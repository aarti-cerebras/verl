# Part B Plan: graft the indexer into MiniCPM3's attention (DSA attention layer)

Second slice of the DSA work (Part A = the standalone `LightningIndexer`, done + tested; see
`docs/dsa_indexer_module_plan.md`, `docs/dsa_minicpm3_plan.md`). Part B wires that module into MiniCPM3-4B's
MLA attention via verl's FSDP/HF monkey-patch path, and implements the **`dense_warmup`** forward (Phase 1):
run the frozen base attention, recompute its head-averaged attention as the distillation target `p`, compute
the indexer scores `I`, and accumulate the per-layer KL so it can be surfaced as the training loss.

Grounded in **Step 0 results** (`dsa_minicpm3_plan.md`): `model_type=minicpm3`, attention class
`MiniCPMFlashAttention2`, MLA sub-layers `q_a_proj`/`q_a_layernorm`/`q_b_proj`/`kv_a_proj_with_mqa`/
`kv_a_layernorm`/`kv_b_proj`, per-layer `rotary_emb`=`MiniCPMLongRoPE` (plain `rotate_half`, fp32, rope dim
32), and the env is transformers **4.57.1 + one `get_usable_length` shim**.

## Scope / non-goals

**In scope:** the build-time shim, the `minicpm3` branch in `apply_monkey_patch`, the patched attention
`forward` with **`dense_warmup`** fully implemented, the `sparse` mode **designed (signature + branch)** but
its full top-k attention deferred, and surfacing the summed per-layer KL on the model so it's testable now.

**Deferred:** the custom FSDP engine + loss wiring (Part C), parameter freezing/optimizer (Part C/D), the
packed-pretrain dataset (separate), training configs/scripts (Part D), full **`sparse`** top-k attention
(Phase 2), and **Ulysses SP > 1** (Part B targets SP=1 first — Step 0 validated SP=1; SP>1 mirrors
`kimi_vl`'s all-to-all and is a follow-up).

## Files

**Create**
- `verl/models/transformers/minicpm_dsa.py` — everything MiniCPM-DSA-specific:
  - `apply_get_usable_length_shim()` — the one Step-0 compat shim (`DynamicCache.get_usable_length →
    get_seq_length`), idempotent.
  - `build_dsa_config(model.config, overrides)` — construct `DSAConfig` from the live config
    (`q_lora_rank=768`, `rope_head_dim=qk_rope_head_dim=32`, `hidden_size`, + `n_heads`/`head_dim`/`top_k`/
    `mode`/`fp8` from training config).
  - `attach_indexers(model, dsa_cfg)` — loop `model.model.layers`, set `attn.indexer = LightningIndexer(...)`
    and `attn.dsa = dsa_cfg` on each `self_attn` instance.
  - `minicpm3_dsa_attn_forward(self, hidden_states, ...)` — the patched `MiniCPMFlashAttention2.forward`
    (mirrors the stock forward + DSA behavior; see below).
  - `install_kl_accumulation(model)` — forward-pre-hook to reset accumulators + forward-hook to sum each
    `layer.self_attn._dsa_kl` into `model._dsa_indexer_kl`.
- `tests/models/test_minicpm_dsa_integration.py` — standalone integration tests on a tiny MiniCPM3 under
  4.57.1 + shim (no full weights, no engine, no training).

**Modify**
- `verl/models/transformers/monkey_patch.py` — add a `minicpm`/`minicpm3` branch (beside `kimi_vl` ~L480),
  gated on `getattr(model.config, "dsa_enabled", False)`: call the shim, `attach_indexers`, set
  `module.MiniCPMFlashAttention2.forward = minicpm3_dsa_attn_forward`, `install_kl_accumulation`, then
  **`return` early** (skip the generic fallthrough at L529-539).
- `verl/utils/flops_counter.py` (~L553) — add `minicpm3` → MLA flops entry.

**Look at (reference)**
- cached `modeling_minicpm.py` `MiniCPMFlashAttention2.forward` (~L534-665) + `_flash_attention_forward`
  (~L671) — the **exact stock forward to mirror** (q_a→q_a_layernorm→q_b, kv split, `apply_rotary_pos_emb`,
  flash call). `qr = q_a_layernorm(q_a_proj(x))` is computed here.
- `verl/models/transformers/kimi_vl.py:91-192` (`_ulysses_flash_attn_forward`) — verl's MLA flash patch
  pattern (and the SP all-to-all for the future SP>1 path).
- `verl/models/transformers/dsa_indexer.py` — `LightningIndexer` / `DSAConfig` (Part A).
- `verl/models/transformers/qwen2_vl.py:164-179` (`prepare_fa2_from_position_ids`) — derive `cu_seqlens`
  from `position_ids == 0` for the per-document mask.
- `verl/workers/engine/fsdp/transformer_impl.py:1101-1109` (`fused_linear_aux` getattr hook) — the Part-C
  precedent for reading the KL out (Part B just exposes `model._dsa_indexer_kl`).

## The patched attention forward (`minicpm3_dsa_attn_forward`)

It is the stock `MiniCPMFlashAttention2.forward` **plus** three additions. Structure:

1. **Projections (mirror stock):** `qr = self.q_a_layernorm(self.q_a_proj(hidden_states))`;
   `q = self.q_b_proj(qr)` → split nope/pe; `kv = ...`; apply `apply_rotary_pos_emb` → build
   `query_states`, `key_states` (post-RoPE), `value_states`. **Capture `qr`, `query_states`, `key_states`,
   and the `cos,sin`** used (the indexer reuses the same cos/sin → identical RoPE to the base).
2. **Base attention output (LM path):** run the stock flash attention to produce `attn_output` exactly as
   today. In `dense_warmup` the base is frozen, so this path is effectively a no-op on grad; we keep its
   output unchanged so the LM forward and deeper layers see the true hidden states.
3. **DSA branch on `self.dsa.mode`:**
   - **`dense_warmup`** (Phase 1):
     - Build the additive mask `[T,T]` (causal **and** per-document, from `position_ids`/`cu_seqlens`).
     - **Target `p`** (detached, `no_grad`): `scores = q·kᵀ·softmax_scale` per head, `+ mask`, softmax over
       keys, **mean over heads** → `p[b,T,T]`. Materialize directly for short `T`; **tile over query blocks
       (`kl_block_size`)** for 32K (reduce over heads immediately → peak ~`B·T`).
     - **Indexer `I`** (grad → indexer only): `I = self.indexer(x=hidden_states, qr=qr, cos, sin,
       attn_bias=mask)` → softmax over keys.
     - **KL:** `KL(p ‖ softmax(I))` over valid (causal, in-doc) keys, mean over valid query positions →
       scalar; store `self._dsa_kl = layer_kl`.
   - **`sparse`** (Phase 2, designed not built): `I = self.indexer(...)`; `top_k` select within-doc/causal;
     gather/mask KV; run attention over the selected set; KL over the selected set; `self._dsa_kl = ...`.
     Implement a `NotImplementedError`-guarded stub now; full impl in the Phase-2 step.
4. **Return** the (unchanged) `attn_output`. The KL leaves only as the scalar on `self`.

Notes: indexer input detach is a Phase-2 requirement; in `dense_warmup` the frozen base already yields
no grad into the base. cos/sin come from `self.rotary_emb` exactly as the stock forward computes them.

## KL surfacing (testable now; consumed in Part C)

- `install_kl_accumulation(model)`: a **forward-pre-hook** on the root model resets per-layer state; a
  **forward-hook** sums `layer.self_attn._dsa_kl` over the 62 layers (normalized by #valid query positions ×
  #layers) into `model._dsa_indexer_kl` (a scalar tensor attribute on the module).
- Part C's custom FSDP engine then sets `model_output["indexer_kl"] = self.module._dsa_indexer_kl` (mirrors
  `fused_linear_aux`, `transformer_impl.py:1101`). Reading off the **module** (not the `ModelOutput` object)
  avoids transformers' `ModelOutput.__setattr__` restrictions.

## Tests (`test_minicpm_dsa_integration.py`, tiny MiniCPM3: 2 layers, vocab 1000, eager+flash, 4.57.1+shim)

1. **Patch applies:** after `apply_monkey_patch` with `dsa_enabled=True`, each `layer.self_attn` has an
   `.indexer` (a `LightningIndexer`) and `.dsa`; `MiniCPMFlashAttention2.forward` is the patched fn; the
   generic fallthrough did **not** also run (early return).
2. **`qr` captured correctly:** the `qr` the forward feeds the indexer equals `q_a_layernorm(q_a_proj(x))`
   recomputed independently.
3. **dense_warmup produces KL:** forward a tiny packed batch → `model._dsa_indexer_kl` is finite and `> 0`;
   per-layer `_dsa_kl` recorded for all layers.
4. **Base LM output unchanged:** logits with DSA patch (dense_warmup) ≈ logits of the un-patched model
   (the dense path must be a no-op on the LM output in Phase 1).
5. **Mask correctness:** pack 2 docs with `position_ids` reset → target `p` and indexer `I` have **zero**
   cross-document / non-causal mass; KL computed only over valid keys.
6. **Target sanity:** at small `T`, the head-averaged `p` rows sum to 1 over the causal prefix; the tiled
   recompute matches the full-matrix recompute within tolerance.
7. **sparse stub:** `mode="sparse"` raises `NotImplementedError` (until Phase 2) — documents the boundary.

## Verification

```
PYTHONPATH=/tmp/tf457lib pytest tests/models/test_minicpm_dsa_integration.py -v     # 4.57.1 + shim env
```
(Plus the Part A suite stays green: `pytest tests/models/test_dsa_indexer.py -v`.)

## Risks / open items
- **Target-recompute cost/memory** — the one real risk (recomputing QK·softmax for `p`). Tiling bounds it;
  validate memory on a longer-seq smoke before 32K.
- **Mask fidelity** — the manual causal+doc mask must match what flash derives from `cu_seqlens`/
  `position_ids` (verified concept; assert in test 5).
- **SP > 1** — deferred; the indexer needs global keys after the MLA all-to-all (mirror `kimi_vl`).
- **Real weights / 62 layers / 32K** — Part B tests use a tiny random model; a real-checkpoint smoke is a
  follow-up before training.
- **`fast_hadamard_transform` missing** — indexer runs the FWHT fallback (fine); install for production FP8.
