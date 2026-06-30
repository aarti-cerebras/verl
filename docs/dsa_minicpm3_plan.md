# Plan: DSA Indexer + Sparse-Attention Adaptation for MiniCPM3-4B (verl, FSDP path)

## Context

We are adapting **MiniCPM3-4B** (dense, MLA backbone, 32K native context) to **DeepSeek Sparse
Attention (DSA)** via continued training, per the Confluence plan
*Q3-2026: MiniCPM3-4B Adaptation*. DSA = a lightweight **lightning indexer** that scores every
query→key pair cheaply, plus **top-k token selection** so main attention only attends to the
k most-relevant keys. The indexer is trained by **self-distillation**: it learns to reproduce the
base model's own (dense) attention distribution — no labels needed.

MiniCPM3-4B is already MLA (`q_lora_rank=768`, `kv_lora_rank=256`, `qk_nope/rope=64/32`, 40 heads,
62 layers), so we can graft the indexer directly instead of re-architecting attention.

This plan covers, in order:
1. **Indexer module** implementation.
2. **DSA attention layer** (a wrapper over MiniCPM3's MLA attention with `dense_warmup` and `sparse` modes).
3. **Wiring the indexer aux-loss** through verl's FSDP SFT trainer.
4. **Phase 1** — dense warm-up, indexer-only (~2B tokens).
5. **Phase 2** — sparse training via **distillation from the frozen dense teacher** + indexer KL (~30B tokens).

**Decisions locked (from user):** FSDP + HF-transformers path; train at **32K native** (no LongRoPE
extension); modules built to support **both** phases, with full detail for both.

### Backend rationale
verl has two paths. We use the **FSDP + HF transformers** path because MiniCPM3-4B is a small dense
model and this path gives us: (a) a pluggable-loss SFT trainer (`verl/trainer/sft_trainer.py`), and
(b) an existing precedent for patching a trust-remote-code MLA attention class —
`verl/models/transformers/kimi_vl.py::_ulysses_flash_attn_forward`, installed in
`verl/models/transformers/monkey_patch.py:480-494` via `module.DeepseekV3FlashAttention2.forward = ...`.
We mirror that precedent for MiniCPM3. (The Megatron `experimental_attention_variant=="dsa"` scaffolding
in `verl/models/mcore/patch.py:316` is DeepSeek-V3/TE-specific and heavier; not used here.)

> **Monkey-patch fallthrough caveat (verified):** for a `model_type` not matched by any branch,
> `apply_monkey_patch` still runs a **generic** `_ulysses_flash_attention_forward` patch +
> `patch_forward_with_backends` at the end (`monkey_patch.py:529-539`). Our MiniCPM3 branch must
> therefore `return` early (exactly like the `kimi_vl` branch at line 494) so the generic path doesn't
> double-patch our DSA attention.

### Step 0 — RESULTS (probed empirically 2026-06-30; **complete ✅**)

**Checks performed** (load + inspect the real MiniCPM3-4B + verl path; see env notes below):

| Check | Result |
|---|---|
| `config.model_type` | **`minicpm3`** |
| Core config | hidden 2560, 62 layers, 40 heads, **q_lora_rank 768**, kv_lora_rank 256, qk_nope 64, **qk_rope 32**, **v_head_dim 64**, vocab 73448, max_pos 32768, LongRoPE |
| Attention class to patch | **`MiniCPMFlashAttention2`** (also `MiniCPMAttention` eager, `MiniCPMSdpaAttention`); chosen via `MINICPM_ATTENTION_CLASSES[_attn_implementation]` |
| MLA sub-layers (exact names) | `q_a_proj`, `q_a_layernorm` (RMSNorm → `qr` = `q_a_layernorm(q_a_proj(x))`), `q_b_proj`, `kv_a_proj_with_mqa`, `kv_a_layernorm`, `kv_b_proj` — all present |
| RoPE | per-attention-module `rotary_emb` = `MiniCPMLongRoPE`; plain `rotate_half` computed in **fp32** (matches the indexer's `_apply_rope`); rope dim 32 |
| FP8 env | H100 **sm_90**, `torch._scaled_mm` present ✅. **`fast_hadamard_transform` NOT installed** → indexer FWHT fallback runs (install for production FP8 quality) |
| verl SFT/FSDP path imports on transformers 4.57.1 | ✅ |
| MiniCPM3 build + **eager** forward+loss (tiny: 2 layers) | ✅ (loss ≈ ln(vocab)) |
| MiniCPM3 build + **flash_attn** forward+loss (H100, bf16) | ✅ — `MiniCPMFlashAttention2` + `flash_attn` run; `qr`/rotary/rope-dim available on the flash module |

**Decision — transformers version (the one blocker found):** MiniCPM3's trust-remote-code modeling targets
transformers ~4.36 and breaks on newer versions at different points (`DynamicCache.get_usable_length` removed
~4.41; `is_torch_fx_available` import + tied-weights-keys list→dict on 5.x). verl, conversely, targets
~4.52–5.x. **Resolution: run DSA training on transformers `4.57.1` + ONE compat shim** —
`DynamicCache.get_usable_length = lambda self, n, layer_idx=0: self.get_seq_length(layer_idx)`, applied by
verl at model-build time (no edits to the HF modeling file). On 4.57.1 this single shim makes MiniCPM3
build + forward + loss pass on both eager and flash. DSA training is the FSDP SFT path (no vllm — even
Phase-2 distillation uses a co-located teacher), so a dedicated **4.57.1 training venv/image** is used; the
stock 5.3.0 serving container is left untouched. See memory `minicpm3-transformers5-incompat`.

**Decided / discarded:**
- ✅ **Chosen:** transformers 4.57.1 + 1 shim, MiniCPM3 loaded via trust_remote_code, **Part B stays the
  monkey-patch plan** (the class loads, so no vendoring needed).
- ❌ **Discarded — pin to a transformers that runs MiniCPM3 unmodified (≲4.40):** breaks verl + the container's
  vllm 0.20.2 / deps; no overlap with verl's window.
- ❌ **Discarded — adapt MiniCPM3 modeling for the installed 5.3.0:** needs several riskier shims (removed
  import + tied-keys-dict + cache + LongRoPE/MLA rope-factor handling) vs. one shim on 4.57.1.
- ❌ **Discarded — map MiniCPM3 weights onto transformers-native `deepseek_v3`:** MiniCPM has its own
  `scale_emb`/`scale_depth`/`dim_model_base` scalings + LongRoPE the native model doesn't replicate.

**Carried into Part B:** apply the `get_usable_length` shim at build time; add a `minicpm3` branch to
`apply_monkey_patch` (patch `MiniCPMFlashAttention2.forward`, attach `LightningIndexer`, `return` early to
skip the generic fallthrough); add a `model_type` → MLA flops entry in `flops_counter.py`; re-validate with
real checkpoint weights / full 62 layers / 32K sequences.

---

## Background: the indexer math (DeepSeek-V3.2, arXiv:2512.02556)

Per layer, for query token `t` and key token `s` (s ≤ t, causal):

```
I[t,s] = Σ_j  w[t,j] · ReLU( q_idx[t,j] · k_idx[s] )        # lightning indexer score
```
- `j` indexes **indexer heads** (reference **64**, head_dim 128; we may downscale — see Part A);
  `q_idx[t,j]` is indexer query head j **derived from the MLA compressed query latent `qr`** (not raw
  hidden states); `k_idx[s]` is a **single shared** indexer key per token (MQA-style, from hidden states
  `x` with LayerNorm); `w[t,j]` is a per-token,per-head fp32 weight from `x`.
- Activation is **ReLU** ("for throughput consideration" — arXiv:2512.02556). The indexer score
  computation runs in **FP8** in the reference ("the lightning indexer has a small number of heads and
  can be implemented in FP8, its computational efficiency is remarkable") — we adopt FP8 too (see
  **FP8 indexer** below).

**Distillation target** (the base model's own attention), per query `t`:
```
p[t,:] = L1normalize_s( Σ_main_heads  A_main[head, t, :] )      # head-summed, then L1-normalized
       = mean over main heads of A_main[head, t, :]             # (each head row already sums to 1)
```
**Indexer loss** (per layer, summed over layers):
```
L_idx = Σ_t  KL( p[t,:]  ‖  softmax_s(I[t,:]) )
```
- Phase 1: KL over the **full causal set** of keys.
- Phase 2: KL **restricted to the top-k selected set** `S_t`.

Module params per layer map exactly to the PEFT names already known to verl
(`verl/utils/megatron_peft_utils.py:40-42`): `wq_b` (indexer query, **from `qr`** — single proj, **no
`wq_a`**), `wk` (indexer key), `weights_proj` (the `w[t,j]` weights). The absence of `wq_a` in that
mapping is the tell that the indexer query reuses MLA's own q-down projection rather than learning its own.

---

## Part A — Indexer module

**New file:** `verl/models/transformers/dsa_indexer.py`

**Mirror the reference exactly** (DeepSeek-V3.2 `inference/model.py` `Indexer`; corroborated by verl's PEFT
mapping `megatron_peft_utils.py:40-42` = `wq_b`/`wk`/`weights_proj`, note **no `wq_a`**):

```python
class LightningIndexer(nn.Module):
    """Per-layer DSA lightning indexer (DeepSeek-V3.2).
    Reference config: index_n_heads=64, index_head_dim=128, index_topk=2048, rope_head_dim=64.
    INPUTS: x = hidden_states; qr = MLA's compressed+normed query latent = q_a_layernorm(q_a_proj(x))."""
    def __init__(self, hidden_size, q_lora_rank, n_heads=64, head_dim=128, rope_head_dim=64, fp8=True):
        # wq_b:         Linear(q_lora_rank, n_heads*head_dim, bias=False)   # indexer query FROM qr (NOT x);
        #                                                                   #   reuses MLA's q-down path → no wq_a
        # wk:           Linear(hidden_size, head_dim, bias=False)           # single shared key from x (MQA-style)
        # k_norm:       LayerNorm(head_dim)                                 # on the key only (not RMSNorm, not q)
        # weights_proj: Linear(hidden_size, n_heads, bias=False, dtype=fp32)# per-head weights from x, in fp32

    def forward(self, x, qr, position_ids, rotary_emb):
        q = self.wq_b(qr).view(b, s, n_heads, head_dim)        # query from compressed latent qr
        q[..., :rope_head_dim] = rope(q[..., :rope_head_dim])  # RoPE on rope-slice only, interleaved=False
        k = self.k_norm(self.wk(x))                            # [b, s, head_dim] single head
        k[..., :rope_head_dim] = rope(k[..., :rope_head_dim])
        weights = self.weights_proj(x.float()) * n_heads**-0.5 # fp32; later folds q_scale * softmax_scale
        # FP8 score (ReLU is INSIDE the kernel):
        #   q_fp8,q_scale = act_quant(q);  k_fp8,k_scale = act_quant(k)     # per-block E4M3
        #   weights = weights * q_scale * softmax_scale
        #   I[t,s] = Σ_j weights[t,j] · ReLU( q_fp8[t,j] · k_fp8[s] )       # fp8_index-style kernel, sum over heads
        return I   # [b, s(query), s(key)] raw scores; tile over query blocks (Part B), never full T×T at once
```

Design choices / corrections:
- **Query input is `qr`, not hidden states.** `qr` = MLA's own `q_a_layernorm(q_a_proj(x))` (q_lora_rank=768
  for MiniCPM3). The indexer reuses that latent and applies a single `wq_b` — there is **no separate `wq_a`**.
  This means the patched MLA forward must **expose `qr`** to the indexer (Part B integration).
- **Key & weights come from hidden states `x`**; key gets **LayerNorm** (`k_norm`); weights are **fp32** and
  scaled by `n_heads**-0.5`. RoPE on the rope-slice of both q and k.
- **Chosen sizing for MiniCPM3 (derived, not copied):** `rope_head_dim=32` (forced = base `qk_rope_head_dim`,
  to reuse RoPE), `head_dim=64` (32 rope + 32 nope; from DeepSeek's `d^I/d_qk≈0.67` and its rope+½·nope rule,
  both of which give 64), and **`n_heads=16` default** with a Phase-1 **sweep {8, 16, 24, 32}**.
- **Why 16 (head-count is the dominant size knob):** the honest yardstick is DeepSeek's indexer fraction of
  **active** params (it's a dense module run every token) = 852M/37B ≈ **2.3%**, *not* the 0.13% storage
  fraction (its 671B denominator is mostly idle MoE experts). On dense MiniCPM3, 2.3% ⇒ ~92M ⇒ ~24 heads;
  the geometric ratio `H^I/H=0.5` ⇒ 20 heads. `n_heads=16` is a deliberately lean start (~61M ≈ **1.5%**);
  the placeholder `64` would have been ~420M ≈ **10.5%** — mis-sized for a 4B dense model. Pick by Phase-1
  top-2048 recall / KL plateau. Indexer params live in fp32 for the optimizer (~1 GB state at 16 heads).
- A `DSAConfig` dataclass (n_heads, head_dim, rope_head_dim, q_lora_rank, top_k, mode, kl_block_size, fp8)
  on the model config so the attention wrapper and loss can read it.

### Indexer head structure (query multi-head, key single-head / MQA)

| Component | # heads | Per-head dim | DeepSeek-V3.2 | MiniCPM3 proposal |
|---|---|---|---|---|
| Indexer **query** (`wq_b`) | `n_heads` (multi-head) | `head_dim` | 64 × 128 | 16 × 64 |
| Indexer **key** (`wk`) | **1 (MQA — shared)** | `head_dim` | **1** × 128 | **1** × 64 |
| Indexer **weights** (`weights_proj`) | `n_heads` (one scalar/query-head) | 1 | 64 | 16 |
| → score `I[t,s]` | all query heads dot the **same** single key, ReLU, weighted-sum over heads | — | sum over 64 | sum over 16 |

The key projection outputs one `head_dim`-wide vector per token (no head multiplicity) → the key cache
stays `head_dim`-wide and FP8-cheap; capacity lives on the query side (`n_heads` + `weights_proj`).

### Indexer weight matrices (per layer; `Linear.weight` is [out, in], all bias-free; `k_norm`=LayerNorm)

| Matrix | Role | Output | DeepSeek (out×in) | DS params | MiniCPM (out×in) | MiniCPM params | dtype |
|---|---|---|---|---|---|---|---|
| `wq_b` | query from `qr` → `n_heads·head_dim` | multi-head q | **8192 × 1536** | 12.58M | **1024 × 768** | 0.79M | bf16→FP8 |
| `wk` | key from `x` → `head_dim` | **1** key head | **128 × 7168** | 0.92M | **64 × 2560** | 0.16M | bf16→FP8 |
| `k_norm` | LayerNorm on key | — | **128** (×2 γ,β) | 0.00M | **64** (×2) | 0.00M | fp32 |
| `weights_proj` | per-head weight from `x` → `n_heads` | `n_heads` | **64 × 7168** | 0.46M | **16 × 2560** | 0.04M | **fp32** |
| **per layer** | | | | **≈13.96M** | | **≈0.99M** | |
| **× layers** | | | (×61) | **≈852M** | (×62) | **≈61.5M** | |

- `wq_b` **in-dim = `q_lora_rank`** (1536/768) because the query is the MLA latent `qr` (the "no `wq_a`" point);
  out reshapes to `[b,s,n_heads,head_dim]`, `head_dim` viewed as `[rope|nope]` = `[64|64]`/`[32|32]`.
- `wk` **out-dim = `head_dim`** (not `n_heads·head_dim`) → the single MQA key.
- Fractions: DeepSeek 852M = **0.13%** of total (671B) / **2.3%** of active (37B); MiniCPM 61.5M = **1.5%** of 4B
  (dense ⇒ total=active). `top_k`=2048 is 1.6% of 128K (DS) but **6.25%** of 32K (MiniCPM) — hence A/B vs 1024.

---

## Part B — DSA attention layer (wrapper over MiniCPM3 MLA)

**New file:** `verl/models/transformers/minicpm_dsa.py` — a patched attention `forward` mirroring
`kimi_vl.py::_ulysses_flash_attn_forward` (same q_a/q_b, kv_a_proj_with_mqa, kv_b_proj, RoPE split,
Ulysses-SP all-to-all), with DSA behavior added. The attention module gets a `.indexer` child
(`LightningIndexer`) attached at patch time and reads `self.config.dsa` for mode.

**Crucial input wiring (from the indexer correction):** the indexer query comes from the **MLA compressed
query latent `qr`**, not raw hidden states. The MLA forward *already* computes it — `qr =
q_a_layernorm(q_a_proj(hidden_states))` (the same tensor fed to `q_b_proj`). The patched forward must
**capture `qr` and call `self.indexer(x=hidden_states, qr=qr, ...)`** — exactly what the mcore DSA path
does via `extra_kwargs["qr"]=q_compressed` (`models/mcore/patch.py:85,97,320`). When `q_lora_rank` is
absent the reference falls back to `qr = hidden_states` (`patch.py:99`); MiniCPM3 has `q_lora_rank=768`,
so `qr` is the 768-dim post-layernorm latent.

**Mode `dense_warmup` (Phase 1):**
1. Run the **normal dense MLA flash-attention** forward unchanged (base is frozen → wrap in
   `torch.no_grad()` for the main path; we only need its output to keep the LM forward well-formed,
   and we need base `q`/`k` for the target).
2. Compute the **distillation target `p[t,:]`** = head-averaged softmax attention weights, **tiled over
   query blocks** to bound memory (see note), under `no_grad()` (target is detached).
3. Compute indexer scores `I[t,:]` (with grad, indexer params only) over the same tiles, and accumulate
   `KL(p ‖ softmax(I))` into a **per-forward aux-loss accumulator**.
4. Return the normal attention output (so the rest of the LM runs) **plus** stash the scalar KL on the
   accumulator. No top-k selection in this phase.

**Mode `sparse` (Phase 2):**
1. Compute indexer scores `I[t,:]`; select **top-k=2048** keys per query (`torch.topk`), build the
   sparse mask / gather indices.
2. Run main MLA attention **only over the selected set** (varlen flash-attn with a block/index mask, or
   gathered KV). This is the real sparse forward used for the LM loss (with grad through base in Phase 2).
3. Also compute `KL(p_S ‖ softmax(I_S))` **restricted to `S_t`** as the indexer aux-loss, where `p_S` is
   the base attention distribution renormalized over the selected set (computed tiled, detached).
4. **Detach the indexer's input from the main graph** (reference requirement) so indexer gradients don't
   flow into the base model.

**Memory note — the dense-target recompute (the one real risk).**
At 32K, a full `T×T` head-summed matrix is ~2 GB/head (bf16); we never form it. Instead we **tile queries**
into blocks of `kl_block_size` (e.g. 1024): for each block we compute `scores_blk = Qblk·Kᵀ` `[B, T]` per
head, softmax over the causal prefix, average over heads → `p_blk [B, T]`; compute `I_blk [B, T]`; add
`KL(p_blk‖softmax(I_blk))`; free. Peak ≈ `B·T` bf16 (1024×32768 ≈ 64M ≈ 128 MB) × few buffers — fine.
Base `q`/`k` for the target are the **post-RoPE query/key states** already built inside the MLA forward;
we capture them (detached) rather than recompute projections. Combine with gradient checkpointing on
the transformer blocks.

**Packing/masking correctness (verified — must not skip).** Phase-1/2 sequences are packed
multi-document at 32K with **`position_ids` reset to 0 per document**; verl's varlen flash-attn derives
`cu_seqlens` from `position_ids == 0` (`verl/models/transformers/qwen2_vl.py:164-179`). Our manual tiled
target recompute and the indexer scores **must replicate the same per-document block-diagonal causal
mask** (mask out keys outside the query's document and keys `> t`). Compute the same `cu_seqlens` from
`position_ids` and apply it as an additive `-inf` mask in both `softmax(scores)` (target) and
`softmax(I)` (indexer) before the KL. If we ignore this, the target leaks across document boundaries and
the indexer learns the wrong distribution. Same constraint applies to Phase-2 top-k selection (top-k must
be taken only within-document, over `s ≤ t`).

**FP8 indexer (matches DeepSeek-V3.2).** The indexer score path is **FP8** by design (`dsa.indexer_fp8`,
default on), since the reference's efficiency win comes from FP8 + few heads + ReLU. Scope and rules:
- **FP8 only on the indexer matmuls** — the `wq_b`/`wk` projections and the `q_idx·k_idx` dot products —
  via `torch._scaled_mm` (E4M3 for q/k/weights) with **per-block scaling** (`act_quant`, per the reference)
  whose `q_scale` is folded into `weights`. Bias-free linears keep this simple.
- **Keep higher precision off the FP8 hot path:** apply ReLU, the per-head weighting `w[t,j]`, the
  `softmax(I)`, the target `p`, and the KL in bf16/fp32. FP8's narrow dynamic range makes softmax/KL
  unstable; only the dot-product accumulation is FP8 (accumulate in fp32, which `_scaled_mm` does).
- **RoPE** is applied before the FP8 cast (rotation in bf16, then quantize q_idx/k_idx to E4M3).
- **Bring-up:** implement a bf16 reference path first and gate FP8 behind the flag; validate FP8 scores
  match bf16 within tolerance (standard FP8 practice) before trusting the KL. This is validation, not a
  fallback — FP8 is the target precision.
- **Kernel availability:** dev/train is **H100 (sm_90)**, so FP8 runs natively via `torch._scaled_mm` (or
  TransformerEngine `Linear`/`fp8_autocast`, which verl already uses on the mcore DeepSeek path) on both
  dev and production. The bf16 reference path is for parity validation only.

**Integration (monkey-patch), mirroring kimi_vl:**
- Add a `minicpm` / `minicpm3` branch in `verl/models/transformers/monkey_patch.py::apply_monkey_patch`
  (next to the `kimi_vl` branch at line 480). It will: import the trust-remote-code module, attach a
  `LightningIndexer` to each attention block, set `<MiniCPMAttentionClass>.forward = minicpm_dsa_forward`,
  and apply Ulysses input slicing if SP>1.
- Gate on a config flag (`model.dsa.enabled`) so non-DSA MiniCPM runs are untouched.

---

## Part C — Wiring the indexer aux-loss through the SFT trainer

**Reviewer correction (verified against code).** The earlier "just put `indexer_kl` in `model_output`"
idea does not work: the engine **rebuilds** `model_output` from scratch in
`prepare_model_outputs` (`transformer_impl.py:1083, 1245-1249`), reading only `output.logits`; any extra
field returned by the HF forward is **dropped**. The loss fn receives that engine-built dict
(`transformer_impl.py:1275-1282`), and `sft_loss` reads only `model_output["log_probs"]`. So we need an
explicit pass-through. There is an exact precedent: the `fused_linear_aux` getattr hook at
`transformer_impl.py:1101-1109` copies aux fields off `raw_output` into `model_output`. We mirror it.

**(C1) Surface the KL from the model.** Each patched attention layer accumulates its scalar KL into a
forward-local buffer on the model (reset via a forward-pre-hook); the patched **model forward** sums them
and attaches `raw_output.indexer_kl` (a scalar tensor) onto the HF output object. Big tiled tensors stay
local to each layer — only a scalar leaves.

**(C2) Custom engine subclass to pass it through.** Add an FSDP engine variant
(`verl/workers/engine/fsdp/`), registered via `EngineRegistry.register(model_type="dsa_language_model",
backend=["fsdp","fsdp2"])`, that overrides `prepare_model_outputs` to do
`model_output["indexer_kl"] = getattr(raw_output, "indexer_kl", None)` (mirroring lines 1101-1109).
Select it via `TrainingWorkerConfig(model_type="dsa_language_model", ...)` in `sft_trainer.py:165`.
(Alternative considered & rejected: routing through the `logits_processor_func` topk path at lines
1221-1228 — that path is shaped for per-token tensors == `log_probs.shape`, not a scalar aux loss.)

**(C3) New losses** in `verl/workers/utils/losses.py` (same `(config, model_output, data, dp_group)`
signature as `sft_loss`; KL/agg helpers from `verl/trainer/distillation/losses.py`):
- `indexer_kl_loss` — Phase 1: returns `model_output["indexer_kl"]` (already aggregated, normalized by
  #valid query positions × #layers), metrics = per-layer KL + indexer/teacher mass. **No LM CE.**
- `dsa_distill_loss` — Phase 2: `agg(forward_kl_topk(student_logits, teacher_topk)) + λ · indexer_kl`
  (`λ` from config). **No raw CE by default** (self-distillation toward dense parity — the teacher's soft
  distribution dominates the hard label; CE conflicts with calibration matching). Optional low-weight
  `μ · CE` **only** in the offline small-k regime as a top-k truncation backstop. Reuses verl's
  `compute_forward_kl_topk` (`verl/trainer/distillation/fsdp/losses.py`) for the distillation term —
  it is logit-agnostic, so a sparse-attention student works unchanged.

**(C4) Loss selection.** Add `loss_mode` (`sft|indexer_kl|dsa_combined`) to the SFT config and branch at
`sft_trainer.py:163` to pick the fn. Small, additive change.

**(C5) Phase-1 compute saving (optional).** In Phase 1 the LM head / `log_probs` over 32K×full-vocab is
pure waste (loss ignores it). The custom engine's `prepare_model_outputs` can short-circuit the
log_probs computation when `loss_mode==indexer_kl`, returning only `indexer_kl`. Keeps Phase-1 memory low.

**(C6) Phase-2 distillation wiring (reuse existing).** The distillation top-k loss is *already* wired into
the engine forward: `prepare_model_outputs` has a `distillation_use_topk` logits-processor branch
(`transformer_impl.py:1221-1228`) that calls `compute_topk_loss` and stashes
`distillation_losses`/`student_mass`/`teacher_mass` into `model_output`. Only the **teacher-logit source**
is RL-specific (online vLLM via the agent loop). So Phase 2: enable `distillation.enabled` + provide
`data["teacher_logprobs"]`/`data["teacher_ids"]` from one of two backends (Part E), set
`use_task_rewards=False`/`use_policy_gradient=False` for pure supervised forward-KL, and have the DSA
engine add `indexer_kl` on top. No change to `compute_forward_kl_topk` itself.

**Freezing + optimizer — reuse the LoRA pattern (verified).** Phase 1 is structurally identical to LoRA
(frozen base + small trainable modules), which verl already supports end-to-end (`is_lora`/`lora_rank`
threaded through the wrap policy at `transformer_impl.py:371` and elsewhere). Concretely:
- Set `requires_grad=False` on base, `True` on `*.indexer.*` **before FSDP wrapping** (in the model-build
  path near `apply_monkey_patch`).
- **Mixed `requires_grad` in one FSDP1 FlatParameter is unsupported** unless `use_orig_params=True`
  (`transformer_impl.py:402`, config-driven) — set it, or use FSDP2 (`fully_shard`, per-param). Document
  this requirement in the Phase-1 config.
- `_build_optimizer` (`transformer_impl.py:451-456`) passes **all** `module.parameters()`; filter to
  `requires_grad` (and/or param groups) so AdamW only tracks indexer params at LR `1e-3`.
- Phase 2: unfreeze base; two param groups — base LR `~7e-6`, indexer LR `~1e-3`
  (`build_optimizer`, `verl/workers/config/optimizer.py:218`).

---

## Part D — Phase 1 training plan: dense warm-up (indexer-only, ~2B tokens)

**Goal:** with base **frozen**, train only the indexer to match MiniCPM3's own dense MLA attention.

- **Init:** MiniCPM3-4B dense checkpoint; attach indexers; `mode=dense_warmup`.
- **Loss:** `indexer_kl_loss` (per-layer KL summed; no LM CE).
- **Optimizer/LR:** AdamW, indexer LR `1e-3`, constant (short linear warmup ~2% steps); base frozen.
- **Data:** `openbmb/InfLLM-V2-data-5B` (use 2.0B of its 5B). **Re-tokenize with MiniCPM3's tokenizer**
  (vocab ≠ MiniCPM4 — ignore shipped IDs). **Genuinely pack to 32K** (short docs defeat the indexer).
  → new packed-pretraining dataset class (see Files); **emit `position_ids` reset to 0 per document** so
  varlen flash-attn builds correct `cu_seqlens` (`qwen2_vl.py:174`) and so our target/indexer masks match.
  Verified: no existing verl dataset packs raw text — all are chat/message-based (`multiturn_sft_dataset.py`
  uses sequential `position_ids`, no reset). Loss mask = all tokens (full-sequence KL).
- **Budget:** ~2B tokens → e.g. ~1000 steps × (global batch ≈ 16 seqs × 32K ≈ 0.5M tok/step)≈ ~4k steps;
  set `total_training_steps` to hit ~2B tokens. Size-independent per the reference.
- **Eval/early-stop:** track per-layer KL going down + indexer top-k **recall** vs dense attention (does
  the indexer's top-2048 cover most of the dense attention mass?). Stop when KL plateaus.

---

## Part E — Phase 2 training plan: sparse training via distillation (~30B tokens)

**Goal:** with **top-k=2048** sparse attention active, make the sparse student **reproduce the frozen dense
teacher's output distribution** (and keep the indexer matching the selected-set attention), recovering
dense quality. This is self-distillation toward parity, which directly targets the ~1% gate.

- **Init:** Phase-1 checkpoint (warmed-up indexer); `mode=sparse`. **Train base + indexer together**
  (decision): base LR `~7e-6`, indexer LR `~1e-3`; cosine or constant; grad-clip 1.0.
- **Teacher:** a **separate frozen snapshot** of the original dense MiniCPM3-4B (full attention, DSA off).
  Must be a *distinct* copy — not the student's base — because the student base is being fine-tuned. Same
  tokenizer (same model), teacher-forced on the **same packed inputs/position_ids** as the student.
- **Objective (confirmed):**
  `L = agg(forward_kl_topk(student_sparse_logits, teacher_topk)) + λ · indexer_KL(on selected set S_t)`.
  **No raw CE by default.** Optional low-weight `μ · CE` **only** in the offline small-k backend as a
  top-k truncation backstop. **Detach indexer input** from the main graph.
- **Teacher-logit source — two configurable backends (both supported, decision):**
  - **(E-online) co-located frozen teacher:** load the dense teacher as a second frozen module in the
    training process; each step run a teacher-forced forward (no_grad, dense attn, FSDP-shardable /
    CPU-offloadable) → top-k over vocab → inject `data["teacher_logprobs"]`/`["teacher_ids"]`. No storage;
    ~+1 forward/step; teacher params resident (offload to save HBM). Best for the 30B streaming corpus.
  - **(E-offline) precomputed top-k:** a new dump utility runs the teacher once over the corpus, writes
    `teacher_logprobs`/`teacher_ids` (top-k, e.g. 64) to parquet beside the packed sequences; training
    reads them with no teacher in memory. Multi-TB at 30B×k — best for smaller/fixed distill sets or
    many epochs. Reuses verl's data contract directly.
  - Both feed the **same** loss path (Part C6): `distillation_use_topk` logits-processor +
    `compute_forward_kl_topk`, with `use_task_rewards=False`/`use_policy_gradient=False` (pure supervised
    forward-KL, no RL/rollouts).
- **Two distinct top-k's (don't conflate):** **attention** top-k = 2048/32K (~6.25%, config-driven, A/B
  k=1024); **distillation vocab** top-k = e.g. 64 (monitor `teacher_mass` — already emitted by
  `compute_forward_kl_topk` — to confirm captured mass is high; raise k if low).
- **Data mix (packed to 32K, MiniCPM3 tokenizer), 30B default (range 15–60B):**
  | Domain | Share | Tokens | Source |
  |---|---|---|---|
  | EN general web | 40% | 12.0B | `openbmb/Ultra-FineWeb` (en) |
  | ZH general web | 15% | 4.5B | `openbmb/Ultra-FineWeb` (zh) |
  | Code | 20% | 6.0B | The Stack v2 / StarCoder (repo/file concat→32K) |
  | Math + STEM | 15% | 4.5B | UltraData-Math + NuminaMath-CoT + OpenR1-Math |
  | Long docs | 10% | 3.0B | remaining InfLLM-V2-data-5B (~3B) + arXiv/books |
- **Guardrails:** decontaminate vs RULER / LongBench-v2 / C-Eval / MMLU / AIME before reporting.
  **Eval gate:** sparse within **~1%** of dense on long-context + task benchmarks; track **student↔teacher
  agreement / KL** as a direct in-training metric. Default 30B, early-stop on parity; push to 60B if a gap
  remains.

---

## Files to create / modify

**Create**
- `verl/models/transformers/dsa_indexer.py` — `LightningIndexer`, `DSAConfig`.
- `verl/models/transformers/minicpm_dsa.py` — patched MLA `forward` (dense_warmup + sparse), per-layer KL
  accumulation, top-k selection; patched **model forward** that sums per-layer KL and attaches
  `raw_output.indexer_kl`; forward-pre-hook to reset the accumulator.
- `verl/workers/engine/fsdp/<dsa_engine>.py` — FSDP engine subclass registered as
  `model_type="dsa_language_model"`, overriding `prepare_model_outputs` to copy `raw_output.indexer_kl`
  into `model_output` (mirrors `fused_linear_aux`, lines 1101-1109); optional Phase-1 log_probs skip.
- `verl/utils/dataset/packed_pretrain_dataset.py` — raw-text → re-tokenize (MiniCPM3) → pack to 32K with
  **per-document `position_ids` reset**; registerable via `data.custom_cls`. Phase-2 offline mode also
  carries `teacher_logprobs`/`teacher_ids` columns.
- `verl/models/transformers/dsa_teacher.py` (or in the DSA engine) — **co-located frozen dense teacher**
  loader + teacher-forced top-k forward (no_grad, DSA off, CPU-offloadable); injects teacher tensors into
  `data` for the online backend.
- `scripts/dsa/dump_teacher_topk.py` — **offline teacher-logit dump** utility (no equivalent exists in
  repo): run the dense teacher over the packed corpus, write top-k `teacher_logprobs`/`teacher_ids` to
  parquet.
- `verl/trainer/config/sft_trainer_minicpm_dsa_phase1.yaml` and `..._phase2.yaml` — Hydra configs
  (model=MiniCPM3 trust_remote_code, `model.dsa.*`, `data.*` packing, `optim.*`, `loss_mode`,
  `engine.use_orig_params=true`; Phase-2 adds `distillation.*` with teacher backend = online|offline).
- `examples/dsa/run_minicpm3_dsa_phase1.sh`, `run_minicpm3_dsa_phase2.sh` — launch scripts.

**Modify**
- `verl/models/transformers/monkey_patch.py` (~line 480, beside `kimi_vl`) — add `minicpm`/`minicpm3`
  DSA branch keyed on the confirmed `model_type` (Step 0): attach indexers, patch attention + model
  forward, Ulysses slicing, then **`return` early** to skip the generic fallthrough patch (lines 529-539).
- `verl/workers/utils/losses.py` — add `indexer_kl_loss`, `dsa_distill_loss` (reuses
  `verl/trainer/distillation/fsdp/losses.py::compute_forward_kl_topk`; that file needs **no change**).
- `verl/trainer/sft_trainer.py` (~line 163-165) — select loss fn by `loss_mode`; set
  `TrainingWorkerConfig(model_type="dsa_language_model")`; enable `distillation.*` for Phase 2.
- Reuse `verl/workers/config/distillation.py` (`DistillationConfig`/`DistillationLossConfig`) for the
  Phase-2 distillation knobs (`topk`, `use_task_rewards=False`, `use_policy_gradient=False`).
- `verl/workers/engine/fsdp/transformer_impl.py` — freeze/unfreeze per phase near model build (before FSDP
  wrap); filter optimizer params to `requires_grad` (or param groups) in `_build_optimizer` (line 451).
- `verl/utils/flops_counter.py` (~line 553) — add MiniCPM3 `model_type` → MLA flops entry.
- (Optional) `verl/utils/megatron_peft_utils.py` already has `wq_b`/`wk`/`weights_proj` mappings — keep
  indexer submodule names consistent with these.

---

## Verification

0. **Step-0 probe:** load MiniCPM3-4B (`trust_remote_code=True`), print `config.model_type`,
   `type(self_attn).__name__`, and the attention sub-module names; confirm the patch targets exist.
1. **Module unit test (no training):** load MiniCPM3-4B with `model.dsa.enabled=true` on a single GPU,
   forward a short packed batch (e.g. 4K) in `dense_warmup`; assert `raw_output.indexer_kl` is finite,
   per-layer KL > 0, shapes of `q_idx/k_idx/w` correct, and base-model logits **unchanged** vs an
   un-patched forward (dense path must be a no-op on the LM output in Phase 1).
1b. **Engine pass-through:** assert the custom engine's `prepare_model_outputs` lands
   `model_output["indexer_kl"]` and that `indexer_kl_loss` returns it as the scalar loss.
1c. **Packed-mask test:** pack 2 short docs into one sequence with `position_ids` reset; assert the
   target `p` and indexer `I` masks give **zero** cross-document attention, matching the varlen
   `cu_seqlens` derived from `position_ids==0`.
2. **Target sanity:** verify head-averaged attention `p` rows sum to 1 over the causal prefix on a tiny
   eager reference; confirm the tiled recompute matches a full-matrix recompute within tolerance at 2K seqlen.
3. **Freeze check:** after Phase-1 build, assert only `*.indexer.*` params have `requires_grad=True`, the
   optimizer's param count matches the indexer param count, and FSDP wrapped without error under
   `use_orig_params=true` (or FSDP2). Confirm a step changes only indexer params (base bit-identical).
4. **Phase-1 smoke run:** ~50 steps on a small InfLLM-V2 shard at 32K (or 8K to fit), confirm KL
   decreases and memory is stable (validates the tiling). Track per-layer KL + indexer top-2048 recall.
5. **Sparse forward test:** flip to `mode=sparse`, k=256 on a 4K seq, assert top-k mask is causal +
   within-document, selected-set attention output is finite; confirm indexer-input detach (no grad on base
   from the KL term).
6. **Teacher backend test:** (online) assert the co-located frozen teacher's top-k `teacher_logprobs`/
   `teacher_ids` match a standalone dense forward on the same packed input; (offline) assert the dumped
   parquet round-trips into `data` with correct shapes. Confirm `forward_kl_topk` produces a finite loss
   on sparse-student logits and that `teacher_mass` is high at the chosen vocab-k.
7. **Phase-2 smoke + eval gate:** short distillation run (`dsa_distill_loss`); confirm student↔teacher KL
   decreases; then the long-context eval harness (RULER / LongBench-v2) comparing sparse-student vs dense
   — the ~1% parity gate.

## Open items / risks
- **MiniCPM3 `model_type` + attention class are unverified from the repo** (Step 0). The repo only has
  `minicpmv`/`minicpmo` vision variants. Must load the checkpoint and read both before coding the patch.
- **Aux-loss requires a custom engine subclass** (Part C2), not just a loss swap — confirmed by tracing
  `prepare_model_outputs`. This is the largest piece of new plumbing.
- **FSDP freezing** needs `use_orig_params=True` (FSDP1) or FSDP2; otherwise mixed `requires_grad` in a
  FlatParameter breaks. Reuse the LoRA path's handling.
- **Packing/masking**: per-document `position_ids` reset is mandatory, and the manual target/indexer
  masks must replicate the varlen per-document causal mask — the top correctness risk.
- **Ulysses SP interaction (Phase 2)**: the indexer needs global keys and top-k spans the full sequence;
  under SP the keys are only complete after the MLA all-to-all gather. Top-k + selected-set gather must
  run on the gathered (full-seq, head-sharded) tensors. Validate SP=1 first, then SP>1.
- **Phase-2 distillation teacher cost**: online backend keeps a *separate* frozen dense 4B teacher
  resident (≈ +8 GB bf16 params) + ~1 extra forward/step — CPU-offload the teacher or shard it to fit
  alongside the student + optimizer on H100. Offline backend trades that for **multi-TB** top-k storage at
  30B tokens (~6 TB at k=32) — only viable for small/fixed distill sets or many epochs. Pick per corpus.
- **Distillation vocab top-k truncation**: small k loses the tail; monitor `teacher_mass` and raise k (or
  add the small CE backstop) if captured mass is low. Teacher must use the **same tokenizer** (same model
  — trivially satisfied) and identical packing/position_ids as the student.
- **FlashAttention varlen sparse kernel** for Phase 2 top-k: start with a gathered-KV / masked
  implementation for correctness, optimize to a block-sparse kernel later.
- **FP8 indexer (design default, matches DeepSeek-V3.2)**: indexer score matmuls in FP8/E4M3 with
  per-tensor scaling; softmax/target/KL stay bf16/fp32. Needs `torch._scaled_mm` or TransformerEngine and
  Hopper-class GPUs — confirm at Step 0. A bf16 reference path exists for validation / unblocking dev on
  non-FP8 GPUs, but FP8 is the production setting.
- **Indexer RoPE / weight-init details** (does the indexer apply RoPE; init scale of `wq_b`/`wk`) should
  be cross-checked against the DeepSeek-V3.2 report before finalizing the module.
- Packing throughput on 5B/1T raw-text datasets — re-tokenization is the long pole for data prep.
