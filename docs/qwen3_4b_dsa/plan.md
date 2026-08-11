# Qwen3-4B → DSA (GQA → sparse, no MLA) — implementation plan

**Goal:** give **Qwen3-4B-Thinking-2507** DeepSeek-style sparse attention (lightning indexer +
top-k key selection) **directly on its GQA attention**, with no MLA conversion, and prove it works
at **32K** without losing long-context or reasoning capability.

**Decisions (2026-07-27):**
- **No MLA.** Indexer bolts onto GQA. This is the single biggest simplification vs. the MiniCPM3
  port: one confound (sparsity) instead of three (MLA conversion + recovery training + sparsity).
- **Target model:** `Qwen/Qwen3-4B-Thinking-2507` (see §1).
- **Target length:** 32K, which is deep inside the model's 262144 native range. No YaRN.
- **Budget:** `top_k = 2048` (DeepSeek-V3.2's deployed value) = 6.25% selection ratio at 32K.
- **Selection granularity:** block-pooled, `Bk = 64` (see §4.2). Token-granular Triton is a later
  optimization, not v1.
- **Phase 2 splits** into 2a (base frozen) and 2b (base unfrozen); 2b is conditional on 2a's result.

---

## 1. Target model

Three different things are called "Qwen3 4B". Only two are GQA.

| | **Qwen3-4B** | **Qwen3-4B-Instruct-2507** | **Qwen3-4B-Thinking-2507** | **Qwen3.5-4B** (on disk) |
|---|---|---|---|---|
| architecture | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | `Qwen3ForCausalLM` | `Qwen3_5ForConditionalGeneration` |
| layers / hidden | 36 / 2560 | 36 / 2560 | 36 / 2560 | 32 / 2560 |
| q / kv heads | 32 / 8 | 32 / 8 | 32 / 8 | 16 / 4 |
| head_dim | 128 | 128 | 128 | **256** |
| intermediate | 9728 | 9728 | 9728 | 9216 |
| rope_theta | 1e6 | 5e6 | **5e6** | 1e7 (mrope, `partial_rotary_factor 0.25`) |
| max_position_embeddings | **40960** | 262144 | **262144** | 262144 |
| rope_scaling | null | null | **null** | — |
| attention | full, GQA | full, GQA | **full, GQA** | **hybrid: 24× linear + 8× full** (`full_attention_interval: 4`) |
| modes | thinking + non-thinking | non-thinking only | **thinking only** | — |
| other | — | — | — | multimodal (vision tower), MTP layer, vocab 248320 |

**Chosen: `Qwen3-4B-Thinking-2507`.**

1. **32K is deep inside its native range** (262144 max_pos, rope_theta 5e6, no rope_scaling). The
   original Qwen3-4B caps at 40960, so 32K sits at the edge of its trained range — and you cannot
   demonstrate "DSA preserves long-context capability" against a dense reference that is itself
   marginal at the target length. The reference curve has to be genuinely good or every Δ is noise
   on a weak baseline.
2. **Thinking mode is the capability worth keeping**, and it is also the regime that stresses sparse
   attention hardest (§1.1). Choosing the non-thinking Instruct-2507 would simplify Phase-2
   behavior cloning but forfeit the more interesting and more valuable result.
3. **Geometry is byte-identical across both 2507 variants and the original 4B**, so every number in
   this document applies to any of them — the choice is reversible at zero code cost.

**`/cb/ml-eng/aarti/models/qwen3p5_4b` is NOT a candidate.** It is hybrid linear-attention +
multimodal; only 8 of 32 layers are full attention, so "GQA → DSA" would touch a quarter of the
stack while the interesting long-context behavior lives in the linear-attention layers. Different
project. (The repo already has `verl/models/transformers/qwen3_5.py` for that family.)

### 1.1 What thinking mode changes

Thinking mode moves the center of gravity from prefill to **decode**, which is a genuinely new
regime relative to the MiniCPM3 work (where code generation p50 was ~406 tokens).

1. **Sparsity bites on every item, including short-prompt benchmarks.** A 500-token AIME prompt
   with an 8K reasoning trace means ~94% of forward passes attend over self-generated context,
   sub-selecting 2048 keys from a growing 8K+. Short-prompt benchmarks stop being short-context
   tests.
2. **The selection ratio degrades *within* a single response.** `k` is fixed; `L` grows every token.
   A 32K prompt + 12K trace slides k/L from 6.25% → 4.6% mid-generation. No fixed-length benchmark
   grid captures this; recall must be measured against **generated-trace position**, not just
   prompt length.
3. **New failure mode with no MiniCPM3 analogue:** losing track of its own earlier reasoning →
   repetition loops, restarted derivations, or never emitting `</think>`. Accuracy alone masks
   this; see the trace-integrity gates in [eval_plan.md](eval_plan.md).
4. **Thinking traces belong in the Phase-1 data mixture**, not just real documents — otherwise the
   indexer never trains on the attention patterns it spends most of inference on.
5. **Sampled decoding, not greedy** (`generation_config.json`: `temperature 0.6, top_p 0.95,
   top_k 20, do_sample: true`). This invalidates the greedy protocol used for MiniCPM3 and forces
   multi-sample scoring with error bars. Full protocol in [eval_plan.md](eval_plan.md) §5.

---

## 2. Architecture deltas vs. the MiniCPM3 (MLA) port

Three assumptions baked into `verl/models/transformers/dsa_indexer.py` and
`verl/models/transformers/minicpm_dsa.py` do not carry over:

| MLA assumption | Qwen3-GQA replacement |
|---|---|
| Indexer query comes from the MLA compressed latent `qr = q_a_layernorm(q_a_proj(x))`; `DSAConfig.q_lora_rank` must equal the base `q_lora_rank` | Qwen3 has no q-LoRA. Give the indexer its **own** rank-512 down-projection `wq_a: hidden → 512`. `q_lora_rank` stops meaning "base latent width" and becomes an indexer-internal knob. |
| `rope_head_dim = 32` (partial RoPE, nope split); base `cos`/`sin` reused verbatim | Qwen3 applies **full** RoPE over all 128 dims. Set `rope_head_dim = head_dim = 128` so base `cos`/`sin` is still reused **verbatim** — this makes the indexer/attention RoPE-mismatch bug class structurally impossible. |
| One latent KV shared by all heads → per-token top-k is natural; `FLASHMLA_SPARSE` kernel exists | 8 KV heads, no shared latent, **no sparse kernel exists**. This is the real work — §4.2. |

**Qwen3 quirks to keep straight:**

- **`head_dim × num_heads = 4096 ≠ hidden_size = 2560`.** Head dim is decoupled
  (`q_proj: 2560→4096`, `o_proj: 4096→2560`). The indexer's query source is `hidden_states` at
  **2560**, not the 4096-wide attention space.
- **QK-norm.** Qwen3 applies per-head RMSNorm to q and k before RoPE, architecturally (not a config
  flag). The indexer keeps its own norms — no coupling. But the KL target `p` must be computed from
  the *actual* post-QK-norm, post-RoPE `query_states`/`key_states` after `repeat_kv` to 32 heads,
  i.e. exactly what the patched forward already receives.
- **`tie_word_embeddings: true`.** Affects FSDP wrapping and `scripts/dsa/consolidate_indexer_ckpt.py`.
  MiniCPM3-4B also ties, so the existing path probably handles it — add a targeted test rather than
  assuming.

---

## 3. What the sparsity actually buys (and why 4B is a better target than 8B)

Qwen3-4B and Qwen3-8B have **identical attention geometry**: 36 layers × 8 KV heads × head_dim 128.
Same KV cache — **147 KB/token, 4.83 GB at 32K** — while the model is half the size. So attention
is a much larger share of the work on 4B.

| at 32K prefill | **Qwen3-4B** | Qwen3-8B |
|---|--:|--:|
| non-attention FLOPs (`2·P·L`) | 263 TFLOP | 524 TFLOP |
| dense attention FLOPs | 317 TFLOP | 317 TFLOP |
| attention share of prefill | **54.7%** | 38% |
| sparse attention @ k=2048 | 40 TFLOP | 40 TFLOP |
| indexer overhead (8 heads × 128, dense O(L²)) | 40 TFLOP | 40 TFLOP |
| **net prefill speedup** | **~1.7×** | ~1.4× |
| decode KV-read reduction | **~16×** | ~16× |
| indexer KV-cache overhead | **+3.1%** (4.6 KB/token FP8 on 147 KB/token) | +1.5% |

**Be honest about the shape of the win:** GQA at 4B does not have an MLA-style KV-memory problem
(4.8 GB at 32K is nothing on an H100). The payoff is **decode bandwidth**, with prefill a secondary
~1.7×. The headline result of this project should be a decode throughput/latency curve vs. context
length, not a memory saving.

**Corollary — keep the indexer small.** The indexer is itself dense O(L²). At 16 heads × 128 dim it
would cost ~80 TFLOP (14% of dense prefill) and eat much of the saving. 8 heads × 128 is the
recommended point: it keeps RoPE-table reuse exact (head_dim == base head_dim) while holding
indexer cost to ~7% of dense prefill.

### 3.1 Indexer sizing

```python
DSAConfig(
    enabled=True,
    n_heads=8,            # 8x128 holds the O(L^2) indexer at ~40 TFLOP (~7% of dense prefill)
    head_dim=128,         # == base head_dim -> reuse base cos/sin VERBATIM
    rope_head_dim=128,    # full RoPE, matching Qwen3 (no nope split)
    q_lora_rank=512,      # now indexer-INTERNAL rank (Qwen3 has no q-LoRA); wq_a: 2560 -> 512
    hidden_size=2560,
    top_k=2048,           # 6.25% selection ratio at 32K; train == deploy
    mode="dense_warmup",  # Phase 1
    kl_block_size=512,
    kl_checkpoint=True,   # required at 32K
    fp8=True,
    fp8_ue8m0=True,       # from day one -- do not repeat the MiniCPM3 retrain
)
```

Per layer: `wq_a` 2560×512 = 1.31M · `wq_b` 512×1024 = 0.52M · `wk` 2560×128 = 0.33M ·
`weights_proj` 2560×8 = 0.02M ≈ **2.18M** → **×36 = 78M params, ~1.9% of the 4.02B model.**

**`fp8_ue8m0=True` from the start.** The ~2% train/serve selection drift documented in
`docs/dsa_eval_report.md` §5 matters *more* at 32K than at 1K: with a larger candidate set, a given
score perturbation flips proportionally more of the top-k.

### 3.2 Family compatibility (free dev ladder)

**Qwen3-0.6B / 1.7B / 4B / 8B all share `head_dim = 128` and 8 KV heads.** One `qwen3_dsa.py`
covers the family, and the local 0.6B/1.7B checkpoints are architecturally faithful smoke targets
rather than toy approximations.

| local checkpoint | layers | hidden | q/kv | head_dim | use |
|---|--:|--:|--:|--:|---|
| `/cb/ml-eng/aarti/models/qwen3_0p6b` | 28 | 1024 | 16/8 | 128 | Phase 0 correctness/parity at 4K |
| `/cb/ml-eng/aarti/models/qwen3_1p7b` | 28 | 2048 | 16/8 | 128 | Phase-1 smoke |
| Qwen3-4B-Thinking-2507 (**needs download**) | 36 | 2560 | 32/8 | 128 | real runs |

---

## 4. Phase 0 — module, kernel, parity (blocking)

New `verl/models/transformers/qwen3_dsa.py`, mirroring `minicpm_dsa.py`'s structure — the
`attach_indexers` / `freeze_base_train_indexer` / `install_kl_accumulation` / `_dense_warmup_kl` /
`_sparse_attn_and_kl` decomposition all keep their shapes.

### 4.1 Decision — selection sharing

**One index set per query token, shared across all 32 q heads / 8 KV groups**, exactly like DSA.
Per-KV-head sets would give higher recall at the same k but cost 8× the gather and far messier
kernels; not for v1.

This makes the **existing KL target correct as-is**: attention probabilities summed over all 32 q
heads, then L1-normalized (already what `_dense_warmup_kl` computes; just add `repeat_kv`).

### 4.2 Decision — selection granularity (determines whether Phase 2 is trainable at all)

Token-granular top-k on GQA *without* a fused kernel means materializing gathered KV per query
tile. At 32K with k=2048 and q-tiles of 128, that is ~537 MB per tile per layer and ~137 GB of
gather traffic per layer per step. Not viable.

> **Decision: pool indexer scores over key blocks of `Bk = 64` tokens, select the top 32 blocks
> (= 2048 tokens), and express the result as a FlexAttention `BlockMask`.**

No materialization, a generated fused kernel, and it maps directly onto vLLM's paged KV blocks at
serving time — which is also the way out of the `enforce_eager` / no-CUDA-graph hole the MiniCPM3
serving path is stuck in. This is NSA's design. Token-granular Triton stays as a later optimization.

**Set `Bk = 64` and serve with `--block-size 64`** so selected block ids *are* page ids and the
serving-side gather becomes the block table that already exists.

#### Selection mechanics

Phase 1 is unaffected — the indexer still emits token-level scores and the KL is still token-level.
Only *selection* is pooled.

```python
# 1. indexer scores, exactly as Phase 1 computes them: raw / unnormalized
I = indexer_scores(q_tile, keys)                        # [q_tile, L]

# 2. mask BEFORE softmax: causal + document boundaries (reuse _causal_doc_bias_block)
I = I + causal_doc_bias                                 # -inf where s > t or different doc

# 3. to predicted attention mass
p = I.softmax(dim=-1)                                   # [q_tile, L], fp32

# 4. pool onto key blocks (L padded to a multiple of Bk)
mass = p.view(q_tile, L // Bk, Bk).sum(-1)              # [q_tile, n_kv_blocks]

# 5. force-include sinks + local window, INSIDE the budget
mass[:, 0] = float("inf")                               # first 64 tokens (attention sinks)
mass.scatter_(1, local_block_ids, float("inf"))         # blocks covering [t-128, t]

# 6. select
idx = mass.topk(n_blocks, dim=-1).indices               # n_blocks = top_k // Bk = 32
```

Masking **before** softmax makes the causal diagonal block correct for free: masked entries are
already zero in `p`, so a partially-visible block's mass sums to exactly its visible portion. No
special-casing.

Force-including *inside* the budget (the `inf` trick) keeps `top_k` exactly 2048, so ladder rows
stay budget-comparable with token-granular k. Sinks + local eat 3 of 32 blocks — negligible.

#### Why mass-sum pooling

The gate metric is `topk_recall` = recovered attention **mass**. Mass is additive and blocks are
disjoint, so "pick the n blocks maximizing total predicted mass" is **greedy-optimal for that
objective** under the indexer's own estimate. Every other pooling is a heuristic approximation.

| pooling | behavior | verdict |
|---|---|---|
| **sum of `softmax(I)`** | maximizes predicted recovered mass | **default** |
| max of `I` in block | "does this block contain a spike" | good for single-needle retrieval, loses diffuse mass; keep as ablation knob |
| mean of `I` | divides a needle's score by 64 | **avoid** — dilutes exactly the retrieval signal being preserved |
| sum of raw `I` | ≈ mean × 64; `I` can be negative (unconstrained `w[t,j]`), row scale varies | avoid |

Sharpness tension: mass-sum can prefer 64 medium keys over one spike, which is the retrieval case.
If NoLiMa/NIAH show that failure, the cheap fix is **sum of the top-4 masses per block** — captures
spikes without mean-pooling's dilution. Measure before building it.

#### The second approximation (the one that actually costs)

FlexAttention's `BlockMask` is defined on **(q_block, kv_block)** pairs, so all queries in a q-block
must share one KV-block set. The q axis must be collapsed too:

```python
mass_q = mass.view(n_qblk, Q_BLOCK, n_kv_blocks).amax(1)   # amax, NOT mean
idx = mass_q.topk(n_blocks, -1).indices                     # [n_qblk, n_blocks]
```

Use `amax`: a needle that only one query in the block wants should survive. This is an independent,
second recall loss stacked on the KV pooling, and it is the one that gets forgotten.

**It only affects prefill.** At decode there is one query per step, so selection is exactly
per-query — and decode is where the ~16× bandwidth win lives.

Measure recall at four points, not two:

| config | what it bounds |
|---|---|
| token-granular, per-query | upper bound (what DSA-on-MLA achieves) |
| block-64 KV, per-query | cost of KV pooling alone — **also the true decode-time recall** |
| block-64 KV, q-block-128 shared | what actually ships for prefill |
| block-64 KV, q-block-32 shared | is the kernel-efficiency trade worth it |

`BlockMask` internally stores `kv_num_blocks` + `kv_indices`, which *is* the `topk` output format —
construct it directly rather than materializing a boolean mask, with the diagonal block as a
partial block carrying the causal `mask_mod`.

Minor known waste: with packed one-doc-per-row data, a 64-block straddling a document boundary
spends budget on masked tokens. Harmless at 32K with mostly-long docs; revisit only if the recall
probe says otherwise.

### 4.3 Decision — always-keep set

Force-include the **first 64 tokens** (attention sinks) and a **128-token local window** on top of
the selected blocks, both inside the budget. Cheap, and it removes the most common catastrophic
failure mode.

### 4.4 Gradients

The `topk` is hard and non-differentiable — **do not backprop through selection.** The indexer's
gradient comes only from the KL loss (Phase 1: full-row KL; Phase 2: selected-set KL), exactly as
V3.2 does and as `_sparse_indexer_kl` already implements. Pooling therefore changes nothing about
the optimizer, the loss, or the warm-start path.

### 4.5 Phase 0 exit criteria

New tests in `tests/dsa/`, following the existing naming:

1. **`top_k ≥ L` reproduces stock Qwen3 logits** to within bf16 noise. This is the faithfulness
   control the whole eval ladder rests on.
2. **Indexer RoPE == base RoPE**, asserted on tensors, not by inspection.
3. **FP8 UE8M0 parity** vs. the serve-time quant (port `test_indexer_fp8_ue8m0_parity.py`).
4. **Block-vs-token recall gap** measured at 4K on Qwen3-0.6B.
5. `dsa_overrides_from_config` / flat `dsa_*` override keys wired for Qwen3.
6. Tied-embedding consolidate/reload round-trip.

---

## 5. Phase 1 — dense warm-up, indexer only, at 32K

Same shape as MiniCPM3 Phase 1. Base **fully frozen**, attention stays **dense**, only the indexer
trains. Loss = `KL(softmax(I) ‖ head-summed, L1-normalized dense attention)`.

The base model's outputs are bit-identical to stock Qwen3 throughout, so this phase carries **zero
risk** to model quality — **assert unchanged logits on a fixed batch** as a standing test.

| Knob | Value | Why |
|---|---|---|
| `seq_len` | **32768** | MiniCPM3 Phase 1 ran 32768 but Phase 2 dropped to 4096. Both phases run at 32K here. |
| `top_k` | 2048 (metrics only; attention is dense) | Matches the deploy budget; k/L = 6.25% at 32K. |
| indexer LR | 1e-3, cosine, 10% warmup | Validated Phase-1 setting. |
| `kl_block_size` / `kl_checkpoint` | 512 / **true** | Required at 32K. Tile is `[1, 32, 512, 32768]` fp32 ≈ 2.1 GB — *smaller* than the proven MiniCPM3 case (62 layers × 40 heads at 32768). |
| `fp8_ue8m0` | **true** | See §3.1. |
| tokens | ~0.5–1B | Base frozen, no base gradients → cheap (§7). |
| data | long real docs, **one doc per row**, **+ self-generated thinking traces** (§1.1), **retokenized with the Qwen3 tokenizer** | Reuse `examples/dsa/prepare_real_data.py`. Never reuse another model's `token_ids`. |

### Phase 1 gates — do not start Phase 2 until all pass

Measured on a held-out long-context val set (`verl/trainer/sft_trainer.py:355` already surfaces
these on val):

1. **`indexer/topk_recall ≥ 0.90`** at k=2048, L=32768, **and** the full recall-vs-k curve over
   {512, 1024, 2048, 4096} recorded — this curve *is* how the deploy budget gets chosen.
2. **Per-layer min recall ≥ 0.80.** The mean hides one broken layer.
3. **Recall bucketed by query position, by key distance, and by generated-trace position.** Plus
   the selected-key position histogram vs. dense attention argmax. **Recency collapse** — selections
   degenerating onto local window + sinks — is the failure that kills 32K, and it is visible at step
   200 instead of after a full Phase 2.
4. `nan_frac == 0`; indexer entropy neither collapsed nor uniform (see `docs/dsa_indexer_metrics.md`).

Smoke on Qwen3-1.7B at 4K before committing 4B GPU-days, mirroring
`examples/dsa/run_minicpm3_dsa_phase1_smoke.sh`.

### 5.1 The granularity sweep is free

Every §4.2 decision — pooling function, `Bk ∈ {32, 64, 128}`, `Q_BLOCK`, force-include set,
top-m-per-block variants — is a **post-hoc function of the token-level scores**. Sweep all of them
**offline against a Phase-1 checkpoint** using the existing `topk_recall` metric, at 4K–32K, before
writing a line of the Phase-2 kernel.

Deliverable: `scripts/dsa/probe_block_selection.py` — load a Phase-1 checkpoint, run the indexer on
held-out 32K docs, emit the four-point recall table from §4.2.

**Decision rule:** if block-64 / q-128 recall at k=2048 comes in **under ~0.85**, pay for the
token-granular Triton kernel instead of shipping block-pooled.

This reorders the risk: Phase 1 is cheap and unchanged, and it *hands you* the granularity decision
with measurements instead of a Phase-0 guess.

---

## 6. Phase 2 — sparse adaptation

Sparse attention on, warm-started from the Phase-1 consolidated indexer, `top_k` = deploy `top_k`.

**Phase 2 splits, because reasoning is more fragile than instruction-following.** The MiniCPM3
Phase 2 cost **−3.0 on HumanEval+ from behavior-cloning drift alone — more than double the −1.3
sparsity cost** (`docs/dsa_eval_report.md` §3). Reasoning chains are more brittle than
instruction-following, so on a thinking model that risk goes up. Meanwhile the realistic token
budget here (~1B) is three orders of magnitude below DeepSeek's 943B sparse stage — which is what
licenses *them* to unfreeze the base.

### 6.1 Phase 2a — sparse ON, base FROZEN, indexer only

Loss = **selected-set KL** (+ optionally LM loss, which still only trains the indexer).

- Base weights never move → **reasoning capability preserved by construction, zero drift, provably.**
- Collapses eval ladder rows 0 and 2 into one config (Phase-2a weights *are* the stock weights).
- Measures the **pure sparsity cost** with no confound whatsoever.
- Cheap — same ~4× saving as Phase 1 (no base gradients).

This answers the central question: *can the indexer alone carry 32K sparse attention on this
model?* If yes, ship without ever touching the base, and "preserves long-context capability" becomes
a statement about the attention mechanism rather than a hope about the data mixture.

There is a real chance 2a underperforms — DeepSeek unfreezes for a reason, and a frozen base gets no
opportunity to adapt to sparse inputs. Finding that out costs a fraction of a 2b run and makes the
result interpretable either way.

### 6.2 Phase 2b — unfreeze the base (conditional on 2a)

Only if 2a's sparsity cost is unacceptable. Then:

| Knob | Value | Why |
|---|---|---|
| loss | **LM loss on real docs (dominant) + selected-set KL + a *smaller* BC self-distillation share** | The V3.2 recipe is LM loss + selected-set KL, no teacher distillation. LM loss on real docs has no drift because it is the pretraining objective. BC is what caused the MiniCPM3 code regression — make it a minority of the mixture, not the whole diet. |
| base LR | **5e-6 – 1e-5** | Low; drift is the enemy. |
| indexer LR | 1e-3 → decay to ~1e-4 | Continues from Phase 1. |
| length mixture | 40% short (<4K) / 30% decode-long / 30% prefill-long — see [data_plan.md](data_plan.md) §5 | All-short reproduces the Phase-2-at-4096 mistake (indexer never sees 6% selection ratios); all-long regresses the short-context scorecard. |
| code share | upweighted from the start | Code is both the most 
-fragile and the most sparsity-sensitive domain. The `--exclude-sha` net-new selection machinery already exists. |
| `top_k` curriculum (optional) | anneal 8192 → 2048 over the first ~30% | Sparse-from-scratch at 6% is the harder optimization problem, and the budget knob is free. |

**Thinking traces are a Phase-2 asset**, not just a cost: self-generated CoT at 8–32K is naturally
long-context training data in exactly the decode-time distribution that matters.

### 6.3 Phase 2 gates

The full ladder, length × top_k grid, and thinking-mode protocol — see
[eval_plan.md](eval_plan.md).

---

## 7. Compute budget (8×H100 80GB)

**Memory is not the constraint at 4B.** Full FSDP finetune ≈ bf16 weights 8 GB + grads 8 GB +
AdamW fp32 states 48 GB ≈ 64 GB → **~8 GB/GPU**. The 32K activations and KL tiles dominate, and the
KL tile (2.1 GB) is smaller than the already-proven MiniCPM3 case.

**Phase 1 and 2a are ~4× cheaper per token than 2b.** With the base frozen, gradients flow only
into indexer params and `hidden_states` are constants, so there is **no backward through the base at
all**: cost ≈ one base forward (needed anyway for the dense KL target) + indexer fwd/bwd.

| Phase | Tokens | Rough wall-clock |
|---|--:|---|
| 1 (dense warm-up, base frozen) | 0.5–1B | ~½ day |
| 2a (sparse, base frozen) | 0.5–1B | ~½ day |
| 2b (sparse, base unfrozen) | 1–2B | ~1–2 days |

**These are FLOP-derived, not measured.** Pin them with a 50-step timing run before scheduling.
Anchor: MiniCPM3-4B is the same parameter class, so existing step times bracket this — adjusted for
36 layers instead of 62, GQA instead of MLA, and 32K instead of 4096.

---

## 8. Reuse map

| Component | Status |
|---|---|
| `dsa_indexer.py` score path, FP8/UE8M0, Hadamard rotation, `select_topk` | **reuse**, minus the MLA-latent coupling (§2) |
| `_dense_warmup_kl`, KL tiling, `kl_checkpoint`, `install_kl_accumulation`, `_causal_doc_bias_block` | **reuse ~as-is** (add `repeat_kv` for GQA) |
| `_sparse_indexer_kl` (selected-set KL) | **reuse** |
| `indexer/topk_recall`, `topk_overlap`, entropy diagnostics | **reuse** — and promote to a length-swept offline probe (§5.1) |
| `consolidate_indexer_ckpt.py`, `randomize_indexer_ckpt.py`, warm-start plumbing, Phase-1/2 launch scripts | **reuse**, re-parameterized |
| `prepare_real_data.py` (one-doc-per-row) | **reuse**, retokenized for Qwen3 |
| `_sparse_attn` (MLA), FlashMLA-sparse serving kernel, 40→64 head pad, latent-576 padding | **discard** — new block-sparse GQA path, training + vLLM |
| `build_vllm_serving_dir.py` | **adapt** — and note it omits `index_topk`, which silently serves dense (`docs/dsa_eval_report.md` §2) |

---

## 9. Open risks

1. **Block-pooled recall may be insufficient** (§5.1 decision rule). Mitigated by measuring it
   offline from Phase 1 before Phase-2 kernel work; fallback is a token-granular Triton kernel.
2. **q-axis sharing in `BlockMask`** is a second, easily-overlooked recall loss; affects prefill
   only (§4.2).
3. **FlexAttention + FSDP + `torch.compile`** interaction is historically finicky. Validate on
   Qwen3-0.6B in Phase 0 before depending on it.
4. **Phase 2a may underperform**, forcing 2b and its drift risk (§6.1).
5. **Eval cost with thinking mode** is the largest schedule risk — 10–50× output tokens × multi-
   sample × 6 ladder rows × 4 lengths. Argues for getting CUDA graphs working via the paged
   block-sparse path rather than repeating `enforce_eager`.
6. **`index_topk` serving gate**: a freshly-built serving dir silently serves **dense**. Re-verify
   at every length; at long context the resulting good scores would be misread as success.
7. **Qwen3-4B-Thinking-2507 is not on disk yet.**
