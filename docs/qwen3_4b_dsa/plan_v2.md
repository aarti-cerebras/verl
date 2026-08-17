# Qwen3-4B GQA → DSA (token-granular, no MLA) — plan v2

**Target:** Qwen3-4B-Thinking-2507 (`/cb/ml-eng/aarti/models/qwen3_4b_thinking_2507`), dense GQA →
**DeepSeek Sparse Attention**: a lightning indexer scores every (query, key) *token* pair, the top-k
**tokens** are selected, and the main attention runs over only those. No MLA is introduced.

**Status:** plan, ready to implement. Supersedes `plan.md` in this directory (v1, never implemented).
Written 2026-08-14 and revised the same day against the **Kwai Keye-VL-2.0** precedent (arXiv 2606.10651) —
the only published DSA-on-GQA system — and against its released SGLang/EffectiveKernels sources.

Three decisions were settled during that review and are load-bearing everywhere below:

1. **Full isolation** (§6). Nothing shared is modified: `dsa_indexer.py`, `minicpm_dsa.py`, `msa_indexer.py`
   and `qwen3_msa.py` are read-only references, the FP8/Hadamard helpers are **copied**, and MiniCPM3-DSA
   and Qwen3-MSA keep running untouched. Exactly two shared files get additive edits.
2. **Serve on vLLM** (§5, §10.6), with Keye as the reference for numeric choices only. It needs no MLA
   (§5.0) and, unlike Keye's route, imposes no GQA-ratio constraint.
3. **The Phase-2 teacher is free and it is the published recipe** (§4.1) — `minicpm_dsa`'s dense recompute
   is the deviation, on both cost and target.

Read **§9** for Keye's shipped indexer config and verbatim equations, and **§5.1** for the serving
constraints the indexer config must satisfy at construction time.

> **Why revisit.** The MSA detour is measured: `docs/qwen3_4b_msa/eval_scorecard_all_ckpts.md` — k16
> (2048 tokens/query, block-granular) beats k8 on all 13 benchmarks and reaches dense parity at short
> context, but still loses **RULER 32K by 6.36 points**. Token-granular selection at the *same* 2048-token
> budget is the obvious next lever: a 128-token block is selected for the sake of one hot token and spends
> the other 127 slots on whatever sits next to it. This plan tests that at equal parameter budget
> (98.2M indexer params vs MSA's 97M) and equal selection budget (2048 tokens).

---

## 1. What actually has to change vs. the two existing implementations

The repo already contains both halves of this port; almost nothing structural is new.

| Concern | Source to reuse | Change needed |
|---|---|---|
**Everything in the "source" column is a read-only reference.** Per the isolation rule in §6, no shared
module is modified — the numerics helpers are copied, not imported, so MiniCPM3-DSA and Qwen3-MSA keep
running byte-identically.

| Concern | Source to read | What differs in ours |
|---|---|---|
| Lightning indexer (FP8/UE8M0, Hadamard, ReLU-dot, per-head gate, top-k) | `verl/models/transformers/dsa_indexer.py` | **Query source:** MLA hands the indexer a free compressed latent `qr`; GQA has none, so `wq` projects straight from hidden states — and a per-head `q_norm` replaces the normalization MLA's `q_a_layernorm` used to provide upstream. Own 64-dim rotary instead of the base's tables. |
| Qwen3 attention plumbing (transformers 5.x forward mirror, `position_ids` side channel, tiling, checkpointing, teacher compile, diagnostics) | `verl/models/transformers/qwen3_msa.py` | Swap block-granular per-group selection for token-granular shared selection, and the per-group teacher for one head-averaged teacher. |
| Phase-1 KL, Phase-2 sparse attention, gradient wiring | `verl/models/transformers/minicpm_dsa.py` | GQA-ify (`key_states` is `[b, H_kv, T, d]`), and **do not** copy `_sparse_indexer_kl` — its dense recompute is both off-recipe and 8× the cost (§4.1). |
| FSDP2 indexer wrap (Option B2), engine `batch_num_valid_queries`, `indexer_kl_loss` / `dsa_sparse_loss`, SFT loss modes, datasets | `fsdp_utils.py:593`, `transformer_impl.py:670`, `workers/utils/losses.py`, `trainer/sft_trainer.py:176` | **Nothing** — all gated on `config.dsa_enabled` and the `_dsa_*` attribute names, which we reuse deliberately. One additive line in `fsdp_utils.py:592` for the new class name (§6). |
| Data | `/cb/ml-eng/aarti/msa/data/…` | **Nothing.** Phase-1 and Phase-2 corpora are already built for this exact tokenizer/model. |

So the new code is: one indexer module, one integration module, one vLLM backend + serving plugin, tests,
launch scripts. Two shared files get additive edits; nothing existing changes behaviour.

---

## 2. Architecture

Qwen3-4B geometry (from `config.json`): `L=36`, `hidden=2560`, `H_q=32`, `H_kv=8`, `d_h=128`,
`rope_theta=5e6`, `max_position_embeddings=262144`, `rms_norm_eps=1e-6`, full rotary
(`partial_rotary_factor` unset → RoPE over all 128 dims).

### 2.1 Indexer, per layer

```
q_idx   = RoPE(wq(x))                       # [b, T, 16, 64]     direct from hidden (no MLA latent)
k_idx   = RoPE(k_norm(wk(x)))               # [b, T, 64]         single MQA key head, shared
w       = weights_proj(x.float()) * 16^-0.5 # [b, T, 16]         fp32
I[t,s]  = sum_j (w[t,j] * d_idx^-0.5) * ReLU(<q_idx[t,j], k_idx[s]>)
S_t     = TopK_s( I[t, :] + causal_doc_bias, k=2048 )
```

Sizing is taken from the **Keye-VL-2.0 precedent** (§9) wherever it published a number, since it is the
only existing DSA-on-GQA system: `sa_config = {indexer_num_heads: 16, indexer_head_dim: 64,
indexer_num_kv_heads: 1, topk: 2048, q_chunk_size: 512, kv_chunk_size: 512}`.

| Knob | Value | Why |
|---|---|---|
| `n_heads` (indexer query heads) | **16** | Keye's `indexer_num_heads`, and the existing `DSAConfig` default from the MiniCPM3 port. Two independent DSA ports landed on 16. |
| `head_dim` (`d_idx`) | **64** | Keye's `indexer_head_dim`, and again the MiniCPM3 default. Halves the indexer's FLOPs vs 128 → `n_idx*d_idx / (2*H_q*d_h) = 1024/8192 = 12.5%` of dense attention (V3.2 sits at 25%). Also the dim our serving indexer already handles: `scripts/dsa/vllm_minicpm3_dsa/indexer.py` was written and parity-tested for exactly `16 × 64`, padding to `32 × 128` for the DeepGEMM kernel. |
| query source | **direct `wq: 2560 → 16*64`** | Keye: *"`q^I_{t,j}` and `w^I_{t,j}` are derived from `h_t`"* — no bottleneck rank appears in `sa_config`. MLA's `wq_a→norm→wq_b` existed only because MLA already had a query latent lying around; on GQA a bottleneck buys nothing at `d_idx=64` (2.62M params direct). |
| `q_norm` | **`RMSNorm(64, eps=1e-6)`, per head** | Keye `keye_indexer.py:156`. Neither DeepSeek's indexer nor our MiniCPM3 port has one — because MLA's `q_a_layernorm` sits upstream of the indexer query and on GQA nothing does. So this is not an addition Keye invented, it is a *relocated* norm: from a shared latent to per-head after projection. **Not retrofittable after Phase 1** (see §10.5). |
| `k_norm` | `LayerNorm(64, eps=1e-6)`, fp32 | Same in DeepSeek, Keye, and our port. Note the deliberate asymmetry with `q_norm` (RMS vs Layer) — both precedents have it. |
| `rope_head_dim` | **64 — all of it, own rotary** | **Mirror the attention being distilled.** DeepSeek ropes 64 of its indexer's 128 dims because *MLA's own attention is only partly position-dependent* — its score splits into `q_nope·k_nope` (128 content dims) + `q_pe·k_pe` (64 position dims), so the indexer's nope/pe split mirrors the function it approximates. Same for MiniCPM3 (`qk_rope_head_dim=32` of `q_head_dim=96`), which our port mirrors at 32 of 64. **Qwen3's attention ropes every dimension** (`head_dim=128`, `partial_rotary_factor` unset → `rotary_dim=128`), with no content-only channel anywhere — so the indexer should have none either. Keye, the only DSA-on-GQA precedent, ropes all of it on a model whose attention is likewise fully rotary. The invariant is the structural match, not the fraction: anchoring on "DeepSeek used half" copies a number instead of its reason. The content-channel worry does not survive that argument (Qwen3's attention lacks one too) and is small anyway: at `theta=5e6` the slowest frequency pair rotates ~0.01 rad across 32K tokens, so low-frequency dims already behave like content channels. **Its own rotary, never a slice of the base tables** — that part is correctness, not preference: `cos[..., :64]` holds 64 *distinct* frequencies but `rotate_half` on a 64-dim vector needs 32 duplicated. A 64-dim rotary at the same `theta` gives exactly the base's even-indexed frequencies, spanning the same spectrum as the attention it distills. Also simpler: no nope/pe split to get wrong. **Ablation:** `rope_head_dim = 32` (DeepSeek's *fraction*), one config line. |
| `top_k` | **2048 tokens** | Keye's `topk`, DeepSeek's `index_topk`, *and* apples-to-apples with MSA k16 (16 × 128). Three reasons for the same number. Sweep 1024 / 512 to price the granularity win. |
| `fp8`, `fp8_ue8m0`, `rotate_activation` | **True, True, True** | Train against serve-time numerics from step 0. `fp8_ue8m0=False` cost the MiniCPM3 run ~2% selection drift (memory `dsa-fp8-ue8m0-fix`). |
| forced local / sink tokens | **none** | Faithful DSA; Keye mentions neither. Log the local-mass diagnostic; if Phase-1 recall stalls, forcing the last 128 tokens is the cheap fix. |
| sparse layers | **all 36** | DeepSeek and Keye both sparsify every layer (Keye reports no dense prefix). `dense_prefix` stays available; decide from Phase-1 per-layer recall (MSA's block-level probe capped *layer 3* at 0.767 — token-granular may not have that problem, and that is worth knowing). |
| query/key tiling | **512** | Keye's `q_chunk_size`/`kv_chunk_size`, and independently our `kl_block_size=512`. |

**Params:** `wq 2.62M + wk 0.16M + weights_proj 0.04M ≈ 2.83M/layer × 36 = 101.7M`
= **2.5%** of the 4.02B backbone (MSA: 97M — close enough for a fair comparison).

### 2.2 One shared selection, not per-group

`S_t` is **one token set per query, shared by all 32 query heads** (DeepSeek-faithful). Not per-KV-group
as in MSA. Keye-VL-2.0 does the same and says so explicitly — *"We apply the same sparse index set `Ω_t`
to all groups"* (Eq. 3) — which is the strongest available evidence for this call, since it is the one
design fork where MSA points the other way.

Three reasons: (a) Keye's precedent, at 256K, reported lossless; (b) vLLM's whole sparse path is built on
a `topk_indices_buffer` of shape `[num_tokens, topk]` — per-group selection means
`[num_tokens, H_kv, topk]` and a rewritten kernel interface; (c) it makes the Phase-1 teacher a single
`[b, T_q, T]` distribution instead of MSA's `[b, H_kv, T_q, T]`, an 8× memory cut on the dominant tensor.
MSA's Appendix-A finding that groups *do* diverge is the argument on the other side; it is the documented
follow-up lever, not the v1 design.

**Caveat specific to Qwen3-4B:** Keye has `H_kv=4` (4 groups × 8 heads); Qwen3-4B has `H_kv=8`
(8 groups × 4 heads). Twice as many group teachers must be satisfied by one shared index set, so the
shared-selection assumption is a somewhat bigger ask here than in the paper. That is what the
group-divergence diagnostic in §3 is for.

### 2.3 Cost accounting (T=32K, per layer)

| | FLOP/layer (train) | Decode bytes/token/layer |
|---|---|---|
| dense GQA attention (causal) | 8.80 T | 134 MB (K+V, bf16) |
| indexer scores (`O(T²)`), `16 × 64` | 1.10 T | 4.3 MB (fp8 index-K cache, **padded** — see below) |
| sparse attention @ k=2048 | 1.10 T | 8.4 MB |
| **DSA total** | **2.20 T (4.0× less)** | **12.7 MB (10.6× less)** |

At 128K: 17.6 + 4.4 vs 140.7 TFLOP and 25 MB vs 537 MB (**21×**).

**The serve-time indexer costs 4× its training cost, because of kernel padding.** Stock DeepGEMM's
`mqa_logits` accepts head counts in `{32, 64, 128}` at `head_dim = 128`
(`vllm_minicpm3_dsa/indexer.py::SUPPORTED_KERNEL_HEADS`), so a `16 × 64` indexer is zero-padded to
`32 × 128` at serve time. The padding is provably lossless — zeros leave the dot product unchanged *and*
leave the per-row `amax` unchanged, so the FP8 scale is identical, and padded heads get zero q rows and
zero gate weights (`w·ReLU(·) = 0`). But it means 4096 MACs/pair instead of 1024, and the **padded** key
is what gets cached: 132 B/token/layer, not 68. Hence the 4.3 MB above.

`16 × 64` is still the right choice: the cheapest natively-supported config, `32 × 128`, costs 4× in
*training* (4.4 TFLOP/layer = 50% of dense attention) and issues identical kernel work at serve. Strictly
dominated. Keye runs `16 × 64` unpadded via their `DeepGEMM` fork, which asserts `head_dim ∈ {32,64,128}`
where stock constrains the head *count* — so the padding tax looks recoverable later by linking a patched
DeepGEMM. Not a v1 concern.

**Do not quote these ratios as expected speedups.** Keye *measured*, on the same `k=2048` at 128K,
">3× prefill and >5× decode" cost reduction — well under the byte/FLOP counts, because real decode is not
purely KV-bandwidth-bound and the token-granular gather does not run at peak. The qualitative shape holds:
the win is decode, and the indexer's `O(T²)` term is what caps the prefill win.

### 2.4 Initialization — re-derived, because adding `q_norm` invalidates the inherited recipe

**Do not copy `dsa_indexer.py::reset_parameters`.** Its rationale ("each linear starts at ~half-unit
variance, so scores are small and `softmax(I)` starts near-uniform") describes a network *without* a
normalized query. With `q_norm` on q and `k_norm` on k, **both sides are normalized**, so `wq`'s and
`wk`'s init std no longer influence the forward score scale at all — RMSNorm and LayerNorm are invariant to
row scaling. What each init actually controls now:

| parameter | affects forward score scale? | what its init controls |
|---|---|---|
| `wq` | **no** (RMSNorm is scale-invariant) | gradient geometry / effective LR on `wq` |
| `wk` | **no** (LayerNorm likewise) | effective LR on `wk` |
| `q_norm.weight` | yes | init `1` — standard RMSNorm, matching Keye's `RMSNorm(head_dim, eps=1e-6)` |
| `k_norm.weight` / `.bias` | yes | init `1` / `0` (identity) |
| `weights_proj` | **yes, solely** | **the entire initial score scale, hence the initial entropy** |

**Derivation.** At init with unit gains: `q` rows have unit RMS → `‖q_j‖ = 8`; `k` has unit per-element
variance → `std(q_j·k) = 8`; `std_s(ReLU(q_j·k)) = 8·√(½ − 1/2π) = 4.67`. Only the *variation* of `I` across
keys survives softmax, so with `w_rms = std_W·√hidden·rms(x)·n^{-1/2}`:

```
σ_I ≡ std_s(I) = softmax_scale · √n · w_rms · 4.67 = 2.335 · w_rms
entropy_frac  ≈ 1 − σ_I² / (2 ln N)        (N = 32768 → ln N = 10.4)
```

**Why the inherited init fails here.** `σ_I ∝ rms(x)`, and `rms(x) ≈ rms(input_layernorm.weight)`.
Measured on this checkpoint, that gain RMS spans **186×** across the 36 layers:

| layer | 0 | 4 | 12 | 18 | 24 | 30 | 34 |
|---|---|---|---|---|---|---|---|
| `rms(gain)` | 0.025 | 0.122 | 0.331 | 0.623 | 1.095 | 2.298 | **4.709** |
| `σ_I` (inherited) | 0.007 | 0.036 | 0.097 | 0.182 | 0.320 | 0.671 | **1.375** |
| `entropy_frac` | 1.0000 | 0.9999 | 0.9996 | 0.9984 | 0.9951 | 0.9784 | **0.9091** |

Harmful at both ends: layers 30–35 start visibly **committed** (0.91–0.98), while layers 0–6 start with
scores of essentially **zero** — and since `wq`/`wk` receive gradient *only* through `w ∝ rms(x)`
(`dsa_grad_norm_debugging.md` issue #2), those layers train ~100× slower. It also defeats
`kl_reduction=mean`, whose whole purpose is a readable `grad_norm`: the layer-mean is dominated by the few
late layers that start peaked. Plausibly a contributor to the MiniCPM3 Phase-1 runs sitting at
`grad_norm ≈ 330` against `clip_grad = 1.0`.

**The fix is one factor.** Solving `σ_I = σ_target`:

```python
# hidden_rms = rms(layer.input_layernorm.weight) — read at attach time; no data, no forward pass,
# deterministic, and before the FSDP wrap, so it stays world-size agnostic like the warm-start path.
nn.init.normal_(self.wq.weight, std=0.5 * hidden_size**-0.5)   # forward-invisible; sets grad scale
nn.init.normal_(self.wk.weight, std=0.5 * hidden_size**-0.5)   # forward-invisible
nn.init.ones_(self.q_norm.weight)
nn.init.ones_(self.k_norm.weight); nn.init.zeros_(self.k_norm.bias)
std_w = SIGMA_TARGET * n_heads**0.5 / (2.335 * hidden_size**0.5 * hidden_rms)   # SIGMA_TARGET = 0.3
nn.init.normal_(self.weights_proj.weight, std=std_w)
```

For Qwen3-4B that is `std_w = 0.0102 / rms(gain_ℓ)` — i.e. **keep the inherited `0.5/√hidden ≈ 0.00988` and
divide by the layer's `rms(gain)`**, which is exactly why it looked correct at mid-depth (`rms ≈ 1` near
layer 24) and drifted 186× either side:

| layer | 0 | 12 | 24 | 34 |
|---|---|---|---|---|
| `std_w` | 0.408 | 0.0308 | 0.00932 | 0.00217 |

`SIGMA_TARGET = 0.3` → `entropy_frac ≈ 0.996`. Anything in `[0.2, 0.5]` works; **equality across layers is
the property that matters, not the value.** The error is asymmetric: too small is self-correcting, because
`dL/dI = softmax(I) − p` is dominated by the peaked teacher, so `weights_proj` receives an O(1) gradient
regardless of `w` and grows to unblock `wq`/`wk`. Too large starts committed and can saturate.

**No weight decay on `q_norm.weight`, `k_norm.weight`, `k_norm.bias`, or `weights_proj.weight`.** Decay
pulls the norm gains toward 0, suppressing the branch — strictly worse than MSA's Gemma-style
parameterization, where decay pulls the gain toward 1 and is harmless. And decaying `weights_proj` toward 0
severs the only gradient path into `wq`/`wk`.

*As implemented (2026-08-15):* the FSDP engine owns optimizer construction
(`transformer_impl.py::_build_optimizer`) and its existing param-group branch is shared with the MiniCPM3
Phase-2 path, so splitting groups there would change behaviour for an existing run — which the §6 isolation
rule forbids. Both launch scripts therefore set **`optim.weight_decay=0.0` globally**. For Phase 1 that is
not a workaround but the correct setting: every trainable parameter is a norm gain, the gate, or a
projection whose scale is forward-invisible because a norm follows it, so decay has nothing useful to act
on. For Phase 2 (base unfrozen) it is a deliberate simplification; `qwen3_dsa.indexer_param_groups` is
written and unit-tested for the day base decay is wanted, and is the helper `_build_optimizer` should call.

**Gate:** per-layer `entropy_frac ∈ [0.99, 1.0]` on the first forward. No new machinery — it is already
computed on forward #1 (`_do_diag` is set there) and already emitted per layer under `log_per_layer=true`.
Make it a unit test *and* a smoke-run check.

*Caveats:* `rms(x) ≈ rms(input_layernorm.weight)` assumes the normalized residual is isotropic across
channels; in practice the gains correlate with per-channel variance, so the true spread is probably somewhat
compressed from 186×. The Gaussian/independence steps above are approximations too. Neither weakens the
conclusion — the prescription is "equalize per layer", and the gate measures the real quantity rather than
trusting the estimate.

**Dead keys:** `ReLU` leaves each head inactive for ~half the keys, so a key with all heads inactive gets
exactly `I = 0`. At `n_heads = 16` that is `2⁻¹⁶ ≈ 1.5e-5`, ~0.5 keys per 32K row — negligible. It would be
6% at 4 heads, so this is a further point in favour of 16.

---

## 3. Phase 1 — indexer warm-up (dense, zero capability risk)

Base frozen, attention stays dense, LM output **bit-identical** to stock. Only the indexer trains, on a
side-channel KL. Loss (per layer, averaged over valid query rows, `mean` over layers):

```
p[t,s]  = (1/H_q) * sum_h softmax_s( q_h[t]·k_{g(h)}[s] * scaling + bias )[s]    # detached teacher
L_KL    = KL( p[t,:] || softmax(I[t,:]) )
```

Per-head softmax **then** the average over all 32 heads — the softmax is inside the sum. Compute it by
accumulating one KV group at a time (`q_group [b,4,T_q,d] × k[:,r]`), summing over the group's 4 heads,
accumulating across the 8 groups, then dividing by `H_q`: peak extra is one group's `[b,4,T_q,T]` bf16
(268 MB at 32K/512) and the retained result is a single fp32 `[b,T_q,T]` (67 MB).

This matches **DeepSeek §2.1.1 verbatim** — *"we first aggregate the main attention scores by summing
across all attention heads. This sum is then L1-normalized along the sequence dimension to produce a
target distribution."* Each head's softmax sums to 1, so the head-sum has L1 norm `H` and dividing by it
*is* the L1 normalization. `minicpm_dsa`'s `+= softmax; /= H` is that sentence transcribed; we keep it and
only reshape the loop for GQA (8 group matmuls of 4 heads each instead of 32 single-head matmuls — same
arithmetic, 4× fewer launches, and the form that `torch.compile`s well).

**It is also gradient-equivalent to Keye's Eq. 4, at 1/G the memory.** Keye writes one teacher *per group*
against the shared indexer distribution, `L = Σ_t Σ_g KL(p_g ‖ q)`. Cross-entropy is linear in its first
argument and the teachers are detached, so the whole thing decomposes exactly:

```
Σ_g KL(p_g ‖ q)  =  G · KL(p̄ ‖ q)  +  G · JSD(p_1 … p_G)      p̄ = mean_g p_g
                                        └── constant in q ──┘
```

Three things fall out of that, and they are the reason this is written down rather than just done:

1. **`q = p̄` is the exact minimizer.** When the student is shared across groups, Keye's per-group loss is
   *already asking for the head-average*. Averaging first is not a cheaper approximation of Eq. 4 — the
   average is the target Eq. 4 defines. This is the principled argument for shared selection (§2.2), which
   otherwise rests only on precedent and buffer layout.
2. **One `[b,T_q,T]` tensor instead of eight** — 67 MB vs 537 MB per tile at 32K/512, on the tensor that
   dominates Phase 1. Memory only: forming `p̄` still needs all 32 per-head softmaxes, so no FLOPs are saved.
3. **The loss has an irreducible floor of `G·JSD`**, so it is not a progress metric — hence the gate is
   `topk_recall`, not KL. And that `JSD` term *is* the group-divergence diagnostic below: the exact measure
   of how much the 8 groups disagree, i.e. how much stress shared selection is under.

The leftover `G` factor is a pure gradient rescale. AdamW divides by its own running magnitude so a
constant washes out; under saturated clipping the norm does too. It bites only in the transition band,
where it changes *which* steps clip — and it makes `grad_norm` unreadable. That, not the optimum, is why
`kl_reduction=mean` is the default. Same reason `λ` is near-inert in Phase 2 (memory
`msa-phase2-kl-lambda-nearly-inert`; do not sweep it). Our logged KL is `1/G` of Keye's *and* omits the
`JSD` term, so never compare magnitudes across papers.

One ambiguity flagged rather than assumed: the paper says it *"aggregate[s] dense attention scores within
each group, normalize[s] them"*. Read post-softmax (sum G distributions, renormalize → the mean) that is
exactly the above. Read pre-softmax (sum logits, then softmax) it is not, and the two differ whenever
heads within a group disagree. Post-softmax is the defensible reading and the one MSA's `kl_loss.md`
independently argued for; note it as an open question if Phase-1 recall underperforms.

Carry over three hard-won implementation rules from `qwen3_msa.py` verbatim — each was a measured bug:

1. **Pre-slice the query tile before the compiled teacher.** Passing `q0`/`q1` as ints into a
   `torch.compile`d function makes Dynamo specialize per tile, blows the recompile limit at tile 8, and
   silently runs the remaining 56 tiles eager (`_group_teacher_impl` docstring).
2. **Build the bias/teacher *inside* the checkpointed tile function.** `torch.utils.checkpoint` retains
   every input tensor; passing a freshly-allocated per-tile `[1,T_q,T]` bias in cost ~142 GB across
   36 layers, and shrinking `kl_block_size` does not help because the total is `n_tiles × per_tile`.
3. **`position_ids` and `attention_mask` come down a side channel** (`cfg._position_ids`), because
   transformers 5.x `Qwen3Attention.forward` receives pre-gathered `position_embeddings`, not
   `position_ids`. Without them the causal/document/padding support is wrong and the teacher normalizes
   over a different key set than the real attention.

**Config:** `kl_block_size=512`, `kl_checkpoint=True` (required at 32K), `kl_reduction=mean`,
`compile_teacher=True`, `diag_interval=10`, `log_per_layer=true`, `GRAD_CKPT=False` (base is frozen, and
recomputing a decoder layer re-fires the `_dsa_kl` side effect), `LR=1e-3` cosine. **Per-layer init per
§2.4, and no weight decay on `q_norm`/`k_norm`/`weights_proj`** — the optimizer param groups have to be
built accordingly, which is a launch-script change, not just a config value.

**Data / budget:** `/cb/ml-eng/aarti/msa/data/longmino_qwen3_32768` — already built: 92,180 windows of
32768 tokens (3.02B), one document per row, Qwen3-tokenized. **Budget 2B tokens** (61,036 windows), gate
at 1B and stop early if it passes. 2B is the converged number across both precedents: Keye's warm-up is
*"approximately 2B multimodal tokens"* and DeepSeek-V3.2's is 2.1B. Our earlier 1B figure came from the
MSA run, not from a DSA precedent.

**Diagnostics** (token-level, from `minicpm_dsa.py`): `indexer/topk_recall` (teacher mass inside the
indexer's top-k — *the* number), `topk_overlap`, `entropy_frac` (≈1.0 at init = healthy), `attn/entropy`,
`score_mean/std`, `nan_frac`, plus two new ones: `local_mass` (fraction of teacher mass in the last 128
keys, i.e. is recency being learned) and **`group_divergence`** — per-group recall of the *shared* top-k
against each group's own teacher, min over groups. The shared-selection design (§2.2) assumes one index
set can serve all 8 groups; this is the metric that falsifies it, and Qwen3-4B has 2× Keye's group count.

**Gate:** `topk_recall ≥ 0.95` at k=2048 on held-out 32K windows, every layer. Token-granular at 2048
should clear this comfortably — MSA's *block*-level oracle ceiling at the same budget was 0.979, and
token granularity raises the ceiling. If a layer stalls, that layer is a `dense_prefix` candidate.

**Budget:** 2B tokens ≈ 7,630 steps at 8 windows/step on 8×H100. Expect 45–55 s/step (MSA measured 42 s
with an 8× larger teacher; ours has a `1/G` teacher and a `16 × 64` indexer) → **4–5 days**, with the 1B
checkpoint at ~2 days as the early-exit gate. Peak memory should land below MSA's measured 46.9 GB;
measure before assuming `ACT_OFFLOAD` is unneeded.

**Optional accelerator, if the teacher dominates as it did for MSA (20.4 s of a 73 s step):** the KL has a
closed-form gradient, `dL/dI = softmax(I) − p`, so the whole `O(T²)` teacher can be kept out of the
autograd graph with a hand-written backward. Two existing implementations to copy from rather than derive:
Megatron-Core's `FusedDSAIndexerLoss` (`experimental_attention_variant/dsa.py`, memory
`dsa-is-mla-only-both-ends`) and `github.com/lemyx/tilelang-dsa` (a TileLang warm-up indexer op that emits
the FA output and the KL together and computes the KL gradient in backward). Both are written for MLA's
MQA mode and need the GQA teacher swapped in. Do this only if measurement demands it — `qwen3_msa.py`'s
compiled teacher already bought 2.70×.

---

## 4. Phase 2 — sparse (train base + indexer)

The sparse path *replaces* the dense attention. Per query tile:

```
I            = indexer(stopgrad(x))                    # grad → indexer params only
S_t          = topk(I + bias, k).detach()              # stop-grad: top-k is non-differentiable
K_g, V_g     = gather(key_states, value_states, S_t)   # [b, H_kv, T_q, k, d]  grad → base
A            = softmax(q·K_g^T * scaling + bias_sel)   # fp32
out          = A @ V_g
p            = A.detach().mean over the 32 heads       # ← the teacher, FREE
L_KL         = KL( p || softmax(gather(I, S_t) + bias_sel) )
L            = L_LM + λ · L_KL
```

### 4.1 The free teacher IS the recipe — `minicpm_dsa` is the deviation

DeepSeek §2.1.1's sparse-stage loss is `L^I = Σ_t KL(p_{t,S_t} ‖ Softmax(I_{t,S_t}))`, with `p` defined
once, in the warm-up paragraph, as the main attention summed over heads and L1-normalized. **In the sparse
stage the main attention *is* sparse** — the model only computes attention over `S_t`, that being the point
of the stage. So the same construction applied to it yields: each head's softmax over `S_t`, summed over
heads, divided by `H`. That is the sparse attention's own softmax, detached and head-averaged. No dense
pass exists anywhere in that forward pass.

Two things confirm the reading. There is no dense distribution to truncate — it is never computed. And if
`p_{t,S_t}` meant the dense `p` sliced at `S_t`, it would not sum to 1 and the KL would be ill-defined; the
paper mentions no renormalization step because none is needed.

Per head, the identity is airtight: a softmax over a subset equals the dense softmax restricted to that
subset and rescaled, because the full normalizer is a common factor that cancels. It holds as long as the
raw per-key scores are unchanged between the two — true here, and worth a parity assert (it would break on
a different scale factor, an extra masking bias, a sink token, or a different temperature).

So `minicpm_dsa._sparse_indexer_kl` is the outlier, and in **two** ways at once:

- **Cost:** it runs a full dense `[b,T_q,T]` attention pass — 8.80 TFLOP/layer at 32K — purely to build a
  loss target. That is 8× the sparse attention it supervises, so a Phase-2 step ends up *more expensive
  than dense training*: the sparsity is simulated for the LM path while the loss re-adds the quadratic cost.
- **Target:** its numbers come from a softmax normalized over all `T` keys, so heads get weighted by how
  much of their mass landed inside `S_t`. The paper's construction gives every head an equal vote, because
  a sparse softmax has no knowledge of what lies outside `S_t`. Illustration with two heads holding 40% and
  88% of their mass inside the selection: `[0.42, 0.58]` (paper) vs `[0.30, 0.70]` (minicpm).

**Why minicpm was written that way** — worth knowing, because the first reason is structural and easy to
repeat: `_sparse_attn` computes the sparse softmax as a local inside its query-tile loop and returns only
`(attn_output, idx, indexer_scores)`, so by the time `_sparse_indexer_kl` runs as a separate function the
softmax is out of scope and it has no choice but to rebuild a target. (Its `_sparse_attn_and_kl` wrapper
fuses the two *calls*, but for checkpointing — not to share the softmax.) Secondarily: it read `p_{t,S_t}`
as "the warm-up `p`, subscripted", it had Phase 1's dense loop to hand, and at MiniCPM3's `top_k` of
128–256 the extra pass was cheap enough to be invisible.

**Therefore:** attention and KL must live in **one function, inside one `checkpoint()`**. That is not a
performance preference — split them and the softmax goes out of scope and the dense recompute comes back.
`qwen3_msa._sparse_tile` is the shape to copy (`p = attn_f32.detach().mean(...)`, from the fp32 softmax
before the bf16 cast so the target is not quantized, detached before averaging so no graph node is built).
The `Σ_g ≡ G·mean_g` identity from §3 applies here too, so one averaged teacher again replaces Keye's eight.

Keye unfreezes **all** parameters in this stage and uses `L_total = L_NTP + λ·L_sparse` (Eq. 7) — the same
shape as ours. It also confirms the detach: *"To reduce gradient interference, the indexer input is
detached from the computation graph."* Its Eq. 6 wording (*"truncated and renormalized over `S_t`"*) is the
one phrasing that reads like average-then-rescale, but its sparse stage runs sparse attention too, so a
full-support distribution is not available to it either; most likely descriptive prose. Flagged as
inference — we have read Keye's serving code, not its training code.

**Diagnostic to add:** per-head share of attention mass falling inside `S_t`. It is already in the sparse
softmax, costs nothing, and its spread across heads is exactly the quantity that says whether the two
teacher orderings would differ measurably on our data. It also converges to 1 as recall rises.

Gradient wiring (arXiv 2512.02556 §2.1.1), all three edges load-bearing:
- LM loss → base, through the gathered K/V. Indexer indices are detached, so the LM loss never reaches
  the indexer.
- KL → indexer only, because the indexer reads **detached** hidden states.
- The teacher is detached, so the KL cannot flatten the base's attention to cheat.

Details to carry over:
- Fuse attention + KL in **one** checkpointed function. A KL stashed as a side effect of a `no_grad`
  first pass carries no graph and contributes exactly zero gradient — silently.
- Guard the all-`-inf` row: a pad query would softmax to NaN and poison the LM path. Force slot 0 open.
- `full_support_kl_prob` (MSA's escape hatch): with small probability, compute the KL over the full
  causal support instead of `S_t`. The restricted KL gives no gradient about tokens that were *not*
  selected, so the ranking outside the top-k slowly decalibrates. Start at 0.0, keep the knob.
- `λ`: with `kl_reduction=mean` a paper-faithful λ needs `λ_ours = λ_paper × 36`. Note that per memory
  `msa-phase2-kl-lambda-nearly-inert`, λ is close to a no-op here (disjoint params + AdamW scale
  invariance) — **do not spend compute sweeping it**.
- Memory: `K_g`/`V_g` are `[1,8,512,2048,128]` bf16 = 2.15 GB **each**. Use `kl_block_size=256`.
  Phase-2 at 32K needs `ACT_OFFLOAD=True` (memory `msa-phase2-32k-needs-act-offload`) and ~1720 GB host
  RAM, which constrains node choice (`msa-phase2-host-ram-1720gb`).

**Data, already generated, no new work:**
- 2a short + decode-long BC: `…/qwen3-4b-thinking-2507__dolci-think-rl-32b__ph2b_full93889_L32768_20260730_231930`
- 2b prefill-long: `…/qwen3-4b-thinking-2507__longctx_tierA__L131072_20260811_022925`

**Gates:** short-context parity (GSM8K/MMLU-Pro/IFEval/LiveCodeBench within ~1 pt of dense) and
**RULER 32K within 2 points of dense** — i.e. beating MSA-k16's −6.36 at the identical 2048-token budget.
That comparison *is* the experiment.

---

## 5. Serving: vLLM, and why it works without MLA

**Decision: vLLM.** Not Keye's SGLang stack. Four reasons, in order of weight:

1. **The GQA-ratio blocker exists only on Keye's route.** Qwen3-4B is `H_q/H_kv = 4` → `topk_block = 32`,
   and EffectiveKernels instantiates only 16 and 64 → `TORCH_CHECK(false)` (§10.4). On vLLM's FA3 route
   there is no ratio constraint at all. Keye's route needs kernel work *before anything runs*; vLLM's does
   not. The ratio problem was never a DSA-on-GQA problem — it is a property of one dedup kernel.
2. **The eval harness is vLLM** (`/cb/ml-eng/aarti/dsa/evals/...`) — Qwen3 baselines, the MSA scorecard we
   are measuring against, the serving-dir builders, the multi-replica serve scripts. The project's whole
   point is an MSA-vs-DSA comparison at equal budget; different engines would put an engine change inside
   the comparison.
3. **We already have most of the vLLM indexer** — `scripts/dsa/vllm_minicpm3_dsa/indexer.py`, 478 lines,
   parity-tested, with the separate `wk`/`weights_proj` layout and the head/dim padding solved.
4. **Keye's route is three pinned forks** (SGLang branch + DeepGEMM fork + EffectiveKernels) on a branch
   named for one model release.

What we forgo is their adjacent-query dedup prefill kernel. That trick is **prefill-only** by construction
— it amortizes one KV load across `128/G` *adjacent query rows*, and a decode step has one query per
sequence, so there is nothing to amortize. On the decode path the two routes do the identical thing. Decode
is where DSA's win lives. And the pieces are separable anyway: **EffectiveKernels is a standalone pip
package** (torch + cutlass-dsl, no SGLang import), so their fast prefill kernel is callable *from* a vLLM
backend later. Choosing vLLM forfeits their scheduler, not their kernels.

### 5.0 Why no MLA is needed

The MLA coupling in vLLM is entirely Python plumbing:

| Piece | MLA-bound? | Evidence |
|---|---|---|
| sparse attention kernel | **no** | `flash_attn_varlen_func` with `block_table = topk_indices` on a page-size-1 *view*. The only MLA-specific bits in `flashattn_mla_sparse.py` are the cache view hardcoding `num_kv_heads=1` (the latent) and FA3's asymmetric `q_v` path (K 64-dim, V 512-dim). GQA uses the ordinary symmetric path — strictly simpler. Keye ships this call on GQA with `pack_gqa=True`. |
| indexer + cache + paged top-k | **no** | `Indexer`, `DeepseekV32IndexerCache`, `SparseAttnIndexer`, `triton_convert_req_index_to_global_index` are shape-generic. `MLAAttentionSpec(num_kv_heads=1)` is a naming artifact; the `assert num_kv_heads == 1` inside is the *indexer's own MQA shape*. Only `qr` is MLA's, and our `wq` replaces it. |
| page-size-1 legality | **no constraint** | No page-size assert in `flash_attn_interface.py`; page size is read from `k.shape[1]`. The only requirement is `assert block_table is None or seqused_k is not None` (`:274`). |
| backend + model registration | **yes — this is what we write** | Sparse backends register as `MLAAttentionImpl` under `backends/mla/`, and every indexer-bearing model subclasses `DeepseekV2MLAAttention`. So: one backend registered as an ordinary attention backend, one Qwen3 model file. |

**No kernel authoring, no MLA in the model.**

### 5.1 Serving constraints to design against

| Constraint | Source | Consequence |
|---|---|---|
| plain FA backend allocation: `[MultipleOf(16)]` | `flash_attn.py:83` | page size must be a multiple of 16 |
| MLA-sparse backend declares `[64]` | `flashattn_mla_sparse.py` | 64 is the proven value |
| indexer paged-logits kernel needs **exactly 64** | Keye asserts ×2 | **`page_size = 64` satisfies all three — the compatibility keystone** |
| `supported_dtypes = [fp16, bf16]` | `flash_attn.py:73` | bf16 |
| KV cache dtype | fp8 is allowed on the plain path, but fp8 scales under a page-1 view are untested | **pin bf16 KV for v1** |
| main `head_dim` | FA3 supports 32/64/96/128/192/256 | Qwen3's 128 ✓, nothing to choose |
| `max_seqlen_k = topk`, per-row truncation via `seqused_k` | the call itself | `top_k=2048` works at any sequence length |
| `topk_indices_buffer` `[max_num_tokens, topk]` int32, preallocated | `deepseek_v2.py:1377` | 268 MB at 32K×2048; scales with `max_num_batched_tokens` |
| `index_topk` on the HF config | `flashattn_mla_sparse.py:102`; memory `dsa-serving-index-topk-gate` | mirror the key or the sparse path silently runs dense |
| indexer linears are `ReplicatedLinear` | both stacks | **no TP constraint** on indexer `n_heads` |
| **stock DeepGEMM: head count ∈ {32,64,128}, `head_dim = 128`** | `vllm_minicpm3_dsa/indexer.py::SUPPORTED_KERNEL_HEADS` | `16 × 64` zero-pads to `32 × 128` (lossless; see §2.3) |
| `128 % n_heads == 0` | Keye `block_q = 128 // num_heads` | 16 ✓ |
| `head_dim` power of 2; bf16 into the Hadamard | Hadamard asserts, both codebases | 64 ✓ |
| **vLLM must contain `deepseek_v2.Indexer` / `SparseAttnIndexer` / `mla/sparse_utils.py`** | — | `.devlibs/vllm-src` has them; the container's **0.20.2 does not**. Confirm which build the eval harness runs — this is the one thing that turns "supported" into "supported after an upgrade". |

Enforce the indexer-side rows in the config's `__post_init__` under a `serving_compat` flag, as `MSAConfig`
does — so a config that cannot be served fails at construction, not after a 4-day run.

### 5.2 Reusable as-is, MLA-independent
- `vllm/model_executor/models/deepseek_v2.py::Indexer` (the serving twin of our module) — its only MLA
  coupling is the `qr` input, which our direct `wq` projection replaces.
- `DeepseekV32IndexerCache` — an `MLAAttentionSpec(num_kv_heads=1, head_size=132)` fp8 cache; "MLA" is
  a naming artifact, it is just a 1-head cache.
- `SparseAttnIndexer` (paged top-k op + prefill workspace) and
  `mla/sparse_utils.py::triton_convert_req_index_to_global_index` (per-request → global token indices).
- `scripts/dsa/vllm_minicpm3_dsa/indexer.py` — our own unfused-projection serving indexer, 478 lines,
  already parity-tested against the training module.

### 5.3 The mechanism, and the one thing left to measure

`flashattn_mla_sparse.py` reduces sparse attention to FA3 with a **page-size-1 block table**:

```python
kv = kv_cache.view(-1, block_size, head_size)
k_cache = kv[:, :, kv_lora_rank:].view(-1, 1, 1, qk_rope_head_dim)   # ← per-token slots, 1 kv head
out = flash_attn_varlen_func(q=..., k=k_cache, v=v_cache,
        max_seqlen_q=1, cu_seqlens_q=arange(n+1),
        max_seqlen_k=topk, seqused_k=valid_counts,
        block_table=topk_indices, causal=True, fa_version=3)
```

Each query token becomes its own length-1 sequence over its own gathered key set; causality is enforced
by the *selection*, not the kernel mask. For GQA the same call should work with the standard
`[num_blocks, block_size, H_kv, d]` cache viewed as `(-1, 1, H_kv, d)` and no `q_v` (qk dim == v dim).
Coalescing is fine — 2 KB contiguous per slot per cache.

**Decode is no longer an unknown.** Keye's production decode path does exactly this on GQA —
`k_cache.view(-1, 1, num_kv_heads, head_dim)` + `flash_attn_with_kvcache(page_table=token_slots,
pack_gqa=True)` — so page-size-1 block tables with `H_kv > 1` work. `pack_gqa=True` is the flag to copy.

**Prefill is the unmeasured regime.** A 32K prefill on this route means one varlen call with 32,768
length-1 sequences and a `[32768, 2048]` block table. Neither codebase demonstrates that shape — Keye went
to a custom dedup kernel instead, which is itself a hint. So the spike
(`tests/dsa/probe_fa3_sparse_gqa.py`) must cover **both**: allocate a paged GQA KV cache, random top-k
indices, compare against a reference gather+SDPA for correctness, then time the decode shape (1 query/seq)
*and* the prefill shape (32K length-1 sequences). Correctness is the gate; the prefill timing is
information, not a gate — if it is bad, §10.6's fallbacks apply and nothing about the training plan changes.

Then the serving build (Phase 3, after Phase 2 converges):
1. `vllm/v1/attention/backends/flash_attn_sparse.py` — new, ~250 lines, mirroring
   `flashattn_mla_sparse.py` minus the MLA bits.
2. `scripts/dsa/vllm_qwen3_dsa/{__init__,model,indexer,attention}.py` — Qwen3 model that builds one
   indexer per layer, allocates the shared `topk_indices_buffer`, and constructs `Attention` with the
   sparse backend.
3. `scripts/dsa/build_qwen3_dsa_serving_dir.py` — **must write `index_topk` into `config.json`**; without
   it vLLM silently falls back to dense (memory `dsa-serving-index-topk-gate`).
4. Parity ladder from `docs/dsa_eval_report.md` §2–§3: dense equivalence at `top_k ≥ T`, then
   train/serve selection overlap ≥ 0.99 at k=2048, then greedy-decode agreement, then the sparsity probe
   that proves the sparse path is actually running.
5. Confirm which vLLM the eval harness runs before starting — the container ships 0.20.2, which predates
   the `vllm/models/` layout that `.devlibs/vllm-src` uses.

### 5.4 Four serving lessons taken from Keye-VL-2.0

1. **Top-k must be deterministic, and `torch.topk` is not.** *"To avoid mismatch between training and
   inference Top-k results, we use deterministic Top-k computation. `flashinfer.topk` replaces
   `torch.topk`, achieving a 2–3× speedup while preserving determinism."* This is a third
   train/serve-drift mechanism alongside the two we already got burned by (`fp8_ue8m0`, the missing
   `index_topk` config gate) — and it is worse than those, because with `k=2048` out of 32K there are
   near-ties in every row, so a nondeterministic tie-break silently perturbs the selected set. **Add to
   the parity ladder:** selection must be bit-identical across repeated runs on the same input, and equal
   between train and serve. It matters even more if this checkpoint ever goes to RL.
2. **Score storage `T×T → T×max_seq`.** *"We reduce score storage to `T×max_seq` and use
   `flashinfer.top_k_ragged_transform` to compute only over valid KV regions."* Our query tiling bounds
   the same tensor, but the ragged form is strictly better for packed/variable-length batches, where a
   dense `T×T` allocation wastes most of its area on padding.
3. **Their answer to scattered token-granular gathers is dedup, not coarser pages.** *"Adjacent queries
   often select similar Top-k KV sets. We deduplicate Top-k sets across adjacent queries and use an MMA
   Thread Layout-Aware Mask inside the attention kernel."* This directly addresses the §8 risk. It is a
   *kernel-internal* optimization, so it is not available from the FA3-with-a-block-table baseline — but it
   is the documented fix if prefill coalescing disappoints, and it is a better answer than my earlier
   "fall back to 16–64-token pages", which would trade away the granularity that is the whole point.
4. **Chunked prefill for the indexer:** *"a chunked indexer is used as a memory-bound fallback"*, with
   `q_chunk_size = kv_chunk_size = 512` in their shipped config — the same 512 we tile at in training.

---

## 6. Exact file-by-file work list

**Ground rule (decided 2026-08-14): full isolation.** MiniCPM3-DSA and Qwen3-MSA must keep running
untouched, so nothing shared gets modified for our benefit. `dsa_indexer.py`, `minicpm_dsa.py`,
`msa_indexer.py` and `qwen3_msa.py` are **read-only references**. The FP8 numerics helpers
(`_fake_quant_fp8` with its straight-through estimator, `_rotate_activation`/`_fwht`, `FP8_DTYPE`/`FP8_MAX`)
are **copied**, not imported — ~60 lines duplicated in exchange for zero coupling. Consequence to accept:
a future fix to those numerics must be applied in both places, and they are exactly the numerics train/serve
parity depends on, so note it in both files.

**New**

| File | ~LOC | Contents |
|---|---|---|
| `verl/models/transformers/qwen3_dsa_indexer.py` | 300 | `Qwen3DSAConfig` + `Qwen3DSAIndexer`, self-contained: `wq` direct from hidden, `q_norm` RMSNorm, `k_norm` LayerNorm(fp32), `weights_proj` (fp32), own 64-dim rotary at base `rope_theta`, copied FP8/UE8M0 + Hadamard helpers, `project`/`scores`/`select_topk`, `reset_parameters`, and `serving_compat` validation of §5.1 |
| `verl/models/transformers/qwen3_dsa.py` | 650 | `dsa_overrides_from_config`, `build_dsa_config` (Qwen3 geometry), `attach_indexers`, `_warmstart_from_consolidated`, `freeze_base_train_indexer`, `install_kl_accumulation` (with the `position_ids` side channel), `_causal_doc_bias_block`, `_head_avg_teacher` (+compiled), `_dense_warmup_kl`, `_sparse_tile` (fused attention+KL), `_sparse_attn_and_kl`, `_accumulate_diag`, `_finalize_diag`, `qwen3_dsa_attn_forward` |
| `tests/dsa/probe_fa3_sparse_gqa.py` | 150 | Phase-0 spike (§5.3) — correctness + decode timing + prefill timing |
| `tests/dsa/test_qwen3_dsa_dense_equivalence.py` | 80 | Phase-1 LM logits bit-identical to stock Qwen3 |
| `tests/dsa/test_qwen3_dsa_kl.py` | 150 | Teacher matches eager attention; tiled == untiled; checkpoint == no-checkpoint; padding/document masking; own-rotary == a reference 64-dim RoPE |
| `tests/dsa/test_qwen3_dsa_sparse.py` | 180 | `top_k ≥ T` ⇒ output equals dense (M0 parity); **free teacher == softmax-over-subset identity, per head**; grad isolation (no LM grad on indexer params, no KL grad on base params); top-k determinism across repeated runs |
| `tests/dsa/test_qwen3_dsa_indexer_parity.py` | 150 | Training indexer vs the serving indexer, including the `16×64 → 32×128` zero-pad (dot and FP8 scale must be unchanged) |
| `examples/dsa/_qwen3_dsa_common.sh` | 150 | **Built (2026-08-15).** Sourced by both phase scripts: `dsa_expand_files`, the CONFIG_TAG/RUN_NAME split, the manifest writer (git sha **+ working-tree diff** + every resolved knob, since env-var prefixes never reach `/proc/<pid>/cmdline`), and the `printf %q` argv logger. Factored rather than copy-pasted twice because these are the parts that have cost runs before. Note `DATA_TAG` must survive the `<model>__<dataset>__<stage>_<ts>` artifact layout — a naive first-token split yields the *model* name, so two BC datasets would share a `CKPT_DIR` and `resume_mode=auto` would resume across them |
| `examples/dsa/run_qwen3_dsa_phase{1,2}.sh` | 250 + 230 | **Built (2026-08-15).** Phase 1: `loss_mode=indexer_kl`, `PackedPretrainDataset`, 7630 steps = 2B tokens, `GRAD_CKPT=False`. Phase 2: `loss_mode=dsa_sparse`, `MSASFTDataset` (**not** `MultiTurnSFTDataset`, which deletes `<think>` traces, and **not** `PackedPretrainDataset`, which drops variable-length BC rows), `KL_BLOCK=256`, `ACT_OFFLOAD=True`, `optim.indexer_lr`, and a hard preflight failure if `WARMSTART` is unset |
| `scripts/dsa/vllm_qwen3_dsa/{__init__,model,indexer,attention}.py` + `build_qwen3_dsa_serving_dir.py` | 800 | Phase 3 only. `indexer.py` adapted from `vllm_minicpm3_dsa/indexer.py` (query from hidden, own rotary, `q_norm`); a non-MLA sparse backend; **`index_topk` written into `config.json`** |

**Modified — exactly two files, both additively**

| File | Change |
|---|---|
| `verl/models/transformers/monkey_patch.py` | One new `elif` **after** the MSA branch: `model_type == "qwen3" and getattr(config, "dsa_enabled", False)` → build config, `attach_indexers`, patch `modeling_qwen3.Qwen3Attention.forward`, `install_kl_accumulation`, freeze base in `dense_warmup`. Plus `assert ulysses_sp_size == 1` (the indexer scores every query against the full key sequence; a sharded sequence silently normalizes over a fragment) and an assert that `dsa_enabled` and `msa_enabled` are not both set. Disjoint from the existing branches by `model_type`/flag, so neither can change behaviour. Ordering constraint: after `from_pretrained`, before FSDP wrap. |
| `verl/utils/fsdp_utils.py:592` | Append `"Qwen3DSAIndexer"` to `_indexer_cls_names`. **Unavoidable:** the FSDP2 Option-B2 wrap keys on the class *name*, and a class not in that tuple silently falls into the broken path — nonzero-but-fake `grad_norm`, flat loss at `world_size > 1` (`docs/dsa_fsdp_sharding_notes.md` §3b/§4). A pure membership extension, so it cannot affect `LightningIndexer` or `MSAIndexer`. The new class must also expose `.cfg.mode`, which the same line checks. |

**Zero-touch, because we reuse the existing flag and attribute names.** Setting `config.dsa_enabled` and
stashing `model._dsa_indexer_kl` / `model._dsa_metrics` makes all of these work unmodified:
`fsdp_utils.py:593` (the `dsa_enabled or msa_enabled` gate), `workers/engine/fsdp/transformer_impl.py:670`
(`batch_num_valid_queries`), `workers/utils/losses.py` (`indexer_kl_loss` / `dsa_sparse_loss`),
`trainer/sft_trainer.py` (`loss_mode ∈ {indexer_kl, dsa_sparse}`), `utils/dataset/*`. Do **not** rename
them for tidiness — the naming *is* the integration.

---

## 7. Sequence, with the decision points

| Step | Work | Gate before continuing |
|---|---|---|
| 0 | FA3 page-size-1 GQA spike (§5.3): correctness, decode timing, **prefill timing** | correctness only — decode is already proven by Keye (§10.2); prefill timing is information |
| 1 | `qwen3_dsa_indexer.py` (new, self-contained) **incl. `q_norm` + per-layer init (§2.4)** + `serving_compat` validation + unit tests | **`entropy_frac ∈ [0.99, 1.0]` for every layer**; grad reaches `wq`/`wk`/`weights_proj`; `16×64 → 32×128` pad leaves dot and FP8 scale unchanged |
| 2 | `qwen3_dsa.py` Phase-1 path + dense-equivalence & KL tests | LM logits bit-identical; teacher matches eager |
| 3 | Phase-1 smoke: 1 node, 20 steps, 32K | ≤ 60 s/step, peak < 70 GB, KL falling, no NaN, **per-layer init entropy in range and `grad_norm` not pinned at the clip** |
| 4 | Phase-1 full run, 2B tokens (gate at 1B) | `topk_recall ≥ 0.95` per layer at k=2048; `group_divergence` acceptable |
| 5 | Phase-2 path + sparse parity/grad-isolation tests | `top_k ≥ T` equals dense; grad isolation holds; top-k deterministic |
| 6 | Phase-2a (short + decode-long BC) | short-context within ~1 pt of dense |
| 7 | Phase-2b (prefill-long) | **RULER 32K within 2 pts of dense** (vs MSA-k16's −6.36) |
| 8 | Serving build + parity ladder | selection overlap ≥ 0.99, decode agreement, ~3× prefill / ~5× decode |

Wall clock: ~2 days spike/impl, 4–5 days Phase 1, 4–6 days Phase 2, 3–5 days serving.

Operational reminders that have each cost a run before: timestamped `RUN_NAME` with a timestamp-free
config-keyed `CKPT_DIR` so `resume_mode=auto` doesn't restart at step 0 (`msa-run-naming-and-resume`);
log the full invocation + resolved env + `git diff` next to the checkpoints (`log-full-invocation`);
`pad_mode=no_padding` (FSDP engine supports nothing else); `torchrun --local-addr 127.0.0.1` on hosts
that can't resolve their own FQDN.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| ~~FA3 page-size-1 with `H_kv>1` is unsupported~~ — **retired.** | Keye ships it in production for GQA decode (§10.2). No longer a risk; the spike is a confirmation. |
| FA3 page-size-1 **prefill** (32,768 length-1 sequences) is pathologically slow. | Measured in Phase 0 as information, not a gate — training is unaffected. Fallbacks in order: dense prefill + sparse decode (zero work, keeps the decode win); then EffectiveKernels' dedup kernel called from vLLM, which needs the `(32, 2048)` instantiation (§10.4). Do *not* fall back to coarser pages — that discards the granularity the project is testing. |
| Indexer `O(T²)` dominates prefill FLOPs, so prefill throughput barely improves. | Expected and quantified (§2.3), and bounded by Keye's measured >3× prefill at 128K. `d_idx=64` (not 128) halves this in training — though **stock DeepGEMM pads it back to `32×128` at serve time** (§2.3), which is a 2× prefill tax recoverable only via a patched DeepGEMM. |
| **Nondeterministic `torch.topk` ties ⇒ train/serve selection mismatch.** | New, from Keye §5.1 item 1. Add a determinism assertion to the parity ladder; `flashinfer.topk` at serve is both deterministic and 2–3× faster. Third instance of this failure class after `fp8_ue8m0` and the `index_topk` gate. |
| Token-granular selection scatters KV reads and loses to blocks in wall-clock despite better recall. | Measure recall *and* tokens/s at the same k. Keye ships token-granular at 256K, so the answer exists; it is dedup + an MMA-layout-aware mask in-kernel. |
| Shared-across-heads selection under-serves GQA groups that want different context (MSA App. A). | Keye does the same at `H_kv=4` and reports lossless 256K, but Qwen3-4B has `H_kv=8` — 2× the group teachers per shared set. Log `group_divergence` in Phase 1; per-group selection is the documented follow-up. |
| Phase-2 selected-set KL decalibrates the ranking outside the top-k. | `full_support_kl_prob` knob, ready but off. Neither precedent mentions needing it. |
| `d_idx=64` ⇒ vLLM's `use_fused_indexer_q` path (which wants `head_dim=128`, `rope_dim=64`) is unavailable. | Our MiniCPM3 serving indexer already pads `16×64 → 32×128` for DeepGEMM and is parity-tested there; accept the unfused path for v1. |

---

## 9. Precedent: Kwai Keye-VL-2.0 (arXiv 2606.10651)

**"The first to adapt DeepSeek Sparse Attention (DSA) to GQA-based multimodal architectures, enabling
lossless 256K context processing."** Keye-VL-2.0-30B-A3B is a 30B-A3B MoE MLLM: `L=48`, `hidden=2048`,
`H_q=32`, **`H_kv=4`**, `d_h=128`, `rope_theta=1e7`, `max_position_embeddings=262144`, mrope. It is the
only published DSA-on-GQA system, so this plan follows it wherever it publishes a number.

Shipped `config.json` → `sa_config`:

```json
{"indexer_head_dim": 64, "indexer_num_heads": 16, "indexer_num_kv_heads": 1,
 "kv_chunk_size": 512, "q_chunk_size": 512, "topk": 2048}
```

Their equations, verbatim:

```
(1)  I_{t,s} = Σ_{j=1..H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
(2)  Ω_t     = { s | I_{t,s} ∈ Top-k(I_{t,:}) }
(3)  u_{t,g} = Attn(h_{t,g}, { c_{s,g} | s ∈ Ω_t })                 # SAME Ω_t for every group
(4)  L^I_warmup = Σ_t Σ_{g=1..G} D_KL( p_{t,:,g} ‖ Softmax(I_{t,:}) )
(5)  S_t    = { s | I_{t,s} ∈ Top-k(I_{t,:}) }
(6)  L^I_sparse = Σ_t Σ_{g=1..G} D_KL( p_{t,S_t,g} ‖ Softmax(I_{t,S_t}) )
(7)  L_total = L_NTP + λ · L^I_sparse
```

with *"`q^I_{t,j}` and `w^I_{t,j}` are derived from `h_t`, and `k^I_s` is the shared key derived from
`h_s`"*; *"By sharing one key head across all query heads, the indexer substantially reduces both
computation and memory traffic. Together with FP8 implementation and the ReLU-based scoring function…"*;
warm-up *"uses approximately 2B multimodal tokens"* with *"most parameters frozen"*; sparse adaptation
*"all parameters are unfrozen"*, *"the indexer input is detached from the computation graph"*.

**Where this plan agrees with them** (all of it independently derived before reading the paper, which is
the reassuring part): GQA backbone, token-granular selection, `k=2048`, MQA single shared index key,
ReLU-weighted multi-head score, `H^I=16`, 512-token chunking, two-stage frozen-then-unfrozen recipe,
per-group teacher distilled into one shared index distribution, selected-set-truncated-and-renormalized
KL in stage 2, detached indexer input, FP8, all layers sparse, no forced local/sink.

**Where the paper changed the plan:**

| | Before | After | Why |
|---|---|---|---|
| `d_idx` | 128 | **64** | Their `indexer_head_dim`; halves indexer FLOPs (25% → 12.5% of dense) and matches the dims our serving indexer is already parity-tested at. |
| Indexer query | `wq_a(512) → norm → wq_b` | **direct `wq`** | *"derived from `h_t`"*; no bottleneck rank in their config, and at `d_idx=64` a bottleneck buys nothing. |
| Indexer RoPE | full 128, base tables | **own 64-dim rotary, all dims roped** | `d_idx=64` means the base's 128-wide tables cannot be reused (slicing breaks `rotate_half`). On *how much* to rope, we follow Keye rather than DeepSeek's fraction — because DeepSeek's half-roped indexer mirrors MLA's own partly-roped attention, while Qwen3's attention is fully roped. §2.1. |
| Warm-up tokens | 1B | **2B** | Theirs ~2B, DeepSeek's 2.1B. Our 1B came from the MSA run. |
| Expected speedup | 2.7× prefill FLOP / 10.6× decode bytes | **~3× prefill / ~5× decode, measured** | Their measured numbers at 128K, `k=2048`. Counted ratios overstate. |
| Serving determinism | not considered | **hard requirement** | §5.1 item 1. |

**What they do NOT publish**, so these remain our own calls: which layers are sparsified (we assume all),
RoPE handling for the indexer, `λ`, learning rates, indexer-key cache format, forced local/sink tokens,
and any ablation of `k`. Their multimodal-specific choice — normalizing the teacher *"over visual and text
tokens"* — has no analogue in our text-only setting.

**Code:** [github.com/Kwai-Keye/Keye](https://github.com/Kwai-Keye/Keye); serving is a custom
[SGLang branch](https://github.com/Kwai-Keye/sglang/tree/keye-vl-v2-30b-release) plus a
[DeepGEMM fork](https://github.com/Kwai-Keye/DeepGEMM) (`keye_support`) and
[EffectiveKernels](https://github.com/Kwai-Keye/EffectiveKernels) — **not vLLM**. Read as source, it is far
more reusable than the paper suggests: see §10. Also relevant:
[github.com/lemyx/tilelang-dsa](https://github.com/lemyx/tilelang-dsa), a TileLang warm-up lightning-indexer
training op with a hand-written KL backward (MLA/MQA mode; would need the GQA teacher swapped in).

---

## 10. Reusing the Keye SGLang branch, and what it constrains

Read at commit `keye-vl-v2-30b-release`. The relevant files are
`python/sglang/srt/layers/attention/keye_topk/keye_indexer.py` (622 L),
`.../attention/keye_sa_backend.py` (760 L), `python/sglang/srt/models/keye_topk_mask.py` (482 L), and the
public [EffectiveKernels](https://github.com/Kwai-Keye/EffectiveKernels) package (29 files).

### 10.1 The text backbone is already Qwen3-shaped

```python
class KeyeTopKMaskDecoderLayer(Qwen3DecoderLayer):   # models/keye_topk_mask.py:284
class KeyeTopKMaskModel(Qwen2Model):                 # :323
class KeyeTopKMaskAttention(nn.Module):              # :36  — GQA + per-head q_norm/k_norm RMSNorm
    self.attn = RadixAttention(...)
    topk_indices = self.sa_indexer(hidden_states, positions, forward_batch, layer_id)
    attn_output = self.attn(q, k, v, forward_batch, topk_indices=topk_indices)
```

Their sparse attention layer subclasses **SGLang's `Qwen3DecoderLayer`** and applies Qwen3's own QK-norm.
Serving Qwen3-4B-DSA on their fork is therefore mostly a text-only `ForCausalLM` wrapper around
`KeyeTopKMaskModel` plus an `sa_config` block in `config.json` — not a new model implementation.

### 10.2 Free, generic over our dimensions

| Component | Notes |
|---|---|
| `KeyeIndexer` | Whole serving indexer: fused `q_proj`(+gate), `k_proj`, q/k norms, own rotary, Hadamard, `act_quant`+ue8m0, fp8 paged index-K cache, `deep_gemm.fp8_mqa_logits` (prefill) / `fp8_paged_mqa_logits` (decode), chunked-logits OOM fallback. Nothing in it depends on the main attention's shape. |
| **Decode sparse attention** | `keye_sa_backend.py:742-759` is *exactly* our Phase-0 hypothesis in production: `k_cache.view(-1, 1, num_kv_heads, head_dim)` + `flash_attn_with_kvcache(page_table=token_slots, pack_gqa=True, causal=True)`. Generic in `num_kv_heads`. **The FA3 page-size-1 GQA spike is de-risked** — it ships. |
| `topk_transform` | `flashinfer.top_k_ragged_transform` / `top_k_page_table_transform` (deterministic), `sgl_kernel.fast_topk_v2` fast path, `torch.topk` slow path. |
| Index-K KV pool | `head_dim + head_dim//block_size*4` bytes/token/layer = **68 B** at `d_idx=64` (vs 132 for DeepSeek's 128). |

### 10.3 Constraints reuse would impose

| Constraint | Where | Qwen3-4B |
|---|---|---|
| main `head_dim = 128` (AOT builder declares `divisibility=128`) | `ops/aot.py:108` | 128 ✓ |
| indexer `head_dim ∈ {32, 64, 128}` | `keye_indexer.py:168` assert | 64 ✓ |
| indexer `num_heads` divides 128 (`block_q = 128 // num_heads`) | `keye_indexer.py:486` | 16 ✓ |
| `topk ∈ {128, 2048}` — **exactly**, not ≤ | `topk_block_unique.h:51-89` dispatch; `fast_topk_v2` fast path keyed on `topk == 2048` | 2048 ✓ |
| deterministic top-k ⟹ `topk ≤ 2048` | `keye_sa_backend.py:318` assert | ✓ |
| **`page_size == 64`** | two asserts, `keye_indexer.py:441,589` (deep_gemm paged MQA logits) | server flag |
| SM90 Hopper, CUDA ≥12.3, `nvidia-cutlass-dsl ≥4.4.2`, bf16 | EK README; `is_hopper_with_cuda_12_3()` gate; Hadamard asserts bf16 | H100 ✓ |
| `mrope_section` must exist and `sum(section)*2 == indexer head_dim` | `keye_indexer.py:110-117` asserts | needs a plain-rope branch, or an `mrope_section` summing to 64 — with 1-D positions broadcast to `[3, N]` (`:198`), MRoPE with identical position channels **is** plain RoPE, so a synthetic `[16,24,24]` is numerically exact |
| **`H_q / H_kv ∈ {8, 2}`** for the prefill kernel | `topk_block = 128 // num_kv_groups`, `keye_sa_backend.py:685` | **4 ✗ — see 10.4** |

### 10.4 The GQA-ratio problem, at two depths

`AOT_CONFIGS` (`ops/aot.py:86-90`) ships `qh8/kv4`, `qh8/kv2`, `qh8/kv1` at `topk=2048`, and `qh2/kv8` at
`topk=128`. Qwen3-4B is `qh4/kv8`. Note the ratio is **TP-invariant** (both head counts divide by TP), so
no launch flag fixes it.

1. **Mechanical.** `topk_block_unique` is an AOT C++ dispatch over exactly two `(topk_block, topk)` pairs —
   `(64, 128)` and `(16, 2048)` — and anything else is `TORCH_CHECK(false, "Unsupported configuration")`.
   Qwen3-4B needs `(32, 2048)`. Fix: one `topk_2048_block_32_unique.cu` instantiating
   `run_topk_block_unique_kernel<256, 32, 2048>`, a dispatch branch, and the two output-buffer size
   formulas (for `(16,2048)` they are `unique_vals[:, topk + 4*128]` and `qmask[:, topk + 4*64]`, which are
   tile-derived and must be re-derived, not copied). The CuTe attention kernel itself is *parameterized*
   over `qhead_per_kvhead` and JIT-compiles, so only this one op blocks.
2. **Fundamental.** `unique_pack_factor = tile_m // qhead_per_kvhead` (`sparse_fwd.py:44`), `tile_m = 128`.
   The kernel unions the top-k sets of `128/G` **adjacent queries** into one 128-row MMA tile. At Keye's
   `G=8` that is 16 queries; at Qwen3-4B's `G=4` it is **32** — twice as many query rows sharing one union,
   so the union is larger and each tile loads more KV for the same `k`. Their prefill design intrinsically
   favors *large* GQA ratios. Expect prefill gains below their ">3×"; decode is unaffected.

### 10.5 Two architecture details their indexer has that DeepSeek's does not

1. **`q_norm = RMSNorm(d_idx, eps=1e-6)` on the indexer query** (`keye_indexer.py:156`). Neither
   DeepSeek's reference nor our MiniCPM3 port has one — because MLA's `q_a_layernorm` sits upstream of the
   indexer query, and on GQA nothing does. So it is a *relocated* norm, not an invented one, and it is a
   real architecture change rather than a weight remap: **retrofitting it after Phase 1 means retraining
   Phase 1.** **Adopted** — it is in §2.1's config table and step 1 of §7. (`k_norm` is
   `LayerNorm(d_idx, eps=1e-6)` in fp32, which both precedents already share.)
2. **Gate `w` fused into `q_proj`** as `num_heads` extra output columns (`hidden → n*d + n`), applied as
   `w.float() * q_scale * softmax_scale` with no `n_heads**-0.5` (ours applies it in `project()`). Moot on
   the vLLM route — our serving indexer keeps `wq`/`weights_proj` separate, as `vllm_minicpm3_dsa` already
   does. Recorded only in case a checkpoint ever has to be loaded by their module, in which case the
   conversion concatenates the two weights and folds the `16**-0.5`. That factor is a uniform positive
   scalar, so it cannot change the selection — only the KL temperature.

Order of application (`_get_q_k_w_bf16`): project → per-head norm → **RoPE (`rotary_dim == head_dim`, neox,
own rotary at base θ)** → Hadamard → fp8 `act_quant`. That matches §2.1 line for line. The full-width rope
is corroboration rather than the source of our choice — §2.1 argues it from the structural match to Qwen3's
fully-rotary attention, which is also *why* Keye's number is right for a GQA backbone and DeepSeek's is not.

### 10.6 Conclusion: vLLM, with Keye as the reference implementation

**Decided (2026-08-14): build on vLLM** — see §5 for the four reasons and §5.0 for why no MLA is needed.
Keye's sources are the reference for every *numeric* choice (dims, norms, rope width, fp8 format, ordering)
and for the constraint table in §5.1; their engine is not adopted.

`KeyeIndexer` is coupled to SGLang's `ForwardBatch` / `token_to_kv_pool` / `BaseIndexerMetadata`, so it is
read, not imported. vLLM already has the matching pieces (`SparseAttnIndexer`, `DeepseekV32IndexerCache`,
`triton_convert_req_index_to_global_index`) over the same DeepGEMM kernels.

Staged, in the order the work should happen:

1. **v1:** vLLM + a non-MLA sparse backend (FA3, page-size-1) serving prefill *and* decode, plus our
   indexer module adapted from `vllm_minicpm3_dsa/indexer.py`. `--block-size 64`, bf16 KV, `index_topk` in
   `config.json`. No ratio problem, no kernel authoring.
2. **If prefill throughput binds:** call EffectiveKernels from inside vLLM — it is a standalone pip package
   (torch + cutlass-dsl, no SGLang import), so this does not mean adopting their engine. That is when the
   `(32, 2048)` instantiation becomes worth writing.
3. **Revisit SGLang only if** the FA3 prefill measurement is bad *and* the EffectiveKernels path fails.

What we knowingly give up: the adjacent-query dedup kernel, hence prefill numbers below their published
>3×. Not decode — the dedup amortizes across *adjacent query rows*, and a decode step has one query per
sequence, so on the decode path the two routes are doing the identical thing.
