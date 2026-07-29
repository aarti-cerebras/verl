# MSA index-branch KL loss — equations and shapes

**Paper:** MiniMax Sparse Attention, arXiv [2606.13392](https://arxiv.org/abs/2606.13392).
§3.2 Training (Eq. 9–11, Algorithm 1), §3.3 Complexity (Eq. 12), §4 Kernel Design, §5.2 metrics.
**All paper claims below are transcribed from the PDF**
(`/cb/ml-eng/aarti/msa/refs/msa_2606.13392.pdf`, 30pp, §3.2 on p.5), not from the HTML — see §7.
Re-verified 2026-07-28; §1 survived unchanged, §2.1–§2.2 had four errors, now corrected inline.

**vLLM reference implementation:** `vllm/models/minimax_m3/` (merged PR #45381, *"[Model] Add MiniMax
M3 support"*, 2026-06-15; claims below pinned to `main` @ `4f56321d`).
**Inference kernel:** [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA) (MIT, **SM100 only**, and
**not vendored inside vLLM** — `vllm/third_party/fmha_sm100` does not exist in the tree).

Target model for all shapes: **Qwen3-4B-Thinking-2507**. Companion to `docs/qwen3_4b_dsa/` (the
DSA-on-GQA variant this supersedes).

---

## 0. Notation

| symbol | meaning | Qwen3-4B |
|---|---|--:|
| `b` | batch size | 1 |
| `N` / `T` | **sequence length** (paper's `N`) | 32768 |
| `T_q` | query tile (`kl_block_size`) | 512 |
| `L` | layers | 36 (3 dense + **33 sparse**) |
| `d_model` | hidden size | 2560 |
| `H_q` | query heads | 32 |
| `H_kv` | KV heads = **GQA groups** = index query heads | 8 |
| `G` | heads per group = `H_q / H_kv` | 4 |
| `H_r` | the `G` query heads in group `r` | — |
| `d_h` | attention head dim | 128 |
| `d_idx` | index head dim | 128 |
| `B_k` | block size | **128** |
| `B` | blocks per row = `ceil(N / B_k)` | **256** |
| `k` | selected blocks per (query, group) | **16** |
| `k·B_k` | token budget | **2048** (6.25%) |
| `I^(r)_i` | selected **block** indices | \|·\| = k |
| `I^(r)_{i,tok}` | selected **token** set (KL support) | ≤ 2048 |

---

## 1. The loss — verbatim from the paper

**Support (p.5):** *"Writing `I^(r)_{i,tok} = (∪_{b ∈ I^(r)_i} B_b) ∩ {1, …, i}` for the causally
visible tokens induced by the selected block indices, for each query position `i` and GQA group `r`,
we define the Index Branch distribution `P^idx` and the Main Branch teacher `P` over this token index
set:"*

**Eq. 9** — the two distributions:

```
                       exp( S^idx,(r)_{i,j} )
P^idx,(r)_{i,j}  =  ─────────────────────────────────                       ← STUDENT
                     Σ_{u ∈ I^(r)_{i,tok}} exp( S^idx,(r)_{i,u} )


                        1              exp( S^(ℓ)_{i,j} )
P^(r)_{i,j}      =  ─── · Σ_{ℓ ∈ H_r} ─────────────────────────────         ← TEACHER
                        G              Σ_{u ∈ I^(r)_{i,tok}} exp( S^(ℓ)_{i,u} )

                                                             j ∈ I^(r)_{i,tok}
```

with the scores defined **immediately after Eq. 9**:

```
S^idx,(r)_{i,j} = (Q^idx)^(r)_i (K^idx)^T_j / sqrt(d_idx)      ← token-level index score
S^(ℓ)_{i,j}     = Q^(ℓ)_i (K^(r)_j)^T      / sqrt(d_h)         ← Main Branch score, head ℓ ∈ H_r
```

*"The teacher `P` averages the per-head Main Branch distributions at the probability level."*

**Eq. 10** — the loss:

```
L_KL  =  1/(N · H_kv) · Σ_{i=1}^{N} Σ_{r=1}^{H_kv}  D_KL( stopgrad(P^(r)_{i,·}) ‖ P^idx,(r)_{i,·} )
```

*"where `N` is the sequence length, and the teacher distribution `P^(r)_{i,·}` is detached from
gradient computation."*

**Eq. 11** — gradient detach:

```
Q^idx = stopgrad(X) W^idx_q ,      K^idx = stopgrad(X) W^idx_k
```

*"The teacher `P` in Equation 9 is detached, so `L_KL` leaves the Main Branch projections untouched;
Equation 11 further prevents it from reaching the backbone through `X`. Under this rule, `L_KL`
updates only `W^idx_q` and `W^idx_k`."*

**Total loss (Algorithm 1 caption):**

```
L  =  L_LM  +  λ · Σ_{layers} L_KL
```

**Note the reduction: the paper's is a SUM over layers times `λ`, not a mean.**

Our implementation defaults to **`kl_reduction='mean'` in both phases** (configurable via
`msa_kl_reduction`), which is *not* a deviation in the optimum: per-layer indexer parameters are
disjoint, so `∂L/∂θ_i = ∂KL_i/∂θ_i` and `sum == mean × n_sparse_layers` — a pure gradient scale,
equivalent to `LR × n_layers`. Asserted in `tests/msa/test_qwen3_msa_phase1.py` §1b
(`mean 4.0021 × 25 = 100.0520 == sum`). Rationale, including the `grad_norm` vs `clip_grad = 1.0`
argument, is in plan §4.1.

**The consequence to carry into Phase 2:** matching a paper `λ` under `mean` needs
`λ_ours = λ_paper × n_sparse_layers` (33). A paper `λ` used verbatim with `mean` weights the KL 33× too
weakly against `L_LM`. In Phase 1 this is moot — with the base frozen there is no `L_LM`, so the
reduction only rescales the index LR.

**Gradient w.r.t. the student logits** — assert in a unit test, KL-direction errors are silent:

```
dL / dS^idx,(r)_{i,j}  =  P^idx,(r)_{i,j}  −  P^(r)_{i,j}
```

### 1.1 What Eq. 9–11 settle

1. **The support is TOKENS**, not blocks: `I^(r)_{i,tok}`, explicitly the causally visible tokens
   *induced by* the selected blocks. Blocks only define the set.
2. **`max`-pooling is NOT in the loss.** The student softmaxes token-level `S^idx`, so max-pool is a
   **selection-time** reduction only. It never enters the backward graph; the gradient to `S^idx` is
   **dense**.
3. **No teacher/student functional mismatch** — both sides are token softmaxes over the same support.
   (A block-level formulation would have one: block mass equals `softmax(logsumexp-pool(S))`, so
   pooling with `max` would make the student a biased estimator. Eq. 9 avoids it.)
4. **Head aggregation: per-head softmax over the restricted support, THEN `(1/G)` average** —
   renormalize-then-average. The two orders differ whenever heads hold different mass in the support
   (§3.1).
5. **The scale is `1/sqrt(d_idx)`** in the paper (and `1/sqrt(d_h)` for the main branch). vLLM's
   kernel omits it — see §2.1; use `1/sqrt(d_idx)` in training to match the paper's temperature.
6. **`L_KL` updates only `W^idx_q` and `W^idx_k`** in the paper's formulation — the index branch has
   **no norm layers**. vLLM's M3 adds them; see §2.1.
7. **One equation covers both stages.** In warm-up (full attention) `I^(r)_{i,tok}` is all causally
   visible tokens; in sparse training it is the tokens of the selected blocks.

> **The split that caused early confusion:** the **loss** (Eq. 9–10) is token-level; the **metrics**
> (§5.2, §8) are block-level, using `P_b` = attention mass summed within block `b`. Same paper,
> different support, on purpose.

**Algorithm 1** (p.6), the layer-level procedure:

```
1: Q, K, V     <- X W_q, X W_k, X W_v
2: Q_idx,K_idx <- stopgrad(X) W^idx_q, stopgrad(X) W^idx_k     # (N,H_kv,d_idx), (N,1,d_idx)
3: M_idx       <- BlockMaxPool(Q_idx, K_idx, B_k)              # (N,H_kv,B); per-group, causal
4: I           <- TopK(M_idx, k)                               # local block included
5: O           <- TopKAttn(Q, K, V, I)                          # (N,H_q,d_h)
6: output      <- O W_o
7: L_KL        <- KLdiv(Q_idx, K_idx, stopgrad(Q), stopgrad(K), I)   # over tokens induced by I
```

Line 2 confirms the index key is **single-head** `(N, 1, d_idx)`. Line 7 confirms the teacher is
built from **detached** main-branch `Q`, `K`.

### 1.2 Worked example

Toy dims: `N = 8`, `B_k = 2` → `B = 4`, `k = 2`, `H_q = 4`, `H_kv = 2`, `G = 2`.
Query `i = 7` (all 8 keys visible), group `r = 0` (heads 0, 1).

**Step 1 — teacher, per head.** Softmax over the support (here all 8 tokens).

| | tok0 | tok1 | tok2 | tok3 | tok4 | tok5 | tok6 | tok7 | Σ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `P^(ℓ=0)` | 0.02 | 0.03 | 0.10 | 0.05 | 0.30 | 0.20 | 0.15 | 0.15 | 1.0 |
| `P^(ℓ=1)` | 0.04 | 0.02 | 0.04 | 0.02 | 0.40 | 0.28 | 0.10 | 0.10 | 1.0 |

**Step 2 — `(1/G)` average at the probability level.**

```
P⁽⁰⁾ = [0.030, 0.025, 0.070, 0.035, 0.350, 0.240, 0.125, 0.125]      Σ = 1.0   ← TEACHER
```

**Step 3 — student: softmax of the token-level index scores over the same support.**
(`S^idx` already includes `/sqrt(d_idx)`.)

```
S^idx,⁽⁰⁾ = [-1.2, -0.8,  1.4, -0.3,  2.1,  1.5,  0.9,  1.1]
P^idx,⁽⁰⁾ = [0.0127, 0.0190, 0.1714, 0.0313, 0.3452, 0.1894, 0.1040, 0.1270]   Σ = 1.0  ← STUDENT
```

**Step 4 — Phase 1 KL: one term per TOKEN in the support.**

```
KL = Σ_j  P_j · ln( P_j / P^idx_j )

 j0: 0.030 · ln(0.030/0.0127) = 0.030 · ( 0.8570) = +0.0257
 j1: 0.025 · ln(0.025/0.0190) = 0.025 · ( 0.2748) = +0.0069
 j2: 0.070 · ln(0.070/0.1714) = 0.070 · (-0.8958) = -0.0627
 j3: 0.035 · ln(0.035/0.0313) = 0.035 · ( 0.1113) = +0.0039
 j4: 0.350 · ln(0.350/0.3452) = 0.350 · ( 0.0138) = +0.0048
 j5: 0.240 · ln(0.240/0.1894) = 0.240 · ( 0.2366) = +0.0568
 j6: 0.125 · ln(0.125/0.1040) = 0.125 · ( 0.1843) = +0.0230
 j7: 0.125 · ln(0.125/0.1270) = 0.125 · (-0.0157) = -0.0020
                                                    ────────
                                             KL  =   0.0564
```

Eight terms, one per token. At the real config Phase 1 has `N = 32768` terms per (query, group).

**Step 5 — selection (separate from the loss).** Where max-pool and blocks appear:

```
block0={0,1}  block1={2,3}  block2={4,5}  block3={6,7}

M⁽⁰⁾ = [max(-1.2,-0.8), max(1.4,-0.3), max(2.1,1.5), max(0.9,1.1)]
     = [-0.8,            1.4,           2.1,          1.1]        ← raw, exp-free top-k

top-2 by M  →  Î = {block2, block1}  →  I^(0)_{7,tok} = tokens {2,3,4,5}
```

**Step 6 — Phase 2 KL: same equation, support shrinks to `{2,3,4,5}`.**

Teacher — per-head renormalize over the support, *then* average:

```
 ℓ=0: [0.10,0.05,0.30,0.20] / 0.65 = [0.1538, 0.0769, 0.4615, 0.3077]
 ℓ=1: [0.04,0.02,0.40,0.28] / 0.74 = [0.0541, 0.0270, 0.5405, 0.3784]
 (1/G) average                     = [0.1040, 0.0520, 0.5010, 0.3430]      Σ = 1.0

 student: softmax([1.4, -0.3, 2.1, 1.5]) = [0.2325, 0.0425, 0.4681, 0.2569]

 KL = 0.1040·ln(0.1040/0.2325) + 0.0520·ln(0.0520/0.0425)
    + 0.5010·ln(0.5010/0.4681) + 0.3430·ln(0.3430/0.2569)
    = -0.0837 + 0.0105 + 0.0340 + 0.0992                =  0.0600
```

**Four terms instead of eight.** Tokens 0,1,6,7 contribute nothing — including block3, which the
index branch *should* have picked. That is the cost in §6.2, and why Phase 1 matters.

**Step 7 — metrics are block-level** (§8), from the full-support teacher:

```
P_b = block mass of P⁽⁰⁾ = [0.055, 0.105, 0.590, 0.250]
I*  = top-2 by mass = {block2, block3}      Î = {block2, block1}      I* ∩ Î = {block2}

block recall  = |I* ∩ Î| / |I*| = 1/2               = 0.50
score recall  = P₂ / (P₂ + P₃)  = 0.59 / 0.84       = 0.70
captured_mass = P₂ + P₁         = 0.59 + 0.105      = 0.695
```

**Summary — what is normalized over what:**

| quantity | normalized over | Phase 1 | Phase 2 |
|---|---|---|---|
| `P^(ℓ)` per-head teacher | tokens in `I_tok` | all `N` | selected `k·B_k` |
| **`P^(r)` teacher** | tokens in `I_tok`, then `(1/G)` avg | all `N` | selected `k·B_k` |
| **`P^idx,(r)` student** | tokens in `I_tok` | all `N` | selected `k·B_k` |
| `M_idx` block scores | **not normalized** — selection only | `B` | `B` |
| `P_b` (metrics only) | blocks | `B` | `B` |
| **KL terms per (i,r)** | — | **`N` = 32768** | **`k·B_k` = 2048** |

---

## 2. Index branch forward

Paper: two projections, `W^idx_q` and `W^idx_k` (single-head key), with `/sqrt(d_idx)`.
vLLM adds per-head RMSNorm — see §2.1.

```python
W_q_idx : [d_model, H_kv * d_idx]     # 2560 -> 1024   (one index query per GQA group)
W_k_idx : [d_model, d_idx]            # 2560 ->  128   (ONE shared index key head, MQA-style)
```

| step | op | shape |
|---|---|---|
| hidden states | — | `[b, N, d_model]` = `[1, 32768, 2560]` |
| index queries | `W_q_idx(stopgrad(h))` → view → transpose | `[b, H_kv, T_q, d_idx]` = `[1, 8, 512, 128]` |
| index keys | `W_k_idx(stopgrad(h))` → unsqueeze group axis | `[b, 1, N, d_idx]` = `[1, 1, 32768, 128]` |
| **Gemma-style RMSNorm** (vLLM) | `index_q_norm(·)`, `index_k_norm(·)`: `x·rsqrt(mean(x²)+eps)·(1+w)`, ONE shared `[128]` gain per branch, `eps = rms_norm_eps` | unchanged |
| **RoPE — AFTER the norm** | the **same rotary module** as main attention | unchanged |
| **token scores `S^idx`** | `q_idx @ k_idx.mT / sqrt(d_idx)` | **`[b, H_kv, T_q, N]`** = `[1, 8, 512, 32768]` |
| mask | `+ causal_doc_bias` (`-inf`) — **before** the max | same |
| block scores `M_idx` | `S.view(b, H_kv, T_q, B, B_k).amax(-1)` | `[b, H_kv, T_q, B]` = `[1, 8, 512, 256]` |
| **selection** | `M_idx.topk(k, -1).indices` — **exp-free**, raw scores | `[b, H_kv, T_q, k]` = `[1, 8, 512, 16]` |
| **student** | `softmax(S^idx over I_tok)` — **from `S`, not `M`** | `[b, H_kv, T_q, |I_tok|]` |

Order is **norm-then-RoPE** per the fused kernel:

```python
q_idx = RoPE( index_q_norm( W_q_idx(stopgrad(h)) ) )
k_idx = RoPE( index_k_norm( W_k_idx(stopgrad(h)) ) )
```

Masking **before** `amax` is required: a `-inf` token must never win the max, or a block straddling a
document boundary or the causal frontier scores garbage.

### 2.1 Paper vs. vLLM implementation — the two divergences

| | paper (Eq. 9, 11) | vLLM `models/minimax_m3` |
|---|---|---|
| index score scale | **`/sqrt(d_idx)`** explicit | **omitted** — `qk = tl.dot(q, k)`; `self.scaling = head_dim ** -0.5` is stored and never applied |
| index branch params | `W^idx_q`, `W^idx_k` only | **+ `index_q_norm`, `index_k_norm`** — Gemma-style `(1+w)`, one shared `[128]` gain each, zero-init |

These are consistent designs, not a contradiction: a learned RMSNorm gain subsumes any
constant factor, so vLLM drops the constant. Consequences:

- **Not a train/serve hazard.** Serving never softmaxes the index scores (exp-free top-k on raw
  values) and top-k is invariant to positive scaling, so a constant cannot change deployed selection.
- The constant only sets the **Eq. 10 softmax temperature**. **Use `/sqrt(d_idx)` in training** to
  match the paper.
- **Trainable index-branch set (following vLLM, which we must to reuse the kernels):**
  `{W_q_idx, W_k_idx, index_q_norm.weight, index_k_norm.weight}`. The paper's Eq. 11 lists only the
  two projections because its index branch has no norms.
- **Train the norms in Gemma form** (`(1+w)`, zero-init, one shared `[128]` gain). It is an exact
  reparameterisation of standard RMSNorm (`gain = 1+w_gemma ≡ w_std`) with identical gradients and —
  for matched init — identical gain trajectories, so nothing is lost and train object == serve object.
  Keep the norms out of weight decay (in Gemma form L2 shrinks toward gain 1, in standard form toward
  gain 0; simplest is to exclude them).
- **Consequence of zero-init + the paper's scale:** at init `gain = 1`, so *serving* raw scores are
  `sqrt(128) = 11.3×` larger than the *training* scores we softmax. Irrelevant to selection
  (scale-invariant), but never compare raw index-score magnitudes across train/serve and use no
  absolute score thresholds in diagnostics. Sentinel headroom is safe by ~13 orders of magnitude.
- What must match train↔serve is the **architecture**, and it must be checked **numerically** — a
  module-type + `eps` assertion cannot catch the `(1+w)` parameterisation, the shared-`[128]` gain
  shape, or the norm→RoPE order. Diff against the fused op (§9 item 2).

### 2.2 Confirmed vLLM configuration

| M3 config field | M3 default | Qwen3-4B |
|---|--:|--:|
| `sparse_index_dim` | 128 | **128** (enforced) |
| `sparse_num_index_heads` | 4 | **8** (= `H_kv`) |
| `sparse_topk_blocks` | 16 | **16** |
| `sparse_block_size` | 128 | **128** |
| `sparse_init_block` | **0** | **0** |
| `sparse_local_block` | **1** | **1** |
| `sparse_score_type` | `"max"` | `"max"` |
| `sparse_attention_freq` | `[0]*3 + [1]*57` | **`[0]*3 + [1]*33`** |
| `partial_rotary_factor` | `0.5` (→ rotary_dim 64) | **`1.0`** (→ rotary_dim 128) |

Two fields that are **not** model config: `indexer_kv_dtype` is a vLLM `AttentionConfig` option
(`vllm/config/attention.py:13,67`, `Literal["bf16","fp8","mxfp4","nvfp4"]`, default `bf16`) and is
absent from M3's `config.json`; on SM90 non-bf16 **raises** `NotImplementedError`
(`common/indexer.py:513-518`), so bf16 is forced, not preferred. And `--block-size` need not be
passed: `get_preferred_block_size` returns `min(supported) = 128`
(`v1/attention/backend.py:194-203`, `platforms/interface.py:628-641`), while a user-specified 16 is
**rejected** with `ValueError("No common block size for 16")` (`v1/worker/utils.py:296-320`) rather
than silently misaligned.

**`sparse_num_index_heads` MUST equal `num_key_value_heads` — and vLLM DOES assert it.**
`MinimaxM3QKVParallelLinearWithIndexer.__init__` raises
`"requires total_num_index_heads == total_num_kv_heads"`
(`vllm/model_executor/layers/linear.py:1462-1465`), and `minimax_m3_index_decode_score` asserts the
same at runtime. The equality matters because two independent derivations exist: the shared top-k
buffer is sized from `sparse_num_index_heads // tp` (`nvidia/model.py:798`), the layer uses
`num_kv_heads // tp` (`:433-436`). M3's shipped config has `num_attention_heads = 64`,
`num_key_value_heads = 4`, `sparse_num_index_heads = 4`. Keep an early check in our own config
validation for the better error message.

**The shared top-k buffer is token-major** — `[padded_num_tokens, num_index_heads, topk]` int32
(`nvidia/model.py:790-808`), transposed to `[H, tokens, topk]` by the attend
(`nvidia/sparse_attention_msa.py:78-82`). It is sized from `max_num_batched_tokens`.

**`d_idx = 128` is mandatory** — `get_supported_head_sizes() → [128]` on **both** backends
(`MiniMaxM3IndexerBackend`, `common/indexer.py:92`; `MiniMaxM3SparseBackend`,
`common/sparse_attention.py:103`), and `MinimaxM3QKVParallelLinearWithIndexer` additionally assumes
`index_head_size == head_size`. Conveniently `d_idx == d_h`, so the base rotary tables are reused
verbatim.

**Index-key side cache** (distinct KV-cache group, `MLAAttentionSpec`): `num_kv_heads = 1`,
`head_size = 128`, `bf16` → `128 × 2 B × 33 sparse layers` = **8.4 KB/token**, **+5.7%** on the
147 KB/token main KV cache (275 MB at 32K).

**Shared rotary — verified literal:** `self.index_rotary_emb = self.rotary_emb`, and the fused kernel
receives a **single `cos_sin_cache` and a single `rotary_dim`** for both branches.

**Full RoPE at `rotary_dim = 128` — VERIFIED supported.** M3 runs partial RoPE
(`partial_rotary_factor = 0.5` → `rotary_dim = 64`); Qwen3 needs 128. In
`csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`, `rotary_dim` is a **runtime**
argument (not a template parameter) and is explicitly admitted:

```cpp
STD_TORCH_CHECK(rotary_dim > 0 && rotary_dim % 8 == 0 &&
                    rotary_dim <= vllm::minimax_m3_fused_ops::kHeadDim,
                "rotary_dim must be a positive multiple of 8 and <= 128");
constexpr int kHeadDim = 128;      // hard-coded; no head_dim dispatch
```

**SM90-capable — verified:** bf16 requires SM80+, and there is an explicit `__CUDA_ARCH__ >= 900`
programmatic-dependent-launch path.

**The fused op is mandatory *inside M3's layer*.** Every reference to `index_q_norm`, `index_k_norm`,
`index_rotary_emb`, `rotary_emb`, `q_norm`, `k_norm` in `nvidia/model.py` occurs only inside the fused
call's argument list (the two call sites are `:370` and `:603`); no platform/env/`use_fused` branch
exists. **But it is optional for our port** — it only fuses RMS-norm + RoPE + the cache writes, all of
which vLLM already ships separately. Using Qwen3's own norm modules and a custom index-cache insert
removes the Gemma `w − 1` conversion of Qwen3's `q_norm`/`k_norm` entirely, at the cost of a few extra
launches per layer. Decide by measurement (plan §7.1).

**Two attention classes.** `MiniMaxM3SparseAttention.forward()` passes index args;
`MiniMaxM3Attention.forward()` omits them (dense layers 0–2); the host signature ends in
`bool skip_index_branch`. **The port needs both classes**, selected per layer by
`sparse_attention_freq`.

**Only 33 of 36 layers are sparse.** Index params `≈ 2.95M × 33 = 97M` (2.4% of 4B); dense layers get
no indexer, no side cache, **no KL term**.

### 2.3 Complexity (paper Eq. 12) and the Qwen3-4B numbers

```
F_GQA(N) = 2 H_q d_h N²
F_MSA(N) = H_kv d_idx N²        +    4 H_q d_h N k B_k
           └── Index Branch ──┘       └── Main Branch ──┘
```

At `N = 32768`, per layer: `F_GQA = 8.80 TFLOP`; `F_MSA = 1.10 (index) + 1.10 (main) = 2.20 TFLOP`
→ **4.0× attention-FLOP reduction per sparse layer.**

Whole model: attention `3 × 8.80 + 33 × 2.20 = 99.0 TFLOP` (vs `316.8` all-dense); non-attention
`2 · 4.02e9 · 32768 = 263.5 TFLOP`. Total `362.5` vs `580.3` → **1.60× prefill FLOP ratio** — not a
speedup: the paper is explicit (§5.4, p.12) that measured gains trail FLOP reduction because of index
construction, top-k, reverse-index materialisation, query gathering and load balancing.

**Decode reads: 8× at 32K, not 16×.** The index-key side cache is full-length and is read in its
entirety every decode step, so per sparse layer: dense `32768×8×2×128×2 B = 134.2 MB` vs sparse
`2048×8×2×128×2 B = 8.39 MB` (main) `+ 32768×128×2 B = 8.39 MB` (index) `= 16.8 MB` → **8.0×**. 16× is
the `N→∞` asymptote of `4096N/(4096·2048 + 256N)`. End-to-end (8.04 GB weights): **~1.4× at batch 1**,
**~4.2× at batch 32**. The paper's 7.6× decode figure is at **1M**, `G = 16`, with MiniMax's own
kernel — and vLLM has **no MSA decode kernel at all** (decode is Triton split-K on every platform).

---

## 3. Teacher construction

Per Eq. 9 — **softmax per head over the support, then average**:

```python
scores  = q @ repeat_kv(k_states).mT / sqrt(d_h) + causal_doc_bias   # [b, H_q, T_q, N]
p_head  = softmax(scores.masked_to(I_tok), dim=-1)                   # [b, H_q, T_q, |I_tok|]
P_group = p_head.view(b, H_kv, G, T_q, -1).mean(2).detach()           # [b, H_kv, T_q, |I_tok|]
```

### 3.1 Order matters — renormalize-then-average ≠ average-then-renormalize

Eq. 9 places the per-head softmax **inside** the `(1/G) Σ`, so each head normalizes over `I_tok`
*before* averaging. On the §1.2 numbers with support `{2,3,4,5}`:

```
Eq. 9 (renormalize per head, THEN average)   →  block1 mass = 0.1559
average over full support, THEN renormalize  →  block1 mass = 0.1511      ← WRONG
```

In Phase 1 (full support) the two coincide. The distinction bites only in Phase 2 — where the sparse
main branch already computes exactly the Eq.-9 quantity (§6.1).

---

## 4. Phase 1 — indexer warm-up

*"During the first few iterations, the model runs full attention in both branches and trains the newly
added index projections with `L_KL`. After warmup, the model switches to sparse attention, and `L_KL`
is computed over the top-`k` selected positions. The same schedule is used when sparsifying a
pretrained full-attention checkpoint"* — i.e. exactly our case.

Base frozen; attention dense (stock FlashAttention, outputs bit-identical to stock Qwen3);
`I^(r)_{i,tok}` = all causally visible tokens.

```python
P     = teacher(full_support).detach()                       # [1, 8, 512, 32768]
logp  = log_softmax(S_idx + causal_doc_bias, dim=-1)         # [1, 8, 512, 32768]
L     = kl_div(logp, P, reduction='none').sum(-1)            # [1, 8, 512]
```

Reduce `1/(N·H_kv)` within a layer; **sum over layers**, scaled by `λ` (Algorithm 1).

Gradient path is **dense** — max-pool is not in the graph:

```
dL/dS_idx  =  P^idx − P        [1, 8, 512, 32768]
dS/d{q_idx, k_idx, index_q_norm, index_k_norm}  : standard backward
```

**Local block forcing applies from Phase 1** — verbatim (p.5): *"the local block containing `i` is
always selected as part of `I^(r)_i` during both training and inference. **This fixed allocation
reserves one block slot and leaves the remaining slots to be chosen by the Index Branch**, preventing
degenerate selections that omit the query's immediate neighbourhood."*

> **So forced blocks are INSIDE the `k` budget.** Effective support is
> `|I^tok| = k × B_k = 16 × 128 = 2048` tokens, of which the local block occupies one of the 16 slots.
> The paper forces **only the local block**; M3 ships `sparse_init_block = 0` — no sink block.
> Appendix C.2 is explicit about why, and about *which* local block: forced sink + a fixed local
> window were early stabilisation devices, removing them changed reasoning/code/PPL and long-context
> retrieval "little" (Table 5), and therefore *"the final recipe does not force the first block or a
> large local window, and only forces the special **incomplete self block**."* Appendix A adds that
> the sink is learned anyway — every head puts substantial mass on the first token without being
> forced to (Fig. 6). vLLM matches: `local_mask = i + off_k >= max(0, valid_blocks - local_blocks)`,
> i.e. the last visible (partial) block.
>
> vLLM implements the forcing by sentinel injection **before** the top-k
> (`index_topk.py`): `score = tl.where(causal_mask & local_mask, 1e29, score)`, with `1e30` for init
> blocks. It also guards NaN: `score = tl.where(score != score, -1e30, score)`, so a fully-masked
> block carries **`-1e30`, not `-inf`**. **Training must replicate the mechanism**, not just the
> counts.

---

## 5. Memory — Phase 1 is the expensive stage

Per query tile, fp32, `N = 32768`, `T_q = 512`:

| tensor | shape | bytes | retained |
|---|---|--:|---|
| `scores` / `p_head` (per-head) | `[1, 32, 512, 32768]` | 2.0 GiB | no — accumulate over heads |
| **`P` teacher** | `[1, 8, 512, 32768]` | **512 MiB** | **yes** |
| **`S_idx` / `log_softmax`** | `[1, 8, 512, 32768]` | **512 MiB** | **yes** |
| `M_idx` block scores | `[1, 8, 512, 256]` | 4 MiB | selection + metrics only |

Token-level support means Phase 1 retains `O(T_q × N)` per layer — same order as DSA's token-level KL.
Mitigations, all existing knobs:

- **`kl_checkpoint = True`** — recompute the KL graph in backward. Required at 32K.
- **`kl_block_size = 256`** if 512 is tight → 256 MiB per retained tensor.
- **Accumulate the teacher over heads** — `for ℓ in H_r: acc += softmax(scores[ℓ])` keeps peak at one
  head (64 MiB) instead of all 32 (2.0 GiB).

Phase 2's support is `k·B_k = 2048` → `[1, 8, 512, 2048]` = **32 MiB**, 16× smaller.

Note: the paper's §4 describes a **sparse KL loss backward kernel**, which is *not* in the released
inference package. Our training KL is therefore a torch implementation.

---

## 6. Phase 2 — sparse training

Main branch block-sparse over the `k = 16` selected blocks; same Eq. 9–10 with
`I^(r)_{i,tok}` = tokens of those blocks.

```python
sel   = M_idx.topk(k, -1).indices.detach()      # [1, 8, 512, 16]  block ids, STOP-GRAD
I_tok = tokens_of(sel)                          # [1, 8, 512, <=2048] causally masked

P     = teacher_over(I_tok).detach()            # [1, 8, 512, 2048]  (see 6.1 — free)
logp  = log_softmax(gather(S_idx, I_tok), -1)   # [1, 8, 512, 2048]
L_kl  = kl_div(logp, P, reduction='none').sum(-1)

L     = L_lm + λ · Σ_layers L_kl
```

### 6.1 The teacher is free in Phase 2

The sparse main branch already softmaxes each head over exactly `I_tok`, so its attention weights
**are** Eq. 9's per-head distributions — no extra dense forward, and §3.1's ordering is correct by
construction:

```python
attn_w  = <main branch sparse attention weights>      # [1, 32, 512, 2048]   (128 MiB)
P_group = attn_w.view(1, 8, 4, 512, 2048).mean(2)     # [1,  8, 512, 2048]   ( 32 MiB)
```

A full-context teacher in Phase 2 would need the dense `O(N²)` attention that sparsity exists to
avoid. **Phase 2's KL is nearly free; Phase 1 is the expensive stage.**

### 6.2 Cost of the restriction

The index branch gets **no gradient about blocks it failed to select** — a block ranked 17th never
appears in the loss (§1.2 step 6). This makes the **Phase-1 warm-up load-bearing, not optional**: it
is the only stage that supervises the full ranking. If calibration drifts, the escape hatch is a
low-weight full-support KL on a small fraction of batches.

### 6.3 Gradient wiring

| path | trains | mechanism |
|---|---|---|
| `L_lm` → base | **base only** | `sel` detached (top-k non-differentiable) |
| `L_kl` → index branch | **index params only** | Eq. 11: `stopgrad(X)` into the index projections; teacher detached (Algorithm 1 line 7 uses `stopgrad(Q), stopgrad(K)`) |

Under **Phase 2a** (base frozen) `L_lm` trains nothing and only `L_kl` is active — the drift-free
control. **Phase 2b** unfreezes the base.

---

## 7. Provenance

**Verified from the PDF** (`/tmp/msa_2606.13392.pdf`, §3.2 p.5, Algorithm 1 p.6, §3.3 p.6, §4 p.6):
Eq. 9 (both distributions + both score definitions incl. `/sqrt(d_idx)`, `/sqrt(d_h)`), Eq. 10 (loss +
`N` = sequence length), Eq. 11 (`stopgrad(X)`; updates only `W^idx_q`, `W^idx_k`), Eq. 12
(complexity), Algorithm 1 (incl. `L = L_LM + λ Σ_layers L_KL`), the `I^(r)_{i,tok}` definition, the
two-stage warm-up schedule, local-block forcing reserving one slot, exp-free selection, `B_k = 128`
`k = 16`, and the per-thread register top-k design.

**Verified from vLLM source** (`main` @ `4f56321d`, post-PR #45381): `select_main_impl_cls` /
`select_indexer_impl_cls` dispatch, incl. `is_device_capability_family(100)` and
`topk_blocks ∈ {4,8,16,32}` gating and the bf16-only Triton indexer;
`SPARSE_BLOCK_SIZE = 128`; `get_supported_head_sizes() → [128]` and
`get_supported_kernel_block_sizes() → [128]` on both backends; token-major top-k buffer
`[padded_tokens, num_index_heads, topk]` int32; `BLOCK_SIZE_H` heuristics on `gqa_group_size`
(prefill `next_pow2`, decode `max(16, next_pow2)`); sentinel `1e30`/`1e29` injection with
`MASK_INIT=False, MASK_LOCAL=False` at the call site, NaN → `-1e30`, `-1` padding past
`valid_blocks`; `-inf` mask before `tl.max`; `self.index_rotary_emb = self.rotary_emb`;
`self.scaling` stored-but-unapplied and `qk = tl.dot(q, k)` with the explicit "score scale is
omitted" comment; **Gemma-style `(1+w)` norms with a shared `[128]` gain** in both
`MiniMAXGemmaRMSNorm` and the fused kernel; fused-kernel `rotary_dim` check, `kHeadDim = 128`, SM80
guard and SM90 PDL path, `skip_index_branch` as the last host arg; the two attention classes;
`MLAAttentionSpec(num_kv_heads=1, head_size=128)` side cache; `AttentionCGSupport.UNIFORM_BATCH`
alongside `@eager_break_during_capture` on the attend; `@torch.no_grad()` on all six kernel wrappers
with no `autograd.Function` anywhere; the `stacked_params_mapping` contract for
`index_q_proj`/`index_k_proj`; `_sparse_attention_layer_ids` returning `∅` for a missing config and
`load_weights` silently skipping unmatched names; `MINIMAX_M3_SPARSE` registry path; the
`total_num_index_heads == total_num_kv_heads` assert in
`MinimaxM3QKVParallelLinearWithIndexer`; block-size auto-selection; and M3's HF `config.json`.

**Verified from the paper's appendices** (2026-07-28): B.2/Fig. 7 (LM-only vs KL-only vs both;
KL-only *without* the index value head loses short-context ability), B.3/Figs. 8-9 (the stopgrad
rationale, and that larger KL coefficients diverge *without* it), B.4/Figs. 10-11 (attention-entropy
collapse motivating warm-up), C.1/Table 4 (block size 32/64/128 ≈ flat), C.2/Table 5 (forced sink and
wide local window removed; only the incomplete self block forced), C.3/Table 6 (the index value head
is droppable once warm-up exists), §5.1/§5.4 and Tables 2-3 (**MSA-CPT**: 2.6T dense → 400B convert
(40B warm-up) → ~140B long-context; residual −2.6 RULER-8K, −3.1 HumanEval).

**Added here, not from either source:** the `captured_mass` gate and oracle decomposition (§8),
`group_divergence` as a *metric* (§8 — the phenomenon is in Appendix A), the Phase-2a/2b split, all
Qwen3-4B shapes/byte counts, the decode-read and end-to-end decode estimates (§2.3), and the memory
mitigations.

**Corrected 2026-07-28 (were wrong here):** the top-k buffer shape; "`sparse_num_index_heads` … NOT
asserted"; `indexer_kv_dtype` as a model-config field; `--block-size 128` as mandatory;
`get_supported_head_sizes` attributed to `MiniMaxM3Indexer`; "per-head RMSNorm" without the Gemma
`(1+w)` and shared-`[128]` details; the `i ≥ 4096` probe floor; and "≈1.60× prefill speedup" /
"~16× fewer KV reads" stated without their qualifiers.

---

## 8. Metrics (paper §5.2) — block-level, unlike the loss

```
I*  = top-k block set by MAIN BRANCH mass       Î = index selection       |I*| = |Î| = k
P_b = main-branch attention probability summed over tokens in block b

block recall  =  |I* ∩ Î| / |I*|                              # set identity
score recall  =  Σ_{b ∈ I*∩Î} P_b  /  Σ_{b ∈ I*} P_b          # oracle-relative mass
```

Block-level analogues of the existing `topk_overlap` / `topk_recall`
(`docs/dsa_indexer_metrics.md`), with one difference: **score recall is oracle-relative**.

The paper observes score recall *above* block recall (§5.2: *"The higher score recall further shows
that the retrieved blocks account for most of the Main Branch attention mass"*, Fig. 3). **That is an
empirical property of a trained selector, not an inequality that holds by construction** — an earlier
version of this line asserted it as a general fact. Measured on a random-init indexer at Qwen3-0.6B
(`tests/msa/test_qwen3_msa_phase1.py`), the direction reverses: block recall 0.689 vs score recall
0.621, because a random selector hits random members of `I*` rather than its high-mass members. So
`score_recall > block_recall` is a *sign that training is working*, worth watching in Phase 1 — not an
assertion to put in a test.

Add an absolute quantity as the **gate** — neither paper metric detects a budget that is too small:

```
captured_mass  =  Σ_{b ∈ Î} P_b            # ÷ 1, over Î (not the intersection)
```

Decomposition against two oracles at the same 2048-token budget:

```
1.0
 ├── unreachable        = 1 − Σ_{b ∈ I*_token} p        (budget too small — irreducible)
 ├── granularity cost   = oracle_token − oracle_block   (cost of whole 128-token blocks)
 └── index quality gap  = oracle_block − captured_mass  (cost of the learned selector)
```

Both oracle terms need **only dense attention, no index branch**, so they are measurable on stock
Qwen3-4B today — **done, see plan §12.** Restrict the sampled query positions so partial blocks at the
causal frontier don't skew it; the run used `min_pos = 2·min(ks)·max(B_k) = 2048` plus a per-`k`
validity mask `vis_blocks > k`, so the effective floor at `k = 16` is position ≥ 2048 (an earlier
version of this line said 4096). Note also that the oracle takes the *unconstrained* top-k by mass
while MSA spends one slot on the forced local block, so `oracle_block` is a slightly **loose** upper
bound for the deployed configuration.

Computing `I*` needs the true attention distribution → these are **periodic** diagnostics; reuse the
existing `diag_interval = 10` gating.

Also log **`group_divergence`**: how much the 8 groups' selections differ — the empirical test of
per-group selection. If all groups pick the same blocks, per-group capacity is wasted. The paper
expects divergence (Appendix A, Fig. 5): *"different groups attend to different long-range stripes
while sharing the common local and sink patterns"*, at both layer 1 and layer 18 — so the metric is
ours, but the phenomenon it tests is reported.

---

## 9. Open items

1. ~~Re-verify Eq. 9 against the PDF~~ — **DONE** (2026-07-28). Corrections applied: the loss is
   **Eq. 10** (Eq. 9 is the distributions); `N` = sequence length; the paper **does** specify
   `/sqrt(d_idx)`; the paper's index branch has **no norms** (Eq. 11); layer reduction is a **sum ×
   λ**, not a mean; local block **reserves one slot inside `k`**.
2. **Numerical parity test** vs `minimax_m3_index_score` / `_topk` / `_decode` — set equality primary;
   plus sentinel (`-1e30`), `-1` padding past `valid_blocks`, forced-block presence, tie census
   (bitonic output is unsorted), and the decode split-K path separately. Plus a **numerical** diff of
   `fused_minimax_m3_qknorm_rope_kv_insert` against a torch reference at `rotary_dim = 128` — this
   replaces the old "architecture match" assertion, which could not catch the Gemma `(1+w)` form, the
   shared-`[128]` gain, or the norm→RoPE order. The full must-match / need-not-match checklist is in
   plan §4.2 #2. See `tests/dsa/test_minicpm3_dsa_indexer_parity.py` for the pattern.
   **The kernels cannot be trained through** — all six wrappers are `@torch.no_grad()`, the score
   kernel applies `tl.max` internally and never emits token-level `S^idx`, its K side is a paged
   cache, and decode selection is a different algorithm. Torch reimplementation + kernel-as-oracle is
   the only route.
3. ~~**Oracle probe** (§8) on stock Qwen3-4B at 32K~~ — **DONE 2026-07-28**, plan §12.
4. **Measure the decode head-axis pad at `G = 4`** — vLLM's decode heuristic is
   `BLOCK_SIZE_H = max(16, next_pow2(G))`, so 4 real heads occupy 16 slots. M3 (`G = 16`) never sees
   this. Decode is bandwidth-bound so the cost is probably small; measure.
5. **Assert `dL/dS_idx = P^idx − P`** in a unit test.
6. **Phase-1 memory** — confirm `kl_checkpoint` + head accumulation fit 33 sparse layers at
   `T_q = 512`; fall back to 256.
7. ~~**Pick `λ`** — check the experiments/appendix~~ — **CLOSED 2026-07-28: the value is nowhere in
   the paper.** A full-text search finds `λ` only symbolically, in Algorithm 1 and Eqs. 18–19. Choose
   it empirically by matching gradient-norm scale; B.3 says that *with* the Eq. 11 stopgrad the
   coefficients that diverged without it were stable, so the realistic failure mode of a bad `λ` here
   is an under-trained indexer, not a destabilised backbone.
8. **Decide the norm/RoPE path** — reuse the fused op (and convert Qwen3's `q_norm`/`k_norm` by
   `w − 1`) or implement norm+RoPE+index-cache-insert ourselves (§2.1, plan §7.1).
