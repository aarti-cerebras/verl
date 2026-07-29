# Qwen3-4B → MSA — training and inference plan

Adapting **Qwen3-4B-Thinking-2507** from dense GQA to **MiniMax Sparse Attention (MSA)**: a
lightweight per-GQA-group Index Branch scores 128-token KV blocks, top-16 are selected, and the Main
Branch runs exact block-sparse attention over only those blocks. Target: **32K context**, preserving
long-context and reasoning capability.

**Loss and index-branch details:** [kl_loss.md](kl_loss.md) — equations, shapes, memory, and the
full paper-vs-vLLM provenance ledger. This document is the plan; that one is the spec.

**Decisions (2026-07-28):**
- **MSA, not DSA-on-GQA** (§2).
- **Target:** `Qwen/Qwen3-4B-Thinking-2507` (§1).
- **Config:** `B_k = 128`, `k = 16`, `d_idx = 128`, `H_kv = 8` index heads, `init 0 / local 1`,
  layers 0–2 dense / 3–35 sparse, `bf16` index cache (§3).
- **Phases:** 0 (implement + parity) → 1 (indexer warm-up, base frozen) → 2a (sparse, base frozen,
  **measurement stage**) → 2b (unfreeze — plan for it, don't hope to skip it).
- **Training:** verl, pure torch, own KL implementation. **Serving:** vLLM ≥ PR #45381, reusing
  `MiniMaxM3SparseTritonImpl` + `MiniMaxM3Indexer` (§7).

**Status:** planning complete, nothing implemented. All feasibility unknowns closed; remaining items
are tests (§10).

> **Audited 2026-07-28** against the paper PDF (`/cb/ml-eng/aarti/msa/refs/msa_2606.13392.pdf`),
> vLLM `main` @ `4f56321d`, both HF `config.json`s, and the §12 artifacts. Ten factual errors were
> found and fixed; the changes are listed in §13. Every claim below that cites a paper page or a
> `file:line` was read at the source. Claims **not** so marked are inference or in-repo prior art.

---

## 1. Target model

`Qwen/Qwen3-4B-Thinking-2507` — 36 layers, hidden 2560, **32 q heads / 8 KV heads, head_dim 128**
(`G = 4`), intermediate 9728, `rope_theta = 5e6`, `rope_scaling = null`,
**`max_position_embeddings = 262144`**, tied embeddings, thinking-only.
*All of the above verified against the HF `config.json`, plus `rms_norm_eps = 1e-6`.*

Chosen over `Qwen3-4B` (rope_theta 1e6, max_pos 40960) and `Qwen3-4B-Instruct-2507`:

1. **32K sits deep inside its native range** — no YaRN. The original 4B caps at 40960, so 32K is at
   the edge of its trained range, and a dense reference that is itself marginal at the target length
   makes every Δ meaningless.
2. **Thinking mode is the capability worth keeping**, and it is the regime that stresses sparse
   attention hardest (§1.1).
3. Geometry is identical across all three, so the choice is reversible at zero code cost.

**Not** `Qwen3.5-4B` (`/cb/ml-eng/aarti/models/qwen3p5_4b`) — hybrid linear attention, only 8 of 32
layers are full attention, plus a vision tower. Different project.

**On disk:** `/cb/ml-eng/aarti/models/qwen3_4b_thinking_2507` — this is the checkpoint the §12 oracle
probe ran against.

**Dev ladder:** local `qwen3_0p6b` (28L, 16q/8kv, hd 128) for Phase-0 correctness at 4K → local
`qwen3_1p7b` (28L, 16q/8kv, hd 128) for a Phase-1 smoke → 4B for real runs. All three share
`head_dim = 128` and 8 KV heads, so the module code is identical. **But both dev models are
`G = 2`, not 4** — and `G` drives the teacher's `1/G` average, the decode head-axis pad
(`max(16, next_pow2(G))`), and the prefill `BLOCK_SIZE_QH` packing. Only the 4B exercises `G = 4`;
budget one 4B parity run before trusting the smoke.

### 1.1 What thinking mode changes

1. **Sparsity bites on every item, including short-prompt benchmarks.** A 500-token AIME prompt with
   an 8K trace means ~94% of forward passes attend over self-generated context.
2. **The selection ratio degrades *within* a response.** `k·B_k` is fixed at 2048; `N` grows every
   token, so a 32K prompt + 12K trace slides 6.25% → 4.6% mid-generation. No fixed-length grid
   captures this — recall must be measured against **generated-trace position**.
3. **New failure mode:** losing track of its own reasoning → repetition loops, restarted derivations,
   never emitting `</think>`. Accuracy masks it; gate on trace integrity.
4. **Thinking traces belong in the Phase-1 data mixture**, not just documents.
5. **Sampled decoding**, not greedy: `temperature 0.6, top_p 0.95, top_k 20, do_sample: true` —
   verified from the model's `generation_config.json` (the card additionally recommends `MinP = 0`).
   This invalidates the MiniCPM3 greedy protocol and forces multi-sample scoring with error bars.
6. **Traces can run past the adaptation window.** The model card recommends a 32,768-token output
   budget, and 81,920 for competition math/code. With a 32K adaptation length, prompt + trace will
   exceed the trained regime on exactly the benchmarks we care about. Either cap output length and
   say so, or state the extrapolation explicitly and expect the sparse Δ to grow past 32K.

---

## 2. Why MSA rather than DSA-on-GQA

We started from DeepSeek DSA (as in the MiniCPM3 work) and moved to MSA. The reasoning, recorded so
it isn't relitigated:

| | DSA on GQA | **MSA** |
|---|---|---|
| designed for | MLA (MQA-shaped latent) | **GQA** — "a blockwise sparse attention built upon Grouped Query Attention" (abstract) |
| selection | per-query, **token**-granular | per-group, **block**-granular (128) |
| arithmetic intensity, Q-outer | **≈ G = 4** for Qwen3-4B (~4 FLOP/byte) | fixed by **KV-outer**: `≈ (2/3)·B_k ≈ 85` |
| serving kernel for GQA | **none exists** — every sparse backend in vLLM's registry is MLA-shaped; must be written, incl. a backward | **exists, merged in vLLM**, Triton path on SM90 |
| training precedent for adaptation | V3.2 2-stage | **MSA §3.2 + §5's MSA-CPT run** (see caveat) |

The decisive facts: MSA's paper formalizes the intensity problem as `≈ G` for a Q-outer loop
(Eqs. 13–14, p.7), and Qwen3-4B's `G = 4` is the worst case (M3's is 16). MSA fixes it with a
KV-outer loop (Eqs. 15–16) that is independent of `G`, and vLLM already ships that kernel with a
Triton path that runs on our H100s. DSA's token-granular path would have required a new GQA sparse
kernel *with a hand-written backward*.

**What the paper actually says about MLA** — it is one sentence of Related Work (p.13), not a design
argument: *"DSA (DeepSeek-AI et al., 2025) sits on top of MLA in its MQA mode: a multi-head
ReLU-based lightning indexer scores tokens individually, all query heads share a single Top-k index,
and selection is token-level. MSA differs from this neighborhood along two axes that are taken up
together: per-GQA-group Top-k sharing combined with block-level selection, which gives multi-group
block-granular retrieval while keeping KV reads contiguous."* The only adjacent claim is the intro's
*"relaxing the head-dimension constraints imposed by prior designs."* There is **no latent-KV
overhead argument in the paper** — don't attribute one.

**The caveat that actually applies (§5, p.9).** The paper has *two* 109B-scale runs, and the second
one is our setting:

- **MSA-PT** — native sparse pretraining, 3T tokens (40B indexer warm-up, then sparse).
- **MSA-CPT** — start from a **GQA full-attention checkpoint trained on 2.6T tokens**, replace dense
  attention with MSA, continue for **400B tokens** (the first **40B** are indexer warm-up), then a
  further **~140B** tokens of long-context training (Table 3).

So a conversion precedent exists — at **400–540B tokens, i.e. 100–350× our planned budget** (§8).
And at that budget MSA-CPT still shows residual gaps (Table 2): RULER-8K 79.8 → 77.2 (**−2.6**),
HumanEval 61.0 → 57.9 (**−3.1**), GSM8K 76.2 → 73.7 (−2.5), while RULER-32K is roughly flat
(75.0 → 75.7) and HELMET-128K overall is −0.60 after the extension stage. **The paper does not itself
clear a uniform `Δ ≤ 2 pts` bar on code and short-context retrieval.** Set our acceptance gate
accordingly (§9) instead of assuming a tighter one is reachable at 1–4B tokens.

---

## 3. Architecture and configuration

### 3.1 Config

Model-side (`config.json` → `sparse_attention_config`):

```python
sparse_index_dim       = 128        # d_idx; ENFORCED by the backends' get_supported_head_sizes() -> [128]
sparse_num_index_heads = 8          # MUST equal num_key_value_heads -- ASSERTED, see below
sparse_topk_blocks     = 16         # k; also inside the SM100 CuTe path's {4,8,16,32}
sparse_block_size      = 128        # B_k; hardcoded SPARSE_BLOCK_SIZE = 128
sparse_init_block      = 0          # paper C.2 + M3 default: no forced sink block
sparse_local_block     = 1          # the local block reserves ONE of the 16 slots
sparse_score_type      = "max"
sparse_attention_freq  = [0]*3 + [1]*33     # layers 0-2 DENSE, 3-35 SPARSE (matches the released M3
                                            # config verbatim: [0]*3 + [1]*57 over 60 layers).
                                            # NOTE: the §12 probe does NOT independently confirm this
                                            # choice -- see §12 Finding 1.
use_sparse_attention   = True
# Qwen3 also needs, for the M3 layer's get_rope() call:
partial_rotary_factor  = 1.0        # -> rotary_dim = 128 (M3 ships 0.5 -> 64)
head_dim               = 128        # read directly as config.head_dim
```

Engine-side (**not** model config): `indexer_kv_dtype` is a vLLM `AttentionConfig` field
(`vllm/config/attention.py:13,67`, `Literal["bf16","fp8","mxfp4","nvfp4"]`, default `bf16`), set via
`--attention-config '{"indexer_kv_dtype": ...}'`. It does not appear in M3's `config.json`.
**On SM90 bf16 is not a preference, it is the only option** — `select_indexer_impl_cls` raises
`NotImplementedError` for any non-bf16 index cache off SM100 (`common/indexer.py:513-518`).

**`--block-size` does not need to be passed.** Both backends declare
`get_supported_kernel_block_sizes() -> [128]`, and when the user has not specified a block size vLLM
calls `get_preferred_block_size`, which returns `min(supported) = 128`
(`v1/attention/backend.py:194-203`, `platforms/interface.py:628-641`). A user-specified 16 is
**rejected** — `try_get_kernel_block_size` raises `ValueError("No common block size for 16")`
(`v1/worker/utils.py:296-320`) — it does not silently misalign. Passing `--block-size 128`
explicitly is harmless and self-documenting.

**`sparse_num_index_heads == num_key_value_heads` is asserted by vLLM**, contrary to an earlier note
here: `MinimaxM3QKVParallelLinearWithIndexer.__init__` raises
`"requires total_num_index_heads == total_num_kv_heads"`
(`model_executor/layers/linear.py:1462-1465`), and the same class assumes
`index_head_size == head_size`. There is a second runtime assert on the decode path
(`assert num_idx_heads == num_kv_heads` in `minimax_m3_index_decode_score`). Two independent
derivations exist — the shared top-k buffer is sized from `sparse_num_index_heads // tp`
(`nvidia/model.py:798`) while the layer uses `num_kv_heads // tp` (`model.py:433-436`) — which is
*why* the equality matters. Keep an early check in our own config validation anyway; it costs
nothing and gives a better message.

**The shared top-k buffer is token-major** `[padded_num_tokens, num_index_heads, topk]` int32
(`nvidia/model.py:790-808`), transposed to `[H, tokens, topk]` by the attend
(`nvidia/sparse_attention_msa.py:78-82`). It is sized from `max_num_batched_tokens`, so a large
chunked-prefill budget costs real memory.

### 3.2 Index branch

Per **sparse** layer (33 of 36):

```python
W_q_idx      : [2560, 8 * 128]      # one index query per GQA group
W_k_idx      : [2560, 128]          # ONE shared index key head (MQA-style; paper Eq. 5, Alg. 1 L2)
index_q_norm : Gemma-style RMSNorm, ONE shared [128] gain across all 8 index heads   <- TRAINABLE
index_k_norm : Gemma-style RMSNorm, ONE shared [128] gain                            <- TRAINABLE
rotary       : the SAME module as main attention (rope_theta 5e6, rotary_dim 128)
```

Forward: `q_idx = RoPE(index_q_norm(W_q_idx(stopgrad(h))))`, likewise `k_idx`. Then
`S^idx = q_idx @ k_idx.mT / sqrt(d_idx)`, mask (`-inf`) → `M = blockmax(S)` → `topk(16)` on raw
scores (exp-free), local block forced via sentinel injection.

**Norm convention — pinned, because the serving kernel hardcodes it.** The fused op computes
`x · rsqrt(mean(x²)+eps) · (1 + w)` (Gemma-style), with `w` zero-initialised
(`csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:17,135-145`;
`MiniMAXGemmaRMSNorm` at `nvidia/model.py:114-141`), and asserts
`index_{q,k}_norm_weight.numel() == 128` with dtype equal to the qkv dtype (`kernel.cu:711-718`).
Consequences:

- **Train the index norms in Gemma form** (`(1+w)`, zero-init). The parameterisation is an exact
  reparameterisation of standard RMSNorm (`gain = 1 + w_gemma ≡ w_std`) with identical gradients and,
  for matched init, identical gain trajectories — so nothing is lost, and train object == serve
  object with no conversion. It is also safer under weight decay (shrinks toward gain 1, not 0).
- **One shared `[128]` gain per branch, not per-head.** "Per-head RMSNorm" here means *applied*
  per head, not *parameterised* per head. A `[H_kv, 128]` implementation is a real architecture
  mismatch that no reparameterisation fixes.
- **Qwen3's own `q_norm`/`k_norm` need a `w − 1` shift** *if* we reuse the fused op. Their gains
  average 1.79 / 2.35 with minima near 0.007, so feeding them raw distorts channels by
  `(1+w)/w` ∈ [1.02×, ~150×] — a loud failure, not a silent one, but a mandatory export step.
  Alternative: our layer is free to do norm + RoPE with Qwen3's own modules and write the index key
  into the side cache separately, in which case Gemma semantics never enter (see §7.1).

**Order is norm-then-RoPE**, and the scale `1/sqrt(d_idx)` is the paper's (vLLM omits it; selection
is scale-invariant, so only the loss temperature is affected). One consequence of zero-init: at init
`gain = 1`, so **serving raw scores are `sqrt(128) = 11.3×` larger than the training scores we
softmax**. Harmless for top-k; but never compare raw index-score magnitudes across train and serve,
and put no absolute thresholds on index scores in diagnostics. Sentinel headroom is a non-issue
(reaching `1e29` would need gains ~`1e13`), but assert it once. Details and provenance:
[kl_loss.md](kl_loss.md) §2.

### 3.3 Cost accounting

Paper Eq. 12: `F_GQA = 2 H_q d_h N²`, `F_MSA = H_kv d_idx N² + 4 H_q d_h N k B_k`.

| at N = 32768 | value |
|---|--:|
| dense attention, per layer | 8.80 TFLOP |
| MSA per sparse layer (index 1.10 + main 1.10) | **2.20 TFLOP** → **4.0×** reduction |
| whole-model attention (3 dense + 33 sparse) | 99.0 TFLOP (vs 316.8 all-dense) |
| non-attention | 263.5 TFLOP |
| **prefill total** | **362.5 vs 580.3 → 1.60× FLOP ratio** |
| index-branch params | 97M (**2.4%** of 4.02B) |
| index-key side cache | 8.4 KB/token (**+5.7%** on 147 KB/token main KV) |
| main KV cache at 32K | 4.83 GB |

`1.60×` is a **FLOP ratio, not a speedup** — the paper is explicit that measured runtime gains trail
FLOP reduction because of index construction, top-k, reverse-index materialisation, query gathering
and load balancing (§5.4, p.12).

**Decode reads: ~8× at 32K, not 16×.** The index-key cache is full-length and is read in its entirety
every decode step (side cache is `MLAAttentionSpec(num_kv_heads=1, head_size=128)`,
`common/indexer.py:159-165`; the decode score kernel scans all blocks). Per sparse layer at 32K:

```
dense  main read      = 32768 × 8 × 2 × 128 × 2 B = 134.2 MB
sparse main read      =  2048 × 8 × 2 × 128 × 2 B =   8.39 MB
sparse index-K read   = 32768 × 1     × 128 × 2 B =   8.39 MB      <- omitted in the old 16× figure
                                                     ---------
                                                       16.8 MB  ->  8.0×
```

16× is the `N → ∞` asymptote (`4096N / (4096·2048 + 256N)`), reached at ~1M, not at 32K. End-to-end,
batch 1: dense ≈ 8.04 GB weights + 4.83 GB KV = 12.9 GB/token vs MSA ≈ 8.23 + 0.96 = 9.2 GB/token →
**~1.4×**; at batch 32 → **~4.2×**. So the decode win is real but strongly batch- and
length-dependent, and the paper's headline 7.6× was measured **at 1M, with G = 16, using MiniMax's
own kernel** — which vLLM does not have for decode at all (§7.1). The deliverable is a measured
**throughput/latency surface over (context length × batch)**, not a memory saving; GQA at 4B has no
MLA-style KV-memory problem.

---

## 4. Phase 0 — implement and prove parity

### 4.1 Build

**`verl/models/transformers/qwen3_msa.py`** — mirror `minicpm_dsa.py`'s decomposition
(`attach_indexers` / `freeze_base_train_indexer` / `install_kl_accumulation` /
`_dense_warmup_kl` / `_sparse_attn_and_kl` / patched forward with `mode ∈ {dense_warmup, sparse}`).

New/changed vs. the DSA module:

| component | action |
|---|---|
| `MSAConfig` (new, or extend `DSAConfig`) | drop FP8/UE8M0/Hadamard; add `B_k`, `k`, `init_blocks`, `local_blocks`, `score_type`, `sparse_layers` |
| index branch | **new, simpler than `dsa_indexer.py`**: 2 projections + 2 Gemma-form RMSNorms (shared `[128]` gains), no ReLU-weighted sum, no FP8 |
| `_dense_warmup_kl` | **reuse the tiling/checkpoint machinery**; change the target to per-group (`repeat_kv`, `.view(b,H_kv,G,T_q,-1).mean(2)`) and the student to `softmax(S^idx)` |
| block max-pool + top-k + sentinel forcing | new, ~30 lines; must replicate vLLM's semantics exactly — see the checklist in §4.2 |
| sparse forward (Phase 2) | **reuse the `_sparse_attn` gather pattern**: gather the 16 selected blocks (2048 tokens) per (query, group), einsum, softmax, einsum. Same cost class as the existing MiniCPM3 Phase-2 path (§8) |
| `_sparse_indexer_kl` | reuse; support becomes the selected blocks' tokens |
| per-layer gating | only 33 of 36 layers get an indexer and a KL term |

**Loss assembly:** `L = L_LM + λ · Σ_{sparse layers} L_KL`, with `1/(N·H_kv)` inside each layer.

**Layer reduction: `kl_reduction='mean'` in both phases (configurable via `msa_kl_reduction`).** Each
layer's indexer parameters are disjoint, so `∂L/∂θ_i = ∂KL_i/∂θ_i` and the reduction over layers is a
pure gradient *scale* — `sum == mean × n_sparse_layers`, same optimum, exactly equivalent to `LR ×
n_layers`. Verified in `tests/msa/test_qwen3_msa_phase1.py` (§1b). We take `mean` because:
- In **Phase 1** the reduction has no meaning at all: with the base frozen there is no `L_LM` to trade
  against, so it only rescales the index LR.
- It keeps the reported `grad_norm` usable as a diagnostic rather than pinned against verl's
  `clip_grad = 1.0` default (`workers/config/optimizer.py:53`). The DSA Phase-1 runs sat at
  `grad_norm ≈ 330` — ~330× clipping every step — and `sum` over 33 layers would push ~33× deeper in,
  where the reduction stops affecting the actual step at all because the magnitude is discarded.

**Phase-2 `λ` conversion — the one place this bites.** Algorithm 1 is
`L = L_LM + λ · Σ_layers L_KL`. Under `mean`, matching a paper `λ` requires
**`λ_ours = λ_paper × n_sparse_layers`** (33). Using a paper `λ` verbatim with `mean` trains the indexer
33× weaker against `L_LM` than intended. Set `msa_kl_reduction=sum` for a literally paper-faithful
Phase 2.

**Why the training path must be re-implemented in torch rather than calling vLLM's kernels** — all
six wrappers are `@torch.no_grad()` with no `autograd.Function` anywhere
(`common/ops/index_topk.py:647,706,759,857`; `common/ops/sparse_attn.py:513,600`); the score kernel
applies `tl.max` *inside* and never materialises token-level `S^idx`, which is what Eq. 9's student
needs; its K side is a **paged cache** (`[num_blocks,128,head_dim]` + `block_table` + `cu_seqlens`),
which does not exist in training; and decode selection is a different algorithm (split-K partial +
merge) with no training analogue. MiniMax describes a sparse-KL backward kernel in §4.3 but released
only the inference kernel (p.1). **So: mirror the semantics, and use the kernels as the test oracle.**

**Checkpoint layout for serving** (settles a question §7 used to leave open): keep the index
projections **separate** — `self_attn.{index_q_proj, index_k_proj}.weight` and
`self_attn.{index_q_norm, index_k_norm}.weight`. vLLM folds q/k/v/index_q/index_k into its single
fused GEMM itself via `stacked_params_mapping` (`nvidia/model.py:903-909`). No manual fusion.

### 4.2 Exit criteria (tests under `tests/msa/`)

1. **Dense equivalence:** with `k ≥ ceil(N / B_k)` (all blocks selected) the sparse path reproduces
   stock Qwen3 logits to bf16 noise. The faithfulness control the whole eval ladder rests on.
   *Training-side only:* at serving time `k = 256` forces the Triton top-k, whose
   `tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)` with `BLOCK_SIZE_T = next_pow2(256)` is violated
   by three of its six autotune configs (`common/ops/index_topk.py:172-183,209`). Verify before
   relying on it as a *served* control; Phase-2a is the cheaper substitute (§9).
2. **Index parity vs. vLLM kernels** — `minimax_m3_index_score` / `_topk` / `_decode`. Selected-block
   **set** equality (primary, target 1.0000 mean and worst-query). Pattern:
   `tests/dsa/test_minicpm3_dsa_indexer_parity.py`. The semantics that must match, read off the
   kernels:
   - block grid anchored to **absolute** position (`blk = pos // 128`), never to the query tile, with
     `valid_blocks = (pos + B_k) // B_k` so the partial block containing `i` counts as visible
     (`index_topk.py:229`);
   - `-inf` mask **before** the max, applied only on the diagonal tile
     (`if q_start < i + BLOCK_SIZE_K`, `:150-152`);
   - `max` pooling (`score_type="max"`);
   - forcing by sentinel **before** the top-k, local = last `local_blocks` valid blocks at `1e29`,
     init at `1e30`, with `MASK_INIT=False, MASK_LOCAL=False` as the prefill wrapper passes
     (`:753-754`) — the *forcing* branch, not the suppression branch;
   - the forced local block consumes **one of the 16** slots, not a 17th;
   - NaN → `-1e30`, fully-masked → `-1e30`, `-1` written into slots beyond `valid_blocks`.

   What need **not** match: the score scale (top-k is scale-invariant — keep the paper's
   `1/sqrt(d_idx)` in training for the KL temperature), output ordering (bitonic output is unsorted —
   compare sets, and census ties), and bit-exact scores (the kernel dots bf16 into fp32; near-threshold
   blocks can flip — this is why the target is set overlap ≈ 1.0000, cf. the MiniCPM3 UE8M0
   experience). Test the decode split-K path separately.
3. **Fused op at `rotary_dim = 128`, numerically** — run
   `fused_minimax_m3_qknorm_rope_kv_insert` on Qwen3-shaped input and diff against a torch reference.
   Admissible by the kernel's own check (`rotary_dim > 0 && %8 == 0 && <= 128`, `kernel.cu:678-680`)
   but never exercised at 128 by M3. **This test subsumes the old "architecture match" assertion:**
   asserting module type and `eps` cannot catch the Gemma `(1+w)` parameterisation, the shared-`[128]`
   gain shape, or the norm→RoPE order — only a numerical diff can.
4. **`dL/dS^idx = P^idx − P`** asserted.
5. **Serving-path sparsity gate** — the checks in §7.3, wired into the harness before any benchmark
   runs.
6. **Oracle probe** (§12) on stock Qwen3-4B — done.
7. **Tied-embedding** consolidate/reload round trip (`consolidate_indexer_ckpt.py`).
8. `dsa_*`-style flat override keys wired for Qwen3.

Smoke everything on local Qwen3-0.6B at 4K before spending 4B GPU-hours — noting that 0.6B is
`G = 2` (§1), so the `G = 4` paths need one 4B run.

---

## 5. Phase 1 — indexer warm-up (base frozen)

Directly per the paper (§3.2, p.5): *"During the first few iterations, the model runs full attention
in both branches and trains the newly added index projections with `L_KL`. After warmup, the model
switches to sparse attention, and `L_KL` is computed over the top-`k` selected positions. The same
schedule is used when sparsifying a pretrained full-attention checkpoint."*

Base **fully frozen**, attention **dense** (stock FlashAttention → outputs bit-identical to stock
Qwen3, so this phase carries **zero capability risk** — assert unchanged logits on a fixed batch).
This holds because the final recipe has **no index value head**: the Index Branch output is
discarded (paper C.3; vLLM sets `sparse_disable_index_value` on every sparse layer). Only
`{W_q_idx, W_k_idx, index_q_norm, index_k_norm}` train.

**The paper's own evidence that this phase is load-bearing** (worth citing, because it is stronger
than our §6.2 argument): B.4/Fig. 10-11 — main-branch attention entropy drops sharply in early
training, so top-k from step zero makes the indexer chase a moving target; warm-up fixed both
short-context and long-context retrieval. B.2/Fig. 7 — *KL-only without the index value head* lost
short-context ability in the pilot, and C.3/Table 6 shows the value head became droppable **only
once warm-up was in place**. Our recipe (KL-only + no value head + warm-up) is the paper's final
recipe; the warm-up is what makes it viable.

| knob | value | why |
|---|---|---|
| `seq_len` | **32768** | the MiniCPM3 run's Phase-2-at-4096 was the mistake to avoid |
| index LR | 1e-3, cosine, 10% warmup | validated Phase-1 setting (in-repo, `docs/dsa_indexer_worklog_2026-07-09.md`) |
| `kl_block_size` / `kl_checkpoint` | 512 / **true** | required at 32K; retains 2 × 512 MiB per tile |
| teacher head accumulation | on | keeps peak at one head (64 MiB) vs all 32 (2.0 GiB) |
| tokens | ~0.5–1B | base frozen, no base backward |
| data | long real docs (one doc per row) **+ self-generated thinking traces**, retokenized with the Qwen3 tokenizer | never reuse another model's `token_ids` |
| weight decay on index norms | **0** | the `(1+w)` form shrinks toward identity, but keep it out of WD entirely so train/serve semantics need no thought |

**Budget calibration — two reference points, and they disagree by 20×:**

| recipe | indexer warm-up | LR | source |
|---|--:|--:|---|
| MSA (109B MoE, both PT and CPT) | **40B tokens** | not given | arXiv 2606.13392 §5.1, p.9 |
| DeepSeek-V3.2 DSA (671B MoE) | **2.1B tokens** (1000 steps × 16 seqs × 128K) | **1e-3** | arXiv 2512.02556 §2.1.1, verbatim |
| **ours (4B dense)** | **0.5–1B** | 1e-3 | this plan |

DeepSeek's 2.1B at LR 1e-3 on a far larger model is the closer analogue, and it makes our 0.5–1B look
far less under-budgeted than the 40B figure alone suggested — it also independently corroborates the
1e-3 peak LR we inherited from the MiniCPM3 runs. The bet remains that warming up 97M parameters
against a frozen teacher is cheaper than warming up an indexer inside a live 109B pretraining run; the
gates below verify it. If gate 1 or 3 stalls, the first lever is more Phase-1 tokens, not Phase 2.

**Gates — do not start Phase 2 until all pass**, on a held-out long-context val set:

1. **`captured_mass ≥ 0.90 × oracle_block`, per layer** — an **oracle-relative** gate, not absolute.
   The §12 probe shows an absolute per-layer floor of 0.80 is *unachievable* at `k = 16` on this
   model: even a perfect selector caps at 0.767 on layer 3 and 0.793 on layer 4. This is exactly why
   the paper's `score recall` is oracle-relative; an absolute gate would fail on layers whose ceiling
   is simply low. Read absolute `captured_mass` alongside it as a deployment-quality number, but gate
   on the ratio.
2. **`score recall`** and **`block recall`** (paper §5.2) trending flat/up, comparable to the paper's
   Figure 3.
3. Recall bucketed by **query position, key distance, and generated-trace position**; plus the
   selected-block position histogram vs. dense argmax. **Recency collapse** — selections degenerating
   onto the local window — is the failure that kills 32K, visible at step 200.
4. **`group_divergence` > 0** — if all 8 groups pick the same blocks, per-group capacity is wasted and
   the design should be reconsidered. The paper expects divergence: Appendix A/Fig. 5 reports that
   *"different groups attend to different long-range stripes while sharing the common local and sink
   patterns"* at both layer 1 and layer 18.
5. No NaNs; index score distribution healthy — in **relative** terms (§3.2: serving magnitudes are
   11.3× training's, so no absolute thresholds).

Recall-vs-`k` curve over {8, 16, 32, 64} recorded — it is how the deploy budget gets justified.

---

## 6. Phase 2 — sparse adaptation

Split, because reasoning is more fragile than instruction-following and our token budget is 100–350×
smaller than MSA-CPT's.

### 6.1 Phase 2a — sparse ON, base FROZEN — a **measurement** stage

Loss = `λ · Σ_layers L_KL` only (`L_LM` trains nothing).

- Base weights never move → **the weights are unchanged; the function is not.** Attention goes dense
  → block-sparse, so capability is *not* preserved by construction. What is preserved is the absence
  of drift, which is what makes the measurement clean.
- Collapses eval ladder rows 0 and 2 into one config: Phase-2a weights *are* stock weights.
- Measures the **pure sparsity cost** with zero confound.
- Cheap — no base gradients.
- **The teacher is free**: the sparse main branch already softmaxes each head over exactly the
  selected support, so its attention weights *are* Eq. 9's per-head distributions.

**Expect little learning here, by construction.** Per §6.2 the restricted-support KL gives no
gradient about blocks the indexer failed to select, so after Phase 1 the only thing 2a can improve is
within-support calibration. And the paper has **no precedent for a frozen backbone during the sparse
stage** — MSA-CPT trains 360B sparse tokens with `L_LM` active. So budget 2a as a short measurement
(hundreds of millions of tokens, not 1B), read the sparsity cost off it, and move on.

Answers the central question: *how much does sparsity alone cost this model at 32K?* If that number
is small enough to ship, stop. §12's caveat says not to count on it.

### 6.2 Phase 2b — unfreeze the base (plan for it)

| knob | value |
|---|---|
| loss | `L_LM` on real docs (dominant) + `λ Σ L_KL` + a **smaller** BC self-distillation share |
| base LR | **5e-6 – 1e-5** |
| index LR | 1e-3 → 1e-4 |
| length mixture | 40% short (<4K) / 30% decode-long / 30% prefill-long — see `../qwen3_4b_dsa/data_plan.md` §5 |
| code share | ≥20% of the long half — code was the most BC-fragile *and* most sparsity-sensitive domain in the MiniCPM3 run (−3.0 HumanEval+ from BC alone, `docs/dsa_eval_report.md` §3), and it is also where MSA-CPT's own residual gap is largest (−3.1 HumanEval, Table 2) |

BC is what caused the MiniCPM3 code regression, so make it a minority of the mixture, not the diet.

### 6.3 Gradient wiring (both phases)

| path | trains | mechanism |
|---|---|---|
| `L_LM` → base | base only | block indices `.detach()`-ed (top-k non-differentiable) |
| `L_KL` → index branch | index params only | Eq. 11 `stopgrad(X)` into the index projections; teacher detached |

The paper's B.3 is the evidence for the stopgrad, and it also tells us what mis-setting `λ` does:
without detach, larger KL coefficients caused gradient-norm spikes and LM-loss divergence within a
few hundred steps, plus a gradual short-context regression attributed to self-distillation; **with**
detach the same coefficients were stable. So under Eq. 11 the risk from a too-small `λ` is an
under-trained indexer, not a destabilised backbone.

---

## 7. Inference / serving

**vLLM**, reusing the merged M3 implementation. Nothing needs to be written from scratch — but note
the environment gap: the working container ships **vLLM 0.20.2, which has no `vllm/models/` package
and no `minimax_m3` at all**. The whole of §7 requires vLLM `main` at or past PR #45381 (*"[Model]
Add MiniMax M3 support"*, merged 2026-06-15), which also collides with the MiniCPM3-DSA plugin pinned
to 0.20.2. **Add the vLLM upgrade/build to the schedule** (§10).

### 7.1 What we reuse vs. write

Kernels actually executed per sparse layer on H100 (SM90) — **1 CUDA kernel + 7 Triton kernels**:

| # | kernel | file | role |
|--:|---|---|---|
| 1 | `fused_minimax_m3_qknorm_rope_kv_insert` (CUDA) | `csrc/libtorch_stable/…_kernel.cu` | Gemma QK-norm + partial-NeoX RoPE on q/k/index_q/index_k, scatter-insert of k/v + index_k into the caches. `kHeadDim=128`; bf16 needs SM80+, PDL at `__CUDA_ARCH__ ≥ 900`. Dense layers call it with `skip_index_branch` |
| 2 | `_index_block_score_kernel` | `common/ops/index_topk.py:82` | prefill index scores → `[H_kv, total_q, max_block]` (max already applied) |
| 3 | `_topk_index_kernel` | `…:186` | prefill top-k, bitonic merge, sentinel forcing |
| 4 | `_decode_index_score_kernel` | `…:295` | decode index scores, split-K, cudagraph-stable grid |
| 5 | `_topk_index_partial_kernel` → `_topk_index_merge_kernel` | `…:409, 549` | decode top-k, split-K + merge |
| 6 | `_gqa_sparse_fwd_kernel` | `common/ops/sparse_attn.py:52` | prefill block-sparse GQA; `BLOCK_SIZE_H = next_pow2(G)`, `BLOCK_SIZE_QH` query packing |
| 7 | `_gqa_sparse_decode_kernel` → `_merge_topk_attn_out_kernel` | `…:234, 419` | decode attend, split-K over the 16 blocks + LSE combine |

Layers 0–2 use none of this: `MiniMaxM3Attention` builds a generic `Attention` layer (FlashAttention
on H100) after kernel #1 with the index branch skipped.

| component | source | status on H100 (SM90) |
|---|---|---|
| main block-sparse GQA attention | `MiniMaxM3SparseTritonImpl` (`common/sparse_attention.py` + `common/ops/sparse_attn.py`) | ✅ **default**; the MSA CuTe path is `is_device_capability_family(100)` + `topk ∈ {4,8,16,32}`-gated |
| index scoring + top-k | `common/ops/index_topk.py` (Triton, `@torch.no_grad()`) | ✅ |
| QK-norm + RoPE + KV insert | `csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu` | ✅ `rotary_dim=128` admitted. **Mandatory inside M3's layer — but optional for ours**, see below |
| indexer module + side cache group | `MiniMaxM3Indexer` (`common/indexer.py`) | ✅ fully parameterised (`num_kv_heads`, `topk_blocks`, `num_index_heads`, `index_head_dim`, `init/local_blocks`, `score_type`) |
| MiniMax CuTe kernels (`fmha_sm100`) | [MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA), MIT | ❌ **not vendored in vLLM at all** (`vllm/third_party/fmha_sm100` does not exist); every reference is a function-local import behind the SM100 gate. Even on SM100 it covers **prefill only** — decode falls back to Triton (`nvidia/sparse_attention_msa.py`) |
| **`Qwen3MSAForCausalLM`** | **we write it** | wires the above into Qwen3's decoder |
| **model registration** | **we write it** | `ModelRegistry.register_model(...)` out-of-tree, exactly like `scripts/dsa/vllm_minicpm3_dsa/__init__.py:182-200` — no vLLM fork needed |

`MiniMaxM3SparseImpl.__init__(num_heads, head_size, scale, num_kv_heads, kv_cache_dtype, *,
topk_blocks, sparse_block_size)` is model-agnostic, so **import first, fork only if that fails**. The
backend itself is registered by dotted path (`MINIMAX_M3_SPARSE →
vllm.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend`,
`v1/attention/backends/registry.py:101`) and bound directly by the layer
(`self.attn_backend = MiniMaxM3SparseBackend`), mirroring `vllm/models/deepseek_v4/` — so a
`vllm/models/qwen3_msa/`-shaped port is idiomatic, but out-of-tree registration is the lower-friction
route given we also need the 0.20.2 → main move.

**Two attention classes are required**, not one class with a flag:
`MiniMaxM3SparseAttention.forward()` passes index args, `MiniMaxM3Attention.forward()` omits them
(dense layers), and the fused op's host signature ends in `bool skip_index_branch`
(`kernel.cu:652`).

**The fused op is optional for our port.** It is mandatory inside M3's layer (no branch: the only
call sites are `model.py:370` and `:603`), but it is only a fusion of ops vLLM already ships — RMS
norm, RoPE, and the cache writes. A `Qwen3MSAForCausalLM` may instead use Qwen3's own norm modules
and write the index key into the side cache itself, which removes the Gemma `w − 1` conversion
entirely (§3.2). Cost: a handful of extra launches per layer plus a custom insert for the
`[num_blocks, 128, head_dim]` index-cache layout. Decide by measurement, not by principle.

### 7.2 Serving properties vs. the MiniCPM3 DSA path

- **No global `--enforce-eager` needed** — both backends declare
  `AttentionCGSupport.UNIFORM_BATCH` (`common/sparse_attention.py:205`, `common/indexer.py:225`), so
  uniform decode batches are capturable. **But the sparse attention itself still runs eagerly:**
  `MiniMaxM3SparseAttention._run_attention` is decorated `@eager_break_during_capture`
  (`nvidia/model.py:631-643`) — *"their split-K kernels read per-request metadata and can't be
  captured into a cudagraph"* — and that mechanism is itself gated on
  `VLLM_USE_BREAKABLE_CUDAGRAPH` (`vllm/compilation/breakable_cudagraph.py:53`). This is a real
  improvement on the DSA path's whole-model `enforce_eager`, but it is not "cuda graphs work".
- **`page_size = B_k = 128`** → selected blocks *are* page ids 1:1; no read amplification, no
  masking-within-page (`get_supported_kernel_block_sizes() -> [128]`, "one sparse block per KV page").
- No head padding (MiniCPM3 needed 40→64) and no latent padding to 576.
- Five kernel launches per sparse layer per decode step (§7.1 #4–#7) — a launch-overhead regime worth
  measuring at small batch.

### 7.3 The silent-dense-fallback gate — verify at every length

The MiniCPM3 DSA path silently served **dense** whenever `index_topk` was missing from `config.json`
(`docs/dsa_eval_report.md` §2). **The same bug exists here, and the mechanism is now confirmed:**
`MiniMaxM3DecoderLayer` selects the sparse class via
`layer_id in _sparse_attention_layer_ids(config)`, which returns an **empty set** when
`sparse_attention_config` or `sparse_attention_freq` is missing (`nvidia/model.py:114-123, 676-695`)
→ every layer is built dense; and then every branch of `load_weights` ends in
`if name not in params_dict: continue` (`nvidia/model.py:939, 954, 979`) → **all `index_*` weights
are silently dropped**. Result: a fluent, benchmark-passing, entirely dense Qwen3.

Wire all three checks into the harness *before* any benchmark runs:

1. **Log grep** for both `info_once` lines: `"MiniMax M3 sparse attention selected %s
   (kv_cache_dtype=%s, topk_blocks=%s)"` (`common/sparse_attention.py:469-474`) and `"MiniMax M3
   indexer: selected Triton (no fmha_sm100) [topk_blocks=…]"` (`common/indexer.py:519-525`).
   Absent ⇒ dense.
2. **Two KV-cache groups** must exist — the indexer registers its own side cache
   (`MLAAttentionSpec(num_kv_heads=1, head_size=128)`). One group ⇒ no index branch. This also
   cross-checks the predicted +5.7% bytes/token.
3. **A positive control:** serve the same weights with a randomised indexer
   (`randomize_indexer_ckpt.py`) and confirm the outputs **change**. If they don't, sparsity isn't
   live.

And prefer a **strict loader** (`AutoWeightsLoader`) in `Qwen3MSAForCausalLM` over a silent-skip
loop, so a mis-specified config raises at startup instead of serving dense.

---

## 8. Compute budget (8×H100 80GB)

Memory is not the constraint at 4B: full FSDP finetune ≈ 64 GB total → **~8 GB/GPU**. Activations and
the Phase-1 KL tiles dominate.

**Phase-2 sparse forward cost, calibrated against the existing MiniCPM3 runs.** The training-side
gather is `[b, H_kv, T_q, k·B_k, d]`:

| per token, all layers | MiniCPM3 Phase 2 (4K, k=512) | **Qwen3-4B MSA (32K, k·B_k=2048)** |
|---|--:|--:|
| KV gathered per token per layer | 6.55 MB (40 heads × 96/64) | **8.4 MB** (8 kv heads × 128, K+V) |
| sparse layers | 62 | **33** |
| **per token, whole model** | 406 MB | **277 MB** (0.68×) |
| tokens/GPU/step | 8 × 4096 = 32,768 | 1 × 32,768 = **32,768** |
| **gather traffic/GPU/step** | ~13.3 TB | **~9.1 TB** |

At identical tokens/GPU/step the MSA path moves ~30% *less* data than the MiniCPM3 Phase-2 runs,
because GQA has 8 KV heads where unabsorbed MLA materialized 40, and only 33 layers are sparse. So
Phase 2 at 32K should land at a comparable step time using the existing torch gather — **but the
table counts only the main-branch gather.** Not counted:

- the index branch's dense `O(N²)` score + block-max per sparse layer (1.10 TFLOP/layer at 32K,
  `[1, 8, 512, 32768]` tiles — ~1 TB/step of tile traffic across 33 layers, small next to 9 TB but
  not free);
- the gather's **peak activation**: `[1, 8, 512, 2048, 128]` bf16 ≈ 2.1 GB per tensor, ~4.2 GB for
  K+V per tile — tractable only with `kl_block_size` tiling + `kl_checkpoint`.

**The MiniCPM3 step-time anchor needs a citation.** The "≈38 s/step measured" figure this comparison
leans on is not recorded in any doc in this repo (the only in-repo step time is
`docs/dsa_kl_checkpoint.md:137`, "~265 s/step for the 32K run"). Pin it to a run id / log path per
our own logging discipline, or replace it with a fresh 50-step timing run.

| phase | tokens | rough wall-clock |
|---|--:|---|
| 1 (dense warm-up, base frozen) | 0.5–1B | ~½–1 day |
| 2a (sparse, base frozen — measurement) | 0.2–0.5B | ~¼–½ day |
| 2b (sparse, base unfrozen) | 1–2B+ | ~1–2 days |

**FLOP-derived, not measured.** Pin with a 50-step timing run before scheduling. Note MiniMax's
KV-outer sparse-attention forward and sparse-KL backward *training* kernels are described in §4 of
the paper but **not released** — only the inference kernel is (p.1).

---

## 9. Evaluation

`../qwen3_4b_dsa/eval_plan.md` **carries over** — benchmarks, the de-confounding ladder, the
thinking-mode protocol, and the acceptance gates are architecture-agnostic. Substitutions:

| eval-plan concept | MSA equivalent |
|---|---|
| ladder row 2, "`top_k ≥ L`" dense-equivalent control | **Phase-2a (stock weights) is the primary control.** `k ≥ ceil(N/B_k)` = 256 blocks at 32K is the backup and may not be servable (§4.2 #1) |
| rows 4/5 random-indexer ablation | randomize `{W_q_idx, W_k_idx, index_q_norm, index_k_norm}` (`randomize_indexer_ckpt.py`) — doubles as the §7.3 positive control |
| `topk_recall` / `topk_overlap` | **`captured_mass`** (gate) + paper `score recall` / `block recall` (§5.2) |
| `index_topk` config gate | the three §7.3 checks |

Spine: **RULER at 4K/8K/16K/32K** (effective context length is the headline metric), NIAH depth×length,
**NoLiMa** (the most diagnostic test of a learned selector), HELMET summarization + ICL, reasoning
(AIME/GPQA/LiveCodeBench v6), and the full short-context regression suite.

**Acceptance gate, recalibrated:** `Δ(N)` **non-increasing in `N`** is the load-bearing criterion — a
Δ that grows with length is the signature that matters. On magnitude, MSA-CPT at 400B+ tokens still
gave −2.6 on RULER-8K and −3.1 on HumanEval (§2), so a uniform `Δ ≤ 2 pts` bar is stricter than the
architecture's own published conversion result. Gate on: long-context Δ ≤ ~2 and flat-or-shrinking
in `N`; short-context and code Δ reported and explicitly accepted or rejected as a product call, not
silently failed.

Data plan: `../qwen3_4b_dsa/data_plan.md`, unchanged.

**Heed MiniMax's own M2 retrospective** ([Why did M2 end up as a full attention model?](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model),
and LMSYS's [*"No Free Lunch"*](https://www.lmsys.org/blog/2025-11-04-miminmax-m2/)): with hybrid
attention, standard benchmarks looked fine while **complex multi-hop reasoning showed clear deficits
at scale**, and after SFT the gap concentrated **above 32K context** — precisely our target
boundary. Scope caveat: that retrospective concerns *content-agnostic* hybrid/linear/SWA attention,
not a learned block-sparse selector — MSA is MiniMax's own answer to it — so it is suggestive rather
than directly transferable. Either way, keep RULER variable-tracking and NoLiMa as load-bearing
gates, not nice-to-haves.

---

## 10. Open items and risks

| # | item | severity |
|--:|---|---|
| 1 | **Index parity test** vs the three vLLM index kernels, incl. the semantics checklist (§4.2 #2) | must-do before training |
| 2 | ~~Oracle probe~~ — **DONE 2026-07-28**, see §12 (with Finding 1 corrected) | done |
| 3 | **Fused op at `rotary_dim = 128`**, numerically, subsuming the architecture-match assertion (§4.2 #3) | must-do, cheap |
| 4 | ~~**`λ`** not given in §3.2 — check the appendix~~ — **CLOSED: it is nowhere in the paper.** `λ` appears only symbolically (Alg. 1, Eqs. 18–19). Pick it empirically; B.3 says the failure mode under Eq. 11 is an under-trained indexer, not divergence | closed |
| 5 | **vLLM upgrade 0.20.2 → main ≥ #45381**, alongside the MiniCPM3-DSA plugin | must-do, blocks all serving |
| 6 | **Gemma-norm decision** (§3.2/§7.1): reuse the fused op + `w − 1` export, or write our own norm/RoPE path | decide in Phase 0 |
| 7 | **Silent dense fallback** (§7.3) — confirmed mechanism, silent by construction | must-gate before any eval |
| 8 | **Decode head-axis pad at `G = 4`**: `BLOCK_SIZE_H = max(16, next_pow2(G))`, so 4 real heads occupy 16 slots (`common/ops/sparse_attn.py:227-229`). M3 (`G=16`) never sees this | measure |
| 9 | **`k = 256` may not compile** in the Triton top-k (§4.2 #1) — affects only the served dense-equivalence control | verify, cheap |
| 10 | **Phase 2a will likely underperform**, making 2b necessary (§6.1, §12) | design expectation, not a risk |
| 11 | **Eval cost with thinking mode** — 10–50× output tokens × multi-sample × ladder rows × lengths, plus the >32K trace-length issue (§1.1 #6) | schedule risk |
| 12 | **No instruction-shaped prefill-long training data** (`data_plan.md` §11) — we train document-shaped and gate on RULER/HELMET | methodology risk; Phase-1 recall probe is the early warning |
| 13 | **MiniCPM3 step-time anchor uncited** (§8) | fix or re-measure |

---

## 11. Working discipline

This plan went through several reversals (KL support block↔token, forced blocks additional↔inside,
backend generic↔model-specific) — every one caused by inferring from notation or signatures instead
of reading the source. The 2026-07-28 audit (§13) found ten more of exactly that kind, including two
in our *own* measured data. Going forward:

- **Verify against the primary source before writing a claim into a doc**, not after. Paper claims
  come from the PDF (WebFetch returns *summaries*; for equations demand verbatim quotes). Code claims
  come from the file, at a pinned revision.
- **Keep the quoted-vs-inferred split explicit** — see [kl_loss.md](kl_loss.md) §7.
- **Negative claims need a real read.** "The summarizer didn't find it" is not evidence of absence —
  and neither is "the plan says it's not asserted."
- **Re-read your own artifacts before quoting them.** §12's layer labels were wrong for a week while
  the JSON sat on disk.

---

## 12. Measured: block-oracle probe on stock Qwen3-4B-Thinking-2507 (2026-07-28)

**Run:** `bash scripts/msa/run_oracle_8gpu.sh` — 8 GPUs, 62 docs at 32K, 512 sampled query positions
per doc, ~90 s wall clock, peak 17 GiB/GPU.
**Artifacts:** `/cb/ml-eng/aarti/msa/oracle/qwen3_4b_thinking_32k/` (merged JSON + per-shard + logs);
the merge is an exact weighted merge of raw sums/counts, and all eight swept configs are in the JSON.
**Corpus:** `/cb/ml-eng/aarti/msa/data/long_docs_qwen3_32768.jsonl` (+ MANIFEST) — built by
`scripts/msa/dump_long_docs.py` from `openbmb/InfLLM-V2-data-5B` @ `deeb03b5bcea`, re-filtered on the
**Qwen3** tokenizer.
**Sampling floor:** `min_pos = 2 · min(ks) · max(B_k) = 2048`, with a per-`k` validity mask
`vis_blocks > k`, so for `k = 16` the effective floor is position ≥ 2048 (not 4096 — `kl_loss.md` §8
overstated it).

| config | tokens | oracle_block | min layer | oracle_token | granularity | unreachable |
|---|--:|--:|--:|--:|--:|--:|
| Bk128_k8 | 1024 | 0.8230 | 0.6296 | 0.9404 | 0.1174 | 0.0596 |
| **Bk128_k16** | **2048** | **0.8879** | **0.7403** | 0.9657 | 0.0779 | 0.0343 |
| Bk128_k32 | 4096 | 0.9345 | 0.8171 | 0.9828 | 0.0484 | 0.0172 |
| Bk128_k64 | 8192 | 0.9687 | 0.8910 | 0.9935 | 0.0248 | 0.0065 |
| Bk64_k16 | 1024 | 0.8532 | 0.6692 | 0.9404 | 0.0872 | 0.0596 |
| Bk64_k32 | 2048 | 0.9066 | 0.7520 | 0.9657 | 0.0592 | 0.0343 |

**Verdict: proceed.** Excluding the dense prefix, **sparse layers 3–35 average `oracle_block = 0.8988`**
with only 2 of 33 below 0.80 (layers 3 and 4) — the top of the "workable" band.

**Bound tightness:** the oracle takes the unconstrained top-k by mass, whereas MSA spends one of its
16 slots on the forced local block. The local block is usually top-mass, so the gap is small, but
`oracle_block` is a slightly **loose** upper bound for the deployed configuration.

### Findings

1. **The worst layers are 1, 2, 3** — 0.7403 / 0.7441 / 0.7669 — then 4 (0.7928) and only then
   **0 (0.8184)**. *(Corrected: an earlier version of this section attributed these values to layers
   0, 1, 2 and claimed the probe "independently confirms" M3's `[0]*3 + [1]*57`. It does not.* Under
   `[0]*3 + [1]*33` layer 3 — third-worst overall — is **sparse**, while layer 0 is kept dense
   despite out-scoring both 3 and 4. A data-driven dense set would be **{1, 2, 3, 4}**, and vLLM
   accepts any per-layer list, not just a prefix
   (`{i for i, f in enumerate(freq) if f != 0}`, `nvidia/model.py:114-123`).) The general trend —
   early layers diffuse, deeper layers block-concentrated — does hold, and is consistent with M3
   keeping a dense prefix; it just isn't a layer-for-layer confirmation.
2. **Block-concentration rises with depth but not monotonically.** Layers 28–35 are
   0.9514 / 0.9278 / 0.9667 / 0.9475 / 0.9503 / 0.9129 / 0.9180 / 0.9518 — i.e. 0.913–0.967 with
   ±0.03 layer-to-layer variation, not "0.950–0.967 monotonic".
3. **Granularity is the minor term; budget is the major one.** At a fixed 2048-token budget, halving
   the block size buys +1.9 pts (`Bk128_k16` 0.8879 → `Bk64_k32` 0.9066). So `B_k = 128` being
   kernel-locked costs little — consistent with the paper's own C.1 ablation, where PPL is flat
   across `B_k ∈ {32, 64, 128}` and RULER moves ≤1.5 pts (RULER-8K *improves* with 128, RULER-32K
   drops 66.1 → 64.6). Raising `k` is what moves the ceiling: 0.823 → 0.888 → 0.935 → 0.969 for
   k = 8/16/32/64.
4. **The absolute per-layer gate in §5 was unachievable** and has been replaced with an
   oracle-relative one. Layers 3 and 4 cap at 0.767 / 0.793 with a *perfect* selector.

### Adjustments

**Adopted now:**

- **Phase-1 gate is oracle-relative** (§5 item 1) — the absolute 0.80 floor was unachievable.
- **`k = 32` is the documented fallback** operating point (0.9345 ceiling, 12.5% ratio, still inside
  the SM100 CuTe path's `{4,8,16,32}`) if `k = 16` underperforms after Phase 1.

**Recommended, pending Phase-1 evidence — config unchanged for now:**

- **Extending the dense prefix from 3 to 5 layers** (`sparse_attention_freq = [0]*5 + [1]*31`) drops
  the two worst sparse layers (3 at 0.767, 4 at 0.793) and lifts the sparse floor to 0.8222
  (layer 17), with the sparse mean rising 0.8988 → 0.9065. Costs 2 more dense layers: attention
  FLOPs 99.0 → 112.2 TFLOP, prefill FLOP ratio **1.60× → 1.55×**, index params 97M → 91M, side cache
  8.4 → 7.9 KB/token, KL denominator 33 → 31.
- **Or, better motivated by the data: dense = {1, 2, 3, 4}, sparse = {0} ∪ {5…35}** — same count as
  `[0]*5`, but it spends the dense budget on the four actually-diffuse layers instead of on layer 0.
  Costs nothing extra; requires only that our config plumbing pass an arbitrary list, which vLLM
  already supports.
  **Neither adopted yet**, deliberately: a low *oracle* on layers 3–4 is not the same as those layers
  failing to train, the oracle-relative gate accommodates a low ceiling, and Phase 2b may lift it.
  Decide from Phase-1 per-layer `captured_mass / oracle_block`.

### The caveat that matters most

This oracle measures **stock, densely-trained** attention. MSA's M3 was trained *natively* sparse, so
its attention adapted to be block-concentrated — its own oracle would be far higher at the same
budget (it runs `k = 16` at **1M** context, a 0.2% ratio, and still reports GQA parity). So 0.8988 is
**not a hard post-adaptation ceiling**: Phase 2b can raise the ceiling itself by reshaping attention
to be block-friendly.

That materially weakens the Phase-2a optimism in §6.1. Phase 2a (base frozen) is bounded by *this*
oracle; if it lands near 0.90 × 0.8988 the sparsity cost may be unacceptable and **Phase 2b is
necessary rather than optional**. The paper agrees by omission: MSA-CPT never freezes the backbone
during its 360B-token sparse stage.

### Known probe limitation

`_build_batches` is called without shard arguments (`probe_block_oracle.py:280`), so its documented
"shard before tokenize" optimisation never runs — every rank tokenizes the whole corpus and then
strides (`:284`). Results are unaffected (the strided slices are disjoint and balanced); only startup
CPU is wasted. Fix the call or the docstring before the next run.

---

## 13. Corrections applied in the 2026-07-28 audit

| # | was | now | source |
|--:|---|---|---|
| 1 | §12: "three worst layers are 0, 1, 2 … independent confirmation of M3's design" | worst are **1, 2, 3**; the claim is withdrawn | merged oracle JSON |
| 2 | §12: layers 28–35 "0.950–0.967", monotonic | 0.913–0.967, non-monotonic | merged oracle JSON |
| 3 | §3.2: "`RMSNorm(128)`, per-head" | **Gemma-style `(1+w)`, one shared `[128]` gain**, bf16, zero-init; train in Gemma form; `w−1` for Qwen3's q/k if the fused op is reused | `kernel.cu:17,135-145,711-718`; `nvidia/model.py:114-141`; M3 `use_gemma_norm: true` |
| 4 | §3.1: "`--block-size 128` MANDATORY; default 16 misaligns" | auto-selected; a user-specified 16 **raises** | `backend.py:194-203`; `interface.py:628-641`; `worker/utils.py:296-320` |
| 5 | §3.1: top-k buffer `[num_kv_heads, total_q, topk]`; equality "NOT asserted" | token-major `[tokens, H, topk]`; equality **is** asserted | `nvidia/model.py:790-808`; `linear.py:1462-1465` |
| 6 | §7.2: "CUDA graphs work … fixes the `enforce_eager` hole" | no *global* enforce-eager, but the attend is an **eager segment** | `nvidia/model.py:631-643`; `breakable_cudagraph.py:53` |
| 7 | §3.1: `indexer_kv_dtype` as a model-config key, fp8 "avoid" | vLLM **engine** option; fp8 **raises** off SM100 | `config/attention.py:13,67`; `common/indexer.py:513-518` |
| 8 | §1/§10: "Qwen3-4B-Thinking-2507 not on disk" | on disk; §12 ran on it | probe JSON `args.model` |
| 9 | §2: "MSA explicitly rejects MLA to avoid latent-KV overhead" | not in the paper; replaced with the actual Related-Work sentence | paper p.13 |
| 10 | §2: "no evidence that adaptation works"; §3.3 "decode ~16× fewer reads" | **MSA-CPT** = 400B + 140B tokens with residual −2.6/−3.1 gaps; decode reads **8×** at 32K (~1.4× end-to-end at batch 1) | paper §5, Tables 2–3; side-cache arithmetic |

Also added: vLLM version gap (§7), the confirmed silent-dense-fallback mechanism and its three gates
(§7.3), the H100 kernel inventory (§7.1), why the kernels can't be trained through (§4.1), the
`k = 256` compile risk (§4.2), Phase-2a reframed as measurement (§6.1), paper appendix evidence for
warm-up and KL-only (§5), the `G = 2` dev-ladder gap (§1), the >32K trace-length tension (§1.1),
`group_divergence`'s paper support (§5), and the uncited step-time anchor (§8).
