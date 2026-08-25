# Qwen3-4B DSA — vLLM serving + evaluation plan (max kernel reuse)

**Question this answers:** how do we get a token-granular DSA Qwen3-4B-Thinking-2507 checkpoint onto
vLLM and evaluated, writing as little kernel code as possible?

**Answer in one line:** reuse **every** kernel — DeepSeek-V3.2's fp8 lightning-indexer stack for
selection and FA3 varlen-with-a-page-size-1-block-table for the sparse attend — and write only
~350 lines of non-MLA *plumbing*: one attention backend, one Qwen3 model shell, one serving-dir
builder. No CUDA, no Triton, no DeepGEMM authored by us.

**Status (2026-08-20).** Nothing is blocked on training, because **training has not started** — there
is no `qwen3*dsa*` run dir under `/cb/ml-eng/aarti/dsa`. That is the plan's biggest lever: the
Phase-0 feasibility spike and ~80% of the serving work need no checkpoint at all, and two training-side
geometry choices (§3) are still free. Do serving *now*, in parallel with (or ahead of) Phase 1.

**Verification discipline.** Every vLLM claim below was read from the environment we will actually
serve on — `.devlibs/vllm026` = **vLLM 0.26.0 / torch 2.11.0+cu130 / transformers 5.14.1** — and is
cited `file:line` relative to that site-packages `vllm/`. Claims not verified are marked
**[UNVERIFIED]** and each has an assigned check in §4. The container's vLLM **0.20.2 cannot be used**:
it predates `models/deepseek_v32/`, `model_executor/layers/sparse_attn_indexer.py` and every
`backends/mla/*sparse*` file.

Companion docs: [plan_v2.md](plan_v2.md) §5 (the feasibility study this executes),
[eval_plan.md](eval_plan.md) (benchmarks, ladder, thinking-mode protocol — reused wholesale),
[../qwen3_4b_msa/serving_plan.md](../qwen3_4b_msa/serving_plan.md) (the proven end-to-end template).

---

## 1. Why this is mostly plumbing

vLLM's DSA implementation is MLA-gated in *Python*, not in kernels. The gate:
`FlashAttnMLASparseBackend.is_mla() -> True` and the sparse attend reads an MLA-shaped cache
(`v1/attention/backends/mla/flashattn_mla_sparse.py:62`, `:238-244`). But the attend itself is
**one ordinary FA3 call** (`:246-259`):

```python
out = flash_attn_varlen_func(
    q=q_rope, k=k_cache, v=v_cache, q_v=q_nope,        # q_v = MLA's asymmetric K64/V512 path
    max_seqlen_q=1, cu_seqlens_q=cu_seqlens_q,          # every query token is its own sequence
    max_seqlen_k=topk_indices.shape[1], seqused_k=valid_counts,
    block_table=topk_indices,                           # page-size-1 view => "pages" are tokens
    softmax_scale=self.scale, causal=True, fa_version=3)
```

Token-level top-k is expressed as a degenerate paged block table. There is no bespoke gather kernel to
port. For GQA we drop `q_v` (qk dim == v dim, the *simpler* symmetric path) and view the standard KV
cache as `(-1, 1, H_kv, head_dim)` instead of `(-1, 1, 1, d)`.

And the precedent that a **non-MLA sparse backend can register itself at all** is already in the same
wheel: `models/minimax_m3/common/sparse_attention.py:77` — `MiniMaxM3SparseBackend`, GQA, `is_sparse()
-> True`, its own `get_kv_cache_shape`, bound directly by the attention module rather than by vLLM's
MLA selector. We serve MSA through it today. So our backend is a **cross** of two files that both
ship: M3's registration/KV-shape/metadata skeleton × the MLA-sparse file's FA3 call.

### 1.1 The reuse ledger

| Component | Source | Reuse |
|---|---|---|
| fp8 paged indexer logits + paged top-k | `model_executor/layers/sparse_attn_indexer.py:296` (`sparse_attn_indexer` op), `:706` (`SparseAttnIndexer`) | **verbatim** |
| radix top-k / DCP merge kernels | `model_executor/kernels/attention/dsa/dcp_indexer_cutedsl.py` | **verbatim** (via the op) |
| fp8 index-K side cache + its KV-cache spec | `model_executor/models/deepseek_v2.py:614` `DeepseekV32IndexerCache` (`MLAAttentionSpec(num_kv_heads=1)` — a naming artifact, it is a 1-head cache) | **verbatim** |
| per-request → global token slot conversion | `v1/attention/backends/mla/sparse_utils.py` `triton_convert_req_index_to_global_index` (`BLOCK_SIZE` is a `tl.constexpr` arg — page-size generic) | **verbatim** |
| sparse attend | `vllm_flash_attn/flash_attn_interface.py:176` `flash_attn_varlen_func` | **verbatim**, called from our backend |
| our serving indexer module (projections, RoPE, Hadamard, fp8, head/dim padding) | `scripts/dsa/vllm_minicpm3_dsa/indexer.py` — 478 lines, already parity-tested (`tests/dsa/test_minicpm3_dsa_indexer_parity.py`) | **adapt**, ~40 lines changed (§2.1) |
| Qwen3 model shell, `topk_indices_buffer` allocation, plugin registration, `_pluginboot` sitecustomize, serve script, serving-dir builder, eval-root cloning, 13-bench queue | `scripts/msa/vllm_qwen3_msa/model.py`, `scripts/msa/serving/*`, `scripts/msa/build_msa_serving_dir.py`, `scripts/msa/setup_eval_root.sh` | **adapt** — same model, same env, same harness |
| **non-MLA token-sparse attention backend** | — | **write (~350 lines)** |

Nothing in the first six rows is a kernel we author. `Indexer` itself
(`deepseek_v2.py:643`) is *not* in the reuse list — see §2.1.

---

## 2. What we write

`scripts/dsa/vllm_qwen3_dsa/` — mirroring `scripts/msa/vllm_qwen3_msa/`, which is the in-repo pattern
that works.

| file | lines | contents |
|---|--:|---|
| `__init__.py` | ~20 | lazy `ModelRegistry.register_model("Qwen3DSAForCausalLM", "…model:Qwen3DSAForCausalLM")` |
| `indexer.py` | ~300 | adapted from the MiniCPM3 plugin (§2.1) |
| `sparse_attention.py` | ~350 | the new backend: `Qwen3SparseBackend` / `Impl` / `MetadataBuilder` (§2.2) |
| `model.py` | ~250 | `Qwen3DSAAttention` / `DecoderLayer` / `Model` / `ForCausalLM`; buffer allocation; `DSA_SPARSE=0` bring-up flag |
| `../build_qwen3_dsa_serving_dir.py` | ~150 | consolidate FSDP shards → HF dir, **write `index_topk`** |
| `../serving/serve_qwen3_dsa.sh` | ~80 | copy of `scripts/msa/serving/serve_msa.sh`, `--block-size 64` |

### 2.1 The indexer: reuse our own port, not vLLM's `Indexer` class

vLLM's `Indexer` (`deepseek_v2.py:643`) is tempting — `qr` is its only MLA coupling and our `wq`
projects straight from hidden states, so `q_lora_rank = hidden_size` would fit. Three verified reasons
we subclass/adapt instead of instantiating it:

1. **It has no `q_norm`.** Our indexer does (`verl/models/transformers/qwen3_dsa_indexer.py:349`), and
   it is load-bearing: GQA has no `q_a_layernorm` upstream to supply the normalization MLA gets for
   free (plan_v2 §2.1). A norm cannot be folded into a GEMM.
2. **Its fp8 cache sizing breaks below head_dim 128.** `deepseek_v2.py:696` sizes the side cache as
   `head_dim + head_dim // quant_block_size * 4` with `quant_block_size = 128`. At our `d_idx = 64`
   that is `64 + 0` — **zero bytes of scale storage**. So the padded construction is mandatory, and our
   MiniCPM3 plugin already does exactly it: `padded_head_dim = 128`, `padded_n_heads = 32`, and it
   hands `SparseAttnIndexer` the *padded* head_dim (`vllm_minicpm3_dsa/indexer.py:243-297`).
3. **`use_fused_indexer_q` is unavailable at `d_idx = 64` regardless** — it requires
   `head_dim == 128 and rope_dim == 64` (`deepseek_v2.py:723-729`, and
   `sparse_attn_indexer.py:210-223` asserts `q.shape[-1] == 128`). We are on the unfused path either
   way, which is precisely what makes inserting `q_norm` free.

Padding is lossless and already parity-tested: norms/RoPE/Hadamard run at the **real** dims, zeros are
appended immediately before fp8 quantization (dot unchanged; row amax unchanged, so the UE8M0 scale is
unchanged), and padded heads carry zero gate weights so `w·ReLU(·) = 0`
(`vllm_minicpm3_dsa/indexer.py:364-380`). Cost: 132 vs 68 B/token/layer of index cache — 152 MB at
32K×36 layers. Irrelevant.

**Deltas from the MiniCPM3 plugin's indexer:** query source `qr → hidden_states`; add `q_norm`
(RMSNorm, real dim) after `wq`; `rope_head_dim 32 → 64`; `n_heads 16 / head_dim 64` unchanged; keep
Hadamard + UE8M0 exactly as trained (`Qwen3DSAConfig.fp8_ue8m0`, `rotate_activation` —
`qwen3_dsa_indexer.py:119-125`, and memory `dsa-fp8-ue8m0-fix`: this is worth ~2% selection overlap).

### 2.2 The sparse attention backend

Structure copied from `MiniMaxM3SparseBackend` (registration, `get_kv_cache_shape`,
`get_kv_cache_stride_order`, metadata builder, `_cudagraph_support`), body copied from
`FlashAttnMLASparseImpl.forward_mqa` minus MLA:

- `get_supported_kernel_block_sizes() -> [64]` — the compatibility keystone: FA3 allocation wants a
  multiple of 16, the MLA-sparse backend proves 64, and the indexer's paged-logits kernel wants
  exactly 64.
- `get_supported_head_sizes() -> [128]`, `supported_dtypes = [bf16]`,
  `supported_kv_cache_dtypes = ["auto","bfloat16"]` — pin bf16 KV for v1; fp8 scales under a page-1
  view are untested.
- `get_kv_cache_shape -> (num_blocks, block_size, num_kv_heads, 2*head_size)` (standard GQA packed
  K/V), viewed as `(-1, 1, num_kv_heads, head_size)` at attend time.
- `is_sparse() -> True`, `is_mla() -> False`, `_cudagraph_support = UNIFORM_BATCH` (both templates
  declare it: `flashattn_mla_sparse.py:144`, `minimax_m3/common/sparse_attention.py:205`) — so decode
  cudagraphs are reachable, unlike the MSA bring-up which shipped `--enforce-eager`.
- `forward`: `triton_convert_req_index_to_global_index(...)` → `flash_attn_varlen_func(...)` with
  `block_table=topk_indices`, `max_seqlen_q=1`, `seqused_k=valid_counts`, `causal=True`, no `q_v`.
- **`index_topk` gate: raise, never fall back.** vLLM's own gate is a *soft* one — a missing
  `index_topk` makes `supports_combination` return a string and the engine silently picks a dense
  backend (`flashattn_mla_sparse.py:96-100`; memory `dsa-serving-index-topk-gate`, which cost us a
  whole misread MiniCPM3 eval). Our backend is bound directly by the module, so we assert instead.

**[UNVERIFIED, the one real unknown]** `flash_attn_varlen_func` does not expose `pack_gqa`
(`flash_attn_interface.py:176-215`; it is hardcoded `None` internally → FA3's own heuristic). Keye-VL
sets it explicitly on their GQA decode path. If FA3's heuristic mis-chooses at `H_q/H_kv = 4` with a
page-1 table, the fix is a wrapper or a bumped wheel, not a redesign. Measured in P0.

---

## 3. Two training-side decisions that are still free (decide before Phase 1 starts)

Both trade training cost against serve-time kernel nativeness. They are cheap now and expensive after
2B tokens of warm-up.

**D1 — `d_idx = 64` (plan_v2's choice, Keye's value) vs `128` (DeepSeek's).**
At 128, `use_fused_indexer_q` turns on, padding disappears, and the fp8 cache sizing is native — the
literal maximum-kernel-reuse geometry. Cost: 2× indexer FLOPs *and* 2× the Phase-1/2 KL teacher's
index-side cost, on the hot path of the entire training budget.
**Recommendation: keep 64.** The padding is proven lossless in-repo, and the fusion it forfeits is one
small pre-attend kernel while the sparse attend and the paged logits dominate. Revisit only if P0
shows the unfused indexer is a measurable fraction of decode.

**D2 — keep `q_norm`.** It has no serving cost on the unfused path (§2.1.3) and plan_v2 §2.4 argues it
is needed for trainability. Recorded here only because it is the reason we cannot instantiate vLLM's
`Indexer` verbatim, which is otherwise the single largest reuse opportunity we are declining.

Enforce both in `Qwen3DSAConfig.__post_init__` under the existing `serving_compat` flag
(`qwen3_dsa_indexer.py:154`) so a config that cannot be served fails at construction — not after a
4-day run.

---

## 4. Order of work

P0–P2 need **no checkpoint**. P3 needs Phase-1 weights (any step). Only P5–P6 need a converged model.

### P0 — the FA3 page-size-1 GQA spike (gates everything; ~1 day)

`tests/dsa/probe_fa3_sparse_gqa.py`, standalone, random weights, no vLLM engine:
allocate a paged GQA KV cache (`block_size=64`, `H_kv=8`, `d=128`, bf16), draw random top-k token
slots, and

1. **correctness** — compare against a reference gather + SDPA. Bit-close is the gate.
2. **decode shape** — 1 query/seq, batch 1–64, `T ∈ {4K, 8K, 32K}`, `k=2048`. Expected to be fine:
   Keye ships this on GQA in production.
3. **prefill shape** — one varlen call with 32,768 length-1 sequences and a `[32768, 2048]` block
   table. **This is the unmeasured regime**; Keye went to a custom dedup kernel instead, which is a
   hint. Timing here is *information, not a gate* — a slow prefill sends us to §6's fallbacks and
   changes nothing about training.
4. `pack_gqa` — check whether FA3's heuristic is sane at `H_q/H_kv = 4`.

**Do this first even if serving slips**, because a hard failure here is the one outcome that would
change the training plan (block-granular MSA is the fallback, and we already have it working).

#### P0 RESULT — 2026-08-20, H100 SM90, `tests/dsa/probe_fa3_sparse_gqa.py`

**Correctness: PASS.** All six cases match a gather+SDPA reference to ≤3.2e-3 relative (bf16
reduction noise): decode 1/8/32 reqs, prefill-shaped 512 and 2048 queries, a mixed 4×64 batch, and
the `top_k > seq_len` dense-equivalence case. **The page-size-1 block table works on GQA
(`H_q/H_kv = 32/8`, `d=128`, page 64) with no `q_v` and no new kernel.** The route is viable.

**Decode: sparse wins, and the win grows with length — this is the whole point.**

| batch | dense paged decode | sparse `k=2048` | speedup |
|---|--:|--:|--:|
| 1 req @ 8K / 32K | 0.038 / 0.059 ms | 0.071 / 0.069 ms | 0.5× / 0.9× |
| 8 reqs @ 8K / 32K | 0.098 / 0.355 ms | 0.071 / 0.072 ms | 1.4× / **4.9×** |
| 32 reqs @ 8K / 32K | 0.356 / 1.373 ms | 0.106 / 0.106 ms | 3.4× / **13×** |

Sparse decode is **flat in sequence length** (8K == 32K to 3 significant figures) while dense scales
linearly and saturates HBM at ~3.1 TB/s. At batch 1 sparse is launch-bound and slightly *loses* —
irrelevant, since eval runs at high concurrency.

**Prefill: sparse is 3.5–19× SLOWER than dense at our target lengths.** Per layer, one request:

| length | dense FA3 causal | sparse `k=2048` | ratio |
|---|--:|--:|--:|
| 4K | 0.223 ms | 4.13 ms | 18.5× slower |
| 8K | 0.784 ms | 9.44 ms | 12.0× slower |
| 32K | 14.4 ms | 51.0 ms | **3.5× slower** |

Sparse prefill does ~8× fewer FLOPs at 32K yet takes 3.5× longer: 21 TFLOP/s vs dense's ~680
TFLOP/s. The crossover is around 90–110K tokens. **Sorting each row's slots ascending does not help
at all** (53.2 vs 51.0 ms at 32K; 4.12 vs 4.13 at 4K) — so this is *not* a coalescing problem, it is
the cost of restructuring a prefill into `T` independent length-1 sequences each carrying a
2048-entry block table. Confirmed as a side effect: sorted and unsorted agree to 4.9e-4, i.e. the
kernel treats the selection as a set, so any future reordering optimization is numerically free.

**Verdict: ship it, and expect end-to-end to still win.** For a 32K prompt + 12K thinking trace at
batch 32, ×36 layers: sparse ≈ 59 s prefill + 46 s decode; dense ≈ 17 s prefill + 588 s decode. The
prefill regression is real and must be reported in the scorecard, but decode dominates eval
wall-clock by an order of magnitude. Two consequences for the rest of the plan:
- **P4's throughput gate becomes a decode-throughput gate at realistic concurrency**, not a
  prefill one. A prefill-only benchmark would read as a failure and would be measuring the wrong
  thing.
- Do **not** "fix" prefill by rounding the token selection up to 64-token pages to use the M3
  block-sparse kernels. That attends to a superset of what was trained and is a different model.
  If prefill ever has to improve, the honest levers are FA4's `mask_mod` (Blackwell) or a dedup
  kernel of the kind Keye wrote.

### P1 — indexer parity, standalone (~1 day, no engine)
`tests/dsa/test_qwen3_dsa_indexer_parity.py`: training `Qwen3DSAIndexer` vs the plugin indexer on the
same random weights — scores to fp8 noise, and **selected sets identical**. Includes the 16×64 → 32×128
pad invariance (dot and UE8M0 scale unchanged) and a **determinism** check: repeated runs on the same
input must give bit-identical top-k. `torch.topk` ties are a third train/serve drift mechanism after
`fp8_ue8m0` and the `index_topk` gate (plan_v2 §5.4.1); with k=2048 of 32K there are near-ties in every
row.

### P2 — plugin loads and serves dense (~1 day)
Build a serving dir from **stock Qwen3-4B-Thinking weights + randomly initialised indexer**, serve with
`DSA_SPARSE=0`. Gate: greedy output identical to stock Qwen3 on a fixed prompt set, i.e. the indexer is
attached and inert. This is `tests/dsa/test_stage0_plugin_load.py` re-pointed, and it is what makes
every later failure attributable.

### P3 — sparse path parity vs the training forward (~2 days, needs Phase-1 weights)
1. **`top_k ≥ T` ⇒ dense.** Serve with `index_topk` larger than the prompt; output must match P2's
   dense run. This is the faithfulness control for the *kernel*, and it is the single most valuable
   test in the plan — it isolates backend bugs from model quality with no reference implementation
   needed.
2. **Teacher-forced prefill parity** against the HF training forward (the training module is
   prefill-only — `_sparse_attn_and_kl` recomputes index keys from current hidden states, so there is
   no HF decode oracle; same limitation the MSA plan hit and accepted).
3. **Selection overlap ≥ 0.99** train vs serve at `k=2048`, per layer, at 4K/8K/32K.
4. **Decode parity** — greedy agreement over ≥256 tokens vs prefill-teacher-forced logits.
5. **Prove sparsity is live**: assert the backend name in the serve log and that
   `valid_counts.max() == index_topk`; then the negative control — a **randomized indexer**
   (`scripts/dsa/randomize_indexer_ckpt.py`) must *collapse*. If it does not, the model is silently
   dense and every score below is meaningless.

### P4 — throughput (~1 day)
Prefill and decode tok/s at 4K/8K/32K vs the dense baseline and vs MSA k16, with cudagraphs on (§2.2)
and off. This is where P0's prefill number turns into a serving decision, and it is the DSA-vs-MSA
efficiency comparison nobody has measured yet.

### P5 — the eval ladder (needs a converged Phase-2 checkpoint)
Reuse the harness verbatim: `bash scripts/msa/setup_eval_root.sh <src> <dst>` clones an isolated eval
root (symlinking the ~12 GB of read-only inputs), and the 13-bench queue runs against an
OpenAI-compatible endpoint served under the **baseline's** model name so the existing OpenCompass
configs need no edit (`scripts/msa/serving/serve_msa.sh:5-7`). Row 0 is already measured —
`/cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking`: IFEval 88.72, AIME25 83.13±0.83, GPQA-d 64.40±1.18,
MMLU-Pro 72.50, LCB v6 53.82, RULER 4K/8K/16K/32K = 97.15 / 96.46 / 96.65 / 95.33.

Ladder rows, benchmarks, gates, and the thinking-mode protocol (sampled decoding, fixed n, paired
per-item deltas, truncation rate as a first-class metric) come from [eval_plan.md](eval_plan.md) §3–§7
unchanged. Two additions:

- **Order `mmlu_g1` before `mmlu_g3`** — `mmlu_g3` always dies at startup racing `g1` on a
  non-atomic cache write, and `run_queue.sh` still reports "13/13 ok" (memory
  `msa-eval-mmlu-g3-dataset-size-race`).
- **`--block-size 64`**, not MSA's 128.

### P6 — the comparison that is the actual point (~1 day on top of P5)
**Paired DSA k=2048 tokens vs MSA k=16 blocks × 128 = 2048 tokens, identical KV budget, identical
samples.** MSA k16 reached dense parity on short context but still fails RULER 32K by **−6.36**
(memory `msa-k16-eval-result`, `docs/qwen3_4b_msa/SCORECARD_msa_longctx.md`), and its
`block_recall` plateaued at 87% of the k=16 ceiling — which is exactly the deficit token granularity
is supposed to remove. Report `Δ_granularity(L) = DSA(L) − MSA(L)` at 4K/8K/16K/32K next to the
throughput ratio from P4. That table is the deliverable of the whole DSA-on-GQA project.

---

## 5. Acceptance gates

Kernel/serving gates (this doc); model-quality gates are [eval_plan.md](eval_plan.md) §7.

1. P0 correctness: page-1 GQA FA3 == gather+SDPA reference.
2. `DSA_SPARSE=0` == stock Qwen3, greedy, exact.
3. `top_k ≥ T` == dense, at every tested length.
4. Selection overlap ≥ 0.99 train↔serve; top-k bit-deterministic across runs.
5. Randomized indexer collapses; the sparse backend is named in the serve log; `index_topk` present
   and *asserted*, never soft-gated.
6. Throughput ≥ dense at 8K and above (otherwise sparsity is buying nothing at serve time and the
   result is a research number, not a deployable one — say so in the scorecard rather than burying it).

---

## 6. Risks and fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| FA3 page-1 prefill is slow at 32K (§4 P0.3) | P0 timing | chunked prefill; a dedup pass (Keye's route); or dense prefill + sparse decode, which is where the bandwidth win lives anyway |
| `pack_gqa` heuristic mis-chooses | P0.4 | wrapper call into `_flash_attn_forward`, or bump the wheel |
| fp8 index cache + UE8M0 drift | P1/P3 overlap < 0.99 | `fp8_ue8m0=True` is already the trained default; re-check the Hadamard scale convention |
| cudagraph capture gives wrong numbers, not a crash | P3 parity with cudagraphs on vs off | `--enforce-eager`, as MSA shipped; costs throughput only |
| 0.26.0 ages out / we need a newer sparse fix | — | `.devlibs/vllm-src` (commit 1206891) is a newer checkout of the same files; the `vllm/models/` layout differs |
| The whole page-1 approach fails | P0 correctness | block-granular MSA already works end-to-end on this exact model; DSA becomes a training-side result reported with an honest serving caveat |

---

## 7. Cost

| phase | wall-clock | needs checkpoint? | GPUs |
|---|---|---|---|
| P0 spike | ~1 d | no | 1 |
| P1 indexer parity | ~1 d | no | 1 |
| P2 dense plugin | ~1 d | no | 1 |
| P3 sparse parity | ~2 d | Phase-1, any step | 1 |
| P4 throughput | ~1 d | Phase-1 | 1 |
| P5 eval ladder | ~3–5 d | Phase-2 converged | 8 (one server/GPU) |
| P6 DSA vs MSA | ~1 d | + MSA k16 (exists) | 8 |

**~6 days of it is unblocked today**, and P0 alone decides whether the token-granular bet is servable.
