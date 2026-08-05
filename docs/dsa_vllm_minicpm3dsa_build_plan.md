# Build plan — `MiniCPM3DSAForCausalLM` on vLLM's sparse path (Tier 3)

> **DECODE BUILD PLAN (post-recon, 2026-07-21) — see the dedicated section at the bottom of this file.**
> Goal chosen: serve the Phase-2 *sparse* model on vLLM for **generation**, to run the MiniCPM3-4B card
> benchmark suite (MMLU/CMMLU/CEval/BBH/GSM8K/MATH/HumanEval/MBPP/IFEval/BFCL/LiveCodeBench — all
> generation/agentic, not teacher-forced) through the existing OpenAI-compatible eval harness.

Companion to `docs/dsa_vllm_serving.md` (source-verified 2026-07-20). This is the **Tier 3** path from that
doc's §6 — building the custom vLLM class so the Phase-2 sparse model can be *generated* from at long context
with real throughput. Only pursue this once Tier 1 (dense ≤512-tok evals) and Tier 2 (HF sparse forward for
logprob/MC/perplexity) are landed and you have a concrete need for **fast long-context generation**.

## Build-gate (decide before Phase 1)
Build Tier 3 **only if** all hold:
- Tier 2 (HF sparse forward) has already validated that the sparse mechanism is correct at long context — so
  we're not debugging model quality *and* a new serving stack at once.
- There is a real generation workload (free-form, long-context) that Tier 2 can't serve (Tier 2 has no
  incremental-decode KV cache).
- Throughput matters enough to justify the effort estimate below (~1.5–3 wk, dominated by Phase 3).

If any fails, stay on Tiers 1–2.

---

## Architecture reality (verified against vllm 0.20.2)
Two facts shape the whole plan:
1. **vLLM's `MiniCPM3Attention` is dense** — it materializes full Q/K/V from the MLA latents and calls the
   standard `Attention` op (`minicpm3.py:52,126,141-156`). There is **no** MiniCPM3 MLA/sparse backend to
   toggle. The sparse machinery lives on `DeepseekV2MLAAttention` (`deepseek_v2.py:855`, `is_v32` at :976),
   which owns the `Indexer` and feeds its `topk_indices_buffer` into a sparse-MLA attention backend.
2. **Indexer RoPE is a config knob**: `is_neox_style = not config.indexer_rope_interleave`
   (`deepseek_v2.py:983`), default non-interleaved — matches our `rotate_half` indexer. Set
   `indexer_rope_interleave=False` (or omit). MiniCPM3's *main* rope is also non-interleaved (do NOT inherit
   DeepSeek's interleaved MLA rope).

Consequence: two viable attention strategies for the custom class —
- **(B) indexer-selects + gathered-KV dense attention** — run indexer → top-k indices → attend over the
  gathered KV (the vLLM analogue of our HF `_sparse_attn`). *Correct and much less code; reuses MiniCPM3's
  existing attention.* Limited throughput/memory win (KV still materialized), but proves the full stack.
- **(A) true sparse-MLA backend** — adapt `DeepseekV2MLAAttention`'s MLA + sparse backend to MiniCPM3 dims.
  Max throughput, but heavy (MLA absorption + backend assume DeepSeek dims). 

**Recommended sequencing: milestone-1 = (B) for correctness, milestone-2 = (A) for throughput.** (B) is a
hard gate on (A) anyway — it isolates weight-loading/indexer bugs from backend bugs.

---

## Phase 0 — DeepGEMM probe — ✅ DONE (2026-07-20, `tests/dsa/probe_deepgemm_indexer.py`)
**Result: GO.** Kernel runs at our (padded) dims; head_dim 64→128 pad is **bit-exact** (`max|Δlogit|=0`);
kernel applies **per-head ReLU**; dropping the Hadamard is a **~2%** top-512 selection effect (not
load-bearing). Two new must-handle items surfaced — both now in the phases below:
1. **`n_heads=16` is kernel-rejected** (only 32/64/128) → pad heads 16→32 with zero q + zero weights (exact).
   Added to Phase 2.
2. **This env's deep_gemm lacks `fp8_fp4_mqa_logits`** (vLLM wrapper → `_missing()` stub; external `~/.local`
   package shadows the vendored one) → **prerequisite for Phases 3–4**, see new Phase 0.5.

Original probe spec (kept for reference / re-runs):

## Phase 0 (spec) — DeepGEMM probe (STEP ZERO, standalone, no model)
Source (Python + CUDA templates) already says there is **no Hadamard** in the kernel and that head_dim only
needs `% 32 == 0` (see serving doc §3/§4). But the runtime dispatch and the K-side quant live in a compiled
`.so` we can't read — so confirm the **compiled** path numerically before writing any model code. This is the
"rotation probe": if it passes, Phase 3's re-inserted Hadamard is the only rotation in play; if it fails, stop
and reconcile.

Standalone script (`tests/dsa/probe_deepgemm_indexer.py`), no vLLM model load:
```python
# reuse exactly what vLLM's Indexer.forward calls
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm.model_executor.layers.sparse_attn_indexer import (fp8_fp4_mqa_logits)  # or the deep_gemm entry
# 1) controlled inputs: real-64 known values + 64 zeros; n_head=16, rope handled upstream
q64 = torch.randn(T, 16, 64, device="cuda", dtype=torch.bfloat16)
k64 = torch.randn(T, 64,     device="cuda", dtype=torch.bfloat16)
q128 = F.pad(q64, (0,64)); k128 = F.pad(k64, (0,64))
# 2) run the compiled path at padded 128
q_fp8, q_scale = per_token_group_quant_fp8(q128.view(-1,128), 128, use_ue8m0=True)
logits_kernel = <mqa logits kernel>(q_fp8, q_scale, k128, weights=...)   # match Indexer.forward wiring
# 3) references
ref_nohad = fp8_reference_mqa_dot(q64, k64)          # plain fp8 dot, NO Hadamard  -> must MATCH kernel
ref_pad   = fp8_reference_mqa_dot(q128, k128)        # padded 128, NO Hadamard     -> must MATCH kernel (exactness)
ref_had128= fp8_reference_mqa_dot(H128(q128),H128(k128))  # rotated over 128       -> must DIFFER (proves no internal rotate)
```
**Pass criteria (gate to Phase 1):**
- `logits_kernel ≈ ref_nohad ≈ ref_pad` within fp8 tolerance → no hidden rotation, padding is exact in the
  compiled path, head_dim=128/n_head=16 dispatch works.
- `logits_kernel` clearly `≠ ref_had128` → the kernel does not expect/undo a 128-wide Hadamard.
- Also insert a known K via the k-cache op and read back → confirm symmetric-absmax fp8 + ue8m0 @ block 128.

**If it fails:** a rotation (or different quant) is baked into the compiled kernel → revisit §4; the fix is to
match whatever the kernel expects (likely rotate over the real 64 then pad, or drop our rotation), re-probe.

---

## Phase 0.5 — fix the deep_gemm install (blocker for serving) — ✅ RESOLVED (2026-07-20)
The probe showed `vllm.utils.deep_gemm.fp8_fp4_mqa_logits` resolves to a `_missing()` stub in this env: the
external `~/.local/lib/python3.12/site-packages/deep_gemm` (hand-staged, no pip metadata) has legacy
`fp8_mqa_logits` but **not** `fp8_fp4_mqa_logits`, and vLLM's importer tries top-level `import deep_gemm`
FIRST (`utils/deep_gemm.py:152-167`) — it succeeds, so vLLM never falls back to the vendored
`vllm.third_party.deep_gemm` that *does* have the symbol. Nothing in `verl/` imports deep_gemm directly, and
dense serving never calls the kernel — so this is a **Tier-3-only** blocker (baseline dense evals are
unaffected).

**Verified fix (`tests/dsa/check_vendored_deepgemm.py`, run in the real serve env):**
- Vendored `vllm.third_party.deep_gemm` imports cleanly (no `_C.init()` failure — wheel is complete here) and
  has `fp8_fp4_mqa_logits`.
- It **runs**: one real kernel call matched our ReLU reference to `max|Δ| = 0.0024` (fp8 tol) — complete, not
  just importable.
- Hiding *only* top-level `deep_gemm` makes vLLM fall back to vendored end-to-end: `_fp8_fp4_mqa_logits_impl`
  goes None→resolved (`vllm.third_party.deep_gemm._C`) and the public wrapper runs.

**Least-invasive mechanism (apply in the Tier-3 serve wrapper, BEFORE `import vllm`):** install a
`sys.meta_path` finder whose `find_spec` raises `ImportError` for `name == "deep_gemm"` and `"deep_gemm."`
prefixes (leave the dotted `vllm.third_party.deep_gemm` alone). Touches neither disk nor `sys.path`, so
flashinfer / fast_hadamard_transform / cupy / sgl_kernel in `~/.local` stay importable. Equivalent packaging:
a one-line `sitecustomize.py`/`.pth` on the serve `PYTHONPATH`. **Do NOT** use `PYTHONNOUSERSITE` (hides the
other ~/.local packages). Fallback if vendored ever regresses: upgrade the external deep_gemm to a build with
`fp8_fp4_mqa_logits`.
```python
# serve wrapper, before `import vllm`:
import sys, importlib.abc
class _BlockExternalDeepGEMM(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "deep_gemm" or name.startswith("deep_gemm."):
            raise ImportError("forcing vendored vllm.third_party.deep_gemm")
        return None
sys.meta_path.insert(0, _BlockExternalDeepGEMM())
```
Note: the vendored kernel has the **same `H ∈ {32,64,128}` constraint** — the 16→32 zero-weight head-pad
(Phase 2) is still required.

## Phase 1 — checkpoint + config prep
- Consolidate a Phase-2 checkpoint to a world-size-agnostic, HF-loadable dir
  (`scripts/dsa/consolidate_indexer_ckpt.py`) with **base + indexer** weights.
- Author the HF `config.json` DSA fields the vLLM Indexer reads: `index_topk=512`, `index_n_heads=16`,
  `index_head_dim=128` *(padded; the custom module holds native-64 weights but sizes cache/k_norm at 128 —
  see Phase 2)*, `qk_rope_head_dim=32`, `q_lora_rank=768`, `indexer_rope_interleave=false`. Add a marker attr
  so our class registers (e.g. `model_type` / `architectures: ["MiniCPM3DSAForCausalLM"]`).
- Decide packaging: **out-of-tree plugin** (register via `vllm.ModelRegistry.register_model` in a small
  importable module) — do NOT edit `dist-packages`. Recommended.

---

## Phase 2 — custom indexer module (`MiniCPM3DSAIndexer`)
Subclass/rewrite vLLM's `Indexer` so weights stay native-64 but the FP8 path runs at padded-128:
- `__init__`: `wq_b = Linear(768, 16*64)` (native 64), `wk`/`weights_proj` handling (see loader below),
  `k_norm = LayerNorm(64)`, `softmax_scale = 64**-0.5` (we own it; top-k-invariant anyway). Reuse
  `DeepseekV32IndexerCache` sized at head_dim **128** (→132-byte entries) and `SparseAttnIndexer` @ block 128.
- `forward` (the padded pipeline from serving-doc §4, with the two padding axes Phase-0 confirmed):
  `wq_b(qr)` → rope on rope-slice (non-interleaved) → `k_norm(wk(x))` over real 64 → **`rotate_activation`
  (Hadamard) over the real 64** (optional — ~2% selection effect per Phase-0; cheap, keep it) →
  `F.pad(...,(0,64))` head_dim 64→128 → **pad heads 16→32 with zero q AND zero weights** (kernel rejects
  16; padded heads contribute 0) → `per_token_group_quant_fp8(...,128, use_ue8m0=True)` → `indexer_op` →
  top-k indices. **Do NOT apply ReLU here** — Phase-0 confirmed the kernel applies per-head ReLU internally.
- Weight loader: our **separate** `wk` + `weights_proj` → vLLM's **fused** `wk_weights_proj`
  (`MergedColumnParallelLinear([64,16])`); or keep them separate in our module and skip the fusion (simplest,
  since we own the module). `wq_b`, `k_norm` load 1:1.

---

## Phase 3 — model class `MiniCPM3DSAForCausalLM`
Subclass vLLM's `MiniCPM3ForCausalLM`; add the indexer per decoder layer and route attention through it.
- **Milestone 1 (B): gathered-KV.** Per layer: build the MLA Q/K/V as MiniCPM3 already does, run
  `MiniCPM3DSAIndexer` → top-k indices, then attend over gathered KV (dense attention masked/gathered to the
  selected set). Mirror HF `_sparse_attn` (`minicpm_dsa.py:432`) semantics; `top_k ≥ T` must reproduce dense
  exactly (the M0 parity test). Correctness milestone — generation works, numerics match HF.
- **Milestone 2 (A): sparse-MLA backend.** Replace the gathered-KV attention with the real sparse-MLA backend
  used by `DeepseekV2MLAAttention` (feed `topk_indices_buffer`), adapting MLA absorption/dims to MiniCPM3.
  Throughput milestone. Higher risk — scope after M1 lands.

---

## Phase 4 — registration + serving smoke
- Plugin module registers `MiniCPM3DSAForCausalLM`; ensure the `.devlibs` transformers-4.57.1 env is used
  (see memory: serve via system python + `PYTHONPATH=.devlibs/tf457lib`).
- Launch a single vLLM replica on the consolidated ckpt; smoke a short prompt (T ≤ 512 → must equal dense),
  then a long prompt (T > 512 → exercises real selection).

---

## Phase 5 — parity + eval
- **Parity:** vLLM sparse-served **LM logits** and **selected-index sets** ≈ HF sparse forward
  (`dsa_mode=sparse, top_k=512`) on sample prompts. Compare LM logits / indices, **not** raw indexer scores
  (softmax_scale differs, §5). Include a `top_k ≥ T` case (must equal dense) and a `T ≫ top_k` case.
- **Eval:** once parity holds, run the long-context *generation* benchmarks that motivated Tier 3.

---

## Risk & effort
| Phase | Effort | Risk | Note |
|---|---|---|---|
| 0 probe | ~0.5 d | low | source says pass; confirms compiled path. Hard gate. |
| 1 ckpt/config | ~0.5 d | low | reuses consolidate script |
| 2 indexer | ~2 d | med | padding + rotate + loader; unit-test vs HF projection |
| 3 M1 gathered-KV | ~3–5 d | med | the integration bulk; parity vs HF |
| 3 M2 sparse-MLA | ~1 wk | high | MLA-absorption/backend adapted to MiniCPM3 dims; optional |
| 4–5 serve/parity/eval | ~2 d | med | |

**Decisions to confirm:** (a) attention strategy — recommend B→A staged; (b) stop at M1 if its throughput is
"good enough" (KV still materialized, but attention cost drops); (c) plugin vs fork — recommend plugin.

---

# DECODE BUILD PLAN (post-recon, 2026-07-21)

**Chosen path:** serve the Phase-2 sparse model on vLLM for generation. The recon settled what's reusable and
what isn't, so this plan has ONE genuinely new component and a lot of reuse.

> **UPDATE (2026-07-21) — PADDING PROBE PASSED → NO CUSTOM KERNEL NEEDED.**
> `tests/dsa/probe_flashmla_padding.py` proved we can **reuse** vLLM's FlashMLA-sparse decode kernel by
> zero-padding MiniCPM3's MLA dims up to DeepSeek's (nope 256→512, rope 32→64 ⇒ head **576**; value
> 256→**512**; heads 40→**64**), BF16 cache. Exactness `max_abs_err=5.5e-4` (bf16 tol); zero-padded output
> dims are exactly 0. So the make-or-break blocker is **solved by padding, not a kernel rewrite**:
> **Stage 2 becomes integration (wire the padded reuse), and Stage 4 (custom Triton kernel) is DROPPED**
> (revisit only if the ~2× KV/compute cost proves prohibitive at long context).
> Key mechanics: callable = `flash_mla_sparse_fwd` (absorbed MQA, runtime `sm_scale=288**-0.5`, no internal
> RoPE — pre-rope q/k), heads must be padded to 64, and **topk must be a multiple of 2·B_TOPK** (128 works,
> 64 rejected; our `top_k=256` = 2×128 is fine — pad indices with -1 / use `topk_length`). Cost: ~2× KV bytes
> + ~2× per-selected-key compute.

> **UPDATE (2026-07-21) — Stage 1 done (MLA math proven), and the dim-lock is BROADER than the sparse kernel.**
> Stage 1 proved the MLA absorbed-latent formulation is numerically exact vs stock materialized-QKV
> (fp32 `max|Δ|=5.7e-5`; bf16 rel `0.56%`), weights load 1:1, muP preserved. BUT it found: (i) vLLM's **dense**
> `MLACommonBackend.get_supported_head_sizes()==[320,576]` also rejects MiniCPM3's head **288**, and (ii)
> `model_type="minicpm3"` isn't in vLLM's `is_deepseek_mla` allowlist → `use_mla=False`. So the whole MLA path
> (metadata `__post_init__`, cache spec, op) — not just the sparse kernel — is 576-locked, and vLLM won't even
> route MiniCPM3 through MLA by default. **Consequence:** Stage-1 parity is a *standalone* proof (the engine
> rejects 288); the **end-to-end engine run is deferred into Stage 2**. Stage 2 now = **(2a)** pad the entire
> MLA path to 576 + force vLLM to treat MiniCPM3DSA as MLA (patch allowlist/flag), and prove the model runs
> end-to-end in DENSE MLA (logits == stock dense); **(2b)** then enable padded FlashMLA-sparse. Also note:
> the Stage-0 test's dense-generation assertion now fails **by design** (post-restructure the main attn is MLA
> → hits the 288 lock until 2a); dense generation still available via the `MiniCPM3StockRefForCausalLM` ref.

## The core finding that shapes everything
- **Reusable as-is (config/shape-driven):** the `Indexer` + `DeepseekV32IndexerCache` (validated Phase 0/2),
  the MLA wrapper (`MultiHeadLatentAttentionWrapper` / `MLAAttention`), automatic 2-cache-group collection,
  and muP (kept via the MiniCPM shells).
- **NOT reusable — the one blocker:** vLLM's sparse **decode attention kernel** (`FlashMLASparseImpl`) is
  hard-locked to DeepSeek latent dims — `get_supported_head_sizes()=[512,576]`, `head_dim_v=512`
  (`flashmla_sparse.py:119-121,965-975`). MiniCPM3 is head_size 288 / d_v 256 → rejected at backend
  selection. **So we write ONE new component: a native-dim (288/256) sparse-MLA attention impl.** Everything
  else is reuse + reconfiguration.
- **Selection is reused, only the attend-over-selected step is new:** the indexer's `SparseAttnIndexer`
  produces `topk_indices_buffer` (works at our dims). Our custom impl only consumes that buffer + gathers the
  selected latent-KV + does MQA. Smaller surface than "a whole sparse backend."

## Class design (out-of-tree plugin `scripts/dsa/vllm_minicpm3_dsa/`)
- `MiniCPM3DSAForCausalLM(MiniCPMForCausalLM)` — keep muP `scale_width`/logits.
- `MiniCPM3DSAModel(MiniCPMModel)` — keep `embed*scale_emb`; add model-level `topk_indices_buffer`
  (`[max_batched_tokens, index_topk]` int32) gated on `index_topk`, threaded to every layer.
- `MiniCPM3DSADecoderLayer(MiniCPMDecoderLayer)` — keep residual `scale_depth/sqrt(L)`; swap `self_attn`.
- `MiniCPM3DSAAttention(DeepseekV2MLAAttention)` — reuse the is_v32/Indexer/MLA wiring, but MiniCPM3 dims,
  separate `q_a_proj`+`kv_a_proj_with_mqa` (vs DeepSeek fused → weight remap), MiniCPM3 scaling (no yarn
  mscale), longrope (no yarn rewrite), non-interleaved indexer rope.
- `MiniCPM3SparseMLAImpl(SparseMLAAttentionImpl)` — **the new kernel**: `is_sparse/is_mla=True`,
  `get_supported_head_sizes()=[288]`; reads `topk_indices_buffer`; gathers selected latent-KV; MQA attend at
  native dims. Reference (torch gather+SDPA) first, Triton later.
- Config: add `index_topk(=256)`, `index_n_heads(=16)`, `index_head_dim(=128 padded)`, `indexer_rope_interleave=false`
  aliases (vLLM reads `index_*`, not our `dsa_*`). Register class + install the Phase-0.5 deep_gemm shim.

## Staged implementation + test-at-each-gate
The golden reference throughout is the **HF sparse forward** (`minicpm_dsa.py::_sparse_attn`) and the
**`top_k ≥ T ⇒ output == dense`** invariant (a free correctness oracle at every stage).

**Stage 0 — plugin skeleton + config plumbing (runnable today).** Build the class hierarchy; route MAIN
attention to the existing dense materialized-QKV path; keep the indexer running.
- *Test:* vLLM loads the serving dir (weights 1:1, no missing/unexpected keys); engine builds BOTH kv-cache
  groups (assert from kv_cache spec); a forward runs; `topk_indices_buffer` is populated with valid indices;
  a short prompt generates coherent text (== base model, since main attn is dense). Proves config + cache +
  indexer runtime with NO custom kernel.

**Stage 1 — MiniCPM3 MLA-latent attention (dense, correct).** Implement `MiniCPM3DSAAttention` as real MLA
(preprocess + kv_b absorption) via `MLAAttention(use_sparse=False)`; handle separate q-proj, MiniCPM3
scaling, longrope.
- *Test:* LM-logit parity of MLA-form dense vs the stock materialized-QKV MiniCPM3 (same weights) on sample
  prompts, within fp tol. Isolates "MLA expressed correctly" from the sparse kernel.

**Stage 2 — custom sparse impl, correctness (reference version).** Write `MiniCPM3SparseMLAImpl` (torch
gather + SDPA/bmm, correctness over speed); register so vLLM selects it for head_size=288 + use_sparse.
- *Test:* (a) `top_k ≥ T` ⇒ output == Stage-1 dense exactly; (b) prefill LM logits vs HF sparse forward within
  tol; (c) selected-index sets match the indexer. Proves the sparse attention is numerically right.

**Stage 3 — decode/generation correctness (the incremental caches).** Verify both caches grow per step,
indexer scores incrementally, impl gathers per step.
- *Test:* greedy generation from the vLLM server matches greedy generation from the HF sparse model
  token-for-token on sample prompts; `top_k≥T` generation == dense generation. Decode correctness gate.

**Stage 4 — kernel perf.** Replace the reference impl with a Triton MQA-over-gathered-topk kernel (native
288/256, num_heads pad 40→64).
- *Test:* numerical equivalence to the Stage-2 reference (tight tol); throughput (tok/s at long context) +
  memory vs dense. (Fallback to eval-tolerable speed on the reference impl if Triton slips.)

**Stage 5 — serve + run the card suite.** Adapt `serve_multi_devlibs.sh` (add deep_gemm shim, point at the
serving dir), one server per GPU; run the existing benchmark clients (mmlu, cmmlu, ceval, bbh, gsm8k, math,
humaneval_plus, mbpp_plus, ifeval, bfcl, livecodebench).
- *Test/eval:* produce a scorecard; compare vs the stock-MiniCPM3 baseline scorecard; this is the true sparse
  model's capability. Sanity: a `top_k≥T` serve run should ≈ the dense baseline on short benchmarks.

## Risks (ranked) & mitigations
1. **[make-or-break, mitigated]** sparse kernel dim-lock → the custom Stage-2 impl (correctness) then Stage-4
   (perf). Confirmed the *selection* side (indexer) already works at our dims.
2. **[med]** q-proj fused-vs-separate weight layout → weight remap in Stage 1 (our ckpt has separate names).
3. **[med]** muP must not be lost → keep the MiniCPM `*ForCausalLM/Model/DecoderLayer` shells (Stage 0).
4. **[low]** `index_topk=256 ∉ {512,1024,2048}` → indexer decode uses `top_k_per_row_decode` (correct, a bit
   slower) not `persistent_topk` (`sparse_attn_indexer.py:323`). Acceptable; revisit only if it dominates.
5. **[low]** rope (bypass DeepSeek yarn, keep longrope; indexer neox-style) — handled in Stage 1.

## Effort (rough)
Stage 0 ~1d · Stage 1 ~2–3d · Stage 2 (ref impl) ~3–4d · Stage 3 ~2d · Stage 4 (Triton) ~4–6d · Stage 5 ~2d.
The custom-kernel stages (2+4) are the bulk; Stages 0–3 give a **correct (if slow) end-to-end sparse
generation server** before any Triton work — so the eval suite can run on the reference impl while Stage 4
optimizes.
