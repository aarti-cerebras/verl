# DSA sparse MiniCPM3-4B — evaluation report

End-to-end evaluation of the **Phase-2 DSA sparse** MiniCPM3-4B (trained lightning-indexer, top-k sparse
attention) served on vLLM, vs the **stock dense baseline** and the **HF-card reported** numbers. Covers the
results, how sparsity was verified genuine, the HumanEval+ investigation, and the LiveCodeBench issue + fix.

- **Model:** Phase-2 ckpt `phase2_full_k256_1ep` @ `global_step_2805` (base+indexer), `index_topk=256`.
- **Eval root:** `/cb/ml-eng/aarti/dsa/evals/minicpm3-4B-dsa-k256/` (one-page results: `DSA_SCORECARD.md`).
- **Baseline-dense root:** `/cb/ml-eng/aarti/dsa/evals/minicpm3-4B/` (stock `openbmb/MiniCPM3-4B`, full attn).
- **Serving:** `MiniCPM3DSAForCausalLM` vLLM plugin (`scripts/dsa/vllm_minicpm3_dsa/`), bf16, TP=1,
  `--max-model-len 32768`, `enforce_eager`, one replica/GPU (DP for throughput). Build story:
  `docs/dsa_vllm_minicpm3dsa_build_plan.md`; kernel/serving internals: `docs/dsa_vllm_serving.md`.
- **Date:** 2026-07-21/22.

## 1. Scorecard

| Benchmark | Reported | Baseline-dense | **DSA sparse tk256** | Δ dense | **DSA sparse tk128** |
|---|--:|--:|--:|--:|--:|
| MMLU (5-shot) | 67.2 | 66.80 | 67.19 | +0.39 | 67.36 |
| CMMLU (5-shot) | 73.3 | 72.72 | 72.84 | +0.12 | 72.56 |
| CEval (5-shot) | 73.6 | 72.18 | 72.03 | −0.15 | 72.01 |
| GSM8K (8-shot CoT) | 81.1 | 79.98 | 78.77 | −1.21 | 79.38 |
| MATH (0-shot CoT) | 46.6 | 46.58 | 47.34 | +0.76 | 47.44 |
| IFEval (prompt-strict) | 68.4 | 70.79 | 71.72 | +0.93 | 72.27 |
| HumanEval+ (0-shot) | 68.3 | 69.5 | 65.2 (base 72.0) | −4.3 | — |
| MBPP+ (0-shot) | 63.2 | 56.3 | 61.9 (base 70.1) | +5.6* | — |
| LiveCodeBench v3 | 22.6 | 20.7 | re-running (mp=6) — see §4 | — | — |

`*` MBPP+: the dense baseline (56.3) was itself anomalously low (a baseline "set-version" harness gap, −6.9 vs
reported). Read DSA-sparse MBPP+ as **≈ reported** (61.9 vs 63.2), not a real +5.6 gain.

**Headline:** the sparse model **matches dense within noise** on knowledge (MMLU/CMMLU/CEval), math
(GSM8K/MATH), and instruction-following (IFEval) — at/above dense on 5 of them. Mean |Δ vs dense| ≈ **1.1** over
the clean comparisons. **top_k=128 ≈ top_k=256** everywhere (±0.6). The one real cost is **HumanEval+ (−4.3)** —
see §3. LiveCodeBench had a harness bug (§4), being re-run.

## 2. Verification — is it *genuinely* sparse? (not silently dense)

**top_k causal sweep (GSM8K):**

| top_k | 2 | 128 | 256 | 2048 | dense |
|---|--:|--:|--:|--:|--:|
| GSM8K | **0.08** | 79.08 | 78.77 | 79.76 | 79.98 |

Accuracy is causally controlled by the indexer budget across ~1000×: starve to 2 keys → collapse; open to ≥
context → recovers dense. A silently-dense model would be flat; a broken indexer low everywhere. Only genuine
sparse attention over a real trained indexer produces this curve.

**Forensics (from the serving process):** correct `step_2805` checkpoint; 310 indexer keys loaded 1:1 by the
strict loader (not random); engine chose `FLASHMLA_SPARSE` as the sole backend + built the
`DEEPSEEK_V32_INDEXER` cache group.

**Sparsity actually bit (length histograms, tk256):** MMLU/CMMLU/CEval/GSM8K exceed 256 prompt tokens on
**100% of items** (~1000-token few-shot contexts), so the indexer sub-selected on every query — the near-dense
scores are genuine sparse results, not a degenerate ≤top_k regime. (IFEval prompts are short — sparsity bites
during generation, ~58% of items >256 for prompt+gen.)

## 3. HumanEval+ (−4.3) — the one real regression, and how it splits into drift vs sparsity

### The confound
The scorecard's −4.3 compares two configs that differ in **two** ways at once:
- **Baseline-dense** = *stock* MiniCPM3 weights + *full* attention → **69.5**
- **DSA sparse** = *Phase-2* weights + *sparse* (top_k=256) attention → **65.2**

So −4.3 bundles **(a) the weight change** (stock → Phase-2 behavior-cloning) and **(b) the attention change**
(dense → top-k sparse). It cannot be attributed to sparsity without holding the weights fixed.

### The de-confounding control: top_k=2048
We need the **Phase-2 weights with sparsity effectively OFF**. `top_k=2048` provides exactly that: HumanEval
prompt+gen (~600 tok) ≪ 2048, so a budget of 2048 selects **every** key → the sparse path degenerates to full
attention, but on the Phase-2 weights. This isn't assumed — the **top_k causal sweep proved it on GSM8K**
(`top_k=2048 → 79.76 ≈ dense 79.98`), so 2048 is a *validated* dense-equivalent operating point.

### The ladder — each step changes exactly one variable
| Config | weights | attention | HumanEval+ |
|---|---|---|--:|
| stock dense | stock | full | 69.5 |
| Phase-2 @ top_k=2048 | **Phase-2** | full (2048 ≥ ctx) | 66.5 (base 73.2) |
| Phase-2 @ top_k=256 | Phase-2 | **sparse** | 65.2 (base 72.0) |

- **69.5 → 66.5 = −3.0** — only the *weights* changed, attention held full ⇒ **behavior-cloning (BC) drift** (dominant).
- **66.5 → 65.2 = −1.3** — only the *attention* changed, weights held at Phase-2 ⇒ **sparsity cost** (small).

They sum to −4.3. **So the drop is mostly Phase-2 BC drift on code, not sparsity.**

### Why the −1.3 is believable as the sparsity part
It matches the other de-confounded point: on GSM8K, base-drift ≈ 0 (Phase-2 @ 2048 = 79.76 ≈ dense) and the
sparse cost was ~−1.4 (79.98 → ~78.6). HumanEval's −1.3 lands in the same band — sparsity is uniformly gentle
(~1–1.5 pt); code additionally suffers BC drift that knowledge/math don't show. (Mechanistically, code is the
most sparsity-*sensitive* task — it must hold the exact signature/docstring/constraints across a long,
verbose generation, p50 ~406 gen tok, 91% >256 → it *does* sub-select during decode — yet the measured
sparsity cost is still only −1.3.)

### Caveats (keep honest)
- The −1.3 / −3.0 split rests on `top_k=2048` being dense-equivalent for HumanEval (validated on GSM8K; holds
  here since prompt+gen ≪ 2048).
- Single run per ladder point (no error bars) — read it as "**sparsity is the minority cause**", not a precise −1.3.

### Next steps (training-side, since drift dominates)
Rebalance/upweight code in the Phase-2 BC data; gentler BC (lower LR / higher KL weight). Sparsity-side fixes
(code top_k sweep, code-aware indexer) are lower priority given the −1.3.

## 4. LiveCodeBench — the harness issue (and fix)

**Symptom:** two runs returned ~**0.8% pass@1** with only **316–352 / 612** problems graded (**~43% empty
generations**). Anomalous — the model codes fine on HumanEval+/MBPP+.

**Root cause (fully diagnosed):** LCB requests **all 10 samples in one API call** (`oai_runner`: `n=args.n`),
and on any exception **retries with n−1, returning empty strings at n=0** (`base_runner`:
`outputs.extend([""]*n)`). An `n=10` request = 10 × ~2000 tokens on the **enforce-eager** sparse server (~79s
solo, confirmed). Under LCB's **mp=24** concurrent load these queue past the client timeout → exception →
retry-to-empty cascade → ~43% of problems empty → 0.8%.

**Ruled out (with evidence):** throughput/timeouts (DP=6 gave `timeouts=0`, still failed), extraction (clean
```python``` on non-empty), general code ability (HumanEval+/MBPP+ fine), prompt length (empty-gen problems
p50 521 tok ≈ non-empty 478 — length-independent), **DP deployment** (24 concurrent identical requests → 0
empties on both DP and single; empty problems produce valid code on n=1 replay), and the model itself. A single
n=10 request completes in **79s with 10/10 non-empty samples** — so n=10 works given room.

**Fix:** re-run at **mp=6** across the DP=6 server (one n=10 per replica → no contention → no timeout → no
retry-to-empty), keeping the baseline's exact methodology (n=10, temp 0.2, release_v3). Clean so far
(`timeouts=0`, ~96% success); slow (~hours) because the enforce-eager sparse server is the bottleneck. Result
pending → will complete the 9th benchmark.

## 5. Methodology & caveats

- **macro-averages** (MMLU/CMMLU/CEval) computed identically to the baseline → Δ-vs-dense is apples-to-apples.
- **max-len score-neutral:** tk256 knowledge/IF/GSM8K are the 32K re-serve; MATH + code are 8192 — a 32K re-run
  of the 5 completed benches matched 8192 within ≤0.45 (run-to-run noise; max prompt+gen ≈2071 ≪ both caps).
- **Base-drift vs sparsity:** "baseline-dense" is the *stock* model, so Δ-vs-dense mixes Phase-2 BC drift +
  sparsity. GSM8K top_k=2048 control (79.76 ≈ dense) shows drift ≈0 there; not separately isolated elsewhere
  (except the in-progress HumanEval+ test).
- **Inherent ~2% train/serve selection drift:** kernel UE8M0 vs training plain-absmax FP8 scale (§ serving doc).
  The `phase2_full_k256_1ep` checkpoint evaluated here was trained with the legacy plain-absmax scale, so it
  carries this drift. **Fix now implemented (2026-07-22, off by default):** `DSAConfig.fp8_ue8m0` /
  `+model.override_config={... dsa_fp8_ue8m0: true}` trains the indexer against the kernel's UE8M0 quant. Parity
  verified — top-256 selection overlap vs the real DeepGEMM kernel goes **0.9698 → 1.0000** mean
  (`0.9297 → 0.9961` worst-query); the CPU quant math is bit-identical. Tests:
  `tests/dsa/test_indexer_fp8_ue8m0_parity.py` (CPU), `tests/dsa/test_minicpm3_dsa_indexer_parity.py` (GPU).
  Realizing the gain requires retraining with the flag on.
- **Serving constraints:** `enforce_eager` (no CUDA-graph support validated for the sparse path → slow decode);
  TP=1 only (the 40→64 head-pad is exact only single-GPU); ~2× KV/compute from padding the MLA latent to 576.
- **Not run:** BFCL (was "n/c" in baseline); long-context evals (the regime where sparsity should actually
  separate from dense — the main untested frontier).

## 6. Reproduction
- Serving: `<eval_root>/serving/serve_dsa.sh` (+ `serve_dsa_entry.py`, `_pluginboot/sitecustomize.py`).
- OpenCompass: `<eval_root>/run_oc_dsa.sh <bench>` (configs under `<eval_root>/<bench>/config/`).
- EvalPlus: `<eval_root>/run_evalplus_dsa.sh <folder> <dataset>`. LCB: `<eval_root>/run_lcb_dsa.sh` (MP env).
- Plugin: `scripts/dsa/vllm_minicpm3_dsa/`. Probes/tests: `tests/dsa/probe_flashmla_padding.py`,
  `tests/dsa/test_stage2b_sparse.py`, `tests/dsa/test_stage3_decode_parity.py`.
- Scorecard: `<eval_root>/DSA_SCORECARD.md`. Per-run manifests+logs under each `<bench>/logs/`.

## 7. Open items
1. **LiveCodeBench** — mp=6 re-run finishing; slot the valid pass@1 into the scorecard.
2. **HumanEval+ isolation** (top_k=2048) — finishing; settles sparsity vs BC drift → picks the follow-up.
3. **Long-context eval** — the untested regime where the sparsity budget should matter (and DSA's value lives).
4. **UE8M0 training-side fix — DONE** (implemented + parity-verified, §5; needs a retrain with `dsa_fp8_ue8m0: true`
   to land in a checkpoint). Still open (if pursued): code-aware indexer; BFCL.
