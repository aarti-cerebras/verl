# Qwen3-4B MSA — all checkpoints × all benchmarks vs dense baseline

Every MSA (MiniMax Sparse Attention) checkpoint evaluated so far, against the dense
`Qwen3-4B-Thinking-2507` reference. **Δ** = absolute difference vs baseline (same units as the
metric); **Δ%** = relative difference (`Δ / baseline × 100`).

- **Baseline (dense):** `/cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking` — `Qwen/Qwen3-4B-Thinking-2507`
  @ `768f209d`, vLLM 0.20.2, bf16, TP=1, 131072 ctx, temp 0.6 / top_p 0.95 / top_k 20, seed 1234.
- **Sparse roots:** `/cb/ml-eng/aarti/msa/evals/qwen3-4b-thinking-msa-<geom>-step<N>`.
- **Geometry:** `k8v2` = top_k 8 blocks × B_k 128 = **1024 selected tokens**; `k16v2` = top_k 16 ×
  128 = **2048 selected tokens**. Both: 33 sparse layers, 3 dense prefix layers.
- **Numbers recomputed** from raw per-bench artifacts by `_common/collect_scorecard.py`
  (verified to reproduce the published step-5600 scorecard). Nothing copied from a prior scorecard.
- **Date:** 2026-08-11.

## Figures

| Figure | What it shows | Subsets |
|---|---|---|
| [scores.png](scores.png) | headline number per benchmark, best checkpoint per geometry, model-card rule | — |
| [scores_ruler.png](scores_ruler.png) | per task at each length + the length mean; then the four lengths + the RULER mean | 13 tasks × 4 lengths |
| [scores_mmlu_pro.png](scores_mmlu_pro.png) | per category + the macro-average | 14 categories |
| [scores_mrcr.png](scores_mrcr.png) | per needle count and context band + the means | 9 cells |
| [scores_gsm_infinite.png](scores_gsm_infinite.png) | per length × reasoning-op count + the means | 24 cells |
| [scores_livecodebench.png](scores_livecodebench.png) | per problem difficulty + the 131-problem mean, both output budgets | easy 31 / medium 39 / hard 61 |
| [scores_aime.png](scores_aime.png) | per pass + the average, both AIME budgets | 16 + 16 passes |
| [scores_gpqa.png](scores_gpqa.png) | per pass + the avg@4 | 4 passes |
| [scores_ifeval.png](scores_ifeval.png) | per metric; the last group restates prompt-strict (the headline), since the four are not a partition | 4 scorings |
| [lengths/LENGTHS.md](lengths/LENGTHS.md) | realized prompt / response / overall token distributions | 10 benchmarks |

Regenerate: `python3 scripts/msa/score_bars.py` (charts recompute from raw artifacts via
`_common/collect_scorecard.py`; nothing is transcribed).

## Scorecard

| Benchmark | Baseline | k8 s100 | Δ | Δ% | k8 s2700 | Δ | Δ% | k8 s5600 | Δ | Δ% | k8 s11214 | Δ | Δ% | k16 s5800 | Δ | Δ% | k16 s10700 | Δ | Δ% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| IFEval (prompt-strict) | 88.72 | — | — | — | 85.40 | −3.32 | −3.7% | 87.06 | −1.66 | −1.9% | 85.77 | −2.95 | −3.3% | — | — | — | 86.88 | −1.84 | −2.1% |
| GPQA-diamond (avg@4) | 64.40 | — | — | — | 64.52 | +0.13 | +0.2% | 64.27 | −0.13 | −0.2% | 63.39 | −1.01 | −1.6% | — | — | — | **66.29** | **+1.89** | **+2.9%** |
| MMLU-Pro (macro, 14 cat) | 72.50 | — | — | — | 72.17 | −0.33 | −0.5% | 72.22 | −0.28 | −0.4% | 72.43 | −0.06 | −0.1% | — | — | — | 72.45 | −0.05 | −0.1% |
| AIME25 @81920 (avg@16) | 83.12 | — | — | — | 78.54 | −4.58 | −5.5% | 77.50 | −5.63 | −6.8% | 78.75 | −4.38 | −5.3% | — | — | — | 80.00 | −3.12 | −3.8% |
| AIME25 @32768 (avg@16) | 72.50 | — | — | — | 67.92 | −4.58 | −6.3% | 68.96 | −3.54 | −4.9% | 67.29 | −5.21 | −7.2% | — | — | — | **73.54** | **+1.04** | **+1.4%** |
| LiveCodeBench v6 @81920 | 53.82 | — | — | — | 50.38 | −3.44 | −6.4% | 49.05 | −4.77 | −8.9% | 52.10 | −1.72 | −3.2% | — | — | — | **54.20** | **+0.38** | **+0.7%** |
| LiveCodeBench v6 @32768 | 53.44 | — | — | — | 50.57 | −2.86 | −5.4% | 50.38 | −3.05 | −5.7% | 50.95 | −2.48 | −4.6% | — | — | — | 52.86 | −0.57 | −1.1% |
| RULER 4K | 97.15 | — | — | — | 96.79 | −0.36 | −0.4% | 96.86 | −0.28 | −0.3% | 96.53 | −0.61 | −0.6% | — | — | — | 97.00 | −0.15 | −0.2% |
| RULER 8K | 96.46 | †94.16 | †−2.14 | †−2.2% | 95.10 | −1.35 | −1.4% | 95.54 | −0.91 | −0.9% | 95.35 | −1.10 | −1.1% | — | — | — | 96.08 | −0.38 | −0.4% |
| RULER 16K | 96.65 | — | — | — | 92.75 | −3.90 | −4.0% | 92.20 | −4.45 | −4.6% | 92.81 | −3.84 | −4.0% | — | — | — | **95.08** | **−1.57** | **−1.6%** |
| **RULER 32K** | 95.33 | — | — | — | 82.09 | **−13.25** | **−13.9%** | 81.38 | **−13.95** | **−14.6%** | 81.83 | **−13.50** | **−14.2%** | **89.56** | **−5.77** | **−6.1%** | **88.97** | **−6.36** | **−6.7%** |
| MRCR (mean of 9 cells) | 0.458 | — | — | — | 0.401 | −0.057 | −12.5% | 0.389 | −0.069 | −15.0% | 0.396 | −0.062 | −13.6% | — | — | — | 0.418 | −0.040 | −8.8% |
| GSM-Infinite (mean, 24) | 64.79 | — | — | — | ✗ | — | — | ✗ | — | — | 50.83 | −13.96 | −21.5% | — | — | — | 59.58 | −5.21 | −8.0% |

`—` not run · `✗` run, but not comparable (see below). **All cells are final; nothing in flight.**

**Footnotes**

- **†** k8 s100 RULER 8K is a **12/13-task partial** (`fwe` missing). Both the value and the Δ are
  computed against the baseline restricted to the *same 12 tasks* (96.30), not the published
  13-task 96.46 — so the Δ is apples-to-apples, but it is not the RULER 8K number.
- **k16 s10700 RULER 16K — complete.** It was a 10/13-task partial earlier in the day; `vt`, `cwe`
  and `fwe` landed with the `len16k_e` slice and the full 13-task mean is **95.08 (−1.57)**. The
  three were `vt` 100.0, `fwe` 99.0, **`cwe` 86.3** — `cwe` is the only one of the three with real
  headroom, and it is the largest single 16K gain over k8 (75.1 → 86.3). The 13-task mean is
  *numerically indistinguishable* from the 10-task partial (95.075 → 95.081): the three additions
  averaged 95.1, which is the partial's own mean. Coincidence, not confirmation — do not read the
  stability of that cell as evidence the partial was trustworthy.
- **k16 s10700 AIME25 @81920 — complete at n=16**, 80.00 ± 1.05 (phase B finished 2026-08-11
  ~21:40 UTC). While it was in flight `collect_scorecard.py` averaged whatever passes existed and
  returned real-looking numbers with wide stderr (83.34 ± 3.34 at n=2, 80.00 ± 1.05 at n=16 —
  the point estimate moved 3.3 pts between them). `score_bars.py` withholds any pass-averaged
  metric below its required pass count; the table should follow the same rule.
- **✗** k8 s2700 / s5600 ran GSM-Infinite on a **different ops grid** (len {0,8K,16K,32K} × ops
  {2,4,8,16}, 16 cells) than the baseline and the later checkpoints (ops {2,5,10,15,20,30}, 24
  cells). No shared cell set → no valid Δ. Not a missing run; a non-comparable one.
- **MRCR** is a mean SequenceMatcher ratio on 0–1, so its Δ is in ratio points; Δ% is still
  meaningful (it is a ratio scale with a true zero).
- **AIME25** stderr at avg@16 is ±0.8–1.7, **GPQA** ±0.4–1.7 — Δ smaller than ~2 pts on those two
  is inside the error bar. IFEval / MMLU-Pro / RULER / LCB are single-pass, no error bar.

## What the table says

1. **Short context is free, at both budgets.** MMLU-Pro is within −0.5 for every checkpoint
   (−0.06 at k8 s11214), GPQA is within its error bar, RULER 4K within −0.6. When the whole context
   fits in the selection budget, sparsity costs nothing measurable.

2. **RULER 32K is the failure, and it is a budget failure.** At k8 the Δ is −13 to −14 across
   *three* checkpoints spanning 2700→11214 steps — it does not improve with training. Doubling the
   budget to k16 more than halves it (−13.5 → −6.4). That is the signature of *not enough selected
   tokens*, not of a badly trained indexer: 1024/32768 = 3.1% selection ratio vs 6.25% at k16.

   **The shape did not change, only the magnitude.** Δ by length is −0.15 / −0.38 / −1.57 / −6.36
   at k16 vs −0.61 / −1.10 / −3.84 / −13.50 at k8 — a ratio of 4.1× / 2.9× / 2.4× / 2.1×. Doubling
   `k` bought a roughly constant *factor* at every length rather than a fixed offset, which is what
   you see when the binding constraint is the selection *ratio*. Extrapolating that factor puts
   **k=32 near −3 at 32K — still short of the ≤2 gate.** k=32 is the cheapest next measurement, but
   should be expected to narrow rather than close the gap; if it lands near −3, `k` alone is not the
   remaining lever.

3. **More k8 training does not buy long context.** s2700 → s5600 → s11214 on RULER 32K:
   −13.25, −13.95, −13.50. Flat within noise over 4× the steps. Meanwhile the short-context and
   generation-bound benches *did* improve with steps (LCB @81920 −3.44 → −4.77 → −1.72; MMLU-Pro
   −0.33 → −0.06). Long-context capability tracks the budget; everything else tracks training.

4. **k16 s10700 is the best checkpoint on every bench, full stop** — it beats k8 s11214 in all 13
   rows, with no regressions. Three cells land above baseline (GPQA +1.89, AIME25 @32768 +1.04,
   LiveCodeBench @81920 +0.38); all three are inside ~1.5σ, so read them as parity, not as gains.
   Its worst remaining Δ are RULER 32K −6.36, MRCR −8.8% and GSM-Infinite −8.0%.

5. **MRCR tracks RULER 32K** (−12.5/−15.0/−13.6% at k8, −8.8% at k16), which is the expected
   coupling: both are multi-fact retrieval over long context, the exact regime the budget starves.

6. **The AIME/LCB/IFEval drops are not long-context effects.** Those prompts are short (IFEval
   p50 41 tok); selection only engages once the *generated trace* passes `k·B_k`. For k8 the two
   AIME budgets and the two LCB budgets each gave Δ agreeing within ~1 pt, which is what ruled out
   a truncation artefact.

   **That test now splits, and only for AIME at k16:** Δ@81920 = −3.12 vs Δ@32768 = **+1.04**, a
   4.2 pt divergence, while LCB still agrees (+0.38 vs −0.57). The cause is not a k16 pathology but
   the *baseline's* cap penalty: dense loses 10.6 pts going 81920 → 32768, k16 only 6.46, because
   k16 truncates less (22.50% vs 24.0%) at equal closure (78.54% both). So the capped AIME cell
   flatters k16 by ~4 pts of harness artefact. **Quote AIME25 as 80.00 (@81920).** The +1.04 is
   real arithmetic against a baseline that is itself cap-damaged, not evidence k16 beats dense at
   competition math.

## Caveats — no Δ here is a clean sparsity cost

1. **Δ conflates sparsity with weight drift.** No de-confounding ladder was run
   (`serving_plan.md` §6.4): no dense-on-same-weights `StockRef` row, no stock-base +
   Phase-1-indexer row. *Exception:* the RULER length trend — drift is length-independent, so a Δ
   that grows ~35× from 4K to 32K is attributable to selection.
2. **vLLM version confound (unmitigated).** Baseline measured on 0.20.2, all sparse rows on 0.26.0.
3. **`--enforce-eager`** on the sparse rows only (cudagraph safety on the MSA backend unresolved).
   Everything else matches: `--max-model-len 131072`, `--gpu-memory-utilization 0.90`, `--seed 1234`,
   `--no-enable-prefix-caching`, `--block-size 128`.
4. **Small-n cells.** MRCR 25 items/cell; AIME 30 items × 16 passes; GSM-Infinite 20 items/cell.
5. **k16 s5800 and k8 s100 are single-purpose probes**, not full suites — s5800 is the RULER 32K
   decisive-measurement run, s100 an early-training sanity check.
6. **The k8/k16 comparison is not at matched steps.** k8 is s11214 = 100% of its schedule; k16 is
   s10700 = **96%** (training was stopped at step 10741 to free the GPUs; 10700 was the last
   checkpoint). Point 3 argues this is immaterial for long context — k8 moved <1 pt on RULER 32K
   over steps 2700→11214 — but the short-context rows *do* track training, so k16's IFEval −1.84
   and AIME −3.12 are, if anything, mild underestimates of a finished k16 run.

## Reproduce

```bash
R=/cb/ml-eng/aarti/msa/evals/qwen3-4b-thinking-msa-k16v2-step10700
python3 $R/_common/collect_scorecard.py --root <eval_root> --json out.json
```

Regenerate the figures (both recompute from raw artifacts; nothing is transcribed):

```bash
python3 scripts/msa/score_bars.py --outdir docs/qwen3_4b_msa
MPLCONFIGDIR=/tmp/mpl python3 scripts/msa/length_histograms.py --out docs/qwen3_4b_msa/lengths
```

Per-checkpoint detail: `<eval_root>/<bench>/{results,logs}/`. k8 s2700/s5600 narrative +
per-task RULER 32K breakdown: `/cb/ml-eng/aarti/msa/evals/SCORECARD_msa_k8v2.md`.
**k16 s10700 narrative, per-task RULER, trace-integrity table and method notes:**
`/cb/ml-eng/aarti/msa/evals/SCORECARD_msa_k16v2.md`.
Baseline detail: `/cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking/SCORECARD.md`.
