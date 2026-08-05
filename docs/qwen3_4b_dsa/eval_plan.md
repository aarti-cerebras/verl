# Qwen3-4B DSA — long-context evaluation plan

**Goal:** prove that the DSA-sparse **Qwen3-4B-Thinking-2507** preserves the dense model's
long-context and reasoning capability at **32K**, and attribute any regression to a specific cause.

**Decisions (2026-07-27):**
- **RULER effective context length** is the headline metric. Everything else supports it.
- **Paired comparison** across ladder rows with identical samples/seeds — the sparsity cost is
  ~1–2 pts, which independent-sample noise would swamp.
- **No YaRN, no lengths above 32K.** The model is natively 262144, so 4K–32K is entirely in range.
- Reuse the one-server / many-clients architecture from `docs/minicpm3_eval_plan.md`.

Read `docs/dsa_eval_report.md` first — its §2 (verify the model is *genuinely* sparse) and §3
(drift-vs-sparsity ladder) are the methodology this plan generalizes to long context.

---

## 1. Why the existing scorecard cannot answer the question

The MiniCPM3 DSA scorecard is entirely ~1K-token prompts. What determines sparsity damage is the
**selection ratio k/L**, not `k`:

| top_k | 1K | 4K | 8K | 32K |
|---|--:|--:|--:|--:|
| 256 | 25% | 6.3% | 3.1% | 0.8% |
| 512 | 50% | 12.5% | 6.3% | 1.6% |
| **2048** | 100% (dense) | 50% | 25% | **6.25%** |

At 1K/256 the model keeps a quarter of the context — which is why those numbers came in at dense
parity. **Sparsity damage is length-dependent by construction**, so every result here is a *curve*
over length, never a single number.

With thinking mode there is a second axis: `L` grows during generation while `k` is fixed, so the
ratio degrades *within* a single response (32K prompt + 12K trace → 6.25% → 4.6%).

---

## 2. Benchmarks

### Tier 1 — the spine (every ladder row, every length)

- **RULER** at **4K / 8K / 16K / 32K**. 13 synthetic tasks: NIAH variants (multi-key, multi-value,
  multi-query), variable tracking (multi-hop), common/frequent-word aggregation, QA. Yields
  **effective context length** = longest length still above the fixed 85.6% threshold (Llama-2-7B
  @ 4K), which is exactly the "did we preserve long context" scalar. Supports OpenAI-compatible
  endpoints → drops into the one-server pattern.
- **NIAH depth × length heatmap.** Cheap (~30 min) smoke test. Necessary, not sufficient — it
  saturates and does not predict downstream performance.

### Tier 2 — the benchmarks that actually stress a top-k selector

- **NoLiMa** — NIAH with *minimal lexical overlap* between question and needle. The single most
  diagnostic benchmark for a lightning indexer, because the indexer scores keys by learned
  similarity: lexical-overlap needles are easy for it, latent-association needles are where a weak
  indexer fails. Expect low absolute scores (11 of 13 frontier models drop below 50% of their short
  baseline at 32K) — compare **only** against our own dense baseline.
- **RULER MK (multi-key with distractors)** — per HELMET, the harder recall tasks are the synthetic
  ones that actually track real-world performance.
- **HELMET** at 8K–32K, especially **summarization** (requires aggregating across the *whole* input —
  a top-k selector that keeps only salient spans should degrade here first) and **ICL with many
  examples**. Neither correlates with the other HELMET categories, so they catch failures RULER
  misses.
- **LongBench v2** for a realistic multiple-choice check.

### Tier 3 — reasoning (new for a thinking model)

- **AIME-style math**, **GPQA**, **LiveCodeBench v6**. These are short-prompt but long-*generation*,
  so they exercise decode-side sparsity over self-generated CoT — the dominant regime (§1.1 of
  [plan.md](plan.md)).
- **Long code / repo-level** (RepoQA or similar) if in scope. Code was the most sparsity-sensitive
  domain in the MiniCPM3 work.

### Tier 4 — short-context regression suite

Re-run the existing pipeline unchanged: MMLU, CMMLU, CEval, GSM8K, MATH, IFEval, HumanEval+, MBPP+.
**Non-negotiable** — the most common way to "fix" long context is to break short context, and only
the paired comparison catches it.

Long-document perplexity (PG19 / proof-pile) is worth logging as a cheap continuous monitor but
**must not be a gate** — it is a poor proxy for retrieval-style long-context ability.

---

## 3. The de-confounding ladder

Each row changes exactly one variable. Because there is **no MLA conversion**, there is only **one
confound** — the biggest methodological win of the GQA-direct approach.

| # | Config | weights | attention | Expectation / purpose |
|---|---|---|---|---|
| 0 | stock Qwen3-4B-Thinking-2507 | stock | dense GQA | Reference curve. **Reproduce a published number first**, or every Δ below is meaningless. |
| 2 | DSA, `top_k ≥ L` | Phase-2 | degenerates to dense | **Faithfulness control.** Proves the sparse code path is correct. Must match row 0/1 within noise at every length. |
| 3 | DSA, `top_k = 2048` | Phase-2 | sparse | The real config. `Δ(L) = row3(L) − row2(L)` is the reported sparsity cost. |
| 4 | DSA, **random indexer**, `top_k = 2048` | Phase-2 | sparse | Must **collapse**. If it scores like row 3, the model is silently dense. |
| 5 | DSA, random indexer, `top_k ≥ L` | Phase-2 | dense-equivalent | Must **recover**. Confirms row 4's collapse came from *selection*, not from broken machinery. |

**Under Phase 2a (base frozen), rows 0 and 2 collapse into one config** — the Phase-2a weights *are*
the stock weights, so there is literally zero weight drift to isolate. This is a large part of 2a's
appeal: the ladder shrinks and `Δ(L)` becomes an unconfounded measurement of sparsity alone.

Rows 4/5 reuse `scripts/dsa/randomize_indexer_ckpt.py`. **Re-verify the `index_topk` config gate at
every length** — a freshly-built serving dir silently serves dense (`docs/dsa_eval_report.md` §2),
and at long context the resulting good scores would be misread as success.

> **The failure signature is not a large Δ at one length — it is a Δ that grows monotonically with
> L.** A flat −1 pt across 4K→32K is a fine sparse model. −0.5 at 4K, −2 at 8K, −6 at 32K means the
> indexer is not scaling, and no amount of downstream tuning fixes it.

---

## 4. Length × top_k grid

Run a RULER subset over `L ∈ {4K, 8K, 16K, 32K} × top_k ∈ {512, 1024, 2048, ≥L}`.

This yields the **deployable budget as a function of length**, which is the actual engineering
output, and enables compute-matched rather than arbitrary comparisons. DeepSeek-V3.2 ships
`index_topk=2048`; do not assume a smaller budget transfers from the MiniCPM3 runs, which operated
at ~1K contexts.

---

## 5. Thinking-mode protocol (differs from the MiniCPM3 methodology)

`Qwen3-4B-Thinking-2507/generation_config.json`: `temperature 0.6, top_p 0.95, top_k 20,
do_sample: true`, eos `[151645, 151643]`.

1. **Greedy is gone.** The entire MiniCPM3 protocol was temperature-0. Sampled decoding makes
   run-to-run variance nonzero, so the "Δ ≤ 2 pts" gate needs error bars to mean anything.
2. **Fix seeds; use identical samples and sample counts across every ladder row**; report **paired**
   per-item deltas. Paired discipline goes from nice-to-have to load-bearing.
3. **Multi-sample scoring.** Qwen reports avg@64 on AIME for these models; that is unaffordable
   across 6 rows × 4 lengths. Pick a fixed **n = 4–8**, use it everywhere, and state it.
4. **`max_new_tokens` generously large**, and **log the truncation rate as a first-class metric in
   every run manifest.** A truncated thinking trace scores zero and looks exactly like a model
   regression — the same failure class as the LiveCodeBench retry-to-empty cascade
   (`docs/dsa_eval_report.md` §4). **Any comparison where rows differ in truncation rate is
   invalid.**
5. **Total sequence = prompt + trace.** A 32K RULER prompt plus a 12K trace is L=44K at the end of
   generation. Well inside the 262144 native range, but it means the *effective* selection ratio at
   end-of-trace is materially lower than the nominal k/L at the prompt length. Log realized
   prompt+gen length distributions per benchmark, as the MiniCPM3 report did.

---

## 6. Intrinsic indexer probes — the primary signal

`indexer/topk_recall` (fraction of dense attention **mass** landing on the indexer's top-k) and
`topk_overlap` already exist (`verl/models/transformers/minicpm_dsa.py:199`,
`docs/dsa_indexer_metrics.md`). Repurpose them as an **offline length-swept probe**, not just a
training scalar. This is the cheapest and most diagnostic signal available: a handful of forward
passes instead of GPU-days, and it is the only measurement that explains *why* a benchmark moved.

Report, on held-out long documents and on held-out thinking traces:

1. **Recall vs. length** at 4K/8K/16K/32K, per layer (mean **and min** — the mean hides one broken
   layer). `≥ 0.90` at the target length is close to a sufficient condition for preserved
   long context.
2. **Recall vs. query position** and **vs. key distance**.
3. **Recall vs. generated-trace position** — the thinking-mode ratio-degradation axis (§1).
4. **Recall vs. needle depth** on NIAH-style inputs, not just the mean.
5. **Selected-key position histogram vs. dense attention argmax.** The characteristic failure of an
   indexer trained on short sequences is degenerating into "local window + attention sinks" —
   recency bias that looks fine at 4K and is fatal at 32K. This is visible long before any benchmark
   run.
6. **The four-point granularity table** from [plan.md](plan.md) §4.2 (token/per-query,
   block-64/per-query, block-64/q-128, block-64/q-32).

Tooling: `scripts/dsa/probe_block_selection.py` (see [plan.md](plan.md) §5.1).

---

## 7. Acceptance gates

Write these down before running anything.

1. **RULER effective context length does not shrink.** Row 3 ≥ row 0 (and ≥ row 2). This is the
   headline claim.
2. **`Δ_sparsity(L) ≤ ~2 pts` at every length, and non-increasing in L.** A growing gap fails the
   gate even if all absolute numbers look acceptable.
3. **Faithfulness control passes:** row 2 ≈ row 0/1 within noise at every length. If not, it is a
   kernel/masking bug, not a model result — stop and fix.
4. **Random-indexer control:** row 4 collapses, row 5 recovers.
5. **`indexer/topk_recall ≥ 0.90`** at 32K, all layers ≥ 0.80; no recency collapse in the position
   histogram.
6. **NIAH ≥ 95%** at all depths up to 32K (necessary only).
7. **NoLiMa and HELMET-summarization** degrade no worse, in *relative* terms, than the dense model's
   own curve.
8. **Short-context suite within noise** of the dense baseline, paired.
9. **Trace integrity** (thinking-mode specific): `</think>` closure rate, trace-length distribution,
   and repetition/degeneration rate all within noise of the dense baseline. Accuracy alone masks
   sparse-over-own-CoT failures.
10. **Truncation rate logged and comparable across rows** (§5.4).

All accuracy gates carry error bars from fixed-n sampled decoding.

---

## 8. Cost and sequencing

The dominant schedule risk. Thinking traces are 10–50× the output tokens of a non-thinking model,
multiplied by n samples, 6 ladder rows, and 4 lengths — on a sparse decode path.

- **Validate the entire harness at 4K–8K first**, on Qwen3-1.7B if possible, before spending 32K
  wall-clock.
- **Get CUDA graphs working** via the paged block-sparse path (`Bk = 64` == vLLM `--block-size 64`,
  [plan.md](plan.md) §4.2) rather than repeating the MiniCPM3 `enforce_eager` situation. This is the
  difference between hours and days per row.
- Subsample RULER cells if needed (100 instead of 500 per task per length) but keep the subsample
  **identical across rows**, and **`log()` what was dropped** — silent truncation of coverage reads
  as "we tested everything" when we didn't.
- Ordering: NIAH heatmap → RULER 4K/8K (harness validation) → RULER 16K/32K → NoLiMa → HELMET →
  Tier 3 reasoning → Tier 4 short-context regression.

---

## 9. Reproduction conventions

Follow `docs/dsa_eval_report.md` §6:

- One persistent vLLM server per config; all harnesses hit the same OpenAI-compatible endpoint so
  chat template and tokenization are identical everywhere.
- Eval root per config under `/cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking-dsa-*/`, with a
  one-page `DSA_SCORECARD.md`.
- Per-run manifests + logs under `<eval_root>/<bench>/logs/`, including the **full launch command
  and consumed env**, sampling params, seeds, sample count, `index_topk`, truncation rate, and the
  realized prompt+gen length histogram.
- Confirm in the serve log that the sparse backend was actually selected — not the dense fallback.
