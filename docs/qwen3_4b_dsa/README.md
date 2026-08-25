# Qwen3-4B → DSA (GQA, no MLA)

> **ACTIVE PLAN: [`plan_v2.md`](plan_v2.md) (2026-08-14).** Token-granular DSA on GQA, revisited with
> MSA's measured result in hand (k16 still loses RULER 32K by 6.36) and its plumbing available for reuse.
> `plan_v2.md` supersedes `plan.md` below; read v2, not v1.

> **v1 was superseded by [`docs/qwen3_4b_msa/`](../qwen3_4b_msa/) (2026-07-28).** We moved from
> token-granular DSA-on-GQA to **MiniMax Sparse Attention (MSA)**, which is designed for GQA and
> already has a merged vLLM implementation with a Triton path that runs on H100. Rationale:
> [`qwen3_4b_msa/plan.md`](../qwen3_4b_msa/plan.md) §2.
>
> **Still current and referenced by the MSA plan:** [`data_plan.md`](data_plan.md) (unchanged) and
> [`eval_plan.md`](eval_plan.md) (with the substitutions in `qwen3_4b_msa/plan.md` §9).
> `plan.md` here is kept for the reasoning trail only — do not implement from it.

Adapting **Qwen3-4B-Thinking-2507** from dense GQA to DeepSeek Sparse Attention, **without**
introducing MLA. Target: sparse attention that works at **32K** and demonstrably preserves the
model's long-context and reasoning capability.

| Doc | Contents |
|---|---|
| [plan.md](plan.md) | Target-model choice, architecture deltas vs. the MiniCPM3 MLA port, indexer sizing, Phase 0 / 1 / 2a / 2b, reuse map, compute budget, risks |
| [data_plan.md](data_plan.md) | Prompts-only/on-policy rule (from the Qwen3-235B draft-head survey), difficulty-not-length selection criterion, dataset evaluation, the three data streams, mixture, filtering, **§11 known gap: no instruction-shaped prefill-long data** |
| [eval_plan.md](eval_plan.md) | Long-context benchmark selection, the de-confounding ladder, length × top_k grid, intrinsic indexer probes, thinking-mode eval protocol, acceptance gates |
| [serving_bringup_results.md](serving_bringup_results.md) | **P0-P4 execution log (2026-08-20): what was built, the token-exact ladder, throughput (1.60x at 21K/4096/conc16), and the five silent failure modes found on the way** |
| [serving_eval_plan.md](serving_eval_plan.md) | **vLLM serving + how the evaluation actually gets run (2026-08-20).** Executes plan_v2 §5: the reuse ledger (every kernel reused, ~350 lines of non-MLA plumbing written), the two still-free training-side geometry decisions, phases P0–P6 with gates, and the DSA-vs-MSA equal-KV-budget comparison |

**Prior art in this repo** (MiniCPM3-4B, MLA + DSA): `docs/dsa_minicpm3_plan.md`,
`docs/dsa_phase2_plan.md`, `docs/dsa_eval_report.md`, `docs/dsa_vllm_serving.md`.
Read `docs/dsa_eval_report.md` first — its §2 (sparsity verification) and §3 (drift-vs-sparsity
ladder) are the methodology this plan generalizes.

**Status (2026-08-20):** training code landed (commit `211c1813`) but **no Qwen3 DSA run has started**.
Serving/eval is planned in [serving_eval_plan.md](serving_eval_plan.md); ~6 days of it (P0–P2) needs no
checkpoint, and P0 (the FA3 page-size-1 GQA spike) gates the whole approach.
