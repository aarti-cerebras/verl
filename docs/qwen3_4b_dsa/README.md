# Qwen3-4B → DSA (GQA, no MLA) — **SUPERSEDED**

> **Superseded by [`docs/qwen3_4b_msa/`](../qwen3_4b_msa/) (2026-07-28).** We moved from
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

**Prior art in this repo** (MiniCPM3-4B, MLA + DSA): `docs/dsa_minicpm3_plan.md`,
`docs/dsa_phase2_plan.md`, `docs/dsa_eval_report.md`, `docs/dsa_vllm_serving.md`.
Read `docs/dsa_eval_report.md` first — its §2 (sparsity verification) and §3 (drift-vs-sparsity
ladder) are the methodology this plan generalizes.

**Status:** planning only. Nothing implemented as of 2026-07-27.
