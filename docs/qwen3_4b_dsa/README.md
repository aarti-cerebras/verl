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

## Dense prefix (`dense_prefix`, added 2026-08-25)

Layers `[0, dense_prefix)` keep **stock full attention**: no indexer, no KL term, and in Phase 2 they
still run dense while the rest go sparse. The launch scripts default to **`DENSE_PREFIX=4`**.

**Why 4.** Per-layer `topk_recall` from the `lr1e-4` Phase-1 run at step ~4150:

| L00 | L01 | L02 | L03 | L04 | L05 | L07+ | L23–L35 |
|---|---|---|---|---|---|---|---|
| 0.791 | 0.790 | 0.816 | 0.876 | 0.903 | 0.921 | ≥0.92 | 0.95–0.97 |

The first four layers are the clear outliers — early layers attend broadly and are hardest to
sparsify. Same shape MSA/M3 found (`msa_indexer.dense_prefix=3`; M3 ships
`sparse_attention_freq = [0]*3 + [1]*57`). A second, independent signal agrees: the
`input_layernorm` RMS spread across layers is **186x** over all 36 layers but only **38x** over
layers 4–35, i.e. the dense prefix is exactly the extreme-scale layers the §2.4 per-layer init has to
work hardest to absorb.

32/36 layers stay sparse, so this costs roughly 11% of the attention savings.

**It is architecture, not a training preference.** It must match at train and serve time or the model
runs dense where it was trained sparse — a silent quality regression with no other symptom. Hence:

- it is in the `CONFIG_TAG` (`_dp4`), so `resume_mode=auto` cannot cross a 36-indexer checkpoint with
  a 32-indexer model;
- `build_qwen3_dsa_serving_dir.py` writes the **resolved** id list to `config.json` as
  `dsa_sparse_layer_ids`, and refuses a checkpoint whose indexer tensors disagree with it *in either
  direction* (missing on a sparse layer, or present on a dense one);
- the vLLM plugin's `sparse_layer_ids()` only **reads** that key — it never re-derives the predicate.

**`DENSE_PREFIX=0` is fully backward compatible**: every layer sparse (the DeepSeek-V3.2 recipe), and
it reproduces the pre-`dense_prefix` `CONFIG_TAG` byte-for-byte, which is what the existing
`p1_*_lr3e-4` / `p1_*_lr1e-4` checkpoints need in order to resume. Serving dirs built before this
change have no `dsa_sparse_layer_ids` key and are treated as all-sparse.

`SPARSE_LAYERS="4,9,17"` sets an arbitrary layer set instead of a prefix (tagged by hash, not count,
so two different sets can never share a checkpoint dir). Setting both is an error.
