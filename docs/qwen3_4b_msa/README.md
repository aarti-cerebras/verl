# Qwen3-4B → MSA (MiniMax Sparse Attention)

Adapting **Qwen3-4B-Thinking-2507** from dense GQA to **MSA**: a per-GQA-group Index Branch scores
128-token KV blocks, top-16 are selected, and the Main Branch runs exact block-sparse attention over
only those blocks. Target **32K**, preserving long-context and reasoning capability.

| Doc | Contents |
|---|---|
| **[plan.md](plan.md)** | Target model, why MSA over DSA-on-GQA, config, cost accounting, Phase 0 / 1 / 2a / 2b, **serving plan**, compute budget, risks |
| **[kl_loss.md](kl_loss.md)** | The loss spec: paper Eq. 9–12 verbatim, worked example, shapes and memory, the paper-vs-vLLM divergences, metrics, provenance ledger |
| **[phase1_implementation.md](phase1_implementation.md)** | Phase 1 as built: notation, the 34.4 TB/layer problem statement, the seven optimisations that bring it to a measured 46.9 GB and 42 s/step, the measured budget that replaces plan §8's estimate, and **§8: every logged metric defined with its equation** |
| **[phase2_plan.md](phase2_plan.md)** | Phase 2 (sparse): the per-token gather decision, the free teacher, the `λ` conversion trap, gradient wiring, code list, 2a/2b gates, Phase-2-specific risks |
| [../qwen3_4b_dsa/data_plan.md](../qwen3_4b_dsa/data_plan.md) | **Carries over unchanged** — prompts-only/on-policy rule, difficulty-not-length selection, dataset evaluation, three data streams, §11 known gap |
| **[phase2_data_gen.md](phase2_data_gen.md)** | The data plan made concrete for the S2/S3 (short + decode-long) halves: `allenai/Dolci-Think-RL-32B` as the prompt bank (verified schema, slice policy, IFEval-format caveat), Qwen3-Thinking sampling config, **the `<think>`-stripping landmine in `MultiTurnSFTDataset` and the `input_ids`+`loss_mask` contract that avoids it**, filters, decontamination, budget, pilot gate, commands |
| [../qwen3_4b_dsa/eval_plan.md](../qwen3_4b_dsa/eval_plan.md) | **Carries over** with the substitutions listed in [plan.md](plan.md) §9 |

## Config, in one place

```python
# config.json -> sparse_attention_config
sparse_index_dim = 128 ; sparse_num_index_heads = 8 ; sparse_topk_blocks = 16
sparse_block_size = 128 ; sparse_init_block = 0 ; sparse_local_block = 1
sparse_score_type = "max" ; sparse_attention_freq = [0]*3 + [1]*33
use_sparse_attention = True
# config.json, top level -- required by the M3 layer's get_rope() call:
partial_rotary_factor = 1.0 ; head_dim = 128
# vLLM ENGINE option, not model config (and bf16 is FORCED on SM90):
#   --attention-config '{"indexer_kv_dtype": "bf16"}'
# --block-size auto-resolves to 128; passing it is optional (a user-set 16 is rejected, not misaligned)
```

**Index norms are Gemma-style** (`x·rsqrt(mean(x²)+eps)·(1+w)`, one shared `[128]` gain per branch) —
train them in that form. See [plan.md](plan.md) §3.2.

## Key references

- **Paper:** MiniMax Sparse Attention, arXiv [2606.13392](https://arxiv.org/abs/2606.13392), local
  copy `/cb/ml-eng/aarti/msa/refs/msa_2606.13392.pdf` — §3.2 training (Eq. 9–11, Algorithm 1), §3.3
  complexity (Eq. 12), §4 kernels, §5.1–5.4 experiments (**MSA-CPT** is our setting), §5.2 metrics,
  appendices B/C (ablations that justify the recipe).
- **vLLM implementation:** `vllm/models/minimax_m3/` (PR #45381, merged 2026-06-15). Triton path runs
  on SM90; the MiniMax CuTe kernel ([MiniMax-AI/MSA](https://github.com/MiniMax-AI/MSA), MIT) is
  **SM100-only, prefill-only, and not vendored in vLLM**. Note the working container ships vLLM
  **0.20.2**, which predates the whole `vllm/models/` layout — the upgrade is a prerequisite
  ([plan.md](plan.md) §7).
- **Prior art in this repo** (MiniCPM3-4B, MLA + DSA): `docs/dsa_minicpm3_plan.md`,
  `docs/dsa_phase2_plan.md`, `docs/dsa_eval_report.md`, `docs/dsa_vllm_serving.md`,
  `docs/dsa_indexer_metrics.md`. Read `dsa_eval_report.md` §2 (verifying genuine sparsity) and §3
  (the drift-vs-sparsity ladder) first — that methodology is reused.
- **Superseded:** [`docs/qwen3_4b_dsa/plan.md`](../qwen3_4b_dsa/plan.md) (the token-granular
  DSA-on-GQA variant). Kept for the reasoning trail; see [plan.md](plan.md) §2 for why we moved.

**Status:** Phase 1 implemented and RUNNING (1B tokens, launched 2026-07-29; ~1.9 days). All feasibility unknowns are
closed; the remaining blockers are tests — see [plan.md](plan.md) §10.
