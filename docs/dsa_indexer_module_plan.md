# Step 1 Plan: Lightning Indexer module (standalone, unit-tested)

First slice of the DSA work (see `docs/dsa_minicpm3_plan.md` for the full design). Goal: implement and
unit-test the `LightningIndexer` + `DSAConfig` as a **self-contained module** on dummy tensors, de-risking
the core math (projections, RoPE, MQA broadcast, ReLU weighted-sum, FP8↔bf16 parity) before any verl wiring.

## Scope / non-goals

**In scope:** the module, its config, and a standalone pytest. **Deferred** (later steps): monkey-patch
attachment, exposing `qr` from the MLA forward, dense-attention target recompute, KL loss, custom FSDP
engine, dataset, training. The module needs **no MiniCPM3 weights and no monkey-patch** to test.

## Files

**Create**
- `verl/models/transformers/dsa_indexer.py` — `DSAConfig` dataclass + `LightningIndexer(nn.Module)`.
- `tests/models/test_dsa_indexer.py` — standalone unit tests (mirrors `tests/models/test_transformers_ulysses.py`).

**Look at (reference)**
- `verl/models/transformers/kimi_vl.py:34-75` — `rotate_half` / `apply_rotary_pos_emb` (RoPE convention to mirror).
- `verl/utils/kernel/fp8_kernel.py:36-38` — `FP8_DTYPE`/`FP8_MAX` constants (reuse for E4M3 quantization).
- `verl/utils/megatron_peft_utils.py:40-42` — submodule names must stay `wq_b`/`wk`/`weights_proj`.
- DeepSeek-V3.2 `inference/model.py` `Indexer` (external) — reference semantics.

## Module spec

`DSAConfig`: `enabled, n_heads=16, head_dim=64, rope_head_dim=32, q_lora_rank=768, hidden_size=2560,
top_k=2048, mode="dense_warmup", kl_block_size=1024, fp8=True`.

`LightningIndexer(cfg, softmax_scale=None)`:
- `wq_b = Linear(q_lora_rank, n_heads*head_dim, bias=False)` — query from `qr` (no `wq_a`).
- `wk = Linear(hidden_size, head_dim, bias=False)` — **single (MQA) key head**.
- `k_norm = LayerNorm(head_dim)`.
- `weights_proj = Linear(hidden_size, n_heads, bias=False)` in **fp32**.
- `softmax_scale = softmax_scale or head_dim**-0.5`.
- `project(x, qr, cos, sin)` → `q_idx[b,s,H,D]`, `k_idx[b,s,D]`, `w[b,s,H]` (RoPE on rope-slice of q & k;
  `w = weights_proj(x.float()) * n_heads**-0.5`).
- `scores(q_idx, k_idx, w, attn_bias=None)` → raw `I[b,q,k] = Σ_h (w_h·softmax_scale)·ReLU(⟨q_h,k⟩)`
  (single key broadcast over heads; `+ attn_bias` additive mask; **no softmax**). bf16 reference path +
  FP8 path behind `cfg.fp8`.
- `forward(x, qr, cos, sin, attn_bias=None) = scores(*project(...), attn_bias)`.

FP8 for step 1 = per-tensor E4M3 quantization of q/k (validated against bf16 within tolerance); a fused
fp8 GEMM (`torch._scaled_mm` / Triton `fp8_index`) is a later perf swap.

## Tests (`tests/models/test_dsa_indexer.py`, dummy tensors b=2 s=16 hidden=2560 q_lora=768 H=16 D=64)

1. shapes of `project`/`scores`; 2. MQA broadcast (single key shared) vs per-head loop; 3. ReLU
weighted-sum matches a naive reference; 4. RoPE touches only the rope-slice; 5. additive `-inf` mask
zeroes entries; 6. FP8≈bf16 parity (gated on cuda+sm90); 7. param names `wq_b/wk/k_norm/weights_proj`,
`weights_proj` is fp32, param count ≈ table.

## Verify
```
pytest tests/models/test_dsa_indexer.py -v
pytest tests/models/test_dsa_indexer.py -v -k fp8   # H100
```
