# DSA Phase-2 sparse training — implementation spec (T1–T7)

Code-level companion to `docs/dsa_phase2_plan.md` (strategy). Implements the paper's decoupled sparse stage
(arxiv 2512.02556 §2.1.1): **base ← LM loss through sparse attention; indexer ← selected-set KL with detached
input; top-k selection is stop-grad between them.** `top_k=512`.

Files: `verl/models/transformers/minicpm_dsa.py` (T1,T2), `verl/models/transformers/dsa_indexer.py` (helpers),
`verl/workers/utils/losses.py` (T3), `verl/trainer/sft_trainer.py` (T3), `verl/models/transformers/monkey_patch.py` (T4).

## T1 — Sparse attention forward  (`minicpm_dsa.py`, replaces the `mode=="sparse"` stub)

New `_sparse_attn(attn, hidden_states, qr, query_states, key_states, value_states, cos, sin, position_ids,
attention_mask)` returning `attn_output [b,T,H*v_head_dim]` (post-`o_proj`). Called from
`minicpm3_dsa_attn_forward` when `dsa.mode=="sparse"`, and its output REPLACES the stock flash path.

```
r = rope_head_dim; cos_g, sin_g = cos[position_ids], sin[position_ids]
q_idx,k_idx,w = attn.indexer(hidden_states, qr, cos_g, sin_g, return_projection=True)  # via __call__ (FSDP2)
I   = attn.indexer.scores(q_idx, k_idx, w, attn_bias=causal_doc_bias)   # [b,T,T]
idx = I.topk(min(top_k, T), dim=-1).indices.detach()                    # [b,T,k]  = S_t (stop-grad)
# tiled over query blocks (block = dsa.kl_block_size), mirror _dense_warmup_kl memory pattern:
for q0,q1:
    idx_b = idx[:, q0:q1]                                        # [b,B,k]
    bias_b= gather(causal_doc_bias[:, q0:q1], -1, idx_b)         # [b,B,k]  -inf for pad/cross-doc keys
    Kg = gather(key_states,  idx_b expanded over H)             # [b,H,B,k,q_head_dim]
    Vg = gather(value_states,idx_b expanded over H)             # [b,H,B,k,v_head_dim]
    s  = einsum('bhBd,bhBkd->bhBk', query_states[:,:,q0:q1], Kg) * softmax_scale + bias_b[:,None]
    a  = softmax(s.float(), -1).to(v.dtype)
    o[:, :, q0:q1] = einsum('bhBk,bhBkd->bhBd', a, Vg)          # [b,H,B,v_head_dim]
attn_output = o.transpose(1,2).reshape(b,T,H*v_head_dim); return attn.o_proj(attn_output)
```

- **Grad:** flows into base via `query/key/value_states` (LM loss trains base); `idx` is `.detach()` → selection
  is stop-grad → no LM-loss grad into the indexer.
- **value gather:** gather `value_states` at its own `v_head_dim` (do NOT pad to `q_head_dim` — that padding is
  only for the stock flash kernel).
- **Parity invariant (M0 test):** `top_k>=T` ⇒ selected set = all causal-doc keys ⇒ output == dense flash
  (single-doc case, atol ~1e-2 bf16).
- Activation-checkpoint each tile when `dsa.kl_checkpoint and training` (reuse Phase-1 pattern).

## T2 — Selected-set indexer KL  (`minicpm_dsa.py`, `sparse` branch of `_dense_warmup_kl` or `_sparse_indexer_kl`)

Reuse the tiling + detached target `p_blk` (head-averaged softmax main attention over causal-doc). Restrict to
`S_t = idx` and renormalize (paper eq. 4):
```
qr_d, x_d = qr.detach(), hidden_states.detach()            # detach indexer input -> no grad into base
I_S   = attn.indexer.scores(<from x_d,qr_d>, attn_bias=bias)   # grad only into indexer params
p_S   = gather(p_blk, -1, idx_b); p_S = p_S / p_S.sum(-1, keepdim=True).clamp_min(eps)   # [b,B,k]
logq_S= log_softmax(gather(I_S, -1, idx_b).float(), -1)
kl    = (p_S * (p_S.clamp_min(eps).log() - logq_S)).sum(-1)    # [b,B]
```
Average over valid (non-pad) query rows (existing `qv` mask), accumulate on `attn._dsa_kl`. `install_kl_accumulation`
(unchanged) sums per-layer → `model._dsa_indexer_kl` + metrics. Keep `topk_recall` as the health gate. `idx` MUST
be the same selection as T1 (compute once, share).

## T3 — Loss + trainer wiring

- `verl/workers/utils/losses.py`: `dsa_sparse_loss(config, model_output, data, dp_group=None, model=None)` =
  `sft_loss(config, model_output, data)` **+ `λ * model._dsa_indexer_kl`** (read off model like `indexer_kl_loss:54`;
  `λ = config` knob). Returns merged metrics (`indexer/*` + LM loss). LM logits come from T1's `o_proj` output → LM head.
- `verl/trainer/sft_trainer.py:~178`: `elif loss_mode == "dsa_sparse": self.loss_fn = lambda **kw:
  dsa_sparse_loss(**kw, config=self.config, model=self.engine.module)`.

## T4 — Unfreeze base + two param groups

- `monkey_patch.py:514-516` freezes only for `dense_warmup` → sparse leaves base trainable (verify no other freeze).
- Optimizer: two groups — base LR `7.3e-6`, indexer (`.indexer.`) LR `1e-3`, grad-clip 1.0. FSDP `use_orig_params=True`;
  indexer stays its own `fully_shard` unit.

## T5 — Config / scripts

`sft_trainer_minicpm_dsa_phase2.yaml` + `examples/dsa/run_minicpm3_dsa_phase2.sh` (mirror Phase-1): `dsa_mode=sparse`,
`dsa_top_k=512`, `loss_mode=dsa_sparse`, `indexer_kl_lambda`, base/indexer LRs, `pad_mode=right`,
`use_remove_padding=False`, `RESUME_PATH`=Phase-1 indexer ckpt, `MultiTurnSFTDataset` on the trajectory `messages` parquet.

## T6 — Tests  `tests/models/test_minicpm_dsa_sparse.py` (M0 gate, CPU/1-GPU)

1. Parity: `top_k>=T` sparse ≡ dense flash (atol ~1e-2).
2. Grad/detach: LM loss → base q/k/v, NOT indexer; selected-set KL → indexer, NOT base.
3. Selected-set KL numerics vs a small reference.
4. top-k respects causal + per-doc bias.
5. Extend `test_minicpm_dsa_overfit.py` with a tiny sparse overfit (loss↓, recall↑).

## T7 — Math-only validation (go/no-go)

Init Phase-1 indexer ckpt → train `dsa_sparse` on the math trajectories (`/cb/ml-eng/aarti/dsa/m3a_gen_*`), `top_k=512`
→ eval vs dense baseline (`/cb/ml-eng/aarti/dsa/evals/minicpm3-4B`): GSM8K/MATH (preservation; seqs >512 → sparsity
active) + HumanEval/MMLU/C-Eval (forgetting). Gate: math ≈ dense within ~1% AND non-math not collapsed.

## Sequencing

| Milestone | Tasks | Gate |
|---|---|---|
| **M0** build + correctness | T1, T2 + T6 | ✅ DONE — parity 1.8e-7; grad decoupling (LM→base, KL→indexer) verified on CPU (`tests/models/test_minicpm_dsa_sparse.py`) |
| **M1** tiny e2e | T3, T4 + overfit | T3/T4 CODE DONE (compile-clean: `dsa_sparse_loss`, `loss_mode=dsa_sparse`, `indexer_lr` two param groups). e2e overfit-run verification PENDING (needs GPU). |
| **M3a-math** validation | T5, T7 | GSM8K/MATH ≈ dense, no forgetting |

**Risks:** MLA `q_head_dim != v_head_dim` (gather value at v_head_dim); gathered-bias must reproduce flash causal
masking for the parity test; per-tile `[b,H,B,k,d]` memory (tile `B`, `k=512`).
