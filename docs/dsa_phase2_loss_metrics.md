# DSA Phase-2 — how the loss & grad-norm metrics are computed

Reference for reading the per-step log / wandb of a `loss_mode=dsa_sparse` run (e.g.
`dsa_runs/phase2-overfit24-6gpu-*`). Explains what each reported number means and how it is reduced across
micro-batches and data-parallel (dp) ranks.

Config used in the examples: **D = 6** dp ranks, **M = 2** micro-batches per rank
(`train_batch_size=12`, `micro_batch_size_per_gpu=1`), so **N = D·M = 12** total micro-batches per step.

> **Note (2026-07-16 change):** `loss/lm` and `loss/kl_weighted` are now **sum-reduced** (like `train/loss`),
> so **`loss/lm + loss/kl_weighted == train/loss`** and each is the true whole-batch average. Previously they
> were *mean*-reduced and came out a factor of `M` (micro-batches per rank) too small, which did not sum to
> `train/loss` — confusing. See §5.

---

## 1. The reported metrics

| metric | meaning | reduction |
|---|---|---|
| `train/loss` | the actual optimized scalar (`CE + λ·KL`, whole-batch avg) | sum over micro-batches, dp-avg |
| `loss/lm` | LM cross-entropy term = ∑CE over valid tokens / num valid tokens | sum over micro-batches, dp-avg |
| `loss/kl_weighted` | `λ · (whole-batch mean selected-set KL)` | sum over micro-batches, dp-avg |
| `indexer/kl` | raw per-batch KL mean over layers + valid queries (diagnostic) | mean |
| `train/kl_lambda` | the KL weight `λ` (constant unless scheduled) | constant |
| `train/lr` | base-group learning rate (first optimizer group) | — |
| `lr/base`, `lr/indexer` | per-optimizer-group LRs (two-group DSA runs) | — |
| `train/grad_norm` | global grad L2 norm over the **whole model** (all params, both LR groups), pre-clip | — |
| `grad_norm/base`, `grad_norm/indexer` | per-optimizer-group grad norms, pre-clip | — |
| `kl_by_layer/L00…L61`, `indexer/kl_layer_*` | per-layer KL diagnostics | mean |

"Valid tokens" = **loss-mask tokens = response/assistant tokens only** (the prompt is masked out).

Code: `verl/workers/utils/losses.py` (`dsa_sparse_loss`, `sft_loss`), `verl/workers/engine/engine_workers.py`
(`_postprocess_output`, per-group LR), `verl/workers/engine/utils.py` (`postprocess_batch_func`),
`verl/trainer/sft_trainer.py` (`_scalarize_metric`), `verl/workers/engine/fsdp/transformer_impl.py`
(`optimizer_step`, `_build_optimizer`, `lr_scheduler_step`).

---

## 2. Per-micro-batch loss terms

For micro-batch `i`, `dsa_sparse_loss` returns two terms, each **already global-batch-normalized**
(`÷ global count × dp_size`, `losses.py:138, 105`):

$$\ell^{(i)}_{lm} = \frac{S_i}{T_{\text{glob}}}\cdot D
\qquad
\ell^{(i)}_{kl} = \lambda\cdot k^{(i)}\cdot\frac{v_i}{V_{\text{glob}}}\cdot D$$

- `S_i` = summed token CE over micro-batch i's loss-mask tokens; `T_glob` = `batch_num_tokens` (global
  loss-token count, all-reduced SUM).
- `k^(i)` = micro-batch i's mean KL (`model._dsa_indexer_kl`); `v_i` = `mb_valid`; `V_glob` = `num_valid`
  (`batch_num_valid_queries`, all-reduced SUM). The backpropped loss is `L^(i) = ℓ^(i)_lm + ℓ^(i)_kl`.

### `mb_valid` and `num_valid`
- **`mb_valid`** = `tu.num_valid_queries(data)` on **this micro-batch** (nested → `offsets().diff().sum()`;
  padded → `attention_mask.sum()`).
- **`num_valid`** = `batch_num_valid_queries` — same count over the **full step batch**, all-reduced SUM over
  dp (`transformer_impl.py:654-659`). So `mb_valid/num_valid × dp_size` weights each micro-batch's KL mean so
  the micro-batches **sum** to the global-mean KL — same normalization `sft_loss` gives the LM term.

---

## 3. Reduction — everything additive is SUMMED, diagnostics are MEANED

`train/loss` (`engine_workers.py:186-189`): per rank `torch.sum` over local micro-batches → `all_reduce(AVG)`
over dp. Because of the `× D`, this reconstructs the **true global objective**:

$$\texttt{train/loss}=\frac{1}{D}\sum_{i=1}^{N}L^{(i)}
=\underbrace{\frac{\sum_i S_i}{T_{\text{glob}}}}_{\text{whole-batch mean CE}}
+\lambda\underbrace{\frac{\sum_i k^{(i)}v_i}{V_{\text{glob}}}}_{\text{whole-batch mean KL}}$$

`loss/lm` and `loss/kl_weighted` are reduced the **same way** (`engine_workers.py`, the `loss/`-prefix branch
in `_postprocess_output`): sum over micro-batches, dp-avg. So:

$$\texttt{loss/lm}=\frac{1}{D}\sum_{i}\ell^{(i)}_{lm}=\frac{\sum_i S_i}{T_{\text{glob}}}
\qquad
\texttt{loss/kl\_weighted}=\frac{1}{D}\sum_{i}\ell^{(i)}_{kl}=\lambda\cdot\frac{\sum_i k^{(i)}v_i}{V_{\text{glob}}}$$

Diagnostics (`indexer/kl`, `kl_by_layer/*`, `perf/*`, `lr/*`, `grad_norm/*`) are **mean**-reduced via
`_scalarize_metric` (`sft_trainer.py:50`) — fine, since they're per-step constants or averages, not additive
components of the loss.

---

## 4. The key identity

$$\boxed{\ \texttt{train/loss}=\texttt{loss/lm}+\texttt{loss/kl\_weighted}\ }$$

`loss/lm` is exactly **∑CE over all valid (response) tokens / total valid tokens** across the whole batch —
invariant to the micro-batch / dp split. `loss/kl_weighted` is `λ ×` the whole-batch mean per-query KL.

---

## 5. Why it used to differ (history)

Before the fix, `loss/lm` / `loss/kl_weighted` were **mean**-reduced over the N micro-batches:
`loss/lm_old = (1/N)∑ℓ^(i)_lm = (D/N)·A = A/M`, where `A` is the true average. So they were a factor of
`M = N/D` (micro-batches per rank) too small and gave `train/loss = M·(loss/lm + loss/kl_weighted)` — e.g.
`train/loss` was 2× their sum. The fix reduces them by sum+dp-avg (matching the loss), removing the factor.

---

## 6. Worked example (real step-100 objective ≈ 0.07575)

At step 100 the optimized objective was `train/loss = 0.07575`, `indexer/kl = 0.0756`, `λ = 1`, LM overfit
(`~0`). After the fix the components read as the true averages and add up:

$$\texttt{loss/lm}\approx 2.88\text{e-}05,\qquad
\texttt{loss/kl\_weighted}\approx \lambda\cdot 0.0756 = 0.0757$$
$$\texttt{loss/lm}+\texttt{loss/kl\_weighted}\approx 0.0757 = \texttt{train/loss}\ \checkmark$$

(Before the fix these logged as `1.44e-05` and `0.03786`, summing to `0.03788` — half of `train/loss`.)

---

## 7. Grad norms & learning rates

- **`train/grad_norm`** (`transformer_impl.py:717`, `fsdp2_clip_grad_norm_(self.module.parameters())`): one
  global L2 norm over **all** grad-carrying params — all 62 layers' base weights, embeddings, `lm_head`, and
  every layer's lightning-indexer params — i.e. both optimizer groups combined, on the **unscaled grads
  before clipping**, FSDP2-shard-aware.
- **`grad_norm/base` / `grad_norm/indexer`**: per-optimizer-group breakdown of that same pre-clip norm
  (`optimizer_step`; groups tagged `"base"`/`"indexer"` in `_build_optimizer`). Use to confirm both LR groups
  are healthy independently (indexer @ 1e-3 not exploding while base @ 7.3e-6 trains gently).
- **`train/lr`** = base-group LR; **`lr/base`, `lr/indexer`** = per-group LRs read off `optimizer.param_groups`
  after the scheduler step (`engine_workers.py`). Both groups follow the same schedule *shape*, scaled to
  their own peak LR.
- **`train/kl_lambda`** = the KL weight `λ` (`dsa_sparse_loss`), logged for provenance.
- Separate from the `[dsa-master]` probe (per-indexer-tensor grad norms + Adam state, gated on
  `DSA_DEBUG_MASTER=1`).

---

## 8. TL;DR

- `train/loss` = `loss/lm + loss/kl_weighted` (all sum-reduced; the real optimized objective).
- `loss/lm` = whole-batch average CE = ∑CE / num valid (response) tokens.
- `loss/kl_weighted` = `λ ×` whole-batch mean KL; `indexer/kl` ≈ the same KL before renormalization (diagnostic).
- `train/lr` + `lr/base` + `lr/indexer` show the two LR groups; `train/grad_norm` + `grad_norm/{base,indexer}`
  show combined and per-group grad norms; `train/kl_lambda` logs the KL weight.
