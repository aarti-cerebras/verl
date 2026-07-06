# DSA Phase-1 Smoke: `grad_norm = 0` Debugging Post-Mortem

Record of the debugging that got the MiniCPM3-4B DSA indexer (Phase-1 dense warm-up) actually
**training** in the verl SFT trainer. The headline symptom was **`train/grad_norm = 0.0` with a flat
loss** — the run executed end-to-end but the indexer never learned. It turned out to be **two
independent root causes** that both manifest identically as zero gradient, plus a related FSDP1 backward
crash and a metrics-visibility issue. This doc captures each, how it was diagnosed, the fix, and the
lesson.

Related: [`dsa_phase1_smoke_plan.md`](dsa_phase1_smoke_plan.md).

---

## TL;DR — what was wrong and the fixes

| # | Symptom | Root cause | Fix | Where |
|---|---------|-----------|-----|-------|
| 1 | FSDP1 backward crash `setStorage: ... storage of size 0` on `wk` `[64,2560]` | FSDP1 FlatParameter mixing frozen base + trainable indexer; reshard freed the compute copy before the narrow backward | Switch to **FSDP2** (`fully_shard`, per-parameter) | `engine.strategy=fsdp2` |
| 2 | `grad_norm = 0`, flat loss (FSDP2) | **FP8 quant had no straight-through estimator** — `.to(float8_e4m3fn)` is non-differentiable, so no gradient reached `wq_b`/`wk`/`k_norm` (248/310 params; only `weights_proj`'s 62 survived, and its exact-zero norm here was co-caused by Issue #3) | Fake-quant **with STE**: `x + (x_q - x).detach()` | `dsa_indexer.py::_fake_quant_fp8` |
| 3 | `grad_norm = 0` still (FSDP2, after STE) | FSDP2 `reshard_after_forward=True` freed params after forward; the indexer-only backward (loss does **not** flow through the model's logits) didn't trigger re-gather → grads landed nowhere | **`reshard_after_forward=False`** (free at world_size=1) | launch script |
| 4 | Gradients flowed (`grad_norm ≈ 330`) but `indexer/*` diagnostics never logged | Custom loss-fn metrics arrive as per-micro-batch **lists** (`append_to_dict`); only `loss/grad_norm/lr/mfu` were reduced to scalars, the rest were dropped by the logger | Reduce any list/tensor metric to a scalar mean before `tracking.log` | `sft_trainer.py` |

Final state: clean 20-step run, `grad_norm ≈ 330` every step, peak mem ~37 GB stable, all `indexer/*`
diagnostics logging, `nan_frac = 0`.

---

## Setup / how the loss reaches the indexer

Phase-1 dense warm-up freezes the whole base model and trains **only** the per-layer lightning indexer.
The loss is the summed per-layer KL between the base model's own (head-averaged) attention distribution
`p` and `softmax(I)` where `I` are the indexer scores. Critically:

- The loss is stashed on the model by a forward hook (`model._dsa_indexer_kl`) and read by a closure
  loss fn (`indexer_kl_loss`). **It does not flow through the model's `logits` output** — it depends only
  on intermediate per-layer indexer scores. This "side-channel" loss is what makes the FSDP backward
  behave differently from a normal LM-CE loss (see #3).
- The trainable params are `*.indexer.*` only: `wq_b`, `wk`, `k_norm`, `weights_proj` (× 62 layers = 310
  params).

---

## Issue #1 — FSDP1 backward crash (`storage of size 0`)

**Symptom.** First real training attempt (FSDP1, `use_orig_params=True`) crashed in `loss.backward()`:

```
RuntimeError: setStorage: sizes [64, 2560], strides [2560, 1], storage offset 14304256,
and itemsize 2 requiring a storage size of 28936192 are out of bounds for storage of size 0
```

`[64, 2560]` is the indexer `wk` weight (`Linear(hidden=2560, head_dim=64)`), itemsize 2 = bf16 (the
mixed-precision compute copy).

**Root cause.** FSDP1 packs parameters into a `FlatParameter`. With mixed `requires_grad` (frozen base +
trainable indexer) and `reshard_after_forward`, the bf16 compute copy of the param is freed after
forward; the narrow indexer-only backward failed to re-materialize it → the autograd view pointed at
freed (size-0) storage.

**Fix.** Switch to **FSDP2** (`fully_shard`), which sharded **per-parameter** (no FlatParameter mixing),
so frozen base and trainable indexer coexist cleanly:

```
engine.strategy=fsdp2
```

**Lesson.** FSDP1 `use_orig_params=True` is required for mixed `requires_grad`, but is still fragile for
adapter-style training where most of the module is frozen. FSDP2 is the robust choice.

---

## Issue #2 — the FP8 quant severed the gradient (the real bug)

After moving to FSDP2 the run completed but **`grad_norm = 0.0` at every step** and the loss was flat
(~712–724 noise). The isolated overfit unit test had passed (KL 0.31 → 0.03) — but it used **`fp8=False`**,
so the FP8 gradient path had never been exercised.

### Diagnosis method

1. **Debug print in the loss fn** (before backward) established the setup was correct:
   `kl.requires_grad=True`, `kl.grad_fn=True`, `grad_enabled=True`, all 310 indexer params
   `requires_grad=True`. So: not the freeze, not FSDP wrap, not the loss plumbing. Backward *ran* but
   produced zero gradient.
2. **Read the FP8 score path** and traced the autograd graph by hand.

### Root cause (autograd trace)

The old FP8 path in `LightningIndexer.scores`:

```python
q_fp8, q_scale = _act_quant(q_idx)                    # x_fp8 = (x / scale).to(FP8_DTYPE)
k_fp8, k_scale = _act_quant(k_idx)
dots = torch.einsum("bqhd,bkd->bqhk", q_fp8.float(), k_fp8.float())
dots = torch.relu(dots) * k_scale[...]
eff_w = (weights * softmax_scale * q_scale)
scores = torch.einsum("bqhk,bqh->bqk", dots, eff_w)
```

- `.to(FP8_DTYPE)` (`float8_e4m3fn`) is the severance point, for two independent reasons:
  1. **Mathematically**: rounding to the FP8 grid is piecewise-constant → derivative is 0 almost
     everywhere and undefined at the steps.
  2. **Implementationally**: `float8_e4m3fn` has **no autograd support** — the cast output has
     `requires_grad=False`, `grad_fn=None`. The edge from `q_idx` (→ `wq_b`) is physically gone.
- `q_fp8.float()` on a non-grad tensor → a fresh **constant**. So `dots` is a constant → `relu(dots)*...`
  is a constant. **`wq_b`, `wk`, `k_norm` (4 of the 5 tensors per layer → 248/310 params) receive exactly
  zero gradient** — and these set the *shape* of `softmax(I)`.
- `q_scale`/`k_scale` come from `amax = x.detach()...` → also detached (correct as scales, but no grad).
- The **only** surviving edge is through `weights` (→ `weights_proj`, the 5th tensor per layer → 62/310
  params) via `eff_w`; that's why the loss still had a `grad_fn`. But its gradient coefficient is the
  detached `dots`, and a single scalar-per-head can't reshape the distribution — so effective training was
  ~nil. (This alone leaves a *small nonzero* `grad_norm` via `weights_proj`; the exact `grad_norm = 0.0`
  observed at this stage was actually the still-active reshard bug of Issue #3, which froze even the
  `weights_proj` path — the two bugs overlapped. Both fixes were needed.)

### Fix — fake-quant with a straight-through estimator (STE)

```python
def _fake_quant_fp8(x):
    with torch.no_grad():                                  # x_q is a value-only reference; keep no graph
        amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
        scale = amax / FP8_MAX
        x_q = (x / scale).to(FP8_DTYPE).float() * scale    # faithful E4M3 round-trip (forward)
    return x + (x_q - x).detach()                          # STE: forward == x_q, backward == identity
```

and take the dot product on the **dequantized** values:

```python
q_dq = _fake_quant_fp8(q_idx); k_dq = _fake_quant_fp8(k_idx)
dots = torch.relu(torch.einsum("bqhd,bkd->bqhk", q_dq, k_dq))
eff_w = (weights * softmax_scale).to(dots.dtype)
scores = torch.einsum("bqhk,bqh->bqk", dots, eff_w)
```

This is **numerically identical** to the old fp8 forward (the per-row positive scales factor out of the
ReLU), so the FP8 numerics DeepSeek-V3.2 uses are preserved — it only restores the gradient.

**Note on `torch.no_grad()`**: wrapping just the *quant value* `x_q` is a valid memory optimization (it's
detached anyway). It works **only because the STE identity add sits outside it** — wrapping the whole STE
expression in `no_grad` would re-sever the gradient. `no_grad` can only remove graph edges; it can never
create the surrogate one the STE provides.

### Verification

- CPU + bf16-autocast tests: all 310 params get nonzero grad through the fp8 path, rotate on/off.
- fp8 vs bf16 score parity ≈ **2.8%** relative (the expected E4M3 noise floor).
- **Regression test added**: `tests/models/test_dsa_indexer.py::test_fp8_scores_are_differentiable`.

**Lesson.** "Fake-quant" for QAT **requires** an STE — a bare de/quantize round-trip is not trainable.
This is standard (Jacob et al. 2018, arXiv:1712.05877; PyTorch `torch.ao.quantization.FakeQuantize`). Any
FP8/INT quant on a path that must receive gradient needs `x + (x_q - x).detach()` (or equivalent).

---

## Issue #3 — FSDP2 `reshard_after_forward` zeroed the gradient

Even after the STE fix, `grad_norm` was **still 0** in the engine — but an **isolated** GPU test of the
exact `project`/`scores` path under **bf16 autocast + a KL loss** gave full gradients (`wq_b`=14,
`wk`=4.4, …). So the scoring/loss/autocast were fine; the **only** remaining variable was FSDP2.

**Root cause.** The indexer params *are* wrapped by FSDP — they sit inside their decoder-layer
`fully_shard` unit (`apply_fsdp2` wraps each `MiniCPMDecoderLayer`; the indexer at `self_attn.indexer` is
absorbed into that unit, not wrapped separately). The bug is about *when* FSDP2 re-gathers them, not
membership. With `reshard_after_forward=True`, each layer frees its all-gathered param copy after forward
and re-gathers lazily via a **pre-backward hook keyed on the wrapped module's forward *output
activations*** (the `hidden_states` the block returns). The indexer KL is a side channel — computed during
forward, stashed on `self._dsa_kl`, summed into `model._dsa_indexer_kl` — and does **not** flow through
the decoder layer's returned `hidden_states` (the LM logits aren't in the loss; base is frozen). So during
backward autograd travels straight into `indexer.scores`/`project` without ever traversing the block's
output, the output-keyed re-gather hook **never fires**, and the params are still resharded when their
grad would accumulate → silently zero. (Same class of failure as Issue #1 — FSDP1 raised a hard error,
FSDP2 silently zeros.)

**Fix.** Keep params gathered through backward — free at world_size=1:

```
engine.reshard_after_forward=False
```

Result: `grad_norm ≈ 330` from step 1.

> **Phase-1-only setting.** This is needed only because Phase 1 is a frozen-base, KL-only loss with no
> gradient flowing through block outputs. In Phase 2 the CE loss re-gathers every layer, so
> `reshard_after_forward=True` is correct and this must be **dropped**. Full mechanics, the four FSDP hooks,
> and multi-rank sharding options are in [`dsa_fsdp_sharding_notes.md`](dsa_fsdp_sharding_notes.md).

**Lesson.** When the training loss depends on **intermediate activations** rather than the wrapped
module's returned output (adapter/probe/aux-loss training), FSDP re-gather hooks keyed on the output may
not fire — disable `reshard_after_forward` (or otherwise ensure params stay materialized for backward).

### Debugging note: the false-negative grad probe

A `register_post_accumulate_grad_hook` on `wq_b.weight` **never fired**, which looked alarming but was a
red herring: it was registered **before** FSDP2 wrapping, and `fully_shard` + `fsdp2_load_full_state_dict`
(`to_empty()` + `set_model_state_dict`) **replace every parameter object** with a new DTensor. The hook
sat on the orphaned pre-wrap tensor. `requires_grad` survives (copied onto the new params) but
tensor-level hooks bound to the old objects do not. **Takeaway: register param hooks *after* FSDP
wrapping, or use module-level hooks.** The reliable signal was `train/grad_norm`, computed over the actual
post-wrap params.

---

## Issue #4 — gradients flowed but `indexer/*` diagnostics weren't logged

`train/loss` and `train/grad_norm` logged fine, but `indexer/kl`, `topk_recall`, `kl_layer_mean`, etc.
were absent.

**Root cause.** `postprocess_batch_func` does `append_to_dict(aggregated_metrics, metrics)` → every
loss-fn metric becomes a **per-micro-batch list**. `_postprocess_output` special-cases only
`loss/grad_norm/lr/mfu` (reduces to scalars) and adds `perf/*`; every other key stays a list. The metrics
dict is stored as a **non-tensor** blob so the lists survive to the trainer, but `tracking.log` (wandb)
silently drops non-scalar values.

**Fix.** In `sft_trainer.py`, reduce any list/tensor metric to a scalar mean before logging:

```python
def _reduce_metric(v):
    if isinstance(v, torch.Tensor): return v.detach().float().mean().item()
    if isinstance(v, (list, tuple)):
        vals = [_reduce_metric(x) for x in v if x is not None]
        return sum(vals) / len(vals) if vals else None
    return v
for k in list(metrics.keys()):
    r = _reduce_metric(metrics[k])
    metrics.pop(k) if r is None else metrics.__setitem__(k, r)
```

**Lesson.** verl's engine reduces only a fixed set of scalar metric keys; **custom loss-fn metrics arrive
as per-micro-batch lists** and must be reduced explicitly, or they never reach the logger.

---

## General takeaways

1. **`grad_norm = 0` has many causes that look identical.** Localize with a layered probe: (a) at loss
   time check `requires_grad`/`grad_fn`/`grad_enabled` and param `requires_grad`; (b) reproduce the exact
   compute path in isolation (no engine/FSDP) to separate "math/graph" bugs from "distributed wrapper"
   bugs; (c) only then suspect FSDP.
2. **Test the production precision path.** The FP8 grad bug survived because the only training test used
   `fp8=False`. A fake-quant path needs a differentiability test (now added).
3. **Quantization on a grad path needs an STE.** Always. `x + (x_q - x).detach()`.
4. **Adapter/aux-loss training stresses FSDP differently** than full-model CE: mixed `requires_grad`
   (→ FSDP2) and side-channel losses that don't flow through the module output
   (→ `reshard_after_forward=False`).
5. **Trust the post-wrap signal (`grad_norm`), not hooks bound to pre-wrap tensors.** FSDP2 swaps
   parameter objects.
6. **Synthetic random data → flat KL/recall is expected.** With a fresh random-token batch each step there
   is no attention structure to distill; real learning is shown by the overfit test and requires real
   long-context data.
