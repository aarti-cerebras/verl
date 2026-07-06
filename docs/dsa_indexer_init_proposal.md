# DSA Lightning-Indexer Initialization — Analysis & Proposal

Proposal for how to initialize the DSA lightning-indexer weights (`wq_b`, `wk`, `k_norm`, `weights_proj`)
for Phase-1 dense warm-up. Today the indexer takes PyTorch framework defaults; this doc analyzes what
those produce, shows the failure mode (layer-dependent softmax temperature), and proposes an explicit,
principled init plus one architectural fix.

Related: [`dsa_indexer_metrics.md`](dsa_indexer_metrics.md), [`dsa_grad_norm_debugging.md`](dsa_grad_norm_debugging.md).

---

## Current state (no explicit init)

`attach_indexers` (`verl/models/transformers/minicpm_dsa.py:83-91`) constructs `LightningIndexer(cfg)` and
relies entirely on PyTorch's default `nn.Linear` / `nn.LayerNorm` init (`dsa_indexer.py:191-196`):

| param | module | default init | resulting scale |
|---|---|---|---|
| `wq_b` | `Linear(768, 1024)` | `kaiming_uniform_(a=√5)` | weight var ≈ `1/(3·768)` → **q components ~ std 0.58** |
| `wk` | `Linear(2560, 64)` | `kaiming_uniform_(a=√5)` | re-normalized by `k_norm` → **k ~ unit var** |
| `k_norm` | `LayerNorm(64)` | γ=1, β=0 | keys unit-variance |
| `weights_proj` | `Linear(2560, 16)` (fp32) | `kaiming_uniform_(a=√5)` | sets the **score temperature** |

Under FSDP2 the load path (`verl/utils/fsdp_utils.py:492-499`) keeps rank-0's CPU init
(`model.to(device)`) and broadcasts it (`set_model_state_dict(..., broadcast_from_rank0=True)`), so **only
rank-0's init actually matters**.

---

## Empirical probe (what the default init produces)

Instantiating `LightningIndexer` at the MiniCPM3-4B sizing (`n_heads=16, head_dim=64, q_lora_rank=768,
hidden=2560`, `fp8=False` for a clean scale probe) and feeding random inputs at varying residual-stream
scale `x_std`:

```
x_std=1  → I std 0.23,  |I|max 1.13,  q·k std 4.63,  softmax entropy 4.14/4.16  (near-uniform ✓)
x_std=3  → I std 0.68,  |I|max 3.73,  q·k std 4.62,  softmax entropy 4.00/4.16
x_std=8  → I std 1.88,  |I|max 9.26,  q·k std 4.62,  softmax entropy 3.17/4.16  (saturating ✗)
```

(`entropy X/4.16`: 4.16 = `log T` = the uniform-distribution entropy; closer = higher-entropy start.)

### Reading

- The `q·k` dot std is **≈ 4.6** (`≈ √head_dim · σ_q · σ_k`), but `softmax_scale` multiplies the *weight*
  `w_g`, not the dot — so the effective softmax temperature is set by `weights_proj(x)`, not by the dot.
- **`I` magnitude scales linearly with ‖x‖.** `weights_proj` reads the *raw, unnormalized* residual
  stream, so the softmax temperature is **layer-dependent**: deeper layers (larger residual norm) start
  with a peakier, partly-saturated `softmax(I)`.
- At unit-variance input the default init is actually near-optimal (near-uniform). The problem is purely
  the coupling to the input scale.

This layer-dependent temperature is the likely driver of the `indexer/kl_layer_min` ↔ `kl_layer_max`
spread and of any `indexer/score_std` growth with depth.

---

## Proposal (prioritized)

### 1. Add an explicit, seeded `reset_parameters` — don't rely on framework defaults

The module is created fresh per attach, and (multi-rank) only rank-0's values survive the broadcast.
Explicit init is reproducible and testable.

### 2. Decouple the score temperature from ‖x‖ — **highest impact**

This is the real fix and it is architectural, not just an init constant: because ‖x‖ varies at *runtime*,
no init constant can normalize it. Apply an RMSNorm / LayerNorm to `x` before `weights_proj` (or normalize
the indexer input generally), so the temperature is layer-independent.

Init-only fallback (if we don't want an architectural change): shrink `weights_proj` std
(e.g. `0.5 · hidden^-0.5 ≈ 0.01`) so deep layers don't start saturated — but measure per-layer ‖x‖ first;
MiniCPM3's depth-scaled residuals may already bound it.

### 3. Do NOT zero-init `weights_proj`

Tempting for a "uniform at init" start, but `∂I/∂wq_b, ∂I/∂wk ∝ w_g` — zeroing `weights_proj` severs the
gradient to `wq_b`/`wk` and **recreates the exact zero-grad failure** from
[`dsa_grad_norm_debugging.md`](dsa_grad_norm_debugging.md) #2. Keep it **small but nonzero**.

### 4. Target near-uniform `softmax(I)` at init

Standard for a distillation / router-style objective: pick `wq_b` / `weights_proj` std so the initial
softmax entropy starts ≥ 0.95 · log T. Concrete `reset_parameters`:

```python
def reset_parameters(self):
    # near-uniform softmax(I) at init: q ~ unit variance, low score temperature.
    # weights_proj stays small-but-NONZERO (zeroing it severs wq_b/wk grads — see grad-norm doc #2).
    nn.init.normal_(self.wq_b.weight, std=self.cfg.q_lora_rank ** -0.5)   # q ~ unit variance
    nn.init.normal_(self.wk.weight,   std=self.cfg.hidden_size ** -0.5)   # k re-normed by k_norm anyway
    nn.init.ones_(self.k_norm.weight)
    nn.init.zeros_(self.k_norm.bias)
    nn.init.normal_(self.weights_proj.weight, std=0.5 * self.cfg.hidden_size ** -0.5)  # low temperature
```

### 5. (Secondary) Add a `q_norm` symmetric to `k_norm`

Currently only keys are normalized; the `q·k` std of 4.6 comes from an unnormalized query whose scale
rides on the `q_a_layernorm` gain and `wq_b`. A query norm makes the dot scale-stable and improves the fp8
`rotate_activation` conditioning — at the cost of diverging slightly from the DeepSeek-V3.2 reference.

---

## Correctness caveat to check regardless of init

In multi-rank runs, `fsdp2_load_full_state_dict` does `to_empty()` on non-rank-0 then
`set_model_state_dict(..., broadcast_from_rank0=True)`. If the indexer keys are **not** present in
`full_state` (they are freshly created, not in the base checkpoint), non-rank-0 ranks could keep the
`to_empty()` garbage instead of rank-0's init. Verify the indexer params are actually in the broadcast
set. The single-GPU smoke run takes the `model.to(device)` branch, so it is unaffected — this only bites
at `NPROC > 1`.

---

## References — how the DeepSeek/MLA family initializes

- **DeepSeek-V2/V3** (MLA): *all* learnable params `normal(0, 0.006)`, where `0.006 ≈ 0.5/√d_model` — a
  single **width-scaled** std ([V3 report 2412.19437](https://arxiv.org/pdf/2412.19437),
  [V2 2405.04434](https://arxiv.org/pdf/2405.04434)).
- **DeepSeek-V3.2 (DSA)**: dense warm-up = 2.1B tokens, indexer-only, frozen base, KL vs aggregated
  attention (= our Phase 1). The **indexer weight-init is not published** — V3.2 is an inference-only
  release ([2512.02556](https://arxiv.org/pdf/2512.02556)), so we reason from the family convention.
- **MiniCPM3 (host)**: every Linear `normal(0, initializer_range=0.1)` **with muP scaling**
  (`scale_depth=1.4`, `dim_model_base=256`, `scale_emb=12`; residual ×`scale_depth/√L ≈ 0.178`). The muP
  residual scaling keeps `‖x‖` bounded across depth, so the layer-dependent-temperature drift here is mild.
- **muP** ([Tensor Programs V, 2203.03466](https://arxiv.org/pdf/2203.03466)): the `σ ∝ 1/√fan_in` principle.

The indexer's output feeds `softmax(I)` (a distribution), **not** the muP-scaled residual, so it needs its
own small init to start near-uniform — the reasoning behind §4.

---

## Status — IMPLEMENTED (per-fan-in, §4)

`LightningIndexer.reset_parameters` (`dsa_indexer.py`) now does **per-fan-in** width-scaled normal init,
`std = 0.5/√fan_in` per projection (`wq_b`: fan_in=`q_lora_rank`; `wk`/`weights_proj`: fan_in=`hidden`);
`k_norm` at identity; `weights_proj` nonzero. Called from `__init__`, deterministic under seed.

**Verified** (`tests/models/test_dsa_indexer.py`):
- per-fan-in std + k_norm identity + weights_proj≠0; seed-determinism; every param gets grad through the
  fp8 path (`test_fp8_scores_are_differentiable`).
- **entropy-at-init** (`test_init_softmax_entropy_is_near_uniform`): `entropy_frac = 0.995` at unit input
  scale (near-uniform ✓); measured drop to `0.96` (×3) and `0.78` (×8) as `‖x‖` grows — the temperature
  effect. A runtime `indexer/entropy_frac` diagnostic now logs this every `diag_interval` during training.

**Not done (deferred):** §2 `weights_proj` input-norm (only needed if a deep layer's `indexer/entropy_frac`
starts well below 1.0 — muP already bounds `‖x‖`); §5 `q_norm` (diverges from reference). The multi-rank
broadcast caveat above is **closed** — see #6 / `docs/dsa_checkpoint_notes.md` (the value-diff confirms the
freshly-created indexer params broadcast correctly).
