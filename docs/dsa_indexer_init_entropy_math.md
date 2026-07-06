# DSA Indexer Init → Entropy: the Math

Why the per-fan-in init (`std = 0.5/√fan_in`) makes `softmax(I)` start near-uniform, how the score
magnitude and `indexer/entropy_frac` depend on the input scale `x_std`, and why it should be that way.
Derivation ties the init to the `indexer/entropy_frac` diagnostic and predicts the observed numbers.

Related: [`dsa_indexer_init_proposal.md`](dsa_indexer_init_proposal.md) (init §4),
[`dsa_indexer_metrics.md`](dsa_indexer_metrics.md) (the entropy metric).

Dims (MiniCPM3-4B): `q_lora_rank = 768`, `hidden = 2560`, `head_dim D = 64`, `n_heads G = 16`,
`softmax_scale = D^{-1/2} = 1/8`. Inputs: `qr` is unit-variance (post `q_a_layernorm`); `x` (hidden state)
has per-component std `x_std`.

---

## Step 1 — variance propagation through the init

For a linear `y = Wx` with `W ~ N(0, σ²)`: `Var(y) = fan_in · σ² · Var(x)`. With **per-fan-in**
`σ = 0.5/√fan_in`, the `fan_in` cancels:

```
Var(y) = fan_in · (0.25 / fan_in) · Var(x) = 0.25 · Var(x)
```

So **each projection outputs exactly `0.5 · std(input)`, regardless of its width** — the point of fan-in
scaling. Applied to the three projections:

| quantity | derivation | result |
|---|---|---|
| **q** `= wq_b(qr)` | `√(768 · (0.5/√768)² · 1) = 0.5` | `std(q) = 0.5` — **x_std-independent** |
| **k** `= k_norm(wk(x))` | `wk(x)` has std `0.5·x_std`, then `k_norm` **renormalizes to unit variance** | `std(k) ≈ 1` — **x_std-independent** |
| **w** `= n_heads^{-1/2} · weights_proj(x)` | `√(2560 · (0.5/√2560)² · x_std²) · (1/4) = 0.125·x_std` | `std(w) = 0.125·x_std` — **∝ x_std** |

RoPE is an orthonormal rotation → preserves these variances.

**Key observation.** `q` and `k` scales are *independent* of `x_std` — `k` because `k_norm` renormalizes,
`q` because it comes from the already-normalized `qr`. The **only** channel through which `x_std` enters the
score is the *unnormalized* `weights_proj(x)` → `w ∝ x_std`.

---

## Step 2 — score magnitude `std(I)`

```
I[i,j] = Σ_{h=1..G} ( w_h · softmax_scale ) · ReLU( q_h · k )
```

- dot `q_h·k = Σ_{d=1..64} q_d k_d`:  `Var = D · Var(q) · Var(k) = 64 · 0.25 · 1 = 16` → `std(dot) = 4`
  (**x_std-independent**).
- `E[ReLU(dot)²] = ½ E[dot²] = 8` (symmetric dot).
- per head `c_h = softmax_scale · w_h · ReLU(dot_h)`, with `w` mean-0 ⊥ `ReLU`:

```
Var(c_h) = (1/64) · E[w²] · E[ReLU²] = (1/64) · (0.125·x_std)² · 8 = 1.95e-3 · x_std²
```

- sum over `G = 16` heads:

```
std(I) = √( 16 · 1.95e-3 ) · x_std = 0.177 · x_std
```

So the **softmax logit spread grows linearly with `x_std`**, and at `x_std = 1` it is `≈ 0.18` —
deliberately `≪ 1`.

---

## Step 3 — from logit spread to `entropy_frac`

For logits with std `s` over `T` keys, expand the softmax entropy to 2nd order (`z_j = s·ε_j`, small `s`):

```
H ≈ ln T − ½ s² (1 − 1/T)
entropy_frac = H / ln T ≈ 1 − s²(1 − 1/T) / (2 ln T)
```

*Derivation sketch:* `q_j ∝ e^{s ε_j}`; to second order `H = ln T − ⟨δ²⟩/2` with
`⟨δ²⟩ = Var(logits) = s²(1 − 1/T)` (sample variance around the sample mean → the `1 − 1/T` factor).

Substituting `s = std(I) = 0.177·x_std` gives a **quadratic falloff in `x_std`**:

```
entropy_frac ≈ 1 − C · x_std² ,     C = 0.177² (1 − 1/T) / (2 ln T)
```

With `T = S = 16` (the unit test's length), `ln 16 = 2.773`, `1 − 1/T = 0.9375` → `C ≈ 0.00529`.

| `x_std` | predicted `1 − C·x_std²` | observed |
|---|---|---|
| 1 | `0.9947` | **0.9954** |
| 3 | `0.9524` | **0.9592** |
| 8 | `0.661` | **0.7765** |

The `x_std = 1` prediction matches to `< 0.001`. At `x_std = 8` the 2nd-order expansion breaks down
(`s ≈ 1.4` is no longer small, so it *over*-predicts the deficit — the true 0.78 is higher), but the
direction and rough magnitude are right.

---

## Why it should be this way

1. **Fan-in scaling → controlled, width-independent activations.** `dot ~ N(0,16)` is `O(few)`, not
   `O(width)`. Combined with `softmax_scale = 1/8`, `n_heads^{-1/2} = 1/4`, and the `0.5` factor,
   `std(I) ≈ 0.18` at unit input — engineered to be `≪ 1` so `entropy_frac ≈ 1` (near-uniform).
2. **Near-uniform is the right start** for KL distillation: unbiased prior, finite/gentle loss, and maximal
   softmax responsiveness `q(1−q)` (steerable). The tiny `s²/2·(…)` deficit means the student begins
   essentially "I don't know yet," which is correct before any training.
3. **The residual sensitivity is entirely `weights_proj(x) ∝ x_std`** — the "score temperature ∝ ‖x‖"
   effect, now derived from first principles. It is benign *because* MiniCPM's muP residual scaling
   (`× scale_depth/√L ≈ 0.178` per layer) keeps `‖x‖ ~ O(1)`, so we sit near `x_std = 1`
   (`entropy_frac ≈ 0.995`), not at `x_std = 8`.
4. **`indexer/entropy_frac` is the guard.** If some deep layer *does* have large `‖x‖`, its step-0
   `entropy_frac` reads well below 1.0 — the exact signal that the §2 fix (normalize `weights_proj`'s input)
   is warranted. Until then, init-only (§4) is correct, and the math says why.

---

## Practical reading of the diagnostic

- `entropy_frac ≈ 1.0` at step 0 → healthy, near-uniform start (expected).
- `entropy_frac` **rising toward a plateau then falling gently** during training → the indexer is learning
  structure (concentrating mass on the keys attention cares about). Some fall is expected and good.
- `entropy_frac ≪ 1.0` **at step 0** → a layer started saturated (large `‖x‖` / wrong init temperature);
  investigate before trusting that layer's learning (this is the `§2`-fix trigger).
- Compare against `indexer/score_std` (raw logit spread `s`) and `indexer/kl_layer_min|max` (per-layer
  spread) — all three move together when the temperature is off on some layers.
