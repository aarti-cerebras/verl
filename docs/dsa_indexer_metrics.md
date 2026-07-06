# DSA `indexer/*` Metrics Reference

Exact definition of every `indexer/*` metric logged during DSA Phase-1 (dense warm-up) training —
input/output shapes, what each sum/mean is taken over, and the equations. All metrics originate in
`verl/models/transformers/minicpm_dsa.py::_dense_warmup_kl` / `install_kl_accumulation` and are surfaced
through `verl/workers/utils/losses.py::indexer_kl_loss`, then reduced for logging in
`verl/trainer/sft_trainer.py`.

Related: [`dsa_grad_norm_debugging.md`](dsa_grad_norm_debugging.md).

---

## Setup and notation

Per decoder layer ℓ, the loss compares two distributions over key positions, for each query position.

| symbol | meaning | shape |
|---|---|---|
| `bsz` | batch size | |
| `T` | sequence length (keys = full seq) | |
| `H` | base-model attention heads | |
| `Dh` | base per-head dim (`q_head_dim`) | |
| `B` | query-block length `q1-q0` (tiled by `kl_block_size`) | |
| `G` | indexer heads (`n_heads` = 16) | |
| `k` | `diag_k = min(top_k, T)` (256 in the smoke run) | |
| `L` | number of decoder layers (MiniCPM3-4B: 62) | |

Both distributions are built per query-block `[q0,q1)` and reduced (`_dense_warmup_kl`,
`minicpm_dsa.py:182–293`).

### Target `p` (base attention, head-averaged, detached)

`minicpm_dsa.py:229–235`. For base head `h`, with `scale = attn.softmax_scale` and the additive
causal + same-doc + non-pad mask `bias ∈ {0, −∞}`:

```
s_h[i,j] = (q_h[i] · k_h[j]) * scale + bias[i,j]
p[i,j]   = (1/H) * Σ_h softmax_j( s_h[i,j] )
```

### Indexer scores `I` (grad flows here only)

`dsa_indexer.py:231–257`, `minicpm_dsa.py:243`. With indexer per-head weights `w`, and the indexer's own
`softmax_scale` (= indexer `head_dim**-0.5`, **distinct** from the base `scale` above):

```
I[i,j] = Σ_{g=1..G} ( w_g[i] * softmax_scale ) * ReLU( q_idx_g[i] · k_idx[j] ) + bias[i,j]
```

Both `p` and `I` are `[bsz, B, T]` per block. `log_q = log_softmax_j(I)`.

---

## The KL family

**Per-layer KL** (`minicpm_dsa.py:246–293`) — for each valid query row `i`, `KL(p_i ‖ softmax(I_i))`
summed over *allowed* keys, then averaged over *valid* (non-pad) query rows across all blocks:

```
KL_i     = Σ_{j : allow[i,j]}  p[i,j] * ( log p[i,j] − log q[i,j] )
KL^(ℓ)   = ( Σ_{i ∈ valid} KL_i ) / #{valid i}
```

`KL^(ℓ)` is a **scalar per layer** (sum over allowed keys `j`; mean over valid queries `i`). These stack
across layers → `kl_stack`, shape `[L]`.

| metric | code | formula | reduction |
|---|---|---|---|
| **`indexer/kl`** | `:141` `kl_stack.sum()` | `Σ_ℓ KL^(ℓ)` | **the loss** — summed over layers |
| **`indexer/kl_layer_mean`** | `:143` `kl_stack.mean()` | `(1/L) Σ_ℓ KL^(ℓ)` | mean over layers |
| **`indexer/kl_layer_min`** | `:144` | `min_ℓ KL^(ℓ)` | min over layers |
| **`indexer/kl_layer_max`** | `:145` | `max_ℓ KL^(ℓ)` | max over layers |

Note `indexer/kl = L · kl_layer_mean`. `kl_layer_min`/`max` show the spread across layers (are some
layers learning while others are stuck).

---

## The diagnostic family (only every `diag_interval` forwards)

Gated by `_do_diag` (`minicpm_dsa.py:130`, computed under `no_grad`, `:257–289`). Each is computed per
query row, summed over valid rows / blocks, divided by valid-row count → **per-layer scalar**, then
`torch.stack(...).mean()` over layers (`:150–151`).

### `indexer/topk_recall` (`:262`, `:284`)

Of the base attention's probability *mass*, how much lands on the indexer's top-`k` picks. Let
`topk_I(i)` = indices of the `k` largest `I[i,:]`:

```
recall_i   = Σ_{j ∈ topk_I(i)} p[i,j]          ∈ [0,1]
recall^(ℓ) = ( Σ_{i ∈ valid} recall_i ) / #{valid i}
```

Sum over the `k` selected keys (mass-weighted); mean over valid queries, then over layers. **The metric
that actually matters** — it measures whether the sparse selection keeps the attention mass.

### `indexer/topk_overlap` (`:263–264`, `:285`)

Unweighted set agreement between the indexer's top-`k` and the *target's* own top-`k` (`topk_p(i)`):

```
overlap_i = | topk_I(i) ∩ topk_p(i) | / k     ∈ [0,1]
```

Difference from recall: overlap is by key *identity* (÷k), recall is by *mass*. A query can have high
recall but lower overlap if the missed keys carry little mass.

### `indexer/score_mean` / `indexer/score_std` (`:274–288`)

Health of the *raw* indexer scores `I` over finite, allowed entries `fin = I[allow]` (excludes
masked/pad keys). Streaming mean/var over `(i,j)` valid entries, per layer, then mean over layers:

```
μ = ( Σ fin ) / N
σ = sqrt( max( (Σ fin²)/N − μ², 0 ) )
```

Detects score collapse/explosion (dead ReLU → μ→0; blow-up → σ huge).

### `indexer/nan_frac` (`:278`, `:288`)

Fraction of valid `I` entries that are NaN, `d_nan / N`. Pure guard; should be exactly 0.

### `indexer/entropy` / `indexer/entropy_frac`

Entropy of the **student** distribution `q = softmax(I)` per query row — how peaked vs. spread the indexer's
selection is. `torch.special.entr` (`= −q·ln q`, with `entr(0)=0` so masked keys contribute nothing),
summed over keys → per-query entropy in **nats**; averaged over valid queries, then over layers.

```
entropy_i      = Σ_j  −q[i,j] · ln q[i,j]           (over allowed keys j)
entropy_frac_i = entropy_i / ln(#valid keys_i)      ∈ [0,1]   (1.0 = uniform; single-valid-key rows → 1.0)
```

`entropy_frac` normalizes by `ln(#allowed keys)` so it's comparable across query positions and sequence
lengths. **Read it as init/training health:** ~1.0 = near-uniform (healthy KL-distillation start — unbiased,
gentle gradients, maximal softmax responsiveness); a low or falling value flags a **saturated / over-committed**
indexer (some keys dominate). Since the score scale grows with `‖x‖`, deeper layers can start peakier — watch
for `entropy_frac` well below 1.0 at step 0, which would indicate the init temperature is too high (see
[`dsa_indexer_init_proposal.md`](dsa_indexer_init_proposal.md)). Empirically the per-fan-in init starts at
`entropy_frac ≈ 0.995` at unit input scale. The full derivation — how the init `σ`, the score spread
`std(I) ≈ 0.177·x_std`, and `entropy_frac ≈ 1 − C·x_std²` connect — is in
[`dsa_indexer_init_entropy_math.md`](dsa_indexer_init_entropy_math.md).

---

## Three reduction levels (easy to conflate)

1. **Within a layer**: sum over keys/blocks, divide by valid-query count → per-layer scalar.
2. **Across layers**: `kl` *sums* (`:141`); everything else *means* (`:143`, `:151`).
3. **Across micro-batches in a step**: `_reduce_metric` in `sft_trainer.py:400–417` takes the list-mean of
   the per-forward scalars before logging.

**Consequence:** because `indexer/kl` sums over `L` (=62) layers while `kl_layer_mean` averages,
`indexer/kl ≈ 62 × kl_layer_mean` — the headline loss magnitude (~712 noise in the debugging doc) is a
62-layer sum, and the per-layer figure is ~11–12.

---

## Interpreting the values

- **Synthetic random-token data → flat `topk_recall`/`overlap`.** With no real attention structure the
  target `p` is near-uniform, so recall/overlap sit near their random-chance floor (≈ `k/T`) and stay
  flat. Expected, not a bug (see [`dsa_grad_norm_debugging.md`](dsa_grad_norm_debugging.md), takeaway 6).
- **`nan_frac` must be 0.** Any nonzero value means NaNs leaked into the raw scores.
- **`kl_layer_min` ≪ `kl_layer_max`** indicates uneven per-layer learning.
- **`score_std → 0`** suggests a collapsing / dead-ReLU indexer.
