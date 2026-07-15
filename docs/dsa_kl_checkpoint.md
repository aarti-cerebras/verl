# DSA Indexer-KL Activation Checkpointing (`dsa_kl_checkpoint` / `KL_CKPT`)

How the `dsa_kl_checkpoint` flag makes the Phase-1 dense-warmup KL fit at 32K context on 4×H100, and why
it is structured as **two nested checkpoints**. This is *activation* (gradient) checkpointing — a runtime
memory/compute trade — **not** a disk checkpoint. Nothing is written to disk. (For the disk save→reload
round-trip see [`dsa_checkpoint_notes.md`](dsa_checkpoint_notes.md); for the KL loss definition see
[`dsa_kl_loss_math.md`](dsa_kl_loss_math.md).)

Config knob: `KL_CKPT` in `examples/dsa/run_minicpm3_dsa_phase1.sh` → `dsa_kl_checkpoint` in the model
override config → `IndexerConfig.kl_checkpoint` (`verl/models/transformers/dsa_indexer.py:86`). Only active
when `self.training` is true (backward exists); a no-op at eval / val-only.

## TL;DR

- At 32K the per-layer indexer-KL score graph is `O(n_layers · T²)` and cannot be retained — without
  checkpointing it OOMs immediately.
- Two nested `torch.utils.checkpoint.checkpoint` calls bound peak retained memory to **one layer × one
  query tile ≈ ~2 GB** instead of `62 layers × 32 tiles`.
- **Numerically identical** either way; cost is one extra forward recompute of the score einsum per tile
  in backward.
- Off by default (short context fits); turn **on** at long context (32K). See the memory note
  *DSA 32K on 4×H100*.

## The tensor that drives everything: `dots`

The indexer score, `verl/models/transformers/dsa_indexer.py:279` / `:283`:

```python
dots = torch.relu(torch.einsum("bqhd,bkd->bqhk", q_dq, k_dq))   # [bsz, B, n_heads, T]
eff_w  = (weights * softmax_scale).to(dots.dtype)
scores = torch.einsum("bqhk,bqh->bqk", dots, eff_w)             # [bsz, B, T]
```

`dots` is the `[bsz, B, n_heads, T]` per-head dot-product tensor. For the Phase-1 32K run
(`bsz=1` micro-batch/gpu, `B = kl_block_size = 1024`, `n_heads = 16`, `T = 32768`):

```
dots = 1 × 1024 × 16 × 32768  ≈  5.4e8 elements  ≈  2.1 GB  (fp32, per tile)
```

Backward through the `relu` needs the pre-activation, and the `bqhd,bkd->bqhk` einsum backward needs
`q_idx`/`k_idx`, so autograd must **retain a `dots`-sized activation per tile** until `.backward()` runs.

## Why the naive path OOMs

The KL is computed in a tile loop over the query axis (`verl/models/transformers/minicpm_dsa.py:285`,
`for q0 in range(0, T, block)`), `T/block = 32` tiles of 1024 queries. Each tile's KL contribution is
summed into a single scalar `total_kl` (`minicpm_dsa.py:315-318`). But summing into a scalar does **not**
free the per-tile graphs — each `kl_blk` keeps a live autograd subgraph back to its tile's `dots` until the
final `.backward()`. So without checkpointing, one layer holds **all 32 tiles simultaneously**:

```
32 tiles × 2.1 GB  ≈  68 GB   — for ONE layer
```

On top of the ~66 GB the frozen MiniCPM3-4B base already occupies, that overflows an 80 GB H100 →
`RuntimeError: CUDA out of memory`. Across all 62 layers it would be `62 × 68 GB ≈ 4 TB` — obviously
impossible.

### Why you can't just shrink `KL_BLOCK` instead

The total retained score graph per layer is `bsz × (Σ tile sizes = T) × n_heads × T` — **independent of
`block`**. A smaller `block` gives more, smaller tiles; the *sum* is unchanged (~68 GB). Tiling only bounds
the **forward transient** (one `p_blk`/`dots` built at a time), not the **backward-retained graph**. So
tiling alone cannot make it fit — you have to actually *not keep* the activations. That is what
checkpointing does; `KL_BLOCK` is pure tiling granularity with no effect on the retained graph or on
numerics.

## The two nested checkpoints

### Outer — per layer (`minicpm_dsa.py:466-471`)

```python
if getattr(dsa, "kl_checkpoint", False) and self.training:
    self._dsa_kl = torch.utils.checkpoint.checkpoint(
        _dense_warmup_kl, self, hidden_states, qr, query_states, key_states,
        cos, sin, position_ids, attention_mask, use_reentrant=False,
    )
```

Wraps the **entire** `_dense_warmup_kl` call. The whole KL branch for a layer is discarded after forward
and re-executed in that layer's backward. Bounds retention across *depth*: **one layer alive at a time**,
not 62.

`use_reentrant=False` is required — the checkpoint inputs (frozen base activations) don't require grad; the
grad-carrying tensors are the indexer params referenced *inside*, and the recompute must re-fire the
indexer's FSDP2 all-gather via `__call__` (see `dsa_fsdp_sharding_notes.md`, Option B2).

### Inner — per query tile (`minicpm_dsa.py:308-311`)

```python
if tile_ckpt:  # kl_checkpoint and attn.training
    kl_blk = torch.utils.checkpoint.checkpoint(
        _tile_kl, q_idx[:, q0:q1], k_idx, weights[:, q0:q1], bias, allow, p_blk,
        use_reentrant=False,
    )
```

Wraps each tile's `_tile_kl` (`minicpm_dsa.py:275-283`), which recomputes `dots`→`scores`→KL for that tile.
Bounds retention across the *sequence* dimension *within* a layer: **one tile alive at a time**, not 32.

### Why both are needed

The outer checkpoint alone is **insufficient**: even a single layer's fully-materialized tiled score graph
(~68 GB) overflows the GPU next to the frozen base. The inner checkpoint is what shrinks a single layer's
backward to one tile. They **nest** — the inner checkpoint fires during the outer's recompute:

| Config | Peak retained KL score memory |
|---|---|
| No checkpointing | `62 × 32 × 2.1 GB` ≈ 4 TB — impossible |
| Outer only | `1 × 32 × 2.1 GB` ≈ **68 GB** — OOM next to the ~66 GB base |
| **Outer + inner (`KL_CKPT=True`)** | `1 × 1 × 2.1 GB` ≈ **~2 GB** — fits |

## What gets recomputed in backward

Because the outer checkpoint re-runs all of `_dense_warmup_kl` during a layer's backward, the following
re-execute (per layer, per backward):

1. **Indexer projection** — `attn.indexer(hidden_states, qr, cos_g, sin_g)` (`minicpm_dsa.py:257`),
   producing `q_idx, k_idx, weights`. This is the grad-carrying part and **re-fires the indexer's FSDP2
   all-gather**.
2. **Teacher distribution `p_blk`** (`minicpm_dsa.py:292-298`), under `torch.no_grad()`: per query block and
   per attention head, `query_states[:,h] @ key_states[:,h].T · scale` → softmax → averaged over `H` heads.
   This is the dominant FLOP cost (`O(H·T²)`); it is detached but still re-runs because it lives inside the
   recomputed function.
3. **Indexer score graph `_tile_kl`** (`minicpm_dsa.py:275-283`): `dots` → `scores` → `log_softmax` → the KL
   summand `p·(log p − log q)` → masked sum. Under the inner checkpoint each tile's `dots` is dropped after
   forward and recomputed just-in-time in backward.

**Not** recomputed: the base transformer's own attention output (the normal frozen-base forward,
`minicpm_dsa.py:481+`) and anything outside the DSA branch.

## Cost / correctness

- **Numerics:** identical — checkpointing only recomputes, it does not approximate.
- **Compute:** one extra forward pass of the score einsum (+ the teacher softmax) per tile, per backward.
  At ~265 s/step for the 32K run this is an acceptable price for fitting in HBM.
- **Default:** `kl_checkpoint = False` (`dsa_indexer.py:86`). Short context (e.g. 4K) retains the whole
  score graph cheaply, so it is off. Turn on at long context.

## Related memory levers (32K)

`KL_CKPT` is the KL-specific lever. The other knobs in `run_minicpm3_dsa_phase1.sh` that trade for memory:

- `MODEL_DTYPE=bf16` — halves the frozen-base footprint (fp32 = accurate KL teacher but ~2× memory).
- `ACT_OFFLOAD=True` — CPU-offloads activations saved for backward; transparent, does not re-trigger the
  `_dsa_kl` side-effect (unlike `GRAD_CKPT`).
- `DIAG_OVERLAP_SAMPLE=256` — caps the `O(k²)` `topk_overlap` diagnostic tensor.
- `KL_BLOCK` — tiling granularity only (bounds the forward transient, **not** the retained graph).

See the memory note *DSA 32K on 4×H100* for the measured footprint (peak ~66/69 GB/GPU with these settings).
