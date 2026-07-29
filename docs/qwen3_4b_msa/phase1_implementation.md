# Phase 1 (indexer warm-up) — implementation and the optimisations that make 32K feasible

What Phase 1 computes, why the naive version needs 34.4 TB per layer, and the six changes that bring it
to a measured 46.9 GB. Written after the first successful 32K run on Qwen3-4B-Thinking-2507
(2026-07-29); every number below is either a formula over the notation in §1 or a measurement, and
measurements are labelled as such.

Companions: [plan.md](plan.md) (the plan), [kl_loss.md](kl_loss.md) (the loss spec and paper
provenance), [phase2_plan.md](phase2_plan.md) (the sparse stage).

---

## 1. Notation

| symbol | meaning | Qwen3-4B value |
|---|---|--:|
| `b` | batch size **per GPU** (global batch 8 over 8 GPUs) | 1 |
| `T` | sequence length (the paper's `N`) | 32,768 |
| `L` | total decoder layers | 36 |
| `L_sparse` | layers carrying an index branch (3–35; 0–2 stay dense) | 33 |
| `d_model` | hidden size | 2,560 |
| `d_ff` | MLP intermediate size | 9,728 |
| `H_q` | query heads | 32 |
| `H_kv` | KV heads **= GQA groups = index query heads** | 8 |
| `G` | query heads per group, `H_q / H_kv` | 4 |
| `d_h` | attention head dim | 128 |
| `d_idx` | index head dim (must be 128 for the vLLM kernels) | 128 |
| `B_k` | KV block size | 128 |
| `n_blocks` | `ceil(T / B_k)` | 256 |
| `k` | blocks selected per (query, group) | 16 |
| `k·B_k` | token budget per (query, group) | 2,048 |
| `T_q` | **query tile** = `kl_block_size` (the tiling knob) | 512 |
| `n_tiles` | `T / T_q` | 64 |

Tensors, indices in order:

| symbol | shape | what it is |
|---|---|---|
| `q`, `k`, `v` | `[b, H_q, T, d_h]`, `[b, H_kv, T, d_h]` | base projections, post-QK-norm and post-RoPE |
| `q_idx` | `[b, H_kv, T, d_idx]` | index queries — **one per group**, not per head |
| `k_idx` | `[b, 1, T, d_idx]` | the single shared index key (MQA-shaped; paper Eq. 5) |
| `S_idx` | `[b, H_kv, T_q, T]` | token-level index scores = the KL **student** logits |
| `P` | `[b, H_kv, T_q, T]` | Eq. 9 **teacher**: per-head softmax, then the `1/G` average |
| `bias` | `[b, T_q, T]` | additive causal / document / padding mask (`0` or `−inf`) |
| `allow` | `[b, T_q, T]` bool | `bias == 0`, i.e. attendable |

Byte sizes: bf16 = 2 B, fp32 = 4 B, bool = 1 B. All code references are
`verl/models/transformers/qwen3_msa.py` unless stated otherwise.

---

## 2. What Phase 1 is for

Top-`k` selection is non-differentiable, so the LM loss cannot teach the index branch which blocks to
pick — MSA §3.2: *"the top-k selection in Equation 7 is non-differentiable, so the language-modeling
loss cannot train the index Q/K projections directly."* The index branch needs its own supervision, and
it needs it **before** it starts routing the main branch, or the model attends to noise while the
indexer is still random (paper B.4/Fig. 10–11).

So Phase 1 = *let the index branch watch dense attention and learn to imitate it, while changing nothing
about the model.* Two properties make it safe, both measured rather than assumed:

* **The LM forward is bit-identical to stock Qwen3** — `max|Δ logits| = 0.000e+00`
  (`tests/msa/test_qwen3_msa_phase1.py` §1). Attention stays dense, and the index branch has no output
  path: the paper's final recipe drops the index value head (C.3) and vLLM sets
  `sparse_disable_index_value` on every sparse layer. There is no mechanism by which Phase 1 can damage
  the model.
* **Only `*.indexer.*` parameters receive gradient** — 0 base tensors, verified in both directions.

---

## 3. The problem statement

Phase 1 trains `{W_q_idx, W_k_idx, index_q_norm, index_k_norm}` by minimising, per sparse layer,

```
L_KL = 1/(T · H_kv) · Σ_{i=1..T} Σ_{r=1..H_kv}  D_KL( P^(r)_i ‖ softmax(S_idx^(r)_i) )
```

over the **full causal support** — every query attends over all `T` keys. That needs two distributions
of shape `[b, H_kv, T, T]`:

```
teacher  P      :  b · H_kv · T · T · 4 B  =  1 · 8 · 32768² · 4  =  34.4 TB   per layer
student  S_idx  :  same                                          =  34.4 TB   per layer
```

against **79.2 GB per GPU**. Everything in §4 is about closing that gap.

---

## 4. The six optimisations

### 4.1 Query tiling — tile the QUERY axis only (`_dense_warmup_kl`, the `for q0` loop)

Process `T_q = 512` query rows at a time. Each tile still scores against **all `T` keys**, because the
KL support is the whole row — only the query axis can be tiled.

```
per tile:  b · H_kv · T_q · T · 4 B  =  1 · 8 · 512 · 32768 · 4  =  537 MB
```

`34.4 TB → 537 MB`. The index projections are hoisted **out** of the loop, so `q_idx`/`k_idx` are
computed once per layer rather than `n_tiles` times — which also gives FSDP2 one all-gather per layer
instead of 64.

### 4.2 Per-tile checkpointing (`_tile_kl`)

Inside a tile, three fp32 `[b, H_kv, T_q, T]` tensors are live — `s`, `log_q`, `term` — i.e.
`3 × 537 MB = 1.61 GB`. Without checkpointing they are retained for backward, per tile, per layer.
Checkpointing recomputes them instead.

**Measured** (`tests/msa/test_qwen3_msa_memory_scaling.py`, T=4096 on Qwen3-0.6B):
**32.52 GB → 5.05 GB, a 6.4× saving.**

### 4.3 Build derived tensors INSIDE the checkpoint — the change that unblocked 32K

`torch.utils.checkpoint` saves its **input tensors** so it can replay the function in backward, and
holds them on the graph node until backward runs. Anything allocated per tile and passed *in* is
therefore retained for the whole layer:

```
retained per tile = P      b·H_kv·T_q·T·4  = 537 MB
                  + bias   b·T_q·T·4       =  67 MB
                  + allow  b·T_q·T·1       =  17 MB
                                             621 MB
× n_tiles   (64)  = 39.7 GB  per layer
× L_sparse  (33)  = 1.31 TB
```

The fix passes only **tile-invariant** tensors and rebuilds the rest inside:

```python
def _tile_kl(q_t, k_all, q_states, k_states, pos_ids, kmask, q0, q1):
    bias_t  = _causal_doc_bias_block(pos_ids, q0, q1, T, q_t.device, key_mask=kmask)
    allow_t = bias_t == 0.0
    with torch.no_grad():
        p_t = _group_teacher(q_states, k_states, q0, q1, bias_t, H_kv, scaling)
    s     = attn.indexer.scores(q_t, k_all, attn_bias=bias_t)
    log_q = torch.log_softmax(s, dim=-1)
    term  = p_t * (torch.log(p_t.clamp_min(_KL_EPS)) - log_q)
    return torch.where(allow_t.unsqueeze(1), term, torch.zeros_like(term)).sum(dim=-1)

args = (q_idx[:, :, q0:q1], k_idx, query_states, key_states, position_ids, key_mask, q0, q1)
kl_blk = torch.utils.checkpoint.checkpoint(_tile_kl, *args, use_reentrant=False)
```

`query_states`, `key_states`, `k_idx`, `position_ids` are the **same objects** for all 64 tiles, so they
are stored once per layer; `q_idx[:, :, q0:q1]` is a view; `q0`/`q1` are ints. **1.31 TB → 0.**

**The property that matters.** Retention was `n_tiles × per_tile`, and `n_tiles × T_q = T`, so the total
is **independent of `T_q`**: halving the tile doubles the count. This is why `T_q` 512 → 256 changed
nothing, and why `kl_block_size` — which [plan.md](plan.md) §5 and [kl_loss.md](kl_loss.md) §5 both
present as *the* memory lever — does not control this term at all. It bounds only the transient working
set of §4.1/§4.2.

**Measured:** OOM at 77.2 of 79.2 GB (dying on a 256 MiB request) → runs at **46.9 GB**.

### 4.4 Teacher accumulated over GROUPS, not heads (`_group_teacher`)

Eq. 9 requires a softmax per *head* (`H_q = 32`) before the `1/G` average. Materialising all heads at
once costs

```
b · H_q · T_q · T · 4 B  =  1 · 32 · 512 · 32768 · 4  =  2.15 GB   per tile
```

so it must be chunked. Two choices:

| chunk | working set per chunk | matmuls per tile |
|---|--:|--:|
| one head, `b·1·T_q·T·4` | 67 MB | `H_q` = **32** |
| **one group, `b·G·T_q·T`** (bf16 134 MB → fp32 softmax 268 MB) | 268 MB | `H_kv` = **8** |

Per-head was memory-minimal but issued `H_q · n_tiles · L_sparse = 32 · 64 · 33 = 67,584` small fp32
matmuls per sequence — launch-bound rather than FLOP-bound. Per group:

```python
for r in range(n_kv_heads):                                                    # H_kv = 8 iterations
    qg = qb[:, r * G : (r + 1) * G]                                            # [b, G, T_q, d_h]
    s  = torch.matmul(qg, key_states[:, r:r+1].transpose(-1, -2)) * scaling    # model dtype
    p[:, r] = torch.softmax(s.float() + bias.unsqueeze(1), dim=-1).sum(dim=1)  # fp32 only here
return p / G
```

**Measured:** teacher alone **13.8 → 9.7 ms/tile (1.43×)**, peak 1.68 → 1.74 GB, numerics differing by
3.3e-4 (fp32 accumulation order — the Eq. 9 ordering test still matches its reference to 1e-12).
End-to-end **~80 → ~73 s/step (1.10×)**.

That 1.43 → 1.10 gap implies the teacher is only **~30% of step time**, not the ~68% originally assumed.
The claim "the teacher dominates Phase 1" is **wrong** and is corrected here.

### 4.5 The frozen base — the largest win, and it is free (`freeze_base_train_indexer`)

```python
p.requires_grad_(".indexer." in name)
```

With `requires_grad=False` on the base, autograd retains **no base activation graph at all**. What that
avoids, per layer at `T = 32768` in bf16:

| tensor | formula | size |
|---|---|--:|
| hidden × 2 | `b·T·d_model·2` | 336 MB |
| q proj | `b·T·H_q·d_h·2` | 268 MB |
| k, v | `b·T·H_kv·d_h·2` each | 134 MB |
| **MLP gate, up** | `b·T·d_ff·2` each | **1,276 MB** |
| per layer | | ~2,014 MB |
| **× L = 36** | | **72.5 GB** |

Phase 1 pays none of it. This is exactly why Phase 1 fits at 46.9 GB while Phase 2 (base training) does
not, and why `tiled_mlp` — which recomputes those MLP intermediates — is a **Phase-2** fix that
contributes nothing here.

### 4.6 FSDP2 Option-B2 — correctness, not memory (`verl/utils/fsdp_utils.py`)

`L_KL` never flows through the decoder layer's output, so the layer unit's backward gates never fire and
the reduce-scatter onto the sharded optimizer master never runs → **nonzero-but-fake `grad_norm` with a
flat loss** (`docs/dsa_fsdp_sharding_notes.md` §3b). Each indexer is wrapped as its own FSDP2 unit,
whose output *does* require grad:

```python
_indexer_cls_names = ("LightningIndexer", "MSAIndexer")   # keyed on the class NAME
indexer_kwargs = {**fsdp_kwargs, "reshard_after_forward": False}
```

This predicate previously matched DSA's class only, so `MSAIndexer` silently fell through to the broken
path — it would have surfaced as a flat loss at world_size > 1, not as an error. Prerequisite: the
indexer must be called through `nn.Module.__call__`, never as a bare method, or FSDP2's forward hooks
never fire.

---

## 5. Correctness details the notation makes precise

**The normaliser counts (query, group) PAIRS**, giving Eq. 10's `1/(T · H_kv)` rather than `1/T`:

```python
total_cnt = total_cnt + qv.sum() * H_kv
```

Using *valid* (non-pad) rows rather than the literal sequence length also means padding cannot dilute
the loss.

**Masked entries must be exact zeros, not `P = 0`.** A fully-masked padding row is `−inf` everywhere, so
its teacher row is NaN and `0 × NaN = NaN` would poison the layer:

```python
torch.where(allow_t.unsqueeze(1), term, torch.zeros_like(term)).sum(dim=-1)
```

**The layer reduction is a pure gradient scale.** Per-layer indexer parameters are disjoint, so
`∂L/∂θ_i = ∂KL_i/∂θ_i` and `sum == mean × L_sparse` — same optimum, gradient scaled by `L_sparse`.
Default `mean` keeps `grad_norm` at 1.0–2.8 instead of ~33× that, which matters because verl's
`clip_grad` is global. Asserted in `test_qwen3_msa_phase1.py` §1b
(`mean 4.0021 × 25 == sum 100.0520`). Consequence for Phase 2: a paper `λ` needs
`λ_ours = λ_paper × L_sparse`.

**Diagnostics rebuild their own tensors under `no_grad`** and are gated on `diag_interval`, so the
coverage family's extra teacher pass retains nothing.

---

## 6. Measured state (2026-07-29, Qwen3-4B-Thinking-2507, 8×H100, real longmino data)

| | |
|---|--:|
| peak memory | **46.9 GB** of 79.2 (≈32 GB headroom) |
| step time | **~73 s** for `b · T · 8 GPUs` = 262,144 tokens |
| throughput | **3,590 tok/s** |
| KL over 6 steps | **6.11 → 1.45** |
| `grad_norm` | 1.0–2.8 |
| indexers attached | 33 sparse / 3 dense, as configured |

Budget implication, replacing plan §8's FLOP-derived estimate:

| tokens | wall clock at 3,590 tok/s |
|---|--:|
| 0.5B | ~1.6 days |
| 1B | ~3.2 days |
| 3B (all data built) | ~9.7 days |

Plan §8 predicted "½–1 day" for 0.5–1B; the measurement is **~3–5× slower**. The 46.9 GB has **not**
been decomposed against a predicted ledger (an a-priori estimate gave ~12 GB), so the difference —
likely base-forward transients not freed promptly across the layer loop, plus fragmentation — is
unexplained and worth a memory snapshot before assuming the headroom is usable.

Since the teacher is only ~30% of step time (§4.4), the remaining ~70% is unprofiled. Removing the
teacher *recompute* via a manual per-tile backward (`dL/dS_idx = P_idx − P` is closed form) is worth at
most ~15%, so profiling should precede any further optimisation. The cheapest untried lever is a
`kl_block_size` sweep at 1024/2048: it does not affect retained memory (§4.3) but means fewer Python
iterations and larger kernels, and there is headroom to spend.

---

## 7. Regression guard

`tests/msa/test_qwen3_msa_memory_scaling.py` asserts the invariant that §4.3 violated:

> **At fixed `T`, peak memory must not grow with `n_tiles`.**

```
kl_block_size=2048  tiles= 2  peak=5.087 GB
kl_block_size=1024  tiles= 4  peak=5.053 GB
kl_block_size= 512  tiles= 8  peak=5.053 GB
kl_block_size= 256  tiles=16  peak=5.053 GB      → max/min = 1.01x
```

It also asserts that `kl_checkpoint=True` peaks *below* `False` (5.05 vs 32.52 GB), i.e. that
checkpointing reduces memory rather than merely relocating it.

It runs at `T = 4096` deliberately. The other suites use `T = 512` with `T_q = 128`, i.e. 4 tiles, where
64× retention is invisible — which is why this bug survived 24/24 + 14/14 + 17/17 passing checks until
the first 32K run. **Any future tiled-loss work should be validated at a tile count ≥ 16.**
