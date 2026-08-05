# Phase 1 (indexer warm-up) — implementation and the optimisations that make 32K feasible

What Phase 1 computes, why the naive version needs 34.4 TB per layer, and the seven changes that bring it
to a measured 46.9 GB and 42 s/step. Written after the first successful 32K run on Qwen3-4B-Thinking-2507
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

## 4. The seven optimisations

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

**That 1.43 → 1.10 inference was itself wrong, and is retracted here.** It concluded the teacher was
"~30% of step time"; direct profiling of the step put it at **56%** (20.4 s of a 73 s step, 40.7 s once
the checkpoint recompute is counted). A component speedup does not divide into an end-to-end ratio like
that — the two measurements were taken under different diagnostics and validation settings, so the ratio
was never a clean lever arm. **Measure step composition directly; never infer a component's share from
two end-to-end numbers.** §4.7 is what the correct number bought.

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

### 4.7 `torch.compile` the teacher — and the int-argument trap that silently disabled it

With the teacher measured at 56% of the step (§4.4), it is the right target. It is **bandwidth-bound, not
compute-bound**: eager runs at ~14 TFLOP/s (~3% of an H100's bf16 peak) because `s.float() + bias`, the
softmax and the group-sum each round-trip a 268 MB fp32 tensor through HBM — ~10 GB of traffic per tile.
Compiling fuses that chain. At 32K/`T_q`=512 on one H100, against an **fp64** reference:

```
eager           9.65 ms   peak 1.74 GB   err 1.62e-04
torch.compile   3.57 ms   peak 2.15 GB   err 8.42e-05     <- 2.70x AND more accurate
```

More accurate because Inductor keeps more of the reduction in registers instead of materialising fp32
temporaries. (TF32 is irrelevant: `allow_tf32` is False on this build and toggling it changes neither
timing nor error.) **Validate a replacement against a higher-precision reference, not against the thing
being replaced** — comparing compiled-to-eager shows only that they differ, not which is right.

**The trap.** The first wiring passed the tile bounds as Python ints:

```python
def _group_teacher_impl(query_states, key_states, q0, q1, bias, ...):   # WRONG
    p = query_states.new_zeros(bsz, n_kv_heads, q1 - q0, T, ...)
```

Dynamo guards on an int argument's **value**, so every tile is a fresh specialisation. At
`kl_block_size=512` and `T`=32768 there are 64 tiles, so the 8th tripped `config.recompile_limit (8)`:

```
torch._dynamo hit config.recompile_limit (8)
   function: '_group_teacher_impl'
   last reason: 0/7: q0 == 3584
```

after which **every remaining tile ran eager** — 8 compilations paid for ~1/8 of the benefit. This is
invisible without reading the warnings: the loss is correct and the step merely slower. The fix is to
slice the tile *outside* the compiled region so no varying int argument reaches it, taking `T` from
`key_states` rather than from the now-shorter query tensor:

```python
q_block = query_states[:, :, q0:q1, :]          # in the wrapper, OUTSIDE the compiled fn
return _TEACHER_COMPILED(q_block, key_states, bias, n_kv_heads, scaling)
```

Verified with `torch._dynamo.utils.counters`: **`unique_graphs == 1` across 16 tiles** (was 8 + fallback).
Any tensor-shape-invariant loop passing loop indices into a compiled callee has this bug latent.

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
coverage family's extra teacher pass retains nothing. `_do_diag` is
`(not model.training) or ((cnt - 1) % interval == 0)` — validation **always** diagnoses, which is why the
per-layer gate panel is read off val rather than off the interval-gated training steps.

**Neither `k` nor `kl_block_size` affects the Phase-1 loss — only the diagnostics.** This follows from
the loss being token-level over the *full* key axis (`log_softmax(s, dim=-1)` over all `T` in `_tile_kl`),
so the selection budget never enters it: `top_k` and `block_size` appear only in
`block_selection_metrics` and in the Phase-2 sparse path. Measured on Qwen3-0.6B, `T`=2048, fp32, 28
sparse layers, identical indexer init per variant:

```
kl_block=256  k=4    KL 4.125661850   block_recall 0.641   main_attn_covered 0.677
kl_block=512  k=4    KL 4.125661850         0.641                0.677
kl_block=1024 k=4    KL 4.125661850         0.641                0.677
kl_block=2048 k=4    KL 4.125661850         0.641                0.677
kl_block=512  k=2    KL 4.125661850         0.569                0.498
kl_block=512  k=8    KL 4.125661850         0.840                0.871
kl_block=512  k=16   KL 4.125661850         1.000                1.000
```

`ΔKL` is **exactly** 0.000e+00 in all seven — `kl_block_size` is pure query-axis tiling (memory/speed
only, and §4.7's compile-guard granularity), while `k` moves only the metrics. Hence the practice of
holding `k` at the Phase-2 deploy value during Phase 1 (plan §5): it costs nothing in the loss and keeps the gate
reading predictive. `block_size` is not a knob at all — `MSAConfig.__post_init__` rejects anything but
128, matching vLLM's `SPARSE_BLOCK_SIZE`. (At `k`=16 with `T`=2048 the metrics saturate at 1.0 because
16×128 spans the whole sequence; keep `k·B_k < T` for the numbers to mean anything.)

---

## 6. Measured state (2026-07-29, Qwen3-4B-Thinking-2507, 8×H100, real longmino data)

| | |
|---|--:|
| peak memory | **46.9 GB** of 79.2 (≈32 GB headroom) |
| step time | **42 s** for `b · T · 8 GPUs` = 262,144 tokens (was 73 s before §4.7) |
| throughput | **6,241 tok/s** |
| `grad_norm` | 1.0–2.8 |
| indexers attached | 33 sparse / 3 dense, as configured |

Step time is from the production run
`p1_qwen3-4b-thinking-2507_longmino_1.00Bt_L32k_bs8_k16_B128_dp3_lr1e-3_20260729_212329`
(`ml-eng-gpu-09`, `DIAG_INTERVAL=10`): per-step deltas 42, 42, 43, then 210 s across steps 4→9 = 42.0 s.
Step 1 costs 97 s (Inductor compile + step-1 diagnostics). Cumulative average at step 9 is **43.09 s/it**,
the excess being the 1-in-10 diagnostics step — so **43 s is the all-in planning number, 42 s the
marginal one.**

The 73 → 42 s drop is attributable to §4.7 and **cross-checks against the profile**: teacher at 56% of a
73 s step ≈ 38 s (with recompute), 38/2.70 ≈ 14 s, giving 30 + 14 = 44 s ≈ the observed 42–43 s. Two
independent measurements agreeing is what promotes this from coincidence to explanation.

Budget implication, replacing plan §8's FLOP-derived estimate:

| tokens | steps at `bs`=8, `T`=32K | wall clock at 43 s/step |
|---|--:|--:|
| 0.5B | 1,908 | **~0.95 days** |
| 1B *(the launched run)* | 3,815 | **~1.9 days** |
| 3.02B (all data built) | 11,526 | **~5.7 days** |

Plan §8 predicted "½–1 day" for 0.5–1B; 0.5B now lands inside that, 1B at ~2× it. Validation (15 passes
× 64 windows at `TEST_FREQ`=250) and 76 hourly checkpoint saves together add well under an hour.

The 46.9 GB has **not** been decomposed against a predicted ledger (an a-priori estimate gave ~12 GB), so
the difference — likely base-forward transients not freed promptly across the layer loop, plus
fragmentation — is unexplained and worth a memory snapshot before assuming the headroom is usable.

Remaining levers, now that the teacher is fused: removing the teacher *recompute* via a manual per-tile
backward (`dL/dS_idx = P_idx − P` is closed form) is the largest, and a `kl_block_size` sweep at
1024/2048 the cheapest — it does not affect retained memory (§4.3) but means fewer Python iterations and
larger kernels, and there is headroom to spend. Note `kl_block_size` is now **also** a `torch.compile`
consideration: it sets the tile count, and §4.7's guard must hold for whatever value is chosen.

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

---

## 8. The logged metrics, defined

Every key that reaches wandb, with its equation. Notation from §1; all code refs are
`verl/models/transformers/qwen3_msa.py` unless stated. Two levels of aggregation apply to everything
below, in this order:

1. **Within a layer**, each quantity is a per-`(batch, group, query-row)` tensor, accumulated as a
   `rows`-weighted sum over query tiles and divided by `rows = Σ valid_rows · H_kv` (`_finalize_diag`).
   Weighting by valid rows means padding cannot dilute any metric.
2. **Across layers**, `_post_hook` takes an unweighted `.mean()` over the `L_sparse = 33` sparse layers.
   With `LOG_PER_LAYER=true` the per-layer values are *also* emitted as separate keys, because the gate is
   per layer and a mean hides the worst one.

Symbols used below, all per query row `i` and group `r`:

| symbol | definition |
|---|---|
| `P_b[c]` | teacher attention mass in block `c` = `Σ_{t ∈ block c} P_i[t]`, so `Σ_c P_b[c] = 1` |
| `V` | the set of causally **visible** blocks for row `i`: `c < (pos_i + B_k) // B_k` |
| `I*` | oracle top-`k` blocks by teacher mass, **restricted to `V` before the top-k** |
| `Î` | the index branch's selection, `select_blocks(M, ·)`, `-1` slots dropped |
| `F` | the **forced** blocks ⊂ `Î`: the `local_blocks` most recent + `init_blocks` sink, ∩ `V` |
| `mass(A)` | `Σ_{c ∈ A} P_b[c]` — teacher mass carried by block set `A` |

### 8.1 The loss itself

| key | equation | notes |
|---|---|---|
| `indexer/kl` | `L_KL` per §3, i.e. `1/(T·H_kv) Σ_i Σ_r D_KL(P_i^(r) ‖ softmax(S_idx,i^(r)))` | **The only optimised quantity.** Logged as the interpretable per-batch mean, *not* the length-normalised value actually backpropagated (`losses.py:73`). Token-level, over the full causal support — so `k` and `block_size` do not enter it (§5). |
| `indexer/kl_layer_mean` / `_min` / `_max` | `mean/min/max` over the 33 per-layer `L_KL` values | `kl_layer_mean == indexer/kl` under `kl_reduction=mean`. The **spread** is the useful part: `min` and `max` identify which layers are hard. |
| `indexer/n_sparse_layers` | `|{layers with an index branch}|` | Constant 33. A tripwire: if this ever drops, layers silently stopped contributing. |
| `kl_by_layer/L<ii>` | per-layer `L_KL` | Starts at `L03` — layers 0–2 are dense by `dense_prefix=3`. |
| `indexer/kl_share_of_loss` | `λ · L_KL / L_LM` | **Phase 2 only** (`losses.py:124`); Phase 1 has no `L_LM`. Neither paper publishes a `λ`, so this ratio is how it gets chosen (target ~0.05–0.2). |

### 8.2 The coverage family — how much attention mass the selection keeps

All from `block_selection_metrics`, which is unit-tested against the worked example in
[kl_loss.md](kl_loss.md) §1.2 step 7.

| key | equation | range / reading |
|---|---|---|
| `indexer/main_attn_covered` | `mass(Î)` | **Absolute** fraction of teacher attention mass the selected blocks carry. This is the deployment-relevant number: it is what the sparse forward will actually see. |
| `indexer/coverage_ceiling` | `mass(I*)` | The best any selector could do at this `k` — so the gap to `main_attn_covered` is selector error, not budget error. |
| `indexer/coverage_from_forced` | `mass(F)` | Mass the **forced** blocks alone carry, with zero learning. On a causal LM the local block dominates. |
| `indexer/coverage_vs_ceiling` | `mass(Î) / mass(I*)` | Oracle-relative coverage. |
| `indexer/learned_coverage` | `clamp( (mass(Î) − mass(F)) / (mass(I*) − mass(F)), 0, 1)` | **The metric that isolates skill.** 0 = the learned slots add nothing beyond forcing; 1 = they reach the ceiling. |

**Which of these is the Phase-1 gate — and a correction.** `coverage_vs_ceiling` is *not* a sound gate on
its own, because it is dominated by the forced local block. The in-code measurement (Qwen3-0.6B, `k`=16)
is explicit: the forced block alone captures **0.783** against an oracle of 0.979, so an **untrained**
indexer already reads `coverage_vs_ceiling = 0.903` — a 0.90 threshold **passes at initialisation**. Use
`learned_coverage` as the gate and read `coverage_vs_ceiling` as the absolute deployment number. Earlier
notes in this doc series that treat `coverage_vs_ceiling ≥ 0.90` as *the* gate are wrong in exactly this
way; plan §5 item 1 should be read with `learned_coverage` substituted.

### 8.3 The paper's two recall metrics

Verbatim from MSA §5.2 (p.9): *"let `I*` be the corresponding Top-k block set induced by the Main Branch
scores and let `Î` be the Index Branch selection. Block recall is `|I* ∩ Î|/|I*|`, while score recall is
`Σ_{b ∈ I*∩Î} P_b / Σ_{b ∈ I*} P_b`."*

| key | equation | notes |
|---|---|---|
| `indexer/block_recall` | `|I* ∩ Î| / |I*|` | Set identity, mass-blind: missing a high-mass block and a zero-mass one cost the same. |
| `indexer/score_recall` | `mass(I* ∩ Î) / mass(I*)` | Mass-weighted, so it is the more meaningful of the two. |

Both are **oracle-relative**, so neither can detect a budget that is simply too small — `I*` is itself
capped at `k`. That is precisely why `main_attn_covered` exists. Because masking restricts `I*` to `V`
before the top-k, `|I*| = |Î| = min(k, |V|)` and the paper's `|I*|` denominator is literally correct;
without that masking, early query rows would report spuriously low recall.

Note `main_attn_covered ≥ mass(I* ∩ Î)` by construction, hence
`coverage_vs_ceiling ≥ score_recall` — the gap is mass the index branch found in blocks the oracle did
not rank top-`k`. There is **no** guaranteed ordering between `score_recall` and `block_recall`
(measured 0.621 vs 0.689 at random init), so do not treat one exceeding the other as an invariant.

### 8.4 Per-group behaviour

| key | equation | reading |
|---|---|---|
| `indexer/group_disagreement` | `clamp( (|⋃_r Î_r| − mean_r|Î_r|) / (mean_r|Î_r| · (H_kv − 1)), 0, 1)` | 0 = all 8 groups pick identical blocks; 1 = fully disjoint. |

This is a **capacity-utilisation** check, not a quality one. The paper reports that groups *do* diverge
(Appendix A / Fig. 5: different groups attend to different long-range stripes while sharing local and
sink patterns), so a value pinned near 0 means the per-group index capacity MSA is built around is being
wasted — 8 groups paying for what one would deliver.

### 8.5 Health / saturation

Computed per group over the visible support, `valid_log = log(max(|allow_i|, 2))`:

| key | equation | reading |
|---|---|---|
| `indexer/entropy_norm` | `H(softmax(S_idx,i^(r))) / log|allow_i|` | **Student** entropy, normalised so 1.0 = uniform. ~1.0 at init is a healthy distillation start; a **falling** value flags an over-committed or saturated index branch. |
| `attn/entropy_norm` | `H(P_i^(r)) / log|allow_i|` | **Teacher** entropy, same normalisation. The floor the student is chasing; it is a property of the frozen base and should be ~flat. |
| `indexer/score_mean`, `indexer/score_std` | mean and `sqrt(E[S²] − E[S]²)` of `S_idx` over unmasked entries | Drift detector. A growing `score_std` with falling `entropy_norm` is the signature of logit blow-up rather than learning. |
| `indexer/nan_frac` | `|{NaN entries of S_idx}| / |unmasked entries|` | Must stay exactly **0**. Non-zero means the index projections diverged. |

### 8.6 Per-layer keys

With `LOG_PER_LAYER=true`, three of the above are additionally emitted per layer as
`indexer/<key>_by_layer/L<ii>` for `learned_coverage`, `coverage_vs_ceiling`, `block_recall`, plus
`kl_by_layer/L<ii>`. Separate keys, deliberately — a logger that averaged them would hide the worst
layer, and the gate is per layer.

### 8.7 When each metric is computed

`_do_diag = (not model.training) or ((cnt - 1) % diag_interval == 0)`. So the whole of §8.2–§8.5 is
computed every `DIAG_INTERVAL`-th training forward but **on every validation batch** — validation is
where the full panel is always available, which is why the gate is read off `val/`. §8.1 is computed
every step.

One asymmetry worth knowing: in **Phase-2 training** only `group_disagreement` is computed, because the
coverage family needs the dense teacher, which the sparse path does not build. `_finalize_diag(full=False)`
therefore emits *only* that key rather than zeros — logging an uncomputed metric as `0.0` would draw a
flat zero line that is indistinguishable from a collapsed indexer.
