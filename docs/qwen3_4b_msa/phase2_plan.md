# Qwen3-4B MSA — Phase 2 (sparse) implementation plan

Phase 1 trains the index branch against a frozen, densely-attending base. Phase 2 turns sparsity **on**:
the Main Branch attends only to the `k = 16` selected blocks, and `L_KL` moves to the restricted support.

Split in two, per plan §6:

| | base | loss | tokens | purpose |
|---|---|---|---|---|
| **2a** | **frozen** | `λ · Σ_layers L_KL` (LM loss measured, not trained) | 0.2–0.5B | **measure the pure sparsity cost** with zero drift confound |
| **2b** | trained | `L_LM + λ · Σ_layers L_KL` (+ minority BC) | 1–2B+ | recover the cost by reshaping attention to be block-friendly |

Status: **not started.** Phase 1 module (`verl/models/transformers/{msa_indexer,qwen3_msa}.py`) is built and
tested; Phase 2 adds one forward path, one loss, and the metrics below.

Prior art to mirror: `minicpm_dsa.py::{_sparse_attn,_sparse_indexer_kl,_sparse_attn_and_kl}` and
`verl/workers/utils/losses.py::dsa_sparse_loss`. Papers: MSA arXiv 2606.13392 §3.1 (Eq. 8), §3.2
(Eq. 9–11), Algorithm 1, appendices B.3/C.3; DeepSeek-V3.2 arXiv 2512.02556 §2.1.1 (Eq. 3–4) for the
contrast in §3.3.

---

## 1. The sparse forward — design decision

**Selection is per query TOKEN, not per query block.** MSA Eq. 7 indexes `I^(r)_i` by query position `i`,
and vLLM's kernel is explicit: *"block_size_q is 1 for M3, so top-k is computed per query token"*
(`common/ops/index_topk.py:167`). Anything coarser changes the function we serve.

That rules out the cheapest option:

| option | exact per-token? | verdict |
|---|---|---|
| **Gather K/V at the selected token indices** (DSA's `_sparse_attn`) | ✅ | **chosen** — same cost class as the MiniCPM3 Phase-2 runs (plan §8) |
| FlexAttention with a `BlockMask` at 128×128 granularity | ❌ — all 128 queries in a q-block would share the union of their selections | rejected: silently changes the attended set |
| FlexAttention with a pointwise `mask_mod` | ✅ | correct, but with per-token selection almost every (q_block, kv_block) pair is *partial*, so it skips little work — no FLOP win over dense. Revisit only as a memory play |
| Union-per-q-tile gather + per-token mask inside the union | ✅ (mask restores semantics) | **the optimisation to try if step time demands it.** Union of 512 queries × 16 blocks approaches all 256 blocks at 32K; at `T_q = 128` it should be 40–80 blocks, a 3–6× saving. Measure before building |

### 1.1 HF gradient checkpointing must stay OFF — for a subtler reason than Phase 1

In Phase 1 it was merely unnecessary (frozen base). In Phase 2 the base trains and normally *wants* it,
but it is still forbidden: `_msa_kl` is stashed as a **side effect** on the attention module. Under HF's
`gradient_checkpointing_enable`, the first forward runs under `no_grad`, so the KL tensor we stash has
**no graph**; recompute overwrites it, but the loss has already captured the graph-less one → the KL
silently contributes zero gradient. The DSA launch script carries the same warning.

The supported pattern instead: the module checkpoints itself. One `torch.utils.checkpoint(...,
use_reentrant=False)` call per layer wrapping a function that **returns both** `(attn_output, kl)`, so both
stay in the live graph. That is exactly what `_sparse_attn_and_kl` does. For the FFN, use verl's
`tiled_mlp` rather than global checkpointing.

---

## 2. The exact computation sequence

One sparse layer, one query tile. Shapes for Qwen3-4B at 32K, batch 1, `T_q = 512`: 32 query heads,
8 KV heads (so **4 heads per group**), head dim 128, blocks of 128 tokens → 256 blocks, `k = 16` blocks
→ 2048 tokens.

### A. Base projections — identical to stock Qwen3

1. Project to `q` `[1, 32, 32768, 128]`, `k` and `v` `[1, 8, 32768, 128]`; per-head RMSNorm on `q`/`k`.
2. RoPE on `q` and `k`.

### B. Index branch — from **detached** hidden states (Eq. 11)

3. `x = hidden_states.detach()`.
4. `q_idx = RoPE(index_q_norm(index_q_proj(x)))` → `[1, 8, 32768, 128]` — one index query per group.
5. `k_idx = RoPE(index_k_norm(index_k_proj(x)))` → `[1, 1, 32768, 128]` — one shared index key.

### C. Selection

6. `S_idx = q_idx_tile @ k_idxᵀ / √128` → `[1, 8, 512, 32768]`; add `-inf` for non-causal, cross-document
   and padding keys. **This is dense over all keys** — see §3.2.
7. Max-pool each 128-token block → `[1, 8, 512, 256]`; blocks with no visible key → `-1e30`.
8. Overwrite the block containing the query with `1e29` so it always wins (paper §3.2 Local Block).
9. `sel = topk(16).indices` → `[1, 8, 512, 16]` block ids, **detached** (top-k is non-differentiable).
10. Expand block ids to token positions → `[1, 8, 512, 2048]`, plus a validity mask for causally
    invisible slots.

### D. Sparse attention — the LM path

11. Gather selected keys/values per (group, query): `K_g`, `V_g` each `[1, 8, 512, 2048, 128]`.
    **~2.1 GB apiece in bf16 — the dominant memory cost.**
12. Reshape the query tile to expose the group axis: `[1, 8, 4, 512, 128]`.
13. `scores = einsum(q, K_g) / √128` → `[1, 8, 4, 512, 2048]`; `-inf` on invalid slots.
14. `attn = softmax(scores, dim=-1)` in fp32. **Each of the 32 heads gets its own distribution** over its
    group's 2048 tokens, summing to 1. Nothing is averaged here.
15. `out = einsum(attn, V_g)` → `[1, 8, 4, 512, 128]` → reshape `[1, 32, 512, 128]`.
16. Concatenate tiles, apply `o_proj` → residual stream → … → logits → `L_LM`.

### E. The KL — same tile, no second attention

17. **Teacher:** `P = attn.detach().mean(over the 4-head axis)` → `[1, 8, 512, 2048]`. Reuses step 14's
    tensor. The mean collapses 4 per-head distributions into the one per-group target the index branch can
    be compared against, because there are 8 index queries, not 32.
18. **Student:** gather step 6's `S_idx` at the same 2048 positions → `[1, 8, 512, 2048]`, then
    `log_softmax` over the last axis. A gather, not a new matmul.
19. `KL = Σ over the 2048 tokens of P · (log P − log_student)` → `[1, 8, 512]`; invalid slots contribute
    exactly 0 (`torch.where`, not reliance on `P = 0`, since masked teacher entries are NaN).
20. Accumulate over tiles; divide by (valid query rows × 8) → one scalar for this layer.

Steps C–E for a tile live inside **one** `torch.utils.checkpoint` call, so only one tile's intermediates
are alive and both `out` and `KL` return in the live graph (§1.1).

### F. Loss

21. Reduce the 33 per-layer KLs (`mean` by default) → `L_KL`.
22. `L_LM` = cross-entropy of step 16's logits against the next tokens.
23. `L = L_LM + λ · L_KL`.

### G. Backward

| path | reaches | blocked by |
|---|---|---|
| `L_LM` → base (`q/k/v/o`, MLP, embeddings) | steps 11–16 and the rest of the network | `sel` detached at step 9; `S_idx` is not in the LM path at all |
| `L_KL` → index branch (`index_{q,k}_proj`, both norms) | step 18 → step 6 → steps 4–5 | `x = h.detach()` at step 3; teacher detached at step 17 |

In **2a** the base has `requires_grad=False`, so step 22 is a *measurement* (the sparsity cost) and trains
nothing. In **2b** the base trains through the `L_LM` path.

---

## 3. Why the teacher is free — precisely

### 3.1 What the claim is

"Free" means: **the teacher requires no additional attention computation, given the restricted support.**
It does **not** mean the KL is free.

| KL component | Phase 2 | Phase 1 |
|---|---|---|
| **teacher** | `.detach().mean(2)` over an existing `[1,32,T_q,2048]` → writes 32 MB. **No matmul, no softmax.** | full dense attention **recompute**, all 32 heads, `[1,8,T_q,32768]` → 512 MB + `O(N²)` FLOPs |
| **student** | gather `S_idx` at `I_tok` (already computed for selection) + `log_softmax` over 2048 | `log_softmax` over all `N` → another 512 MB |
| **KL terms** | elementwise over 32 MB | elementwise over 512 MB |

The mechanism is a support identity: MSA Eq. 9's teacher and Eq. 8's forward normalise over **the same
set** `I^(r)_{i,tok} = (∪_{b∈I} B_b) ∩ {1..i}`. So the teacher *is* the main branch's own distribution.
Phase 1 is the same equation over the full causal set — which the dense flash kernel computes internally
but never returns, so it must be rebuilt. The `1/G` mean is incidental to the claim; what is free is that
the 32 softmax rows already exist.

**The invariance that makes this work:** because each head is normalised over `I_tok` *before* the `1/G`
average, MSA's teacher is **invariant to the scores of unselected keys**. Nothing outside the selected set
can change it, so nothing outside the selected set needs computing.

### 3.2 What is NOT free — the index branch is dense in both phases

Selection requires scoring every key (step 6): `S_idx` is `[1, 8, T_q, 32768]`, the `H_kv·d_idx·N²` term
of Eq. 12 = **1.10 TFLOP/layer** at 32K. That is inherent to MSA and unaffected by any of this. What makes
the design work is the asymmetry:

```
index branch, dense over all keys:    H_kv · d_idx · N²        = 1.10 TFLOP
main branch,  dense over all keys:  2 · H_q  · d_h  · N²        = 8.80 TFLOP
main branch,  sparse (2048 keys):   4 · H_q  · d_h · N·k·B_k    = 1.10 TFLOP
```

The index branch is ~8× cheaper per unit of context, which is why it can afford to look everywhere. **A
dense teacher in Phase 2 would cost the 8.80 instead of the 1.10** — i.e. undo the sparsity, per layer,
per step. That also prices risk R2's escape hatch: `full_support_kl_prob` costs one dense attention pass
on the batches where it fires.

Nothing is "missing from memory" — `k` and `v` exist for all 32768 tokens. What's missing is the `q·k`
matrix for unselected pairs, and computing it *is* the dense attention sparsity exists to avoid.

### 3.3 Why DSA cannot do this — verified against V3.2

DeepSeek-V3.2 §2.1.1, verbatim:

> *"for the `t`-th query token, we first aggregate the main attention scores by **summing across all
> attention heads**. This sum is then **L1-normalized along the sequence dimension** to produce a target
> distribution `p_{t,:} ∈ R^t`."* (Eq. 3, dense warm-up)
>
> *"we also keep aligning the indexer outputs to the main attention distribution, but considering only the
> selected token set `S_t`:"* `L_I = Σ_t D_KL( p_{t,S_t} ‖ Softmax(I_{t,S_t}) )` (Eq. 4, sparse stage)

`p` is defined **once**, in the warm-up section — sum over heads, then L1-normalise along the *sequence*.
Eq. 4 takes a **slice** of that full-sequence distribution. So the normalisation precedes the restriction,
and the two orders are not interchangeable:

```
MSA  (Eq. 9):   P(j) = (1/G) Σ_h A^h_S(j)                        ← normalise per head over S, then average
DSA  (Eq. 3/4): P(j) = Σ_h A^h_F(j) / Σ_{u∈S} Σ_h A^h_F(u)       ← average over the FULL sequence, then crop
              = Σ_h m_h · A^h_S(j) / Σ_h m_h,   m_h = exp(LSE_S^h − LSE_F^h)
```

`m_h` is head `h`'s **coverage** of `S`. So DSA's target is a *coverage-weighted* head average and MSA's is
*uniform*; they coincide only when all heads in the group have equal coverage (including Phase 1, where
`S = F` and every `m_h = 1` — which is why our teacher-ordering test agrees at full support and diverges by
0.21 on a restricted one).

For one head the normaliser cancels and either order is free. For a **sum** over heads it does not cancel,
because each head has a different `Z^h`. Concretely — 4 keys, 2 heads, `S = {1,2}`, and changing only key
3's score for head 2 (a key that never enters the KL):

```
head 2 = [1, 1, 80, 18]  → DSA teacher over S = [0.880, 0.120]
head 2 = [1, 1,  8, 18]  → DSA teacher over S = [0.860, 0.140]      ← moved
                            MSA teacher over S = [0.694, 0.306]      ← identical in both
```

So DSA needs a full-context pass in Phase 2 **by definition**, not by implementation choice.

**Consequence for computation counts.** DSA Phase 2 runs three attention-like computations per layer per
tile; MSA runs two:

| | DSA | MSA |
|---|---|---|
| index/indexer scores, dense over all keys | ✅ → selection + KL student | ✅ → selection + KL student |
| main attention, sparse over selected | ✅ → **logits only** | ✅ → **logits AND KL teacher** |
| main attention, dense over all keys | ✅ → **KL teacher** | — |

**An available DSA optimisation, for the record.** Even keeping Eq. 4 exactly, the second pass need not be
a full attention: `A^h_S` and `LSE_S^h` are free from the sparse forward, so the only missing quantity is
`LSE_F^h` — **one scalar per (head, query)** (≈4 MB at our shapes vs gigabytes of weights). A flash-style
LSE-only pass gives it: the `q·k` FLOPs, but no `[T_q, N]` materialisation and no `A·V` matmul.
`_sparse_indexer_kl` currently materialises the full per-head distribution (lines 526–536), which is
heavier than the definition requires. Not our project, but worth knowing if the DSA path is revisited.

### 3.4 The claim is contingent on materialising weights

`attn` exists only because our forward computes `softmax(...)` into a variable (§1's gather). If we later
swap in FlexAttention or a fused Triton kernel returning only `(out, LSE)`, the teacher stops being free.
MiniMax hit exactly this wall — §4.3: *"we optimize this by emitting these LSE values directly to global
memory during the main pass, allowing us to skip the KL loss forward pass entirely… The backward kernel
then loads these scalars directly into the softmax."* Same principle as §3.3's optimisation: keep the
scalars, discard the matrix, rebuild the distribution when needed.

### 3.5 Two implementation details this dictates

1. **Detach before reducing**, so no graph node is built:
   ```python
   P = attn.detach().view(b, H_kv, G, T_q, M).mean(dim=2)   # not attn.view(...).mean(2).detach()
   ```
   In 2b `attn` is part of the live LM graph; Algorithm 1 line 7's `stopgrad(Q), stopgrad(K)` is precisely
   this detach.
2. **Compute the KL inside the same checkpointed region as the attention.** Under per-layer checkpointing
   `attn` is transient — live during the region, recomputed in backward. Computing the KL outside it would
   force retaining `attn` across the whole layer stack, reintroducing the memory the checkpointing removes.

---

## 4. `λ` — now live, and the conversion is a trap

Phase 1 didn't need `λ` (frozen base → no `L_LM` to trade against). Phase 2 does, and:

- **The MSA paper never gives a value.** Full-text search finds `λ` only symbolically (Algorithm 1,
  Eqs. 18–19).
- **Our default `kl_reduction='mean'` rescales it.** Algorithm 1 is a SUM over layers, so matching a paper
  `λ` under `mean` requires **`λ_ours = λ_paper × n_sparse_layers`** (33). A paper value used verbatim
  weights the KL 33× too weakly.
- What the paper *does* tell us (B.3): without the Eq. 11 stopgrad, larger `λ` caused gradient-norm spikes
  and LM divergence within a few hundred steps; **with** it, the same values were stable. So under our
  wiring the realistic failure of a bad `λ` is an under-trained indexer, not a blow-up.

**Approach:** pick `λ` so that `λ · Σ L_KL ≈ 0.05–0.2 × L_LM` on the first batch of 2b (logged as
`indexer/kl_share_of_loss`), then hold it fixed. Sweep one order of magnitude either way on a 200-step
1.7B run before the real launch.

---

## 5. Gradient wiring

| path | trains | mechanism |
|---|---|---|
| `L_LM` → base | base only (2b; nothing in 2a) | `sel` is `.detach()`-ed — top-k is non-differentiable, so no LM gradient reaches the index branch |
| `L_KL` → index branch | index params only | Eq. 11 `stopgrad(X)` into the index projections; teacher detached (Algorithm 1 line 7) |

Two Phase-2-specific consequences for existing infrastructure:

1. **FSDP2 Option-B2 must NOT apply.** In Phase 2 the loss flows through each decoder layer's output, so
   the standard layer gates fire and the indexer needs no special unit. `verl/utils/fsdp_utils.py` already
   gates the B2 wrapping on `mode == "dense_warmup"` — correct as written, do not widen it.
2. **Two optimizer param groups.** `optim.indexer_lr` already exists (`workers/config/optimizer.py:115`,
   `transformer_impl.py:456-462`, keyed on `".indexer." in name`) and works for MSA unchanged, since our
   parameters are named `…self_attn.indexer.*`.

---

## 6. Code to write

| # | item | where | notes |
|--:|---|---|---|
| 1 | `_sparse_attn` | `qwen3_msa.py` | steps 11–16; tiled over `kl_block_size`; returns `(out, attn_w, sel)` |
| 2 | `_sparse_indexer_kl` | `qwen3_msa.py` | steps 17–20; teacher from the forward's own `attn_w` |
| 3 | `_sparse_attn_and_kl` | `qwen3_msa.py` | the single checkpointed region returning both (§1.1, §3.5) |
| 4 | `mode == "sparse"` branch | `qwen3_msa_attn_forward` | currently `raise NotImplementedError`; **replaces** the dense attention call |
| 5 | `msa_sparse_loss` | `workers/utils/losses.py` | LM CE + `kl_lambda ·` KL; generalise `dsa_sparse_loss` or add a sibling reading `_msa_indexer_kl` |
| 6 | `loss_mode=msa_sparse` | `trainer/sft_trainer.py` | one branch beside `indexer_kl` / `dsa_sparse` |
| 7 | `full_support_kl_prob` | `MSAConfig` | risk R2; costs one dense attention pass when it fires (§3.2) |
| 8 | `run_qwen3_msa_phase2a.sh` / `_phase2b.sh` | `examples/msa/` | clone Phase 1's; 2a sets `msa_mode=sparse` + frozen base, 2b adds `optim.indexer_lr` and the data mixture |

Reused unchanged from Phase 1: `MSAIndexer` in full (`select_blocks` and `block_token_mask` were written
for this), `attach_indexers`, `install_kl_accumulation`, the diagnostics, the monkey-patch hook. Reused
from `minicpm_dsa.py`: the tiling loop, `gather(bias, idx)` for masked selected keys, fp32 softmax with
cast-back, the detach discipline, and the `(output, kl)`-from-one-checkpoint pattern.

---

## 7. Metrics and gates

Everything from Phase 1 keeps working (the coverage family is computed from the teacher, which still
exists), **plus** the two numbers only Phase 2 can produce:

1. **`lm_loss_sparse − lm_loss_dense`, same weights, same batch — the pure sparsity cost.** In 2a this is
   *the* result: base weights are stock, so any LM-loss gap is attributable to sparsity alone. Two forwards
   → val set only.
2. **`indexer/kl_share_of_loss`** = `λ Σ L_KL / L_LM`, so `λ` is visible rather than implicit.

**2a exit criteria (decides whether 2b is mandatory):**
- sparsity cost above ~0.05 nats on held-out long docs ⇒ 2b is mandatory (expected — see plan §12);
- `learned_coverage` and `coverage_vs_ceiling` under the *restricted* support hold at their Phase-1 values
  (a drop means the restriction destabilised selection);
- no NaN, no selection collapse onto the forced local block — which needs the position-bucketed metrics
  from Phase 1's remaining item 2. **Prerequisite, not optional**: the forced block alone covers a large
  share of attention mass, so a collapse reads as success without them.

**2b exit criteria:** the eval ladder in `../qwen3_4b_dsa/eval_plan.md` with plan §9's substitutions, and
the recalibrated gate — `Δ(N)` non-increasing in `N` is load-bearing; magnitude judged against MSA-CPT's
own residual (−2.6 RULER-8K, −3.1 HumanEval at 400B tokens).

---

## 8. Risks specific to Phase 2

**R1 — Selection churn.** As the index branch trains, the support moves under the base. The paper's answer
is the warm-up (which we have). Monitor: selection stability on a *fixed* val batch across checkpoints
(fraction of blocks retained). A churning support under a training base is the plausible cause if 2b
diverges.

**R2 — Starved blocks.** A block ranked 17th never enters the loss, so the index branch gets no gradient
about what it failed to select (plan §6.2) — and §3.1's invariance property is the precise reason: the
teacher literally cannot see outside `S`. This is why Phase 1 is load-bearing. Escape hatch, config-gated:
`full_support_kl_prob` — run Phase 1's full-support KL on a small fraction of batches (e.g. 5%) to keep the
full ranking calibrated. Costs one dense attention pass (8.80 TFLOP/layer) when it fires, so 5% ≈ +4% on
attention FLOPs. Cheap to add now, expensive to retrofit after a bad run.

**R3 — Self-referential teacher (2b only).** The teacher is the *sparse* model's own attention, so the KL
can be lowered by the backbone reshaping attention to be easy to index rather than by the indexer
improving. Not speculation: paper B.3 reports exactly this (*"the backbone can lower the KL loss by
simplifying the Main Branch attention distribution"*) with short-context regression (Fig. 9), which is why
Eq. 11's stopgrad exists. Our stopgrad blocks the *KL* route, but in 2b `L_LM` still trains the backbone
and nothing stops it drifting toward index-friendly attention. **Monitor `attn/entropy_norm`** — a sharp
fall in 2b that Phase 1 didn't show is the signature. Mitigation: lower `λ`, raise the BC share.

**R4 — Serving parity becomes load-bearing.** In Phase 1 selection only fed diagnostics; in Phase 2 it
determines the function. The vLLM kernel parity test (plan §4.2 #2) moves from "nice before training" to
**blocking**, and needs a vLLM `main` build (the container ships 0.20.2).

**R5 — Step time.** Plan §8 predicts ~9.1 TB/step of gather traffic, ~30% *less* than the MiniCPM3 Phase-2
runs, but that comparison rests on a "~38 s/step" anchor with no in-repo citation. Re-measure on the 4B
before scheduling 2b.

---

## 9. Order of work

1. **32K sparse-forward probe** on the 4B: memory and step time for the gather at `T_q ∈ {512, 256}`,
   before writing the loss around it. Decides whether §1's union optimisation is needed on day one.
2. `_sparse_attn` + a **dense-equivalence test**: with `k ≥ ceil(N/B_k)` the sparse path must reproduce the
   Phase-1 (dense) logits to bf16 noise. Same control as plan §4.2 #1, now on the path that changes the
   function.
3. `_sparse_indexer_kl` + `_sparse_attn_and_kl` + checkpointing, with tests that (a) the KL gradient
   survives checkpointing (§1.1's trap), (b) `dL/dS = P_idx − P` still holds on the restricted support, and
   (c) the teacher equals the group-mean of the forward's own weights (§3.1).
4. Loss + trainer branch + 2a launch script; run 2a; report the sparsity cost.
5. vLLM parity (parallel track, blocking for 2b).
6. 2b data mixture (`../qwen3_4b_dsa/data_plan.md` §5; build plan for the short + decode-long halves:
   [phase2_data_gen.md](phase2_data_gen.md)) and launch. Its §6 and §10 are prerequisites — the default
   SFT dataset path silently deletes Qwen3 reasoning traces, and generation is a ~day-long H100 job, so
   both land before step 4's 2a run finishes.

Phase 1's position-bucketed metrics are a **prerequisite** for step 4.
