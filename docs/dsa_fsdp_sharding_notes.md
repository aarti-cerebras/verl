# DSA × FSDP2 Sharding Notes

How FSDP2 (`fully_shard`) interacts with the DSA lightning-indexer training, why the Phase-1 indexer KL
hits `grad_norm = 0` under default resharding, the exact hook mechanics behind it, the sharding options,
and why Phase 1 and Phase 2 need **opposite** FSDP settings.

Related: [`dsa_grad_norm_debugging.md`](dsa_grad_norm_debugging.md) (Issue #3),
[`dsa_indexer_init_proposal.md`](dsa_indexer_init_proposal.md).

---

## 1. Where the indexer lands in the FSDP wrap

`apply_fsdp2` (`verl/utils/fsdp_utils.py:559-586`) applies `fully_shard` to three kinds of module:

1. each transformer decoder layer (matched by `_no_split_modules` → `MiniCPMDecoderLayer`),
2. `embed_tokens` / `lm_head`,
3. the root model.

The indexer lives at `layer.self_attn.indexer` (`minicpm_dsa.py:91`). It is **not** wrapped separately, so
it is absorbed into its enclosing **decoder-layer FSDP unit** — sharded and resharded exactly like the rest
of that layer. So the indexer params *are* FSDP-managed; the Phase-1 bug is about *when they get
re-gathered*, not membership.

Under the load path (`fsdp2_load_full_state_dict`, `fsdp_utils.py:492-499`), rank-0 keeps its CPU init
(`model.to(device)`) and broadcasts it; non-rank-0 does `to_empty()` then receives the broadcast. So only
rank-0's init survives (see the init proposal for the caveat about freshly-created indexer keys not being
in `full_state` at `NPROC>1`).

---

## 2. FSDP2's parameter lifecycle and the four hooks

A sharded parameter has two physical states:

- **sharded** — the persistent `1/N` slice (small, always resident).
- **unsharded** — the full param, reconstructed by an **all-gather**, allocated into a temporary buffer for
  compute only.

`reshard_after_forward=True` frees the unsharded buffer after a module's forward; it must be re-gathered
before backward.

> **At world_size=1 this still runs.** The shard *is* the full data, so the all-gather is a no-op copy —
> but FSDP2 still allocates/frees the separate unsharded buffer and runs all the same hook machinery. That
> is why the Phase-1 bug bites even on the single-GPU smoke run: no memory is saved, but the free/re-gather
> bookkeeping still happens, and that bookkeeping is what breaks.

### 2a. The normal lifecycle from the ground up

For one wrapped module (say a decoder layer) with `reshard_after_forward=True`, a full training step walks
through six states:

```
1. all-gather      pre-forward hook reconstructs the full param into the unsharded buffer
2. FORWARD         module computes with the full param
3. free            post-forward hook drops the unsharded buffer → only the 1/N shard remains
   ── (time passes; other layers run their forward, then loss, then backward begins) ──
4. re-gather       pre-backward gate all-gathers the full param again, BEFORE this layer's backward
5. BACKWARD        module computes param grads against the now-live full param
6. reduce-scatter  post-backward gate sums grads across ranks, keeps each rank its 1/N grad shard, frees
```

Steps 1–3 and 6 keep peak memory at ~`1/N` of the params most of the time; the full param is only resident
during the module's own forward (2) and backward (5). The load-bearing question is **step 4**: FSDP has to
know *when* to re-gather. It can't re-gather "just before backward" by wall-clock — it has to be triggered
by the autograd engine reaching this module during the backward pass.

### 2b. The gate intuition

Think of each wrapped module as having two gates spliced into the autograd graph:

- **EXIT gate** — sits on the module's forward **outputs**. When backward flows *into* those outputs, it
  fires **step 4 (re-gather)** *before* the module's internal backward runs.
- **ENTRY gate** — sits on the module's forward **inputs**. When backward flows *out* toward the inputs
  (i.e. the module's internal backward is done), it fires **step 6 (reduce-scatter)**.

A gate only exists if backprop actually traverses the tensor it's attached to. That is the entire crux: a
gate keyed on the output only fires if the loss's gradient reaches that output.

A minimal sketch of the same idea (single-rank, no real collectives):

```python
class ToyFSDP(nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped
        wrapped.register_forward_pre_hook(self._pre_forward)   # step 1: unshard
        wrapped.register_forward_hook(self._post_forward)      # step 3: reshard + arm EXIT gate

    def _pre_forward(self, mod, args):
        self.all_gather_params()                               # materialize full params
        return args

    def _post_forward(self, mod, args, out):
        self.free_unsharded_params()                           # reshard
        if out.requires_grad:                                  # ← the crucial guard
            out.register_hook(self._pre_backward)              # EXIT gate on the OUTPUT
        return out

    def _pre_backward(self, grad):                             # fires ONLY if backward reaches `out`
        self.all_gather_params()                               # step 4: RE-GATHER before backward
        return grad
```

### 2c. Normal loss (CE) — the gates fire

```
loss(CE) ── logits ── block_N.out ── block_{N-1}.out ── … ── block_0.out
                          │              │
                      EXIT gate      EXIT gate     ← backward passes THROUGH each block output
```

The CE loss depends on the logits, which depend on the last block's output, which depends on the previous
block's output, and so on. So backward enters each block **through its output tensor** → the EXIT gate
fires → params re-gathered (step 4) → the block's backward runs (5) → the ENTRY gate reduce-scatters (6).
Every gate is on the gradient's path, so everything works. This is the case FSDP was designed around.

### 2d. reshard=True vs =False, as a timeline

```
reshard=True : [all-gather] → FORWARD → [FREE] → … → (backward needs step-4 re-gather) → EXIT gate → BACKWARD
reshard=False: [all-gather] → FORWARD → (stays resident) …………………………………………………………………………→ BACKWARD
```

With `reshard=False`, steps 3 and 4 are removed: the full param is never freed, so backward never needs the
EXIT gate to re-materialize it. That is exactly why it sidesteps the Phase-1 failure below.

### 2e. The four hooks (the precise wiring)

FSDP2 wires the lifecycle above with four hooks (torch 2.11 `_fully_shard`):

| gate | mechanism | torch source | job |
|---|---|---|---|
| pre-forward | `module.register_forward_pre_hook` | `_fsdp_state.py:114` → `_pre_forward:243` | **unshard** (all-gather) |
| post-forward | `module.register_forward_hook` | `_fsdp_state.py:117` → `_post_forward:275` | **reshard** + arm the EXIT gate |
| **EXIT gate** (pre-backward) | `Tensor.register_hook` on each **output** | `_register_pre_backward_hook:356-362` → `_pre_backward:302` | **re-gather** (unshard for backward) |
| **ENTRY gate** (post-backward) | `autograd.Function` on each **input** | `_register_post_backward_hook:708-728`, `RegisterPostBackwardFunction:876` | **reduce-scatter** grads + free |

Crucially, both backward gates are only installed **if the tensor they attach to requires grad**:

```python
# EXIT gate — _fsdp_state.py:356
def _register_pre_backward_hook(self, output):
    for t in tree_flatten(output)[0]:
        if torch.is_tensor(t) and t.requires_grad:   # ← guard
            t.register_hook(self._pre_backward)       # fires when grad reaches this output
    return output
```

```python
# ENTRY gate — _fsdp_param_group.py:876
class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param_group, *inputs):
        ctx.param_group = param_group
        return inputs                    # identity
    @staticmethod
    def backward(ctx, *grads):
        ctx.param_group.post_backward()  # reduce-scatter runs here
        return (None, *grads)
```

There is also a backstop: `_root_post_backward_final_callback` (`_fsdp_state.py:313`), queued via
`Variable._execution_engine.queue_callback` **from inside `_pre_backward`** (`:304`). It runs
`post_backward` for groups whose backward didn't run — but it is only *queued if some `_pre_backward`
fires*, and it never unshards. So it cannot rescue a unit whose EXIT gate never fired.

---

## 3. Why Phase 1 hits `grad_norm = 0`

Phase-1 dense warm-up freezes the whole base and trains **only** the indexer with a **KL-only** loss (no
LM cross-entropy). The KL is a side channel: computed inside the attention forward, stashed on
`self._dsa_kl`, summed into `model._dsa_indexer_kl`, and returned as the loss. Its autograd graph runs
`KL → indexer.scores → indexer.project → indexer params`, branching off the block's **input** `x` — it
never touches the block's **output** `hidden_states`.

Because the base is frozen, the block output is computed from frozen params and non-grad inputs, so
`hidden_states.requires_grad == False`. Therefore:

1. **The EXIT gate is never even installed** (the `requires_grad` guard fails) → no re-gather → params
   stay sharded → the indexer-weight `.grad` is never populated.
2. **The ENTRY gate is never installed** either (frozen inputs don't require grad).
3. **The backstop never queues** (no `_pre_backward` ever fires anywhere, including the root, since the KL
   doesn't flow through any output).

FSDP1 kept a view onto the freed unsharded storage → hard crash `setStorage: … storage of size 0`. FSDP2
silently leaves `.grad` empty → `grad_norm = 0`. (`_finalize_backward:340` may also `warning_once` that
"N modules … did not run forward before backward" — the same skipped-boundary-logic symptom.)

**Current fix:** `engine.reshard_after_forward=False` — the unsharded buffer is never freed, so no
re-gather is needed and plain autograd finds live params. Free at world_size=1; costs memory at
world_size>1 (the whole frozen base stays resident on every rank).

---

## 4. Sharding options for the trainable indexer

### Option A — global `reshard_after_forward=False` (current)

Simplest, correct. Keeps every unit (including the ~8 GB frozen base) unsharded per rank. Right for
single-GPU; wasteful of memory at multi-rank.

### Option B — indexer as its own `fully_shard` unit

The indexer's output *does* require grad (its params are trainable), so a separately-wrapped indexer
**would** have its EXIT gate armed and fired by the KL backward — re-gather works. Two sub-variants:

- **B1**: indexer unit `reshard=True` — works via the fired gate.
- **B2 (recommended for multi-rank)**: indexer unit `reshard=False`, base layers `reshard=True`. The huge
  frozen base still reshards (where the memory is); the tiny indexer (~1 M params/layer, ~62 M total) stays
  resident, so you don't even rely on gate-firing. `fully_shard` takes `reshard_after_forward` per call.

**Hard prerequisite for Option B:** FSDP2 forward hooks fire only on `nn.Module.__call__`. The integration
calls the indexer via **plain methods** — `attn.indexer.project(...)` (`minicpm_dsa.py:209`) and
`attn.indexer.scores(...)` (`:243`) — which bypass `__call__`, so the pre-forward all-gather never fires
and (at `NPROC>1`) forward would run on sharded params. The param-bearing compute must be routed through
`indexer.__call__`. Convenient split: `scores()` is **param-free** (only `softmax_scale`/cfg); **all params
are in `project()`**, which is called **once per layer forward** — so wrapping/routing just the projection
gives one all-gather/reshard cycle per layer, not per query block.

**Cost of Option B:** 62 tiny FSDP units → 62 extra small-tensor collectives per step (poor bandwidth
utilization). Coarse units are generally preferred; only worth it when memory-bound at multi-rank. At
world_size=1 it buys nothing over Option A.

---

## 5. Phase 1 vs Phase 2 need OPPOSITE settings

Phase 2 (sparse) is stubbed today (`minicpm_dsa.py:352-353`), but its loss structure inverts the FSDP
situation:

| | Phase 1 (dense warm-up) | Phase 2 (sparse) |
|---|---|---|
| base params | **frozen** | **train** (LM/CE loss) |
| loss | KL only | CE + λ·KL |
| loss flows through block outputs? | **no** (side-channel KL) | **yes** (CE → logits → every block out) |
| FSDP gates arm/fire? | no (frozen output) | **yes** (trainable output) |
| `reshard_after_forward` | **False** (workaround) | **True** (standard; just works) |
| indexer re-gather | needs Option A or B | free — CE backward re-gathers each layer |

So `reshard_after_forward=False` is a **Phase-1-only** setting and should be **removed in Phase 2**, not
carried forward.

**Phase-2 gotcha (independent of FSDP):** `select_topk` uses `.topk().indices` — **non-differentiable**. So
gradient does *not* reach the indexer through the sparse-attention selection; the indexer is still trained
by the differentiable KL, while CE trains the base through the selected keys. Design the Phase-2 loss as
`CE + λ·KL`; if you want gradient through the selection itself, that needs an STE on the top-k
(cf. the FP8 STE in [`dsa_grad_norm_debugging.md`](dsa_grad_norm_debugging.md) #2).

---

## 6. 32K / 4-GPU memory notes (FSDP-relevant)

- At 32K, **activations dominate, not params.** Resharding the 8 GB base saves ~6 GB/GPU — secondary on
  H100-80GB. So Option A (base resident) likely fits; Option B is a memory optimization, not a correctness
  need.
- FSDP shards **params, not activations** — DP=4 alone does not shrink the 32K activation footprint. If it
  doesn't fit even with gradient checkpointing, you need **sequence/context parallelism** (Ulysses), which
  shards the sequence dim.
- **SP risk:** the KL target recompute needs the **full key sequence** per layer
  (`s = q_blk @ k_all.T`, `minicpm_dsa.py:233`). Under Ulysses the keys are sharded → the custom KL/indexer
  path needs SP-aware collectives (all-gather K, or ring-style KL). It will **not** work out of the box.
  Prefer to size Phase 1 so pure DP + checkpointing fits 32K and SP is avoided.
- Memory knobs, in priority order: (1) gradient checkpointing on, (2) lower `kl_block_size`, (3) reshard
  the base (Option B), (4) sequence parallelism (last resort — integration work on the KL path).
