# DSA × FSDP2 Sharding Notes

How FSDP2 (`fully_shard`) interacts with the DSA lightning-indexer training, why the Phase-1 indexer KL
hits `grad_norm = 0` under default resharding (§3) — and why, even after the `reshard_after_forward=False`
workaround, a *nonzero* `grad_norm` still doesn't train because the optimizer's params never get a grad
(§3b) — the exact hook mechanics behind it, the sharding options, and why Phase 1 and Phase 2 need
**opposite** FSDP settings.

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

## 3b. Why a *nonzero* `grad_norm` still doesn't train — the master vs compute-copy split

`reshard_after_forward=False` (§3) removes the `grad_norm = 0` symptom but **not** the underlying bug. The
loss now moves off zero (`grad_norm ≈ 330`), yet training is still dead: the indexer params show a `.grad`,
but the **optimizer's** params show `grad = None`, and `optimizer.step()` no-ops → flat loss. This is a
different, sharper face of the same missing reduce-scatter, and it is worth understanding precisely because
**a healthy-looking `grad_norm` reads as "it's training" when it isn't.**

### Two parameter objects per weight

Under `MixedPrecisionPolicy(param_dtype=bf16)`, FSDP2 keeps **two** tensors per weight:

- the **sharded master** — the persistent fp32 `DTensor` (`fsdp_param.sharded_param`). *This is what the
  optimizer captured at build time* — `build_optimizer(module.parameters(), …)`
  (`transformer_impl.py:454`), called right after the wrap while params are in the sharded state
  (`:564,569`).
- the **unsharded compute copy** — the bf16 full tensor (`fsdp_param.unsharded_param`), materialized by the
  pre-forward all-gather. With `reshard_after_forward=False`, `post_forward` does **not** reshard
  (`_fsdp_param_group.py:461-462`), so this copy stays live *and stays registered as the module's
  parameter*. So after forward, `module.named_parameters()` returns the **unsharded bf16 copy**, a different
  object from the optimizer's sharded master.

### Autograd fills the copy; only `post_backward` fills the master

Backward computes grad against the live (unsharded) param, so it writes **`unsharded_param.grad`**. The
sharded master's `.grad` is written by nothing in autograd — it is populated **only** inside
`post_backward`, which reads the copy's grad and reduce-scatters it onto the master:

```python
# _fsdp_param_group.py:525-528, 567
elif fsdp_param.unsharded_param.grad is not None:
    unsharded_grads.append(fsdp_param.unsharded_grad_data)
    fsdp_param.unsharded_param.grad = None          # copy's grad consumed here
...
foreach_reduce(fsdp_params_with_grad, unsharded_grads, …)   # → writes the sharded master's .grad
```

But per §3, in the frozen-base side-channel case **`post_backward` never runs** (neither backward gate is
armed). So `foreach_reduce` never executes, and the bridge from copy→master is never crossed:

```
KL.backward()  →  unsharded_param.grad  ✅ (autograd)  ──✗ post_backward never runs ✗──►  sharded_param.grad  ❌ (None)
                  = module.parameters()                                                    = optimizer.param_groups
```

### Why `grad_norm` lies here

The two engine code paths read the two different parameter sets:

| step | reads | has grad? |
|---|---|---|
| `optimizer_step` grad-norm — `fsdp2_clip_grad_norm_(self.module.parameters(), …)` (`transformer_impl.py:690`) | unsharded **copies** | ✅ → `grad_norm ≈ 330` |
| `self.optimizer.step()` (`transformer_impl.py:744`) | sharded **masters** | ❌ `None` → AdamW no-ops |

So a nonzero `grad_norm` measures the copy's grad, which the optimizer never sees. This is exactly what the
`DSA_DEBUG_MASTER` probe (`transformer_impl.py:699-766`) was added to expose: it reads the optimizer's
**own** params (the masters), and prints `grad_norm=None` / `NO STATE` / `max|Δ init|=0` for the indexer
masters even while the run reports `train/grad_norm ≈ 330`.

### Takeaway

Under an FSDP2 side-channel loss, **`grad_norm > 0` is not evidence of training.** The only reliable signal
is that the *optimizer's* params get a grad and move — i.e. `post_backward`/`foreach_reduce` actually ran.
`reshard_after_forward=False` alone does not guarantee that; you need the EXIT/ENTRY gate to fire
(Option B, §4) or to take the indexer out of FSDP's master/copy management entirely.

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

---

## 7. Appendix — FSDP2 backward walkthrough (torch 2.11 source)

A ground-up trace of how FSDP2 runs a step and why the loss has to reach a wrapped module's output,
straight from `torch/distributed/fsdp/_fully_shard` in torch 2.11.

### How FSDP2 runs — the four hooks

`fully_shard(module)` registers two module hooks — `_pre_forward` and `_post_forward`
(`_fsdp_state.py:114-118`). Everything else is bootstrapped from autograd hooks those two install. The
lifecycle for one wrapped unit (e.g. a `MiniCPMDecoderLayer`):

**1. pre_forward** (`_fsdp_state.py:243` → `_fsdp_param_group.py:442-452`)
- `unshard()` — all-gathers the sharded shards into the full parameter, casts to the bf16 compute copy.
- `_register_post_backward_hook(args)` — wraps the module's **input** tensors in
  `RegisterPostBackwardFunction`, but *only* the ones with `requires_grad=True`:
  ```python
  for i, obj in enumerate(args_kwargs_list):
      if torch.is_tensor(obj) and obj.requires_grad:   # _fsdp_param_group.py:723
          inp_tensors.append(obj)
  if len(inp_tensors) == 0:
      return args, kwargs                              # nothing registered
  inp_tensors = RegisterPostBackwardFunction.apply(self, *inp_tensors)  # :728
  ```

**2. forward** runs on the full (unsharded) params.

**3. post_forward** (`_fsdp_state.py:275` → `_fsdp_param_group.py:454-468`)
- `reshard()` — frees the full param again (gated on `reshard_after_forward`; this is the flag flipped in
  Issue #3).
- `_register_pre_backward_hook(output)` — registers `_pre_backward` on the module's **output** tensors,
  again *only* those with `requires_grad`:
  ```python
  for t in flat_outputs:
      if torch.is_tensor(t) and t.requires_grad:       # _fsdp_state.py:361
          t.register_hook(self._pre_backward)
  ```

**4. backward** — two things must happen: re-gather the params, then reduce-scatter the grads.
- When grad reaches an **output** tensor, `_pre_backward` fires (`_fsdp_state.py:302-311`): it
  `_register_root_post_backward_final_callback()` (queues the reduce-scatter finalizer, `:304`) and calls
  `pre_backward()` → `unshard()` again to re-gather params for the backward math
  (`_fsdp_param_group.py:491`).
- When grad reaches an **input** tensor, `RegisterPostBackwardFunction.backward` fires → `post_backward()`
  directly.
- `post_backward` (`_fsdp_param_group.py:496-595`) is the one that matters: it reads the autograd-computed
  grad off the **unsharded** param and reduce-scatters it into the **sharded** param's `.grad`:
  ```python
  elif fsdp_param.unsharded_param.grad is not None:    # :525
      unsharded_grads.append(fsdp_param.unsharded_grad_data)
      fsdp_param.unsharded_param.grad = None           # :528
  ...
  foreach_reduce(fsdp_params_with_grad, unsharded_grads, ...)  # :567 -> writes sharded .grad
  ```
- Finally the root callback `_root_post_backward_final_callback` (`_fsdp_state.py:313-338`) sweeps every
  param group and runs `post_backward()` for any that didn't already run.

### Why the loss must be on the output

The gradient the optimizer consumes lives on the **sharded** parameter (that's what the optimizer captured
at build time). Autograd never writes there — autograd only fills `unsharded_param.grad` (the compute
copy). The *only* thing that moves grad from the unsharded copy to the sharded param is `post_backward` /
`foreach_reduce` (`:525→567`).

And `post_backward` is reachable only through hooks armed on **boundary tensors that require grad**:

- the output-tensor hook `_pre_backward` (`_fsdp_state.py:361`) — which is also what queues the root
  finalizer (`:304, :369`), and
- the input-tensor `RegisterPostBackwardFunction` (`_fsdp_param_group.py:723`).

Both are gated on `requires_grad` (`:361`, `:723`). So if backward never crosses a wrapped module's
input/output tensor that requires grad:
- `_pre_backward` never fires → the root finalizer is never queued → the whole reduce-scatter sweep never
  runs;
- `RegisterPostBackwardFunction.backward` never fires either.

Result: `post_backward` never executes, `unsharded_param.grad` is never reduced into `sharded_param.grad`,
and the sharded param (the optimizer param) keeps `grad=None`.

There's even an explicit fallback for the "frozen-input" case in the root finalizer:
```python
# Run post-backward in case forward inputs did not require
# gradient so the autograd backward did not run
fsdp_param_group.post_backward()     # _fsdp_state.py:323-325
```
but note it only runs *if the finalizer was queued*, and the finalizer is queued only from `_pre_backward`
(`:304`) — i.e. only if some **output** tensor required grad. So even the safety net needs at least one
requires-grad output to bootstrap it.

### Mapping to the DSA run

In Phase-1 the KL graph is `KL → indexer.scores → indexer weights`, with `hidden_states`/`qr` entering as
**constant** inputs (base is frozen, no `enable_input_require_grads`), and the loss is stashed on
`model._dsa_indexer_kl` — it never flows through any wrapped layer's returned output. So:
- No wrapped module has a requires-grad input **or** output on the KL path → neither hook is armed →
  `post_backward` never runs.
- Backward still computes `unsharded_param.grad` on the indexer (that's what `module.parameters()` shows,
  and what `clip_grad_norm_` sums to ~330).
- `foreach_reduce` never runs → the sharded master's `.grad` stays `None` → optimizer no-ops → flat loss.

That's why "put the loss on the output" is the *intended* FSDP2 contract — it's literally the only trigger
torch arms. And it's also why it can't work for a frozen base without forcing the base activations to
require grad (the expensive/ad-hoc thing): there's simply no requires-grad boundary tensor for FSDP to hang
the hook on.
