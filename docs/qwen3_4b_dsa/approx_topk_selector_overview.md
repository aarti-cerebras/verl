# Qwen3 DSA approximate top-k selector overview

**Status:** reference implementation and CUDA-graph correctness validation, 2026-08-28.

This note defines the exact and copied-top-k controls, explains how the current partial-radix
selector changes the attention set, and distinguishes semantic approximation from selector-speed
optimization. The full validation and rollout plan remains in
[`approx_topk_serving_plan.md`](approx_topk_serving_plan.md).

## Exact and copied-top-k controls

### Exact

The `exact` arm loads `Qwen3DSAForCausalLM` from `scripts/dsa/vllm_qwen3_dsa`. For each query, the
lightning indexer scores every causal key and vLLM's stock selector emits the `k` highest-scoring
request-local indices. The current serving configuration uses `k = 2048`.

Conceptually:

```python
selected = scores.topk(k).indices
```

### Copied top-k

The `copied-topk` arm loads the isolated `Qwen3DSAApproxForCausalLM` implementation from
`scripts/dsa/vllm_qwen3_dsa_approx`, but configures it with:

```text
dsa_selector = topk
dsa_selector_backend = vllm_stock
```

This arm is **not approximate**. The selector runtime is inactive and the hook delegates to vLLM's
stock top-k kernel. It is a control that asks whether copying the model into an approximate-capable
plugin changed behavior before any radix rule was enabled:

```text
original exact implementation
              versus
isolated implementation with approximation disabled
```

Any mismatch between these two arms must be separated from process-level numerical variation before
it can be blamed on `radix_floor` or `radix_ceil`. Fresh-process replication showed that both the
exact and copied plugins can choose either side of the same borderline decision: exact-plugin runs
produced both token 382 with a 0.125 logprob margin and token 271 with a zero margin. Copied-plugin
runs produced margins of 0.0, 0.125, and 0.25. Contract-check and custom-op-namespace controls did
not determine the outcome.

Consequently, cross-process greedy token equality is diagnostic rather than a hard gate. The hard
end-to-end numerical gate compares centered top-token logprobs only while generated prefixes remain
identical, using a tolerance calibrated above the exact-vs-exact variability envelope. Fixed-input
selector and CUDA-replay fixtures retain deterministic equality requirements.

## Score and index flow

For each sparse layer:

```text
hidden state
  -> lightning-indexer query/key projections
  -> one score per causal key
  -> exact or partial-radix selector
  -> request-local indices with a -1 suffix
  -> global KV-cache slots
  -> FA3 sparse attention
```

The selector sees a score matrix shaped `[query rows, keys]`. A row may only emit indices at or
before its request-local query position. FA3 receives the number of valid selected indices and
ignores the remaining `-1` suffix.

## Partial-radix threshold rule

The exact top-k boundary is the k-th-largest causal score, called `Tq`. The reference selector:

1. Masks noncausal scores.
2. Uses exact `torch.topk` to obtain `Tq` and the exact comparison set.
3. Converts scores and `Tq` to FP16.
4. Maps each FP16 bit pattern to a monotonic unsigned 16-bit radix key.
5. Reconstructs a threshold after omitting the key's four low bits.
6. Retains every causal score whose radix key is at least the reconstructed threshold.

The monotonic transform preserves numeric order:

```text
larger finite FP16 score -> larger radix key
```

Omitting four bits groups 16 adjacent radix encodings into a bucket. Given monotonic threshold key
`m`, the implemented reconstruction rules are:

```python
floor    = m & ~0xF
midpoint = (m & ~0xF) | 8
ceil     = min((m & ~0xF) + 16, 0xFFFF)
```

Membership is then:

```python
keep = causal & (mono_score >= reconstructed_threshold)
```

The authoritative implementation is in
[`radix_rules.py`](../../scripts/dsa/vllm_qwen3_dsa_approx/radix_rules.py) and
[`radix_selector_reference.py`](../../scripts/dsa/vllm_qwen3_dsa_approx/radix_selector_reference.py).

## Selector modes

| Mode | Threshold behavior | Expected relationship to exact top-k | Capacity |
|---|---|---|---:|
| `topk` | Stock vLLM exact selection | Exactly `k` | `k` |
| `exact_ge` | Unrounded FP16 `Tq`, including ties | Exact set plus possible threshold ties | `k + margin` |
| `radix_floor` | Lower edge of the radix bucket | Contains exact top-k; may add keys | `k + margin` |
| `radix_midpoint` | Middle of the radix bucket | May add or drop keys | `k + margin` |
| `radix_ceil` | Start of the next radix bucket | Subset of exact top-k; may drop keys | `k` |

For example, with `k = 2048`, the validated floor configuration uses `index_topk = 4096`. The
logical target remains 2048, while the larger buffer holds keys admitted by the lower threshold.
If the true selection exceeds capacity, the implementation fails closed instead of truncating it.

`radix_ceil` raises the boundary, so it cannot over-capture relative to exact top-k and needs no
capacity margin.

## Special cases and safety contract

- A causal prefix with at most `k` keys retains every causal key; no threshold approximation is
  needed.
- If an active row unexpectedly selects nothing, the highest candidate is rescued and telemetry
  records the event.
- CUDA-graph padding rows have query position `-1`. They are inactive, stay empty, and are excluded
  from validation and telemetry.
- Active output rows must contain unique, causal, request-local indices as a valid prefix followed
  only by `-1`.
- Capacity overflow, an invalid index, a noncausal index, a duplicate, or malformed padding fails
  closed.

## What is and is not optimized today

The current backend approximates the **set reaching attention**, but it does not yet accelerate the
selector. It deliberately calls exact `torch.topk` to obtain `Tq`, then applies the partial-radix
emission rule. This supports:

- quality and recall measurement against an exact set;
- capacity sizing for over-capturing rules;
- CUDA-graph and padded-batch validation;
- integration with request-local indexing and FA3;
- verification of the intended floor, midpoint, and ceil semantics.

It cannot demonstrate an approximate-selector speedup. A production implementation must replace
exact threshold discovery with radix passes or bucket counting while preserving the same membership,
capacity, safety, and telemetry contracts.
