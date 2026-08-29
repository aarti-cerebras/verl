# Modulo-Bucket Local Top-k for Qwen3 DSA Serving

**Status (2026-08-29):** eager implementation validated, including an attention budget independent
of the checkpoint's `dsa_top_k`. The isolated reference and
`vllm_stock_per_bucket` paths, builder, boot path, launcher, and CPU/non-interference tests pass.
On an H100, real vLLM CUDA prefill/decode selected-set parity passed, an eager sparse engine loaded
all 36 bucketed layers, and a 4.5k-token equal-budget needle fixture passed with token-identical
exact/bucket outputs. `FULL_DECODE_ONLY` capture and replay also pass on H100, including seven active
decode rows padded to an eight-row captured graph and additional 3-to-4, 9-to-16, and 33-to-40
replays. Exploratory graph/eager decode measurements are retained, but full/prefill graphs,
and durable performance claims remain unvalidated. Bucket telemetry is implemented and validated
on H100 in eager `verify_exact` and decode-graph `graph_safety` modes. A separate
`graph_verify_exact` mode now follows the approximate selector's split: eager prefill remains
host-folded while captured decode updates fixed-address exact-quality moments on device. The
`vllm_stock_batched_buckets` backend validates the requested 500-by-12 geometry: 6,000 logical
positions are carried in a minimally 128-aligned 6,016-slot buffer whose 16-slot tail is always
invalid. End-to-end eager generation passes at this geometry; full-engine decode graph capture has
not yet been validated for it.

**Scope:** Qwen3 DSA vLLM serving and evaluation only. This plan does not change training,
checkpoint weights, sampling top-k, or the existing exact and radix-selector servers.

## 1. Objective

Add an isolated Qwen3 DSA selector that groups causal key tokens by request-local position:

```text
bucket(position) = position % bucket_count
```

and selects a fixed exact local top-k from the indexer scores in every bucket. The union of those
positions is the key set consumed by sparse GQA attention.

The selector/GQA capacity is:

```text
bucket_total_k = bucket_count * bucket_top_k
index_topk = round_up(bucket_total_k, 128)
```

`bucket_total_k` is the logical maximum selected count. `index_topk` is the physical buffer width
required by vLLM's request-index conversion; any alignment tail is permanently `-1` padded and is
excluded by FA3's `valid_counts`, so it does not increase the keys consumed by GQA.

`dsa_top_k` remains the source checkpoint's logical/training value. Omitting `--bucket-top-k`
preserves the old equal-budget behavior by deriving it from `dsa_top_k`; passing it explicitly may
choose an independent maximum GQA attention budget. Telemetry compares against global exact
top-`bucket_total_k`, so cardinality remains comparable even when the source value differs.

Example:

```text
dsa_top_k      = 2048
bucket_count   = 8
bucket_top_k   = 256
index_topk     = 2048
```

An explicitly larger geometry is also valid:

```text
dsa_top_k      = 2048  # retained source metadata
bucket_count   = 500
bucket_top_k   = 12
bucket_total_k = 6000  # maximum valid positions consumed by GQA
index_topk     = 6016  # physical width; final 16 slots are -1
```

## 2. Non-interference requirement

Bucketed local top-k is an alternative selector, not a preprocessing or postprocessing step for
the existing radix selectors. The modes are mutually exclusive:

```text
topk                 -> stock global exact top-k
radix_floor          -> existing global radix-floor selector
radix_midpoint       -> existing global radix-midpoint selector
radix_ceil           -> existing global radix-ceil selector
modulo_bucket_topk   -> exact local top-k in each position bucket
```

Two stock-reuse layouts implement the same selector: `vllm_stock_per_bucket` issues one strided
top-k call per bucket and remains the simple rollback path; `vllm_stock_batched_buckets`
materializes all bucket rows and invokes stock top-k once. The batched layout is intended for large
bucket counts such as 500, where the per-bucket layout measured 18,000 launches per 36-layer model
forward and did not finish engine warmup within 45 minutes.

Do not implement either of these compositions:

```text
ceil/floor -> modulo bucketing -> local top-k
modulo bucketing -> local top-k -> ceil/floor
```

The operations do not commute. Combining them changes the selected count, threshold semantics,
and containment relationship with global exact top-k.

For maximum experimental isolation, the bucket implementation must not import or modify:

- `scripts/dsa/vllm_qwen3_dsa_approx/radix_rules.py`;
- `scripts/dsa/vllm_qwen3_dsa_approx/radix_selector_reference.py`;
- the approximate selector's threshold, rescue, or over-capture logic;
- the exact Qwen3 DSA plugin.

## 3. Isolated plugin layout

Create a sibling plugin rather than adding another branch to the approximate-selector runtime:

```text
scripts/dsa/vllm_qwen3_dsa_bucketed/
    __init__.py
    model.py
    indexer.py
    sparse_attention.py
    bucket_selector_hooks.py
    bucket_selector_runtime.py
    bucket_topk_reference.py
    bucket_topk_stock.py
    bucket_topk_triton.py       # optional, only after profiling justifies it
```

Register a distinct architecture:

```text
Qwen3DSABucketedForCausalLM
```

Add separate construction and serving entry points:

```text
scripts/dsa/build_qwen3_dsa_bucketed_serving_dir.py
scripts/dsa/serving/serve_qwen3_dsa_bucketed_entry.py
scripts/dsa/serving/serve_qwen3_dsa_bucketed.sh
scripts/dsa/serving/_pluginboot_bucketed/sitecustomize.py
```

Snapshot the model, indexer, and sparse-attention plumbing from the exact Qwen3 DSA server at a
recorded repository revision. Add only the bucket selector and its configuration. This incurs some
copy maintenance but prevents bucket experiments from changing the ceil/floor implementation or
its evidence.

The build manifest must record:

- source serving directory and source-config digest;
- source and derived architectures;
- repository revision;
- `bucket_count`, `bucket_top_k`, `dsa_top_k`, and `index_topk`;
- selector backend and telemetry mode;
- whether a selector speed claim is valid;
- confirmation that the source directory was not modified.

## 4. Selector semantics

For score row `r` with inclusive request-local query position `q[r]`, the valid keys are:

```text
V[r] = {p | 0 <= p <= q[r]}
```

Bucket `b` contains:

```text
V[r, b] = {p in V[r] | p % bucket_count == b}
```

The selector emits:

```text
S[r, b] = exact_topk(indexer_score[r, p], p in V[r, b], bucket_top_k)
S[r]    = union over b of S[r, b]
```

If a bucket contains fewer than `bucket_top_k` causal keys, select the entire bucket. Therefore:

```text
selected_count[r] = min(q[r] + 1, bucket_count * bucket_top_k)
```

for a contiguous causal prefix. Prefixes no longer than `total_k` are attended densely.

Positions are request-local. Bucketing restarts at position zero for every request. Never bucket
packed-batch columns, block-table indices, or physical KV-cache slots.

CUDA-graph padding rows have `q[r] == -1` and must emit an all-`-1` row.

### 4.1 Output layout

The existing GQA handoff requires a valid prefix followed by `-1`. Use rank-major bucket layout:

```text
rank 0: bucket 0, bucket 1, ..., bucket B-1
rank 1: bucket 0, bucket 1, ..., bucket B-1
...
```

The output slot for bucket `b`, local rank `j` is:

```text
output_slot = j * bucket_count + b
```

Modulo buckets of a contiguous prefix differ in population by at most one, with lower-numbered
buckets receiving the extra token. Rank-major layout consequently preserves the required valid
prefix without a separate data-dependent compaction kernel.

The selected set is the semantic contract. The reference implementation should use a stable output
order and may use smallest request-local position first at a tied local top-k boundary. Stock vLLM
top-k is not assumed to make the same boundary choice. Parity gates must require identical sets when
the boundary is unique; for a tied boundary, they must require all strictly greater scores and the
correct number of threshold-equal scores rather than a particular tied position.

## 5. Reference implementation

Implement a graph-safe PyTorch reference first in `bucket_topk_reference.py`.

Given scores `[rows, key_count]`:

1. mask noncausal positions to negative infinity;
2. pad the key dimension to a multiple of `bucket_count`;
3. reshape `[rows, groups, buckets]` and transpose to `[rows, buckets, groups]`;
4. run exact `torch.topk(bucket_top_k)` along `groups`;
5. convert group indices back to request-local positions with
   `position = group * bucket_count + bucket`;
6. transpose local rank before bucket and flatten to rank-major order;
7. write a valid prefix followed by `-1` into the fixed output buffer.

This backend validates selector semantics and model quality. It does not support a selector-speed
claim because it materializes the bucketed view and invokes generic top-k.

The bucket selector should have its own result type rather than reuse radix-specific threshold and
rescue fields:

```python
@dataclass(frozen=True)
class BucketSelectionResult:
    selected_count: torch.Tensor
    effective_k: torch.Tensor
```

## 6. Reuse the stock DSA top-k

The primary production backend should reuse vLLM's saved original DSA top-k implementation rather
than introduce a new selection kernel. The existing operation is global, so it cannot enforce all
bucket quotas in one unchanged invocation. Invoke it independently for each bucket.

For bucket `b` and sequence length `L`:

```text
bucket_scores  = scores[:, b::bucket_count]
bucket_length  = max(0, (L + bucket_count - 1 - b) // bucket_count)
local_indices  = stock_topk(bucket_scores, bucket_length, bucket_top_k)
local_position = local_indices * bucket_count + b
```

Write local rank `j` directly or through a fixed scratch buffer to:

```text
output[row, j * bucket_count + b]
```

This preserves the existing exact score precision and top-k implementation while keeping the
bucket selector independent of radix ceil/floor.

### 6.1 Preferred stock backends

Evaluate the following approaches in order:

1. **Strided per-bucket calls.** Pass `scores[:, b::bucket_count]` and its strides to the saved
   original top-k operation. Use a fixed Python loop over the compile-time bucket count. This uses
   `bucket_count` top-k launches but avoids score materialization.
2. **One materialized batched call.** Gather scores into contiguous
   `[rows, bucket_count, groups]`, flatten to `[rows * bucket_count, groups]`, and invoke stock
   top-k once. This needs a preallocated score-sized scratch buffer but provides a contiguous
   layout accepted by stock kernels.
3. **Hybrid bucket groups.** Materialize or process a small fixed number of buckets per call if
   neither launch count nor full materialization is acceptable.

Do not assume that every vLLM dispatch supports arbitrary strides. Validate generic prefill,
generic decode, cooperative decode, and persistent decode separately. In particular, native decode
entry points expose fewer explicit stride controls than the generic wrappers.

The pinned vLLM 0.26 cooperative and persistent entry points accept only
`k in {512, 1024, 2048}`. When the global `dsa_top_k` dispatch enters either specialized path but
`bucket_top_k` is another value (for example 256), the isolated hook redirects each bucket to the
saved generic `top_k_per_row_decode` operation. That operation accepts explicit score strides, so
the initial stock backend uses `scores[:, b::bucket_count]` without materializing bucket scores.

If the stock kernel cannot write rank-major strided output directly, write to a preallocated
`[bucket_count, rows, bucket_top_k]` scratch tensor and use a graph-safe remap operation. Allocation
must occur before warmup and graph capture.

### 6.2 Prefill addressing

For prefill row `r`:

```text
key_start      = cu_seqlen_ks[r]
query_length   = cu_seqlen_ke[r] - cu_seqlen_ks[r]
local_position = bucket + group_index * bucket_count
score_column   = key_start + local_position
```

The selector must load `score_column` but emit `local_position`. A correctness-first implementation
may retain eager per-request handling. A graph-capturable implementation must instead derive these
offsets on device for all rows and avoid the approximate hook's CPU `.tolist()` request discovery.

Decode uses `seq_lens[r]` as `query_length` and has no per-request score-column offset.

### 6.3 Backend identity and speed claims

Name the first production backend explicitly, for example:

```text
vllm_stock_per_bucket
```

Its manifest must report the number of top-k launches and whether bucket scores were materialized.
It may support an end-to-end speed claim only after profiling; reuse alone does not establish that
`bucket_count` launches are cheaper than global stock top-k.

## 7. Optional fused Triton selector

A custom Triton selector is deferred until profiling shows that the stock reuse path is inadequate.
Reasons to consider it include excessive per-bucket launch overhead, unsupported strided layouts,
costly score materialization, or a requirement for one all-request prefill selector launch.

The candidate fused kernel would consume indexer logits and write request-local positions directly
into the shared buffer. It would fuse causal masking, modulo grouping, exact local selection,
boundary-tie handling, compaction, and rank-major emission. It would not initially fuse indexer
score production.

The candidate design is a persistent program per `(query_row, bucket)` using an exact FP32 radix
threshold: scan fixed score tiles, find the local k-th threshold four bits at a time, then emit all
scores above it plus deterministic position-ordered boundary ties. Prefill may instead use one
program per row or a hybrid bucket grouping to improve memory coalescing.

Before implementation, compare this design with the measured stock path. Do not add or maintain a
custom kernel solely because it might be faster.

## 8. CUDA graph capture

Stock top-k launches and a fixed bucket loop can be graph-captured. A custom Triton kernel is not a
prerequisite. Prefill capture nevertheless requires replacing the host-side request-splitting path.

The capturable selector path must satisfy all of the following:

- compile and warm every selected stock or custom kernel before graph capture;
- treat bucket geometry, launch count, and any tile geometry as static;
- use stable device addresses for scores, metadata, output, and any scratch storage;
- derive request offsets and causal lengths on device;
- represent shorter requests and inactive graph-padding rows with device-side masks;
- perform no `.cpu()`, `.tolist()`, `.item()`, tensor-dependent Python control flow, or
  per-request Python launch loop;
- perform no dynamic allocation during capture or replay;
- keep telemetry on device in persistent fixed-address buffers, or disable it.

This makes the selector compatible with prefill capture. Complete prefill graph capture still
depends on vLLM capturing the surrounding indexer score computation, KV-cache writes, sparse FA3
call, and model path. Selector capture compatibility must not be reported as proof that the whole
prefill is captured.

## 9. Configuration and validation

Add the following derived-config fields:

```json
{
  "architectures": ["Qwen3DSABucketedForCausalLM"],
  "dsa_selector": "modulo_bucket_topk",
  "dsa_selector_backend": "vllm_stock_per_bucket",
  "dsa_bucket_count": 8,
  "dsa_bucket_top_k": 256,
  "dsa_top_k": 2048,
  "index_topk": 2048
}
```

Fail closed unless:

- `dsa_bucket_count > 0`;
- `dsa_bucket_top_k > 0`;
- `index_topk` is the smallest multiple of 128 greater than or equal to
  `dsa_bucket_count * dsa_bucket_top_k`;
- every slot from the logical bucket total through `index_topk` is `-1` padded;
- the local `dsa_bucket_top_k` does not exceed the validated stock top-k kernel limit;
- the backend is one of the bucket plugin's explicitly supported reference, stock-reuse, or
  optional Triton backends;
- the configured graph and telemetry modes are compatible.

The builder must derive a new serving directory and refuse to overwrite a nonempty output or
modify the exact source directory.

## 10. Telemetry

Bucket telemetry must remain independent of radix threshold telemetry. Do not emit radix
`threshold`, `rescued`, over-capture, or under-capture claims.

Record bounded summaries for:

- selected count and effective k;
- selected count per bucket;
- empty or underfilled buckets;
- recall, precision, and Jaccard against global exact top-`bucket_total_k`;
- global exact top-k keys dropped by bucketing;
- bucket-selected keys added relative to global exact top-k;
- indexer-score mass or score-rank loss relative to global exact top-k;
- selection distribution by query-position and distance bands;
- invalid, noncausal, duplicate, and non-prefix output violations.

Telemetry used during graph replay must update preallocated device counters. Host-folded detailed
telemetry is limited to eager runs.

Every run artifact must identify the selector as bucketed exact local top-k so it cannot be
mistaken for radix approximation or global exact top-k.

## 11. Tests

### 11.1 CPU/reference semantics

- Hand-constructed scores produce the expected positions for every modulo bucket.
- Non-divisible sequence lengths assign the extra tokens to the correct lower buckets.
- Buckets shorter than `bucket_top_k` select all their causal keys.
- Prefixes of length at most `total_k` select the full causal prefix.
- Long prefixes select exactly `total_k` unique causal positions.
- Inactive graph-padding rows remain all `-1`.
- Exact boundary ties satisfy the threshold membership and cardinality contract.
- Output always has a valid prefix followed by `-1`.
- Bucketing restarts at zero for each request in a multi-request prefill batch.

### 11.2 Stock top-k reuse parity

- Compare stock-per-bucket and reference selected sets on randomized score tensors.
- Cover positive, negative, repeated, and extreme finite scores.
- Sweep bucket counts, local k values, row counts, sequence lengths, and non-power-of-two widths.
- Cover strided and materialized stock backends where supported.
- Validate every vLLM prefill and decode top-k dispatch path independently.
- Compare eager and CUDA-graph replay outputs.
- Treat tied-boundary position differences separately from true set errors.

If an optional Triton backend is later implemented, add the same parity matrix against both the
reference and stock-reuse backends.

### 11.3 Hook and GQA integration

- Cover generic prefill, generic decode, cooperative decode, and persistent decode dispatches.
- Verify emitted positions convert to unique valid KV-cache slots.
- Verify GQA consumes exactly the selected set.
- Cover heterogeneous multi-request batches and inactive capture-bucket rows.
- Confirm exact total budget equality with the global top-k control.

### 11.4 Non-interference

- Assert the bucket plugin does not import approximate radix modules.
- Assert the approximate plugin does not import bucket modules.
- Run existing radix floor, midpoint, and ceil tests unchanged.
- Compare their selected outputs before and after the bucket plugin lands.
- Verify importing or serving one plugin does not install the other plugin's hooks.
- Verify the exact server remains bit-identical to its protected snapshot.

### 11.5 Performance and graph evidence

- Measure selector latency separately from indexer score computation and FA3 attention.
- Compare PyTorch reference, stock global top-k, strided stock-per-bucket, and materialized
  stock-per-bucket.
- Add Triton to the comparison only if that backend is implemented.
- Report prefill and decode independently across context and batch-size buckets.
- Confirm profiler evidence for the number of selector launches and absence of bucket
  materialization.
- Confirm whether the selector alone or the complete prefill path is captured; do not conflate the
  two claims.

## 12. Implementation stages

### Current implementation checkpoint (2026-08-28)

- Stage A eager exit criteria pass. CPU/reference, builder, source-isolation, worker boot, model
  load, and two-request generation all have retained evidence.
- Stage B eager decode selected-set parity passes against the reference on real H100 CUDA for both
  cooperative and persistent global dispatch interception at `total_k=2048`, `local_k=256`.
- Stage C eager prefill selected-set parity passes against the reference on real H100 CUDA across
  packed requests. It deliberately retains host request discovery, so CUDA graphs fail closed and
  prefill graph capture is not claimed.
- The graph follow-on permits only `FULL_DECODE_ONLY` for the stock backend. Full and piecewise
  prefill capture still fail closed. A fixed-shape decode replay test with seven active rows plus
  one inactive padding row passes on real CUDA. End-to-end 3-to-4, 7-to-8, 9-to-16, and 33-to-40
  padded replays also pass.
- An exploratory five-repeat H100 comparison records median graph/eager decode-window ratios of
  7.355x at batch 3, 5.729x at batch 9, and 1.906x at batch 33. The controlled batch-33 point uses
  2,839-to-3,542-token prompts, all above the 2,048 selection budget. These are engine-level graph
  versus eager results, not isolated selector-kernel timings or a durable speed claim.
- No Stage D Triton kernel was added. There is no evidence yet that one is needed.
- Telemetry defaults to `off`; manifests keep `selector_speed_claim_valid` false. Eager `summary`
  and `verify_exact` use bounded host-folded summaries and fail closed with CUDA graphs.
  `graph_safety` uses persistent fixed-address per-layer/per-bucket device counters.
  `graph_verify_exact` additionally captures the global stock top-k reference and accumulates
  overlap, score-mass, position-band, distance-band, and per-bucket moments on device for every
  decode replay; eager prefill continues to use the bounded host-folded summaries. Both graph modes
  use `FULL_DECODE_ONLY`. Manifests report decode and prefill graph validation separately and only
  mark the tested 8-by-256 stock geometry as decode-graph validated.
- Explicit `bucket_top_k` now decouples the selected attention budget from checkpoint
  `dsa_top_k`. The logical budget is `bucket_count * bucket_top_k`; the physical `index_topk` is
  its minimal 128-aligned capacity. Both values and the padding width are recorded in the build
  manifest. The materialized `vllm_stock_batched_buckets` backend performs one existing stock
  top-k call per selection and is the practical large-bucket-count path; no custom Triton selector
  was added.
- At 500-by-12, CUDA reference parity, both stock backend parity checks, direct selector graph
  replay, and the padded vLLM request-index conversion pass. The eager engine loads all 36 sparse
  layers and consumes 6,000 valid positions from a 6,016-slot buffer. The final smoke served
  6,518- and 6,514-token prompts, generated 16 tokens for each, and retrieved the expected needle.

Retained GPU evidence:

- `.agents/gpu_jobs/20260828T201624Z-qwen3-dsa-bucketed-eager/result.md` — CUDA top-k parity passed;
  the overall job is `ERROR` only because its first smoke harness used an overlong Unix socket path.
- `.agents/gpu_jobs/20260828T202407Z-qwen3-dsa-bucketed-e2e/result.md` — corrected eager end-to-end
  job `PASS`, including `36/36 layers sparse`, two completed requests, and unchanged source digest.
- `.agents/gpu_jobs/20260828T205314Z-qwen3-dsa-bucketed-longctx-8192/result.md` — eager long-context
  job `PASS`: exact and bucket arms both retrieved item 71 at 4,484/4,479 prompt tokens and emitted
  token-identical 48-token completions.
- `.agents/gpu_jobs/20260828T210745Z-qwen3-dsa-bucketed-decode-graph/result.md` — decode-only graph
  job `PASS`: four CUDA selector tests, 51 captured graphs, seven 4.35k-to-4.48k-token requests with
  32 decode tokens each, safe 7-to-8 padding, and successful needle retrieval.
- `.agents/gpu_jobs/20260828T214342Z-qwen3-dsa-bucketed-decode-bench-v2/result.md` — five-repeat
  graph/eager job with valid batch-3 and batch-9 measurements and safe 3-to-4, 9-to-16, and 33-to-40
  replay. Its terminal status is `FAIL` because the batch-33 prompt fixture crossed below 2,048;
  do not use that job's batch-33 ratio or its superseded direction-of-bias inference.
- `.agents/gpu_jobs/20260828T220547Z-qwen3-dsa-bucketed-b33-long-bench/result.md` — corrected
  batch-33 job `PASS`: all prompts 2,839-to-3,542 tokens, five safe 33-to-40 replays, median decode
  throughput 105.290 tok/s eager versus 200.664 tok/s graph (1.906x), and all needles retrieved.
- `.agents/gpu_jobs/20260828T225011Z-qwen3-dsa-bucket-telemetry/result.md` — eager
  `verify_exact` telemetry passed end to end (36 layers, eight buckets/layer, zero violations); the
  overall job is `FAIL` because its first graph-safety attempt exposed missing prefill attribution.
- `.agents/gpu_jobs/20260828T232757Z-qwen3-dsa-bucket-graph-telemetry-fix/result.md` — corrected
  graph telemetry job `PASS`: 51 decode graphs captured, post-capture reset succeeded, all 36
  prefill and decode layers exported persistent counters, and six safety-violation classes were
  zero over 1,440 observed rows.
- `.agents/gpu_jobs/20260829T000314Z-qwen3-dsa-bucket-graph-verify-exact/result.md` — first
  graph-exact job `FAIL`: direct replay and all 51 engine graph captures succeeded, exposing only a
  missing exported `total` field and a four-token needle fixture too short to contain its answer.
- `.agents/gpu_jobs/20260829T002139Z-qwen3-dsa-bucket-graph-verify-exact-fix/result.md` — corrected
  field/fixture job `FAIL`: long-context generation and decode-quality export passed (1,080 rows,
  mean recall 0.9913, balanced added/dropped totals, zero safety violations), but prefill collapsed
  under `unattributed` and a float32 expected mean used an impossible absolute tolerance.
- `.agents/gpu_jobs/20260829T003723Z-qwen3-dsa-bucket-graph-verify-exact-attribution/result.md` —
  follow-up contract routes graph-exact prefill through pinned layer recovery and computes the
  independent expected intersection mean in float64.
- `.agents/gpu_jobs/20260829T023927Z-qwen3-dsa-bucket-500x12-aligned-eager/result.md` — aligned
  6,016-slot capacity removes vLLM's multiple-of-128 rejection and passes eight CUDA tests, but the
  500-launch-per-layer rollback backend times out during warmup after 45 minutes.
- `.agents/gpu_jobs/20260829T033425Z-qwen3-dsa-bucket-500x12-batched-eager/result.md` — one-call
  materialized stock backend passes 11 CUDA tests, loads all 36 layers, and reduces warmup to
  125.37 seconds. Its terminal `FAIL` is limited to a two-token needle fixture that stopped after
  emitting `" The code"`.
- `.agents/gpu_jobs/20260829T034254Z-qwen3-dsa-bucket-500x12-needle/result.md` — corrected eager
  end-to-end job `PASS`: 126.16-second warmup, 6,518/6,514-token prompts, 16 decode tokens each,
  successful needle retrieval, unchanged implementation/artifact digests, and 74.07 GiB peak GPU
  memory. This job does not claim full-engine decode graph validation.

### Stage A: isolated reference

1. Snapshot the exact Qwen3 DSA serving plugin into the bucketed architecture.
2. Add separate builder, boot path, launcher, configuration, and manifest.
3. Implement the PyTorch bucket selector and all CPU/reference tests.
4. Validate request-local indexing through the existing sparse GQA handoff.
5. Run existing exact and radix suites as non-interference gates.

Exit criterion: reference semantics, isolation checks, and end-to-end eager serving pass. No speed
claim.

### Stage B: stock top-k reuse for decode

1. Implement `vllm_stock_per_bucket` using the saved original top-k operation.
2. Validate strided score input and output behavior for generic, cooperative, and persistent
   decode dispatches.
3. Add preallocated scratch and graph-safe remapping only where required.
4. Establish reference parity away from ties, valid boundary-tie behavior, and graph replay parity.
5. Benchmark launch overhead and memory traffic against global stock top-k and the reference.

Exit criterion: selected-set parity with the bucket reference away from tied boundaries,
threshold-valid tied selections, and no selector safety violations under heterogeneous decode
concurrency.

Current status: correctness and `FULL_DECODE_ONLY` graph-safety exit criteria pass for the tested
8-by-256 geometry. Exploratory engine-level graph/eager measurements pass; selector-only profiling,
comparison against global stock top-k/reference, and realistic continuous-batching evaluation remain
open, so `selector_speed_claim_valid` stays false.

### Stage C: stock top-k reuse for prefill

1. Establish eager prefill correctness using stock top-k per bucket.
2. Compare strided per-bucket calls with one contiguous materialized batched call.
3. Consume `cu_seqlen_ks` and `cu_seqlen_ke` directly on device for the capture path.
4. Remove CPU metadata copies and data-dependent per-request Python launches.
5. Validate selector graph capture with fixed capture buckets and padding rows.

Exit criterion: reference parity and captured replay parity where the surrounding vLLM path permits
capture. One selector invocation is desirable but not required; a static captured bucket loop is
acceptable if its measured performance is sufficient.

Current status: eager reference parity passes. Prefill graph work remains deliberately isolated and
unimplemented; it is not implied by the passing decode-only graph result.

### Stage D: optional Triton optimization

1. Use Stage B/C profiles to identify a concrete stock-path bottleneck.
2. Implement a fused selector only if it addresses that bottleneck.
3. Establish selected-set parity away from ties, valid boundary-tie behavior, and graph replay
   parity.
4. Retain stock-per-bucket as the simpler rollback backend.

Exit criterion: the Triton backend is kept only if it provides a measured benefit without weakening
correctness, graph safety, or isolation.

### Stage E: end-to-end evaluation

1. Compare global exact top-k, bucket reference, and stock-per-bucket at an equal total attention
   budget; include Triton only if implemented.
2. Evaluate model quality, selected-set recall, long-context behavior, and throughput.
3. Attribute time separately to score computation, selection, index conversion, and GQA.
4. Publish only claims supported by retained manifests, telemetry, profiler traces, and test logs.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `bucket_count` stock launches add overhead, especially at geometries such as 500×12 | compare strided, materialized single-call, and hybrid/fused approaches before adding a kernel |
| Materializing batched bucket rows duplicates the score tensor temporarily | retain the strided rollback backend and measure peak memory at target prompt/batch shapes |
| Larger `index_topk` increases shared-buffer, index-conversion, and GQA work | record the derived capacity in the manifest and measure memory/latency for each explicit geometry |
| Stock kernels reject strided input or output | use fixed preallocated contiguous scratch and a graph-safe remap |
| Bucket materialization adds score-sized traffic | prefer strided calls where supported and decide from phase-specific profiles |
| Optional FP32 radix requires repeated score scans | implement only after a measured stock-path bottleneck justifies it |
| Boundary ties differ from stock top-k | validate threshold membership and cardinality instead of specific tied positions |
| Selected-key order changes FA3 rounding | document order and validate output tolerances end to end |
| Graph padding emits bogus keys | explicit `query_length == 0` mask and mandatory inactive-row assertions |
| Shared global hooks affect another selector | separate architecture, entry point, boot path, runtime, and hook installer |
| Plugin snapshots drift | record source revision and retain exact-copy parity tests |

## 14. Non-goals

- Combining modulo bucketing with radix ceil, midpoint, or floor.
- Changing indexer training or checkpoint parameters.
- Changing sampling behavior.
- Claiming reduced indexer score-computation cost.
- Claiming complete prefill graph capture from selector capture alone.
- Supporting a variable per-bucket budget in the first implementation.
- Requiring a custom Triton kernel before the stock top-k reuse path has been measured.

## 15. Acceptance criteria

The feature is complete only when:

1. the bucketed architecture and serving artifacts are isolated from exact and radix plugins;
2. configuration records the logical `bucket_count * bucket_top_k`, enforces the minimally
   128-aligned physical `index_topk`, and records when the logical budget differs from source
   `dsa_top_k`;
3. reference and stock-per-bucket implementations select identical sets at unique boundaries and
   satisfy the same threshold-membership contract at tied boundaries;
4. all outputs are causal, unique, request-local, and valid-prefix/`-1`-suffix;
5. heterogeneous prefill and decode batches, including CUDA-graph padding rows, pass;
6. existing exact and radix selector tests remain unchanged and pass;
7. telemetry clearly distinguishes bucket selection from radix selection;
8. performance and graph-capture claims are limited to directly measured scopes.
