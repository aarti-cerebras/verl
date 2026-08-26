# Approximate Top-k for Qwen3 DSA Serving

**Status (2026-08-26):** the isolated `dsa-csx` PyTorch reference selector, hooks, variable-capacity
FA3 handoff, telemetry, and eager GPU validation are implemented locally. CUDA-graph support is in
Stage A: active selectors are capture-safe only with `dsa_telemetry=off`; persistent device-side
telemetry for graph replay is still pending. A native partial-radix Triton/CUDA kernel remains
explicitly deferred.

**Stage-A GPU gate (2026-08-26): PASS.** At 9,168/9,164 prompt tokens with `dsa_top_k=2048`, the
`radix_ceil` reference selector produced token-identical eager and `FULL_DECODE_ONLY` CUDA-graph
outputs. The complete seven-arm campaign also passed exact-vs-copy top-k parity, top-k eager-vs-graph
parity, and ceil eager-vs-graph parity. Ceil, midpoint, and floor each recorded 662,184 eager
telemetry rows with zero hard safety violations, capacity saturation, or rescue. The graph arm's
zero telemetry rows are intentional under the Stage-A `dsa_telemetry=off` contract. Artifacts:
`.agents/dsa_approx_gpu_validation/cg_stage_a_20260826_222937/`.

**Stage B implementation in progress:** `dsa_telemetry=graph_safety` allocates persistent
fixed-address counters before CUDA capture and records rows, calls, core selector safety failures,
capacity saturation, and rescue per phase and sparse layer on every replay. The existing post-warmup
RPC reset zeros those counters in place, preserving captured addresses. Full selected-count,
overlap, and position histograms remain host-folded and are not yet graph-supported.

**Stage-B graph-safety gate (2026-08-26): PASS.** Eager and CUDA-graph ceil both recorded exactly
662,184 post-warmup rows with zero safety violations, saturation, or rescue. Persistent replay
counters attributed all 36 sparse layers in both phases: prefill recorded 659,952 rows / 72 calls and
decode recorded 2,232 rows / 1,152 calls. All Stage-A token-parity gates remained exact. Artifacts:
`.agents/dsa_approx_gpu_validation/cg_stage_b_20260826_224854/`.

**Scope:** Qwen3 DSA evaluation and vLLM serving only. This plan does not change
training, checkpoint weights, sampling `top_k`, or distillation top-k.

## 1. Objective

Add an optional `dsa-csx`-equivalent approximate selector in an isolated copy
of Qwen3 DSA serving so its evaluation quality and selection behavior can be
measured without modifying the working exact server, while preserving:

- the existing Qwen3 indexer projections and FP8 score path;
- request-local, causal token selection;
- the existing page-size-one FA3 attention handoff;
- exact top-k as a baseline and rollback mode;
- sufficient telemetry to validate selection quality, safety, and model quality.

The near-term reference backend still executes exact top-k internally to obtain
the k-th threshold. It produces a genuinely approximate attention set, so model
quality and selection telemetry are valid, but it cannot support an
approximate-selector speedup claim. Validation runs may retain the exact set
from the same score tensor for comparison, while only the approximate selection
is sent to attention.

## 2. Source design in `dsa-csx`

The source rule is the GLM 5.2 partial-radix study in `dsa-csx`:

1. Cast scores onto the FP16 grid.
2. Convert FP16 representations to order-preserving `uint16` keys.
3. Resolve the bucket containing the k-th score using 4-bit radix digits from
   the most significant side.
4. Stop before resolving the final low bits.
5. Reconstruct a per-query threshold from the resolved prefix.

The selector modes are:

| Selector mode | Threshold rule | Relationship to exact top-k |
| --- | --- | --- |
| `topk` | exact fixed-size selection | baseline, exactly effective k |
| `exact_ge` | all scores greater than or equal to exact `Tq` | contains exact top-k, including all threshold ties |
| `radix_floor` | unresolved low bits set to zero | contains exact top-k; variable over-capture |
| `radix_midpoint` | unresolved region represented by its midpoint | may over- or under-capture |
| `radix_ceil` | threshold advanced to the next bucket | subset of exact top-k; no over-capture |

Here `Tq` is the exact k-th-largest valid score for one query row. The expected
containment chain is:

```text
radix_ceil ⊆ topk ⊆ exact_ge ⊆ radix_floor
```

The `dsa-csx` Python emitters are correctness and measurement vehicles. They
still use exact `topk` to obtain `Tq` and may sort an output-width window. The
near-term plan deliberately reuses this behavior to evaluate approximate
attention semantics. A later performance phase may replace only the selector
backend with a native partial-radix kernel while preserving the same rules,
telemetry, configuration, and FA3 handoff.

GLM results do not determine which selector is best for Qwen3. Qwen3 uses a
different model, score distribution, FP8 path, attention implementation, and
serving workload. Selection mode and capacity will be chosen from Qwen-specific
measurements.

## 3. Current Qwen3 serving path

The current dataflow is:

```text
hidden states
  -> Qwen3DSAServingIndexer projections and FP8 quantization
  -> vLLM SparseAttnIndexer score kernel
  -> vLLM exact top-k selector
  -> topk_indices_buffer [query tokens, dsa_top_k]
  -> request-local indices converted to global KV-cache slots
  -> valid_counts
  -> FA3 with block_table=selected slots and seqused_k=valid_counts
```

vLLM 0.26 can reach four exact selection paths:

- prefill `top_k_per_row_prefill`;
- decode `cooperative_topk`;
- decode `persistent_topk`;
- decode `top_k_per_row_decode` fallback.

All four call sites must be covered. Replacing only one would make selector
behavior depend silently on request shape or decode dispatch.

The FA3 handoff already supports the required variable-count representation:
valid indices in a fixed-width buffer, invalid entries represented by `-1`, and
per-row `valid_counts`. The approximate emitter must make every valid selection
a contiguous prefix followed only by `-1`.

### 3.1 Isolation from the working exact server

The existing exact-serving implementation is frozen for this experiment:

```text
scripts/dsa/vllm_qwen3_dsa/
scripts/dsa/serving/_pluginboot/
scripts/dsa/serving/serve_qwen3_dsa.sh
scripts/dsa/serving/serve_qwen3_dsa_entry.py
scripts/dsa/build_qwen3_dsa_serving_dir.py
```

Approximate work is developed side by side in a copied package and separate
entry points:

```text
scripts/dsa/vllm_qwen3_dsa_approx/
scripts/dsa/serving/_pluginboot_approx/
scripts/dsa/serving/serve_qwen3_dsa_approx.sh
scripts/dsa/serving/serve_qwen3_dsa_approx_entry.py
scripts/dsa/build_qwen3_dsa_approx_serving_dir.py
```

The architectures are distinct:

```text
Qwen3DSAForCausalLM        # existing exact server
Qwen3DSAApproxForCausalLM  # isolated approximate server
```

The approximate package starts as a snapshot of the working exact package. The
snapshot revision is recorded in its source comments and every run manifest.
Changes to selection, buffer capacity, hooks, and telemetry occur only in the
approximate copy.

The approximate bootstrap is the only bootstrap that installs selector hooks.
Because those hooks replace process-global vLLM operations, they must never be
imported by an exact-server process.

The approximate serving-directory builder derives a new directory from an
existing exact serving directory. It writes a copied and modified `config.json`
while symlinking unchanged weights and tokenizer assets. The source exact
serving directory is never modified.

## 4. Serving configuration contract

Add the following fields only to the derived approximate-serving
`config.json`. The original exact-serving configuration remains byte-for-byte
unchanged, and the derived configuration sets:

```json
{
  "architectures": ["Qwen3DSAApproxForCausalLM"]
}
```

The approximate selector fields are:

```json
{
  "dsa_top_k": 2048,
  "dsa_selector": "radix_ceil",
  "dsa_selector_backend": "dsa_csx_reference",
  "dsa_radix_omit_bits": 4,
  "dsa_selector_margin": 0,
  "dsa_telemetry": "summary",
  "index_topk": 2048
}
```

Definitions:

- `dsa_top_k`: logical target k used by the selection rule.
- `dsa_selector`: `topk`, `exact_ge`, `radix_floor`, `radix_midpoint`, or
  `radix_ceil`. `exact_ge` is primarily a tie-aware validation control.
- `dsa_selector_backend`: near-term approximate mode is
  `dsa_csx_reference`; exact control mode is `vllm_stock`.
- `dsa_radix_omit_bits`: number of unresolved low radix bits; initially 4.
- `dsa_selector_margin`: additional output capacity for an over-capturing
  selector.
- `dsa_telemetry`: `off`, `summary`, or `verify_exact`.
- `index_topk`: derived fixed buffer capacity used by vLLM and FA3.

Capacity is derived rather than independently configured:

```text
topk or radix_ceil:
    index_topk = dsa_top_k

exact_ge, radix_floor, or radix_midpoint:
    index_topk = round_up(dsa_top_k + dsa_selector_margin,
                          required_alignment)
```

The approximate serving-directory builder writes the resolved values and
refuses ambiguous or inconsistent combinations. It never writes into its source
exact-serving directory. Approximate-server construction asserts the same
contract; the exact server neither parses nor recognizes these experimental
fields.
There is no silent fallback from an approximate selector to exact top-k.
The server log and run artifact prominently identify the reference backend and
state that its timings do not represent a future native partial-radix kernel.
The artifact derives the read-only field `selector_speed_claim_valid=false`;
this is not a user-configurable claim.

## 5. Selector implementation

The near-term implementation ports the smallest backend-independent pieces of
`dsa-csx` into this repository with source attribution and matching tests. It
does not add an absolute runtime dependency on
`/net/.../dsa-csx`; serving artifacts and PRs must remain reproducible without
that checkout. The reused sources are:

- `glm_52/study_core/names.py` for selector names and schemes;
- `glm_52/study_core/rules.py` for monotonic keys, `Tq`, thresholds, and counts;
- `glm_52/study_core/summaries.py` for distribution summaries;
- `glm_52/vllm_study/selector/selection.py` for emission and verification;
- `glm_52/vllm_study/selector/telemetry.py` and
  `observers/passive.py` for telemetry structure;
- the request grouping, runtime context, and hook-lifetime patterns in
  `vllm_study/selector/{hooks,runtime,engine}.py`.

GLM model/scorer code, FlashMLA-specific attention bounding, and vLLM 0.23
call-site assumptions are not copied.

### 5.1 Shared rule module

Add `scripts/dsa/vllm_qwen3_dsa_approx/radix_rules.py` containing:

- FP16-to-monotonic-`uint16` conversion;
- threshold reconstruction for floor, midpoint, and ceil;
- membership and selected-count functions;
- the degenerate rule: select the whole valid prefix when valid count is no
  greater than k;
- FP16 finiteness and overflow checks;
- containment assertions.

This is the only definition of selector semantics used by the reference path,
backend tests, telemetry, and serving code.

### 5.2 Reference selector

Add `scripts/dsa/vllm_qwen3_dsa_approx/radix_selector_reference.py`.

The reference selector may use exact `torch.topk`. It will:

1. apply the request-valid range;
2. obtain exact `Tq`;
3. apply the selected threshold rule;
4. compact selected request-local indices;
5. emit a valid prefix followed by `-1`;
6. return selected counts and safety information.

This path is the near-term active serving backend as well as the oracle for
tests and validation telemetry. It produces the requested approximate attention
set, but its latency is reference-backend overhead and not an approximate-top-k
performance result.

### 5.3 Deferred native partial-radix kernel

Future work may add
`scripts/dsa/vllm_qwen3_dsa_approx/radix_selector_triton.py` without changing the
serving or telemetry contracts.

For each score row, the kernel will:

1. exclude columns outside the request-valid range;
2. cast valid scores to FP16 monotonic keys;
3. run 4-bit histograms from the most significant digit;
4. narrow the candidate bucket until the configured early-stop point;
5. construct the floor, midpoint, or ceil threshold;
6. compact qualifying request-local indices into the fixed-capacity buffer;
7. write `-1` to the complete unused suffix;
8. record selected count, overflow, and safety counters.

That future production kernel must not invoke `torch.topk`, vLLM exact top-k,
or another full selection internally. A profiler check will enforce this before
any selector speedup claim.

Kernel implementation, tuning, selector-only performance acceptance, and any
CUDA extension are outside the near-term plan. If pursued, Triton must first
outperform the existing Hopper exact selector before considering a CUDA
extension.

## 6. vLLM interception and layer attribution

Add:

```text
scripts/dsa/vllm_qwen3_dsa_approx/selector_hooks.py
scripts/dsa/vllm_qwen3_dsa_approx/selector_runtime.py
```

The hooks replace selection, not scoring. They preserve existing FP8 score
generation and side-cache insertion and cover all four vLLM selection paths.

Prefill hooks use `cu_seqlen_ks` and `cu_seqlen_ke` to derive each request's
valid key slice and request-local output indices. Decode hooks use `seq_lens`.

Each copied `Qwen3DSAApproxServingIndexer` supplies a context-scoped layer
identifier around its runtime op so telemetry can be attributed to:

```text
request/sample x layer x phase x absolute query position
```

The integration is pinned to the tested vLLM version and expected function
signatures. A mismatch is a startup error. Speculative decode, prefill/decode
context parallelism, and other unvalidated geometries initially fail explicitly.

Installation occurs in the EngineCore subprocess through
`scripts/dsa/serving/_pluginboot_approx`. The existing `_pluginboot` is not
changed and cannot install approximate-selector hooks.

## 7. Buffer and FA3 changes

Only in the copied approximate indexer, allocate the shared selection buffer
as:

```text
[max_num_batched_tokens, index_topk]
```

rather than `[max_num_batched_tokens, dsa_top_k]`.

The indexer retains both:

```text
rule_k = dsa_top_k
capacity = index_topk
```

Only the copied approximate sparse-attention adapter is changed to consume the
wider/variable-count buffer. Before FA3, it validates:

- all selected indices are request-local and causal;
- selected indices are unique;
- valid entries form a contiguous prefix;
- every remaining entry is `-1`;
- selector count equals converted `valid_counts`;
- count does not exceed capacity;
- converted global cache slots are valid and unique.

## 8. Telemetry modes

### 8.1 `off`

Run the reference approximate selector and approximate attention while
retaining mandatory fatal safety checks but no measurement distributions. This
measures the reference backend's operational overhead only. It is not a clean
partial-radix performance profile because exact top-k is still used to obtain
`Tq`.

### 8.2 `summary`

Run the reference approximate selector and approximate attention. Track
GPU-side counters and distributions:

- selected count;
- effective k;
- `delta_k = selected_count - effective_k`;
- selection-count ratio;
- selection density;
- capacity utilization;
- rescue, FP16 overflow, and saturation;
- duplicates and invalid/noncausal indices;
- prefix/`-1`-suffix violations;
- selector count versus FA3 `valid_counts`;
- calls and rows split by layer and prefill/decode.

This is the default mode for normal evaluation runs after its overhead relative
to `off` is measured.

### 8.3 `verify_exact`

Compute exact and approximate selections from the same score tensor:

```text
indexer scores
  |-- approximate selector -> live buffer -> FA3 attention
  `-- stock exact top-k    -> scratch buffer -> comparison only
```

Only approximate attention executes. Add:

- `added = |A \ E|`;
- `dropped = |E \ A|`;
- `intersection = |A intersection E|`;
- `exact_recall = intersection / |E|`;
- `precision = intersection / |A|`;
- strict and tie-aware overlap;
- rows verified against the stock selector;
- containment violations.

This mode validates the selector on the active approximate trajectory. At later
layers, `E` answers "what exact top-k would select from the current approximate
trajectory's scores," not "what a fully exact model trajectory would select."
The latter requires a separate complete exact run.

Because the reference backend already executes exact top-k, `verify_exact`
primarily adds scratch output retention, set comparison, and detailed
telemetry. It remains excluded from timing comparisons even against the
reference backend.

## 9. Metric granularity and retention

Metrics are computed at row granularity:

```text
benchmark sample/request x layer x query position
```

For prefill, the query position is request-local and absolute within the prompt.
For decode, it includes the generated-token step and absolute sequence position.

Retaining every raw layer-position record is too large for routine 32K
evaluation. The retention policy is:

1. Compute every metric per row on device.
2. Retain per-sample/per-position summaries across sparse layers in
   `verify_exact` mode.
3. Retain aggregate per-layer and position-bin distributions.
4. Retain full raw records only for configured samples/layers/position stride.
5. Always retain the complete raw record for every safety or containment
   violation.

At one sample and query position, percentiles describe the cross-layer
distribution. Because this is a small population, retain the maximum and its
responsible layer alongside p99.

## 10. Distribution summaries

Every numeric aggregate uses a common schema:

```json
{
  "n": 102400,
  "mean": 2076.3,
  "std": 48.7,
  "min": 2048,
  "p01": 2048,
  "p10": 2049,
  "p25": 2052,
  "p50": 2061,
  "p75": 2084,
  "p90": 2122,
  "p95": 2157,
  "p99": 2241,
  "p999": 2468,
  "max": 2712
}
```

Primary reporting points are p50, p90, p95, p99, p99.9, and max. Lower
percentiles remain important for an under-capturing selector such as ceil.

Apply this schema to:

- selected count and `delta_k`;
- selection ratio, density, and capacity utilization;
- added and dropped counts;
- exact recall, precision, and Jaccard overlap;
- `Tq`;
- selector and total indexer time;
- TTFT, inter-token latency, and request latency;
- output agreement metrics from paired benchmark runs.

Binary safety properties use count and rate rather than percentiles.

Report distributions globally and split by:

- sample;
- layer;
- prefill/decode;
- query-position bin;
- context-length bucket;
- steady rows where valid count is at least k.

## 11. Top-k overlap and position histograms

For exact set `E`, approximate set `A`, and query position `q`, define:

```text
I = E intersection A
D = E \ A
G = A \ E
```

### 11.1 Per-row overlap histogram

Histogram the following across query rows:

```text
exact recall = |I| / |E|
precision    = |I| / |A|
Jaccard      = |I| / |E union A|
```

Use tail-sensitive edges near 1.0, for example:

```text
0, 0.5, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0
```

### 11.2 Agreement by query position

For fixed absolute-position bins, report:

- rows;
- recall, precision, and Jaccard distributions;
- added and dropped distributions;
- the worst sample, layer, and query position.

Use bins suitable for the evaluated length, such as:

```text
0-2K, 2K-4K, 4K-8K, 8K-16K, 16K-24K, 24K-32K
```

### 11.3 Agreement by key distance

For selected key position `j`, define distance `q - j`. Count exact,
approximate, overlapping, added, and dropped keys in logarithmic distance bins:

```text
0-15, 16-63, 64-255, 256-1023, 1K-4K, 4K-16K, 16K+
```

This reveals whether approximation preferentially changes recent or distant
context.

### 11.4 Agreement by exact rank

Sort the exact selection by score and report retention in rank bands:

```text
1-16, 17-64, 65-256, 257-1024, 1025-2048
```

Approximation should normally affect the lowest-confidence ranks nearest `Tq`,
not the highest-scoring keys.

### 11.5 Two-dimensional heatmaps

Generate query-position-bin by key-distance-bin heatmaps for:

- overlap;
- exact recall;
- added keys;
- dropped keys.

Report strict overlap against the literal stock exact set and tie-aware overlap
that treats keys with score equal to `Tq` as interchangeable.

## 12. Evaluation metrics

An approximate-only server can report normal benchmark metrics unchanged:

- accuracy or exact match;
- pass@1/pass@k;
- retrieval and needle accuracy;
- RULER per-task and aggregate results;
- instruction-following pass rate;
- judge score or win rate;
- sample count and confidence interval.

Run a separate exact-top-k server with identical prompts, decoding parameters,
seeds, concurrency, and evaluation version. Report:

- exact and approximate task scores;
- absolute score delta and relative retention;
- paired confidence interval;
- improved/regressed/unchanged sample counts;
- output length difference;
- token exact-match prefix and first divergence;
- top-token agreement where logits are captured.

Full-vocabulary output KL requires two complete, same-prefix model executions:
one exact and one approximate. Same-pass exact-set verification is insufficient
because only the approximate attention trajectory continues to output logits.
For clean KL, generate an exact reference sequence and replay the same token
prefix through both selectors.

## 13. Performance metrics

The reference backend is expected to be slower than stock exact top-k because it
uses exact top-k and then applies the approximate rule. Its timings are useful
for operational sizing and telemetry-overhead checks, not as evidence for the
future kernel. Run with `dsa_telemetry=off`, then compare with `summary` and
report:

- reference-selector time;
- total indexer time, including scoring;
- TTFT p50/p90/p99;
- inter-token latency p50/p90/p99;
- prompt and output throughput;
- end-to-end latency;
- requests per second;
- peak GPU memory;
- context length, batch size, and concurrency;
- eager or CUDA-graph mode.

Every report must state `selector_speed_claim_valid=false`. Profiling the
absence of exact top-k and claiming approximate-selector speedup are deferred
until a native backend exists.

## 14. Artifact format

Write one JSON artifact per serving/evaluation run:

```text
selector_artifacts/<timestamp>_<selector>_k<k>_cap<capacity>.json
```

Top-level schema:

```json
{
  "meta": {},
  "provenance": {},
  "prefill": {},
  "decode": {},
  "per_layer": {},
  "per_sample": {},
  "position_histograms": {},
  "safety": {},
  "quality": {},
  "performance": {}
}
```

Provenance includes selector configuration, model/checkpoint, git revision,
vLLM version, selector backend, CUDA-graph mode, benchmark revision,
sampling parameters, and seed. The serve manifest records the artifact path.

## 15. Tests and acceptance gates

Add tests for:

- snapshot parity: the untouched exact server and the initial approximate copy
  in `dsa_selector=topk` mode produce identical selected indices, logits, and
  deterministic tokens on fixed fixtures;
- isolation: importing the exact bootstrap does not install selector hooks and
  starting the approximate bootstrap does not mutate files or configuration in
  the exact serving directory;
- FP16 monotonic ordering, negative values, ties, and signed zero;
- FP16 overflow rejection;
- short-prefix behavior;
- floor/midpoint/ceil threshold semantics;
- ported-rule parity with the `dsa-csx` source tests;
- reference-emitter equality with `dsa-csx` on shared fixtures;
- all four vLLM interception paths;
- request-local and causal output indices;
- suffix-only `-1` padding;
- capacity overflow;
- same-pass exact verification;
- layer and query-position attribution;
- eager/CUDA-graph parity;
- repeated-run determinism;
- unchanged exact-top-k mode.

Fatal safety gates:

```text
invalid indices              == 0
noncausal indices            == 0
duplicate indices            == 0
padding violations           == 0
FA3 count mismatches         == 0
FP16 overflow                == 0
capacity saturation          == 0
unexpected empty rescue      == 0
containment violations       == 0
```

Selector-specific gates:

```text
radix_ceil:  added == 0
radix_floor: dropped == 0
exact_ge:    dropped == 0
```

No selector speedup claim is permitted for `dsa_csx_reference`. Quality
acceptance thresholds will be fixed before the full benchmark campaign and
evaluated with paired confidence intervals. Native-kernel performance gates are
deferred.

## 16. Implementation and evaluation sequence

1. Record the source revision, snapshot the working exact plugin into
   `vllm_qwen3_dsa_approx`, rename its architecture/indexer classes, and add
   separate approximate bootstrap, builder, entry point, and serve script.
2. Before any selector change, pass a parity gate between the untouched exact
   server and the copied server in `dsa_selector=topk` mode. Stop if selected
   indices, logits, or deterministic tokens differ.
3. Add approximate-only configuration parsing, hard validation, and manifest
   fields. Verify that the exact serving directory and exact source paths have
   no diff.
4. Port the attributed `dsa-csx` rules, summaries, and reference emitter with
   their source tests and shared fixtures.
5. Add version-pinned vLLM hooks and layer/request/position attribution only to
   the approximate bootstrap and package.
6. Add `summary` and `verify_exact` telemetry plus JSON artifacts.
7. Resize the shared buffer and strengthen the FA3 contract checks in the
   approximate copy only.
8. Re-run the exact-versus-approximate-copy `topk` parity gate after each
   integration stage.
9. Run same-pass Qwen validation for ceil, midpoint, and floor at 4K, 16K,
   21K needle, and 32K.
10. Use the Qwen count tails and overlap histograms to select floor capacity and
   the leading selector candidate.
11. Run paired exact-server-versus-approximate-server benchmark evaluations
    using separate processes and the
   `dsa_csx_reference` backend.
12. Report reference-backend operational latency and telemetry overhead with an
    explicit prohibition on selector speedup claims.
13. Select the preferred Qwen selector semantics and capacity from safety and
    quality evidence. Keep exact top-k as the explicit control and rollback.
14. Defer native partial-radix kernel implementation and performance gating to
    a future phase that preserves the established rule and telemetry APIs.

## 17. Planned file changes

New isolated approximate snapshot and implementation files:

```text
scripts/dsa/vllm_qwen3_dsa_approx/__init__.py
scripts/dsa/vllm_qwen3_dsa_approx/model.py
scripts/dsa/vllm_qwen3_dsa_approx/indexer.py
scripts/dsa/vllm_qwen3_dsa_approx/sparse_attention.py
scripts/dsa/vllm_qwen3_dsa_approx/radix_rules.py
scripts/dsa/vllm_qwen3_dsa_approx/radix_selector_reference.py
scripts/dsa/vllm_qwen3_dsa_approx/selector_hooks.py
scripts/dsa/vllm_qwen3_dsa_approx/selector_runtime.py
scripts/dsa/vllm_qwen3_dsa_approx/selector_telemetry.py
scripts/dsa/serving/_pluginboot_approx/sitecustomize.py
scripts/dsa/serving/serve_qwen3_dsa_approx.sh
scripts/dsa/serving/serve_qwen3_dsa_approx_entry.py
scripts/dsa/build_qwen3_dsa_approx_serving_dir.py
```

Deferred future implementation file:

```text
scripts/dsa/vllm_qwen3_dsa_approx/radix_selector_triton.py
```

Protected exact-serving paths (no planned edits):

```text
scripts/dsa/vllm_qwen3_dsa/
scripts/dsa/serving/_pluginboot/
scripts/dsa/serving/serve_qwen3_dsa.sh
scripts/dsa/serving/serve_qwen3_dsa_entry.py
scripts/dsa/build_qwen3_dsa_serving_dir.py
```

New tests:

```text
tests/dsa/approx/test_exact_copy_parity.py
tests/dsa/approx/test_qwen3_dsa_radix_rules.py
tests/dsa/approx/test_qwen3_dsa_radix_selector.py
tests/dsa/approx/test_qwen3_dsa_selector_hooks.py
tests/dsa/approx/test_qwen3_dsa_selector_telemetry.py
```

CI/review adds an allowlist check that rejects a change if any protected exact
path appears in the implementation diff. This is in addition to runtime parity,
not a substitute for it.

Before proposing a PR, run the repository-mandated duplicate-work checks and
record the exact test and benchmark commands in the PR description.
