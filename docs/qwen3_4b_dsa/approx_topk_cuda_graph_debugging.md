# Approximate top-k CUDA-graph debugging conclusions

**Date:** 2026-08-28
**Scope:** Qwen3-4B DSA approximate selectors in vLLM 0.26, especially concurrent decode under
`FULL_DECODE_ONLY` CUDA graphs.

This report records two investigations that initially appeared to be one problem:

1. a real CUDA-graph padded-row safety defect that killed serving workers; and
2. a later greedy-token parity failure caused by cross-process numerical variability, not by the
   padding fix or the copied approximate plugin.

Keeping those conclusions separate is essential. The first required a selector fix. The second
required a validation-gate fix.

## 1. Production failure

A `radix_floor`, `dsa_top_k=2048`, `index_topk=4096`, `graph_verify_exact` evaluation against the
step-2200 checkpoint failed several minutes into concurrent serving. Five of eight vLLM servers
exited with:

```text
CUDA error: device-side assert triggered
approximate selector emitted an invalid or noncausal request-local index
```

The named assertion originated in the approximate selector's mandatory safety gates. The failing
scheduler dump was a pure decode batch containing seven one-token requests.

### Evaluation-harness traps

Two independent harness behaviors initially obscured the failure:

- the evaluation client swallowed most connection failures, wrote a zero-score result, and exited
  successfully; and
- the outer queue treated those zero-return-code jobs as successful.

Telemetry polled before the workers died was clean, but it described only rows processed before the
fatal batch. It could not prove that later rows were safe.

The affected evaluation outputs were quarantined rather than treated as model-quality evidence.

## 2. Why earlier stages missed it

The original Stage A/B/C gates and `qwen3_dsa_offline_smoke.py` submitted only two nearly
equal-length prompts. They covered two-request decode but never exercised a non-power-of-two batch
that vLLM had to pad to a captured CUDA-graph shape.

The production scheduler submitted seven real decode requests. vLLM replayed the eight-row capture
bucket:

```text
7 active request rows + 1 inactive graph-padding row
```

That shape was absent from the historical smoke fixture. The historical PASS claims remain valid
for their two-request workloads, but they were too narrow to establish production concurrency
safety.

## 3. Debug arms and attribution

### Telemetry disabled

The failing workload was repeated with selector telemetry disabled. It failed at the same assertion.

**Conclusion:** telemetry was not the cause and `telemetry=off` was not a workaround. Mandatory
safety gates remain active independently of telemetry.

### Enforce eager

The same multi-request load was run with `--enforce-eager`. Eight concurrent decodes ran cleanly
with no assertions or HTTP 500 responses.

**Conclusion:** the trigger depended on CUDA-graph execution. Eager mode does not add an inactive
padding row to a captured batch.

### Resource checks

Host memory, GPU memory pressure, GPU contention, and selector capacity saturation were checked and
did not explain the failure.

**Conclusion:** the device assertion was a selector-input/graph-shape defect rather than an
infrastructure failure.

## 4. Root cause

The decode hook derives request-local query positions from sequence lengths:

```python
query_position = sequence_length - 1
```

For an unused CUDA-graph row, vLLM supplies sequence length zero, producing query position `-1`.
The selector originally treated every row as an active request. Its ordinary empty-selection rescue
could manufacture key zero for the inactive row, after which causal/index validation correctly
rejected the result.

The safety assertion was therefore doing its job: the selector had violated its input contract by
treating graph padding as a request.

## 5. Implemented fix

The reference selector now defines:

```python
active = query_position >= 0
```

Inactive rows:

- remain entirely `-1` padded;
- do not run empty-selection rescue;
- do not contribute to selected counts;
- are excluded from quality and safety telemetry row accounting; and
- still participate in explicit padding-contract validation.

Active rows retain all existing fail-closed checks for invalid, noncausal, duplicate, malformed,
rescued, or capacity-saturating selections.

## 6. Padding-fix validation

### Fixed-input CUDA replay

Focused CUDA tests captured both `radix_floor` and `radix_ceil`, mutated query positions, logits, and
the padding row between replays, and completed cleanly. The focused artifact reported two passing
tests.

### Concurrent Stage fixture

The smoke harness was expanded to submit seven heterogeneous long prompts in one `LLM.generate`
call and force all requests to decode the complete output window. This deliberately exercises the
7-to-8 graph bucket.

The expanded eight-arm run established:

- all eight processes completed;
- all seven requests completed in every arm;
- prompt lengths were 9,034–9,168 tokens, above `k=2048`;
- all completions contained 32 forced tokens;
- 51 decode graph shapes captured without stream-capture failure;
- graph decode telemetry counted only the seven active rows;
- floor and ceil reported zero hard safety violations, rescue, or capacity saturation; and
- floor/ceil set-containment invariants held.

This is the direct end-to-end evidence that the padded-row defect is fixed.

## 7. Secondary token-parity failure

The expanded runner still returned a failure because it required independent eager and graph
processes to emit bit-identical greedy tokens and bit-identical aggregate telemetry. Several pairs
differed at request 0, generation position 12:

```text
token 382: punctuation with a period
token 271: the alternate no-period/newline continuation
```

The remaining tokens often reconverged. At first, the mismatch appeared to follow the isolated
`Qwen3DSAApproxForCausalLM` copy.

## 8. Numerical-parity experiments

### Same-GPU repeats

Initial two-run controls appeared self-consistent: exact repeatedly chose token 382 while copied
top-k repeatedly chose token 271. That sample was too small.

### Batch-size sweep

Exact versus copied-top-k results were:

| Concurrent requests | Greedy token equality |
|---:|---|
| 2 | different at position 12 |
| 4 | exact |
| 6 | exact |
| 7 | different at position 12 in the expanded Stage run |
| 8 | exact |

**Conclusion:** there was no monotonic concurrency threshold and no signature specific to graph
padding. These were eager processes.

### Logprob capture

The smoke fixture was extended to record top-token logprobs. One batch-size-two comparison showed:

```text
exact:       token 382 beats 271 by 0.125
copied-topk: tokens 382 and 271 tie at 0.0 margin
```

Relative logprob differences were present before the greedy-token divergence, so the difference was
not merely an additive log-softmax normalization shift.

### Exact architecture on the copied directory

The exact architecture was loaded from the same copied-topk model directory. That run chose token
382 with a 0.125 margin.

At that point the evidence appeared to implicate the copied architecture, but later replication
showed that this single control was also insufficient.

### Contract-check bypass

A diagnostic-only copied-topk run skipped the additional selector-to-FA3 contract checks. It still
produced the zero-margin decision.

**Conclusion:** the safety checks did not cause the numerical variation. They remain mandatory.

### Rotate custom-op namespace

Six fresh copied-topk processes were run in interleaved pairs. Three used the ordinary
`qwen3_dsa_approx::rotate_activation` namespace and three used the exact
`qwen3_dsa::rotate_activation` namespace, with identical implementation and all safety checks
enabled.

Observed token-382-versus-271 margins were:

```text
plain copied-topk:   0.125, 0.250, 0.125
exact-namespace arm: 0.125, 0.000, 0.125
```

**Conclusion:** the namespace did not determine the result. The replicated runs instead exposed
fresh-process numerical variability.

### Deciding exact-plugin replication

Three fresh exact-architecture processes were then run back-to-back using the same copied model
directory:

```text
exact r1: token 382, margin 0.125
exact r2: token 382, margin 0.125
exact r3: token 271, margin 0.000
```

All three completed successfully with the same seed and configuration.

**Conclusion:** the exact plugin is also numerically variable across fresh vLLM processes. The
apparent exact-versus-copy asymmetry was sampling noise around a borderline BF16 decision, not an
approximate-plugin defect.

The diagnostic bypass and namespace switches were removed after reaching this conclusion.

## 9. Revised validation policy

Cross-process greedy token equality remains useful evidence, but it is no longer a hard gate. A
fixed seed does not make independent vLLM processes bit-identical at floating-point decision
boundaries.

### Deterministic hard gates

The following remain exact:

- fixed-input selector reference tests;
- fixed-input CUDA capture and replay tests;
- active/padding row accounting;
- invalid, noncausal, duplicate, malformed-padding, count-mismatch, rescue, and capacity counters;
- `radix_floor`: `dropped == 0` and `exact_recall == 1`;
- `radix_ceil`: `added == 0` and `precision == 1`;
- complete graph capture without invalidation; and
- all concurrent requests, forced output lengths, long prompts, and needle retrieval.

### Tolerance-based end-to-end gates

For independent eager/graph processes, the revised runner:

- records top-20 logprobs at every generation step;
- compares distributions only while generated prefixes remain identical;
- subtracts each distribution's best shared logprob to remove additive normalization shifts;
- hard-gates the mean and maximum centered-logprob deltas against tolerances above the measured
  exact-versus-exact variability envelope;
- compares eager/graph quality means with bounded tolerance; and
- reports token equality, agreement rate, and first divergence diagnostically.

The default output tolerances are:

```text
mean centered logprob delta <= 0.20
max centered logprob delta  <= 1.0
quality mean relative delta <= 1%
quality mean absolute floor <= 0.001
```

The new summary writes explicit `hard_failure_reasons`; a false `token_exact` value alone cannot
fail the run.

## 10. Current conclusion and remaining validation

The production device assertion and the later token mismatch have different resolutions:

| Finding | Resolution |
|---|---|
| Seven-request graph batch crashed on its padded eighth row | Fix selector inactive-row handling |
| Historical two-request stages missed the crash | Add heterogeneous seven-request fixture |
| Telemetry-off still crashed | Keep mandatory safety gates independent of telemetry |
| Eager concurrent load was clean | Confirm graph-padding trigger |
| Exact/copy/eager/graph tokens sometimes differ | Treat cross-process greedy equality diagnostically |
| Exact plugin also chooses both borderline tokens | Fix the gate, not the copied plugin |

The revised runner and CPU tests are complete. The remaining step is a fresh eight-GPU Stage run
using the new logprob-tolerance verdict. Production evaluations should resume only after that run
reports:

```text
passed: true
hard_failure_reasons: []
```

## 11. Relevant files

- `scripts/dsa/vllm_qwen3_dsa_approx/radix_selector_reference.py`: inactive-row selection rule.
- `scripts/dsa/vllm_qwen3_dsa_approx/selector_runtime.py`: graph replay safety and quality accounting.
- `tests/dsa/qwen3_dsa_offline_smoke.py`: heterogeneous concurrent fixture and logprob artifacts.
- `tests/dsa/run_qwen3_dsa_approx_gpu_validation.sh`: eight-arm GPU campaign.
- `tests/dsa/qwen3_dsa_approx_validation_summary.py`: revised verdict and explicit failure reasons.
- `tests/dsa/approx/test_qwen3_dsa_validation_summary.py`: CPU tests for the revised gate.
- [`approx_topk_selector_overview.md`](approx_topk_selector_overview.md): selector terminology and
  partial-radix rules.
- [`approx_topk_serving_plan.md`](approx_topk_serving_plan.md): full implementation and rollout plan.
