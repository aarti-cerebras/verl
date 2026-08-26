import json

import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_approx import selector_runtime as runtime_module
from scripts.dsa.vllm_qwen3_dsa_approx.radix_selector_reference import (
    select_prefix_reference,
)
from scripts.dsa.vllm_qwen3_dsa_approx.selector_runtime import (
    SelectorConfig,
    SelectorRuntime,
)
from scripts.dsa.vllm_qwen3_dsa_approx.selector_telemetry import (
    COUNT_BINS,
    BoundedDist,
    dist_values,
    integer_spec,
    ratio_spec,
)


def _selector_config(*, selector: str = "radix_ceil", telemetry: str = "off") -> SelectorConfig:
    return SelectorConfig(
        selector=selector,
        backend="vllm_stock" if selector == "topk" else "dsa_csx_reference",
        rule_k=4,
        capacity=4,
        omit_bits=4,
        telemetry=telemetry,
    )


@pytest.mark.parametrize("telemetry", ["summary", "verify_exact"])
def test_cuda_graph_rejects_host_folded_telemetry(telemetry: str) -> None:
    with pytest.raises(ValueError, match="require dsa_telemetry='off' or 'graph_safety'"):
        _selector_config(telemetry=telemetry).validate_execution(cudagraph_enabled=True)


def test_cuda_graph_accepts_telemetry_off_and_topk_control() -> None:
    _selector_config(telemetry="off").validate_execution(cudagraph_enabled=True)
    _selector_config(telemetry="graph_safety").validate_execution(cudagraph_enabled=True)
    _selector_config(selector="topk", telemetry="verify_exact").validate_execution(
        cudagraph_enabled=True
    )


def test_graph_safety_counters_are_persistent_attributed_and_reset_in_place() -> None:
    runtime = SelectorRuntime()
    runtime.configure(_selector_config(telemetry="graph_safety"))
    runtime.initialize_graph_safety(["layer.0", "layer.1"], device=torch.device("cpu"))
    assert runtime._graph_safety is not None
    storage_address = runtime._graph_safety.data_ptr()

    logits = torch.randn(3, 12)
    qpos = torch.tensor([7, 9, 11])
    for layer, phase in (("layer.0", "prefill"), ("layer.1", "decode")):
        output = torch.empty(3, 4, dtype=torch.int32)
        result = select_prefix_reference(logits, qpos, 4, output, "radix_ceil")
        with runtime.layer(layer):
            runtime.validate_and_record(
                phase=phase,
                output=output,
                query_positions=qpos,
                result=result,
                stock_reference=None,
                key_count=logits.shape[1],
            )

    artifact = runtime.artifact()
    assert artifact["safety"]["rows"] == 6
    assert not any(artifact["safety"]["counts"].values())
    replay = artifact["graph_replay_safety"]
    assert replay["global"]["rows"] == 6
    assert replay["global"]["calls"] == 2
    assert replay["phases"]["prefill"]["per_layer"]["layer.0"]["rows"] == 3
    assert replay["phases"]["decode"]["per_layer"]["layer.1"]["rows"] == 3

    runtime.reset()
    assert runtime._graph_safety.data_ptr() == storage_address
    reset_artifact = runtime.artifact()
    assert reset_artifact["safety"]["rows"] == 0
    assert reset_artifact["graph_replay_safety"]["global"]["calls"] == 0


def test_distribution_contract_contains_tail_percentiles() -> None:
    summary = dist_values(torch.arange(1000))
    assert set(("p50", "p90", "p95", "p99", "p999", "max")) <= set(
        summary["percentiles"]
    )


def test_verify_exact_artifact_contains_overlap_and_position() -> None:
    runtime = SelectorRuntime()
    runtime.configure(
        SelectorConfig(
            selector="radix_ceil",
            backend="dsa_csx_reference",
            rule_k=4,
            capacity=4,
            omit_bits=4,
            telemetry="verify_exact",
        )
    )
    torch.manual_seed(41)
    logits = torch.randn(3, 12)
    qpos = torch.tensor([5, 8, 11])
    approximate = torch.empty(3, 4, dtype=torch.int32)
    exact = torch.empty_like(approximate)
    result = select_prefix_reference(logits, qpos, 4, approximate, "radix_ceil")
    select_prefix_reference(logits, qpos, 4, exact, "topk")
    with runtime.layer("model.layers.7.self_attn.indexer"):
        runtime.note_call("prefill")
        runtime.validate_and_record(
            phase="prefill",
            output=approximate,
            query_positions=qpos,
            result=result,
            stock_reference=exact,
            key_count=logits.shape[1],
        )
    artifact = runtime.artifact()
    layer = artifact["prefill"]["per_layer"]["model.layers.7.self_attn.indexer"]
    assert layer["query_position"]["n"] == 3
    assert layer["exact_recall"]["n"] == 3
    assert layer["precision"]["n"] == 3
    assert layer["rank_recall_1_16"]["n"] == 3
    # `distance_*` is gated off by default (DSA_APPROX_DISTANCE): it is 21 of the 39 folded fields
    # and ~48% of verify_exact's cost. Absence is recorded, so an artifact cannot be misread as
    # "distance recall was zero" when the group was simply not collected.
    assert not [name for name in layer if name.startswith("distance_")]
    assert artifact["meta"]["retention"]["distance_telemetry"] is False
    assert artifact["position_histograms"]["prefill"][0]["rows"] == 3
    assert artifact["meta"]["selector_speed_claim_valid"] is False


def test_distance_telemetry_is_collected_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "DISTANCE_TELEMETRY", True)
    runtime = SelectorRuntime()
    runtime.configure(
        SelectorConfig(
            selector="radix_ceil",
            backend="dsa_csx_reference",
            rule_k=4,
            capacity=4,
            omit_bits=4,
            telemetry="verify_exact",
        )
    )
    torch.manual_seed(41)
    logits = torch.randn(3, 12)
    qpos = torch.tensor([5, 8, 11])
    approximate = torch.empty(3, 4, dtype=torch.int32)
    exact = torch.empty_like(approximate)
    result = select_prefix_reference(logits, qpos, 4, approximate, "radix_ceil")
    select_prefix_reference(logits, qpos, 4, exact, "topk")
    with runtime.layer("model.layers.7.self_attn.indexer"):
        runtime.note_call("prefill")
        runtime.validate_and_record(
            phase="prefill",
            output=approximate,
            query_positions=qpos,
            result=result,
            stock_reference=exact,
            key_count=logits.shape[1],
        )
    artifact = runtime.artifact()
    layer = artifact["prefill"]["per_layer"]["model.layers.7.self_attn.indexer"]
    assert layer["distance_recall_0_16"]["n"] == 3
    assert layer["distance_dropped_0_16"]["n"] == 3
    assert layer["distance_added_0_16"]["n"] == 3
    # every band, all three metrics
    expected = 3 * len(runtime_module.DISTANCE_BANDS)
    assert len([name for name in layer if name.startswith("distance_")]) == expected
    assert artifact["meta"]["retention"]["distance_telemetry"] is True


def test_empty_band_rate_is_absent_rather_than_zero(monkeypatch) -> None:
    """An empty band has no recall; it must not fold a 0.0 that reads as total failure.

    Every query here sits within 12 tokens, so only the 0-16 band holds exact keys. The far bands
    used to divide by a clamped denominator and record recall 0.0 for every row, making "no keys
    were ever this far away" indistinguishable from "every distant key was dropped".
    """

    monkeypatch.setattr(runtime_module, "DISTANCE_TELEMETRY", True)
    runtime = SelectorRuntime()
    runtime.configure(
        SelectorConfig(
            selector="radix_ceil",
            backend="dsa_csx_reference",
            rule_k=4,
            capacity=4,
            omit_bits=4,
            telemetry="verify_exact",
        )
    )
    torch.manual_seed(41)
    logits = torch.randn(3, 12)
    qpos = torch.tensor([5, 8, 11])
    approximate = torch.empty(3, 4, dtype=torch.int32)
    exact = torch.empty_like(approximate)
    result = select_prefix_reference(logits, qpos, 4, approximate, "radix_ceil")
    select_prefix_reference(logits, qpos, 4, exact, "topk")
    with runtime.layer("model.layers.0.self_attn.indexer"):
        runtime.note_call("prefill")
        runtime.validate_and_record(
            phase="prefill",
            output=approximate,
            query_positions=qpos,
            result=result,
            stock_reference=exact,
            key_count=logits.shape[1],
        )
    layer = runtime.artifact()["prefill"]["per_layer"]["model.layers.0.self_attn.indexer"]

    # the populated band carries real rows
    assert layer["distance_recall_0_16"]["n"] == 3
    assert layer["distance_recall_0_16"]["mean"] > 0.0

    # every empty band contributes no samples at all, rather than three zeros
    for label in ("16_64", "64_256", "256_1024", "1024_4096", "4096_16384", "16384_plus"):
        summary = layer[f"distance_recall_{label}"]
        assert summary["n"] == 0, label
        assert "mean" not in summary, label

    # counts are NOT rates: an empty band genuinely dropped and added nothing
    for label in ("16_64", "16384_plus"):
        assert layer[f"distance_dropped_{label}"]["n"] == 3
        assert layer[f"distance_dropped_{label}"]["mean"] == 0.0
        assert layer[f"distance_added_{label}"]["n"] == 3


def test_artifact_is_strict_json_with_undefined_rates(monkeypatch) -> None:
    """NaN must not reach the artifact: undefined per-row values serialize as null."""

    monkeypatch.setattr(runtime_module, "DISTANCE_TELEMETRY", True)
    monkeypatch.setattr(runtime_module, "RAW_ROWS_PER_GROUP", 8)
    runtime = SelectorRuntime()
    runtime.configure(
        SelectorConfig(
            selector="radix_ceil",
            backend="dsa_csx_reference",
            rule_k=4,
            capacity=4,
            omit_bits=4,
            telemetry="verify_exact",
        )
    )
    torch.manual_seed(41)
    logits = torch.randn(3, 12)
    qpos = torch.tensor([0, 64, 128])
    approximate = torch.empty(3, 4, dtype=torch.int32)
    exact = torch.empty_like(approximate)
    result = select_prefix_reference(logits, qpos, 4, approximate, "radix_ceil")
    select_prefix_reference(logits, qpos, 4, exact, "topk")
    with runtime.layer("model.layers.0.self_attn.indexer"):
        runtime.note_call("prefill")
        runtime.validate_and_record(
            phase="prefill",
            output=approximate,
            query_positions=qpos,
            result=result,
            stock_reference=exact,
            key_count=logits.shape[1],
        )
    # allow_nan=False raises on any NaN/Infinity that leaked into a per-row record
    json.loads(json.dumps(runtime.artifact(), allow_nan=False))


# --------------------------------------------------------------------------------------------- #
# Retention (plan §9). Telemetry used to accumulate every raw per-row record on the host, which
# both grew without bound and eventually hit torch.quantile's hard 2**24-element ceiling at
# artifact-dump time -- i.e. AFTER a whole 32K benchmark run had already been paid for.


def _clean_runtime(selector: str = "radix_ceil", rule_k: int = 4, capacity: int = 4):
    runtime = SelectorRuntime()
    runtime.configure(
        SelectorConfig(
            selector=selector,
            backend="dsa_csx_reference",
            rule_k=rule_k,
            capacity=capacity,
            omit_bits=4,
            telemetry="verify_exact",
        )
    )
    return runtime


def _record(runtime, logits, qpos, rule_k, capacity, selector, layer, phase="prefill"):
    approximate = torch.empty(logits.shape[0], capacity, dtype=torch.int32)
    exact = torch.empty(logits.shape[0], rule_k, dtype=torch.int32)
    result = select_prefix_reference(logits, qpos, rule_k, approximate, selector)
    select_prefix_reference(logits, qpos, rule_k, exact, "topk")
    with runtime.layer(layer):
        runtime.note_call(phase)
        runtime.validate_and_record(
            phase=phase,
            output=approximate,
            query_positions=qpos,
            result=result,
            stock_reference=exact,
            key_count=logits.shape[1],
        )


def test_count_percentiles_are_exact_against_torch_quantile() -> None:
    # §16 step 10 sizes floor capacity from the count tails, so a coarse bin there would mis-size
    # the buffer. Count metrics get one bin per value and must be exact, not approximate.
    torch.manual_seed(61)
    population = torch.randint(0, 2305, (200_000,))
    accumulator = BoundedDist(integer_spec(0, 2304, COUNT_BINS))
    for chunk in population.split(4093):
        accumulator.add(chunk)
    summary = accumulator.summary()

    assert summary["exact_percentiles"] is True
    assert summary["n"] == population.numel()
    for name, fraction in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p999", 0.999)):
        assert summary["percentiles"][name] == float(
            torch.quantile(population.float(), fraction, interpolation="lower")
        )
    assert summary["min"] == float(population.min())
    assert summary["max"] == float(population.max())
    assert abs(summary["mean"] - float(population.double().mean())) < 1e-6


def test_ratio_percentiles_stay_within_one_bin() -> None:
    torch.manual_seed(62)
    population = torch.rand(200_000)
    accumulator = BoundedDist(ratio_spec())
    accumulator.add(population)
    summary = accumulator.summary()
    for name, fraction in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p999", 0.999)):
        error = abs(summary["percentiles"][name] - float(torch.quantile(population, fraction)))
        assert error <= accumulator.spec.width


def test_accumulator_cost_is_independent_of_row_count() -> None:
    accumulator = BoundedDist(integer_spec(0, 2304, COUNT_BINS))
    footprint = accumulator._counts.numel()
    for _ in range(200):
        accumulator.add(torch.randint(0, 2304, (4096,)))
    assert accumulator.summary()["n"] == 200 * 4096
    assert accumulator._counts.numel() == footprint


def test_merging_layers_preserves_the_exact_population() -> None:
    torch.manual_seed(63)
    parts = [torch.randint(0, 300, (5_000,)) for _ in range(4)]
    merged = BoundedDist(integer_spec(0, 2304, COUNT_BINS))
    for part in parts:
        piece = BoundedDist(integer_spec(0, 2304, COUNT_BINS))
        piece.add(part)
        merged.merge(piece)
    whole = torch.cat(parts)
    summary = merged.summary()
    assert summary["n"] == whole.numel()
    assert summary["max"] == float(whole.max())
    assert summary["percentiles"]["p99"] == float(
        torch.quantile(whole.float(), 0.99, interpolation="lower")
    )


def test_samples_outside_the_spec_are_reported_not_hidden() -> None:
    accumulator = BoundedDist(ratio_spec())
    accumulator.add(torch.tensor([0.5, 1.5, -0.25]))
    summary = accumulator.summary()
    assert summary["out_of_range"] == {"below": 1, "above": 1}
    assert summary["max"] == 1.5 and summary["min"] == -0.25


def test_dist_values_survives_a_population_above_the_quantile_limit() -> None:
    # torch.quantile refuses anything above 2**24 elements; one 32K prompt across the sparse
    # layers reaches that, so the helper has to degrade to the histogram instead of raising.
    rows = (1 << 24) + 1024
    with pytest.raises(RuntimeError, match="too large"):
        torch.quantile(torch.zeros(rows), torch.tensor([0.5]))
    summary = dist_values(torch.rand(rows))
    assert summary["n"] == rows
    assert 0.4 < summary["percentiles"]["p50"] < 0.6


def test_violation_rows_are_retained_in_full() -> None:
    # Every score shares one FP16 bucket, so ceil's threshold clears the row and each row is
    # rescued to its best key. Rescue is a §15 gate, so the whole raw record must be kept.
    runtime = _clean_runtime()
    _record(runtime, torch.ones(3, 12), torch.tensor([5, 8, 11]), 4, 4, "radix_ceil", "layer.0")
    safety = runtime.artifact()["safety"]

    assert safety["rows"] == 3
    assert safety["counts"]["rescued"] == 3
    assert safety["rates"]["rescued"] == 1.0
    assert len(safety["violation_records"]) == 3
    record = safety["violation_records"][0]
    assert record["flags"] == ["rescued"]
    assert record["layer"] == "layer.0" and record["phase"] == "prefill"
    assert record["selected_count"] == 1.0
    assert "query_position" in record and "exact_recall" in record


def test_a_clean_run_reports_every_safety_counter_at_zero() -> None:
    torch.manual_seed(64)
    runtime = _clean_runtime(selector="radix_floor", rule_k=4, capacity=8)
    _record(runtime, torch.randn(6, 24), torch.arange(6) * 4 + 3, 4, 8, "radix_floor", "layer.0")
    safety = runtime.artifact()["safety"]
    assert safety["counts"] == {
        "count_mismatch": 0,
        "padding_violation": 0,
        "invalid_index": 0,
        "noncausal_index": 0,
        "duplicate_index": 0,
        "capacity_saturation": 0,
        "rescued": 0,
        "containment_added": 0,
        "containment_dropped": 0,
    }
    assert safety["violation_records"] == []


def test_raw_retention_honours_the_configured_stride_and_cap(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "RAW_ROWS_PER_GROUP", 3)
    monkeypatch.setattr(runtime_module, "RAW_POSITION_STRIDE", 4)
    torch.manual_seed(65)
    runtime = _clean_runtime(selector="radix_floor", rule_k=2, capacity=4)
    qpos = torch.arange(16)
    _record(runtime, torch.randn(16, 16), qpos, 2, 4, "radix_floor", "layer.0")
    raw = runtime.artifact()["prefill"]["raw_records"]["layer.0"]

    # Positions 0, 4, 8, 12 pass the stride; the cap keeps the first three.
    assert raw["rows"] == 3
    assert raw["position_stride"] == 4
    assert raw["fields"]["query_position"] == [0.0, 4.0, 8.0]


def test_raw_retention_is_off_by_default() -> None:
    torch.manual_seed(66)
    runtime = _clean_runtime(selector="radix_floor", rule_k=2, capacity=4)
    _record(runtime, torch.randn(8, 16), torch.arange(8) + 4, 2, 4, "radix_floor", "layer.0")
    assert "raw_records" not in runtime.artifact()["prefill"]


def test_aggregates_are_bounded_but_still_cover_every_row() -> None:
    torch.manual_seed(67)
    runtime = _clean_runtime(selector="radix_floor", rule_k=8, capacity=16)
    logits = torch.randn(64, 128)
    qpos = torch.randint(8, 128, (64,))
    for layer in range(6):
        for _ in range(5):
            _record(runtime, logits, qpos, 8, 16, "radix_floor", f"model.layers.{layer}")
    artifact = runtime.artifact()

    # Every row is represented in the aggregate even though nothing raw was retained.
    assert artifact["prefill"]["global"]["selected_count"]["n"] == 6 * 5 * 64
    assert artifact["safety"]["rows"] == 6 * 5 * 64
    assert len(artifact["prefill"]["per_layer"]) == 6
    assert set(artifact["prefill"]["max_by_layer"]["delta_k"]) == {"max", "layer"}
    assert artifact["prefill"]["max_by_layer"]["delta_k"]["layer"].startswith("model.layers.")
    assert artifact["meta"]["retention"]["raw_rows_per_group"] == 0


def test_identical_passes_produce_an_identical_artifact() -> None:
    def run() -> str:
        torch.manual_seed(68)
        runtime = _clean_runtime(selector="radix_midpoint", rule_k=4, capacity=8)
        logits = torch.randn(12, 64)
        qpos = torch.randint(4, 64, (12,))
        for layer in range(3):
            _record(runtime, logits, qpos, 4, 8, "radix_midpoint", f"layer.{layer}")
        return json.dumps(runtime.artifact(), sort_keys=True)

    assert run() == run()


def test_reset_clears_telemetry_but_keeps_configuration() -> None:
    torch.manual_seed(69)
    runtime = _clean_runtime(selector="radix_floor", rule_k=4, capacity=8)
    _record(runtime, torch.randn(4, 32), torch.arange(4) + 8, 4, 8, "radix_floor", "layer.0")
    runtime.reset()
    artifact = runtime.artifact()
    assert "prefill" not in artifact
    assert artifact["safety"]["rows"] == 0
    assert runtime.config is not None and runtime.config.selector == "radix_floor"


def test_runtime_assigns_exact_bins_to_every_count_metric() -> None:
    # The exactness guarantee has to hold through the runtime's own field->spec mapping, not just
    # for a hand-built spec: this is what a change to _spec would silently coarsen.
    torch.manual_seed(70)
    runtime = _clean_runtime(selector="radix_floor", rule_k=8, capacity=16)
    _record(runtime, torch.randn(10, 64), torch.randint(8, 64, (10,)), 8, 16, "radix_floor", "l0")
    layer = runtime.artifact()["prefill"]["per_layer"]["l0"]

    for field in (
        "selected_count",
        "effective_k",
        "delta_k",
        "intersection",
        "added",
        "dropped",
        "duplicates",
        "rescued",
    ):
        assert layer[field]["exact_percentiles"] is True, field
        assert "out_of_range" not in layer[field], field
    for field in ("capacity_utilization", "exact_recall", "precision", "jaccard"):
        assert layer[field]["exact_percentiles"] is False, field


def test_delta_k_spec_admits_an_under_capturing_arm() -> None:
    # radix_ceil is a subset of exact top-k, so delta_k is negative. A non-negative spec would
    # clamp those rows into bin zero and quietly report the tail as zero.
    runtime = _clean_runtime(selector="radix_ceil", rule_k=4, capacity=4)
    _record(runtime, torch.ones(2, 12), torch.tensor([9, 11]), 4, 4, "radix_ceil", "l0")
    delta = runtime.artifact()["prefill"]["per_layer"]["l0"]["delta_k"]
    assert delta["max"] < 0
    assert delta["exact_percentiles"] is True
    assert "out_of_range" not in delta


def test_spec_stays_exact_at_serving_capacity() -> None:
    # The real serving geometry is k=2048 with a floor capacity of 2304, and delta_k spans
    # [-2304, 2304]. A bin budget below that span is what silently coarsens the count tails, and a
    # capacity-16 fixture is far too small to notice.
    runtime = SelectorRuntime()
    runtime.configure(
        SelectorConfig(
            selector="radix_floor",
            backend="dsa_csx_reference",
            rule_k=2048,
            capacity=2304,
            omit_bits=4,
            telemetry="summary",
        )
    )
    counts = torch.zeros(4, dtype=torch.int64)
    for field in ("selected_count", "delta_k", "added", "dropped", "intersection"):
        spec = runtime._spec(field, counts)
        assert spec.exact, f"{field} bins are coarser than one per value: {spec}"
    assert runtime._spec("delta_k", counts).low == -2304
    assert runtime._spec("selected_count", counts).low == 0
    assert runtime._spec("exact_recall", torch.zeros(4)).exact is False
