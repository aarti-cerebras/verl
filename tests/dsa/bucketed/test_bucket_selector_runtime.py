from types import SimpleNamespace

import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_runtime import (
    BucketSelectorConfig,
    BucketSelectorRuntime,
    config_from_hf,
)
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_topk_reference import (
    BucketSelectionResult,
    select_bucket_topk_reference,
)


def _config(*, telemetry: str = "off") -> BucketSelectorConfig:
    return BucketSelectorConfig(
        selector="modulo_bucket_topk",
        backend="vllm_stock_per_bucket",
        bucket_count=2,
        bucket_top_k=2,
        total_k=4,
        capacity=4,
        telemetry=telemetry,
    )


def test_config_from_hf_fixed_budget() -> None:
    config = SimpleNamespace(
        dsa_selector="modulo_bucket_topk",
        dsa_selector_backend="vllm_stock_per_bucket",
        dsa_bucket_count=8,
        dsa_bucket_top_k=256,
        dsa_top_k=2048,
        index_topk=2048,
    )
    observed = config_from_hf(config)
    assert observed.bucket_count == 8
    assert observed.bucket_top_k == 256


def test_config_from_hf_uses_bucket_product_capacity_not_trained_top_k() -> None:
    config = SimpleNamespace(
        dsa_selector="modulo_bucket_topk",
        dsa_selector_backend="vllm_stock_per_bucket",
        dsa_bucket_count=500,
        dsa_bucket_top_k=12,
        dsa_top_k=2048,
        index_topk=6016,
    )
    observed = config_from_hf(config)
    assert observed.total_k == 6000
    assert observed.capacity == 6016


def test_config_rejects_nonminimal_capacity_padding() -> None:
    config = BucketSelectorConfig(
        selector="modulo_bucket_topk",
        backend="vllm_stock_per_bucket",
        bucket_count=500,
        bucket_top_k=12,
        total_k=6000,
        capacity=6144,
    )
    with pytest.raises(ValueError, match="smallest 128-aligned"):
        config.validate()


def test_config_rejects_budget_mismatch() -> None:
    config = BucketSelectorConfig(
        selector="modulo_bucket_topk",
        backend="vllm_stock_per_bucket",
        bucket_count=8,
        bucket_top_k=128,
        total_k=2048,
        capacity=2048,
    )
    with pytest.raises(ValueError, match="bucket budget"):
        config.validate()


def test_stock_backend_allows_decode_only_cuda_graphs() -> None:
    config = _config()
    config.validate_execution(cudagraph_mode="NONE")
    config.validate_execution(cudagraph_mode="FULL_DECODE_ONLY")


@pytest.mark.parametrize("mode", ["FULL", "PIECEWISE", "FULL_AND_PIECEWISE"])
def test_prefill_cuda_graph_modes_fail_closed(mode: str) -> None:
    config = BucketSelectorConfig(
        selector="modulo_bucket_topk",
        backend="vllm_stock_per_bucket",
        bucket_count=4,
        bucket_top_k=2,
        total_k=8,
        capacity=8,
    )
    with pytest.raises(ValueError, match="FULL_DECODE_ONLY"):
        config.validate_execution(cudagraph_mode=mode)


def test_reference_backend_fails_closed_on_cuda_graphs() -> None:
    config = BucketSelectorConfig(
        selector="modulo_bucket_topk",
        backend="torch_reference",
        bucket_count=4,
        bucket_top_k=2,
        total_k=8,
        capacity=8,
    )
    with pytest.raises(ValueError, match="vLLM stock backend"):
        config.validate_execution(cudagraph_mode="FULL_DECODE_ONLY")


@pytest.mark.parametrize("telemetry", ["summary", "verify_exact"])
def test_host_folded_telemetry_fails_closed_with_decode_graphs(telemetry: str) -> None:
    with pytest.raises(ValueError, match="host-folded"):
        _config(telemetry=telemetry).validate_execution(cudagraph_mode="FULL_DECODE_ONLY")


@pytest.mark.parametrize("telemetry", ["graph_safety", "graph_verify_exact"])
def test_graph_telemetry_allows_decode_only_graphs(telemetry: str) -> None:
    _config(telemetry=telemetry).validate_execution(cudagraph_mode="FULL_DECODE_ONLY")


def _selection_fixture() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    BucketSelectionResult,
]:
    logits = torch.tensor(
        [
            [100.0, 99.0, 98.0, 1.0, 97.0, 2.0, 96.0, 3.0],
            [1.0, 8.0, 7.0, 6.0, 5.0, -9.0, -9.0, -9.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    qpos = torch.tensor([7, 4, -1])
    output = torch.empty(3, 4, dtype=torch.int32)
    result = select_bucket_topk_reference(
        logits,
        qpos,
        output,
        bucket_count=2,
        bucket_top_k=2,
    )
    exact = torch.full_like(output, -1)
    lengths = (qpos + 1).clamp(min=0)
    valid = torch.arange(logits.shape[1])[None, :] < lengths[:, None]
    top = logits.masked_fill(~valid, float("-inf")).topk(4, dim=-1)
    keep = torch.arange(4)[None, :] < lengths.clamp(max=4)[:, None]
    exact.copy_(torch.where(keep, top.indices.to(torch.int32), -1))
    return logits, qpos, output, exact, result


def test_verify_exact_emits_bucket_identity_quality_and_bounded_views() -> None:
    runtime = BucketSelectorRuntime()
    runtime.configure(_config(telemetry="verify_exact"))
    logits, qpos, output, exact, result = _selection_fixture()
    with runtime.layer("model.layers.0.self_attn.indexer"):
        runtime.record(
            phase="decode",
            logits=logits,
            output=output,
            query_positions=qpos,
            result=result,
            exact_reference=exact,
        )

    artifact = runtime.artifact()
    assert artifact["selector_identity"]["radix_selector"] is False
    assert artifact["selector_identity"]["global_exact_topk"] is False
    assert artifact["safety"]["rows"] == 2
    assert not any(artifact["safety"]["counts"].values())
    metrics = artifact["decode"]["global"]
    assert metrics["selected_count"]["mean"] == 4
    assert metrics["recall"]["min"] < 1
    assert metrics["added"]["max"] > 0
    layer = artifact["decode"]["per_layer"]["model.layers.0.self_attn.indexer"]
    assert len(layer["per_bucket"]) == 2
    assert artifact["decode"]["query_position_bands"]
    assert artifact["decode"]["distance_bands"]


def test_host_telemetry_can_defer_until_post_warmup_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DSA_BUCKET_DEFER_HOST_TELEMETRY", "1")
    runtime = BucketSelectorRuntime()
    runtime.configure(_config(telemetry="verify_exact"))
    logits, qpos, output, exact, result = _selection_fixture()
    runtime.record(
        phase="decode",
        layer_name="layer.0",
        logits=logits,
        output=output,
        query_positions=qpos,
        result=result,
        exact_reference=exact,
    )
    before = runtime.artifact()
    assert before["recording_enabled"] is False
    assert before["safety"]["rows"] == 0
    assert "decode" not in before

    runtime.reset()
    runtime.record(
        phase="decode",
        layer_name="layer.0",
        logits=logits,
        output=output,
        query_positions=qpos,
        result=result,
        exact_reference=exact,
    )
    after = runtime.artifact()
    assert after["recording_enabled"] is True
    assert after["safety"]["rows"] == 2


def test_graph_safety_counters_are_persistent_per_layer_per_bucket_and_reset() -> None:
    runtime = BucketSelectorRuntime()
    runtime.configure(_config(telemetry="graph_safety"))
    runtime.initialize_graph_safety(["layer.0"], device=torch.device("cpu"))
    assert runtime._graph is not None
    graph_address = runtime._graph.data_ptr()
    logits, qpos, output, _, result = _selection_fixture()
    runtime.record(
        phase="decode",
        layer_name="layer.0",
        logits=logits,
        output=output,
        query_positions=qpos,
        result=result,
        exact_reference=None,
    )

    replay = runtime.artifact()["graph_replay"]
    assert replay["global"]["rows"] == 2
    assert replay["global"]["calls"] == 1
    assert replay["global"]["selected_total"] == 8
    layer = replay["phases"]["decode"]["per_layer"]["layer.0"]
    assert [bucket["selected_total"] for bucket in layer["per_bucket"]] == [4, 4]
    assert not any(
        layer[name]
        for name in (
            "count_mismatch",
            "padding_violation",
            "prefix_violation",
            "invalid_index",
            "noncausal_index",
            "duplicate_index",
        )
    )

    runtime.reset()
    assert runtime._graph.data_ptr() == graph_address
    assert runtime.artifact()["graph_replay"]["global"]["rows"] == 0


def test_graph_verify_exact_matches_eager_decode_moments_and_resets_in_place() -> None:
    logits, qpos, output, exact, result = _selection_fixture()
    eager = BucketSelectorRuntime()
    eager.configure(_config(telemetry="verify_exact"))
    eager.record(
        phase="decode",
        layer_name="layer.0",
        logits=logits,
        output=output,
        query_positions=qpos,
        result=result,
        exact_reference=exact,
    )

    graph = BucketSelectorRuntime()
    graph.configure(_config(telemetry="graph_verify_exact"))
    graph.initialize_graph_quality(["layer.0"], device=torch.device("cpu"))
    assert graph._graph_quality is not None
    quality_address = graph._graph_quality.data_ptr()
    graph.record(
        phase="decode",
        layer_name="layer.0",
        logits=logits,
        output=output,
        query_positions=qpos,
        result=result,
        exact_reference=exact,
    )

    eager_metrics = eager.artifact()["decode"]["global"]
    artifact = graph.artifact()
    graph_metrics = artifact["graph_replay_quality"]["phases"]["decode"]["global"]
    for field in (
        "selected_count",
        "intersection",
        "added",
        "dropped",
        "recall",
        "precision",
        "jaccard",
        "selected_score_mass",
        "global_exact_score_mass",
        "score_mass_gap",
        "score_mass_ratio",
    ):
        assert graph_metrics[field] == eager_metrics[field]
    per_bucket = artifact["graph_replay_quality"]["phases"]["decode"]["per_layer"][
        "layer.0"
    ]["per_bucket"]
    assert len(per_bucket) == 2
    assert artifact["graph_replay_quality"]["phases"]["decode"][
        "query_position_bands"
    ]
    assert artifact["graph_replay_quality"]["phases"]["decode"]["distance_bands"]

    graph.reset()
    assert graph._graph_quality.data_ptr() == quality_address
    reset_metrics = graph.artifact()["graph_replay_quality"]["phases"]["decode"][
        "global"
    ]
    assert all(summary["n"] == 0 for summary in reset_metrics.values())


def test_graph_verify_exact_keeps_prefill_host_folded() -> None:
    runtime = BucketSelectorRuntime()
    runtime.configure(_config(telemetry="graph_verify_exact"))
    runtime.initialize_graph_quality(["layer.0"], device=torch.device("cpu"))
    logits, qpos, output, exact, result = _selection_fixture()
    runtime.record(
        phase="prefill",
        layer_name="layer.0",
        logits=logits,
        output=output,
        query_positions=qpos,
        result=result,
        exact_reference=exact,
    )

    artifact = runtime.artifact()
    assert artifact["prefill"]["global"]["recall"]["n"] == 2
    graph_prefill = artifact["graph_replay_quality"]["phases"]["prefill"]["global"]
    assert all(summary["n"] == 0 for summary in graph_prefill.values())
