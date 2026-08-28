from types import SimpleNamespace

import pytest

from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_runtime import (
    BucketSelectorConfig,
    config_from_hf,
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
    config = BucketSelectorConfig(
        selector="modulo_bucket_topk",
        backend="vllm_stock_per_bucket",
        bucket_count=4,
        bucket_top_k=2,
        total_k=8,
        capacity=8,
    )
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
    with pytest.raises(ValueError, match="vllm_stock_per_bucket"):
        config.validate_execution(cudagraph_mode="FULL_DECODE_ONLY")
