from pathlib import Path

import pytest

from scripts.dsa.bucket_telemetry_rpc import BucketTelemetryExtension
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_runtime import (
    RUNTIME,
    BucketSelectorConfig,
)


def _configure(telemetry: str) -> None:
    RUNTIME.configure(
        BucketSelectorConfig(
            selector="modulo_bucket_topk",
            backend="vllm_stock_per_bucket",
            bucket_count=2,
            bucket_top_k=2,
            total_k=4,
            capacity=4,
            telemetry=telemetry,
        )
    )


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    RUNTIME.reset_for_test()
    yield
    RUNTIME.reset_for_test()


def test_extension_exposes_only_enabled_bucket_runtime() -> None:
    extension = BucketTelemetryExtension()
    _configure("off")
    assert extension.bucket_telemetry_artifact() is None
    assert extension.bucket_telemetry_reset() is False

    RUNTIME.reset_for_test()
    _configure("summary")
    artifact = extension.bucket_telemetry_artifact()
    assert artifact is not None
    assert artifact["selector_identity"]["selector"] == "modulo_bucket_topk"
    assert extension.bucket_telemetry_reset() is True


def test_bucket_rpc_does_not_reference_approximate_runtime() -> None:
    source = Path(__file__).parents[3] / "scripts/dsa/bucket_telemetry_rpc.py"
    text = source.read_text()
    assert "vllm_qwen3_dsa_approx" not in text
    assert "selector_telemetry" not in text
