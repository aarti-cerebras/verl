"""Process-local configuration for isolated modulo-bucket Qwen3 DSA selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BACKENDS = ("torch_reference", "vllm_stock_per_bucket")
SELECTOR = "modulo_bucket_topk"
TELEMETRY_MODES = ("off",)


@dataclass(frozen=True)
class BucketSelectorConfig:
    selector: str
    backend: str
    bucket_count: int
    bucket_top_k: int
    total_k: int
    capacity: int
    telemetry: str = "off"

    def validate(self) -> None:
        if self.selector != SELECTOR:
            raise ValueError(f"dsa_selector={self.selector!r}; expected {SELECTOR!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"dsa_selector_backend={self.backend!r}; expected one of {BACKENDS}")
        if self.bucket_count <= 0 or self.bucket_top_k <= 0:
            raise ValueError(
                f"dsa_bucket_count and dsa_bucket_top_k must be positive; got {self.bucket_count}/{self.bucket_top_k}"
            )
        derived = self.bucket_count * self.bucket_top_k
        if derived != self.total_k:
            raise ValueError(
                "bucket budget must equal dsa_top_k: "
                f"{self.bucket_count} * {self.bucket_top_k} = {derived} != {self.total_k}"
            )
        if self.capacity != self.total_k:
            raise ValueError(
                f"index_topk must equal dsa_top_k for fixed bucket selection: {self.capacity} != {self.total_k}"
            )
        if self.telemetry not in TELEMETRY_MODES:
            raise ValueError(f"dsa_bucket_telemetry={self.telemetry!r}; expected one of {TELEMETRY_MODES}")

    def validate_execution(self, *, cudagraph_mode: str) -> None:
        self.validate()
        if cudagraph_mode == "NONE":
            return
        if self.backend != "vllm_stock_per_bucket":
            raise ValueError(
                "Qwen3 DSA bucket CUDA graphs require backend='vllm_stock_per_bucket'; "
                f"got {self.backend!r}"
            )
        # Prefill request discovery still copies cu_seqlen metadata to the host. Keep it outside
        # capture while allowing the fixed-shape decode loop to be captured and replayed.
        if cudagraph_mode != "FULL_DECODE_ONLY":
            raise ValueError(
                "Qwen3 DSA bucketed selection supports CUDA graphs only with "
                "cudagraph_mode=FULL_DECODE_ONLY; full/piecewise prefill capture remains disabled"
            )


def config_from_hf(config: Any) -> BucketSelectorConfig:
    value = BucketSelectorConfig(
        selector=str(getattr(config, "dsa_selector", SELECTOR)),
        backend=str(getattr(config, "dsa_selector_backend", "vllm_stock_per_bucket")),
        bucket_count=int(config.dsa_bucket_count),
        bucket_top_k=int(config.dsa_bucket_top_k),
        total_k=int(config.dsa_top_k),
        capacity=int(config.index_topk),
        telemetry=str(getattr(config, "dsa_bucket_telemetry", "off")),
    )
    value.validate()
    return value


class BucketSelectorRuntime:
    def __init__(self) -> None:
        self._config: BucketSelectorConfig | None = None

    @property
    def config(self) -> BucketSelectorConfig | None:
        return self._config

    @property
    def active(self) -> bool:
        return self._config is not None

    def configure(self, config: BucketSelectorConfig) -> None:
        config.validate()
        if self._config is not None and self._config != config:
            raise RuntimeError(f"bucket selector configuration changed in one worker: {self._config} -> {config}")
        self._config = config

    def reset_for_test(self) -> None:
        self._config = None


RUNTIME = BucketSelectorRuntime()
