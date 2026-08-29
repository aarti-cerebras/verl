"""Process-local configuration and telemetry for isolated modulo-bucket selection."""

from __future__ import annotations

import contextlib
import math
import os
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import torch

from .bucket_topk_reference import BucketSelectionResult

BACKENDS = ("torch_reference", "vllm_stock_per_bucket", "vllm_stock_batched_buckets")
SELECTOR = "modulo_bucket_topk"
TELEMETRY_MODES = ("off", "summary", "verify_exact", "graph_safety", "graph_verify_exact")
PHASES = ("prefill", "decode")
PHASE_SLOT = {name: index for index, name in enumerate(PHASES)}
GRAPH_FIELDS = (
    "rows",
    "calls",
    "selected_total",
    "effective_k_total",
    "underfilled_buckets",
    "empty_buckets",
    "count_mismatch",
    "padding_violation",
    "prefix_violation",
    "invalid_index",
    "noncausal_index",
    "duplicate_index",
)
GRAPH_FIELD_SLOT = {name: index for index, name in enumerate(GRAPH_FIELDS)}
GRAPH_BUCKET_FIELDS = ("selected_total", "underfilled_rows", "empty_rows")
GRAPH_BUCKET_FIELD_SLOT = {name: index for index, name in enumerate(GRAPH_BUCKET_FIELDS)}
GRAPH_QUALITY_FIELDS = (
    "query_position",
    "selected_count",
    "effective_k",
    "underfilled_buckets",
    "empty_buckets",
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
)
GRAPH_BUCKET_QUALITY_FIELDS = (
    "selected_count",
    "underfilled",
    "empty",
    "intersection",
    "added",
    "dropped",
)
GRAPH_DISTANCE_QUALITY_FIELDS = (
    "selected_count",
    "global_exact_count",
    "intersection",
    "recall",
)
GRAPH_STAT_FIELDS = ("n", "total", "total_sq", "min", "max")
GRAPH_STAT_SLOT = {name: index for index, name in enumerate(GRAPH_STAT_FIELDS)}
QUERY_POSITION_BANDS = (
    (0, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, None),
)
DISTANCE_BANDS = (
    (0, 16),
    (16, 64),
    (64, 256),
    (256, 1024),
    (1024, 4096),
    (4096, 16384),
    (16384, None),
)


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
                "bucket budget must equal selector total_k: "
                f"{self.bucket_count} * {self.bucket_top_k} = {derived} != {self.total_k}"
            )
        if self.capacity < self.total_k:
            raise ValueError(
                f"index_topk capacity cannot hold bucket total_k: {self.capacity} < {self.total_k}"
            )
        if self.capacity != self.total_k and (
            self.capacity % 128 != 0 or self.capacity - self.total_k >= 128
        ):
            raise ValueError(
                f"padded index_topk must be the smallest 128-aligned capacity above bucket "
                f"total_k: capacity={self.capacity}, total_k={self.total_k}"
            )
        if self.telemetry not in TELEMETRY_MODES:
            raise ValueError(f"dsa_bucket_telemetry={self.telemetry!r}; expected one of {TELEMETRY_MODES}")

    def validate_execution(self, *, cudagraph_mode: str) -> None:
        self.validate()
        if cudagraph_mode == "NONE":
            return
        if self.backend not in ("vllm_stock_per_bucket", "vllm_stock_batched_buckets"):
            raise ValueError(
                "Qwen3 DSA bucket CUDA graphs require a vLLM stock backend; "
                f"got {self.backend!r}"
            )
        if cudagraph_mode != "FULL_DECODE_ONLY":
            raise ValueError(
                "Qwen3 DSA bucketed selection supports CUDA graphs only with "
                "cudagraph_mode=FULL_DECODE_ONLY; full/piecewise prefill capture remains disabled"
            )
        if self.telemetry in ("summary", "verify_exact"):
            raise ValueError(
                "FULL_DECODE_ONLY with bucket telemetry requires dsa_bucket_telemetry='off' "
                "or a graph telemetry mode; summary/verify_exact are host-folded"
            )


def config_from_hf(config: Any) -> BucketSelectorConfig:
    capacity = int(config.index_topk)
    bucket_count = int(config.dsa_bucket_count)
    bucket_top_k = int(config.dsa_bucket_top_k)
    value = BucketSelectorConfig(
        selector=str(getattr(config, "dsa_selector", SELECTOR)),
        backend=str(getattr(config, "dsa_selector_backend", "vllm_stock_per_bucket")),
        bucket_count=bucket_count,
        bucket_top_k=bucket_top_k,
        total_k=bucket_count * bucket_top_k,
        capacity=capacity,
        telemetry=str(getattr(config, "dsa_bucket_telemetry", "off")),
    )
    value.validate()
    return value


class _Summary:
    """Fixed-memory exact moments; no raw token records are retained."""

    __slots__ = ("n", "total", "total_sq", "minimum", "maximum")

    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, values: torch.Tensor) -> None:
        flat = values.detach().to("cpu", torch.float64).flatten()
        flat = flat[torch.isfinite(flat)]
        if not flat.numel():
            return
        self.n += int(flat.numel())
        self.total += float(flat.sum())
        self.total_sq += float(flat.square().sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))

    def merge(self, other: _Summary) -> None:
        if not other.n:
            return
        self.n += other.n
        self.total += other.total
        self.total_sq += other.total_sq
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    def artifact(self) -> dict[str, float | int]:
        if not self.n:
            return {"n": 0}
        mean = self.total / self.n
        variance = max(self.total_sq / self.n - mean * mean, 0.0)
        return {
            "n": self.n,
            "total": self.total,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def _band_label(low: int, high: int | None) -> str:
    return f"{low}_{high if high is not None else 'plus'}"


def _membership(needles: torch.Tensor, haystack: torch.Tensor, sentinel: int) -> torch.Tensor:
    needle_valid = needles >= 0
    haystack_valid = haystack >= 0
    ordered = torch.where(haystack_valid, haystack.long(), sentinel).sort(-1).values
    safe = needles.long().clamp(min=0, max=sentinel)
    offsets = torch.searchsorted(ordered, safe).clamp(max=max(ordered.shape[1] - 1, 0))
    return needle_valid & (ordered.gather(-1, offsets) == safe)


class BucketSelectorRuntime:
    """Owns bucket-only eager summaries and persistent graph-replay counters."""

    def __init__(self) -> None:
        self._config: BucketSelectorConfig | None = None
        self._layer = "unattributed"
        self._lock = threading.Lock()
        self._dists: dict[tuple[str, str], dict[str, _Summary]] = defaultdict(dict)
        self._position_dists: dict[tuple[str, str], dict[str, _Summary]] = defaultdict(dict)
        self._bucket_dists: dict[tuple[str, str, int], dict[str, _Summary]] = defaultdict(dict)
        self._distance_dists: dict[tuple[str, str], dict[str, _Summary]] = defaultdict(dict)
        self._calls: dict[tuple[str, str], int] = defaultdict(int)
        self._safety_rows = 0
        self._safety: dict[str, int] = defaultdict(int)
        self._host_recording_enabled = True
        self._graph: torch.Tensor | None = None
        self._graph_buckets: torch.Tensor | None = None
        self._graph_layer_slots: dict[str, int] = {}
        self._graph_quality: torch.Tensor | None = None
        self._graph_position_quality: torch.Tensor | None = None
        self._graph_bucket_quality: torch.Tensor | None = None
        self._graph_distance_quality: torch.Tensor | None = None

    @property
    def config(self) -> BucketSelectorConfig | None:
        return self._config

    @property
    def active(self) -> bool:
        return self._config is not None

    @property
    def telemetry_active(self) -> bool:
        return self._config is not None and self._config.telemetry != "off"

    @property
    def current_layer(self) -> str:
        return self._layer

    def configure(self, config: BucketSelectorConfig) -> None:
        config.validate()
        with self._lock:
            if self._config is not None and self._config != config:
                raise RuntimeError(f"bucket selector configuration changed in one worker: {self._config} -> {config}")
            self._config = config
            if config.telemetry in ("summary", "verify_exact", "graph_verify_exact"):
                self._host_recording_enabled = os.environ.get(
                    "DSA_BUCKET_DEFER_HOST_TELEMETRY", "0"
                ) in ("0", "", "false", "False")

    @contextlib.contextmanager
    def layer(self, name: str) -> Iterator[None]:
        previous = self._layer
        self._layer = name
        try:
            yield
        finally:
            self._layer = previous

    def initialize_graph_safety(self, layer_names: list[str], *, device: torch.device) -> None:
        config = self._config
        if config is None or config.telemetry not in ("graph_safety", "graph_verify_exact"):
            raise RuntimeError(
                "graph storage requires dsa_bucket_telemetry='graph_safety' or "
                "'graph_verify_exact'"
            )
        if not layer_names or len(set(layer_names)) != len(layer_names):
            raise ValueError(f"graph_safety requires unique sparse layer names, got {layer_names!r}")
        slots = {name: index for index, name in enumerate(layer_names)}
        shape = (len(PHASES), len(layer_names), len(GRAPH_FIELDS))
        bucket_shape = (len(PHASES), len(layer_names), config.bucket_count, len(GRAPH_BUCKET_FIELDS))
        with self._lock:
            if self._graph is not None:
                if self._graph_layer_slots != slots or tuple(self._graph.shape) != shape:
                    raise RuntimeError("bucket graph telemetry layout changed after allocation")
                return
            self._graph_layer_slots = slots
            self._graph = torch.zeros(shape, dtype=torch.int64, device=device)
            self._graph_buckets = torch.zeros(bucket_shape, dtype=torch.int64, device=device)

    def initialize_graph_quality(self, layer_names: list[str], *, device: torch.device) -> None:
        """Allocate fixed-address exact-quality moments before decode graph capture."""

        self.initialize_graph_safety(layer_names, device=device)
        config = self._config
        if config is None or config.telemetry != "graph_verify_exact":
            raise RuntimeError(
                "graph quality storage requires dsa_bucket_telemetry='graph_verify_exact'"
            )
        quality_shape = (
            len(PHASES),
            len(layer_names),
            len(GRAPH_QUALITY_FIELDS),
            len(GRAPH_STAT_FIELDS),
        )
        position_shape = (
            len(PHASES),
            len(QUERY_POSITION_BANDS),
            len(GRAPH_QUALITY_FIELDS),
            len(GRAPH_STAT_FIELDS),
        )
        bucket_shape = (
            len(PHASES),
            len(layer_names),
            config.bucket_count,
            len(GRAPH_BUCKET_QUALITY_FIELDS),
            len(GRAPH_STAT_FIELDS),
        )
        distance_shape = (
            len(PHASES),
            len(DISTANCE_BANDS),
            len(GRAPH_DISTANCE_QUALITY_FIELDS),
            len(GRAPH_STAT_FIELDS),
        )
        with self._lock:
            if self._graph_quality is not None:
                observed = (
                    tuple(self._graph_quality.shape),
                    tuple(self._graph_position_quality.shape),
                    tuple(self._graph_bucket_quality.shape),
                    tuple(self._graph_distance_quality.shape),
                )
                expected = (quality_shape, position_shape, bucket_shape, distance_shape)
                if observed != expected:
                    raise RuntimeError("bucket graph quality layout changed after allocation")
                return
            self._graph_quality = torch.empty(quality_shape, dtype=torch.float64, device=device)
            self._graph_position_quality = torch.empty(
                position_shape, dtype=torch.float64, device=device
            )
            self._graph_bucket_quality = torch.empty(
                bucket_shape, dtype=torch.float64, device=device
            )
            self._graph_distance_quality = torch.empty(
                distance_shape, dtype=torch.float64, device=device
            )
            self._reset_graph_quality_storage()

    def _reset_graph_quality_storage(self) -> None:
        for stats in (
            self._graph_quality,
            self._graph_position_quality,
            self._graph_bucket_quality,
            self._graph_distance_quality,
        ):
            if stats is None:
                continue
            stats.zero_()
            stats[..., GRAPH_STAT_SLOT["min"]].fill_(float("inf"))
            stats[..., GRAPH_STAT_SLOT["max"]].fill_(float("-inf"))

    @staticmethod
    def _safety_tensors(
        output: torch.Tensor,
        query_positions: torch.Tensor,
        result: BucketSelectionResult,
        key_count: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        qpos = query_positions.reshape(-1).to(output.device, torch.int64)
        active = qpos >= 0
        valid = output >= 0
        observed = valid.sum(-1)
        sentinel = torch.full_like(output, key_count)
        ordered = torch.where(valid, output, sentinel).sort(-1).values
        duplicates = ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != key_count)).any(-1)
        flags = {
            "count_mismatch": active & (observed != result.selected_count),
            "padding_violation": (~active) & valid.any(-1),
            "prefix_violation": ((~valid[:, :-1]) & valid[:, 1:]).any(-1),
            "invalid_index": ((output < -1) | (output >= key_count)).any(-1),
            "noncausal_index": (valid & (output > qpos[:, None])).any(-1),
            "duplicate_index": duplicates,
        }
        return active, flags, observed

    def _record_graph(
        self,
        *,
        phase: str,
        layer_name: str,
        output: torch.Tensor,
        query_positions: torch.Tensor,
        result: BucketSelectionResult,
        key_count: int,
    ) -> None:
        if self._graph is None or self._graph_buckets is None:
            raise RuntimeError("graph_safety telemetry storage was not initialized before execution")
        try:
            phase_slot = PHASE_SLOT[phase]
            layer_slot = self._graph_layer_slots[layer_name]
        except KeyError as exc:
            raise RuntimeError(f"no graph telemetry slot for phase={phase!r}, layer={layer_name!r}") from exc
        active, flags, observed = self._safety_tensors(output, query_positions, result, key_count)
        target = self._graph[phase_slot, layer_slot]
        target[GRAPH_FIELD_SLOT["rows"]].add_(active.to(torch.int64).sum())
        target[GRAPH_FIELD_SLOT["calls"]].add_(1)
        target[GRAPH_FIELD_SLOT["selected_total"]].add_((observed * active).sum())
        target[GRAPH_FIELD_SLOT["effective_k_total"]].add_((result.effective_k * active).sum())
        assert self._config is not None
        underfilled = (result.bucket_counts < self._config.bucket_top_k) & active[:, None]
        empty = (result.bucket_counts == 0) & active[:, None]
        target[GRAPH_FIELD_SLOT["underfilled_buckets"]].add_(underfilled.sum())
        target[GRAPH_FIELD_SLOT["empty_buckets"]].add_(empty.sum())
        for name, values in flags.items():
            scoped = values if name == "padding_violation" else values & active
            target[GRAPH_FIELD_SLOT[name]].add_(scoped.to(torch.int64).sum())
        bucket_target = self._graph_buckets[phase_slot, layer_slot]
        bucket_target[:, GRAPH_BUCKET_FIELD_SLOT["selected_total"]].add_(
            (result.bucket_counts * active[:, None]).sum(0)
        )
        bucket_target[:, GRAPH_BUCKET_FIELD_SLOT["underfilled_rows"]].add_(underfilled.sum(0))
        bucket_target[:, GRAPH_BUCKET_FIELD_SLOT["empty_rows"]].add_(empty.sum(0))

    @staticmethod
    def _graph_moments_add(
        target: torch.Tensor,
        values: torch.Tensor,
        include: torch.Tensor,
    ) -> None:
        """Accumulate fixed-shape moments without leaving the captured device graph."""

        values = values.to(torch.float64)
        finite = torch.isfinite(values) & include[..., None]
        finite_float = finite.to(torch.float64)
        safe = torch.where(finite, values, torch.zeros_like(values))
        target[..., GRAPH_STAT_SLOT["n"]].add_(finite_float.sum(0))
        target[..., GRAPH_STAT_SLOT["total"]].add_(safe.sum(0))
        target[..., GRAPH_STAT_SLOT["total_sq"]].add_(safe.square().sum(0))
        minimum = torch.where(
            finite, values, torch.full_like(values, float("inf"))
        ).amin(0)
        maximum = torch.where(
            finite, values, torch.full_like(values, float("-inf"))
        ).amax(0)
        target_min = target[..., GRAPH_STAT_SLOT["min"]]
        target_max = target[..., GRAPH_STAT_SLOT["max"]]
        target_min.copy_(torch.minimum(target_min, minimum))
        target_max.copy_(torch.maximum(target_max, maximum))

    def _record_graph_quality(
        self,
        *,
        phase: str,
        layer_name: str,
        logits: torch.Tensor,
        output: torch.Tensor,
        query_positions: torch.Tensor,
        result: BucketSelectionResult,
        exact_reference: torch.Tensor,
    ) -> None:
        """Fold exact overlap and score quality into persistent tensors on replay."""

        storage = (
            self._graph_quality,
            self._graph_position_quality,
            self._graph_bucket_quality,
            self._graph_distance_quality,
        )
        if any(value is None for value in storage):
            raise RuntimeError(
                "dsa_bucket_telemetry='graph_verify_exact' is active but persistent quality "
                "storage was not initialized before execution"
            )
        quality, position_quality, bucket_quality, distance_quality = storage
        assert quality is not None
        assert position_quality is not None
        assert bucket_quality is not None
        assert distance_quality is not None
        config = self._config
        assert config is not None
        try:
            phase_slot = PHASE_SLOT[phase]
            layer_slot = self._graph_layer_slots[layer_name]
        except KeyError as exc:
            raise RuntimeError(
                f"no graph quality slot for phase={phase!r}, layer={layer_name!r}"
            ) from exc

        key_count = logits.shape[1]
        active, _, observed = self._safety_tensors(
            output, query_positions, result, key_count
        )
        qpos = query_positions.reshape(-1).to(output.device, torch.int64)
        bucket_in_exact = _membership(output, exact_reference, key_count)
        exact_in_bucket = _membership(exact_reference, output, key_count)
        exact_valid = exact_reference >= 0
        exact_count = exact_valid.sum(-1)
        intersection = bucket_in_exact.sum(-1)
        union = observed + exact_count - intersection
        underfilled = (result.bucket_counts < config.bucket_top_k).sum(-1)
        empty = (result.bucket_counts == 0).sum(-1)
        safe_bucket = output.clamp(min=0, max=max(key_count - 1, 0)).long()
        safe_exact = exact_reference.clamp(min=0, max=max(key_count - 1, 0)).long()
        bucket_mass = (logits.gather(-1, safe_bucket).float() * (output >= 0)).sum(-1)
        exact_mass = (logits.gather(-1, safe_exact).float() * exact_valid).sum(-1)
        ratio = torch.where(
            exact_mass.abs() > 1e-12,
            bucket_mass / exact_mass,
            torch.full_like(exact_mass, float("nan")),
        )
        metrics = torch.stack(
            (
                qpos,
                observed,
                result.effective_k,
                underfilled,
                empty,
                intersection,
                observed - intersection,
                exact_count - intersection,
                intersection.float() / exact_count.clamp(min=1),
                intersection.float() / observed.clamp(min=1),
                intersection.float() / union.clamp(min=1),
                bucket_mass,
                exact_mass,
                exact_mass - bucket_mass,
                ratio,
            ),
            dim=-1,
        )
        self._graph_moments_add(quality[phase_slot, layer_slot], metrics, active)

        position_masks = []
        for low, high in QUERY_POSITION_BANDS:
            mask = active & (qpos >= low)
            if high is not None:
                mask &= qpos < high
            position_masks.append(mask)
        position_include = torch.stack(position_masks, dim=-1)
        position_values = metrics[:, None, :].expand(
            -1, len(QUERY_POSITION_BANDS), -1
        )
        self._graph_moments_add(
            position_quality[phase_slot], position_values, position_include
        )

        bucket_selected = result.bucket_counts
        # Accumulate by modulo id without materializing [rows, total_k, bucket_count] one-hot
        # tensors. At 500x12 and vLLM's largest decode capture shape those broadcasts require more
        # than 11 GiB apiece; scatter keeps the temporary footprint proportional to
        # [rows, total_k] plus the small [rows, bucket_count] result.
        output_bucket_ids = output.remainder(config.bucket_count).long()
        exact_bucket_ids = exact_reference.remainder(config.bucket_count).long()
        bucket_intersection = torch.zeros_like(bucket_selected).scatter_add(
            1,
            output_bucket_ids,
            (bucket_in_exact & (output >= 0)).to(bucket_selected.dtype),
        )
        bucket_exact = torch.zeros_like(bucket_selected).scatter_add(
            1,
            exact_bucket_ids,
            exact_valid.to(bucket_selected.dtype),
        )
        bucket_metrics = torch.stack(
            (
                bucket_selected,
                bucket_selected < config.bucket_top_k,
                bucket_selected == 0,
                bucket_intersection,
                bucket_selected - bucket_intersection,
                bucket_exact - bucket_intersection,
            ),
            dim=-1,
        )
        self._graph_moments_add(
            bucket_quality[phase_slot, layer_slot],
            bucket_metrics,
            active[:, None].expand(-1, config.bucket_count),
        )

        selected_distance = qpos[:, None] - output
        exact_distance = qpos[:, None] - exact_reference
        distance_metrics = []
        for low, high in DISTANCE_BANDS:
            selected_band = (output >= 0) & (selected_distance >= low)
            exact_band = exact_valid & (exact_distance >= low)
            if high is not None:
                selected_band &= selected_distance < high
                exact_band &= exact_distance < high
            selected_band_count = selected_band.sum(-1)
            exact_band_count = exact_band.sum(-1)
            overlap = (exact_in_bucket & exact_band).sum(-1)
            distance_metrics.append(
                torch.stack(
                    (
                        selected_band_count,
                        exact_band_count,
                        overlap,
                        torch.where(
                            exact_band_count > 0,
                            overlap.float() / exact_band_count.clamp(min=1),
                            torch.full_like(overlap.float(), float("nan")),
                        ),
                    ),
                    dim=-1,
                )
            )
        distance_values = torch.stack(distance_metrics, dim=1)
        self._graph_moments_add(
            distance_quality[phase_slot],
            distance_values,
            active[:, None].expand(-1, len(DISTANCE_BANDS)),
        )

    @staticmethod
    def _add(group: dict[str, _Summary], name: str, values: torch.Tensor) -> None:
        summary = group.get(name)
        if summary is None:
            summary = group[name] = _Summary()
        summary.add(values)

    def _record_host(
        self,
        *,
        phase: str,
        layer_name: str,
        logits: torch.Tensor,
        output: torch.Tensor,
        query_positions: torch.Tensor,
        result: BucketSelectionResult,
        exact_reference: torch.Tensor | None,
    ) -> None:
        config = self._config
        assert config is not None
        key_count = logits.shape[1]
        active, flags, observed = self._safety_tensors(output, query_positions, result, key_count)
        qpos = query_positions.reshape(-1).to(output.device, torch.int64)
        underfilled = (result.bucket_counts < config.bucket_top_k).sum(-1)
        empty = (result.bucket_counts == 0).sum(-1)
        metrics: dict[str, torch.Tensor] = {
            "query_position": qpos,
            "selected_count": observed,
            "effective_k": result.effective_k,
            "underfilled_buckets": underfilled,
            "empty_buckets": empty,
        }
        bucket_in_exact = exact_in_bucket = None
        if exact_reference is not None:
            bucket_in_exact = _membership(output, exact_reference, key_count)
            exact_in_bucket = _membership(exact_reference, output, key_count)
            intersection = bucket_in_exact.sum(-1)
            exact_count = (exact_reference >= 0).sum(-1)
            union = observed + exact_count - intersection
            metrics.update(
                {
                    "intersection": intersection,
                    "added": observed - intersection,
                    "dropped": exact_count - intersection,
                    "recall": intersection.float() / exact_count.clamp(min=1),
                    "precision": intersection.float() / observed.clamp(min=1),
                    "jaccard": intersection.float() / union.clamp(min=1),
                }
            )
            safe_bucket = output.clamp(min=0, max=max(key_count - 1, 0)).long()
            safe_exact = exact_reference.clamp(min=0, max=max(key_count - 1, 0)).long()
            bucket_scores = logits.gather(-1, safe_bucket).float() * (output >= 0)
            exact_scores = logits.gather(-1, safe_exact).float() * (exact_reference >= 0)
            bucket_mass = bucket_scores.sum(-1)
            exact_mass = exact_scores.sum(-1)
            metrics["selected_score_mass"] = bucket_mass
            metrics["global_exact_score_mass"] = exact_mass
            metrics["score_mass_gap"] = exact_mass - bucket_mass
            metrics["score_mass_ratio"] = torch.where(
                exact_mass.abs() > 1e-12,
                bucket_mass / exact_mass,
                torch.full_like(exact_mass, float("nan")),
            )

        active_metrics = {name: values[active] for name, values in metrics.items()}
        with self._lock:
            self._calls[(phase, layer_name)] += 1
            self._safety_rows += int(active.sum().to("cpu"))
            for name, values in flags.items():
                scoped = values if name == "padding_violation" else values & active
                self._safety[name] += int(scoped.sum().to("cpu"))
            group = self._dists[(phase, layer_name)]
            for name, values in active_metrics.items():
                self._add(group, name, values)

            qpos_host = qpos.detach().to("cpu")
            active_host = active.detach().to("cpu")
            for low, high in QUERY_POSITION_BANDS:
                mask = active_host & (qpos_host >= low)
                if high is not None:
                    mask &= qpos_host < high
                if not mask.any():
                    continue
                band_group = self._position_dists[(phase, _band_label(low, high))]
                active_band = mask[active_host]
                for name, values in active_metrics.items():
                    self._add(band_group, name, values.detach().to("cpu")[active_band])

            for bucket in range(config.bucket_count):
                bucket_group = self._bucket_dists[(phase, layer_name, bucket)]
                chosen = result.bucket_counts[:, bucket]
                self._add(bucket_group, "selected_count", chosen[active])
                self._add(bucket_group, "underfilled", (chosen < config.bucket_top_k)[active])
                self._add(bucket_group, "empty", (chosen == 0)[active])
                if bucket_in_exact is not None and exact_reference is not None:
                    output_bucket = (output.remainder(config.bucket_count) == bucket) & (output >= 0)
                    exact_bucket = (exact_reference.remainder(config.bucket_count) == bucket) & (exact_reference >= 0)
                    overlap = (bucket_in_exact & output_bucket).sum(-1)
                    self._add(bucket_group, "intersection", overlap[active])
                    self._add(bucket_group, "added", (output_bucket.sum(-1) - overlap)[active])
                    self._add(bucket_group, "dropped", (exact_bucket.sum(-1) - overlap)[active])

            selected_distance = qpos[:, None] - output
            exact_distance = qpos[:, None] - exact_reference if exact_reference is not None else None
            for low, high in DISTANCE_BANDS:
                selected_band = (output >= 0) & (selected_distance >= low)
                if high is not None:
                    selected_band &= selected_distance < high
                distance_group = self._distance_dists[(phase, _band_label(low, high))]
                self._add(distance_group, "selected_count", selected_band.sum(-1)[active])
                if exact_distance is not None and exact_reference is not None and exact_in_bucket is not None:
                    exact_band = (exact_reference >= 0) & (exact_distance >= low)
                    if high is not None:
                        exact_band &= exact_distance < high
                    overlap = (exact_in_bucket & exact_band).sum(-1)
                    exact_band_count = exact_band.sum(-1)
                    self._add(distance_group, "global_exact_count", exact_band_count[active])
                    self._add(distance_group, "intersection", overlap[active])
                    has_exact = active & (exact_band_count > 0)
                    self._add(
                        distance_group,
                        "recall",
                        (overlap.float() / exact_band_count.clamp(min=1))[has_exact],
                    )

    def record(
        self,
        *,
        phase: str,
        logits: torch.Tensor,
        output: torch.Tensor,
        query_positions: torch.Tensor,
        result: BucketSelectionResult,
        exact_reference: torch.Tensor | None,
        layer_name: str | None = None,
    ) -> None:
        config = self._config
        if config is None or config.telemetry == "off":
            return
        host_folded = config.telemetry in ("summary", "verify_exact") or (
            config.telemetry == "graph_verify_exact" and phase != "decode"
        )
        if host_folded and not self._host_recording_enabled:
            return
        if phase not in PHASES:
            raise ValueError(f"unknown bucket telemetry phase {phase!r}")
        layer = layer_name or self._layer
        if config.telemetry in ("graph_safety", "graph_verify_exact") and (
            config.telemetry == "graph_safety" or phase == "decode"
        ):
            self._record_graph(
                phase=phase,
                layer_name=layer,
                output=output,
                query_positions=query_positions,
                result=result,
                key_count=logits.shape[1],
            )
            if config.telemetry == "graph_verify_exact":
                if exact_reference is None:
                    raise RuntimeError(
                        "graph_verify_exact bucket telemetry requires a global exact top-k reference"
                    )
                self._record_graph_quality(
                    phase=phase,
                    layer_name=layer,
                    logits=logits,
                    output=output,
                    query_positions=query_positions,
                    result=result,
                    exact_reference=exact_reference,
                )
            return
        if config.telemetry in ("verify_exact", "graph_verify_exact") and exact_reference is None:
            raise RuntimeError(
                f"{config.telemetry} bucket telemetry requires a global exact top-k reference"
            )
        self._record_host(
            phase=phase,
            layer_name=layer,
            logits=logits,
            output=output,
            query_positions=query_positions,
            result=result,
            exact_reference=exact_reference,
        )

    @staticmethod
    def _summaries(groups: list[dict[str, _Summary]]) -> dict[str, dict[str, float | int]]:
        merged: dict[str, _Summary] = {}
        for group in groups:
            for name, summary in group.items():
                target = merged.setdefault(name, _Summary())
                target.merge(summary)
        return {name: summary.artifact() for name, summary in sorted(merged.items())}

    def _graph_artifact(self) -> dict[str, Any]:
        if self._graph is None or self._graph_buckets is None:
            return {"initialized": False}
        graph = self._graph.detach().to("cpu")
        buckets = self._graph_buckets.detach().to("cpu")
        layers = [name for name, _ in sorted(self._graph_layer_slots.items(), key=lambda item: item[1])]
        phases: dict[str, Any] = {}
        for phase, phase_slot in PHASE_SLOT.items():
            per_layer = {}
            for layer_slot, layer in enumerate(layers):
                values = graph[phase_slot, layer_slot]
                per_layer[layer] = {name: int(values[slot]) for name, slot in GRAPH_FIELD_SLOT.items()}
                per_layer[layer]["per_bucket"] = [
                    {
                        name: int(buckets[phase_slot, layer_slot, bucket, slot])
                        for name, slot in GRAPH_BUCKET_FIELD_SLOT.items()
                    }
                    for bucket in range(buckets.shape[2])
                ]
            phases[phase] = {"per_layer": per_layer}
        global_values = graph.sum((0, 1))
        return {
            "initialized": True,
            "storage": "persistent_device_counters",
            "global": {name: int(global_values[slot]) for name, slot in GRAPH_FIELD_SLOT.items()},
            "phases": phases,
        }

    @staticmethod
    def _stats_artifact(stats: torch.Tensor) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for field_slot, field in enumerate(GRAPH_QUALITY_FIELDS):
            values = stats[field_slot]
            count = int(values[GRAPH_STAT_SLOT["n"]])
            if not count:
                result[field] = {"n": 0}
                continue
            total = float(values[GRAPH_STAT_SLOT["total"]])
            total_sq = float(values[GRAPH_STAT_SLOT["total_sq"]])
            mean = total / count
            result[field] = {
                "n": count,
                "total": total,
                "mean": mean,
                "std": math.sqrt(max(total_sq / count - mean * mean, 0.0)),
                "min": float(values[GRAPH_STAT_SLOT["min"]]),
                "max": float(values[GRAPH_STAT_SLOT["max"]]),
            }
        return result

    @staticmethod
    def _named_stats_artifact(
        stats: torch.Tensor, fields: tuple[str, ...]
    ) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        for field_slot, field in enumerate(fields):
            values = stats[field_slot]
            count = int(values[GRAPH_STAT_SLOT["n"]])
            if not count:
                result[field] = {"n": 0}
                continue
            total = float(values[GRAPH_STAT_SLOT["total"]])
            total_sq = float(values[GRAPH_STAT_SLOT["total_sq"]])
            mean = total / count
            result[field] = {
                "n": count,
                "total": total,
                "mean": mean,
                "std": math.sqrt(max(total_sq / count - mean * mean, 0.0)),
                "min": float(values[GRAPH_STAT_SLOT["min"]]),
                "max": float(values[GRAPH_STAT_SLOT["max"]]),
            }
        return result

    @staticmethod
    def _merge_stats(stats: torch.Tensor, dimensions: tuple[int, ...]) -> torch.Tensor:
        merged = stats.sum(dimensions)
        merged[..., GRAPH_STAT_SLOT["min"]] = stats[..., GRAPH_STAT_SLOT["min"]].amin(
            dimensions
        )
        merged[..., GRAPH_STAT_SLOT["max"]] = stats[..., GRAPH_STAT_SLOT["max"]].amax(
            dimensions
        )
        return merged

    def _graph_quality_artifact(self) -> dict[str, Any]:
        storage = (
            self._graph_quality,
            self._graph_position_quality,
            self._graph_bucket_quality,
            self._graph_distance_quality,
        )
        if any(value is None for value in storage):
            return {"initialized": False}
        quality, position_quality, bucket_quality, distance_quality = (
            value.detach().to("cpu") for value in storage if value is not None
        )
        layers = [
            name
            for name, _ in sorted(
                self._graph_layer_slots.items(), key=lambda item: item[1]
            )
        ]
        phases: dict[str, Any] = {}
        for phase, phase_slot in PHASE_SLOT.items():
            phase_quality = quality[phase_slot]
            per_layer = {}
            for layer_slot, layer in enumerate(layers):
                per_layer[layer] = {
                    "metrics": self._stats_artifact(phase_quality[layer_slot]),
                    "per_bucket": [
                        self._named_stats_artifact(
                            bucket_quality[phase_slot, layer_slot, bucket],
                            GRAPH_BUCKET_QUALITY_FIELDS,
                        )
                        for bucket in range(bucket_quality.shape[2])
                    ],
                }
            phases[phase] = {
                "global": self._stats_artifact(
                    self._merge_stats(phase_quality, (0,))
                ),
                "per_layer": per_layer,
                "query_position_bands": {
                    _band_label(low, high): self._stats_artifact(
                        position_quality[phase_slot, band]
                    )
                    for band, (low, high) in enumerate(QUERY_POSITION_BANDS)
                },
                "distance_bands": {
                    _band_label(low, high): self._named_stats_artifact(
                        distance_quality[phase_slot, band],
                        GRAPH_DISTANCE_QUALITY_FIELDS,
                    )
                    for band, (low, high) in enumerate(DISTANCE_BANDS)
                },
            }
        return {
            "initialized": True,
            "storage": "persistent_device_moments",
            "fields": list(GRAPH_QUALITY_FIELDS),
            "phases": phases,
        }

    def artifact(self) -> dict[str, Any]:
        config = self._config
        if config is None:
            return {"available": False, "reason": "bucket selector runtime is not configured"}
        with self._lock:
            artifact: dict[str, Any] = {
                "schema_version": 1,
                "selector_identity": {
                    "selector": SELECTOR,
                    "semantics": "exact_local_topk_per_request_local_position_modulo_bucket",
                    "global_exact_topk": False,
                    "radix_selector": False,
                },
                "config": asdict(config),
                "telemetry_mode": config.telemetry,
                "recording_enabled": (
                    True
                    if config.telemetry in ("off", "graph_safety")
                    else self._host_recording_enabled
                ),
            }
            if config.telemetry in ("graph_safety", "graph_verify_exact"):
                artifact["graph_replay"] = self._graph_artifact()
                if config.telemetry == "graph_safety":
                    return artifact
                artifact["graph_replay_quality"] = self._graph_quality_artifact()
            artifact["safety"] = {"rows": self._safety_rows, "counts": dict(sorted(self._safety.items()))}
            for phase in PHASES:
                phase_groups = [group for (group_phase, _), group in self._dists.items() if group_phase == phase]
                layers = {
                    layer: {
                        "calls": self._calls[(phase, layer)],
                        "metrics": {name: summary.artifact() for name, summary in sorted(group.items())},
                        "per_bucket": [
                            {
                                name: summary.artifact()
                                for name, summary in sorted(self._bucket_dists.get((phase, layer, bucket), {}).items())
                            }
                            for bucket in range(config.bucket_count)
                        ],
                    }
                    for (group_phase, layer), group in sorted(self._dists.items())
                    if group_phase == phase
                }
                if not phase_groups:
                    continue
                artifact[phase] = {
                    "calls": sum(count for (call_phase, _), count in self._calls.items() if call_phase == phase),
                    "global": self._summaries(phase_groups),
                    "per_layer": layers,
                    "query_position_bands": {
                        band: {name: summary.artifact() for name, summary in sorted(group.items())}
                        for (group_phase, band), group in sorted(self._position_dists.items())
                        if group_phase == phase
                    },
                    "distance_bands": {
                        band: {name: summary.artifact() for name, summary in sorted(group.items())}
                        for (group_phase, band), group in sorted(self._distance_dists.items())
                        if group_phase == phase
                    },
                }
            return artifact

    def reset(self) -> None:
        with self._lock:
            self._dists.clear()
            self._position_dists.clear()
            self._bucket_dists.clear()
            self._distance_dists.clear()
            self._calls.clear()
            self._safety_rows = 0
            self._safety.clear()
            self._host_recording_enabled = True
            if self._graph is not None:
                self._graph.zero_()
                assert self._graph_buckets is not None
                self._graph_buckets.zero_()
                self._reset_graph_quality_storage()
                if self._graph.device.type == "cuda":
                    torch.cuda.synchronize(self._graph.device)

    def reset_for_test(self) -> None:
        self.reset()
        with self._lock:
            self._config = None
            self._layer = "unattributed"
            self._graph = None
            self._graph_buckets = None
            self._graph_quality = None
            self._graph_position_quality = None
            self._graph_bucket_quality = None
            self._graph_distance_quality = None
            self._graph_layer_slots.clear()
            self._host_recording_enabled = True


RUNTIME = BucketSelectorRuntime()
