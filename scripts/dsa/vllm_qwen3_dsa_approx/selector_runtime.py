"""Process-local configuration, attribution, safety checks, and telemetry."""

from __future__ import annotations

import atexit
import contextlib
import json
import math
import os
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from .radix_selector_reference import SELECTORS, SelectionResult
from .selector_telemetry import (
    COUNT_BINS,
    HIST_BINS,
    BoundedDist,
    DistSpec,
    integer_spec,
    ratio_spec,
)

TELEMETRY_MODES = ("off", "summary", "verify_exact", "graph_safety", "graph_verify_exact")
BACKENDS = ("vllm_stock", "dsa_csx_reference")
GRAPH_PHASES = ("prefill", "decode")
GRAPH_SAFETY_FIELDS = (
    "rows",
    "calls",
    "count_mismatch",
    "padding_violation",
    "invalid_index",
    "noncausal_index",
    "duplicate_index",
    "capacity_saturation",
    "rescued",
)
GRAPH_PHASE_SLOT = {name: index for index, name in enumerate(GRAPH_PHASES)}
GRAPH_SAFETY_SLOT = {name: index for index, name in enumerate(GRAPH_SAFETY_FIELDS)}
RANK_BANDS = ((0, 16), (16, 64), (64, 256), (256, 1024), (1024, 2048))
GRAPH_QUALITY_FIELDS = (
    "selected_count",
    "effective_k",
    "delta_k",
    "capacity_utilization",
    "intersection",
    "added",
    "dropped",
    "exact_recall",
    "precision",
    "jaccard",
    *(f"rank_recall_{low + 1}_{high}" for low, high in RANK_BANDS),
)
GRAPH_STAT_FIELDS = ("n", "total", "total_sq", "min", "max")
GRAPH_STAT_SLOT = {name: index for index, name in enumerate(GRAPH_STAT_FIELDS)}
DISTANCE_BANDS = (
    (0, 16),
    (16, 64),
    (64, 256),
    (256, 1024),
    (1024, 4096),
    (4096, 16384),
    (16384, None),
)
QUERY_POSITION_BANDS = (
    (0, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
    (16384, 24576),
    (24576, 32768),
    (32768, None),
)
_BAND_EDGES = torch.tensor([low for low, _ in QUERY_POSITION_BANDS[1:]], dtype=torch.int64)

# Retention policy knobs (plan §9). Raw per-row retention is OFF by default: §9 only asks for it
# on CONFIGURED samples/layers/stride, and one 32K prompt across ~30 sparse layers is ~1M rows per
# field. Aggregates are always kept, at fixed memory, via BoundedDist.
POSITION_BOUND = int(os.environ.get("DSA_APPROX_MAX_POSITION", str(1 << 18)))
RAW_ROWS_PER_GROUP = int(os.environ.get("DSA_APPROX_RAW_ROWS", "0"))
RAW_POSITION_STRIDE = max(1, int(os.environ.get("DSA_APPROX_RAW_POSITION_STRIDE", "64")))
RAW_LAYERS = tuple(name for name in os.environ.get("DSA_APPROX_RAW_LAYERS", "").split(",") if name)
VIOLATION_ROWS = int(os.environ.get("DSA_APPROX_VIOLATION_ROWS", "256"))
# UNFINISHED -- `distance_*` is parked, not validated. It is OFF by default and has not been read
# on a run that actually exercises the threshold rule, so treat any distance number as provisional
# until that happens. Revisit when long-context selection quality is the question.
#
# Cost: 21 of the 39 folded fields (7 DISTANCE_BANDS x 3 metrics), ~48% of verify_exact's per-call
# cost (16.63 -> 8.10 ms/call/layer measured with it off). Enable it only for a run whose prompts
# exceed rule_k -- below that every row takes its whole causal prefix (see ScoreView.degenerate) and
# no threshold is applied, so the group costs full price to report that nothing was approximated.
# Empty bands are excluded rather than recorded as 0.0 (see `_rate`), so they no longer mislead, but
# they still cost time. Enabled state is in the artifact's retention block.
DISTANCE_TELEMETRY = os.environ.get("DSA_APPROX_DISTANCE", "0") not in ("0", "", "false", "False")


def _finite_or_none(value: float) -> float | None:
    """JSON null for an undefined rate, so per-row records stay strict JSON."""

    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class SelectorConfig:
    selector: str
    backend: str
    rule_k: int
    capacity: int
    omit_bits: int
    telemetry: str
    artifact_path: str | None = None

    def validate(self) -> None:
        if self.selector not in SELECTORS:
            raise ValueError(f"dsa_selector={self.selector!r}; expected one of {SELECTORS}")
        if self.backend not in BACKENDS:
            raise ValueError(f"dsa_selector_backend={self.backend!r}; expected one of {BACKENDS}")
        if self.telemetry not in TELEMETRY_MODES:
            raise ValueError(f"dsa_telemetry={self.telemetry!r}; expected one of {TELEMETRY_MODES}")
        if self.rule_k <= 0 or self.capacity < self.rule_k:
            raise ValueError(f"invalid rule_k/capacity: {self.rule_k}/{self.capacity}")
        if self.omit_bits != 4:
            raise ValueError("the reused dsa-csx reference rule currently requires omit_bits=4")
        if self.selector == "topk" and self.backend != "vllm_stock":
            raise ValueError("topk control mode requires dsa_selector_backend='vllm_stock'")
        if self.selector != "topk" and self.backend != "dsa_csx_reference":
            raise ValueError("approximate selectors require dsa_selector_backend='dsa_csx_reference'")

    def validate_execution(self, *, cudagraph_enabled: bool) -> None:
        """Validate execution-mode combinations that affect measurement correctness.

        The reference selector itself is tensor-only and can be captured. Summary and verify_exact
        are host-folded Python, however, so they execute during capture but not on graph replay.
        graph_safety and graph_verify_exact instead update persistent device storage whose addresses
        are retained by every replay.
        """

        self.validate()
        if (
            cudagraph_enabled
            and self.selector != "topk"
            and self.telemetry not in ("off", "graph_safety", "graph_verify_exact")
        ):
            raise ValueError(
                "CUDA graphs for an approximate DSA selector currently require "
                "dsa_telemetry='off', 'graph_safety', or 'graph_verify_exact'; "
                "summary/verify_exact are host-folded and would not run on graph replay."
            )


def config_from_hf(config: Any) -> SelectorConfig:
    selector = str(getattr(config, "dsa_selector", "topk"))
    default_backend = "vllm_stock" if selector == "topk" else "dsa_csx_reference"
    value = SelectorConfig(
        selector=selector,
        backend=str(getattr(config, "dsa_selector_backend", default_backend)),
        rule_k=int(config.dsa_top_k),
        capacity=int(config.index_topk),
        omit_bits=int(getattr(config, "dsa_radix_omit_bits", 4)),
        telemetry=str(getattr(config, "dsa_telemetry", "off")),
        artifact_path=(
            os.environ.get("DSA_SELECTOR_ARTIFACT")
            or getattr(config, "dsa_telemetry_artifact", None)
        ),
    )
    value.validate()
    return value


class SelectorRuntime:
    """Bounded-memory telemetry for the approximate selector.

    Per-row metrics fold into fixed-width histograms as they arrive (plan §9): aggregates cost the
    same whether a run is one prompt or a whole benchmark. Two accumulator tables are kept, one
    keyed by layer and one by query-position band, so the per-layer and position-bin views of §9-§11
    both fall out without ever retaining the raw (row, layer, position) product.
    """

    def __init__(self) -> None:
        self._config: SelectorConfig | None = None
        # A PLAIN attribute, not a ContextVar. Dynamo cannot trace `ContextVar.set` (gb0156), and
        # this scope is entered from `indexer.forward`, which vLLM 0.26 aot_compiles in fullgraph
        # mode -- so a ContextVar here made every ACTIVE selector uncapturable and forced the
        # approximate server to eager. dsa-csx attributes layers the same way, with a plain
        # attribute and manual save/restore (glm_52/vllm_study/selector/runtime.py:120-153).
        #
        # The tradeoff is that this is not task- or thread-local. That is sound here because the
        # model forward runs single-threaded per worker, and the attribute is only ever read on the
        # same call stack that set it. `_lock` still guards the accumulators it feeds.
        self._layer = "unattributed"
        self._lock = threading.Lock()
        self._layer_dists: dict[tuple[str, str], dict[str, BoundedDist]] = defaultdict(dict)
        self._band_dists: dict[tuple[str, int], dict[str, BoundedDist]] = defaultdict(dict)
        self._calls: dict[tuple[str, str], int] = defaultdict(int)
        self._safety: dict[str, int] = defaultdict(int)
        self._violations: list[dict[str, Any]] = []
        self._raw: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Persistent graph-replay counters. Allocated once after model construction, before warmup
        # and capture, then zeroed in place after warmup. Captured graphs retain these addresses and
        # replay the in-place increments without running Python.
        self._graph_safety: torch.Tensor | None = None
        self._graph_layer_slots: dict[str, int] = {}
        self._graph_quality_stats: torch.Tensor | None = None
        self._graph_quality_histograms: torch.Tensor | None = None
        self._graph_quality_band_stats: torch.Tensor | None = None
        self._graph_quality_band_histograms: torch.Tensor | None = None
        self._graph_position_edges: torch.Tensor | None = None
        self._graph_quality_lows: torch.Tensor | None = None
        self._graph_quality_widths: torch.Tensor | None = None
        self._graph_quality_bin_max: torch.Tensor | None = None
        self._graph_quality_offset_tensor: torch.Tensor | None = None
        self._graph_quality_field_ids: torch.Tensor | None = None
        self._graph_quality_specs: dict[str, DistSpec] = {}
        self._graph_quality_hist_offsets: dict[str, int] = {}
        self._dump_registered = False
        self._hook_provenance: dict[str, Any] = {}

    @property
    def config(self) -> SelectorConfig | None:
        return self._config

    @property
    def active(self) -> bool:
        return self._config is not None and self._config.selector != "topk"

    def configure(self, config: SelectorConfig) -> None:
        config.validate()
        with self._lock:
            if self._config is not None and self._config != config:
                raise RuntimeError(
                    f"approximate selector configuration changed across layers: "
                    f"{self._config!r} != {config!r}"
                )
            self._config = config
            if config.artifact_path and not self._dump_registered:
                atexit.register(self.dump_artifact)
                self._dump_registered = True

    def reset(self) -> None:
        """Drop all accumulated telemetry, keeping configuration and provenance."""

        with self._lock:
            self._layer_dists.clear()
            self._band_dists.clear()
            self._calls.clear()
            self._safety.clear()
            self._violations.clear()
            self._raw.clear()
            if self._graph_safety is not None:
                self._graph_safety.zero_()
                self._reset_graph_quality_storage()
                # Reset is an out-of-graph lifecycle RPC after capture. Complete it before the API
                # server accepts traffic, including when replay uses a different CUDA stream.
                if self._graph_safety.device.type == "cuda":
                    torch.cuda.synchronize(self._graph_safety.device)

    def initialize_graph_safety(self, layer_names: list[str], *, device: torch.device) -> None:
        """Allocate stable replay counters before CUDA graph capture.

        Layer names are explicit and immutable for the worker lifetime. Reinitializing with a
        different layout would invalidate captured addresses and therefore fails closed.
        """

        if not layer_names or len(set(layer_names)) != len(layer_names):
            raise ValueError(f"graph_safety requires unique sparse layer names, got {layer_names!r}")
        slots = {name: index for index, name in enumerate(layer_names)}
        shape = (len(GRAPH_PHASES), len(layer_names), len(GRAPH_SAFETY_FIELDS))
        with self._lock:
            if self._graph_safety is not None:
                if self._graph_layer_slots != slots or tuple(self._graph_safety.shape) != shape:
                    raise RuntimeError("graph_safety layout changed after persistent allocation")
                return
            self._graph_layer_slots = slots
            self._graph_safety = torch.zeros(shape, dtype=torch.int64, device=device)

    def initialize_graph_quality(self, layer_names: list[str], *, device: torch.device) -> None:
        """Allocate fixed-address quality accumulators before decode graph capture."""

        self.initialize_graph_safety(layer_names, device=device)
        config = self._config
        if config is None or config.telemetry != "graph_verify_exact":
            raise RuntimeError("graph quality storage requires dsa_telemetry='graph_verify_exact'")
        specs = {
            field: (
                ratio_spec()
                if field == "capacity_utilization"
                or "recall" in field
                or field in ("precision", "jaccard")
                else integer_spec(
                    -config.capacity if field == "delta_k" else 0,
                    config.capacity,
                    COUNT_BINS,
                )
            )
            for field in GRAPH_QUALITY_FIELDS
        }
        offsets: dict[str, int] = {}
        histogram_bins = 0
        for field in GRAPH_QUALITY_FIELDS:
            offsets[field] = histogram_bins
            histogram_bins += specs[field].bins
        stats_shape = (
            len(GRAPH_PHASES),
            len(layer_names),
            len(GRAPH_QUALITY_FIELDS),
            len(GRAPH_STAT_FIELDS),
        )
        histogram_shape = (len(GRAPH_PHASES), len(layer_names), histogram_bins)
        band_stats_shape = (
            len(GRAPH_PHASES),
            len(QUERY_POSITION_BANDS),
            len(GRAPH_QUALITY_FIELDS),
            len(GRAPH_STAT_FIELDS),
        )
        band_histogram_shape = (len(GRAPH_PHASES), len(QUERY_POSITION_BANDS), histogram_bins)
        with self._lock:
            if self._graph_quality_stats is not None:
                if (
                    self._graph_quality_specs != specs
                    or self._graph_quality_hist_offsets != offsets
                    or tuple(self._graph_quality_stats.shape) != stats_shape
                ):
                    raise RuntimeError("graph quality layout changed after persistent allocation")
                return
            self._graph_quality_specs = specs
            self._graph_quality_hist_offsets = offsets
            self._graph_quality_stats = torch.zeros(stats_shape, dtype=torch.float64, device=device)
            self._graph_quality_histograms = torch.zeros(
                histogram_shape, dtype=torch.int64, device=device
            )
            self._graph_quality_band_stats = torch.zeros(
                band_stats_shape, dtype=torch.float64, device=device
            )
            self._graph_quality_band_histograms = torch.zeros(
                band_histogram_shape, dtype=torch.int64, device=device
            )
            self._graph_position_edges = _BAND_EDGES.to(device=device)
            self._graph_quality_lows = torch.tensor(
                [specs[field].low for field in GRAPH_QUALITY_FIELDS],
                dtype=torch.float64,
                device=device,
            )
            self._graph_quality_widths = torch.tensor(
                [specs[field].width for field in GRAPH_QUALITY_FIELDS],
                dtype=torch.float64,
                device=device,
            )
            self._graph_quality_bin_max = torch.tensor(
                [specs[field].bins - 1 for field in GRAPH_QUALITY_FIELDS],
                dtype=torch.int64,
                device=device,
            )
            self._graph_quality_offset_tensor = torch.tensor(
                [offsets[field] for field in GRAPH_QUALITY_FIELDS],
                dtype=torch.int64,
                device=device,
            )
            self._graph_quality_field_ids = torch.arange(
                len(GRAPH_QUALITY_FIELDS), dtype=torch.int64, device=device
            )
            self._reset_graph_quality_storage()

    def _reset_graph_quality_storage(self) -> None:
        for stats in (self._graph_quality_stats, self._graph_quality_band_stats):
            if stats is None:
                continue
            stats.zero_()
            stats[..., GRAPH_STAT_SLOT["min"]].fill_(float("inf"))
            stats[..., GRAPH_STAT_SLOT["max"]].fill_(float("-inf"))
        for histograms in (
            self._graph_quality_histograms,
            self._graph_quality_band_histograms,
        ):
            if histograms is not None:
                histograms.zero_()

    def set_hook_provenance(self, value: dict[str, Any]) -> None:
        self._hook_provenance = dict(value)

    @contextlib.contextmanager
    def layer(self, name: str) -> Iterator[None]:
        previous = self._layer
        self._layer = name
        try:
            yield
        finally:
            self._layer = previous

    def note_call(self, phase: str) -> None:
        with self._lock:
            self._calls[(phase, self._layer)] += 1

    def _record_graph_safety(
        self,
        *,
        phase: str,
        layer_name: str,
        rows: int,
        count_mismatch: torch.Tensor,
        padding_violation: torch.Tensor,
        invalid_index: torch.Tensor,
        noncausal_index: torch.Tensor,
        duplicate_index: torch.Tensor,
        capacity_saturation: torch.Tensor,
        rescued: torch.Tensor,
    ) -> None:
        counters = self._graph_safety
        if counters is None:
            raise RuntimeError(
                "dsa_telemetry='graph_safety' is active but persistent counters were not "
                "initialized before model execution"
            )
        try:
            phase_slot = GRAPH_PHASE_SLOT[phase]
            layer_slot = self._graph_layer_slots[layer_name]
        except KeyError as exc:
            raise RuntimeError(
                f"graph_safety has no stable slot for phase={phase!r}, layer={layer_name!r}"
            ) from exc

        target = counters[phase_slot, layer_slot]
        # Every operation below is fixed-shape and device-side. During capture these in-place adds
        # become graph nodes; replay updates the same persistent storage without executing Python.
        target[GRAPH_SAFETY_SLOT["rows"]].add_(rows)
        target[GRAPH_SAFETY_SLOT["calls"]].add_(1)
        for name, values in (
            ("count_mismatch", count_mismatch),
            ("padding_violation", padding_violation),
            ("invalid_index", invalid_index),
            ("noncausal_index", noncausal_index),
            ("duplicate_index", duplicate_index),
            ("capacity_saturation", capacity_saturation),
            ("rescued", rescued),
        ):
            target[GRAPH_SAFETY_SLOT[name]].add_(values.to(torch.int64).sum())

    def _graph_quality_record(
        self,
        *,
        phase: str,
        layer_name: str,
        record: dict[str, torch.Tensor],
        position_bands: torch.Tensor,
    ) -> None:
        stats = self._graph_quality_stats
        histograms = self._graph_quality_histograms
        band_stats = self._graph_quality_band_stats
        band_histograms = self._graph_quality_band_histograms
        lows = self._graph_quality_lows
        widths = self._graph_quality_widths
        bin_max = self._graph_quality_bin_max
        offsets = self._graph_quality_offset_tensor
        field_ids = self._graph_quality_field_ids
        if any(
            value is None
            for value in (
                stats,
                histograms,
                band_stats,
                band_histograms,
                lows,
                widths,
                bin_max,
                offsets,
                field_ids,
            )
        ):
            raise RuntimeError(
                "dsa_telemetry='graph_verify_exact' is active but persistent quality "
                "accumulators were not initialized before model execution"
            )
        assert stats is not None
        assert histograms is not None
        assert band_stats is not None
        assert band_histograms is not None
        assert lows is not None
        assert widths is not None
        assert bin_max is not None
        assert offsets is not None
        assert field_ids is not None
        phase_slot = GRAPH_PHASE_SLOT[phase]
        layer_slot = self._graph_layer_slots[layer_name]

        # Fold every field together. Fifteen separate Python-level folds would bake hundreds of
        # redundant conversion/reduction nodes per layer into each captured decode shape.
        values = torch.stack([record[field] for field in GRAPH_QUALITY_FIELDS], dim=-1).to(
            torch.float64
        )
        finite = torch.isfinite(values)
        finite_float = finite.to(torch.float64)
        safe = torch.where(finite, values, torch.zeros_like(values))
        minimum = torch.where(
            finite, values, torch.full_like(values, float("inf"))
        ).amin(0)
        maximum = torch.where(
            finite, values, torch.full_like(values, float("-inf"))
        ).amax(0)
        target = stats[phase_slot, layer_slot]
        target[:, GRAPH_STAT_SLOT["n"]].add_(finite_float.sum(0))
        target[:, GRAPH_STAT_SLOT["total"]].add_(safe.sum(0))
        target[:, GRAPH_STAT_SLOT["total_sq"]].add_(safe.square().sum(0))
        target_min = target[:, GRAPH_STAT_SLOT["min"]]
        target_max = target[:, GRAPH_STAT_SLOT["max"]]
        target_min.copy_(torch.minimum(target_min, minimum))
        target_max.copy_(torch.maximum(target_max, maximum))

        raw_bins = torch.floor((safe - lows) / widths).to(torch.int64)
        bins = torch.minimum(raw_bins.clamp(min=0), bin_max)
        histogram_indices = bins + offsets
        histograms[phase_slot, layer_slot].scatter_add_(
            0, histogram_indices.flatten(), finite.flatten().to(torch.int64)
        )

        # Position-band storage is shared across layers. FULL_DECODE_ONLY replays are serialized
        # by one EngineCore worker/stream; scatter_add handles repeated rows within a replay.
        band = position_bands.flatten().to(torch.int64)
        band_field = band[:, None] * len(GRAPH_QUALITY_FIELDS) + field_ids
        band_target = band_stats[phase_slot]
        band_target[:, :, GRAPH_STAT_SLOT["n"]].view(-1).scatter_add_(
            0, band_field.flatten(), finite_float.flatten()
        )
        band_target[:, :, GRAPH_STAT_SLOT["total"]].view(-1).scatter_add_(
            0, band_field.flatten(), safe.flatten()
        )
        band_target[:, :, GRAPH_STAT_SLOT["total_sq"]].view(-1).scatter_add_(
            0, band_field.flatten(), safe.square().flatten()
        )
        band_target[:, :, GRAPH_STAT_SLOT["min"]].view(-1).scatter_reduce_(
            0,
            band_field.flatten(),
            torch.where(finite, values, torch.full_like(values, float("inf"))).flatten(),
            reduce="amin",
            include_self=True,
        )
        band_target[:, :, GRAPH_STAT_SLOT["max"]].view(-1).scatter_reduce_(
            0,
            band_field.flatten(),
            torch.where(finite, values, torch.full_like(values, float("-inf"))).flatten(),
            reduce="amax",
            include_self=True,
        )
        flat_band_histogram = band_histograms[phase_slot].view(-1)
        band_bin = band[:, None] * band_histograms.shape[-1] + histogram_indices
        flat_band_histogram.scatter_add_(
            0, band_bin.flatten(), finite.flatten().to(torch.int64)
        )

    def _record_graph_quality(
        self,
        *,
        phase: str,
        layer_name: str,
        output: torch.Tensor,
        ordered_output: torch.Tensor,
        valid: torch.Tensor,
        observed_count: torch.Tensor,
        qpos: torch.Tensor,
        result: SelectionResult,
        stock_reference: torch.Tensor,
        key_count: int,
    ) -> None:
        """Fold exact-overlap quality into persistent tensors on every graph replay."""

        stock_valid = stock_reference >= 0
        stock_count = stock_valid.sum(-1)
        safe_stock = stock_reference.clamp(min=0, max=max(key_count - 1, 0)).long()
        # Membership without a [rows, key_count] dense bitmap: sort the emitted indices once and
        # binary-search every stock index. All shapes depend only on the captured decode shape.
        insertion = torch.searchsorted(ordered_output.long(), safe_stock).clamp(
            max=max(output.shape[1] - 1, 0)
        )
        exact_retained = (
            ordered_output.long().gather(-1, insertion) == safe_stock
        ) & stock_valid
        intersection = exact_retained.sum(-1)
        added = observed_count - intersection
        dropped = stock_count - intersection
        record: dict[str, torch.Tensor] = {
            "selected_count": observed_count,
            "effective_k": result.effective_k,
            "delta_k": observed_count - result.effective_k,
            "capacity_utilization": observed_count.float() / output.shape[1],
            "intersection": intersection,
            "added": added,
            "dropped": dropped,
            "exact_recall": intersection.float() / stock_count.clamp(min=1),
            "precision": intersection.float() / observed_count.clamp(min=1),
            "jaccard": intersection.float()
            / (observed_count + stock_count - intersection).clamp(min=1),
        }
        for low, high in RANK_BANDS:
            stop = min(high, stock_reference.shape[1])
            if low >= stop:
                denominator = torch.zeros_like(stock_count)
                numerator = torch.zeros_like(stock_count)
            else:
                denominator = stock_valid[:, low:stop].sum(-1)
                numerator = exact_retained[:, low:stop].sum(-1)
            record[f"rank_recall_{low + 1}_{high}"] = self._rate(numerator, denominator)

        selector = self._config.selector if self._config is not None else ""
        if selector == "radix_ceil":
            self._assert_tensor((added == 0).all(), "radix_ceil added keys outside exact top-k")
        if selector in ("radix_floor", "exact_ge"):
            self._assert_tensor((dropped == 0).all(), f"{selector} dropped exact top-k keys")

        edges = self._graph_position_edges
        if edges is None:
            raise RuntimeError("graph quality position bands were not initialized before capture")
        position_bands = torch.bucketize(qpos, edges, right=True)
        self._graph_quality_record(
            phase=phase,
            layer_name=layer_name,
            record=record,
            position_bands=position_bands,
        )

    @staticmethod
    def _assert_tensor(condition: torch.Tensor, message: str) -> None:
        if condition.device.type == "cuda":
            torch._assert_async(condition, message)
        elif not bool(condition):
            raise RuntimeError(message)

    # Only delta_k can go negative (an under-capturing arm selects fewer than effective k); every
    # other integer metric is a count bounded by the output buffer. Anything that lands outside a
    # spec clamps into the edge bin and is reported under `out_of_range`, so min/max stay exact and
    # a surprise is visible rather than silent.
    SIGNED_FIELDS = ("delta_k",)

    @staticmethod
    def _rate(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """A per-row rate that is NaN where the population is empty, not 0.0.

        A band containing no exact keys has no recall -- the quantity is undefined, not zero. The
        previous `numerator / denominator.clamp(min=1)` folded a 0.0 for those rows, which made an
        empty band indistinguishable from one where every key was dropped: at a context shorter
        than the widest band, `distance_recall_16384_plus` read 0.0 for every row and looked like
        total failure. BoundedDist.add already drops non-finite samples, so NaN excludes the row
        from n/mean/std/min/max and the histogram, and the surviving `n` reports exactly how many
        rows actually had a population.
        """

        rate = numerator.float() / denominator.clamp(min=1)
        return torch.where(denominator > 0, rate, torch.full_like(rate, float("nan")))

    def _spec(self, field: str, values: torch.Tensor) -> DistSpec:
        """Bin layout per field. Every float metric is a rate in [0, 1]; the rest are counts."""

        if values.dtype.is_floating_point:
            return ratio_spec()
        if field == "query_position":
            # Coarse is fine here: positions are reported through exact bands, not percentiles.
            return integer_spec(0, POSITION_BOUND, HIST_BINS)
        capacity = self._config.capacity if self._config is not None else 1
        low = -capacity if field in self.SIGNED_FIELDS else 0
        return integer_spec(low, capacity, COUNT_BINS)

    def _fold(
        self,
        table: dict[Any, dict[str, BoundedDist]],
        key: Any,
        field: str,
        values: torch.Tensor,
    ) -> None:
        leaf = table[key]
        accumulator = leaf.get(field)
        if accumulator is None:
            accumulator = leaf[field] = BoundedDist(self._spec(field, values))
        accumulator.add(values)

    @torch.no_grad()
    def validate_and_record(
        self,
        *,
        phase: str,
        output: torch.Tensor,
        query_positions: torch.Tensor,
        result: SelectionResult,
        stock_reference: torch.Tensor | None,
        key_count: int,
        layer_name: str | None = None,
    ) -> None:
        rows, capacity = output.shape
        valid = output >= 0
        observed_count = valid.sum(-1)
        qpos = query_positions.reshape(-1).to(output.device)
        count_mismatch = observed_count != result.selected_count
        padding_violation = ((~valid[:, :-1]) & valid[:, 1:]).any(-1)
        invalid_index = (output < -1).any(-1)
        noncausal_index = (output > qpos[:, None]).any(-1)
        sentinel = torch.full_like(output, key_count)
        ordered = torch.where(valid, output, sentinel).sort(-1).values
        duplicates = ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != key_count)).sum(-1)

        # Mandatory fatal gates (§15). Live in every telemetry mode, including `off`.
        self._assert_tensor(
            ~count_mismatch.any(),
            "selector count does not match emitted non-negative indices",
        )
        self._assert_tensor(
            ~padding_violation.any(),
            "approximate selector output is not valid-prefix/-1-suffix",
        )
        self._assert_tensor(
            ~(invalid_index | noncausal_index).any(),
            "approximate selector emitted an invalid or noncausal request-local index",
        )
        self._assert_tensor(
            (duplicates == 0).all(),
            "approximate selector emitted duplicate request-local indices",
        )

        config = self._config
        if config is None or config.telemetry == "off":
            return
        capacity_saturation = (
            observed_count >= capacity
            if capacity > config.rule_k
            else torch.zeros_like(count_mismatch)
        )
        if config.telemetry == "graph_safety" or (
            config.telemetry == "graph_verify_exact" and phase == "decode"
        ):
            graph_layer = layer_name or self._layer
            self._record_graph_safety(
                phase=phase,
                layer_name=graph_layer,
                rows=rows,
                count_mismatch=count_mismatch,
                padding_violation=padding_violation,
                invalid_index=invalid_index,
                noncausal_index=noncausal_index,
                duplicate_index=duplicates > 0,
                capacity_saturation=capacity_saturation,
                rescued=result.rescued,
            )
            if config.telemetry == "graph_verify_exact":
                if stock_reference is None:
                    raise RuntimeError("graph_verify_exact requires the stock exact-k reference")
                self._record_graph_quality(
                    phase=phase,
                    layer_name=graph_layer,
                    output=output,
                    ordered_output=ordered,
                    valid=valid,
                    observed_count=observed_count,
                    qpos=qpos,
                    result=result,
                    stock_reference=stock_reference,
                    key_count=key_count,
                )
            return
        record: dict[str, torch.Tensor] = {
            "query_position": qpos,
            "selected_count": observed_count,
            "effective_k": result.effective_k,
            "delta_k": observed_count - result.effective_k,
            "capacity_utilization": observed_count.float() / capacity,
            "rescued": result.rescued.to(torch.int64),
            "duplicates": duplicates,
        }
        # An over-capturing arm that fills the buffer exactly may have been truncated; for an arm
        # whose capacity IS k, a full row is the normal case and not a saturation signal.
        flags: dict[str, torch.Tensor] = {
            "count_mismatch": count_mismatch,
            "padding_violation": padding_violation,
            "invalid_index": invalid_index,
            "noncausal_index": noncausal_index,
            "duplicate_index": duplicates > 0,
            "capacity_saturation": capacity_saturation,
            "rescued": result.rescued,
        }
        if stock_reference is not None:
            stock_valid = stock_reference >= 0
            stock_count = stock_valid.sum(-1)
            approx_flat = torch.where(valid, output.long(), torch.full_like(output, key_count).long())
            stock_flat = torch.where(
                stock_valid,
                stock_reference.long(),
                torch.full_like(stock_reference, key_count).long(),
            )
            approx_set = torch.zeros(
                rows, key_count + 1, dtype=torch.bool, device=output.device
            ).scatter_(-1, approx_flat, True)[:, :key_count]
            stock_set = torch.zeros(
                rows, key_count + 1, dtype=torch.bool, device=output.device
            ).scatter_(-1, stock_flat, True)[:, :key_count]
            intersection = (approx_set & stock_set).sum(-1)
            added = (approx_set & ~stock_set).sum(-1)
            dropped = (~approx_set & stock_set).sum(-1)
            record.update(
                {
                    "intersection": intersection,
                    "added": added,
                    "dropped": dropped,
                    "exact_recall": intersection.float() / stock_count.clamp(min=1),
                    "precision": intersection.float() / observed_count.clamp(min=1),
                    "jaccard": intersection.float()
                    / (observed_count + stock_count - intersection).clamp(min=1),
                }
            )
            safe_stock = stock_reference.clamp(min=0, max=max(key_count - 1, 0)).long()
            exact_retained = approx_set.gather(-1, safe_stock) & stock_valid
            for low, high in RANK_BANDS:
                if low >= stock_reference.shape[1]:
                    continue
                stop = min(high, stock_reference.shape[1])
                denominator = stock_valid[:, low:stop].sum(-1)
                record[f"rank_recall_{low + 1}_{high}"] = self._rate(
                    exact_retained[:, low:stop].sum(-1), denominator
                )

            if DISTANCE_TELEMETRY:
                exact_distance = qpos[:, None] - stock_reference
                approx_distance = qpos[:, None] - output
                exact_overlap = exact_retained
                safe_approx = output.clamp(min=0, max=max(key_count - 1, 0)).long()
                approx_added = (~stock_set.gather(-1, safe_approx)) & valid
                for low, high in DISTANCE_BANDS:
                    label = f"{low}_{high if high is not None else 'plus'}"
                    exact_band = stock_valid & (exact_distance >= low)
                    approx_band = valid & (approx_distance >= low)
                    if high is not None:
                        exact_band &= exact_distance < high
                        approx_band &= approx_distance < high
                    exact_denominator = exact_band.sum(-1)
                    overlap_in_band = (exact_overlap & exact_band).sum(-1)
                    # The two counts below stay counts: an empty band genuinely dropped and
                    # added nothing. Only the rate is undefined.
                    record[f"distance_recall_{label}"] = self._rate(
                        overlap_in_band, exact_denominator
                    )
                    record[f"distance_dropped_{label}"] = exact_denominator - overlap_in_band
                    record[f"distance_added_{label}"] = (approx_added & approx_band).sum(-1)
            selector = config.selector
            if selector == "radix_ceil":
                self._assert_tensor((added == 0).all(), "radix_ceil added keys outside exact top-k")
            if selector in ("radix_floor", "exact_ge"):
                self._assert_tensor((dropped == 0).all(), f"{selector} dropped exact top-k keys")
            flags["containment_added"] = (added > 0) if selector == "radix_ceil" else torch.zeros_like(
                count_mismatch
            )
            flags["containment_dropped"] = (
                (dropped > 0)
                if selector in ("radix_floor", "exact_ge")
                else torch.zeros_like(count_mismatch)
            )

        # One host transfer per field, then every aggregate is computed on CPU at fixed cost.
        host = {name: values.detach().to("cpu") for name, values in record.items()}
        host_flags = {name: values.detach().to("cpu").bool() for name, values in flags.items()}
        band = torch.bucketize(host["query_position"].to(torch.int64), _BAND_EDGES, right=True)
        layer = self._layer

        with self._lock:
            self._safety["rows"] += rows
            violating = torch.zeros(rows, dtype=torch.bool)
            for name, flag in host_flags.items():
                self._safety[name] += int(flag.sum())
                violating |= flag
            for name, values in host.items():
                self._fold(self._layer_dists, (phase, layer), name, values)
            for band_index in band.unique().tolist():
                mask = band == band_index
                for name, values in host.items():
                    self._fold(self._band_dists, (phase, int(band_index)), name, values[mask])
            # §9 item 5: the complete raw record for every safety/containment violation, always.
            for index in torch.nonzero(violating).flatten().tolist():
                if len(self._violations) >= VIOLATION_ROWS:
                    break
                self._violations.append(
                    {
                        "phase": phase,
                        "layer": layer,
                        "flags": sorted(
                            name for name, flag in host_flags.items() if bool(flag[index])
                        ),
                        **{
                            name: _finite_or_none(float(values[index]))
                            for name, values in host.items()
                        },
                    }
                )
            # §9 item 4: raw records only for configured layers, at a position stride, capped.
            if RAW_ROWS_PER_GROUP and (not RAW_LAYERS or layer in RAW_LAYERS):
                leaf = self._raw[(phase, layer)]
                remaining = RAW_ROWS_PER_GROUP - len(leaf.get("query_position", ()))
                if remaining > 0:
                    keep = (host["query_position"].to(torch.int64) % RAW_POSITION_STRIDE) == 0
                    index = torch.nonzero(keep).flatten()[:remaining]
                    if int(index.numel()):
                        for name, values in host.items():
                            leaf[name].extend(
                                _finite_or_none(value) for value in values[index].tolist()
                            )

    def _graph_safety_snapshot(self) -> dict[str, Any] | None:
        counters = self._graph_safety
        if counters is None:
            return None
        host = counters.detach().to("cpu")
        names_by_slot = [None] * len(self._graph_layer_slots)
        for name, slot in self._graph_layer_slots.items():
            names_by_slot[slot] = name

        phases: dict[str, Any] = {}
        total = {field: 0 for field in GRAPH_SAFETY_FIELDS}
        for phase_slot, phase in enumerate(GRAPH_PHASES):
            per_layer: dict[str, Any] = {}
            phase_total = {field: 0 for field in GRAPH_SAFETY_FIELDS}
            for layer_slot, layer in enumerate(names_by_slot):
                values = {
                    field: int(host[phase_slot, layer_slot, field_slot])
                    for field_slot, field in enumerate(GRAPH_SAFETY_FIELDS)
                }
                per_layer[str(layer)] = values
                for field, value in values.items():
                    phase_total[field] += value
                    total[field] += value
            phases[phase] = {"global": phase_total, "per_layer": per_layer}
        return {
            "mode": "persistent_device_counters",
            "semantics": (
                "fixed-address counters updated by captured selector safety ops on every CUDA-graph "
                "replay; reset in place after engine warmup"
            ),
            "fields": list(GRAPH_SAFETY_FIELDS),
            "global": total,
            "phases": phases,
        }

    def _graph_dist(
        self,
        field: str,
        stats: torch.Tensor,
        histogram: torch.Tensor,
    ) -> BoundedDist:
        accumulator = BoundedDist(self._graph_quality_specs[field])
        count = int(stats[GRAPH_STAT_SLOT["n"]])
        if count:
            accumulator.n = count
            accumulator._total = float(stats[GRAPH_STAT_SLOT["total"]])
            accumulator._total_sq = float(stats[GRAPH_STAT_SLOT["total_sq"]])
            accumulator._min = float(stats[GRAPH_STAT_SLOT["min"]])
            accumulator._max = float(stats[GRAPH_STAT_SLOT["max"]])
            accumulator._counts.copy_(histogram)
        return accumulator

    def _graph_quality_snapshot(self) -> dict[str, Any] | None:
        stats = self._graph_quality_stats
        histograms = self._graph_quality_histograms
        band_stats = self._graph_quality_band_stats
        band_histograms = self._graph_quality_band_histograms
        if any(
            value is None
            for value in (stats, histograms, band_stats, band_histograms)
        ):
            return None
        assert stats is not None
        assert histograms is not None
        assert band_stats is not None
        assert band_histograms is not None
        host_stats = stats.detach().to("cpu")
        host_histograms = histograms.detach().to("cpu")
        host_band_stats = band_stats.detach().to("cpu")
        host_band_histograms = band_histograms.detach().to("cpu")
        names_by_slot = [None] * len(self._graph_layer_slots)
        for name, slot in self._graph_layer_slots.items():
            names_by_slot[slot] = name

        phases: dict[str, Any] = {}
        position_histograms: dict[str, list[dict[str, Any]]] = {}
        for phase_slot, phase in enumerate(GRAPH_PHASES):
            per_layer: dict[str, Any] = {}
            merged = {
                field: BoundedDist(self._graph_quality_specs[field])
                for field in GRAPH_QUALITY_FIELDS
            }
            worst: dict[str, tuple[float, str]] = {}
            for layer_slot, layer in enumerate(names_by_slot):
                layer_metrics: dict[str, Any] = {}
                for field in GRAPH_QUALITY_FIELDS:
                    field_slot = GRAPH_QUALITY_FIELDS.index(field)
                    offset = self._graph_quality_hist_offsets[field]
                    bins = self._graph_quality_specs[field].bins
                    accumulator = self._graph_dist(
                        field,
                        host_stats[phase_slot, layer_slot, field_slot],
                        host_histograms[phase_slot, layer_slot, offset : offset + bins],
                    )
                    layer_metrics[field] = accumulator.summary()
                    merged[field].merge(accumulator)
                    maximum = accumulator.maximum
                    if maximum is not None and (
                        field not in worst or maximum > worst[field][0]
                    ):
                        worst[field] = (maximum, str(layer))
                per_layer[str(layer)] = layer_metrics
            if not any(accumulator.n for accumulator in merged.values()):
                continue
            phases[phase] = {
                "per_layer": per_layer,
                "global": {
                    field: accumulator.summary() for field, accumulator in merged.items()
                },
                "max_by_layer": {
                    field: {"max": value, "layer": layer}
                    for field, (value, layer) in worst.items()
                },
            }
            bands: list[dict[str, Any]] = []
            for band_slot, (low, high) in enumerate(QUERY_POSITION_BANDS):
                metrics: dict[str, Any] = {}
                rows = 0
                for field in GRAPH_QUALITY_FIELDS:
                    field_slot = GRAPH_QUALITY_FIELDS.index(field)
                    offset = self._graph_quality_hist_offsets[field]
                    bins = self._graph_quality_specs[field].bins
                    accumulator = self._graph_dist(
                        field,
                        host_band_stats[phase_slot, band_slot, field_slot],
                        host_band_histograms[
                            phase_slot, band_slot, offset : offset + bins
                        ],
                    )
                    metrics[field] = accumulator.summary()
                    if field == "selected_count":
                        rows = accumulator.n
                if rows:
                    bands.append(
                        {"start": low, "end": high, "rows": rows, "metrics": metrics}
                    )
            position_histograms[phase] = bands
        return {
            "mode": "persistent_device_histograms",
            "reference": "literal_vllm_stock_topk",
            "semantics": (
                "fixed-address moments and histograms updated by captured decode quality ops on "
                "every CUDA-graph replay; eager prefill remains host-folded"
            ),
            "fields": list(GRAPH_QUALITY_FIELDS),
            "phases": phases,
            "position_histograms": position_histograms,
        }

    def artifact(self) -> dict[str, Any]:
        config = self._config
        with self._lock:
            layer_items = [(key, dict(fields)) for key, fields in self._layer_dists.items()]
            band_items = sorted(
                ((key, dict(fields)) for key, fields in self._band_dists.items()),
                key=lambda item: item[0][1],
            )
            calls = dict(self._calls)
            safety = dict(self._safety)
            violations = list(self._violations)
            raw = {key: {name: list(values) for name, values in fields.items()}
                   for key, fields in self._raw.items()}

        grouped: dict[str, Any] = {}
        for (phase, layer), fields in layer_items:
            leaf = grouped.setdefault(phase, {}).setdefault("per_layer", {}).setdefault(layer, {})
            for name, accumulator in fields.items():
                leaf[name] = accumulator.summary()
        for phase in list(grouped):
            merged: dict[str, BoundedDist] = {}
            worst: dict[str, tuple[float, str]] = {}
            for (other_phase, layer), fields in layer_items:
                if other_phase != phase:
                    continue
                for name, accumulator in fields.items():
                    target = merged.get(name)
                    if target is None:
                        target = merged[name] = BoundedDist(accumulator.spec)
                    target.merge(accumulator)
                    top = accumulator.maximum
                    if top is not None and (name not in worst or top > worst[name][0]):
                        worst[name] = (top, layer)
            grouped[phase]["global"] = {
                name: accumulator.summary() for name, accumulator in merged.items()
            }
            # §9: at one position the cross-layer population is small, so name the layer that owns
            # the maximum instead of leaving p99 to stand in for it.
            grouped[phase]["max_by_layer"] = {
                name: {"max": value, "layer": layer} for name, (value, layer) in worst.items()
            }
        for (phase, layer), count in calls.items():
            grouped.setdefault(phase, {}).setdefault("calls", {})[layer] = count
        for (phase, layer), fields in raw.items():
            grouped.setdefault(phase, {}).setdefault("raw_records", {})[layer] = {
                "rows": len(fields.get("query_position", ())),
                "position_stride": RAW_POSITION_STRIDE,
                "fields": fields,
            }

        position_histograms: dict[str, Any] = {}
        for (phase, band_index), fields in band_items:
            low, high = QUERY_POSITION_BANDS[band_index]
            position_histograms.setdefault(phase, []).append(
                {
                    "start": low,
                    "end": high,
                    "rows": fields["query_position"].n if "query_position" in fields else 0,
                    "metrics": {
                        name: accumulator.summary() for name, accumulator in fields.items()
                    },
                }
            )

        rows = safety.pop("rows", 0)
        graph_safety = self._graph_safety_snapshot()
        graph_quality = self._graph_quality_snapshot()
        graph_quality_meta = (
            {
                name: value
                for name, value in graph_quality.items()
                if name not in ("phases", "position_histograms")
            }
            if graph_quality is not None
            else None
        )
        if graph_safety is not None:
            graph_global = graph_safety["global"]
            rows += int(graph_global["rows"])
            for name in GRAPH_SAFETY_FIELDS:
                if name not in ("rows", "calls"):
                    safety[name] = safety.get(name, 0) + int(graph_global[name])
        if graph_quality is not None:
            for phase, quality_phase in graph_quality["phases"].items():
                if phase in grouped and grouped[phase].get("global"):
                    raise RuntimeError(
                        f"both host and graph quality telemetry were recorded for phase={phase!r}"
                    )
                grouped[phase] = quality_phase
                if graph_safety is not None:
                    calls = graph_safety["phases"][phase]["per_layer"]
                    grouped[phase]["calls"] = {
                        layer: values["calls"] for layer, values in calls.items()
                    }
            for phase, bands in graph_quality["position_histograms"].items():
                if bands:
                    position_histograms[phase] = bands
        return {
            "meta": {
                "selector": asdict(config) if config else None,
                "selector_speed_claim_valid": False,
                "semantics": (
                    "dsa_csx_reference uses exact torch.topk internally; only approximate "
                    "attention quality and selector telemetry are valid"
                ),
                "retention": {
                    "histogram_bins": HIST_BINS,
                    "raw_rows_per_group": RAW_ROWS_PER_GROUP,
                    "raw_position_stride": RAW_POSITION_STRIDE,
                    "raw_layers": list(RAW_LAYERS) or "all",
                    "violation_row_cap": VIOLATION_ROWS,
                    "distance_telemetry": DISTANCE_TELEMETRY,
                    "note": (
                        "n/mean/std/min/max are exact over every row; percentiles come from "
                        "fixed-width histograms and are exact where exact_percentiles is true. "
                        "Rate fields (rank_recall_*, distance_recall_*) are undefined where the "
                        "band held no exact keys; those rows are excluded, so n is the number of "
                        "rows that actually had a population and may be below the row count"
                    ),
                },
            },
            "provenance": {"hooks": self._hook_provenance},
            "safety": {
                "rows": rows,
                "counts": safety,
                "rates": {
                    name: (count / rows if rows else 0.0) for name, count in safety.items()
                },
                "violation_records": violations,
                "violation_records_capped": len(violations) >= VIOLATION_ROWS,
            },
            "position_histograms": position_histograms,
            **({"graph_replay_safety": graph_safety} if graph_safety is not None else {}),
            **(
                {"graph_replay_quality": graph_quality_meta}
                if graph_quality_meta is not None
                else {}
            ),
            **grouped,
        }

    def dump_artifact(self) -> None:
        config = self._config
        if config is None or not config.artifact_path or int(os.environ.get("RANK", "0")) != 0:
            return
        path = Path(config.artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.artifact(), indent=2) + "\n")
        temporary.replace(path)


RUNTIME = SelectorRuntime()
