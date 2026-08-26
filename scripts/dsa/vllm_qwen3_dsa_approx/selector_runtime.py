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


TELEMETRY_MODES = ("off", "summary", "verify_exact", "graph_safety")
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
DISTANCE_BANDS = ((0, 16), (16, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, None))
QUERY_POSITION_BANDS = ((0, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 24576), (24576, 32768), (32768, None))
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

        The reference selector itself is tensor-only and can be captured. The existing summary and
        verify telemetry is host-folded Python, however, so it executes during capture but not on
        CUDA-graph replay. Allowing that combination would produce a plausible, incomplete artifact.
        Stage A therefore supports active CUDA graphs only with telemetry disabled; persistent
        device accumulators will lift this restriction in Stage B.
        """

        self.validate()
        if (
            cudagraph_enabled
            and self.selector != "topk"
            and self.telemetry not in ("off", "graph_safety")
        ):
            raise ValueError(
                "CUDA graphs for an approximate DSA selector currently require "
                "dsa_telemetry='off' or 'graph_safety'; summary/verify_exact are host-folded and "
                "would not run on graph replay. Use eager mode for distribution telemetry."
            )


def config_from_hf(config: Any) -> SelectorConfig:
    selector = str(getattr(config, "dsa_selector", "topk"))
    default_backend = "vllm_stock" if selector == "topk" else "dsa_csx_reference"
    value = SelectorConfig(
        selector=selector,
        backend=str(getattr(config, "dsa_selector_backend", default_backend)),
        rule_k=int(getattr(config, "dsa_top_k")),
        capacity=int(getattr(config, "index_topk")),
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
        if config.telemetry == "graph_safety":
            self._record_graph_safety(
                phase=phase,
                layer_name=layer_name or self._layer,
                rows=rows,
                count_mismatch=count_mismatch,
                padding_violation=padding_violation,
                invalid_index=invalid_index,
                noncausal_index=noncausal_index,
                duplicate_index=duplicates > 0,
                capacity_saturation=capacity_saturation,
                rescued=result.rescued,
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
        if graph_safety is not None:
            graph_global = graph_safety["global"]
            rows += int(graph_global["rows"])
            for name in GRAPH_SAFETY_FIELDS:
                if name not in ("rows", "calls"):
                    safety[name] = safety.get(name, 0) + int(graph_global[name])
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
