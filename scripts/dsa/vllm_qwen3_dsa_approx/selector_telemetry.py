"""Distribution summaries for approximate-selector serving telemetry.

The histogram helpers originate in
``dsa-csx/glm_52/study_core/summaries.py`` and are extended with the percentile
contract used by the Qwen3 serving plan.
"""

from __future__ import annotations

import dataclasses
import math
import os

import torch


N_BUCKETS = int(os.environ.get("DSA_APPROX_COLLECT_BUCKETS", "64"))
PCT = [
    ("min", 0.0),
    ("p01", 0.01),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.5),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p999", 0.999),
    ("max", 1.0),
]


def dist_values(x: torch.Tensor, n_buckets: int | None = None) -> dict:
    """Summarize finite samples as moments, percentiles, and a histogram."""

    bucket_count = N_BUCKETS if n_buckets is None else n_buckets
    x = x.float().flatten()
    x = x[torch.isfinite(x)]
    n = int(x.numel())
    if n == 0:
        return {"n": 0}
    if n > QUANTILE_LIMIT:
        # torch.quantile hard-fails above 2**24 elements, so fall back to the bounded
        # histogram path rather than dying on the population this helper was handed.
        return bounded_dist_values(x)
    low, high = float(x.min()), float(x.max())
    quantiles = torch.quantile(
        x, torch.tensor([fraction for _, fraction in PCT], device=x.device)
    ).tolist()
    counts = (
        torch.histc(x, bins=bucket_count, min=low, max=high).long().tolist()
        if high > low
        else [n] + [0] * (bucket_count - 1)
    )
    return {
        "n": n,
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "percentiles": {
            name: value for (name, _), value in zip(PCT, quantiles, strict=True)
        },
        "histogram": {
            "counts": counts,
            "edges": torch.linspace(low, high, bucket_count + 1).tolist(),
        },
    }


def low4_occupancy(low4: torch.Tensor) -> dict:
    """Report which of the sixteen 4-bit values a population occupies, and how often."""

    counts = torch.bincount(low4.long(), minlength=16)
    return {
        "n": int(low4.numel()),
        "occupied_values": [
            index for index, count in enumerate(counts.tolist()) if count
        ],
        "histogram": {"values": list(range(16)), "counts": counts.tolist()},
    }


def tq_source_precision(tq: torch.Tensor, *, note: bool = False) -> dict:
    """Summarize the raw IEEE FP16 mantissa occupancy of a threshold population."""

    from .radix_rules import mono, mono_to_bits

    if not tq.numel():
        return {"n": 0}
    mantissa = (mono_to_bits(mono(tq)) & 0x03FF).long()
    used = [bool((mantissa & (1 << bit)).any()) for bit in range(10)]
    effective = 10 - next((bit for bit, occupied in enumerate(used) if occupied), 10)
    output = {
        "n": int(tq.numel()),
        "effective_mantissa_bits": effective,
        "mantissa_low4": low4_occupancy(mantissa & 0xF),
    }
    if note:
        output["note"] = (
            "Raw IEEE FP16 mantissa occupancy after mono()'s cast. This detects source precision "
            "(including a BF16-derived m=7 population) but not the low nibble the radix rule "
            "rounds."
        )
    return output


def tq_rule_low4(tq: torch.Tensor, *, note: bool = False) -> dict:
    """Summarize low-nibble occupancy in the monotonic-uint16 space the rule rounds in."""

    from .radix_rules import mono

    if not tq.numel():
        return {"n": 0}
    output = low4_occupancy(mono(tq) & 0xF)
    if note:
        output["note"] = (
            "Low-nibble occupancy in monotonic-uint16 rule space. These are the exact bits "
            "consumed by round_mono; negative FP16 values are complemented by design."
        )
    return output


def _bin_ids(length: int, n_bins: int) -> tuple[torch.Tensor, int]:
    """Map ``[0, length)`` into at most ``n_bins`` contiguous non-empty bins."""

    bin_count = min(int(n_bins), length)
    return (
        (torch.arange(length) * bin_count // length).clamp(max=bin_count - 1),
        bin_count,
    )


def bin_by_position(
    values: torch.Tensor,
    n_bins: int,
    percentiles: tuple[str, ...] = ("p50", "p75", "p99", "p100"),
    positions: torch.Tensor | None = None,
) -> dict:
    """Aggregate a token-aligned series into contiguous position bins."""

    values_host = values.detach().to("cpu").float().flatten()
    length = int(values_host.numel())
    if length == 0:
        return {"n": 0}
    bin_ids, bin_count = _bin_ids(length, n_bins)
    if positions is None:
        positions_host = torch.arange(length, dtype=torch.float32)
    else:
        positions_host = positions.detach().to("cpu").float().flatten()
        if positions_host.numel() != length:
            raise ValueError(
                f"positions has {positions_host.numel()} entries but values has {length}"
            )
    counts = torch.zeros(bin_count).scatter_add(
        0, bin_ids, torch.ones(length)
    ).clamp(min=1)
    result = {
        "n": length,
        "positions": (
            torch.zeros(bin_count).scatter_add(0, bin_ids, positions_host) / counts
        ).tolist(),
        "mean": (
            torch.zeros(bin_count).scatter_add(0, bin_ids, values_host) / counts
        ).tolist(),
    }
    order = torch.argsort(
        bin_ids * (length + 1) + torch.argsort(torch.argsort(values_host))
    )
    sorted_values = values_host[order]
    starts = torch.zeros(bin_count, dtype=torch.long)
    starts[1:] = counts.long().cumsum(0)[:-1]
    integer_counts = counts.long()
    for name in percentiles:
        fraction = float(name[1:]) / 100.0
        rank = torch.round(
            torch.tensor(fraction) * (integer_counts - 1).clamp(min=0).float()
        ).long()
        result[name] = sorted_values[
            (starts + rank).clamp(0, length - 1)
        ].tolist()
    return result


# --------------------------------------------------------------------------------------------- #
# Bounded streaming accumulation (docs/qwen3_4b_dsa/approx_topk_serving_plan.md §9)
#
# Retaining every raw layer/position record is not viable for routine 32K evaluation: one 32K
# prompt across ~30 sparse layers is ~1M rows per field, and `torch.quantile` REFUSES any input
# above 2**24 elements ("quantile() input tensor is too large"). A few hundred eval prompts
# therefore used to fail at artifact-dump time, after the whole run, having already consumed
# tens of GB of host RAM.
#
# So metrics fold into a fixed-width histogram plus exact moments as they arrive. `n`, `mean`,
# `std`, `min`, and `max` stay EXACT for the whole population; percentiles come from the
# histogram and are exact whenever the spec has one bin per representable value (integer counts,
# the fields whose tails §10 reports on). `exact_percentiles` states which case a summary is.
QUANTILE_LIMIT = int(os.environ.get("DSA_APPROX_QUANTILE_LIMIT", str(8 << 20)))
HIST_BINS = int(os.environ.get("DSA_APPROX_HIST_BINS", "512"))
# Count metrics get one bin per value, so their percentiles stay EXACT: §16 step 10 sizes
# floor capacity off the count tails, where a coarse bin would mis-size the buffer.
COUNT_BINS = int(os.environ.get("DSA_APPROX_COUNT_BINS", "8192"))
SPARSE_HISTOGRAM_CAP = int(os.environ.get("DSA_APPROX_SPARSE_HIST_CAP", "128"))


@dataclasses.dataclass(frozen=True)
class DistSpec:
    """Fixed bin layout: bin ``b`` covers ``[low + b*width, low + (b+1)*width)``."""

    low: float
    width: float
    bins: int
    integral: bool = False

    @property
    def exact(self) -> bool:
        """True when every bin holds exactly one representable value."""

        return self.integral and self.width == 1.0


def integer_spec(low: int, high: int, max_bins: int | None = None) -> DistSpec:
    """Bin every integer in ``[low, high]``, widening bins only past ``max_bins``."""

    limit = max(1, HIST_BINS if max_bins is None else max_bins)
    span = max(1, int(high) - int(low) + 1)
    width = -(-span // limit)
    return DistSpec(float(low), float(width), -(-span // width), True)


def ratio_spec(bins: int | None = None) -> DistSpec:
    """Bin the unit interval, the range of every rate/recall/precision metric."""

    count = max(1, HIST_BINS if bins is None else bins)
    return DistSpec(0.0, 1.0 / count, count, False)


class BoundedDist:
    """Fixed-memory streaming distribution accumulator."""

    __slots__ = ("spec", "n", "_total", "_total_sq", "_min", "_max", "_counts", "_below", "_above")

    def __init__(self, spec: DistSpec) -> None:
        self.spec = spec
        self.n = 0
        self._total = 0.0
        self._total_sq = 0.0
        self._min = math.inf
        self._max = -math.inf
        self._counts = torch.zeros(spec.bins, dtype=torch.int64)
        self._below = 0
        self._above = 0

    def add(self, values: torch.Tensor) -> None:
        """Fold a block of samples in. Cost and memory are independent of the block size."""

        flat = values.detach().to("cpu").flatten().to(torch.float64)
        flat = flat[torch.isfinite(flat)]
        count = int(flat.numel())
        if not count:
            return
        self.n += count
        self._total += float(flat.sum())
        self._total_sq += float(flat.square().sum())
        self._min = min(self._min, float(flat.min()))
        self._max = max(self._max, float(flat.max()))
        raw = ((flat - self.spec.low) / self.spec.width).floor()
        # Out-of-range samples clamp into the edge bins, so record how many did: min/max stay
        # exact, but a clamped percentile would otherwise understate the tail silently.
        self._below += int((raw < 0).sum())
        self._above += int((raw >= self.spec.bins).sum())
        self._counts += torch.bincount(
            raw.clamp_(0, self.spec.bins - 1).to(torch.int64), minlength=self.spec.bins
        )

    def merge(self, other: "BoundedDist") -> None:
        """Absorb another accumulator over the same bin layout."""

        if other.spec != self.spec:
            raise ValueError(f"cannot merge {other.spec} into {self.spec}")
        if other.n == 0:
            return
        self.n += other.n
        self._total += other._total
        self._total_sq += other._total_sq
        self._min = min(self._min, other._min)
        self._max = max(self._max, other._max)
        self._counts += other._counts
        self._below += other._below
        self._above += other._above

    @property
    def maximum(self) -> float | None:
        return None if self.n == 0 else self._max

    def _quantile(self, fraction: float) -> float:
        if fraction <= 0.0:
            return self._min
        if fraction >= 1.0:
            return self._max
        # Rank of the requested quantile under 'lower' interpolation, then the first bin whose
        # cumulative count covers it. With one bin per value this reproduces torch.quantile's
        # lower interpolation exactly; otherwise it is the containing bin's left edge.
        target = int(fraction * (self.n - 1)) + 1
        cumulative = self._counts.cumsum(0)
        index = int(torch.searchsorted(cumulative, torch.tensor(target, dtype=cumulative.dtype)))
        value = self.spec.low + min(index, self.spec.bins - 1) * self.spec.width
        return min(max(value, self._min), self._max)

    def _histogram(self) -> dict:
        """Emit occupancy compactly: sparse pairs, or folded super-bins when too dense."""

        occupied = torch.nonzero(self._counts, as_tuple=False).flatten()
        if int(occupied.numel()) <= SPARSE_HISTOGRAM_CAP:
            return {
                "low": self.spec.low,
                "width": self.spec.width,
                "bins": self.spec.bins,
                "nonzero": [[int(b), int(self._counts[b])] for b in occupied.tolist()],
            }
        factor = -(-self.spec.bins // SPARSE_HISTOGRAM_CAP)
        padded = torch.zeros(factor * SPARSE_HISTOGRAM_CAP, dtype=torch.int64)
        padded[: self.spec.bins] = self._counts
        return {
            "low": self.spec.low,
            "width": self.spec.width * factor,
            "bins": SPARSE_HISTOGRAM_CAP,
            "counts": padded.view(SPARSE_HISTOGRAM_CAP, factor).sum(-1).tolist(),
            "folded_from_bins": self.spec.bins,
        }

    def summary(self) -> dict:
        """Return the §10 distribution schema for everything accumulated so far."""

        if self.n == 0:
            return {"n": 0}
        mean = self._total / self.n
        variance = max(self._total_sq / self.n - mean * mean, 0.0)
        output = {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self._min,
            "max": self._max,
            "percentiles": {name: self._quantile(fraction) for name, fraction in PCT},
            "exact_percentiles": self.spec.exact,
            "histogram": self._histogram(),
        }
        if self._below or self._above:
            output["out_of_range"] = {"below": self._below, "above": self._above}
        return output


def bounded_dist_values(x: torch.Tensor, spec: DistSpec | None = None) -> dict:
    """Summarize an already-materialized population without a size ceiling."""

    flat = x.detach().to("cpu").flatten().float()
    flat = flat[torch.isfinite(flat)]
    if not flat.numel():
        return {"n": 0}
    if spec is None:
        low, high = float(flat.min()), float(flat.max())
        integral = bool(x.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool))
        spec = (
            integer_spec(int(low), int(high))
            if integral
            else DistSpec(low, max((high - low) / HIST_BINS, torch.finfo(torch.float32).tiny), HIST_BINS)
        )
    accumulator = BoundedDist(spec)
    accumulator.add(flat)
    return accumulator.summary()
