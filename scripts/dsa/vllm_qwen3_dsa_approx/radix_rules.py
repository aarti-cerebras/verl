"""FP16 partial-radix threshold rules used by approximate Qwen3 DSA serving.

Ported from ``dsa-csx/glm_52/study_core/rules.py``. The source implementation
is the behavioral oracle; this local copy avoids a serving-time dependency on a
neighboring checkout.
"""

from __future__ import annotations

import dataclasses
import os

import torch

ARM_SCHEME = {
    "ceil": "ceil",
    "exact_ge": "exact",
    "floor": "floor",
    "midpoint": "midpoint",
}
THRESHOLD_ARMS = tuple(ARM_SCHEME)


DEBUG = os.environ.get("GLM_SELECT_DEBUG", "0") == "1"


def mono(f16: torch.Tensor) -> torch.Tensor:
    """Map FP16 values to monotonic uint16 keys stored as int32.

    The mapping preserves the total order of finite FP16 bit patterns. It deliberately orders
    ``-0`` immediately below ``+0`` because the radix operates on representations. NaNs round-trip
    through the inverse, but are outside the scorer contract and have no numeric ordering.
    """

    u = f16.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    mask = ((u >> 15) & 1) * 0xFFFF
    return (u ^ (mask | 0x8000)) & 0xFFFF


def mono_to_bits(m: torch.Tensor) -> torch.Tensor:
    """Invert :func:`mono` into the original FP16 bit pattern."""

    m = m.to(torch.int64)
    mask = ((m >> 15) & 1) * 0xFFFF
    return (m ^ (0x8000 | (~mask & 0xFFFF))) & 0xFFFF


def mono_to_f16(m: torch.Tensor) -> torch.Tensor:
    """Invert monotonic uint16 keys into FP16 values represented as float32."""

    return mono_to_bits(m).to(torch.int16).view(torch.float16).float()


def round_mono(m: torch.Tensor, scheme: str) -> torch.Tensor:
    """Reconstruct a threshold after dropping the radix key's low four bits."""

    if scheme == "exact":
        return m
    if scheme == "floor":
        return m & ~0xF
    if scheme == "midpoint":
        return (m & ~0xF) | 8
    if scheme == "ceil":
        return torch.clamp((m & ~0xF) + 16, max=0xFFFF)
    raise ValueError(f"unknown rounding scheme {scheme!r} (want exact|floor|midpoint|ceil)")


def row_threshold(arm: str, mono_tq: torch.Tensor) -> torch.Tensor:
    """Return an arm's per-row threshold from the exact monotonic ``Tq`` key."""

    try:
        scheme = ARM_SCHEME[arm]
    except KeyError:
        raise ValueError(
            f"unknown threshold arm {arm!r} (want one of {tuple(ARM_SCHEME)}; note "
            f"'topk' is the exact-set baseline and has no threshold)"
        ) from None
    return round_mono(mono_tq, scheme)


def member_ge(
    mono_s: torch.Tensor, valid: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    """Return which valid keys meet each row's monotonic threshold."""

    return (mono_s >= threshold[:, None]) & valid


def count_ge(
    mono_s: torch.Tensor, valid: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    """Count valid keys meeting each row's monotonic threshold."""

    return member_ge(mono_s, valid, threshold).sum(-1)


@dataclasses.dataclass
class ScoreView:
    """Shared derivation of one block of lightning-indexer score rows."""

    s_masked: torch.Tensor
    valid: torch.Tensor
    n_valid: torch.Tensor
    qpos: torch.Tensor
    tq: torch.Tensor
    keff: torch.Tensor
    topi: torch.Tensor
    mono_s: torch.Tensor
    mono_tq: torch.Tensor
    B: int
    mb: int

    @property
    def degenerate(self) -> torch.Tensor:
        """Rows where every causal key must be selected because the prefix has at most k keys."""

        return self.n_valid <= self.keff


def check_f16_survives(
    s_masked: torch.Tensor, valid: torch.Tensor, where: str = ""
) -> None:
    """Reject valid scores that become non-finite on the FP16 radix grid."""

    values = s_masked[valid]
    if bool(torch.isfinite(values.to(torch.float16)).all()):
        return
    max_abs = float(values.abs().max()) if values.numel() else float("nan")
    raise AssertionError(
        f"indexer score{' at ' + where if where else ''} does not survive mono()'s f16 cast: "
        f"|score| reaches {max_abs:.3e} against f16's 65504 limit. Saturated scores collapse "
        f"into one monotonic bucket, so the threshold compare sees ties that are not in the data "
        f"and every arm's count and selection is wrong in the same direction."
    )


def exact_tq(
    s_masked: torch.Tensor,
    valid: torch.Tensor,
    k: int,
    return_indices: bool = False,
):
    """Return each row's k-th largest valid score and effective k."""

    width = s_masked.shape[-1]
    keff = torch.clamp(valid.sum(-1), max=k)
    top = s_masked.topk(min(k, width), dim=-1)
    tq = top.values.gather(1, (keff - 1).clamp(min=0)[:, None]).squeeze(1)
    return (tq, keff, top.indices) if return_indices else (tq, keff)


def score_view(
    idx_scores: torch.Tensor,
    qpos: torch.Tensor,
    k: int,
    where: str = "",
    *,
    debug: bool | None = None,
) -> ScoreView:
    """Build a causal score view from ``[batch, query, key]`` scores and query positions."""

    batch, query_rows, key_count = idx_scores.shape
    device = idx_scores.device
    scores = idx_scores.reshape(batch * query_rows, key_count)
    positions = qpos.expand(batch, query_rows).reshape(batch * query_rows)
    valid = torch.arange(key_count, device=device)[None, :] <= positions[:, None]
    s_masked = torch.where(
        valid,
        scores.float(),
        scores.new_full((), float("-inf"), dtype=torch.float32),
    )
    debug_enabled = DEBUG if debug is None else debug
    if debug_enabled:
        check_f16_survives(s_masked, valid, where)
    tq, keff, topi = exact_tq(s_masked, valid, k, return_indices=True)
    return ScoreView(
        s_masked=s_masked,
        valid=valid,
        n_valid=valid.sum(-1),
        qpos=positions,
        tq=tq,
        keff=keff,
        topi=topi,
        mono_s=mono(s_masked),
        mono_tq=mono(tq),
        B=batch,
        mb=query_rows,
    )


def count_for_arm(view: ScoreView, arm: str) -> torch.Tensor:
    """Count an arm's true selection over each row's whole causal prefix."""

    count = count_ge(view.mono_s, view.valid, row_threshold(arm, view.mono_tq))
    return torch.where(view.degenerate, view.keff, count)


def delta_k_all(
    view: ScoreView, arms: tuple[str, ...] = THRESHOLD_ARMS
) -> dict[str, torch.Tensor]:
    """Return ``selected_count - effective_k`` for every requested threshold arm."""

    return {arm: count_for_arm(view, arm) - view.keff for arm in arms}
