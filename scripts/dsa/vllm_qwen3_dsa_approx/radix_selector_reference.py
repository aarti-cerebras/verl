"""Reference approximate selector ported from the dsa-csx emission rules.

This backend deliberately uses ``torch.topk`` to obtain the exact k-th
threshold and to order the fixed-capacity output window. It changes the set
that reaches attention, but it is not a selector-speed implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import radix_rules as rule


SELECTORS = ("topk", "exact_ge", "radix_floor", "radix_midpoint", "radix_ceil")


def normalize_arm(selector: str) -> str:
    if selector == "topk":
        return selector
    arm = selector.removeprefix("radix_")
    if arm not in rule.ARM_SCHEME:
        raise ValueError(f"unknown DSA selector {selector!r}; expected one of {SELECTORS}")
    return arm


def _assert_tensor(condition: torch.Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


@dataclass(frozen=True)
class SelectionResult:
    selected_count: torch.Tensor
    effective_k: torch.Tensor
    exact_indices: torch.Tensor
    threshold: torch.Tensor
    rescued: torch.Tensor


@torch.no_grad()
def select_prefix_reference(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    rule_k: int,
    output: torch.Tensor,
    selector: str,
) -> SelectionResult:
    """Write request-local indices as a valid prefix followed by ``-1``.

    ``logits`` is ``[rows, keys]`` and each row is valid through its inclusive
    request-local ``query_positions`` value.
    """

    if logits.ndim != 2 or output.ndim != 2 or logits.shape[0] != output.shape[0]:
        raise ValueError(
            f"selector expects logits/output [rows,width], got {tuple(logits.shape)} and "
            f"{tuple(output.shape)}"
        )
    rows, key_count = logits.shape
    capacity = output.shape[1]
    if rows == 0:
        empty = torch.empty(0, dtype=torch.int64, device=logits.device)
        return SelectionResult(empty, empty, output.long(), empty.float(), empty.bool())
    if rule_k <= 0 or capacity < min(rule_k, key_count):
        raise ValueError(
            f"invalid selector geometry: rule_k={rule_k}, capacity={capacity}, keys={key_count}"
        )

    qpos = query_positions.reshape(-1).to(device=logits.device, dtype=torch.int64)
    if qpos.numel() != rows:
        raise ValueError(f"got {qpos.numel()} query positions for {rows} score rows")
    # vLLM pads FULL CUDA graph batches to a captured batch size by setting the unused sequence
    # lengths to zero. The decode hook consequently presents those rows with qpos=-1. They are not
    # requests and must stay empty; in particular, the ordinary empty-selection rescue must not
    # manufacture key 0 for them.
    active = qpos >= 0
    valid = torch.arange(key_count, device=logits.device)[None, :] <= qpos[:, None]
    scores = logits.float().masked_fill(~valid, float("-inf"))
    # Boolean indexing materializes a data-dependent-length tensor. CUDA graph capture rejects that
    # operation even though the final reduction is scalar. Keep the check at the static logits shape
    # and mask invalid positions after the FP16 cast; invalid positions intentionally hold -inf.
    finite_on_grid = torch.isfinite(scores.to(torch.float16)) | ~valid
    _assert_tensor(
        finite_on_grid.all(),
        "approximate DSA selector received a valid score outside finite FP16 range",
    )

    effective_k = valid.sum(-1).clamp(max=rule_k)
    exact_width = min(rule_k, key_count)
    exact = scores.topk(exact_width, dim=-1)
    threshold = exact.values.gather(
        -1, (effective_k - 1).clamp(min=0)[:, None]
    ).squeeze(-1)

    arm = normalize_arm(selector)
    output.fill_(-1)
    if arm == "topk":
        keep = torch.arange(exact_width, device=logits.device)[None, :] < effective_k[:, None]
        output[:, :exact_width].copy_(
            torch.where(keep, exact.indices.to(torch.int32), -1)
        )
        return SelectionResult(
            effective_k,
            effective_k,
            exact.indices,
            threshold,
            torch.zeros(rows, dtype=torch.bool, device=logits.device),
        )

    window = min(capacity, key_count)
    candidates = scores.topk(window, dim=-1)
    finite = torch.isfinite(candidates.values)
    mono_threshold = rule.row_threshold(arm, rule.mono(threshold))
    keep = rule.member_ge(rule.mono(candidates.values), finite, mono_threshold)
    degenerate = valid.sum(-1) <= effective_k
    keep = torch.where(degenerate[:, None], finite, keep)

    selected_count = rule.member_ge(rule.mono(scores), valid, mono_threshold).sum(-1)
    selected_count = torch.where(degenerate, effective_k, selected_count)
    rescued = active & ~keep.any(-1)
    keep[:, 0] = keep[:, 0] | rescued
    selected_count = torch.where(rescued, torch.ones_like(selected_count), selected_count)
    _assert_tensor(
        (selected_count <= capacity).all(),
        "approximate DSA selection exceeds index_topk capacity",
    )
    output[:, :window].copy_(
        torch.where(keep, candidates.indices.to(torch.int32), -1)
    )
    return SelectionResult(
        selected_count,
        effective_k,
        exact.indices,
        rule.mono_to_f16(mono_threshold),
        rescued,
    )
