"""Tensor reference for exact local top-k over request-local modulo-position buckets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _assert_tensor(condition: torch.Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


@dataclass(frozen=True)
class BucketSelectionResult:
    selected_count: torch.Tensor
    effective_k: torch.Tensor
    bucket_counts: torch.Tensor


def bucket_lengths(
    sequence_lengths: torch.Tensor,
    bucket_count: int,
) -> torch.Tensor:
    """Return causal population per modulo bucket as ``[..., bucket_count]``."""

    lengths = sequence_lengths.to(torch.int64).clamp(min=0)
    buckets = torch.arange(bucket_count, device=lengths.device, dtype=torch.int64)
    return ((lengths[..., None] + bucket_count - 1 - buckets) // bucket_count).clamp(min=0)


@torch.no_grad()
def select_bucket_topk_reference(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    *,
    bucket_count: int,
    bucket_top_k: int,
) -> BucketSelectionResult:
    """Write exact per-bucket top-k request-local positions into ``output``.

    ``logits`` is ``[rows, keys]``. Each row is causal through its inclusive request-local
    ``query_positions`` value. Output is rank-major across buckets and therefore a valid prefix
    followed by ``-1`` for every contiguous causal prefix.
    """

    if logits.ndim != 2 or output.ndim != 2 or logits.shape[0] != output.shape[0]:
        raise ValueError(
            f"bucket selector expects logits/output [rows,width], got {tuple(logits.shape)} and {tuple(output.shape)}"
        )
    if bucket_count <= 0 or bucket_top_k <= 0:
        raise ValueError(f"bucket_count and bucket_top_k must be positive, got {bucket_count}/{bucket_top_k}")
    capacity = bucket_count * bucket_top_k
    if output.shape[1] != capacity:
        raise ValueError(f"bucket output width must be bucket_count * bucket_top_k = {capacity}, got {output.shape[1]}")

    rows, key_count = logits.shape
    qpos = query_positions.reshape(-1).to(device=logits.device, dtype=torch.int64)
    if qpos.numel() != rows:
        raise ValueError(f"got {qpos.numel()} query positions for {rows} score rows")

    sequence_lengths = (qpos + 1).clamp(min=0, max=key_count)
    populations = bucket_lengths(sequence_lengths, bucket_count)
    chosen_per_bucket = populations.clamp(max=bucket_top_k)
    selected_count = chosen_per_bucket.sum(-1)
    effective_k = sequence_lengths.clamp(max=capacity)
    _assert_tensor(
        (selected_count == effective_k).all(),
        "modulo bucket populations do not sum to the fixed total-k contract",
    )

    output.fill_(-1)
    if rows == 0 or key_count == 0:
        return BucketSelectionResult(selected_count, effective_k, chosen_per_bucket)

    positions = torch.arange(key_count, device=logits.device, dtype=torch.int64)
    causal = positions[None, :] <= qpos[:, None]
    _assert_tensor(
        (torch.isfinite(logits) | ~causal).all(),
        "bucket selector received a non-finite causal indexer score",
    )
    scores = logits.float().masked_fill(~causal, float("-inf"))

    groups = (key_count + bucket_count - 1) // bucket_count
    padded_width = groups * bucket_count
    if padded_width != key_count:
        scores = F.pad(scores, (0, padded_width - key_count), value=float("-inf"))
    bucket_scores = scores.view(rows, groups, bucket_count).transpose(1, 2)

    local_width = min(bucket_top_k, groups)
    top = bucket_scores.topk(local_width, dim=-1)
    bucket_ids = torch.arange(bucket_count, device=logits.device, dtype=torch.int64)[None, :, None]
    selected_positions = top.indices.to(torch.int64) * bucket_count + bucket_ids
    valid = torch.isfinite(top.values) & (selected_positions < key_count) & (selected_positions <= qpos[:, None, None])

    # Rank-major flattening gives valid-prefix/-1-suffix because a contiguous modulo partition has
    # bucket populations differing by at most one, with the lower bucket ids receiving the extra.
    selected_positions = selected_positions.transpose(1, 2).reshape(rows, -1)
    valid = valid.transpose(1, 2).reshape(rows, -1)
    width = selected_positions.shape[1]
    output[:, :width].copy_(torch.where(valid, selected_positions.to(torch.int32), -1))

    observed = (output >= 0).sum(-1)
    _assert_tensor(
        (observed == selected_count).all(),
        "bucket selector count does not match emitted request-local positions",
    )
    return BucketSelectionResult(selected_count, effective_k, chosen_per_bucket)
