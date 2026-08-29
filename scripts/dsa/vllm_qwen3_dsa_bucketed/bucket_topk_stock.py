"""Exact modulo-bucket selection driven by an injected stock top-k row operation."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from .bucket_topk_reference import BucketSelectionResult, _assert_tensor, bucket_lengths

StockTopK = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None]


@torch.no_grad()
def select_bucket_topk_stock(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    *,
    bucket_count: int,
    bucket_top_k: int,
    stock_topk: StockTopK,
) -> BucketSelectionResult:
    """Invoke ``stock_topk`` once per strided modulo bucket and remap its indices."""

    if logits.ndim != 2 or output.ndim != 2 or logits.shape[0] != output.shape[0]:
        raise ValueError(
            f"bucket selector expects logits/output [rows,width], got {tuple(logits.shape)} and {tuple(output.shape)}"
        )
    capacity = bucket_count * bucket_top_k
    if bucket_count <= 0 or bucket_top_k <= 0 or output.shape[1] != capacity:
        raise ValueError(
            f"invalid bucket geometry count={bucket_count} local_k={bucket_top_k} output_width={output.shape[1]}"
        )

    rows, key_count = logits.shape
    qpos = query_positions.reshape(-1).to(device=logits.device, dtype=torch.int64)
    if qpos.numel() != rows:
        raise ValueError(f"got {qpos.numel()} query positions for {rows} score rows")
    sequence_lengths = (qpos + 1).clamp(min=0, max=key_count)
    populations = bucket_lengths(sequence_lengths, bucket_count)
    chosen_per_bucket = populations.clamp(max=bucket_top_k)
    selected_count = chosen_per_bucket.sum(-1)
    effective_k = sequence_lengths.clamp(max=capacity)

    output.fill_(-1)
    for bucket in range(bucket_count):
        bucket_logits = logits[:, bucket::bucket_count]
        lengths = populations[:, bucket].clamp(max=bucket_logits.shape[1])
        scratch = torch.full(
            (rows, bucket_top_k),
            -1,
            dtype=output.dtype,
            device=output.device,
        )
        stock_topk(bucket_logits, lengths, scratch, bucket_top_k)
        valid = scratch >= 0
        mapped = scratch.to(torch.int64) * bucket_count + bucket
        valid &= mapped < key_count
        valid &= mapped <= qpos[:, None]
        output[:, bucket::bucket_count].copy_(torch.where(valid, mapped.to(output.dtype), -1))

    valid = output >= 0
    observed = valid.sum(-1)
    _assert_tensor(
        (observed == selected_count).all(),
        "stock-per-bucket count does not match the modulo bucket budget",
    )
    _assert_tensor(
        ~((~valid[:, :-1]) & valid[:, 1:]).any(),
        "stock-per-bucket output is not valid-prefix/-1-suffix",
    )
    sentinel = torch.full_like(output, key_count)
    ordered = torch.where(valid, output, sentinel).sort(-1).values
    _assert_tensor(
        ~((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != key_count)).any(),
        "stock-per-bucket output contains duplicate request-local positions",
    )
    return BucketSelectionResult(selected_count, effective_k, chosen_per_bucket)


@torch.no_grad()
def select_bucket_topk_stock_batched(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    *,
    bucket_count: int,
    bucket_top_k: int,
    stock_topk: StockTopK,
) -> BucketSelectionResult:
    """Materialize all modulo buckets as rows and invoke ``stock_topk`` once."""

    if logits.ndim != 2 or output.ndim != 2 or logits.shape[0] != output.shape[0]:
        raise ValueError(
            f"bucket selector expects logits/output [rows,width], got {tuple(logits.shape)} and {tuple(output.shape)}"
        )
    capacity = bucket_count * bucket_top_k
    if bucket_count <= 0 or bucket_top_k <= 0 or output.shape[1] != capacity:
        raise ValueError(
            f"invalid bucket geometry count={bucket_count} local_k={bucket_top_k} output_width={output.shape[1]}"
        )

    rows, key_count = logits.shape
    qpos = query_positions.reshape(-1).to(device=logits.device, dtype=torch.int64)
    if qpos.numel() != rows:
        raise ValueError(f"got {qpos.numel()} query positions for {rows} score rows")
    sequence_lengths = (qpos + 1).clamp(min=0, max=key_count)
    populations = bucket_lengths(sequence_lengths, bucket_count)
    chosen_per_bucket = populations.clamp(max=bucket_top_k)
    selected_count = chosen_per_bucket.sum(-1)
    effective_k = sequence_lengths.clamp(max=capacity)

    groups = (key_count + bucket_count - 1) // bucket_count
    padded_width = groups * bucket_count
    scores = logits if padded_width == key_count else F.pad(logits, (0, padded_width - key_count), value=float("-inf"))
    bucket_logits = scores.view(rows, groups, bucket_count).transpose(1, 2).reshape(rows * bucket_count, groups)
    lengths = populations.reshape(-1).clamp(max=groups)
    scratch = torch.full(
        (rows * bucket_count, bucket_top_k),
        -1,
        dtype=output.dtype,
        device=output.device,
    )
    stock_topk(bucket_logits, lengths, scratch, bucket_top_k)

    local = scratch.view(rows, bucket_count, bucket_top_k)
    buckets = torch.arange(bucket_count, device=output.device, dtype=torch.int64)[None, :, None]
    mapped = local.to(torch.int64) * bucket_count + buckets
    valid = local >= 0
    valid &= mapped < key_count
    valid &= mapped <= qpos[:, None, None]
    mapped = mapped.transpose(1, 2).reshape(rows, capacity)
    valid = valid.transpose(1, 2).reshape(rows, capacity)
    output.copy_(torch.where(valid, mapped.to(output.dtype), -1))

    observed = valid.sum(-1)
    _assert_tensor(
        (observed == selected_count).all(),
        "batched-stock bucket count does not match the modulo bucket budget",
    )
    _assert_tensor(
        ~((~valid[:, :-1]) & valid[:, 1:]).any(),
        "batched-stock bucket output is not valid-prefix/-1-suffix",
    )
    sentinel = torch.full_like(output, key_count)
    ordered = torch.where(valid, output, sentinel).sort(-1).values
    _assert_tensor(
        ~((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != key_count)).any(),
        "batched-stock bucket output contains duplicate request-local positions",
    )
    return BucketSelectionResult(selected_count, effective_k, chosen_per_bucket)
