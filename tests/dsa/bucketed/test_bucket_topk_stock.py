import torch

from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_topk_reference import (
    select_bucket_topk_reference,
)
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_topk_stock import (
    select_bucket_topk_stock,
)


def _torch_stock_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
) -> None:
    output.fill_(-1)
    if logits.shape[1] == 0:
        return
    valid = torch.arange(logits.shape[1])[None, :] < lengths[:, None]
    scores = logits.masked_fill(~valid, float("-inf"))
    width = min(top_k, logits.shape[1])
    top = scores.topk(width, dim=-1)
    keep = torch.arange(width)[None, :] < lengths.clamp(max=top_k)[:, None]
    output[:, :width].copy_(torch.where(keep, top.indices.to(torch.int32), -1))


def _sets(output: torch.Tensor) -> list[set[int]]:
    return [set(row[row >= 0].tolist()) for row in output]


def test_stock_callback_matches_reference_sets() -> None:
    torch.manual_seed(13)
    logits = torch.randn(8, 37)
    # Make boundaries unique so exact set equality, rather than tie validity, is the right gate.
    logits += torch.arange(37, dtype=torch.float32)[None, :] * 1e-5
    qpos = torch.tensor([0, 3, 7, 11, 19, 28, 36, -1])
    reference = torch.empty(8, 12, dtype=torch.int32)
    stock = torch.empty_like(reference)

    left = select_bucket_topk_reference(
        logits,
        qpos,
        reference,
        bucket_count=4,
        bucket_top_k=3,
    )
    right = select_bucket_topk_stock(
        logits,
        qpos,
        stock,
        bucket_count=4,
        bucket_top_k=3,
        stock_topk=_torch_stock_topk,
    )

    assert _sets(stock) == _sets(reference)
    assert torch.equal(right.selected_count, left.selected_count)
    assert torch.equal(right.bucket_counts, left.bucket_counts)


def test_stock_callback_receives_strided_bucket_views_and_local_lengths() -> None:
    logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    output = torch.empty(2, 6, dtype=torch.int32)
    seen: list[tuple[int, list[int]]] = []

    def recording_stock(scores, lengths, target, top_k):
        seen.append((scores.stride(1), lengths.tolist()))
        _torch_stock_topk(scores, lengths, target, top_k)

    select_bucket_topk_stock(
        logits,
        torch.tensor([9, 4]),
        output,
        bucket_count=3,
        bucket_top_k=2,
        stock_topk=recording_stock,
    )

    assert seen == [(3, [4, 2]), (3, [3, 2]), (3, [3, 1])]
    assert [len(row) for row in _sets(output)] == [6, 5]
