import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_topk_reference import (
    bucket_lengths,
    select_bucket_topk_reference,
)


def _sets(output: torch.Tensor) -> list[set[int]]:
    return [set(row[row >= 0].tolist()) for row in output]


def test_hand_constructed_rank_major_selection() -> None:
    logits = torch.tensor([[0.0, 10.0, 9.0, 8.0, 7.0, 6.0]])
    output = torch.empty(1, 4, dtype=torch.int32)
    result = select_bucket_topk_reference(
        logits,
        torch.tensor([5]),
        output,
        bucket_count=2,
        bucket_top_k=2,
    )

    assert output.tolist() == [[2, 1, 4, 3]]
    assert result.selected_count.tolist() == [4]
    assert result.bucket_counts.tolist() == [[2, 2]]


def test_short_prefix_is_dense_and_valid_prefix() -> None:
    logits = torch.tensor([[0.0, 1.0, 9.0, 100.0, 100.0]])
    output = torch.empty(1, 4, dtype=torch.int32)
    result = select_bucket_topk_reference(
        logits,
        torch.tensor([2]),
        output,
        bucket_count=2,
        bucket_top_k=2,
    )

    assert output.tolist() == [[2, 1, 0, -1]]
    assert _sets(output) == [{0, 1, 2}]
    assert result.selected_count.tolist() == [3]


@pytest.mark.parametrize("length", range(0, 18))
def test_bucket_lengths_partition_contiguous_prefix(length: int) -> None:
    observed = bucket_lengths(torch.tensor([length]), 5)[0]
    expected = torch.tensor([sum(position % 5 == bucket for position in range(length)) for bucket in range(5)])
    assert torch.equal(observed, expected)


def test_nondivisible_width_and_inactive_padding_rows() -> None:
    torch.manual_seed(7)
    logits = torch.randn(3, 11)
    qpos = torch.tensor([10, 6, -1])
    output = torch.empty(3, 6, dtype=torch.int32)
    result = select_bucket_topk_reference(
        logits,
        qpos,
        output,
        bucket_count=3,
        bucket_top_k=2,
    )

    assert result.selected_count.tolist() == [6, 6, 0]
    assert bool((output[2] == -1).all())
    valid = output >= 0
    assert not bool(((~valid[:, :-1]) & valid[:, 1:]).any())
    assert bool(((output == -1) | (output <= qpos[:, None])).all())


def test_boundary_ties_satisfy_membership_and_cardinality() -> None:
    # Bucket 0 positions have scores 5, 4, 4. Either threshold-equal position is valid at k=2.
    logits = torch.tensor([[5.0, 0.0, 4.0, 0.0, 4.0, 0.0]])
    output = torch.empty(1, 4, dtype=torch.int32)
    select_bucket_topk_reference(
        logits,
        torch.tensor([5]),
        output,
        bucket_count=2,
        bucket_top_k=2,
    )

    bucket_zero = {position for position in output[0].tolist() if position >= 0 and position % 2 == 0}
    assert 0 in bucket_zero
    assert len(bucket_zero) == 2
    assert bucket_zero <= {0, 2, 4}


def test_nonfinite_causal_score_fails_closed() -> None:
    logits = torch.tensor([[1.0, float("nan"), 2.0]])
    output = torch.empty(1, 2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="non-finite causal"):
        select_bucket_topk_reference(
            logits,
            torch.tensor([2]),
            output,
            bucket_count=2,
            bucket_top_k=1,
        )
