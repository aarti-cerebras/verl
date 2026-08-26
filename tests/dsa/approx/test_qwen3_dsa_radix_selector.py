import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_approx.radix_selector_reference import (
    select_prefix_reference,
)


@pytest.mark.parametrize(
    "selector",
    ["topk", "exact_ge", "radix_floor", "radix_midpoint", "radix_ceil"],
)
def test_reference_selector_emits_valid_prefix(selector: str) -> None:
    torch.manual_seed(7)
    logits = torch.randn(4, 32)
    qpos = torch.tensor([3, 9, 17, 31])
    output = torch.empty(4, 16, dtype=torch.int32)
    result = select_prefix_reference(logits, qpos, 8, output, selector)

    valid = output >= 0
    assert not bool(((~valid[:, :-1]) & valid[:, 1:]).any())
    assert torch.equal(valid.sum(-1), result.selected_count)
    assert bool(((output == -1) | ((output >= 0) & (output <= qpos[:, None]))).all())
    for row in range(output.shape[0]):
        selected = output[row][valid[row]]
        assert selected.unique().numel() == selected.numel()


def test_selector_set_invariants() -> None:
    torch.manual_seed(11)
    logits = torch.randn(8, 64)
    qpos = torch.full((8,), 63)
    exact_output = torch.empty(8, 32, dtype=torch.int32)
    exact = select_prefix_reference(logits, qpos, 16, exact_output, "topk")
    exact_sets = [set(row[: int(count)].tolist()) for row, count in zip(exact_output, exact.selected_count)]

    for selector, relation in (("radix_ceil", "subset"), ("radix_floor", "superset"), ("exact_ge", "superset")):
        output = torch.empty_like(exact_output)
        selected = select_prefix_reference(logits, qpos, 16, output, selector)
        for row, count, exact_set in zip(output, selected.selected_count, exact_sets):
            approximate_set = set(row[: int(count)].tolist())
            if relation == "subset":
                assert approximate_set <= exact_set
            else:
                assert approximate_set >= exact_set


def test_capacity_saturation_fails_closed() -> None:
    # All scores share the same FP16 radix bucket, so floor wants the complete row.
    logits = torch.ones(1, 64)
    output = torch.empty(1, 8, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="exceeds index_topk capacity"):
        select_prefix_reference(logits, torch.tensor([63]), 8, output, "radix_floor")


@pytest.mark.parametrize(
    "selector",
    ["topk", "exact_ge", "radix_floor", "radix_midpoint", "radix_ceil"],
)
def test_short_prefix_rows_take_every_causal_key(selector: str) -> None:
    # Row 0 has 3 causal keys against k=8. The degenerate rule takes the whole prefix, so no arm
    # may thin an already-too-small row -- least of all the under-capturing ceil.
    torch.manual_seed(13)
    logits = torch.randn(2, 16)
    qpos = torch.tensor([2, 15])
    output = torch.empty(2, 8, dtype=torch.int32)
    result = select_prefix_reference(logits, qpos, 8, output, selector)

    assert int(result.selected_count[0]) == 3
    assert set(output[0][output[0] >= 0].tolist()) == {0, 1, 2}
    assert int(result.effective_k[0]) == 3


def test_scores_outside_fp16_range_fail_closed() -> None:
    logits = torch.tensor([[1.0, 70000.0, 3.0, 4.0]])
    output = torch.empty(1, 2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="outside finite FP16 range"):
        select_prefix_reference(logits, torch.tensor([3]), 2, output, "radix_floor")


def test_noncausal_fp16_overflow_is_masked_without_dynamic_indexing() -> None:
    # The graph-safe finiteness check stays at the static logits shape. Values beyond the causal
    # prefix are irrelevant even when they overflow FP16, while the two valid values remain usable.
    logits = torch.tensor([[0.1, 0.2, 70000.0, -70000.0]])
    output = torch.empty(1, 2, dtype=torch.int32)
    result = select_prefix_reference(logits, torch.tensor([1]), 2, output, "radix_floor")
    assert int(result.selected_count[0]) == 2
    assert sorted(output[0].tolist()) == [0, 1]


def test_noncausal_scores_are_ignored_not_selected() -> None:
    # The largest scores sit beyond the query position; they must never reach the output.
    logits = torch.tensor([[0.1, 0.2, 9.0, 9.0]])
    output = torch.empty(1, 2, dtype=torch.int32)
    result = select_prefix_reference(logits, torch.tensor([1]), 2, output, "radix_floor")
    assert sorted(output[0][output[0] >= 0].tolist()) == [0, 1]
    assert int(result.selected_count[0]) == 2


@pytest.mark.parametrize(
    "selector",
    ["topk", "exact_ge", "radix_floor", "radix_midpoint", "radix_ceil"],
)
def test_repeated_selection_is_bit_identical(selector: str) -> None:
    torch.manual_seed(17)
    logits = torch.randn(6, 48)
    qpos = torch.randint(8, 48, (6,))
    first = torch.empty(6, 16, dtype=torch.int32)
    second = torch.empty(6, 16, dtype=torch.int32)
    left = select_prefix_reference(logits, qpos, 8, first, selector)
    right = select_prefix_reference(logits, qpos, 8, second, selector)
    assert torch.equal(first, second)
    assert torch.equal(left.selected_count, right.selected_count)
    assert torch.equal(left.threshold, right.threshold)


def test_rescue_keeps_the_best_key_when_ceil_clears_the_row() -> None:
    # Every score shares one FP16 bucket, so ceil's threshold lands above all of them. The row is
    # rescued to its single best key rather than emitted empty.
    logits = torch.ones(1, 8)
    output = torch.empty(1, 4, dtype=torch.int32)
    result = select_prefix_reference(logits, torch.tensor([7]), 4, output, "radix_ceil")
    assert bool(result.rescued[0])
    assert int(result.selected_count[0]) == 1
    assert int(output[0, 0]) >= 0 and int(output[0, 1]) == -1
