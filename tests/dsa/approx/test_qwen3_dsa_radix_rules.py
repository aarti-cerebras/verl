import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_approx import radix_rules as rule


def test_monotonic_fp16_mapping_round_trips_and_orders() -> None:
    values = torch.tensor(
        [-65504.0, -3.0, -0.0, 0.0, 0.5, 12.0, 65504.0], dtype=torch.float16
    )
    keys = rule.mono(values)
    assert bool((keys[1:] > keys[:-1]).all())
    torch.testing.assert_close(rule.mono_to_f16(keys), values.float(), rtol=0, atol=0)


def test_low_nibble_threshold_rounding() -> None:
    key = torch.tensor([0x8123], dtype=torch.int32)
    assert rule.round_mono(key, "floor").item() == 0x8120
    assert rule.round_mono(key, "midpoint").item() == 0x8128
    assert rule.round_mono(key, "ceil").item() == 0x8130
    assert rule.round_mono(key, "exact").item() == 0x8123


def test_arm_count_containment() -> None:
    scores = torch.tensor(
        [[[0.9, 0.8, 0.7, 0.6, 0.5], [0.9, 0.8, 0.7, 0.6, 0.5]]]
    )
    view = rule.score_view(scores, torch.tensor([1, 4]), k=3)
    counts = {arm: rule.count_for_arm(view, arm) for arm in rule.THRESHOLD_ARMS}
    assert bool((counts["ceil"] <= view.keff).all())
    assert bool((counts["exact_ge"] >= view.keff).all())
    assert bool((counts["floor"] >= view.keff).all())
    assert bool((counts["midpoint"] >= counts["ceil"]).all())
    assert bool((counts["midpoint"] <= counts["floor"]).all())


def test_signed_zero_orders_below_positive_zero() -> None:
    # mono() operates on REPRESENTATIONS, so -0 sits immediately below +0 by design. The radix
    # compares these keys, so the ordering has to be total even where FP16 equality is not.
    keys = rule.mono(torch.tensor([-0.0, 0.0], dtype=torch.float16))
    assert int(keys[0]) == int(keys[1]) - 1


def test_equal_scores_share_one_monotonic_key() -> None:
    values = torch.tensor([2.5, 2.5, 2.5], dtype=torch.float16)
    keys = rule.mono(values)
    assert len(set(keys.tolist())) == 1


def test_exact_ge_captures_every_key_tied_at_tq() -> None:
    # Four keys tie at the k-th score. exact_ge is the tie-aware control, so it must take all of
    # them; the fixed-size baseline necessarily cuts the tie at k.
    scores = torch.tensor([[[0.9, 0.5, 0.5, 0.5, 0.5]]])
    view = rule.score_view(scores, torch.tensor([4]), k=2)
    assert int(rule.count_for_arm(view, "exact_ge")) == 5
    assert int(view.keff) == 2


def test_degenerate_rows_select_their_whole_causal_prefix() -> None:
    # Row 0 sees 2 keys with k=4, so every arm must take both rather than apply a threshold.
    scores = torch.tensor([[[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]]])
    view = rule.score_view(scores, torch.tensor([1, 3]), k=4)
    assert view.degenerate.tolist() == [True, True]
    for arm in rule.THRESHOLD_ARMS:
        assert rule.count_for_arm(view, arm).tolist() == [2, 4]


def test_scores_outside_fp16_range_are_rejected() -> None:
    # A saturating score collapses into one monotonic bucket, which invents ties at the threshold
    # and biases every arm's count in the same direction -- so it must fail rather than degrade.
    scores = torch.tensor([[1.0, 1e6, 3.0]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    with pytest.raises(AssertionError, match="does not survive mono"):
        rule.check_f16_survives(scores, valid, where="unit-test")
    with pytest.raises(AssertionError, match="does not survive mono"):
        rule.score_view(scores[None], torch.tensor([2]), k=2, debug=True)


def test_finite_edge_of_the_fp16_grid_is_accepted() -> None:
    scores = torch.tensor([[65504.0, -65504.0, 0.0]])
    rule.check_f16_survives(scores, torch.ones_like(scores, dtype=torch.bool))
    view = rule.score_view(scores[None], torch.tensor([2]), k=2, debug=True)
    assert int(rule.count_for_arm(view, "floor")) >= int(view.keff)


def test_rule_evaluation_is_deterministic() -> None:
    torch.manual_seed(19)
    scores = torch.randn(1, 6, 24)
    qpos = torch.arange(6) * 4 + 3
    first = rule.delta_k_all(rule.score_view(scores, qpos, k=8))
    second = rule.delta_k_all(rule.score_view(scores, qpos, k=8))
    for arm in rule.THRESHOLD_ARMS:
        assert torch.equal(first[arm], second[arm])
