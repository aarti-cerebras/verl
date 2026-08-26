"""Parity against the dsa-csx sources the approximate selector was ported from.

The plan (§5) requires the port to have no ABSOLUTE dependency on the neighbouring checkout: the
serving artifacts and PRs must reproduce without it. So this module skips when dsa-csx is not
present, and is the only place that reaches for it. Point ``DSA_CSX_ROOT`` at a checkout to run it.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_approx import radix_rules as port
from scripts.dsa.vllm_qwen3_dsa_approx.radix_selector_reference import (
    select_prefix_reference,
)


DEFAULT_ROOT = Path("/cb/home/aarti/ws/code/ws_repos/dsa/dsa-csx")
ARM_FOR_SELECTOR = {
    "topk": "topk",
    "exact_ge": "exact_ge",
    "radix_floor": "floor",
    "radix_midpoint": "midpoint",
    "radix_ceil": "ceil",
}


def _source_root() -> Path:
    return Path(os.environ.get("DSA_CSX_ROOT", DEFAULT_ROOT))


def _load(module: str):
    root = _source_root() / "glm_52"
    if not (root / "study_core" / "rules.py").is_file():
        pytest.skip(f"dsa-csx checkout not available at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as error:  # pragma: no cover - environment-dependent
        pytest.skip(f"dsa-csx module {module!r} not importable: {error}")


@pytest.fixture(scope="module")
def source_rules():
    return _load("study_core.rules")


@pytest.fixture(scope="module")
def source_selection():
    return _load("vllm_study.selector.selection")


def test_arm_scheme_matches_the_source(source_rules) -> None:
    assert port.ARM_SCHEME == source_rules.ARM_SCHEME
    assert set(port.THRESHOLD_ARMS) == set(source_rules.THRESHOLD_ARMS)


def test_monotonic_map_matches_the_source(source_rules) -> None:
    torch.manual_seed(101)
    values = torch.cat(
        [
            torch.tensor([-65504.0, -1.0, -0.0, 0.0, 6.1e-5, 1.0, 65504.0]),
            torch.randn(4096) * 40,
        ]
    ).to(torch.float16)
    assert torch.equal(port.mono(values), source_rules.mono(values))
    keys = port.mono(values)
    assert torch.equal(port.mono_to_bits(keys), source_rules.mono_to_bits(keys))
    assert torch.equal(port.mono_to_f16(keys), source_rules.mono_to_f16(keys))


@pytest.mark.parametrize("scheme", ["exact", "floor", "midpoint", "ceil"])
def test_threshold_rounding_matches_the_source(source_rules, scheme: str) -> None:
    keys = torch.arange(0, 0x10000, 7, dtype=torch.int32)
    assert torch.equal(
        port.round_mono(keys, scheme), source_rules.round_mono(keys, scheme)
    )


def test_counts_and_delta_k_match_the_source(source_rules) -> None:
    torch.manual_seed(103)
    scores = torch.randn(2, 24, 96)
    qpos = torch.randint(0, 96, (24,))
    ours = port.score_view(scores, qpos, k=16)
    theirs = source_rules.score_view(scores, qpos, k=16)

    assert torch.equal(ours.mono_tq, theirs.mono_tq)
    assert torch.equal(ours.keff, theirs.keff)
    assert torch.equal(ours.degenerate, theirs.degenerate)
    for arm in port.THRESHOLD_ARMS:
        assert torch.equal(
            port.row_threshold(arm, ours.mono_tq),
            source_rules.row_threshold(arm, theirs.mono_tq),
        )
        assert torch.equal(
            port.count_for_arm(ours, arm), source_rules.count_for_arm(theirs, arm)
        )
    mine, source = port.delta_k_all(ours), source_rules.delta_k_all(theirs)
    for arm in port.THRESHOLD_ARMS:
        assert torch.equal(mine[arm], source[arm])


@pytest.mark.parametrize("selector", sorted(ARM_FOR_SELECTOR))
@pytest.mark.parametrize("keys,k,capacity", [(96, 16, 32), (64, 64, 64), (8, 4, 4)])
def test_reference_emitter_matches_the_source(
    source_selection, selector: str, keys: int, k: int, capacity: int
) -> None:
    """Our emitter must reproduce dsa-csx's buffer byte-for-byte on shared fixtures.

    The source's fast path handles one full-prefix row at a time, which is exactly a decode row,
    so the fixtures are single rows whose query position is the last key.
    """

    torch.manual_seed(211 + keys + k)
    logits = torch.randn(1, keys)
    qpos = torch.tensor([keys - 1])
    ours = torch.empty(1, capacity, dtype=torch.int32)
    theirs = torch.empty(1, capacity, dtype=torch.int32)

    select_prefix_reference(logits, qpos, k, ours, selector)
    source_selection.select_reference_decode(
        logits, qpos, k, theirs, ARM_FOR_SELECTOR[selector]
    )
    assert torch.equal(ours, theirs), (
        f"{selector}: ours={ours.tolist()} source={theirs.tolist()}"
    )
