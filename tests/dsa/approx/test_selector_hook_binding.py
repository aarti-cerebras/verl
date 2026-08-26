"""Pin the four vLLM selection call sites against the REAL installed vLLM.

The other hook tests drive `_prefill_hook`/`_decode_hook` with stand-in stock callables, so they
pass even if vLLM renames or re-signatures the ops underneath. These tests check the thing that
actually breaks on a vLLM upgrade: that the symbols exist, that their signatures still match the
pin, that every path is patched, and that uninstall restores the originals. Interception is by
module/namespace attribute, which only works because both call sites resolve late
(`ops.top_k_per_row_*` and `torch.ops._C.*`) -- a `from ... import` at the call site would make
the patch invisible, so this is a real contract worth pinning.
"""

import inspect

import pytest
import torch

vllm = pytest.importorskip("vllm")

from scripts.dsa.vllm_qwen3_dsa_approx import selector_hooks  # noqa: E402


INSTALLED = str(vllm.__version__).split("+", 1)[0]
pinned_only = pytest.mark.skipif(
    INSTALLED != selector_hooks.PINNED_VLLM_VERSION,
    reason=f"vLLM {INSTALLED} is not the pinned {selector_hooks.PINNED_VLLM_VERSION}",
)

# `install_hooks` treats these two as OPTIONAL (`getattr(torch.ops._C, name, None)`) and only
# *requires* them when `current_platform.is_cuda()`. A test that reads them unconditionally is
# therefore stricter than the contract and dies with AttributeError on a build that does not carry
# them, rather than skipping. Gate on presence so the failure mode matches the code's own posture.
OPTIONAL_CUDA_OPS = ("cooperative_topk", "persistent_topk")
cuda_ops_only = pytest.mark.skipif(
    not all(hasattr(torch.ops._C, name) for name in OPTIONAL_CUDA_OPS),
    reason=f"optional CUDA selection ops absent: {OPTIONAL_CUDA_OPS}",
)


@pytest.fixture
def uninstalled():
    """Leave the process exactly as clean as it was found, installed or not."""

    selector_hooks.uninstall_hooks()
    try:
        yield
    finally:
        selector_hooks.uninstall_hooks()


@pinned_only
def test_call_site_signatures_still_match_the_pin() -> None:
    from vllm import _custom_ops as ops

    assert tuple(inspect.signature(ops.top_k_per_row_prefill).parameters) == (
        "logits", "cu_seqlen_ks", "cu_seqlen_ke", "raw_topk_indices",
        "num_rows", "stride0", "stride1", "topk_tokens",
    )
    assert tuple(inspect.signature(ops.top_k_per_row_decode).parameters) == (
        "logits", "next_n", "seq_lens", "raw_topk_indices", "num_rows",
        "stride0", "stride1", "topk_tokens",
    )


@pinned_only
@cuda_ops_only
def test_install_covers_all_four_paths_then_restores(uninstalled) -> None:
    from vllm import _custom_ops as ops

    before = {
        "prefill": ops.top_k_per_row_prefill,
        "decode": ops.top_k_per_row_decode,
        "cooperative_topk": torch.ops._C.cooperative_topk,
        "persistent_topk": torch.ops._C.persistent_topk,
    }
    installation = selector_hooks.install_hooks()

    # All four exact selection paths vLLM 0.26 can reach are covered; replacing a subset would make
    # selector behaviour depend silently on request shape and decode dispatch.
    assert set(installation.originals) == set(before)
    assert ops.top_k_per_row_prefill is not before["prefill"]
    assert ops.top_k_per_row_decode is not before["decode"]
    assert torch.ops._C.cooperative_topk is not before["cooperative_topk"]
    assert torch.ops._C.persistent_topk is not before["persistent_topk"]
    for name, original in before.items():
        assert installation.originals[name] is original

    selector_hooks.uninstall_hooks()
    assert ops.top_k_per_row_prefill is before["prefill"]
    assert ops.top_k_per_row_decode is before["decode"]
    assert torch.ops._C.cooperative_topk is before["cooperative_topk"]
    assert torch.ops._C.persistent_topk is before["persistent_topk"]


@pinned_only
def test_install_is_idempotent(uninstalled) -> None:
    first = selector_hooks.install_hooks()
    assert selector_hooks.install_hooks() is first


def test_a_different_vllm_version_fails_closed(monkeypatch, uninstalled) -> None:
    # A silent signature drift would corrupt selection, so a version the hooks were not validated
    # against must be a startup error rather than a fallback.
    monkeypatch.setattr(selector_hooks, "_INSTALLATION", None)
    monkeypatch.setattr(selector_hooks, "PINNED_VLLM_VERSION", "0.0.0-never")
    with pytest.raises(RuntimeError, match="require vLLM 0.0.0-never"):
        selector_hooks.install_hooks()
