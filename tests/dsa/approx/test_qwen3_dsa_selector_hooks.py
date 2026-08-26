from collections.abc import Callable

import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_approx.radix_selector_reference import (
    select_prefix_reference,
)
from scripts.dsa.vllm_qwen3_dsa_approx.selector_hooks import (
    _cuda_decode_hook,
    _decode_hook,
    _prefill_hook,
)
from scripts.dsa.vllm_qwen3_dsa_approx.selector_runtime import (
    RUNTIME,
    SelectorConfig,
)


@pytest.fixture(autouse=True)
def configured_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUNTIME, "_config", None)
    RUNTIME.reset()
    RUNTIME.configure(
        SelectorConfig(
            selector="radix_ceil",
            backend="dsa_csx_reference",
            rule_k=4,
            capacity=4,
            omit_bits=4,
            telemetry="verify_exact",
        )
    )


def _stock_prefix(logits: torch.Tensor, qpos: torch.Tensor) -> Callable[[torch.Tensor], None]:
    def stock(output: torch.Tensor) -> None:
        select_prefix_reference(logits, qpos, 4, output, "topk")

    return stock


def test_prefill_hook_replaces_exact_selection() -> None:
    logits = torch.tensor([[9.0, 1.0, 0.0], [0.2, 0.8, 0.5]])
    starts = torch.tensor([0, 0], dtype=torch.int32)
    ends = torch.tensor([1, 3], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)

    def stock(*args) -> None:
        target = args[3]
        _stock_prefix(logits, ends - starts - 1)(target)

    _prefill_hook(
        {"prefill": stock}, logits, starts, ends, output, 2, 3, 1, 4
    )
    assert bool(((output == -1) | (output >= 0)).all())
    assert not bool(((output[:, :-1] < 0) & (output[:, 1:] >= 0)).any())


def test_generic_decode_hook() -> None:
    torch.manual_seed(31)
    logits = torch.randn(2, 8)
    seq_lens = torch.tensor([[8], [6]], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)

    def stock(logits, next_n, seq_lens, target, *unused) -> None:
        _stock_prefix(logits, seq_lens.reshape(-1) - 1)(target)

    _decode_hook(
        {"decode": stock}, logits, 1, seq_lens, output, 2, 8, 1, 4
    )
    assert bool((output <= (seq_lens.reshape(-1) - 1)[:, None]).all())


@pytest.mark.parametrize("name", ["cooperative_topk", "persistent_topk"])
def test_cuda_decode_dispatch_hooks(name: str) -> None:
    torch.manual_seed(37)
    logits = torch.randn(2, 8)
    seq_lens = torch.tensor([[8], [7]], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)

    def stock(logits, seq_lens, target, workspace, topk_tokens, max_seq_len) -> None:
        _stock_prefix(logits, seq_lens.reshape(-1) - 1)(target)

    _cuda_decode_hook(
        name,
        {name: stock},
        logits,
        seq_lens,
        output,
        torch.empty(1, dtype=torch.uint8),
        4,
        8,
    )
    assert not bool(((output[:, :-1] < 0) & (output[:, 1:] >= 0)).any())


def test_topk_control_mode_defers_entirely_to_stock(monkeypatch: pytest.MonkeyPatch) -> None:
    """`dsa_selector=topk` is the baseline and rollback, so it must not go near the hooks.

    The control arm has to be the UNCHANGED exact path: if the hooks did any of their own work
    here, the parity gate between the exact server and this copy would be measuring the hooks
    rather than the snapshot.
    """

    monkeypatch.setattr(
        RUNTIME,
        "_config",
        SelectorConfig(
            selector="topk",
            backend="vllm_stock",
            rule_k=4,
            capacity=4,
            omit_bits=4,
            telemetry="verify_exact",
        ),
    )
    RUNTIME.reset()
    assert RUNTIME.active is False

    torch.manual_seed(53)
    logits = torch.randn(2, 8)
    seq_lens = torch.tensor([[8], [6]], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)
    calls: list[str] = []

    def stock(logits, next_n, seq_lens, target, *unused) -> str:
        calls.append("stock")
        _stock_prefix(logits, seq_lens.reshape(-1) - 1)(target)
        return "stock-return"

    assert _decode_hook({"decode": stock}, logits, 1, seq_lens, output, 2, 8, 1, 4) == "stock-return"
    assert calls == ["stock"]
    # No telemetry, no counters, nothing recorded: the control arm leaves no trace.
    assert RUNTIME.artifact()["safety"]["rows"] == 0
    assert "decode" not in RUNTIME.artifact()


def test_telemetry_off_keeps_selection_and_safety_but_skips_host_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RUNTIME,
        "_config",
        SelectorConfig(
            selector="radix_ceil",
            backend="dsa_csx_reference",
            rule_k=4,
            capacity=4,
            omit_bits=4,
            telemetry="off",
        ),
    )
    RUNTIME.reset()

    def host_counter_must_not_run(phase: str) -> None:
        raise AssertionError(f"host telemetry ran with telemetry=off: {phase}")

    monkeypatch.setattr(RUNTIME, "note_call", host_counter_must_not_run)
    logits = torch.randn(2, 8)
    seq_lens = torch.tensor([[8], [6]], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)

    def stock(logits, next_n, seq_lens, target, *unused) -> None:
        _stock_prefix(logits, seq_lens.reshape(-1) - 1)(target)

    _decode_hook({"decode": stock}, logits, 1, seq_lens, output, 2, 8, 1, 4)
    assert not bool(((output[:, :-1] < 0) & (output[:, 1:] >= 0)).any())
    assert RUNTIME.artifact()["safety"]["rows"] == 0
    assert "decode" not in RUNTIME.artifact()
