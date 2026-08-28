import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_hooks import (
    _cuda_decode_hook,
    _decode_hook,
    _prefill_hook,
)
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_runtime import (
    RUNTIME,
    BucketSelectorConfig,
)


def _configure(backend: str = "vllm_stock_per_bucket") -> None:
    RUNTIME.configure(
        BucketSelectorConfig(
            selector="modulo_bucket_topk",
            backend=backend,
            bucket_count=2,
            bucket_top_k=2,
            total_k=4,
            capacity=4,
        )
    )


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    RUNTIME.reset_for_test()
    yield
    RUNTIME.reset_for_test()


def _fill_stock(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    target: torch.Tensor,
    top_k: int,
) -> None:
    target.fill_(-1)
    width = min(top_k, logits.shape[1])
    if width == 0:
        return
    flat_lengths = lengths.reshape(-1).long()
    valid = torch.arange(logits.shape[1])[None, :] < flat_lengths[:, None]
    top = logits.masked_fill(~valid, float("-inf")).topk(width, dim=-1)
    keep = torch.arange(width)[None, :] < flat_lengths.clamp(max=top_k)[:, None]
    target[:, :width].copy_(torch.where(keep, top.indices.to(torch.int32), -1))


def test_generic_decode_reuses_saved_stock_per_bucket() -> None:
    _configure()
    logits = torch.tensor([[0.0, 10.0, 9.0, 8.0, 7.0, 6.0], [1.0, 2.0, 3.0, 99.0, 99.0, 99.0]])
    seq_lens = torch.tensor([[6], [3]], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)
    calls: list[tuple[tuple[int, ...], int]] = []

    def stock(logits, next_n, seq_lens, target, num_rows, stride0, stride1, topk_tokens):
        calls.append((tuple(logits.shape), stride1))
        _fill_stock(logits, seq_lens, target, topk_tokens)

    _decode_hook({"decode": stock}, logits, 1, seq_lens, output, 2, 6, 1, 4)

    assert output.tolist()[0] == [2, 1, 4, 3]
    assert set(output[1][output[1] >= 0].tolist()) == {0, 1, 2}
    assert calls == [((2, 3), 2), ((2, 3), 2)]


def test_reference_decode_does_not_call_stock() -> None:
    _configure("torch_reference")
    logits = torch.randn(2, 7)
    seq_lens = torch.tensor([[7], [0]], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)

    def stock(*args):
        raise AssertionError("reference backend called stock top-k")

    _decode_hook({"decode": stock}, logits, 1, seq_lens, output, 2, 7, 1, 4)
    assert bool((output[1] == -1).all())


def test_prefill_restarts_modulo_positions_per_request() -> None:
    _configure()
    # Request 0 has two query rows over key columns [0, 3); request 1 has one row over [3, 5).
    logits = torch.tensor(
        [
            [2.0, 9.0, 0.0, -9.0, -9.0],
            [1.0, 2.0, 3.0, -9.0, -9.0],
            [-9.0, -9.0, -9.0, 7.0, 8.0],
        ]
    )
    starts = torch.tensor([0, 0, 3], dtype=torch.int32)
    ends = torch.tensor([1, 3, 5], dtype=torch.int32)
    output = torch.empty(3, 4, dtype=torch.int32)

    def stock(logits, starts, ends, target, num_rows, stride0, stride1, topk_tokens):
        assert bool((starts == 0).all())
        _fill_stock(logits, ends, target, topk_tokens)

    _prefill_hook({"prefill": stock}, logits, starts, ends, output, 3, 5, 1, 4)

    assert set(output[0][output[0] >= 0].tolist()) == {0}
    assert set(output[1][output[1] >= 0].tolist()) == {0, 1, 2}
    # The second request emits 0 and 1, not packed score columns 3 and 4.
    assert set(output[2][output[2] >= 0].tolist()) == {0, 1}


@pytest.mark.parametrize("name", ["cooperative_topk", "persistent_topk"])
def test_native_decode_redirects_to_strided_generic_stock(name: str) -> None:
    _configure()
    logits = torch.randn(2, 9)
    seq_lens = torch.tensor([[9], [5]], dtype=torch.int32)
    output = torch.empty(2, 4, dtype=torch.int32)
    generic_calls: list[tuple[tuple[int, ...], int, int]] = []

    def generic(logits, next_n, seq_lens, target, num_rows, stride0, stride1, topk_tokens):
        generic_calls.append((tuple(logits.shape), stride1, topk_tokens))
        _fill_stock(logits, seq_lens, target, topk_tokens)

    def native(*args):
        raise AssertionError("local bucket selection called the restricted native top-k")

    _cuda_decode_hook(
        name,
        {name: native, "decode": generic},
        logits,
        seq_lens,
        output,
        torch.empty(1, dtype=torch.uint8),
        4,
        9,
    )

    assert generic_calls == [((2, 5), 2, 2), ((2, 4), 2, 2)]
    assert (output >= 0).sum(-1).tolist() == [4, 4]


def test_inactive_runtime_defers_to_original() -> None:
    logits = torch.randn(1, 4)
    seq_lens = torch.tensor([[4]], dtype=torch.int32)
    output = torch.empty(1, 2, dtype=torch.int32)
    calls: list[str] = []

    def stock(logits, next_n, seq_lens, target, *unused):
        calls.append("stock")
        target.copy_(torch.tensor([[3, 2]], dtype=torch.int32))
        return "stock-return"

    observed = _decode_hook({"decode": stock}, logits, 1, seq_lens, output, 1, 4, 1, 2)
    assert observed == "stock-return"
    assert calls == ["stock"]
    assert output.tolist() == [[3, 2]]
