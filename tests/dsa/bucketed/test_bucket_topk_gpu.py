"""GPU-only correctness checks for the real vLLM 0.26 stock top-k operations."""

from __future__ import annotations

import pytest
import torch

from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_hooks import (
    _cuda_decode_hook,
    _prefill_hook,
)
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_runtime import (
    RUNTIME,
    BucketSelectorConfig,
    _membership,
)
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_topk_reference import select_bucket_topk_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    RUNTIME.reset_for_test()
    yield
    RUNTIME.reset_for_test()


def _configure(bucket_count: int, bucket_top_k: int, *, telemetry: str = "off") -> None:
    total_k = bucket_count * bucket_top_k
    RUNTIME.configure(
        BucketSelectorConfig(
            selector="modulo_bucket_topk",
            backend="vllm_stock_per_bucket",
            bucket_count=bucket_count,
            bucket_top_k=bucket_top_k,
            total_k=total_k,
            capacity=total_k,
            telemetry=telemetry,
        )
    )


def _assert_same_position_sets(observed: torch.Tensor, expected: torch.Tensor) -> None:
    sentinel = torch.iinfo(observed.dtype).max
    observed_sorted = torch.where(observed >= 0, observed, sentinel).sort(-1).values
    expected_sorted = torch.where(expected >= 0, expected, sentinel).sort(-1).values
    torch.testing.assert_close(observed_sorted.cpu(), expected_sorted.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("native_name", ["cooperative_topk", "persistent_topk"])
def test_specialized_decode_entry_redirects_to_real_stride_aware_stock_topk(native_name: str) -> None:
    """A global k=2048 specialized call must support the local k=256 bucket geometry."""

    from vllm import _custom_ops

    device = torch.device("cuda")
    bucket_count, bucket_top_k = 8, 256
    _configure(bucket_count, bucket_top_k)
    generator = torch.Generator(device=device).manual_seed(20260828)
    logits = torch.randn((2, 4103), generator=generator, device=device, dtype=torch.float32)
    seq_lens = torch.tensor([[4103], [1173]], device=device, dtype=torch.int32)
    observed = torch.empty((2, 2048), device=device, dtype=torch.int32)
    expected = torch.empty_like(observed)
    select_bucket_topk_reference(
        logits,
        seq_lens.to(torch.int64) - 1,
        expected,
        bucket_count=bucket_count,
        bucket_top_k=bucket_top_k,
    )

    def restricted_native_must_not_run(*args: object) -> None:
        raise AssertionError("restricted cooperative/persistent top-k was called with local k=256")

    _cuda_decode_hook(
        native_name,
        {native_name: restricted_native_must_not_run, "decode": _custom_ops.top_k_per_row_decode},
        logits,
        seq_lens,
        observed,
        torch.empty(1, device=device, dtype=torch.uint8),
        2048,
        logits.shape[1],
    )
    torch.cuda.synchronize()

    _assert_same_position_sets(observed, expected)
    assert (observed >= 0).sum(-1).cpu().tolist() == [2048, 1173]


def test_prefill_uses_real_stock_topk_per_request_and_bucket() -> None:
    from vllm import _custom_ops

    device = torch.device("cuda")
    bucket_count, bucket_top_k = 4, 3
    _configure(bucket_count, bucket_top_k)
    generator = torch.Generator(device=device).manual_seed(20260829)
    logits = torch.randn((3, 22), generator=generator, device=device, dtype=torch.float32)
    starts = torch.tensor([0, 0, 13], device=device, dtype=torch.int32)
    ends = torch.tensor([5, 13, 22], device=device, dtype=torch.int32)
    observed = torch.empty((3, 12), device=device, dtype=torch.int32)

    _prefill_hook(
        {"prefill": _custom_ops.top_k_per_row_prefill},
        logits,
        starts,
        ends,
        observed,
        3,
        logits.stride(0),
        logits.stride(1),
        12,
    )
    torch.cuda.synchronize()

    expected = torch.empty_like(observed)
    select_bucket_topk_reference(
        logits[:2, :13],
        torch.tensor([4, 12], device=device),
        expected[:2],
        bucket_count=bucket_count,
        bucket_top_k=bucket_top_k,
    )
    select_bucket_topk_reference(
        logits[2:, 13:22],
        torch.tensor([8], device=device),
        expected[2:],
        bucket_count=bucket_count,
        bucket_top_k=bucket_top_k,
    )

    _assert_same_position_sets(observed, expected)
    assert (observed >= 0).sum(-1).cpu().tolist() == [5, 12, 9]


def test_decode_cuda_graph_replay_handles_inactive_padding_row() -> None:
    """Replay must consume new lengths and leave a 7-to-8 graph padding row inactive."""

    from vllm import _custom_ops

    device = torch.device("cuda")
    bucket_count, bucket_top_k = 8, 256
    _configure(bucket_count, bucket_top_k)
    generator = torch.Generator(device=device).manual_seed(20260830)
    logits = torch.randn((8, 4103), generator=generator, device=device, dtype=torch.float32)
    seq_lens = torch.full((8, 1), 4103, device=device, dtype=torch.int32)
    observed = torch.empty((8, 2048), device=device, dtype=torch.int32)
    workspace = torch.empty(1, device=device, dtype=torch.uint8)

    def restricted_native_must_not_run(*args: object) -> None:
        raise AssertionError("restricted native top-k ran inside the bucket decode graph")

    originals = {
        "cooperative_topk": restricted_native_must_not_run,
        "decode": _custom_ops.top_k_per_row_decode,
    }

    def select() -> None:
        _cuda_decode_hook(
            "cooperative_topk",
            originals,
            logits,
            seq_lens,
            observed,
            workspace,
            2048,
            logits.shape[1],
        )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        select()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()

    replay_lengths = torch.tensor(
        [[4103], [3501], [3077], [2049], [2048], [1173], [17], [0]],
        device=device,
        dtype=torch.int32,
    )
    seq_lens.copy_(replay_lengths)
    logits.copy_(torch.randn(logits.shape, generator=generator, device=device))
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.empty_like(observed)
    select_bucket_topk_reference(
        logits,
        replay_lengths.to(torch.int64) - 1,
        expected,
        bucket_count=bucket_count,
        bucket_top_k=bucket_top_k,
    )
    _assert_same_position_sets(observed, expected)
    assert (observed >= 0).sum(-1).cpu().tolist() == [2048, 2048, 2048, 2048, 2048, 1173, 17, 0]
    assert bool((observed[-1] == -1).all())


def test_graph_safety_telemetry_updates_persistent_counters_on_replay() -> None:
    """Replay must update stable bucket-only counters without executing telemetry Python."""

    from vllm import _custom_ops

    device = torch.device("cuda")
    bucket_count, bucket_top_k = 8, 256
    _configure(bucket_count, bucket_top_k, telemetry="graph_safety")
    RUNTIME.initialize_graph_safety(["layer.0"], device=device)
    assert RUNTIME._graph is not None
    graph_address = RUNTIME._graph.data_ptr()

    generator = torch.Generator(device=device).manual_seed(20260831)
    logits = torch.randn((8, 4103), generator=generator, device=device)
    seq_lens = torch.full((8, 1), 4103, device=device, dtype=torch.int32)
    output = torch.empty((8, 2048), device=device, dtype=torch.int32)
    workspace = torch.empty(1, device=device, dtype=torch.uint8)

    def restricted_native_must_not_run(*args: object) -> None:
        raise AssertionError("restricted native top-k ran inside the telemetry graph")

    originals = {
        "cooperative_topk": restricted_native_must_not_run,
        "decode": _custom_ops.top_k_per_row_decode,
    }

    def select() -> None:
        with RUNTIME.layer("layer.0"):
            _cuda_decode_hook(
                "cooperative_topk",
                originals,
                logits,
                seq_lens,
                output,
                workspace,
                2048,
                logits.shape[1],
            )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        select()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    torch.cuda.synchronize()
    RUNTIME.reset()
    assert RUNTIME._graph.data_ptr() == graph_address

    replay_lengths = torch.tensor(
        [[4103], [3501], [3077], [2049], [2048], [1173], [17], [0]],
        device=device,
        dtype=torch.int32,
    )
    seq_lens.copy_(replay_lengths)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    replay = RUNTIME.artifact()["graph_replay"]
    assert replay["global"]["rows"] == 14
    assert replay["global"]["calls"] == 2
    assert replay["global"]["selected_total"] == 2 * sum(
        min(int(length), 2048) for length in replay_lengths.cpu().flatten()
    )
    layer = replay["phases"]["decode"]["per_layer"]["layer.0"]
    assert not any(
        layer[name]
        for name in (
            "count_mismatch",
            "padding_violation",
            "prefix_violation",
            "invalid_index",
            "noncausal_index",
            "duplicate_index",
        )
    )
    assert len(layer["per_bucket"]) == bucket_count

    RUNTIME.reset()
    assert RUNTIME._graph.data_ptr() == graph_address
    assert RUNTIME.artifact()["graph_replay"]["global"]["rows"] == 0


def test_graph_verify_exact_updates_device_quality_on_replay() -> None:
    """Exact comparison must replay into stable device moments without host folding."""

    from vllm import _custom_ops

    device = torch.device("cuda")
    bucket_count, bucket_top_k = 8, 256
    _configure(bucket_count, bucket_top_k, telemetry="graph_verify_exact")
    RUNTIME.initialize_graph_quality(["layer.0"], device=device)
    assert RUNTIME._graph_quality is not None
    quality_address = RUNTIME._graph_quality.data_ptr()

    generator = torch.Generator(device=device).manual_seed(20260829)
    logits = torch.randn((8, 4103), generator=generator, device=device)
    seq_lens = torch.full((8, 1), 4103, device=device, dtype=torch.int32)
    output = torch.empty((8, 2048), device=device, dtype=torch.int32)
    workspace = torch.empty(1, device=device, dtype=torch.uint8)

    def restricted_native_must_not_run(*args: object) -> None:
        raise AssertionError("restricted native top-k ran inside graph_verify_exact")

    originals = {
        "cooperative_topk": restricted_native_must_not_run,
        "decode": _custom_ops.top_k_per_row_decode,
    }

    def select() -> None:
        with RUNTIME.layer("layer.0"):
            _cuda_decode_hook(
                "cooperative_topk",
                originals,
                logits,
                seq_lens,
                output,
                workspace,
                2048,
                logits.shape[1],
            )

    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        select()
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        select()
    torch.cuda.synchronize()
    RUNTIME.reset()
    assert RUNTIME._graph_quality.data_ptr() == quality_address

    replay_lengths = torch.tensor(
        [[4103], [3501], [3077], [2049], [2048], [1173], [17], [0]],
        device=device,
        dtype=torch.int32,
    )
    seq_lens.copy_(replay_lengths)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    artifact = RUNTIME.artifact()
    assert artifact["graph_replay"]["global"]["rows"] == 14
    quality = artifact["graph_replay_quality"]
    assert quality["storage"] == "persistent_device_moments"
    decode = quality["phases"]["decode"]
    assert decode["global"]["recall"]["n"] == 14
    assert 0.0 <= decode["global"]["recall"]["min"] <= 1.0
    assert 0.0 <= decode["global"]["precision"]["min"] <= 1.0
    assert decode["global"]["selected_count"]["total"] == 2 * sum(
        min(int(length), 2048) for length in replay_lengths.cpu().flatten()
    )
    assert len(decode["per_layer"]["layer.0"]["per_bucket"]) == bucket_count

    flat_lengths = replay_lengths.flatten().long()
    valid_keys = torch.arange(logits.shape[1], device=device)[None, :] < flat_lengths[:, None]
    exact = logits.masked_fill(~valid_keys, float("-inf")).topk(2048, dim=-1).indices.to(torch.int32)
    keep = torch.arange(2048, device=device)[None, :] < flat_lengths.clamp(max=2048)[:, None]
    exact = torch.where(keep, exact, -1)
    active = flat_lengths > 0
    intersection = _membership(output, exact, logits.shape[1]).sum(-1)
    expected_recall = (
        intersection.float() / (exact >= 0).sum(-1).clamp(min=1)
    )[active]
    assert decode["global"]["recall"]["mean"] == pytest.approx(
        float(expected_recall.mean()), abs=1e-7
    )
    assert decode["global"]["intersection"]["mean"] == pytest.approx(
        float(intersection[active].double().mean()), abs=1e-7
    )

    RUNTIME.reset()
    assert RUNTIME._graph_quality.data_ptr() == quality_address
    assert RUNTIME.artifact()["graph_replay"]["global"]["rows"] == 0
