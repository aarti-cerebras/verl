"""Version-pinned interception of vLLM 0.26 top-k for isolated bucket selection."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .bucket_selector_runtime import RUNTIME
from .bucket_topk_reference import BucketSelectionResult, _assert_tensor, select_bucket_topk_reference
from .bucket_topk_stock import StockTopK, select_bucket_topk_stock

PINNED_VLLM_VERSION = "0.26.0"


@dataclass(frozen=True)
class Installation:
    containers: dict[str, Any]
    originals: dict[str, Callable[..., Any]]


_INSTALLATION: Installation | None = None


def _captured_indexer_layer() -> str:
    """Recover the static indexer prefix from vLLM's pinned custom-op frame.

    During CUDA capture the Python ``RUNTIME.layer`` scope may have run only while Dynamo traced
    the model. The sparse-indexer implementation still carries its k-cache prefix, and selecting a
    persistent counter slice from that prefix during capture bakes the right address into replay.
    """

    frame = inspect.currentframe()
    try:
        while frame is not None:
            if (
                frame.f_code.co_name == "sparse_attn_indexer"
                and frame.f_globals.get("__name__")
                == "vllm.model_executor.layers.sparse_attn_indexer"
                and "k_cache_prefix" in frame.f_locals
            ):
                prefix = frame.f_locals["k_cache_prefix"]
                value = getattr(prefix, "value", prefix)
                return str(value).removesuffix(".k_cache")
            frame = frame.f_back
    finally:
        del frame
    if RUNTIME.current_layer != "unattributed":
        return RUNTIME.current_layer
    raise RuntimeError("graph telemetry could not recover the bucket indexer layer prefix")


def _validate_output(
    output: torch.Tensor,
    query_positions: torch.Tensor,
    result: BucketSelectionResult,
    key_count: int,
) -> None:
    valid = output >= 0
    qpos = query_positions.reshape(-1).to(output.device, torch.int64)
    _assert_tensor(
        (valid.sum(-1) == result.selected_count).all(),
        "bucket selector count does not match emitted positions",
    )
    _assert_tensor(
        ~((~valid[:, :-1]) & valid[:, 1:]).any(),
        "bucket selector output is not valid-prefix/-1-suffix",
    )
    _assert_tensor(
        ~((output < -1) | (output > qpos[:, None])).any(),
        "bucket selector emitted an invalid or noncausal request-local position",
    )
    sentinel = torch.full_like(output, key_count)
    ordered = torch.where(valid, output, sentinel).sort(-1).values
    _assert_tensor(
        ~((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != key_count)).any(),
        "bucket selector emitted duplicate request-local positions",
    )


def _select(
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    *,
    phase: str,
    stock_topk: StockTopK,
) -> None:
    config = RUNTIME.config
    if config is None:
        raise RuntimeError("bucket selector hook ran before model configuration")
    if output.shape[1] != config.capacity:
        raise RuntimeError(f"bucket selector output width {output.shape[1]} != configured capacity {config.capacity}")
    if config.backend == "torch_reference":
        result = select_bucket_topk_reference(
            logits,
            query_positions,
            output,
            bucket_count=config.bucket_count,
            bucket_top_k=config.bucket_top_k,
        )
    else:
        result = select_bucket_topk_stock(
            logits,
            query_positions,
            output,
            bucket_count=config.bucket_count,
            bucket_top_k=config.bucket_top_k,
            stock_topk=stock_topk,
        )
    _validate_output(output, query_positions, result, logits.shape[1])
    exact_reference = None
    if config.telemetry in ("verify_exact", "graph_verify_exact"):
        exact_reference = torch.full_like(output, -1)
        sequence_lengths = (query_positions.reshape(-1).to(torch.int64) + 1).clamp(
            min=0, max=logits.shape[1]
        )
        stock_topk(logits, sequence_lengths, exact_reference, config.total_k)
    RUNTIME.record(
        phase=phase,
        logits=logits,
        output=output,
        query_positions=query_positions,
        result=result,
        exact_reference=exact_reference,
        layer_name=(
            _captured_indexer_layer()
            if config.telemetry in ("graph_safety", "graph_verify_exact")
            else None
        ),
    )


def _prefill_hook(
    originals: dict[str, Callable[..., Any]],
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> None:
    if not RUNTIME.active:
        return originals["prefill"](
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )
    config = RUNTIME.config
    assert config is not None
    if int(num_rows) != logits.shape[0] or int(topk_tokens) != config.capacity:
        raise RuntimeError(
            f"bucket prefill geometry rows={num_rows}/{logits.shape[0]} topk={topk_tokens}/{config.capacity}"
        )

    # Stage-A eager request discovery. validate_execution rejects CUDA graphs until this CPU sync
    # is replaced by an all-rows device path and validated through the GPU handoff.
    bounds = torch.stack((cu_seqlen_ks, cu_seqlen_ke)).to("cpu", torch.int64)
    starts, ends = bounds[0].tolist(), bounds[1].tolist()
    query_positions = (cu_seqlen_ke - cu_seqlen_ks - 1).to(torch.int64)
    request_start = 0
    request_ends = [index for index in range(1, len(starts)) if starts[index] != starts[index - 1]] + [len(starts)]
    for request_end in request_ends:
        key_start = starts[request_start]
        key_count = ends[request_end - 1] - key_start
        if key_count <= 0 or key_start + key_count > logits.shape[1]:
            raise RuntimeError(
                f"invalid bucket prefill request rows [{request_start},{request_end}) with key "
                f"slice [{key_start},{key_start + key_count}) for logits {tuple(logits.shape)}"
            )
        rows = slice(request_start, request_end)

        def stock_topk(
            bucket_logits: torch.Tensor,
            bucket_lengths: torch.Tensor,
            target: torch.Tensor,
            local_k: int,
        ) -> None:
            starts_local = torch.zeros_like(bucket_lengths, dtype=torch.int32)
            ends_local = bucket_lengths.to(torch.int32)
            originals["prefill"](
                bucket_logits,
                starts_local,
                ends_local,
                target,
                bucket_logits.shape[0],
                bucket_logits.stride(0),
                bucket_logits.stride(1),
                local_k,
            )

        _select(
            logits[rows, key_start : key_start + key_count],
            query_positions[rows],
            raw_topk_indices[rows],
            phase="prefill",
            stock_topk=stock_topk,
        )
        request_start = request_end
    return None


def _decode_selection(
    *,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    stock_topk: StockTopK,
) -> None:
    lengths = seq_lens.reshape(-1)
    if lengths.numel() != output.shape[0]:
        raise RuntimeError(
            f"bucket decode supports next_n=1 only; got {lengths.numel()} lengths for {output.shape[0]} rows"
        )
    _select(
        logits,
        lengths.to(torch.int64) - 1,
        output,
        phase="decode",
        stock_topk=stock_topk,
    )


def _decode_hook(
    originals: dict[str, Callable[..., Any]],
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    raw_topk_indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    topk_tokens: int,
) -> Any:
    if not RUNTIME.active:
        return originals["decode"](
            logits,
            next_n,
            seq_lens,
            raw_topk_indices,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )
    config = RUNTIME.config
    assert config is not None
    if int(next_n) != 1 or int(num_rows) != logits.shape[0] or int(topk_tokens) != config.capacity:
        raise RuntimeError(
            f"unsupported bucket decode geometry next_n={next_n}, rows={num_rows}/"
            f"{logits.shape[0]}, topk={topk_tokens}/{config.capacity}"
        )

    def stock_topk(
        bucket_logits: torch.Tensor,
        bucket_lengths: torch.Tensor,
        target: torch.Tensor,
        local_k: int,
    ) -> None:
        local_lengths = bucket_lengths.to(seq_lens.dtype).reshape_as(seq_lens)
        originals["decode"](
            bucket_logits,
            1,
            local_lengths,
            target,
            bucket_logits.shape[0],
            bucket_logits.stride(0),
            bucket_logits.stride(1),
            local_k,
        )

    return _decode_selection(
        logits=logits,
        seq_lens=seq_lens,
        output=raw_topk_indices,
        stock_topk=stock_topk,
    )


def _cuda_decode_hook(
    name: str,
    originals: dict[str, Callable[..., Any]],
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    workspace: torch.Tensor,
    topk_tokens: int,
    max_seq_len: int,
) -> Any:
    if not RUNTIME.active:
        return originals[name](logits, seq_lens, topk_indices, workspace, topk_tokens, max_seq_len)
    config = RUNTIME.config
    assert config is not None
    if int(topk_tokens) != config.capacity:
        raise RuntimeError(f"bucket native decode topk={topk_tokens} != configured capacity {config.capacity}")

    def stock_topk(
        bucket_logits: torch.Tensor,
        bucket_lengths: torch.Tensor,
        target: torch.Tensor,
        local_k: int,
    ) -> None:
        # The native cooperative/persistent kernels only accept k in {512, 1024, 2048} and expose
        # no score strides. Redirect their global-k call to vLLM's saved generic, stride-aware
        # top-k so arbitrary fixed local k values work without materializing each bucket.
        local_lengths = bucket_lengths.to(seq_lens.dtype).reshape_as(seq_lens)
        originals["decode"](
            bucket_logits,
            1,
            local_lengths,
            target,
            bucket_logits.shape[0],
            bucket_logits.stride(0),
            bucket_logits.stride(1),
            local_k,
        )

    return _decode_selection(
        logits=logits,
        seq_lens=seq_lens,
        output=topk_indices,
        stock_topk=stock_topk,
    )


def install_hooks() -> Installation:
    """Install once, failing closed on a vLLM version or Python signature mismatch."""

    global _INSTALLATION
    if _INSTALLATION is not None:
        return _INSTALLATION

    import vllm
    from vllm import _custom_ops

    observed_version = str(vllm.__version__).split("+", 1)[0]
    if observed_version != PINNED_VLLM_VERSION:
        raise RuntimeError(f"bucket selector hooks require vLLM {PINNED_VLLM_VERSION}, found {vllm.__version__}")
    expected = {
        "prefill": (
            "logits",
            "cu_seqlen_ks",
            "cu_seqlen_ke",
            "raw_topk_indices",
            "num_rows",
            "stride0",
            "stride1",
            "topk_tokens",
        ),
        "decode": (
            "logits",
            "next_n",
            "seq_lens",
            "raw_topk_indices",
            "num_rows",
            "stride0",
            "stride1",
            "topk_tokens",
        ),
    }
    originals: dict[str, Callable[..., Any]] = {
        "prefill": _custom_ops.top_k_per_row_prefill,
        "decode": _custom_ops.top_k_per_row_decode,
    }
    for name in ("cooperative_topk", "persistent_topk"):
        symbol = getattr(torch.ops._C, name, None)
        if symbol is not None:
            originals[name] = symbol
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        missing = {"cooperative_topk", "persistent_topk"} - originals.keys()
        if missing:
            raise RuntimeError(f"CUDA bucket hook targets are missing: {sorted(missing)}")
    for name in ("prefill", "decode"):
        parameters = tuple(inspect.signature(originals[name]).parameters)
        if parameters != expected[name]:
            raise RuntimeError(f"vLLM {name} selector signature is {parameters}, expected {expected[name]}")

    _custom_ops.top_k_per_row_prefill = lambda *args, **kwargs: _prefill_hook(originals, *args, **kwargs)
    _custom_ops.top_k_per_row_decode = lambda *args, **kwargs: _decode_hook(originals, *args, **kwargs)
    if "cooperative_topk" in originals:
        torch.ops._C.cooperative_topk = lambda *args, **kwargs: _cuda_decode_hook(
            "cooperative_topk", originals, *args, **kwargs
        )
    if "persistent_topk" in originals:
        torch.ops._C.persistent_topk = lambda *args, **kwargs: _cuda_decode_hook(
            "persistent_topk", originals, *args, **kwargs
        )
    _INSTALLATION = Installation({"custom_ops": _custom_ops, "cuda_ops": torch.ops._C}, originals)
    return _INSTALLATION


def uninstall_hooks() -> None:
    global _INSTALLATION
    if _INSTALLATION is None:
        return
    installation = _INSTALLATION
    installation.containers["custom_ops"].top_k_per_row_prefill = installation.originals["prefill"]
    installation.containers["custom_ops"].top_k_per_row_decode = installation.originals["decode"]
    if "cooperative_topk" in installation.originals:
        installation.containers["cuda_ops"].cooperative_topk = installation.originals["cooperative_topk"]
    if "persistent_topk" in installation.originals:
        installation.containers["cuda_ops"].persistent_topk = installation.originals["persistent_topk"]
    _INSTALLATION = None
