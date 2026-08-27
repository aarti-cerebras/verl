"""Version-pinned interception of every vLLM 0.26 exact top-k dispatch path."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .radix_selector_reference import select_prefix_reference
from .selector_runtime import RUNTIME

PINNED_VLLM_VERSION = "0.26.0"


@dataclass(frozen=True)
class Installation:
    containers: dict[str, Any]
    originals: dict[str, Callable[..., Any]]


_INSTALLATION: Installation | None = None


def _captured_indexer_layer() -> str:
    """Recover the static layer name from vLLM's enclosing custom-op implementation.

    Dynamo may execute the Python ``RUNTIME.layer`` scope only while tracing, whereas the hooked
    top-k call runs later inside vLLM's opaque ``sparse_attn_indexer`` implementation during CUDA
    capture. That implementation already carries the encoded per-layer k-cache prefix. Reading it
    from the version-pinned caller frame happens once at capture; the selected persistent counter
    slice is then baked into the graph and needs no Python attribution on replay.
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
    raise RuntimeError(
        "graph_safety could not recover the layer prefix from vLLM's pinned "
        "sparse_attn_indexer call frame"
    )


def _selection(
    *,
    phase: str,
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    output: torch.Tensor,
    stock: Callable[[torch.Tensor], Any],
) -> Any:
    config = RUNTIME.config
    if config is None or config.selector == "topk":
        return stock(output)
    result = select_prefix_reference(
        logits,
        query_positions,
        config.rule_k,
        output,
        config.selector,
    )
    reference = None
    if config.telemetry in ("verify_exact", "graph_verify_exact"):
        reference = torch.full_like(output, -1)
        if output.shape[1] == config.rule_k:
            # This is the literal stock vLLM exact set and therefore preserves
            # its tie-breaking behavior.
            stock(reference)
        else:
            # Stock kernels select `topk_tokens`, which is capacity on this
            # copied server. For an over-capturing arm capacity may exceed the
            # logical k, so construct the logical exact-k scratch set from the
            # same score tensor instead.
            width = min(config.rule_k, reference.shape[1], result.exact_indices.shape[1])
            keep = torch.arange(width, device=output.device)[None, :] < result.effective_k[:, None]
            reference[:, :width].copy_(
                torch.where(keep, result.exact_indices[:, :width].to(torch.int32), -1)
            )
    # Host-folded telemetry runs only in eager mode. graph_verify_exact keeps that path for eager
    # prefill but records decode into persistent device storage. Do not count a decode capture-time
    # Python call that would never repeat on replay.
    if config.telemetry not in ("off", "graph_safety") and not (
        config.telemetry == "graph_verify_exact" and phase == "decode"
    ):
        RUNTIME.note_call(phase)
    RUNTIME.validate_and_record(
        phase=phase,
        output=output,
        query_positions=query_positions,
        result=result,
        stock_reference=reference,
        key_count=logits.shape[1],
        layer_name=(
            _captured_indexer_layer()
            if config.telemetry in ("graph_safety", "graph_verify_exact")
            and (config.telemetry == "graph_safety" or phase == "decode")
            else None
        ),
    )
    return None


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
    def stock_all(output: torch.Tensor) -> Any:
        return originals["prefill"](
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            output,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )

    if not RUNTIME.active:
        return stock_all(raw_topk_indices)
    if int(num_rows) != logits.shape[0]:
        raise RuntimeError(f"prefill num_rows={num_rows} does not match logits {tuple(logits.shape)}")

    bounds = torch.stack((cu_seqlen_ks, cu_seqlen_ke)).to("cpu", torch.int64)
    starts, ends = bounds[0].tolist(), bounds[1].tolist()
    query_positions = (cu_seqlen_ke - cu_seqlen_ks - 1).to(torch.int64)
    request_start = 0
    request_ends = [
        index for index in range(1, len(starts)) if starts[index] != starts[index - 1]
    ] + [len(starts)]
    for request_end in request_ends:
        key_start = starts[request_start]
        key_count = ends[request_end - 1] - key_start
        if key_count <= 0 or key_start + key_count > logits.shape[1]:
            raise RuntimeError(
                f"invalid prefill request rows [{request_start},{request_end}) with key slice "
                f"[{key_start},{key_start + key_count}) for logits {tuple(logits.shape)}"
            )
        rows = slice(request_start, request_end)

        def stock_request(output: torch.Tensor, *, _rows=rows) -> Any:
            scratch = torch.full_like(raw_topk_indices, -1)
            stock_all(scratch)
            output.copy_(scratch[_rows])
            return None

        _selection(
            phase="prefill",
            logits=logits[rows, key_start : key_start + key_count],
            query_positions=query_positions[rows],
            output=raw_topk_indices[rows],
            stock=stock_request,
        )
        request_start = request_end
    return None


def _decode_selection(
    *,
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    stock: Callable[[torch.Tensor], Any],
) -> Any:
    lengths = seq_lens.reshape(-1)
    if lengths.numel() != output.shape[0]:
        raise RuntimeError(
            "approximate reference decode currently supports next_n=1 only; "
            f"got {lengths.numel()} lengths for {output.shape[0]} rows"
        )
    return _selection(
        phase="decode",
        logits=logits,
        query_positions=lengths.to(torch.int64) - 1,
        output=output,
        stock=stock,
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
    def stock(output: torch.Tensor) -> Any:
        return originals["decode"](
            logits,
            next_n,
            seq_lens,
            output,
            num_rows,
            stride0,
            stride1,
            topk_tokens,
        )

    if not RUNTIME.active:
        return stock(raw_topk_indices)
    if int(next_n) != 1 or int(num_rows) != logits.shape[0]:
        raise RuntimeError(
            f"unsupported decode geometry next_n={next_n}, num_rows={num_rows}, "
            f"logits={tuple(logits.shape)}"
        )
    return _decode_selection(logits=logits, seq_lens=seq_lens, output=raw_topk_indices, stock=stock)


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
    def stock(output: torch.Tensor) -> Any:
        return originals[name](
            logits, seq_lens, output, workspace, topk_tokens, max_seq_len
        )

    if not RUNTIME.active:
        return stock(topk_indices)
    return _decode_selection(logits=logits, seq_lens=seq_lens, output=topk_indices, stock=stock)


def install_hooks() -> Installation:
    """Install once, failing closed on a vLLM version or signature mismatch."""

    global _INSTALLATION
    if _INSTALLATION is not None:
        return _INSTALLATION

    import vllm
    from vllm import _custom_ops

    observed_version = str(vllm.__version__).split("+", 1)[0]
    if observed_version != PINNED_VLLM_VERSION:
        raise RuntimeError(
            f"approximate selector hooks require vLLM {PINNED_VLLM_VERSION}, found {vllm.__version__}"
        )
    expected = {
        "prefill": (
            "logits", "cu_seqlen_ks", "cu_seqlen_ke", "raw_topk_indices",
            "num_rows", "stride0", "stride1", "topk_tokens",
        ),
        "decode": (
            "logits", "next_n", "seq_lens", "raw_topk_indices", "num_rows",
            "stride0", "stride1", "topk_tokens",
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
        missing = {
            "cooperative_topk",
            "persistent_topk",
        } - originals.keys()
        if missing:
            raise RuntimeError(f"CUDA selector hook targets are missing: {sorted(missing)}")
    for name in ("prefill", "decode"):
        parameters = tuple(inspect.signature(originals[name]).parameters)
        if parameters != expected[name]:
            raise RuntimeError(
                f"vLLM {name} selector signature is {parameters}, expected {expected[name]}"
            )

    _custom_ops.top_k_per_row_prefill = lambda *args, **kwargs: _prefill_hook(
        originals, *args, **kwargs
    )
    _custom_ops.top_k_per_row_decode = lambda *args, **kwargs: _decode_hook(
        originals, *args, **kwargs
    )
    if "cooperative_topk" in originals:
        torch.ops._C.cooperative_topk = lambda *args, **kwargs: _cuda_decode_hook(
            "cooperative_topk", originals, *args, **kwargs
        )
    if "persistent_topk" in originals:
        torch.ops._C.persistent_topk = lambda *args, **kwargs: _cuda_decode_hook(
            "persistent_topk", originals, *args, **kwargs
        )
    containers = {"custom_ops": _custom_ops, "cuda_ops": torch.ops._C}
    _INSTALLATION = Installation(containers, originals)
    RUNTIME.set_hook_provenance(
        {
            "vllm_version": str(vllm.__version__),
            "pinned_version": PINNED_VLLM_VERSION,
            "symbols": list(originals),
        }
    )
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
