# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-granular sparse attention backend for **GQA** — the one piece vLLM 0.26.0 does not ship.

`docs/qwen3_4b_dsa/serving_eval_plan.md` §2.2. This is plumbing, not a kernel: the attend is one
ordinary FA3 varlen call over a **page-size-1 view** of the standard KV cache, exactly as vLLM's MLA
sparse backend does (`v1/attention/backends/mla/flashattn_mla_sparse.py:238-259`) — minus the two
MLA-specific parts (a cache view hardcoding ``num_kv_heads=1``, and FA3's asymmetric ``q_v`` path for
MLA's K64/V512 geometry). GQA uses the symmetric path, which is strictly simpler.

Every query token becomes its own length-1 sequence over its own gathered key set; causality comes
from the *selection*, not from the kernel mask. ``seqused_k`` truncates each row to its valid prefix,
so ``top_k`` may exceed the sequence length (that is the dense-equivalence control, plan §4 P3.1).

**Measured before this file was written** (`tests/dsa/probe_fa3_sparse_gqa.py`, H100, 2026-08-20):
correctness matches a gather+SDPA reference to <=3.2e-3 rel on six shapes; decode is flat in
sequence length and 13x faster than dense paged decode at 32 reqs x 32K; **prefill is 3.5x slower
than dense at 32K** (21 vs ~680 TFLOP/s) and sorting the block table does not help, so the penalty
is the length-1-sequence restructuring, not coalescing. See the plan's "P0 RESULT" for the
end-to-end arithmetic that makes this a net win at eval concurrency.

Registered as ``AttentionBackendEnum.CUSTOM`` — the sanctioned slot for out-of-tree backends
(`v1/attention/backends/registry.py:233`, `register_backend`). ``Attention(..., attn_backend=...)``
then takes it directly, which is what lets us reuse vLLM's KV-cache allocation, its
``static_forward_context`` registration, cudagraph plumbing and cache-write op unchanged.
"""

import os
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import AttentionSpec

# The compatibility keystone (plan §2.2): plain-FA allocation wants a multiple of 16, the MLA sparse
# backend proves 64, and the DSA indexer's paged-logits kernel wants exactly 64.
DSA_KERNEL_BLOCK_SIZE = 64

# `DSA_DEBUG_SELECTION=N` prints, for the first N attention calls, hard numbers proving the
# indexer's selection is what the attend consumes: how many keys each query actually reads, out of
# how many it could. Answering "are we really sparse, and really using the indexer?" from the
# INSIDE of a running engine, rather than inferring it from a config or a benchmark score. Off by
# default; when on it forces a device sync per call, so it is a diagnostic, not a serving mode --
# and it REQUIRES --enforce-eager (EAGER=1): a sync inside a CUDA-graph-captured region aborts
# capture with `cudaErrorStreamCaptureInvalidated` and the engine never starts. The selection code
# path is identical either way, so the numbers it reports hold for the captured configuration too;
# to check the captured path itself, use the behavioural control (DSA_RANDOM_INDEXER=1).
_DEBUG_SELECTION = int(os.environ.get("DSA_DEBUG_SELECTION", "0"))


def page_size_1_view(cache: torch.Tensor) -> torch.Tensor:
    """``[num_blocks, block_size, H_kv, D]`` -> ``[num_blocks * block_size, 1, H_kv, D]``, no copy.

    ``.view()`` cannot do this: vLLM packs K and V into the trailing dim, so the split halves have a
    head stride of ``2 * D`` and are non-contiguous. ``as_strided`` expresses the same reinterpretation
    (dim 0 walks tokens with the old dim-1 stride; the size-1 page dim's stride is irrelevant), and
    FA3 accepts the result — verified against a contiguous-copy control before this backend was
    written. The alternative, defining a bespoke separate-K/V cache shape, would have diverged from
    the layout vLLM's memory manager and prefix caching expect for a non-MLA backend.
    """
    nb, bs, h, d = cache.shape
    s = cache.stride()
    return torch.as_strided(cache, (nb * bs, 1, h, d), (s[1], s[1], s[2], s[3]))


class Qwen3DSASparseBackend(AttentionBackend):
    """Non-MLA, token-granular sparse attention over a page-size-1 block table."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    # bf16 KV only for v1: fp8 KV is allowed on the dense FA path, but fp8 scales under a page-1
    # view are untested and would fail as slightly-wrong numbers, not as an error (plan §2.2).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16", "float16"]

    @staticmethod
    def get_name() -> str:
        # Must be a member of AttentionBackendEnum, because `Attention.__init__` does
        # `AttentionBackendEnum[self.attn_backend.get_name()]` (attention.py:437).
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [DSA_KERNEL_BLOCK_SIZE]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]  # Qwen3-4B; FA3 also does 32/64/96/192/256 if this is ever reused

    @staticmethod
    def get_impl_cls() -> type["Qwen3DSASparseImpl"]:
        return Qwen3DSASparseImpl

    @staticmethod
    def get_builder_cls() -> type["Qwen3DSASparseMetadataBuilder"]:
        return Qwen3DSASparseMetadataBuilder

    @classmethod
    def is_mla(cls) -> bool:
        return False

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FA3. The MLA sparse backend restricts itself to `major == 9`; we match it rather than
        # guess about Blackwell, where the sparse path goes through CuTe kernels instead.
        return capability.major == 9

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Identical to FlashAttentionBackend (flash_attn.py:131-141): K and V packed into the
        # content dim, logical (B, H, N, 2*D). Staying on the standard layout is deliberate.
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            return (1, 0, 3, 2, 4)
        elif cache_layout == "NHD":
            return (0, 2, 1, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            return (1, 2, 0, 3, 4)
        elif cache_layout == "HND":
            return (0, 1, 2, 3)
        raise ValueError(f"Unknown cache layout format {cache_layout}.")


# `Attention.__init__` resolves the enum member to a class path; register ours into the CUSTOM slot.
register_backend(
    AttentionBackendEnum.CUSTOM,
    "scripts.dsa.vllm_qwen3_dsa_approx.sparse_attention.Qwen3DSASparseBackend",
)


@dataclass
class Qwen3DSASparseMetadata(AttentionMetadata):
    num_actual_tokens: int
    block_table: torch.Tensor  # int32 [num_reqs, max_blocks_per_req]
    req_id_per_token: torch.Tensor  # int32 [num_tokens] -- which request each query token belongs to
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    block_size: int = DSA_KERNEL_BLOCK_SIZE


class Qwen3DSASparseMetadataBuilder(AttentionMetadataBuilder[Qwen3DSASparseMetadata]):
    """Minimal builder: the sparse attend needs only the block table and a token->request map.

    No decode/prefill split and no ``reorder_batch_threshold``: this backend treats **every** token
    identically (one length-1 query sequence each), so there is nothing to reorder. That is a real
    simplification over both templates — the MLA sparse builder carries a chunked-prefill workspace
    and a decode threshold because its base class serves the dense-prefill path too.
    """

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        assert self.block_size == DSA_KERNEL_BLOCK_SIZE, (
            f"Qwen3 DSA requires --block-size {DSA_KERNEL_BLOCK_SIZE}, got {self.block_size}. "
            "The indexer's paged-logits kernel and the top-k -> slot conversion both assume it."
        )
        # Persistent buffer, filled per step, so `build` allocates nothing on the hot path
        # (SparseMLACommonMetadataBuilder does the same, sparse_mla_attention.py:56-60).
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def _req_id_per_token(self, common: CommonAttentionMetadata) -> torch.Tensor:
        starts = np.asarray(common.query_start_loc_cpu, dtype=np.int32)
        seg = np.diff(starts)
        rep = np.repeat(np.arange(seg.shape[0], dtype=np.int32), seg)
        n = rep.shape[0]
        self.req_id_per_token_buffer[:n].copy_(
            torch.from_numpy(rep).pin_memory(), non_blocking=True
        )
        return self.req_id_per_token_buffer[: common.num_actual_tokens]

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> Qwen3DSASparseMetadata:
        return Qwen3DSASparseMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            block_table=common_attn_metadata.block_table_tensor,
            req_id_per_token=self._req_id_per_token(common_attn_metadata),
            slot_mapping=common_attn_metadata.slot_mapping,
            seq_lens=common_attn_metadata.seq_lens,
            block_size=self.block_size,
        )


class Qwen3DSASparseImpl(AttentionImpl):
    """FA3 varlen over the indexer's selected tokens.

    The selection arrives through ``topk_indices_buffer``, a shared preallocated tensor the indexer
    writes and this impl reads. That indirection is not stylistic: on CUDA vLLM wraps attention in an
    opaque custom op (``Attention.use_direct_call`` is False), so there is no way to pass a tensor
    from the model's indexer call into ``forward`` as an argument.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if any((alibi_slopes, sliding_window, logits_soft_cap)):
            raise NotImplementedError(
                "Qwen3DSASparseImpl does not support alibi, sliding window or logit soft cap."
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(f"Unsupported attention type {attn_type}")
        if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
            raise NotImplementedError(
                f"Qwen3DSASparseImpl supports only bf16/fp16 KV cache, got {kv_cache_dtype!r}"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        # This is the sparse path's live wire. Missing it is the failure mode that has cost this
        # project a whole misread eval before (memory `dsa-serving-index-topk-gate`): vLLM's own
        # gate is soft -- an absent `index_topk` merely makes the MLA sparse backend decline, and
        # the engine silently serves dense. Here it is a hard error at construction.
        if topk_indices_buffer is None:
            raise ValueError(
                "Qwen3DSASparseImpl requires topk_indices_buffer -- without the indexer's "
                "selection there is nothing sparse to attend over. Refusing to fall back to dense."
            )
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = topk_indices_buffer.shape[-1]

    _dbg_calls: int = 0

    @torch.no_grad()
    def _log_selection(self, topk_indices, topk_slots, valid_counts, md) -> None:
        """Report what the attend is actually reading. Every number here is measured, not derived.

        ``topk_indices`` is the indexer's output (per-request key indices, -1 padded);
        ``topk_slots`` is the same after conversion to global cache slots; ``valid_counts`` is what
        FA3 receives as ``seqused_k``, i.e. the number of keys each query attends to.
        """
        vc = valid_counts.to(torch.int64)
        sel = (topk_indices >= 0).sum(dim=1)
        seq_max = int(md.seq_lens.max().item()) if md.seq_lens is not None else -1
        row = int(torch.argmax(vc).item())
        uniq = int(torch.unique(topk_slots[row][topk_slots[row] >= 0]).numel())
        width = topk_indices.shape[1]
        print(
            f"[DSA-SELECT] tokens={topk_indices.shape[0]} topk_width={width} "
            f"seqused_k(min/mean/max)={int(vc.min())}/{float(vc.float().mean()):.1f}/{int(vc.max())} "
            f"indexer_selected(min/max)={int(sel.min())}/{int(sel.max())} "
            f"max_seq_len={seq_max} density={int(vc.max()) / max(seq_max, 1):.3f} "
            f"unique_slots_in_widest_row={uniq} "
            f"buffer_all_negative={bool((topk_indices < 0).all().item())}",
            flush=True,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,
        kv_cache: torch.Tensor,  # (num_blocks, num_kv_heads, block_size, 2 * head_size) logical
        attn_metadata: Qwen3DSASparseMetadata,
        output: torch.Tensor,  # [num_tokens, num_heads * head_size]
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for Qwen3DSASparseImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)  # profiling / dummy run

        num_actual_tokens = attn_metadata.num_actual_tokens

        # (B, H, N, 2*D) -> two (B, N, H, D) views, as the dense FA backend does.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # Per-request token indices from the indexer -> global cache slots, plus each row's valid
        # count. Reused verbatim from the MLA sparse path; `BLOCK_SIZE` is a kernel arg, so it is
        # page-size generic and nothing about it is MLA-specific.
        topk_indices = self.topk_indices_buffer[:num_actual_tokens]
        topk_slots, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        # These are mandatory selector-to-FA3 contract checks. They remain live
        # even with telemetry disabled and synchronize only if CUDA reports a
        # violation.
        valid = topk_indices >= 0
        torch._assert_async(
            ~((~valid[:, :-1]) & valid[:, 1:]).any(),
            "Qwen3 DSA approximate indices are not valid-prefix/-1-suffix",
        )
        torch._assert_async(
            (valid.sum(-1).to(valid_counts.dtype) == valid_counts).all(),
            "Qwen3 DSA selector count differs from FA3 valid_counts",
        )
        slot_sentinel = torch.full_like(topk_slots, torch.iinfo(topk_slots.dtype).max)
        sorted_slots = torch.where(valid, topk_slots, slot_sentinel).sort(-1).values
        torch._assert_async(
            ~(
                (sorted_slots[:, 1:] == sorted_slots[:, :-1])
                & (sorted_slots[:, 1:] != slot_sentinel[:, 1:])
            ).any(),
            "Qwen3 DSA approximate selection maps to duplicate global KV slots",
        )

        if _DEBUG_SELECTION and self._dbg_calls < _DEBUG_SELECTION:
            self._dbg_calls += 1
            self._log_selection(topk_indices, topk_slots, valid_counts, attn_metadata)

        cu_seqlens_q = torch.arange(
            0, num_actual_tokens + 1, dtype=torch.int32, device=query.device
        )
        out_view = output.view(-1, self.num_heads, self.head_size)
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=page_size_1_view(key_cache),
            v=page_size_1_view(value_cache),
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=topk_indices.shape[1],
            seqused_k=valid_counts,
            block_table=topk_slots,
            softmax_scale=self.scale,
            # With one query per sequence, FA3's bottom-right causal alignment lets that query see
            # all `seqused_k` selected keys. Causality is already in the selection.
            causal=True,
            fa_version=3,
            out=out_view[:num_actual_tokens],
        )
        return output
