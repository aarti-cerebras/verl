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
"""``MiniCPM3DSAAttention`` — MiniCPM3 attention expressed in MLA-latent form.

Stage 1 of docs/dsa_vllm_minicpm3dsa_build_plan.md "DECODE BUILD PLAN".

This mirrors vLLM's ``DeepseekV2MLAAttention`` (deepseek_v2.py:855-1040) MLA
wiring — it builds the ``MLAModules`` + a real ``MLAAttention`` op (kv_b_proj
absorption, latent KV cache) — so the same weights that the stock dense
materialized-QKV ``MiniCPM3Attention`` consumes are now run through the MLA path.
Mathematically MLA-latent dense attention == full materialized-QKV attention, so
this is a numerically-equivalent restructuring that later (Stage 2) can feed the
sparse selected-set path.

Differences vs ``DeepseekV2MLAAttention`` (deliberate — MiniCPM3, not DeepSeek):

* **Separate q_a / kv_a projections.** DeepSeek fuses them into
  ``fused_qkv_a_proj`` (deepseek_v2.py:904-910); MiniCPM3's checkpoint keeps
  ``q_a_proj`` and ``kv_a_proj_with_mqa`` SEPARATE, so we keep them separate here
  and load weights 1:1 (no fused-proj remap). Because of this we build the
  ``MLAAttention`` op directly and write our own ``forward`` (the stock
  ``MultiHeadLatentAttentionWrapper.forward`` assumes the fused proj when
  ``q_lora_rank is not None``).
* **MiniCPM3 scaling.** ``scaling = qk_head_dim**-0.5`` with NO YaRN mscale
  (DeepSeek multiplies by ``mscale**2`` at deepseek_v2.py:967-974).
* **longrope, not deepseek_yarn.** ``get_rope`` is called exactly as stock
  MiniCPM3 does (minicpm3.py:121-125): ``is_neox_style`` left at its ``True``
  default (non-interleaved), ``rope_parameters=config.rope_parameters`` (longrope).
  DeepSeek rewrites ``rope_type`` to ``deepseek_yarn`` and forces
  ``is_neox_style=False`` (deepseek_v2.py:953-965) — we do NOT.

The lightning indexer is NOT built here; for Stage 1 the main attention is dense
(``use_sparse=False``) and the indexer is attached post-hoc by
``MiniCPM3DSAForCausalLM.__init__`` (as in Stage 0) purely so its weights load.
"""

import os

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope

# --------------------------------------------------------------------------- #
# Stage 2a — zero-pad MiniCPM3's MLA latent up to DeepSeek's dims.
#
# vLLM's MLA path is hard-locked to DeepSeek latent head sizes:
#   * ``MLACommonBackend.get_supported_head_sizes() == [320, 576]`` and
#     ``MLACommonMetadata.__post_init__`` reject any other ``head_dim``
#     (the metadata ``head_dim`` == ``model_config.get_head_size()``);
#   * every dense MLA backend (FLASH_ATTN_MLA / FLASHMLA / ...) is selected via
#     that same head size.
# MiniCPM3's native MLA head_size is ``kv_lora_rank(256) + qk_rope_head_dim(32)
# = 288`` (∉ {320, 576}) → rejected. We ZERO-PAD the whole MLA path up to
# DeepSeek's ``kv_lora_rank=512`` / ``qk_rope=64`` (⇒ head_size 576) so the
# backend selector, metadata validation, kv-cache spec and the ``MLAAttention``
# op all accept it. Zeros contribute nothing to the MQA dot / weighted-sum and
# the absorbed weights (``kv_b_proj`` → ``W_UK``/``W_UV``) are padded the same
# way, so the result is numerically identical to native-dim MLA. This is the
# exact scheme validated in tests/dsa/probe_flashmla_padding.py.
# --------------------------------------------------------------------------- #
MLA_PAD_KV_LORA = 512  # MiniCPM3 kv_lora_rank 256 -> 512 (nope/value latent)
MLA_PAD_ROPE = 64      # MiniCPM3 qk_rope_head_dim 32 -> 64  (=> head_size 576)
MLA_PAD_HEAD_SIZE = MLA_PAD_KV_LORA + MLA_PAD_ROPE  # 576  (== supported head size)

# --------------------------------------------------------------------------- #
# Stage 2b — turn ON the padded FlashMLA-*sparse* decode path.
#
# vLLM's sparse backend (``FlashMLASparseImpl._bf16_flash_mla_kernel``) hard-pads
# the query head count to a multiple of 64 on Hopper (``prefill_padding``) and
# asserts ``prefill_padding % num_heads == 0`` — i.e. it only accepts ``num_heads``
# that DIVIDE 64 or are a MULTIPLE of 64. MiniCPM3 has 40 heads (neither), which
# the BF16 sparse kernel rejects. The padding probe (probe_flashmla_padding.py
# GOTCHA D) already showed the underlying ``flash_mla_sparse_fwd`` needs
# ``h_q % 64 == 0`` and that zero-q head padding leaves the real heads exact
# (GOTCHA C). So for the SPARSE path we run the whole MLA op at ``num_heads == 64``
# (heads 40->64), zero-padding the q rows and the absorbed ``kv_b_proj``
# (W_UK/W_UV) on the extra 24 heads. The padded heads carry zero q AND zero
# W_UK/W_UV, so they contribute nothing and are sliced off before ``o_proj``.
# The DENSE (Stage-2a) path keeps the native 40 heads (its backend accepts 40).
# --------------------------------------------------------------------------- #
MLA_SPARSE_PAD_NUM_HEADS = 64


def _wrap_pad_input_dim_loader(param: torch.nn.Parameter, native_in: int) -> None:
    """Wrap a linear weight's ``weight_loader`` so a checkpoint weight whose LAST
    (input) dim is ``native_in`` is zero-padded up to the parameter's (larger)
    input dim before the original loader runs.

    Used for ``kv_b_proj`` (native in=256 kv-latent) declared at the padded
    in=512: the real weight lands in ``[..., :256]`` and ``[..., 256:]`` stays
    zero, so the absorbed ``W_UK``/``W_UV`` produce zeros on the padded latent
    dims (matching the zero-padded ``kv_c``). The pad is on the INPUT dim, which
    ``ColumnParallelLinear`` does NOT shard (only the output dim is sharded), so
    delegating to the original loader keeps TP sharding correct.
    """
    orig_loader = param.weight_loader

    def _padded_loader(p, loaded_weight, *args, **kwargs):
        if loaded_weight.shape[-1] == native_in and p.shape[-1] > native_in:
            padded = loaded_weight.new_zeros(
                *loaded_weight.shape[:-1], p.shape[-1]
            )
            padded[..., :native_in] = loaded_weight
            loaded_weight = padded
        return orig_loader(p, loaded_weight, *args, **kwargs)

    param.weight_loader = _padded_loader


def _wrap_pad_kv_b_loader(
    param: torch.nn.Parameter, native_in: int, native_out: int
) -> None:
    """Wrap ``kv_b_proj``'s loader to zero-pad BOTH the input (latent 256->512) AND
    the output (heads 40->64) dims of the checkpoint weight before delegating.

    Used only on the SPARSE path where the MLA op runs at 64 heads. The native
    ``[40*(nope+v), 256]`` weight lands in ``[:native_out, :native_in]`` of the
    declared ``[64*(nope+v), 512]`` param; the extra output rows (padded heads
    40:64) and input cols (padded latent 256:512) stay zero, so the absorbed
    W_UK/W_UV emit zero on the padded heads/latent. The kv_b_proj OUTPUT layout is
    head-major (``head0[nope;v], head1[nope;v], ...``), so the real 40 heads are a
    contiguous prefix ``[0:40*(nope+v)]`` and the head pad is a trailing zero-row
    append — exact for TP=1 (sparse is asserted TP=1 in __init__).
    """
    orig_loader = param.weight_loader

    def _padded_loader(p, loaded_weight, *args, **kwargs):
        lw = loaded_weight
        if lw.shape[-1] == native_in and p.shape[-1] > native_in:
            padded = lw.new_zeros(*lw.shape[:-1], p.shape[-1])
            padded[..., :native_in] = lw
            lw = padded
        if lw.shape[0] == native_out and p.shape[0] > native_out:
            padded = lw.new_zeros(p.shape[0], *lw.shape[1:])
            padded[:native_out] = lw
            lw = padded
        return orig_loader(p, lw, *args, **kwargs)

    param.weight_loader = _padded_loader


class MiniCPM3DSAAttention(nn.Module):
    """MiniCPM3 attention in MLA-latent form (dense; ``use_sparse=False``).

    Weight names match the checkpoint 1:1 (``self_attn.{q_a_proj, q_a_layernorm,
    q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj}``), so a
    Phase-2 MiniCPM3-DSA checkpoint loads with no remap.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # ---- Stage 2b: sparse path ----
        use_sparse: bool = False,
        vllm_config=None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.use_sparse = use_sparse
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        # NATIVE MiniCPM3 rope / latent dims (drive the projections, RoPE, scale).
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # 96 (native)
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank  # 256 (native latent, cached-before-pad)
        self.num_heads = num_heads

        # PADDED dims handed to the MLA op / kv-cache / metadata so the whole MLA
        # path clears vLLM's 576-lock (see MLA_PAD_* note at top of file). Zeros
        # on the padded dims are numerically inert.
        self.kv_lora_rank_pad = max(kv_lora_rank, MLA_PAD_KV_LORA)   # 512
        self.qk_rope_head_dim_pad = max(qk_rope_head_dim, MLA_PAD_ROPE)  # 64

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        # Head count handed to the MLA op. DENSE keeps the native local head count
        # (its backend accepts 40); SPARSE pads to 64 for the FlashMLA-sparse kernel
        # (see MLA_SPARSE_PAD_NUM_HEADS note). Op-level head padding is exact only
        # for TP=1 (real heads must stay a contiguous prefix), which is all Stage-2b
        # tests use; guard it explicitly.
        if self.use_sparse:
            assert tp_size == 1, (
                "Sparse MiniCPM3-DSA head padding (40->64) is only implemented for "
                "tensor_parallel_size=1 (Stage 2b); TP>1 needs per-shard head "
                "padding — deferred to a later stage."
            )
            self.num_op_heads = MLA_SPARSE_PAD_NUM_HEADS
        else:
            self.num_op_heads = self.num_local_heads

        # MiniCPM3 scaling — NO YaRN mscale (contrast DeepSeek deepseek_v2.py:967-974).
        # NOTE: the real MiniCPM3 per-head scale is qk_head_dim**-0.5 == 96**-0.5,
        # computed from the NATIVE dims — NOT the padded 128/576. The probe
        # (probe_flashmla_padding.py GOTCHA A) confirmed the kernel takes the
        # softmax scale as a runtime arg, so passing 96**-0.5 keeps the padded
        # attention numerically identical to native.
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        # ---- SEPARATE q_a / kv_a projections (checkpoint-native; NOT fused) ----
        self.q_a_proj = ReplicatedLinear(
            self.hidden_size, self.q_lora_rank, bias=False, quant_config=quant_config,
            prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        # kv_a_layernorm stays NATIVE (256) — it normalises the real cached latent
        # before we zero-pad it to 512 in forward().
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        # kv_b_proj (the absorbed W_UK/W_UV source) is declared at the PADDED input
        # dim (512) so MLAAttention.process_weights_after_loading sees
        # kv_lora_rank == 512; the native [.., 256] checkpoint weight is zero-padded
        # to [.., 512] on load (cols 256: == 0), so W_UK/W_UV emit zeros on the
        # padded latent dims — matching the zero-padded kv_c. On the SPARSE path the
        # OUTPUT dim is ALSO padded (heads 40->64): the op runs at 64 heads and the
        # padded head rows stay zero (=> zero W_UK/W_UV => zero contribution).
        native_kv_b_out = self.num_local_heads * (self.qk_nope_head_dim + self.v_head_dim)
        op_kv_b_out = self.num_op_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank_pad,
            op_kv_b_out,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        if self.use_sparse:
            _wrap_pad_kv_b_loader(
                self.kv_b_proj.weight,
                native_in=self.kv_lora_rank,
                native_out=native_kv_b_out,
            )
        else:
            _wrap_pad_input_dim_loader(self.kv_b_proj.weight, native_in=self.kv_lora_rank)
        # o_proj stays NATIVE (40 heads): the sparse op's padded heads (40:64) are
        # sliced off before o_proj (their attention output is zero anyway).
        self.o_proj = RowParallelLinear(
            self.num_local_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # longrope, non-interleaved — exactly like stock MiniCPM3 (minicpm3.py:121-125).
        # is_neox_style defaults to True; do NOT rewrite rope_type to deepseek_yarn.
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )

        # ---- lightning indexer (Stage 2b: built + wired only on the sparse path) ----
        # Must exist BEFORE the MLAAttention op: the sparse backend reads
        # ``indexer.topk_indices_buffer`` at construction and selects
        # FLASHMLA_SPARSE via ``use_sparse=True``. Built with a live vllm_config so
        # its DeepseekV32IndexerCache registers the SECOND kv-cache group and its
        # SparseAttnIndexer serve op is runnable. Weights load 1:1 under
        # ``self_attn.indexer.*``.
        self.indexer = None
        if self.use_sparse:
            from .indexer import MiniCPM3DSAIndexer

            self.indexer = MiniCPM3DSAIndexer(
                n_heads=int(
                    getattr(config, "dsa_n_heads", getattr(config, "index_n_heads", 16))
                ),
                head_dim=int(getattr(config, "dsa_head_dim", 64)),
                rope_head_dim=int(
                    getattr(config, "dsa_rope_head_dim", self.qk_rope_head_dim)
                ),
                top_k=int(
                    getattr(config, "index_topk", getattr(config, "dsa_top_k", 256))
                ),
                q_lora_rank=self.q_lora_rank,
                hidden_size=self.hidden_size,
                fp8=bool(getattr(config, "dsa_fp8", True)),
                dtype=torch.get_default_dtype(),
                vllm_config=vllm_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                prefix=f"{prefix}.indexer",
            )
            assert self.indexer._serve_ready, (
                "sparse path requires a runtime indexer (vllm_config must be "
                "supplied so the SparseAttnIndexer op + DeepseekV32IndexerCache "
                "are built)."
            )

        # ---- MLA op ----
        # Built with PADDED latent/rope dims (kv_lora 512, rope 64 => head_size
        # 576) so backend selection + metadata + kv-cache spec all clear the
        # 576-lock; the real MiniCPM3 scale (96**-0.5) is still passed through.
        # SPARSE: num_heads=64 (padded) + use_sparse=True + indexer => the engine
        # builds the FLASHMLA_SPARSE backend and reads topk_indices_buffer.
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=self.indexer,
            indexer_rotary_emb=self.rotary_emb if self.use_sparse else None,
            is_sparse=self.use_sparse,
            topk_indices_buffer=topk_indices_buffer,
        )
        # Keep the modules dataclass around for introspection / Stage 2 reuse.
        self.mla_modules = mla_modules

        self.mla_attn = MLAAttention(
            num_heads=self.num_op_heads,  # 40 (dense) / 64 (sparse, padded)
            scale=self.scaling,  # 96**-0.5 (native MiniCPM3), NOT a 576-based scale
            qk_nope_head_dim=self.qk_nope_head_dim,       # 64 (native)
            qk_rope_head_dim=self.qk_rope_head_dim_pad,   # 64 (padded 32->64)
            v_head_dim=self.v_head_dim,                   # 64 (native)
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank_pad,           # 512 (padded 256->512)
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.use_sparse,
            indexer=self.indexer,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]
        H = self.num_local_heads

        # q latent -> q_a_layernorm -> q_b_proj  (MiniCPM3 separate q_a path).
        q_c, _ = self.q_a_proj(hidden_states)
        q_c = self.q_a_layernorm(q_c)
        q, _ = self.q_b_proj(q_c)

        # kv latent (compressed kv_c + k_pe), separate proj — NATIVE dims.
        kv_lora, _ = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())  # [T, 256]

        q = q.view(T, H, self.qk_head_dim)  # [T, H, 96] (native)

        # RoPE on the pe slice only (longrope, neox/non-interleaved), on the
        # NATIVE 32-d rope slice. vLLM's neox rope forward expects the heads
        # FLATTENED into the last dim (``[T, H*rope]`` query / ``[T, rope]`` key) —
        # exactly how stock MiniCPM3 calls it (minicpm3.py:156-159). Passing the
        # unflattened ``[T, H, rope]`` (as the DeepSeek MLA wrapper does for its
        # NON-neox rope) breaks here.
        q_pe = q[..., self.qk_nope_head_dim :].reshape(
            T, H * self.qk_rope_head_dim
        )
        k_pe = k_pe.reshape(T, self.qk_rope_head_dim)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_pe = q_pe.view(T, H, self.qk_rope_head_dim)  # [T, H, 32]
        k_pe = k_pe.view(T, self.qk_rope_head_dim)     # [T, 32]

        # ---- run the lightning indexer (Stage 2b) BEFORE the MLA op ----
        # It writes the shared ``topk_indices_buffer`` that the FLASHMLA_SPARSE
        # backend reads inside self.mla_attn (mirrors the DeepSeek MLA wrapper's
        # ``self.indexer(...)`` then ``self.mla_attn(...)`` ordering). ``q_c`` is the
        # compressed+layernormed query latent (the indexer's qr). Result unused
        # here — the side effect (topk_indices_buffer) is what matters.
        if self.use_sparse:
            self.indexer(hidden_states, q_c, positions, self.rotary_emb)
            # DEBUG/CONTROL knob: force an EFFECTIVE top-k of DSA_FORCE_TOPK by
            # keeping only the first k selected indices per query and masking the
            # rest to -1 (the kernel's "no key" sentinel, already handled for
            # short sequences). Lets us probe extreme-low top_k (e.g. 2) that the
            # FlashMLA kernel width (multiple of 128) can't express directly.
            # Default 0 => no-op (normal top_k from config). See top_k sweep.
            _force_k = int(os.environ.get("DSA_FORCE_TOPK", "0") or "0")
            if _force_k > 0 and self.indexer.topk_indices_buffer is not None:
                self.indexer.topk_indices_buffer[:, _force_k:] = -1

        # ---- zero-pad the MLA path to 576/512 (see MLA_PAD_* note) and, on the
        # SPARSE path, pad the head count 40->64 (see MLA_SPARSE_PAD_NUM_HEADS). ----
        # q_pad [T, num_op_heads, qk_nope(64) + rope_pad(64) = 128]:
        #   heads [0:H]:  [.., :64]=q_nope, [.., 64:96]=roped q_pe, [.., 96:128]=0
        #   heads [H:Hop]: 0  (zero q => zero attention contribution, sliced off)
        rope_pad = self.qk_rope_head_dim_pad
        qk_head_dim_pad = self.qk_nope_head_dim + rope_pad
        Hop = self.num_op_heads
        q_pad = q.new_zeros(T, Hop, qk_head_dim_pad)
        q_pad[:, :H, : self.qk_nope_head_dim] = q[..., : self.qk_nope_head_dim]
        q_pad[:, :H, self.qk_nope_head_dim : self.qk_nope_head_dim + self.qk_rope_head_dim] = q_pe

        # kv_c_pad [T, 512]: [:256] = kv_c_normed, [256:] = 0
        kv_c_pad = kv_c_normed.new_zeros(T, self.kv_lora_rank_pad)
        kv_c_pad[:, : self.kv_lora_rank] = kv_c_normed

        # k_pe_pad [T, 1, 64]: [.., :32] = roped k_pe, [.., 32:] = 0
        k_pe_pad = k_pe.new_zeros(T, 1, rope_pad)
        k_pe_pad[:, 0, : self.qk_rope_head_dim] = k_pe

        attn_out = self.mla_attn(
            q_pad,
            kv_c_pad,
            k_pe_pad,
            output_shape=(T, Hop * self.v_head_dim),
        )
        # slice the real 40 heads off the padded (64-head) output before o_proj;
        # padded heads carry zero (their W_UV is zero). No-op on the dense path.
        if Hop != H:
            attn_out = attn_out[:, : H * self.v_head_dim]
        output, _ = self.o_proj(attn_out)
        return output
