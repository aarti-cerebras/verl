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
"""``Qwen3MSAForCausalLM`` — Qwen3-4B with MiniMax Sparse Attention, on vLLM 0.26.0.

P3 of docs/qwen3_4b_msa/serving_plan.md, modelled on ``scripts/dsa/vllm_minicpm3_dsa/model.py``.

**Everything sparse is reused, not written** (serving_plan §3). ``MiniMaxM3SparseAttention``
reads only generic config fields plus ``config.sparse_attention_config``, and builds
``self.impl`` (block-sparse attend) and ``self.indexer`` (score + top-k + its own side KV
cache) from plain integers — so it takes our Qwen3 geometry unmodified, including
``rope_theta``/``partial_rotary_factor`` which it reads flat off the config (which is why the
P2 exporter writes them flat; tf5 nests them under ``rope_parameters``). We therefore use that
class **directly**: no subclass, no attention.py. What this file adds is the Qwen3-shaped
scaffolding M3's own decoder layer cannot provide — dense MLP instead of MoE, standard
RMSNorms, stock ``Qwen3Attention`` on the dense prefix, and the shared top-k buffer.

Three details that are load-bearing and easy to get wrong:

1. **Do not build a dense attention on sparse layers and then replace it.** ``Attention``
   registers itself into ``compilation_config.static_forward_context`` at construction, so a
   discarded one leaves a stale entry and vLLM allocates KV cache for a layer that no longer
   exists. Hence ``Qwen3MSADecoderLayer.__init__`` replicates ``Qwen3DecoderLayer.__init__``
   and *chooses* the attention class, rather than calling ``super().__init__()``. Same reason
   ``MiniCPM3DSADecoderLayer`` replicates its parent (``vllm_minicpm3_dsa/model.py:101-113``).

2. **The top-k buffer is created once at model level, before the layers**, and threaded down —
   mirroring ``MiniCPM3DSAModel._init_layers`` and M3's own ``nvidia/model.py:795-806``. Shape
   ``[pad4(max_num_batched_tokens), num_index_heads, topk_blocks]`` int32, token-major, padded
   to a multiple of 4 for ``build_k2q_csr``'s int4 loads. It is a plain tensor attribute, never
   a Parameter/buffer, so it stays out of ``state_dict``.

3. **The weight mapper must fold ``index_q_proj``/``index_k_proj`` into the fused
   ``qkv_proj``.** Qwen3 loads via ``AutoWeightsLoader`` + ``hf_to_vllm_mapper``; we extend that
   mapper with the two index entries, matching what M3 does through its explicit
   ``stacked_params_mapping`` (``nvidia/model.py:902-911``). Without them the 132 index tensors
   are silently dropped and the model serves dense (serving_plan §2.2).

Env knob: ``MSA_SPARSE=0`` builds every layer dense (indexer never constructed) — the
staged-bring-up switch, mirroring ``DSA_SPARSE``. Used by S0 to separate "does the plugin load
and generate" from "do the sparse kernels work".
"""

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3MLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    maybe_prefix,
)
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors


def _sparse_enabled() -> bool:
    """``MSA_SPARSE=0`` -> every layer dense (no indexer built). Mirrors ``DSA_SPARSE``."""
    return os.environ.get("MSA_SPARSE", "1") not in ("0", "", "false", "False")


def _ensure_m3_rope_fields(config: PretrainedConfig) -> None:
    """Give the live config the two FLAT rope attributes ``MiniMaxM3SparseAttention`` reads.

    It builds RoPE from ``config.rope_theta`` and ``config.partial_rotary_factor``
    (``nvidia/model.py:468-475``) — fine for ``MiniMaxM3Config``, which keeps them as attributes.
    ``Qwen3Config`` under transformers 5 does NOT: it folds rope settings into the nested
    ``rope_parameters`` dict and drops the flat key, so ``config.rope_theta`` raises even though
    P2 writes it into config.json. (That write is still worth keeping — it makes the serving dir
    self-describing — but it is not what makes this work.)

    Qwen3 is full-rotary, so ``partial_rotary_factor = 1.0`` gives ``rotary_dim = head_dim = 128``
    — the setting P1 validated the fused kernel at.
    """
    rope = getattr(config, "rope_parameters", None) or {}
    if not hasattr(config, "rope_theta"):
        theta = rope.get("rope_theta")
        assert theta is not None, "config has neither rope_theta nor rope_parameters.rope_theta"
        config.rope_theta = float(theta)
    if not hasattr(config, "partial_rotary_factor"):
        config.partial_rotary_factor = float(rope.get("partial_rotary_factor", 1.0))



def _consumers_transpose_topk_buffer() -> bool:
    """Does this vLLM transpose `topk_indices_buffer` itself before handing it to the kernels?

    True  -> upstream is fixed; allocate token-major exactly as `nvidia/model.py` does.
    False -> vLLM 0.26.0; we must supply the head-major view (see the call site).

    Detected from the source rather than `vllm.__version__` so that an upgrade flips this
    automatically instead of silently double-transposing.
    """
    import inspect

    try:
        from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerTritonImpl

        return "transpose(0, 1)" in inspect.getsource(MiniMaxM3IndexerTritonImpl.forward)
    except Exception:  # noqa: BLE001 - unknown layout: assume unfixed, matching 0.26.0
        return False


def sparse_layer_ids(config: PretrainedConfig) -> set[int]:
    """Layer ids carrying an index branch, read the way vLLM reads it.

    Identical semantics to ``vllm/models/minimax_m3/nvidia/model.py:95-103`` — an absent
    ``sparse_attention_config`` or ``sparse_attention_freq`` yields the EMPTY SET, i.e. an
    entirely dense model. That is the silent-dense-fallback failure (serving_plan §7.3); the
    model asserts against it below rather than serving a fluent dense Qwen3.
    """
    cfg = getattr(config, "sparse_attention_config", None)
    if not cfg:
        return set()
    freq = cfg.get("sparse_attention_freq")
    if freq is None:
        return set()
    return {i for i, f in enumerate(freq) if f != 0}


class Qwen3MSADecoderLayer(nn.Module):
    """Qwen3 decoder layer whose ``self_attn`` is sparse on the non-prefix layers.

    Replicates ``Qwen3DecoderLayer`` (vllm/model_executor/models/qwen3.py:173-245) with the
    attention class chosen per layer; ``forward`` is byte-for-byte the parent's.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        sparse_ids: set[int] | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        layer_id = extract_layer_index(prefix)
        self.layer_id = layer_id
        self.is_sparse = bool(sparse_ids) and layer_id in sparse_ids

        if self.is_sparse:
            # Reused verbatim: owns qkv_proj (fused with index_q/index_k), Gemma QK norms,
            # RoPE, the fused kernel, self.impl (attend) and self.indexer (score + top-k +
            # side cache). Reads rope_theta / partial_rotary_factor flat off the config.
            self.self_attn = MiniMaxM3SparseAttention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
            )
        else:
            # Dense prefix (layers [0, msa_dense_prefix)) — exactly as they trained.
            self.self_attn = Qwen3Attention(
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                max_position=config.max_position_embeddings,
                num_kv_heads=config.num_key_value_heads,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                head_dim=getattr(config, "head_dim", None),
                cache_config=cache_config,
                quant_config=quant_config,
                rope_parameters=config.rope_parameters,
                prefix=f"{prefix}.self_attn",
            )

        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3MSAModel(Qwen2Model):
    """Qwen3 model that allocates the shared top-k buffer before building the layers."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            **Qwen2Model.hf_to_vllm_mapper.orig_to_new_stacked,
            # Fold the index branch into the same fused GEMM, as M3 does via
            # stacked_params_mapping (nvidia/model.py:907-908). NB no collision with the
            # ".q_proj" entry: ".index_q_proj" does not contain the substring ".q_proj".
            ".index_q_proj": (".qkv_proj", "index_q"),
            ".index_k_proj": (".qkv_proj", "index_k"),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config.get_text_config()
        sparse_ids = sparse_layer_ids(config) if _sparse_enabled() else set()
        if sparse_ids:
            _ensure_m3_rope_fields(config)  # must precede layer construction
        self.msa_sparse_ids = sparse_ids

        buf = None
        if sparse_ids:
            sac = config.sparse_attention_config
            # Token-major [pad4(max_num_batched_tokens), num_index_heads, topk], int32
            # (nvidia/model.py:795-806). Plain attribute, NOT a Parameter/buffer, so it never
            # enters state_dict or gets pickled onto hf_config (DSA model.py:152-154).
            max_toks = vllm_config.scheduler_config.max_num_batched_tokens
            n_idx_heads = int(sac["sparse_num_index_heads"])
            buf = torch.empty(
                (max_toks + 3) // 4 * 4,
                n_idx_heads,
                int(sac["sparse_topk_blocks"]),
                dtype=torch.int32,
                device=current_platform.device_type,
            )
            if not _consumers_transpose_topk_buffer():
                # vLLM 0.26.0 BUG: `nvidia/model.py` allocates this buffer token-major
                # [tokens, heads, topk], but BOTH consumers index it head-major --
                # `indexer.py` writes `out=buf[:, nd:, :]` and `sparse_attention.py` reads
                # `topk[:, :nd, :]`, where `nd` is a DECODE TOKEN COUNT. Neither transposes.
                # Fixed after 0.26.0 by adding the transposes upstream; we hand them the
                # already-transposed view, which is bit-identical (contiguous [T,H,K] has
                # strides (H*K, K, 1); .transpose(0,1) gives [H,T,K] strides (K, H*K, 1) --
                # exactly the tensor the fixed upstream builds).
                #
                # Symptom if wrong in EITHER direction: `CUDA error: an illegal memory access`
                # once a decode batch exceeds num_index_heads (8 here) -- so it is invisible to
                # single-request testing and only appears under real concurrency. Guarded by
                # source inspection rather than a version string so an upgrade cannot silently
                # double-transpose; see tests/msa/test_concurrency_regression.py.
                buf = buf.transpose(0, 1)
                assert buf.shape[0] == n_idx_heads, "pre-transpose left the wrong layout"
        self.topk_indices_buffer = buf

        def _layer(config, cache_config, quant_config, prefix):
            return Qwen3MSADecoderLayer(
                config, cache_config, quant_config, prefix,
                sparse_ids=sparse_ids, topk_indices_buffer=buf,
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=_layer)


class Qwen3MSAForCausalLM(nn.Module, SupportsPP):
    """Qwen3-4B MSA on vLLM. Mirrors ``Qwen3ForCausalLM`` with our model and weight mapper."""

    hf_to_vllm_mapper = Qwen3MSAModel.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj", "index_q_proj", "index_k_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        self.model = Qwen3MSAModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        n_sparse = len(getattr(self.model, "msa_sparse_ids", set()))
        if _sparse_enabled():
            # serving_plan §7.3 check: an absent/mis-shaped sparse_attention_config yields an
            # empty set -> a fluent, benchmark-passing, entirely DENSE model. Refuse to build it.
            assert n_sparse > 0, (
                "MSA_SPARSE=1 but sparse_attention_config produced no sparse layers -- the "
                "serving dir's config.json is missing 'sparse_attention_config' or its "
                "'sparse_attention_freq' (serving_plan §5.2). Refusing to serve dense silently."
            )
        self.msa_num_sparse_layers = n_sparse

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size, config.hidden_size,
                    quant_config=self.quant_config, prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Strict load — a missing/unexpected key must raise, not be skipped.

        M3's own loader ends every branch in ``if name not in params_dict: continue``
        (nvidia/model.py:940-941), which is exactly how all 132 index tensors can vanish in
        silence. ``AutoWeightsLoader`` raises instead (serving_plan §4.2, §8 R4).
        """
        if not _sparse_enabled():
            # MSA_SPARSE=0 bring-up mode ONLY: every layer is a stock Qwen3Attention, which has no
            # index_* parameters, so a strict load of a checkpoint containing them must fail. Drop
            # them, exactly as MiniCPM3DSA's dense path arranges for its indexer weights to be
            # accounted for (vllm_minicpm3_dsa/model.py:229-265).
            # NB this is a DEBUG switch to isolate "scaffolding + registration" from "sparse
            # kernels" -- it is NOT a servable configuration. Serving it would silently evaluate a
            # dense model, which is the failure §6.6 exists to prevent.
            # Also UNDO P2's w-1 shift: the dense Qwen3Attention uses standard RMSNorm (x*w), but
            # the export stored Gemma-convention (w-1) for the fused kernel. Without this the dense
            # path computes x*(w-1) and emits fluent-looking garbage -- which makes the mode useless
            # as a diagnostic, since you cannot tell a broken scaffold from the expected garbage.
            shift = bool(getattr(self.config, "msa_norm_shift_applied", False))
            sparse = sparse_layer_ids(self.config)

            def _fix(ws):
                for n, w in ws:
                    if ".index_" in n:
                        continue
                    if shift and n.endswith((".q_norm.weight", ".k_norm.weight")):
                        try:
                            if int(n.split("layers.")[1].split(".")[0]) in sparse:
                                w = (w.float() + 1.0).to(w.dtype)
                        except (IndexError, ValueError):
                            pass
                    yield n, w

            weights = _fix(weights)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
