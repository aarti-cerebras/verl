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
"""``MiniCPM3DSAForCausalLM`` — Stage-1 vLLM decode plugin for the Phase-2 DSA
MiniCPM3-4B checkpoint.

Stage 1 (docs/dsa_vllm_minicpm3dsa_build_plan.md "DECODE BUILD PLAN"): the MAIN
attention is now expressed in **MLA-latent form** (``MiniCPM3DSAAttention``,
``use_sparse=False``) instead of the stock dense materialized-QKV path. This is a
numerically-equivalent restructuring (MLA-latent dense == full attention) that
later feeds the sparse selected-set path (Stage 2). Everything else is preserved
by keeping the MiniCPM (muP) shells:

* ``MiniCPM3DSAForCausalLM(MiniCPMForCausalLM)`` — keeps ``scale_width`` / logits.
* ``MiniCPM3DSAModel(MiniCPMModel)`` — keeps ``embed * scale_emb``.
* ``MiniCPM3DSADecoderLayer(MiniCPMDecoderLayer)`` — keeps the residual
  ``scale_depth / sqrt(L)`` scaling; ``_init_attn_block`` builds the MLA attn.

As in Stage 0 we still attach a ``MiniCPM3DSAIndexer`` to every layer's
``self_attn`` so the Phase-2 ``indexer.*`` weights load 1:1; but with
``use_sparse=False`` the indexer's result is UNUSED this stage (the MLA main
attention is dense). Set env ``DSA_WIRE_INDEXER=1`` to additionally build+call the
indexer serve op (populating ``topk_indices_buffer``) — off by default so the
Stage-1 parity gate exercises only the dense MLA path.

``MiniCPM3StockRefForCausalLM`` is a thin parity reference: stock vLLM
``MiniCPM3ForCausalLM`` (dense materialized-QKV) that simply ignores the
``indexer.*`` weights, so the SAME serving checkpoint can be loaded through the
stock dense path and compared against the MLA path (tests/dsa/test_stage1_mla_parity.py).
"""

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.minicpm import (
    MiniCPMDecoderLayer,
    MiniCPMForCausalLM,
    MiniCPMModel,
)
from vllm.model_executor.models.minicpm3 import MiniCPM3ForCausalLM
from vllm.model_executor.models.utils import make_layers
from vllm.platforms import current_platform

from .attention import MiniCPM3DSAAttention
from .indexer import MiniCPM3DSAIndexer


def _sparse_enabled() -> bool:
    """Stage 2b: sparse (indexer + FLASHMLA_SPARSE) is ON by default. Set
    ``DSA_SPARSE=0`` to fall back to the Stage-2a dense-MLA path (used by the
    dense parity reference and the gate-A degeneracy baseline)."""
    return os.environ.get("DSA_SPARSE", "1") not in ("0", "", "false", "False")


def _wire_indexer_enabled() -> bool:
    return os.environ.get("DSA_WIRE_INDEXER", "0") not in ("0", "", "false", "False")


# --------------------------------------------------------------------------- #
# Class hierarchy — MiniCPM (muP) shells, MiniCPM3-DSA MLA attention.
# --------------------------------------------------------------------------- #
class MiniCPM3DSADecoderLayer(MiniCPMDecoderLayer):
    """MiniCPM decoder layer whose ``self_attn`` is the MLA-latent DSA attention.

    ``__init__`` is overridden (vs the stock ``_init_attn_block``-only override) so
    the Stage-2b sparse wiring (``use_sparse`` / shared ``topk_indices_buffer`` /
    live ``vllm_config``) is threaded into ``MiniCPM3DSAAttention`` — the sparse
    backend needs the indexer + buffer at construction time. The muP residual
    scaling (``scale_depth / sqrt(num_hidden_layers)``) in
    ``MiniCPMDecoderLayer.forward`` is inherited UNCHANGED.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        use_sparse: bool = False,
        topk_indices_buffer: torch.Tensor | None = None,
        vllm_config: VllmConfig | None = None,
    ) -> None:
        # Replicate MiniCPMDecoderLayer.__init__ so we can stash the sparse wiring
        # BEFORE _init_attn_block runs (attrs must be set after nn.Module.__init__).
        nn.Module.__init__(self)
        self.config = config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.prefix = prefix
        self._dsa_use_sparse = use_sparse
        self._dsa_topk_buf = topk_indices_buffer
        self._dsa_vllm_config = vllm_config
        self._init_attn_block()
        self._init_ffn_block()

    def _init_attn_block(self):
        self.input_layernorm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )
        self.self_attn = MiniCPM3DSAAttention(
            config=self.config,
            hidden_size=self.hidden_size,
            num_heads=self.config.num_attention_heads,
            qk_nope_head_dim=self.config.qk_nope_head_dim,
            qk_rope_head_dim=self.config.qk_rope_head_dim,
            v_head_dim=self.config.v_head_dim,
            q_lora_rank=self.config.q_lora_rank,
            kv_lora_rank=self.config.kv_lora_rank,
            max_position_embeddings=self.max_position_embeddings,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.self_attn",
            use_sparse=self._dsa_use_sparse,
            vllm_config=self._dsa_vllm_config,
            topk_indices_buffer=self._dsa_topk_buf,
        )


class MiniCPM3DSAModel(MiniCPMModel):
    def _init_layers(
        self,
        prefix: str,
        config: PretrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
    ):
        # Create the shared, model-level topk_indices_buffer BEFORE building the
        # layers and thread it (+ the live vllm_config) down to every attention's
        # indexer — mirrors DeepseekV2Model (deepseek_v2.py:1181-1216). The live
        # config is available via get_current_vllm_config() (vLLM builds models
        # under set_current_vllm_config); NB _init_layers only receives config /
        # cache_config / quant_config. The buffer is a plain tensor attribute (NOT
        # a Parameter/buffer), so it never enters state_dict / named_parameters and
        # is never pickled onto the (picklable) hf_config.
        vllm_config = get_current_vllm_config()
        use_sparse = _sparse_enabled() and hasattr(config, "index_topk")
        self.dsa_use_sparse = use_sparse
        self.topk_indices_buffer = None
        if use_sparse:
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                int(config.index_topk),
                dtype=torch.int32,
                device=current_platform.device_type,
            )

        buf = self.topk_indices_buffer

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MiniCPM3DSADecoderLayer(
                config,
                cache_config,
                quant_config,
                prefix=prefix,
                use_sparse=use_sparse,
                topk_indices_buffer=buf,
                vllm_config=vllm_config,
            ),
            prefix=f"{prefix}.layers",
        )


class MiniCPM3DSAForCausalLM(MiniCPMForCausalLM):
    """MiniCPM3-DSA on vLLM.

    * **Stage 2b (default, ``DSA_SPARSE=1``):** every layer's ``self_attn`` is the
      MLA-latent attention with ``use_sparse=True`` — the lightning indexer runs,
      writes the shared ``topk_indices_buffer``, and the FLASHMLA_SPARSE backend
      attends over the selected set. The indexer is built INSIDE the attention
      (``self_attn.indexer``), so its weights load 1:1 and its
      ``DeepseekV32IndexerCache`` forms the second kv-cache group.
    * **Stage 2a (``DSA_SPARSE=0``):** dense MLA (``use_sparse=False``); an indexer
      is attached post-hoc per layer ONLY so the ``indexer.*`` weights still load.

    muP (``scale_emb`` / ``scale_depth`` / ``scale_width``) is inherited from the
    MiniCPM shells.
    """

    packed_modules_mapping = {
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def _init_model(self, *, vllm_config: VllmConfig, prefix: str = ""):
        return MiniCPM3DSAModel(vllm_config=vllm_config, prefix=prefix)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        # Buffer created by the model (sparse) or None (dense).
        self.topk_indices_buffer = getattr(self.model, "topk_indices_buffer", None)
        self._dsa_sparse = bool(getattr(self.model, "dsa_use_sparse", False))
        self._dsa_wire_indexer = _wire_indexer_enabled()

        # Sparse path: indexers are already built inside each attention module
        # (self_attn.indexer). Nothing to attach here.
        self._dsa_indexer_layers = sum(
            1
            for layer in self.model.layers
            if getattr(getattr(layer, "self_attn", None), "indexer", None) is not None
        )
        if self._dsa_sparse:
            return

        # ---- DENSE (Stage 2a): attach an indexer post-hoc so indexer.* weights
        # load 1:1. use_sparse=False => the indexer result is UNUSED. ----
        ref_param = next(self.model.parameters())
        idx_dtype = ref_param.dtype
        n_heads = int(getattr(config, "dsa_n_heads", getattr(config, "index_n_heads", 16)))
        head_dim = int(getattr(config, "dsa_head_dim", 64))
        rope_head_dim = int(getattr(config, "dsa_rope_head_dim", getattr(config, "qk_rope_head_dim", 32)))
        top_k = int(getattr(config, "index_topk", getattr(config, "dsa_top_k", 256)))
        q_lora_rank = int(config.q_lora_rank)
        hidden_size = int(config.hidden_size)
        fp8 = bool(getattr(config, "dsa_fp8", True))

        cache_config = vllm_config.cache_config
        idx_vllm_config = vllm_config if self._dsa_wire_indexer else None

        for layer in self.model.layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:  # PP/Stage missing layer
                continue
            if getattr(self_attn, "indexer", None) is not None:
                continue
            indexer = MiniCPM3DSAIndexer(
                n_heads=n_heads,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                top_k=top_k,
                q_lora_rank=q_lora_rank,
                hidden_size=hidden_size,
                fp8=fp8,
                dtype=idx_dtype,
                vllm_config=idx_vllm_config,
                cache_config=cache_config,
                topk_indices_buffer=self.topk_indices_buffer,
                prefix=f"{layer.prefix}.self_attn.indexer",
            )
            self_attn.indexer = indexer
            self._dsa_indexer_layers += 1

            if self._dsa_wire_indexer and indexer._serve_ready:
                _install_indexer_forward_hook(self_attn)


class MiniCPM3StockRefForCausalLM(MiniCPM3ForCausalLM):
    """Stock dense MiniCPM3 (materialized-QKV) parity reference.

    Identical to vLLM's ``MiniCPM3ForCausalLM`` except that it silently drops any
    ``indexer.*`` weights present in the checkpoint, so the SAME Phase-2 serving
    directory can be loaded through the stock dense path (no MLA, no indexer) and
    used as the golden reference for the Stage-1 MLA parity test.
    """

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = ((name, w) for name, w in weights if ".indexer." not in name)
        return super().load_weights(filtered)


def _install_indexer_forward_hook(self_attn) -> None:
    """Wrap ``self_attn.forward`` so that, after the normal DENSE MLA attention,
    the attached indexer's serve-path op runs to populate ``topk_indices_buffer``.

    Sub-goal-(b) probe path only. Recomputes ``q_c`` (the compressed query latent)
    cheaply from the same hidden_states; the dense attention output is returned
    unchanged, so LM numerics are identical.
    """
    import types

    orig_forward = self_attn.forward

    def forward_with_indexer(module, positions, hidden_states):
        out = orig_forward(positions, hidden_states)
        indexer = getattr(module, "indexer", None)
        if indexer is not None and getattr(indexer, "_serve_ready", False):
            # q_c = q_a_layernorm(q_a_proj(hidden_states)); matches the MLA forward.
            q_c, _ = module.q_a_proj(hidden_states)
            q_c = module.q_a_layernorm(q_c)
            indexer(hidden_states, q_c, positions, module.rotary_emb)
        return out

    self_attn.forward = types.MethodType(forward_with_indexer, self_attn)
