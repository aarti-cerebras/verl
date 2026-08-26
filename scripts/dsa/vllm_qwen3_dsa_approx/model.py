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
"""Isolated approximate-selector copy of Qwen3 DSA serving on vLLM 0.26.0.

`docs/qwen3_4b_dsa/serving_eval_plan.md` §2. Structure follows ``scripts/msa/vllm_qwen3_msa/model.py``
(the proven in-repo pattern); the sparse machinery is the isolated
``vllm_qwen3_dsa_approx/{indexer,sparse_attention}.py`` snapshot.

Per layer the replaced region is exactly: hidden states in -> attention output. Embeddings, MLP,
layernorms, ``lm_head`` and sampling are stock Qwen3.

**Which layers are sparse is read from the config, not assumed.** Like MSA, DSA supports a dense
prefix: layers in ``[0, dsa_dense_prefix)`` keep stock full attention and carry no indexer. The
serving dir's ``config.json`` holds the resolved list in ``dsa_sparse_layer_ids``, written by
``build_qwen3_dsa_serving_dir.py`` from the training config, and ``sparse_layer_ids()`` below only
looks it up. Re-deriving it here would let training and serving disagree, which presents as a quality
regression and nothing else. Older serving dirs have no such key and are all-sparse.

Three details that are load-bearing:

1. **Never build a dense ``Attention`` and then replace it.** ``Attention.__init__`` registers itself
   in ``compilation_config.static_forward_context``; a discarded one leaves a stale entry and vLLM
   allocates KV cache for a layer that no longer exists. So ``Qwen3DSAAttention`` *replicates*
   ``Qwen3Attention.__init__`` rather than calling it -- same reason the MSA and MiniCPM3 plugins
   replicate their parents.
2. **The indexer runs before ``self.attn``, and communicates through the shared buffer.** On CUDA
   vLLM wraps attention in an opaque custom op, so the selection cannot be passed as an argument;
   the indexer writes ``topk_indices_buffer`` and the impl reads it.
3. **The weights load 1:1 with no mapper for the index branch.** Training nests the branch in an
   ``indexer`` submodule (``self_attn.indexer.wq`` etc.) and our module has the same shape, so --
   unlike MSA, whose index projections fold into a fused GEMM -- nothing needs renaming. The
   exporter only has to consolidate the FSDP shards.

Env knob ``DSA_SPARSE=0`` builds every layer as stock dense ``Qwen3Attention`` with no indexer: the
staged-bring-up switch that separates "does the plugin load and generate" from "do the sparse kernels
work". It is a DEBUG mode, not a servable configuration -- serving it would silently evaluate a dense
model, which is the exact failure the hard gates below exist to prevent.
"""

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Attention, Qwen3MLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .indexer import Qwen3DSAApproxServingIndexer
from .selector_runtime import RUNTIME, config_from_hf
from .sparse_attention import DSA_KERNEL_BLOCK_SIZE, Qwen3DSASparseBackend


def _sparse_enabled() -> bool:
    """``DSA_SPARSE=0`` -> dense bring-up mode (no indexer built anywhere)."""
    return os.environ.get("DSA_SPARSE", "1") not in ("0", "", "false", "False")


def sparse_layer_ids(config: PretrainedConfig) -> set[int]:
    """The layers that carry an indexer, read from the serving dir's ``config.json``.

    The builder writes the RESOLVED id list (``dsa_sparse_layer_ids``), so this is a lookup, not a
    reimplementation of training's ``dense_prefix`` predicate -- the two agreeing is the whole contract.
    A config without the key predates ``dense_prefix`` and means every layer is sparse, which is what
    those checkpoints actually are.
    """
    n_layers = int(config.num_hidden_layers)
    ids = getattr(config, "dsa_sparse_layer_ids", None)
    if ids is None:
        prefix = int(getattr(config, "dsa_dense_prefix", 0) or 0)
        assert prefix == 0, (
            f"config.dsa_dense_prefix={prefix} but dsa_sparse_layer_ids is missing. Rebuild the serving "
            "dir with build_qwen3_dsa_serving_dir.py -- guessing the layer set here is how a model gets "
            "served dense where it was trained sparse."
        )
        return set(range(n_layers))
    out = {int(i) for i in ids}
    assert out and max(out) < n_layers and min(out) >= 0, (
        f"config.dsa_sparse_layer_ids={sorted(out)[:8]}... out of range for {n_layers} layers"
    )
    return out


def _random_indexer() -> bool:
    """``DSA_RANDOM_INDEXER=1`` -> replace the trained indexer with random weights of the same scale.

    This is eval-ladder row 4 (eval_plan.md §3), the control that proves the model is *genuinely*
    sparse: with a random selector the model must **collapse**. If it scores like the trained one,
    the selection is not reaching the attend and we are measuring a dense model. Combined with row 5
    (random indexer at ``top_k >= T``, which must recover), it separates "selection is broken" from
    "the sparse machinery is broken".
    """
    return os.environ.get("DSA_RANDOM_INDEXER", "0") not in ("0", "", "false", "False")


def dsa_params(config: PretrainedConfig) -> dict[str, Any]:
    """Indexer geometry from the flat ``dsa_*`` keys the training config carries.

    Defaults mirror ``verl/models/transformers/qwen3_dsa_indexer.py::Qwen3DSAConfig`` so a config
    written by an older run still resolves to what that run actually trained -- except for
    ``dsa_top_k``, which has no safe default and must be present.
    """
    rope = getattr(config, "rope_parameters", None) or {}
    theta = getattr(config, "rope_theta", None) or rope.get("rope_theta")
    assert theta is not None, "config has neither rope_theta nor rope_parameters.rope_theta"
    top_k = getattr(config, "dsa_top_k", None)
    assert top_k is not None, "config.dsa_top_k missing -- cannot guess the trained top-k"
    return dict(
        hidden_size=int(config.hidden_size),
        n_heads=int(getattr(config, "dsa_n_heads", 16)),
        head_dim=int(getattr(config, "dsa_head_dim", 64)),
        rope_head_dim=int(getattr(config, "dsa_rope_head_dim", 64)),
        rope_theta=float(theta),
        top_k=int(top_k),
        capacity=int(getattr(config, "index_topk", top_k)),
        fp8=bool(getattr(config, "dsa_fp8", True)),
        fp8_ue8m0=bool(getattr(config, "dsa_fp8_ue8m0", True)),
        rotate_activation=bool(getattr(config, "dsa_rotate_activation", True)),
    )


def assert_servable_sparse(config: PretrainedConfig) -> None:
    """Refuse to serve a config that would silently produce a dense model.

    Three checks, each of which has burned this project or its MiniCPM3 predecessor:
      * ``dsa_enabled`` absent -> the checkpoint is not a DSA checkpoint at all;
      * ``dsa_mode != "sparse"`` -> a Phase-1 (dense-warmup) checkpoint, whose LM path is
        bit-identical to stock Qwen3 by construction, so benchmarks would look *fine*;
      * ``index_topk`` absent -> the documented vLLM sparse gate. vLLM's own use of it is SOFT
        (a missing key merely makes the MLA sparse backend decline and the engine picks a dense
        one -- memory ``dsa-serving-index-topk-gate``). Ours is hard, and it must agree with
        ``dsa_top_k`` so the config cannot describe one budget while serving another.
    """
    assert getattr(config, "dsa_enabled", False), (
        "config.dsa_enabled is not set -- this serving dir was not built from a DSA checkpoint"
    )
    mode = getattr(config, "dsa_mode", None)
    assert mode == "sparse", (
        f"config.dsa_mode={mode!r}; only 'sparse' is servable. A 'dense_warmup' (Phase-1) "
        "checkpoint computes the stock dense LM function -- serving it would produce baseline "
        "scores that read as success."
    )
    top_k = int(getattr(config, "dsa_top_k"))
    index_topk = getattr(config, "index_topk", None)
    assert index_topk is not None, (
        "config.index_topk missing. Add it in the serving dir (build_qwen3_dsa_serving_dir.py "
        "writes it): it is the documented sparse gate, and its absence is how a sparse model "
        "silently serves dense."
    )
    assert int(index_topk) >= top_k, (
        f"index_topk ({index_topk}) < dsa_top_k ({top_k}) -- capacity cannot hold logical k"
    )
    selector = str(getattr(config, "dsa_selector", "topk"))
    if selector in ("topk", "radix_ceil"):
        assert int(index_topk) == top_k, (
            f"selector {selector!r} requires index_topk == dsa_top_k; got "
            f"{index_topk} != {top_k}"
        )


class Qwen3DSAAttention(nn.Module):
    """Qwen3 GQA attention whose attend runs over the indexer's top-k tokens.

    Replicates ``Qwen3Attention`` (vllm/model_executor/models/qwen3.py) with two changes: ``self.attn``
    is constructed with our sparse backend and the shared top-k buffer, and ``forward`` runs the
    indexer first.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
        topk_indices_buffer: torch.Tensor,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", None) or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # The sparse attend. `attn_backend=` bypasses vLLM's backend selector (which is MLA-gated
        # for sparse) while keeping its KV-cache allocation, forward-context registration and
        # cudagraph plumbing; `topk_indices_buffer` reaches the impl through **extra_impl_args.
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
            attn_backend=Qwen3DSASparseBackend,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.indexer = Qwen3DSAApproxServingIndexer(
            **dsa_params(config),
            vllm_config=self._vllm_config(),
            cache_config=cache_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.indexer",
        )

    @staticmethod
    def _vllm_config() -> VllmConfig:
        from vllm.config import get_current_vllm_config

        return get_current_vllm_config()

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        # Selection first: it writes topk_indices_buffer, which self.attn reads inside the opaque
        # attention op. Same hidden states the training indexer sees (post input_layernorm).
        self.indexer(hidden_states, positions)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DSADecoderLayer(nn.Module):
    """Qwen3 decoder layer; ``self_attn`` is sparse unless ``DSA_SPARSE=0``."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        sparse: bool = True,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = extract_layer_index(prefix)
        self.is_sparse = sparse

        if sparse:
            assert topk_indices_buffer is not None
            self.self_attn: nn.Module = Qwen3DSAAttention(
                config, cache_config, quant_config, f"{prefix}.self_attn", topk_indices_buffer
            )
        else:
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
class Qwen3DSAModel(Qwen2Model):
    """Qwen3 model that allocates the shared top-k buffer before building the layers."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config.get_text_config()
        sparse = _sparse_enabled()
        if sparse:
            assert_servable_sparse(config)
            selector_config = config_from_hf(config)
            selector_config.validate_execution(
                cudagraph_enabled=(
                    vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                )
            )
            RUNTIME.configure(selector_config)
        self.dsa_sparse = sparse
        # Per-layer: `dense_prefix` layers keep stock full attention. Empty in DSA_SPARSE=0 bring-up mode,
        # where nothing is sparse.
        self.dsa_sparse_layers = sparse_layer_ids(config) if sparse else set()

        buf = None
        if sparse:
            # [max_num_batched_tokens, top_k] int32, one row per QUERY TOKEN -- DSA selects per
            # token, unlike MSA which selects per (token, index head). Same shape stock vLLM
            # allocates for DeepSeek-V3.2 (deepseek_v2.py:1362-1367). Plain attribute, never a
            # Parameter or registered buffer, so it stays out of state_dict.
            capacity = int(config.index_topk)
            buf = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                capacity,
                dtype=torch.int32,
                device=current_platform.device_type,
            )
        self.topk_indices_buffer = buf

        sparse_ids = self.dsa_sparse_layers

        def _layer(config, cache_config, quant_config, prefix):
            # extract_layer_index is vLLM's own prefix parser ("model.layers.7" -> 7), the same one
            # Qwen3DSADecoderLayer uses for self.layer_id, so the ids line up with config.json's.
            return Qwen3DSADecoderLayer(
                config,
                cache_config,
                quant_config,
                prefix,
                sparse=extract_layer_index(prefix) in sparse_ids,
                topk_indices_buffer=buf,
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=_layer)
        if sparse:
            if RUNTIME.config is not None and RUNTIME.config.telemetry == "graph_safety":
                layer_names = [
                    layer.self_attn.indexer.layer_name
                    for layer in self.layers
                    if getattr(layer, "is_sparse", False)
                ]
                if layer_names:
                    assert self.topk_indices_buffer is not None
                    RUNTIME.initialize_graph_safety(
                        layer_names,
                        device=self.topk_indices_buffer.device,
                    )
            n_sparse = sum(1 for lyr in self.layers if getattr(lyr, "is_sparse", False))
            print(
                f"[Qwen3DSA-approx] {n_sparse}/{config.num_hidden_layers} layers sparse; "
                f"selector={config.dsa_selector} backend={config.dsa_selector_backend} "
                f"k={config.dsa_top_k} capacity={config.index_topk} telemetry={config.dsa_telemetry} "
                f"(dense: {sorted(set(range(config.num_hidden_layers)) - sparse_ids) or 'none'})"
            )


class Qwen3DSAApproxForCausalLM(nn.Module, SupportsPP):
    """Qwen3-4B DSA on vLLM. Mirrors ``Qwen3ForCausalLM`` with our model."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        cache_block = vllm_config.cache_config.block_size
        assert cache_block == DSA_KERNEL_BLOCK_SIZE, (
            f"Qwen3 DSA needs --block-size {DSA_KERNEL_BLOCK_SIZE} (got {cache_block}): the "
            "indexer's paged-logits kernel and the top-k -> global-slot conversion both assume it."
        )

        self.model = Qwen3DSAModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.dsa_sparse = self.model.dsa_sparse

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
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
        """Strict load: a missing or unexpected key raises.

        Stock vLLM loaders end their branches in ``if name not in params_dict: continue``, which is
        exactly how an entire index branch can vanish in silence and leave a fluent dense model.
        ``AutoWeightsLoader`` raises instead.
        """
        if not self.dsa_sparse:
            # DSA_SPARSE=0: stock Qwen3Attention has no indexer parameters, so a strict load of a
            # DSA checkpoint would fail. Drop them. No norm-convention fixups are needed (unlike
            # MSA's Route A `w - 1` shift) because our main q/k norms are Qwen3's own, untouched.
            weights = ((n, w) for n, w in weights if ".indexer." not in n)

        elif _random_indexer():
            # Randomize in the weight STREAM rather than after load: the model lives in the
            # EngineCore subprocess, so there is no post-construction handle to it from the caller.
            # Preserve each tensor's own std so only the learned structure is destroyed, not the
            # score scale -- a collapse caused by a broken score magnitude would prove nothing.
            def _rand(ws):
                g = torch.Generator().manual_seed(1234)
                n = 0
                for name, w in ws:
                    if ".indexer." in name and (
                        name.endswith("wq.weight")
                        or name.endswith("wk.weight")
                        or name.endswith("weights_proj.weight")
                    ):
                        std = w.float().std().clamp(min=1e-8)
                        w = (torch.randn(w.shape, generator=g, dtype=torch.float32) * std).to(w.dtype)
                        n += 1
                    yield name, w
                # Sparse layers, not all layers: at dense_prefix=4 on a 36-layer model the
                # checkpoint carries 32 indexers, and asserting 36 would fail the control run.
                n_expected = 3 * len(self.model.dsa_sparse_layers)
                assert n == n_expected, (
                    f"DSA_RANDOM_INDEXER randomized {n} tensors, expected {n_expected} "
                    f"({len(self.model.dsa_sparse_layers)} sparse layers x 3)"
                )

            print(
                "[Qwen3DSA] DSA_RANDOM_INDEXER=1 -- indexer projections randomized. This is the "
                "eval ladder's negative control; the model MUST collapse. Never a serving config."
            )
            weights = _rand(weights)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
