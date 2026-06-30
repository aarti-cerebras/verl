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
"""DSA integration for MiniCPM3-4B (Part B): graft the lightning indexer onto MiniCPM3's MLA attention.

This wires `LightningIndexer` (see dsa_indexer.py) into MiniCPM3's `MiniCPMFlashAttention2` via verl's
monkey-patch path. The patched forward mirrors the stock flash forward exactly (so the LM output is
unchanged) and, in `dense_warmup` mode (Phase 1), additionally:
  * captures the MLA compressed query latent `qr = q_a_layernorm(q_a_proj(x))` (the indexer query input),
  * recomputes the base attention's head-averaged distribution `p` (the distillation target, detached),
  * computes indexer scores `I` and accumulates the per-layer KL `KL(p || softmax(I))` on `self._dsa_kl`.

A forward hook on the root model sums the per-layer KL into `model._dsa_indexer_kl` (a scalar tensor),
which the Part-C custom FSDP engine reads into `model_output["indexer_kl"]`.

`sparse` mode (Phase 2: top-k selection + sparse attention) is stubbed (raises) — designed but not built.

See docs/dsa_partB_integration_plan.md / docs/dsa_minicpm3_plan.md.
"""

import sys
from typing import Optional

import torch
import torch.nn.functional as F

from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer

_KL_EPS = 1e-12


def apply_get_usable_length_shim() -> None:
    """Restore `DynamicCache.get_usable_length` (removed ~transformers 4.41) so MiniCPM3's frozen
    trust-remote-code forward runs. Idempotent. See memory `minicpm3-transformers5-incompat`."""
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)


def build_dsa_config(model_config, **overrides) -> DSAConfig:
    """Build a `DSAConfig` from a live MiniCPM3 config. `q_lora_rank`/`rope_head_dim`/`hidden_size` are
    forced from the model; `n_heads`/`head_dim`/`top_k`/`mode`/`fp8` come from `overrides` (training cfg)."""
    kw = dict(
        enabled=True,
        q_lora_rank=model_config.q_lora_rank,
        rope_head_dim=model_config.qk_rope_head_dim,
        hidden_size=model_config.hidden_size,
    )
    kw.update(overrides)
    return DSAConfig(**kw)


def attach_indexers(model, dsa_cfg: DSAConfig) -> None:
    """Attach a `LightningIndexer` (+ the DSAConfig) to every decoder layer's `self_attn` instance,
    matching the attention's device/dtype."""
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.dsa = dsa_cfg
        idx = LightningIndexer(dsa_cfg)  # indexer uses its own softmax_scale = head_dim**-0.5
        ref = next(attn.parameters())
        attn.indexer = idx.to(device=ref.device, dtype=ref.dtype)


def install_kl_accumulation(model) -> None:
    """Reset per-layer KL each forward (pre-hook) and sum it into `model._dsa_indexer_kl` (post-hook)."""
    layers = model.model.layers

    def _pre_hook(module, args, kwargs):
        for layer in layers:
            layer.self_attn._dsa_kl = None
        return None

    def _post_hook(module, args, output):
        kls = [layer.self_attn._dsa_kl for layer in layers if getattr(layer.self_attn, "_dsa_kl", None) is not None]
        # sum of per-layer (mean-over-positions) KL; Part C/D applies final normalization
        model._dsa_indexer_kl = torch.stack(kls).sum() if kls else None
        return output

    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model.register_forward_hook(_post_hook)


def _causal_doc_bias_block(position_ids, q0: int, q1: int, T: int, device) -> torch.Tensor:
    """Additive mask `[bsz, q1-q0, T]` (0 where key j is attendable by query i in [q0,q1), else -inf):
    causal (j<=i) AND same-document (doc boundary = `position_ids == 0`)."""
    doc_id = (position_ids == 0).cumsum(dim=-1)  # [bsz, T] — increments at each doc start
    q_pos = torch.arange(q0, q1, device=device)  # [B]
    k_pos = torch.arange(T, device=device)  # [T]
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [B, T]
    same_doc = doc_id[:, q0:q1].unsqueeze(2) == doc_id.unsqueeze(1)  # [bsz, B, T]
    allow = causal.unsqueeze(0) & same_doc  # [bsz, B, T]
    return torch.zeros(allow.shape, device=device, dtype=torch.float32).masked_fill(~allow, float("-inf"))


def _build_causal_doc_bias(position_ids: Optional[torch.Tensor], T: int, device, dtype) -> torch.Tensor:
    """Full `[bsz, T, T]` causal+per-document additive mask (thin wrapper over the block builder)."""
    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0)
    return _causal_doc_bias_block(position_ids, 0, T, T, device).to(dtype)


def _dense_warmup_kl(attn, hidden_states, qr, query_states, key_states, cos, sin, position_ids):
    """Per-layer indexer KL for Phase 1, tiled over query blocks to bound memory at 32K.

    Args:
        query_states, key_states: ``[bsz, num_heads, T, q_head_dim]`` (post-RoPE, pre-flash-transpose).
        cos, sin: rotary tables ``[seq_len, rope_head_dim]`` from MiniCPM's `rotary_emb`.
    Returns scalar `KL(p || softmax(I))` averaged over valid query positions. The target `p` (head-averaged
    softmax attention, detached) is accumulated head-by-head so peak is ~`[bsz, block, T]`, not
    `[bsz, H, T, T]`. `block = dsa.kl_block_size`.
    """
    bsz, H, T, _ = query_states.shape
    device, compute_dtype = query_states.device, query_states.dtype
    scale = attn.softmax_scale
    block = getattr(attn.dsa, "kl_block_size", 0) or T

    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0)
    cos_g, sin_g = cos[position_ids], sin[position_ids]  # [bsz, T, rope_head_dim]
    # indexer projections once; score per query-block (key set is the full sequence)
    q_idx, k_idx, weights = attn.indexer.project(hidden_states, qr, cos_g, sin_g)

    total_kl = query_states.new_zeros((), dtype=torch.float32)
    total_cnt = 0
    for q0 in range(0, T, block):
        q1 = min(q0 + block, T)
        bias = _causal_doc_bias_block(position_ids, q0, q1, T, device)  # [bsz, B, T]

        # --- target p_blk (detached); accumulate over heads to bound memory ---
        with torch.no_grad():
            qb = query_states[:, :, q0:q1, :]  # [bsz, H, B, Dh]
            p_blk = query_states.new_zeros(bsz, q1 - q0, T, dtype=torch.float32)
            for h in range(H):
                s = torch.matmul(qb[:, h], key_states[:, h].transpose(1, 2)) * scale  # [bsz, B, T]
                p_blk += torch.softmax(s.float() + bias, dim=-1)
            p_blk /= H

        # --- indexer I_blk (grad into indexer only) ---
        I_blk = attn.indexer.scores(q_idx[:, q0:q1], k_idx, weights[:, q0:q1], attn_bias=bias)  # [bsz, B, T]
        log_q = torch.log_softmax(I_blk.float(), dim=-1)

        allow = bias == 0.0
        term = p_blk * (torch.log(p_blk.clamp_min(_KL_EPS)) - log_q)
        kl_blk = torch.where(allow, term, torch.zeros_like(term)).sum(dim=-1)  # [bsz, B]
        total_kl = total_kl + kl_blk.sum()
        total_cnt += kl_blk.numel()

    return (total_kl / max(total_cnt, 1)).to(compute_dtype)


def minicpm3_dsa_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Patched `MiniCPMFlashAttention2.forward`: stock flash attention (unchanged LM output) + DSA.

    Mirrors the stock projection/RoPE/flash path verbatim, captures `qr`/`query_states`/`key_states`, and
    in `dense_warmup` accumulates the per-layer indexer KL on `self._dsa_kl`.
    """
    mod = sys.modules[type(self).__module__]
    apply_rotary_pos_emb = mod.apply_rotary_pos_emb

    bsz, q_len, _ = hidden_states.size()

    # --- projections (mirror stock); capture qr for the indexer ---
    qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(qr).view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )
    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    kv_seq_len = value_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads, q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    # --- DSA branch (before flash transpose/value-pad; uses query_states/key_states) ---
    dsa = getattr(self, "dsa", None)
    if dsa is not None and dsa.enabled:
        if dsa.mode == "dense_warmup":
            self._dsa_kl = _dense_warmup_kl(self, hidden_states, qr, query_states, key_states, cos, sin, position_ids)
        elif dsa.mode == "sparse":
            raise NotImplementedError("DSA 'sparse' mode (Phase 2 top-k attention) is not implemented yet.")
        else:
            raise ValueError(f"unknown DSA mode: {dsa.mode}")

    # --- LM attention output: stock flash path (unchanged) ---
    if self.q_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    dropout_rate = self.attention_dropout if self.training else 0.0

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, softmax_scale=self.softmax_scale
    )
    if self.q_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
