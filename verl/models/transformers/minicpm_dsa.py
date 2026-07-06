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


def dsa_overrides_from_config(model_config) -> dict:
    """Collect DSA overrides from the (HF) model config, supporting two injection styles:

    * a `dsa_overrides` dict attribute (used by tests / programmatic setup), or
    * flat scalar attributes `dsa_<field>` (e.g. `dsa_n_heads`) — required for verl's `override_config`,
      whose `update_model_config` recurses into nested dict values (so a nested `dsa_overrides` dict can't
      be injected, but scalars `setattr` fine).
    """
    ov = getattr(model_config, "dsa_overrides", None)
    if isinstance(ov, dict):
        return dict(ov)
    out = {}
    for field in (
        "n_heads",
        "head_dim",
        "rope_head_dim",
        "top_k",
        "mode",
        "kl_block_size",
        "kl_reduction",
        "fp8",
        "diag_interval",
        "log_per_layer",
    ):
        val = getattr(model_config, f"dsa_{field}", None)
        if val is not None:
            out[field] = val
    return out


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


def freeze_base_train_indexer(model) -> list:
    """Phase-1 freeze: set ``requires_grad=False`` on all params except the DSA indexer (``*.indexer.*``).

    Returns the list of trainable (indexer) params — hand these to the optimizer (LR ~1e-3). With the base
    frozen, autograd retains no base activation graph and AdamW no-ops the frozen params, so only the indexer
    trains. Call this BEFORE FSDP wrapping; the FSDP wrap must use ``use_orig_params=True`` (FSDP1) or FSDP2
    for mixed ``requires_grad`` in a flat parameter to be legal.
    """
    trainable = []
    for name, p in model.named_parameters():
        is_indexer = ".indexer." in name
        p.requires_grad_(is_indexer)
        if is_indexer:
            trainable.append(p)
    return trainable


def install_kl_accumulation(model) -> None:
    """Wire the per-layer indexer KL + monitoring into model attributes each forward.

    Pre-hook: reset per-layer state, bump a forward counter, and set the shared `dsa._do_diag` flag every
    `diag_interval` forwards (diagnostics add top-k compute, so they're gated). Post-hook: sum the per-layer
    KL into `model._dsa_indexer_kl` (the loss) and build `model._dsa_metrics` — plain floats for wandb (the
    SFT trainer logs the metrics dict raw; it does NOT unwrap `Metric` objects). KL layer min/max/mean are
    cheap and always logged; recall/overlap/score-health only on diag forwards.
    """
    layers = model.model.layers

    def _pre_hook(module, args, kwargs):
        for layer in layers:
            layer.self_attn._dsa_kl = None
            layer.self_attn._dsa_diag = None
        cnt = getattr(model, "_dsa_fwd_count", 0) + 1
        model._dsa_fwd_count = cnt
        dsa = layers[0].self_attn.dsa  # shared across layers
        interval = max(1, getattr(dsa, "diag_interval", 1))
        dsa._do_diag = (cnt - 1) % interval == 0  # diag on the 1st forward, then every `interval`
        return None

    def _post_hook(module, args, output):
        attns = [layer.self_attn for layer in layers]
        # keep the true layer index alongside each value so per-layer keys are labelled correctly
        kl_items = [(i, a._dsa_kl) for i, a in enumerate(attns) if getattr(a, "_dsa_kl", None) is not None]
        if not kl_items:
            model._dsa_indexer_kl = None
            model._dsa_metrics = {}
            return output
        kl_stack = torch.stack([kl for _, kl in kl_items])
        # loss reduction over layers (configurable, keeps grad): "mean" (default) | "sum" (reference). "mean"
        # gives an interpretable per-layer loss scale; same optimum, gradient just rescaled by 1/n_layers.
        reduction = getattr(layers[0].self_attn.dsa, "kl_reduction", "mean")
        model._dsa_indexer_kl = kl_stack.mean() if reduction == "mean" else kl_stack.sum()
        metrics = {
            "indexer/kl_layer_mean": kl_stack.mean().item(),
            "indexer/kl_layer_min": kl_stack.min().item(),
            "indexer/kl_layer_max": kl_stack.max().item(),
        }
        diag_items = [(i, a._dsa_diag) for i, a in enumerate(attns) if getattr(a, "_dsa_diag", None) is not None]
        if diag_items:
            diags = [d for _, d in diag_items]
            rename = {"recall": "topk_recall", "overlap": "topk_overlap"}
            for key in ("recall", "overlap", "score_mean", "score_std", "nan_frac", "entropy", "entropy_frac"):
                metrics[f"indexer/{rename.get(key, key)}"] = torch.stack([d[key] for d in diags]).mean().item()
        # optional per-layer breakdown (debug): emit SEPARATE scalar keys so the logger doesn't collapse them
        # to a mean. kl_by_layer every step; entropy_frac_by_layer only on diag forwards.
        if getattr(layers[0].self_attn.dsa, "log_per_layer", False):
            for i, kl in kl_items:
                metrics[f"indexer/kl_by_layer/L{i:02d}"] = kl.item()
            for i, d in diag_items:
                metrics[f"indexer/entropy_frac_by_layer/L{i:02d}"] = d["entropy_frac"].item()
        model._dsa_metrics = metrics
        return output

    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model.register_forward_hook(_post_hook)


def _causal_doc_bias_block(position_ids, q0: int, q1: int, T: int, device, key_mask=None) -> torch.Tensor:
    """Additive mask `[bsz, q1-q0, T]` (0 where key j is attendable by query i in [q0,q1), else -inf):
    causal (j<=i) AND same-document (doc boundary = `position_ids == 0`) AND, if `key_mask` is given,
    the key is a real (non-pad) token. This mirrors exactly what the base attention attends over, so the
    recomputed target `p` matches the base attention distribution over the same positions."""
    doc_id = (position_ids == 0).cumsum(dim=-1)  # [bsz, T] — increments at each doc start
    q_pos = torch.arange(q0, q1, device=device)  # [B]
    k_pos = torch.arange(T, device=device)  # [T]
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [B, T]
    same_doc = doc_id[:, q0:q1].unsqueeze(2) == doc_id.unsqueeze(1)  # [bsz, B, T]
    allow = causal.unsqueeze(0) & same_doc  # [bsz, B, T]
    if key_mask is not None:
        allow = allow & key_mask[:, None, :]  # exclude pad keys ([bsz, 1, T])
    return torch.zeros(allow.shape, device=device, dtype=torch.float32).masked_fill(~allow, float("-inf"))


def _build_causal_doc_bias(position_ids: Optional[torch.Tensor], T: int, device, dtype) -> torch.Tensor:
    """Full `[bsz, T, T]` causal+per-document additive mask (thin wrapper over the block builder)."""
    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0)
    return _causal_doc_bias_block(position_ids, 0, T, T, device).to(dtype)


def _dense_warmup_kl(attn, hidden_states, qr, query_states, key_states, cos, sin, position_ids, attention_mask=None):
    """Per-layer indexer KL for Phase 1, tiled over query blocks to bound memory at 32K.

    Args:
        query_states, key_states: ``[bsz, num_heads, T, q_head_dim]`` (post-RoPE, pre-flash-transpose).
        cos, sin: rotary tables ``[seq_len, rope_head_dim]`` from MiniCPM's `rotary_emb`.
        attention_mask: optional ``[bsz, T]`` real-token mask (1=real, 0=pad). When present, the valid set
            mirrors the base attention exactly — pad keys are excluded and pad queries are dropped from the
            loss/normalization — so padding never leaks into the target/loss.
    Returns scalar `KL(p || softmax(I))` averaged over valid query positions. The target `p` (head-averaged
    softmax attention, detached) is accumulated head-by-head so peak is ~`[bsz, block, T]`, not
    `[bsz, H, T, T]`. `block = dsa.kl_block_size`.
    """
    bsz, H, T, _ = query_states.shape
    device, compute_dtype = query_states.device, query_states.dtype
    scale = attn.softmax_scale
    block = getattr(attn.dsa, "kl_block_size", 0) or T
    do_diag = bool(getattr(attn.dsa, "_do_diag", False))  # set by the KL pre-hook every diag_interval
    diag_k = min(getattr(attn.dsa, "top_k", T), T)

    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, T)
    # real-token mask (exclude right-padding); only 2D [bsz, T] padding masks are used
    key_mask = attention_mask.bool() if (attention_mask is not None and attention_mask.dim() == 2) else None

    cos_g, sin_g = cos[position_ids], sin[position_ids]  # [bsz, T, rope_head_dim]
    # indexer projections once; score per query-block (key set is the full sequence)
    q_idx, k_idx, weights = attn.indexer.project(hidden_states, qr, cos_g, sin_g)

    total_kl = query_states.new_zeros((), dtype=torch.float32)
    total_cnt = query_states.new_zeros((), dtype=torch.float32)
    # monitoring diagnostics (only when do_diag): recall/overlap of top-k, indexer score health
    d_recall = query_states.new_zeros((), dtype=torch.float32)
    d_overlap = query_states.new_zeros((), dtype=torch.float32)
    d_rc = query_states.new_zeros((), dtype=torch.float32)
    d_ssum = query_states.new_zeros((), dtype=torch.float32)
    d_ssq = query_states.new_zeros((), dtype=torch.float32)
    d_scnt = query_states.new_zeros((), dtype=torch.float32)
    d_nan = query_states.new_zeros((), dtype=torch.float32)
    d_ent = query_states.new_zeros((), dtype=torch.float32)  # softmax(I) entropy (nats), summed over valid queries
    d_entfrac = query_states.new_zeros((), dtype=torch.float32)  # entropy / log(#valid keys) in [0,1]

    for q0 in range(0, T, block):
        q1 = min(q0 + block, T)
        bias = _causal_doc_bias_block(position_ids, q0, q1, T, device, key_mask=key_mask)  # [bsz, B, T]
        allow = bias == 0.0
        qv = key_mask[:, q0:q1].to(torch.float32) if key_mask is not None else None  # [bsz, B] real-query mask

        # --- target p_blk (detached); accumulate over heads to bound memory ---
        with torch.no_grad():
            qb = query_states[:, :, q0:q1, :]  # [bsz, H, B, Dh]
            p_blk = query_states.new_zeros(bsz, q1 - q0, T, dtype=torch.float32)
            for h in range(H):
                s = torch.matmul(qb[:, h], key_states[:, h].transpose(1, 2)) * scale  # [bsz, B, T]
                p_blk += torch.softmax(s.float() + bias, dim=-1)
            p_blk /= H  # NOTE: rows with no valid keys (pad queries) are NaN here — never used (see below)

        if getattr(attn.dsa, "_capture_p", False):
            # test/debug only: stash the recomputed target distribution for validation vs eager attention.
            # Assumes a single block (kl_block_size >= T) so this is the full [bsz, T, T].
            attn._dsa_p = p_blk.detach()

        # --- indexer I_blk (grad into indexer only) ---
        I_blk = attn.indexer.scores(q_idx[:, q0:q1], k_idx, weights[:, q0:q1], attn_bias=bias)  # [bsz, B, T]
        log_q = torch.log_softmax(I_blk.float(), dim=-1)

        term = p_blk * (torch.log(p_blk.clamp_min(_KL_EPS)) - log_q)
        # `where` yields 0 for every masked key, so an all-masked (pad-query) row sums to a finite 0 — the
        # NaN in p_blk at such rows is confined to masked entries and discarded, so kl_blk is always finite.
        kl_blk = torch.where(allow, term, torch.zeros_like(term)).sum(dim=-1)  # [bsz, B]
        if qv is not None:
            total_kl = total_kl + (kl_blk * qv).sum()  # count only real (non-pad) query rows
            total_cnt = total_cnt + qv.sum()
        else:
            total_kl = total_kl + kl_blk.sum()
            total_cnt = total_cnt + kl_blk.numel()

        if do_diag:
            with torch.no_grad():
                idf = I_blk.detach().float()
                k = min(diag_k, T)
                topk_i = idf.topk(k, dim=-1).indices  # indexer-selected keys [bsz, B, k]
                recall = p_blk.gather(-1, topk_i).sum(-1)  # [bsz, B] (NaN at pad-query rows)
                topk_p = p_blk.topk(k, dim=-1).indices
                overlap = (topk_i.unsqueeze(-1) == topk_p.unsqueeze(-2)).any(-1).float().sum(-1) / k
                if qv is not None:
                    qb_bool = qv.bool()
                    recall = torch.where(qb_bool, recall, torch.zeros_like(recall))  # drop NaN pad rows
                    overlap = torch.where(qb_bool, overlap, torch.zeros_like(overlap))
                    d_rc = d_rc + qv.sum()
                else:
                    d_rc = d_rc + recall.numel()
                d_recall = d_recall + recall.sum()
                d_overlap = d_overlap + overlap.sum()
                fin = idf[allow]  # finite valid scores only (excludes pad keys / masked / pad rows)
                d_ssum = d_ssum + fin.sum()
                d_ssq = d_ssq + (fin * fin).sum()
                d_scnt = d_scnt + fin.numel()
                d_nan = d_nan + torch.isnan(fin).float().sum()
                # softmax(I) entropy: how peaked the student distribution is. entropy_frac ~1.0 => near-uniform
                # (healthy KL-distillation start); a low/falling value flags a saturated or over-committed init.
                q_dist = torch.softmax(idf, dim=-1)  # masked keys -> 0 (idf carries -inf on disallowed)
                ent = torch.special.entr(q_dist).sum(-1)  # [bsz, B] per-query entropy (nats); entr(0)=0
                valid = allow.sum(-1)  # [bsz, B] attendable keys per query (causal + same-doc)
                ent_frac = torch.where(
                    valid > 1, ent / valid.float().clamp_min(2).log(), torch.ones_like(ent)
                )  # single-valid-key rows are trivially uniform -> 1.0
                if qv is not None:
                    ent = torch.where(qb_bool, ent, torch.zeros_like(ent))
                    ent_frac = torch.where(qb_bool, ent_frac, torch.zeros_like(ent_frac))
                d_ent = d_ent + ent.sum()
                d_entfrac = d_entfrac + ent_frac.sum()

    if do_diag:
        mean = d_ssum / d_scnt.clamp_min(1)
        var = (d_ssq / d_scnt.clamp_min(1) - mean * mean).clamp_min(0)
        attn._dsa_diag = {
            "recall": (d_recall / d_rc.clamp_min(1)).detach(),
            "overlap": (d_overlap / d_rc.clamp_min(1)).detach(),
            "score_mean": mean.detach(),
            "score_std": var.sqrt().detach(),
            "nan_frac": (d_nan / d_scnt.clamp_min(1)).detach(),
            "entropy": (d_ent / d_rc.clamp_min(1)).detach(),
            "entropy_frac": (d_entfrac / d_rc.clamp_min(1)).detach(),
        }
    else:
        attn._dsa_diag = None

    return (total_kl / total_cnt.clamp_min(1)).to(compute_dtype)


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
            self._dsa_kl = _dense_warmup_kl(
                self, hidden_states, qr, query_states, key_states, cos, sin, position_ids, attention_mask
            )
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
