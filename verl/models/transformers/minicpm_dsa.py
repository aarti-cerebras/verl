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
import torch.utils.checkpoint

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
        "diag_overlap_sample",
        "kl_checkpoint",
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
        # eval/validation (module in eval mode): always compute diagnostics so every held-out batch gets
        # recall/overlap/entropy. training: gate on `interval` (diag adds top-k compute). diag on 1st forward.
        dsa._do_diag = (not model.training) or ((cnt - 1) % interval == 0)
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

            def _layer_mean(key):
                return torch.stack([d[key] for d in diags]).mean().item()

            # `indexer/` section: indexer training-health scalars (top-k fidelity + score stats)
            metrics["indexer/topk_recall"] = _layer_mean("recall")
            metrics["indexer/topk_overlap"] = _layer_mean("overlap")
            for key in ("score_mean", "score_std", "nan_frac"):
                metrics[f"indexer/{key}"] = _layer_mean(key)
            # Split the two entropies into SEPARATE wandb sections (the wandb section is the key's first path
            # segment): the student (indexer softmax(I)) entropy joins the `indexer/` section alongside the
            # kl/topk/score health above, and the teacher (base attention `p`) entropy goes to its own `attn/`
            # section. So indexer panes and attn panes never share a group.
            metrics["indexer/entropy"] = _layer_mean("entropy")
            metrics["indexer/entropy_frac"] = _layer_mean("entropy_frac")
            metrics["attn/entropy"] = _layer_mean("attn_entropy")
            metrics["attn/entropy_frac"] = _layer_mean("attn_entropy_frac")
        # optional per-layer breakdown (debug): emit SEPARATE scalar keys so the logger doesn't collapse them
        # to a mean, each in its own wandb section. kl_by_layer every step; entropy by-layer only on diag forwards.
        if getattr(layers[0].self_attn.dsa, "log_per_layer", False):
            for i, kl in kl_items:
                metrics[f"kl_by_layer/L{i:02d}"] = kl.item()
            for i, d in diag_items:
                metrics[f"indexer/entropy_frac_by_layer/L{i:02d}"] = d["entropy_frac"].item()
                metrics[f"attn/entropy_frac_by_layer/L{i:02d}"] = d["attn_entropy_frac"].item()
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
    diag_ov_sample = int(getattr(attn.dsa, "diag_overlap_sample", 0) or 0)  # >0 -> subsample rows for topk_overlap
    tile_ckpt = bool(getattr(attn.dsa, "kl_checkpoint", False)) and attn.training  # per-tile checkpoint the score graph

    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, T)
    # real-token mask (exclude right-padding); only 2D [bsz, T] padding masks are used
    key_mask = attention_mask.bool() if (attention_mask is not None and attention_mask.dim() == 2) else None

    cos_g, sin_g = cos[position_ids], sin[position_ids]  # [bsz, T, rope_head_dim]
    # indexer projections once; score per query-block (key set is the full sequence). Route through
    # __call__ (return_projection=True), NOT attn.indexer.project(...): a direct method call bypasses
    # nn.Module.__call__, so FSDP2's forward hooks on a separately-wrapped indexer unit never fire (no
    # all-gather, no pre-backward gate -> no grad reduce-scatter to the master). See dsa_fsdp_sharding_notes B2.
    q_idx, k_idx, weights = attn.indexer(hidden_states, qr, cos_g, sin_g, return_projection=True)

    total_kl = query_states.new_zeros((), dtype=torch.float32)
    total_cnt = query_states.new_zeros((), dtype=torch.float32)
    # monitoring diagnostics (only when do_diag): recall/overlap of top-k, indexer score health
    d_recall = query_states.new_zeros((), dtype=torch.float32)
    d_overlap = query_states.new_zeros((), dtype=torch.float32)
    d_rc = query_states.new_zeros((), dtype=torch.float32)
    d_ovcnt = query_states.new_zeros((), dtype=torch.float32)  # overlap's own row count (may be subsampled)
    d_ssum = query_states.new_zeros((), dtype=torch.float32)
    d_ssq = query_states.new_zeros((), dtype=torch.float32)
    d_scnt = query_states.new_zeros((), dtype=torch.float32)
    d_nan = query_states.new_zeros((), dtype=torch.float32)
    d_ent = query_states.new_zeros((), dtype=torch.float32)  # softmax(I) entropy (nats), summed over valid queries
    d_entfrac = query_states.new_zeros((), dtype=torch.float32)  # entropy / log(#valid keys) in [0,1]
    d_attn_ent = query_states.new_zeros((), dtype=torch.float32)  # base-model attention entropy (nats), summed
    d_attn_entfrac = query_states.new_zeros((), dtype=torch.float32)  # attn entropy / log(#valid keys) in [0,1]

    def _tile_kl(q_idx_t, k_idx_t, w_t, bias_t, allow_t, p_t):
        """Per-tile grad-carrying KL. Recomputes the indexer scores (the [bsz, B, n_heads, T] ``dots``) so that,
        under checkpoint, only ONE tile's score graph is live in backward. Returns per-query-row KL [bsz, B].
        ``where`` yields 0 for masked keys, so an all-masked (pad-query) row sums to a finite 0 (its p_t NaN is
        confined to masked entries and discarded)."""
        I_blk = attn.indexer.scores(q_idx_t, k_idx_t, w_t, attn_bias=bias_t)  # [bsz, B, T]
        log_q = torch.log_softmax(I_blk.float(), dim=-1)
        term = p_t * (torch.log(p_t.clamp_min(_KL_EPS)) - log_q)
        return torch.where(allow_t, term, torch.zeros_like(term)).sum(dim=-1)  # [bsz, B]

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

        # --- indexer KL for this tile (grad into indexer only). When kl_checkpoint is on, checkpoint the score
        # graph so its [B, n_heads, T] `dots` is recomputed in backward -> only one tile is ever live (nested
        # under the per-layer checkpoint at the attn-forward call site). ---
        if tile_ckpt:
            kl_blk = torch.utils.checkpoint.checkpoint(
                _tile_kl, q_idx[:, q0:q1], k_idx, weights[:, q0:q1], bias, allow, p_blk, use_reentrant=False
            )  # [bsz, B]
        else:
            kl_blk = _tile_kl(q_idx[:, q0:q1], k_idx, weights[:, q0:q1], bias, allow, p_blk)  # [bsz, B]
        if qv is not None:
            total_kl = total_kl + (kl_blk * qv).sum()  # count only real (non-pad) query rows
            total_cnt = total_cnt + qv.sum()
        else:
            total_kl = total_kl + kl_blk.sum()
            total_cnt = total_cnt + kl_blk.numel()

        if do_diag:
            with torch.no_grad():
                # recompute scores here (I_blk now lives inside _tile_kl); no_grad -> transient, no graph held
                idf = attn.indexer.scores(q_idx[:, q0:q1], k_idx, weights[:, q0:q1], attn_bias=bias).float()
                k = min(diag_k, T)
                topk_i = idf.topk(k, dim=-1).indices  # indexer-selected keys [bsz, B, k]
                recall = p_blk.gather(-1, topk_i).sum(-1)  # [bsz, B] (NaN at pad-query rows)
                if qv is not None:
                    qb_bool = qv.bool()
                    recall = torch.where(qb_bool, recall, torch.zeros_like(recall))  # drop NaN pad rows
                    d_rc = d_rc + qv.sum()
                else:
                    d_rc = d_rc + recall.numel()
                d_recall = d_recall + recall.sum()
                # topk_overlap is O(B*k^2) (materializes [bsz, B, k, k]); at long context cap the query rows
                # sampled for it (diag_overlap_sample>0) so peak stays bounded. Its own counter d_ovcnt keeps
                # the average correct over the (possibly subsampled) rows; recall/entropy/score use all rows.
                B = q1 - q0
                sel = (
                    torch.linspace(0, B - 1, diag_ov_sample, device=device).round().long()
                    if (diag_ov_sample and B > diag_ov_sample)
                    else slice(None)
                )
                p_sel = p_blk[:, sel]
                topk_p = p_sel.topk(k, dim=-1).indices  # [bsz, n, k] target-selected keys
                # overlap = |indexer-topk ∩ target-topk| / k via a boolean membership mask over keys (O(n*T))
                # instead of the O(n*k^2) pairwise compare, so the [.., k, k] tensor never forms (fits at 32K).
                mask = torch.zeros_like(p_sel, dtype=torch.bool).scatter_(-1, topk_p, True)  # target keys
                overlap = mask.gather(-1, topk_i[:, sel]).float().sum(-1) / k  # [bsz, n] frac of indexer topk in target
                ov_qv = qv[:, sel] if qv is not None else None
                if ov_qv is not None:
                    overlap = torch.where(ov_qv.bool(), overlap, torch.zeros_like(overlap))
                    d_ovcnt = d_ovcnt + ov_qv.sum()
                else:
                    d_ovcnt = d_ovcnt + overlap.numel()
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
                # base-model attention entropy (the teacher `p`): how peaked the true attention is per query.
                # Rising KL-target entropy => flatter attention the indexer must match; a very low value means
                # attention is near-argmax (easy to top-k). Masked keys contribute 0 (entr(0)=0); pad-query rows
                # are NaN in p_blk but dropped below by the same qb_bool mask, so the sum stays finite.
                attn_ent = torch.special.entr(p_blk).sum(-1)  # [bsz, B] per-query attention entropy (nats)
                attn_ent_frac = torch.where(
                    valid > 1, attn_ent / valid.float().clamp_min(2).log(), torch.ones_like(attn_ent)
                )
                if qv is not None:
                    attn_ent = torch.where(qb_bool, attn_ent, torch.zeros_like(attn_ent))
                    attn_ent_frac = torch.where(qb_bool, attn_ent_frac, torch.zeros_like(attn_ent_frac))
                d_attn_ent = d_attn_ent + attn_ent.sum()
                d_attn_entfrac = d_attn_entfrac + attn_ent_frac.sum()

    if do_diag:
        mean = d_ssum / d_scnt.clamp_min(1)
        var = (d_ssq / d_scnt.clamp_min(1) - mean * mean).clamp_min(0)
        attn._dsa_diag = {
            "recall": (d_recall / d_rc.clamp_min(1)).detach(),
            "overlap": (d_overlap / d_ovcnt.clamp_min(1)).detach(),
            "score_mean": mean.detach(),
            "score_std": var.sqrt().detach(),
            "nan_frac": (d_nan / d_scnt.clamp_min(1)).detach(),
            "entropy": (d_ent / d_rc.clamp_min(1)).detach(),
            "entropy_frac": (d_entfrac / d_rc.clamp_min(1)).detach(),
            "attn_entropy": (d_attn_ent / d_rc.clamp_min(1)).detach(),
            "attn_entropy_frac": (d_attn_entfrac / d_rc.clamp_min(1)).detach(),
        }
    else:
        attn._dsa_diag = None

    return (total_kl / total_cnt.clamp_min(1)).to(compute_dtype)


def _sparse_attn(
    attn, hidden_states, qr, query_states, key_states, value_states, cos, sin, position_ids, attention_mask=None
):
    """DSA Phase-2 sparse LM attention (T1). Selects top-k keys per query via the lightning indexer and attends
    ONLY over the selected set (gathered-KV, tiled over query blocks to bound the ``[b, H, B, k, d]`` peak).

    Gradient wiring (the crux, per arxiv 2512.02556 §2.1.1):
      * The LM path (``attn_output``) uses ``query/key/value_states`` (base activations) gathered at the
        selected indices, so the **LM loss trains the base**. ``idx`` is ``.detach()``-ed — top-k is
        non-differentiable — so **no LM-loss gradient reaches the indexer**.
      * The indexer is called on **detached** base activations, so its scores carry gradient only into the
        indexer's own params (consumed by the selected-set KL, T2), **never into the base**.

    Returns ``(attn_output [bsz, T, num_heads*v_head_dim] (post o_proj), idx [bsz, T, k], indexer_scores
    [bsz, T, T])``. ``idx``/``indexer_scores`` are reused by ``_sparse_indexer_kl`` (T2).

    Parity: with ``top_k >= T`` the selected set is the full causal+document set, so the output equals the
    dense flash path (softmax is permutation-invariant and V is gathered in the same order as K; masked keys
    carry ``-inf`` bias -> zero weight). This is the M0 parity test.
    """
    bsz, H, T, dqk = query_states.shape
    dv = value_states.shape[-1]
    device = query_states.device
    scale = attn.softmax_scale
    top_k = min(getattr(attn.dsa, "top_k", T) or T, T)
    block = getattr(attn.dsa, "kl_block_size", 0) or T

    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, T)
    key_mask = attention_mask.bool() if (attention_mask is not None and attention_mask.dim() == 2) else None
    cos_g, sin_g = cos[position_ids], sin[position_ids]

    # Indexer scores from DETACHED base activations -> gradient only into indexer params (not the base).
    # Route through __call__ (return_projection=True) so FSDP2 hooks on the indexer unit fire (see fsdp notes B2).
    q_idx, k_idx, w = attn.indexer(hidden_states.detach(), qr.detach(), cos_g, sin_g, return_projection=True)
    indexer_scores = attn.indexer.scores(q_idx, k_idx, w)  # [bsz, T, T] raw (causal/doc bias applied per block)

    o_tiles, idx_tiles = [], []
    for q0 in range(0, T, block):
        q1 = min(q0 + block, T)
        B = q1 - q0
        bias_b = _causal_doc_bias_block(position_ids, q0, q1, T, device, key_mask=key_mask)  # [bsz, B, T]
        idx_b = (indexer_scores[:, q0:q1] + bias_b).topk(top_k, dim=-1).indices.detach()  # [bsz, B, k] (stop-grad)
        idx_tiles.append(idx_b)
        bias_sel = torch.gather(bias_b, 2, idx_b)  # [bsz, B, k] (-inf where a selected key is masked/cross-doc)
        # gather the k selected K/V per (batch, query). index_select avoids the O(B*T) broadcast; stack keeps
        # autograd clean (grad flows back to key/value_states at the selected positions).
        Kg = torch.stack([key_states[b].index_select(1, idx_b[b].reshape(-1)).reshape(H, B, top_k, dqk)
                          for b in range(bsz)])  # [bsz, H, B, k, dqk]
        Vg = torch.stack([value_states[b].index_select(1, idx_b[b].reshape(-1)).reshape(H, B, top_k, dv)
                          for b in range(bsz)])  # [bsz, H, B, k, dv]
        s = torch.einsum("bhBd,bhBkd->bhBk", query_states[:, :, q0:q1], Kg) * scale + bias_sel[:, None]
        a = torch.softmax(s.float(), dim=-1).to(Vg.dtype)
        o_tiles.append(torch.einsum("bhBk,bhBkd->bhBd", a, Vg))  # [bsz, H, B, dv]

    o = torch.cat(o_tiles, dim=2)  # [bsz, H, T, dv]
    idx_full = torch.cat(idx_tiles, dim=1)  # [bsz, T, k]
    attn_output = attn.o_proj(o.transpose(1, 2).reshape(bsz, T, H * dv).contiguous())
    return attn_output, idx_full, indexer_scores


def _sparse_indexer_kl(attn, query_states, key_states, indexer_scores, idx, position_ids, attention_mask=None):
    """DSA Phase-2 selected-set indexer KL (T2), paper eq. 4: ``sum_t KL(p_{t,S_t} || softmax(I_{t,S_t}))``.

    ``indexer_scores`` come from ``_sparse_attn`` (computed on DETACHED base activations, so gradient flows
    only into the indexer's params — never the base). ``idx`` is the stop-grad top-k selection ``S_t``. The
    target ``p`` = head-averaged softmax of the **main** attention over the causal+document set, restricted to
    ``S_t`` and renormalized, and is **detached** (a fixed target). So this loss trains ONLY the indexer.

    Tiled over query blocks (``block = dsa.kl_block_size``); the target ``p`` accumulates head-by-head to keep
    peak at ~``[bsz, block, T]``. Averaged over valid (non-pad) query rows. Masked/cross-doc selected keys
    carry ``-inf`` bias and are zeroed out of the KL (avoids ``0*-inf`` NaNs). Feeds ``attn._dsa_kl``.
    """
    bsz, H, T, _ = query_states.shape
    device, compute_dtype = query_states.device, query_states.dtype
    scale = attn.softmax_scale
    block = getattr(attn.dsa, "kl_block_size", 0) or T
    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, T)
    key_mask = attention_mask.bool() if (attention_mask is not None and attention_mask.dim() == 2) else None

    total_kl = query_states.new_zeros((), dtype=torch.float32)
    total_cnt = query_states.new_zeros((), dtype=torch.float32)
    for q0 in range(0, T, block):
        q1 = min(q0 + block, T)
        B = q1 - q0
        bias_b = _causal_doc_bias_block(position_ids, q0, q1, T, device, key_mask=key_mask)  # [bsz, B, T]
        idx_b = idx[:, q0:q1]  # [bsz, B, k]
        bias_sel = torch.gather(bias_b, 2, idx_b)  # [bsz, B, k]
        allow_sel = bias_sel == 0.0
        qv = key_mask[:, q0:q1].to(torch.float32) if key_mask is not None else None  # [bsz, B] real-query mask

        # target p_blk (detached): head-averaged softmax main attention over causal+doc, accumulate over heads
        with torch.no_grad():
            qb = query_states[:, :, q0:q1, :]
            p_blk = query_states.new_zeros(bsz, B, T, dtype=torch.float32)
            for h in range(H):
                sdot = torch.matmul(qb[:, h], key_states[:, h].transpose(1, 2)) * scale  # [bsz, B, T]
                p_blk += torch.softmax(sdot.float() + bias_b, dim=-1)
            p_blk /= H  # rows with no valid keys (pad queries) are NaN here — dropped by qv below

        p_S = torch.gather(p_blk, 2, idx_b)  # [bsz, B, k]
        p_S = torch.where(allow_sel, p_S, torch.zeros_like(p_S))
        p_S = p_S / p_S.sum(dim=-1, keepdim=True).clamp_min(_KL_EPS)  # renormalize over the selected set
        I_S = torch.gather(indexer_scores[:, q0:q1], 2, idx_b) + bias_sel  # [bsz, B, k] (-inf at masked)
        logq_S = torch.log_softmax(I_S.float(), dim=-1)
        term = p_S * (p_S.clamp_min(_KL_EPS).log() - logq_S)
        kl_bB = torch.where(allow_sel, term, torch.zeros_like(term)).sum(dim=-1)  # [bsz, B]
        if qv is not None:
            kl_bB = torch.where(qv.bool(), kl_bB, torch.zeros_like(kl_bB))  # drop pad-query (NaN) rows
            total_kl = total_kl + kl_bB.sum()
            total_cnt = total_cnt + qv.sum()
        else:
            total_kl = total_kl + kl_bB.sum()
            total_cnt = total_cnt + kl_bB.numel()

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
            # Activation-checkpoint the KL only when training (backward exists) and enabled: recompute the
            # O(T^2) score graph in backward instead of holding it across all layers. use_reentrant=False is
            # required — the inputs (frozen base) don't require grad; the grad-carrying tensors are the indexer
            # params referenced inside, and the recompute re-fires the indexer FSDP2 all-gather via __call__.
            if getattr(dsa, "kl_checkpoint", False) and self.training:
                self._dsa_kl = torch.utils.checkpoint.checkpoint(
                    _dense_warmup_kl,
                    self, hidden_states, qr, query_states, key_states, cos, sin, position_ids, attention_mask,
                    use_reentrant=False,
                )
            else:
                self._dsa_kl = _dense_warmup_kl(
                    self, hidden_states, qr, query_states, key_states, cos, sin, position_ids, attention_mask
                )
        elif dsa.mode == "sparse":
            # Phase-2: attend only over the top-k selected keys (T1). This REPLACES the dense flash path, so we
            # return directly. The selected-set indexer KL (T2) will be attached to self._dsa_kl here.
            attn_output, idx, indexer_scores = _sparse_attn(
                self, hidden_states, qr, query_states, key_states, value_states, cos, sin, position_ids, attention_mask
            )
            self._dsa_kl = _sparse_indexer_kl(
                self, query_states, key_states, indexer_scores, idx, position_ids, attention_mask
            )
            return attn_output, None, past_key_value
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
