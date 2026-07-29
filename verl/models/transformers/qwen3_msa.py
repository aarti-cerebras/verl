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
"""MSA integration for Qwen3: graft the MiniMax Sparse Attention index branch onto Qwen3's GQA.

Wires `MSAIndexer` (see msa_indexer.py) into `Qwen3Attention` via verl's monkey-patch path. The patched
forward mirrors stock `Qwen3Attention.forward` exactly — so in Phase 1 the LM output is BIT-IDENTICAL to
the stock model — and, in `dense_warmup` mode, additionally:

  * projects the index branch from `stopgrad(hidden_states)` (Eq. 11),
  * recomputes the per-GQA-group Main Branch distribution `P` over the full causal support (Eq. 9's
    teacher: per-head softmax, THEN the 1/G average — that order matters, see `_group_teacher`),
  * computes token-level index scores `S_idx` and accumulates the per-layer
    `KL(P || softmax(S_idx)) / (N * H_kv)` (Eq. 10) on `self._msa_kl`.

A forward hook on the root model reduces the per-layer KLs into `model._msa_indexer_kl` (`sum` over
layers, per Algorithm 1's `L = L_LM + lambda * SUM_layers L_KL`) which the custom FSDP engine reads into
`model_output["indexer_kl"]`.

Phase-1 properties worth stating because they are what make the phase safe:
  * attention stays DENSE (the stock attention interface is called with the stock arguments), and the
    index branch has NO output path — the paper's final recipe drops the index value head (C.3) and
    vLLM sets `sparse_disable_index_value` on every sparse layer. So the LM forward cannot change.
  * only `*.indexer.*` parameters require grad, and Eq. 11's `stopgrad(X)` keeps the KL gradient out of
    the backbone entirely.
  * `lambda` is irrelevant here: with the base frozen `L_LM` trains nothing, so the KL is the only live
    term and `lambda` merely rescales the index LR. It first matters in Phase 2.

`sparse` mode (Phase 2) is not built yet and raises.

See docs/qwen3_4b_msa/plan.md §4-§5 and docs/qwen3_4b_msa/kl_loss.md.
"""

from typing import Optional

import torch
import torch.utils.checkpoint

from verl.models.transformers.msa_indexer import MSAConfig, MSAIndexer

_KL_EPS = 1e-12


def msa_overrides_from_config(model_config) -> dict:
    """Collect MSA overrides from the (HF) model config, supporting two injection styles:

    * an `msa_overrides` dict attribute (tests / programmatic setup), or
    * flat scalar attributes `msa_<field>` (e.g. `msa_top_k`) — required for verl's `override_config`,
      whose `update_model_config` recurses into nested dict values (so a nested dict cannot be injected,
      but scalars `setattr` fine). Mirrors the `dsa_*` convention.
    """
    ov = getattr(model_config, "msa_overrides", None)
    if isinstance(ov, dict):
        return dict(ov)
    out = {}
    for field in (
        "index_dim",
        "block_size",
        "top_k",
        "init_blocks",
        "local_blocks",
        "score_type",
        "dense_prefix",
        "sparse_layers",
        "mode",
        "index_score_scale",
        "kl_block_size",
        "kl_reduction",
        "kl_checkpoint",
        "diag_interval",
        "log_per_layer",
        "warmstart_path",
        "serving_compat",
        "compile_teacher",
    ):
        val = getattr(model_config, f"msa_{field}", None)
        if val is not None:
            out[field] = val
    return out


def build_msa_config(model_config, **overrides) -> MSAConfig:
    """Build an `MSAConfig` from a live Qwen3 config. Geometry is FORCED from the model (an override that
    disagreed with the backbone would produce a checkpoint that cannot be served); everything else comes
    from `overrides`.

    `num_kv_heads` doubles as the index query head count — vLLM asserts
    `total_num_index_heads == total_num_kv_heads` in `MinimaxM3QKVParallelLinearWithIndexer.__init__`, so
    there is nothing to choose here.
    """
    head_dim = getattr(model_config, "head_dim", None) or (
        model_config.hidden_size // model_config.num_attention_heads
    )
    kw = dict(
        enabled=True,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        index_dim=head_dim,  # d_idx == d_h == 128 for every Qwen3; also what the kernels require
        rms_norm_eps=model_config.rms_norm_eps,
    )
    kw.update(overrides)
    return MSAConfig(**kw)


def attach_indexers(model, msa_cfg: MSAConfig) -> None:
    """Attach the shared `MSAConfig` to every layer's `self_attn`, and an `MSAIndexer` to the SPARSE ones.

    Dense layers (`layer_is_sparse(i) == False`) get the config but no indexer and no KL term — matching
    M3, whose `sparse_attention_freq = [0]*3 + [1]*57` leaves the first three attention layers dense.
    The patched forward branches on `hasattr(attn, "indexer")`, so dense layers run the stock path.
    """
    n_sparse = 0
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.msa = msa_cfg
        attn.msa_layer_idx = i
        if msa_cfg.layer_is_sparse(i):
            idx = MSAIndexer(msa_cfg)
            ref = next(attn.parameters())
            attn.indexer = idx.to(device=ref.device, dtype=ref.dtype)
            n_sparse += 1
    global _COMPILE_TEACHER
    _COMPILE_TEACHER = bool(getattr(msa_cfg, "compile_teacher", True))
    assert n_sparse > 0, "MSA: no sparse layers — check dense_prefix / sparse_layers"
    print(f"MSA: attached {n_sparse} indexers ({len(model.model.layers) - n_sparse} layers stay dense)")
    path = getattr(msa_cfg, "warmstart_path", None)
    if path:
        _warmstart_from_consolidated(model, path)


def _warmstart_from_consolidated(model, path: str) -> None:
    """Warm-start from a CONSOLIDATED (world-size-agnostic) state dict — indexer-only (Phase-1 ckpt; base
    stays stock) or full base+indexer (Phase-2 ckpt). Runs BEFORE FSDP wrap, so it is GPU-count-agnostic
    and yields a FRESH optimizer + step 0. Loaded with `strict=False`, but we assert the file HAS indexer
    keys (guards a typo'd path silently no-op'ing) and has NO unexpected keys (guards a name mismatch).
    """
    sd = torch.load(path, weights_only=False, map_location="cpu")
    ref = next(model.parameters())
    sd = {k: v.to(device=ref.device, dtype=ref.dtype) for k, v in sd.items()}
    n_idx = sum(1 for k in sd if ".indexer." in k)
    assert n_idx > 0, f"warmstart_path {path} has no *.indexer.* keys — is it a consolidated dict?"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"warmstart_path {path} has keys absent from the model: {list(unexpected)[:5]}"
    print(
        f"MSA warm-start: loaded {len(sd)} tensors ({n_idx} indexer + {len(sd) - n_idx} base) from {path}; "
        f"{len(missing)} params kept from base init (fresh optimizer, step 0)"
    )


def freeze_base_train_indexer(model) -> list:
    """Phase-1 freeze: `requires_grad=False` everywhere except `*.indexer.*`.

    Returns the trainable (indexer) params — hand these to the optimizer (LR ~1e-3). With the base frozen
    autograd retains no base activation graph and AdamW no-ops the frozen params. Call BEFORE FSDP wrap;
    the wrap must use `use_orig_params=True` (FSDP1) or FSDP2 for mixed `requires_grad` to be legal.
    """
    trainable = []
    for name, p in model.named_parameters():
        is_indexer = ".indexer." in name
        p.requires_grad_(is_indexer)
        if is_indexer:
            trainable.append(p)
    return trainable


def install_kl_accumulation(model) -> None:
    """Wire per-layer KL reduction + monitoring into model attributes on every forward.

    Pre-hook: reset per-layer state, stash `position_ids`/`attention_mask` on the shared config (Qwen3's
    attention forward does NOT receive `position_ids` in transformers 5.x — it gets pre-gathered
    `position_embeddings` — so the document/padding mask has to come down this side channel), bump the
    forward counter, and set `_do_diag` every `diag_interval` forwards.

    Post-hook: reduce the per-layer KLs into `model._msa_indexer_kl` and build `model._msa_metrics`
    (plain floats — the SFT trainer logs the metrics dict raw and does not unwrap `Metric` objects).
    """
    layers = model.model.layers

    def _pre_hook(module, args, kwargs):
        cfg = None
        for layer in layers:
            layer.self_attn._msa_kl = None
            layer.self_attn._msa_diag = None
            cfg = layer.self_attn.msa
        cfg._position_ids = kwargs.get("position_ids", None)
        cfg._attention_mask = kwargs.get("attention_mask", None)
        cnt = getattr(model, "_msa_fwd_count", 0) + 1
        model._msa_fwd_count = cnt
        interval = max(1, getattr(cfg, "diag_interval", 1))
        # eval/validation: always diagnose, so every held-out batch reports recall. training: gate on
        # the interval (block diagnostics cost two top-ks). Always diagnose on the first forward.
        cfg._do_diag = (not model.training) or ((cnt - 1) % interval == 0)
        return None

    def _post_hook(module, args, output):
        attns = [layer.self_attn for layer in layers]
        kl_items = [(i, a._msa_kl) for i, a in enumerate(attns) if getattr(a, "_msa_kl", None) is not None]
        if not kl_items:
            model._msa_indexer_kl = None
            model._msa_metrics = {}
            return output
        cfg = attns[0].msa
        kl_stack = torch.stack([kl for _, kl in kl_items])
        # Paper Algorithm 1: L = L_LM + lambda * SUM_layers L_KL. "mean" rescales lambda by n_sparse_layers.
        model._msa_indexer_kl = kl_stack.sum() if cfg.kl_reduction == "sum" else kl_stack.mean()
        metrics = {
            "indexer/kl_layer_mean": kl_stack.mean().item(),
            "indexer/kl_layer_min": kl_stack.min().item(),
            "indexer/kl_layer_max": kl_stack.max().item(),
            "indexer/n_sparse_layers": float(len(kl_items)),
        }
        diag_items = [(i, a._msa_diag) for i, a in enumerate(attns) if getattr(a, "_msa_diag", None) is not None]
        if diag_items:
            diags = [d for _, d in diag_items]

            def _layer_mean(key):
                return torch.stack([d[key] for d in diags]).mean().item()

            # Block-level selection quality (paper §5.2 + our main_attn_covered gate). `coverage_vs_ceiling`
            # IS the Phase-1 gate (plan §5 item 1: >= 0.90, per layer) — it is oracle-RELATIVE because an
            # absolute floor is unachievable on this model (the §12 probe caps layer 3 at 0.767).
            # Only emit what was measured: the Phase-2 training path computes just
            # `group_disagreement` (the coverage family needs the dense teacher and runs in eval only).
            present = set(diags[0].keys())
            for key in ("main_attn_covered", "coverage_ceiling", "coverage_from_forced", "learned_coverage",
                        "coverage_vs_ceiling", "block_recall", "score_recall", "group_disagreement",
                        "score_mean", "score_std", "nan_frac", "entropy_norm"):
                if key in present:
                    metrics[f"indexer/{key}"] = _layer_mean(key)
            if "attn_entropy_norm" in present:
                metrics["attn/entropy_norm"] = _layer_mean("attn_entropy_norm")
        if getattr(cfg, "log_per_layer", False):
            # Per-layer breakdown as SEPARATE keys so the logger cannot collapse them into a mean. The
            # per-layer captured/oracle ratio is what the §5 gate is actually read off.
            for i, kl in kl_items:
                metrics[f"kl_by_layer/L{i:02d}"] = kl.item()
            for i, d in diag_items:
                for key in ("learned_coverage", "coverage_vs_ceiling", "block_recall"):
                    if key in d:
                        metrics[f"indexer/{key}_by_layer/L{i:02d}"] = d[key].item()
        model._msa_metrics = metrics
        return output

    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model.register_forward_hook(_post_hook)


def _causal_doc_bias_block(position_ids, q0: int, q1: int, T: int, device, key_mask=None) -> torch.Tensor:
    """Additive mask `[bsz, q1-q0, T]` — 0 where key `j` is attendable by query `i` in `[q0, q1)`, else
    `-inf`: causal (`j <= i`) AND same-document (boundary = `position_ids == 0`) AND, if `key_mask` is
    given, the key is a real (non-pad) token. This must mirror exactly what the base attention attends
    over, or the recomputed teacher `P` is a distribution over a different support than the real one.
    """
    doc_id = (position_ids == 0).cumsum(dim=-1)  # [bsz, T] — increments at each document start
    q_pos = torch.arange(q0, q1, device=device)
    k_pos = torch.arange(T, device=device)
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [B, T]
    same_doc = doc_id[:, q0:q1].unsqueeze(2) == doc_id.unsqueeze(1)  # [bsz, B, T]
    allow = causal.unsqueeze(0) & same_doc
    if key_mask is not None:
        allow = allow & key_mask[:, None, :]
    return torch.zeros(allow.shape, device=device, dtype=torch.float32).masked_fill(~allow, float("-inf"))


def _group_teacher_impl(query_states, key_states, q0, q1, bias, n_kv_heads, scaling) -> torch.Tensor:
    """Eq. 9's per-GQA-group teacher `P` for a query tile: `[bsz, H_kv, B, T]`, fp32, detached.

    Per head: softmax over the (masked) support. THEN the `1/G` average WITHIN each group — the paper
    places the softmax INSIDE the `(1/G) sum`, so each head normalises over the support BEFORE averaging.
    Renormalise-then-average != average-then-renormalise whenever heads hold different mass in the
    support; in Phase 1 (full support) the two coincide, but writing it the wrong way here would silently
    become wrong the moment Phase 2 restricts the support.

    Heads are accumulated one at a time, so the peak extra allocation is one head's `[bsz, B, T]`
    (64 MiB at 32K/512) rather than all `H_q` at once (2.0 GiB).
    """
    bsz, H, T, _ = query_states.shape
    G = H // n_kv_heads
    p = query_states.new_zeros(bsz, n_kv_heads, q1 - q0, T, dtype=torch.float32)
    qb = query_states[:, :, q0:q1, :]
    # Accumulate one GROUP at a time (G heads per matmul), not one head at a time. Per-head looping issued
    # H_q * n_tiles * n_sparse_layers small matmuls -- 67,584 launches per 32K sequence at H_q=32 -- and
    # ran them in fp32, so the teacher was launch-bound rather than FLOP-bound. Per group it is 4x fewer,
    # 4x larger matmuls, with the dot in the model dtype and only the softmax/accumulation in fp32.
    # Peak extra is one group's [bsz, G, T_q, T] (bf16 268 MiB at 32K/512) instead of one head's 64 MiB --
    # still far below the tile's own working set.
    for r in range(n_kv_heads):
        qg = qb[:, r * G : (r + 1) * G]  # [bsz, G, T_q, d]
        s = torch.matmul(qg, key_states[:, r : r + 1].transpose(-1, -2)) * scaling  # [bsz, G, T_q, T]
        p[:, r] = torch.softmax(s.float() + bias.unsqueeze(1), dim=-1).sum(dim=1)
    return p / G  # rows with no valid key (pad queries) are NaN here and are masked out by the caller


_COMPILE_TEACHER = True  # set from MSAConfig.compile_teacher by attach_indexers
_TEACHER_COMPILED = None


def _group_teacher(query_states, key_states, q0, q1, bias, n_kv_heads, scaling) -> torch.Tensor:
    """Eq. 9 teacher, routed through a lazily `torch.compile`d implementation when enabled.

    The teacher is the single largest cost in Phase 1 -- profiled at 20.4 s of a 73 s step (and 40.7 s
    once the checkpoint recompute is counted). It is bandwidth-bound, not compute-bound: eager runs at
    ~14 TFLOP/s (~3% of an H100's bf16 peak) because the `s.float() + bias` upcast, the softmax and the
    group-sum each round-trip a 268 MB fp32 tensor through HBM -- ~10 GB of traffic per tile.

    Compiling fuses that chain. Measured at 32K/T_q=512 on one H100, against an fp64 reference:

        eager               9.65 ms   peak 1.74 GB   err 1.62e-04
        torch.compile       3.57 ms   peak 2.15 GB   err 8.42e-05    <- 2.70x AND more accurate

    It is more accurate because Inductor keeps more of the reduction in registers instead of
    materialising fp32 temporaries. (TF32 is irrelevant here -- `allow_tf32` is False on this build and
    toggling it changes neither timing nor error.) Compilation is lazy so imports stay cheap and CPU-only
    tests are unaffected, and it falls back to eager if compilation fails.
    """
    global _TEACHER_COMPILED
    if not _COMPILE_TEACHER or not query_states.is_cuda:
        return _group_teacher_impl(query_states, key_states, q0, q1, bias, n_kv_heads, scaling)
    if _TEACHER_COMPILED is None:
        try:
            _TEACHER_COMPILED = torch.compile(_group_teacher_impl, dynamic=False)
        except Exception as e:  # never let a compile failure break training
            print(f"MSA: torch.compile of the teacher failed ({type(e).__name__}), using eager: {e}")
            _TEACHER_COMPILED = _group_teacher_impl
    return _TEACHER_COMPILED(query_states, key_states, q0, q1, bias, n_kv_heads, scaling)


def _dense_warmup_kl(attn, hidden_states, query_states, key_states, cos, sin, rope_fn):
    """Per-layer index-branch KL for Phase 1 (Eq. 10), tiled over query blocks to bound memory at 32K.

    Args:
        query_states: `[bsz, H_q, T, d_h]`, post-QK-norm and post-RoPE (exactly what the attention uses).
        key_states:   `[bsz, H_kv, T, d_h]`, likewise, NOT repeat_kv'd — group `r` is `key_states[:, r]`.
        cos, sin:     the base model's rotary tables as passed to the attention (`position_embeddings`).
        rope_fn:      the base model's own `apply_rotary_pos_emb`, handed to the indexer so its positional
                      encoding cannot drift from the attention it distills.

    Returns the scalar `1/(N*H_kv) * sum_i sum_r KL(P || P_idx)` for this layer, in the compute dtype.
    Normalisation counts VALID (query, group) pairs, so padding never enters the denominator.
    """
    cfg = attn.msa
    bsz, H, T, _ = query_states.shape
    H_kv = cfg.num_kv_heads
    device, compute_dtype = query_states.device, query_states.dtype
    scaling = attn.scaling
    block = getattr(cfg, "kl_block_size", 0) or T
    tile_ckpt = bool(getattr(cfg, "kl_checkpoint", False)) and attn.training
    do_diag = bool(getattr(cfg, "_do_diag", False))

    position_ids = getattr(cfg, "_position_ids", None)
    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, T)
    am = getattr(cfg, "_attention_mask", None)
    key_mask = am.bool() if (am is not None and am.dim() == 2) else None

    # Index projections once for the whole sequence; scored per query tile against the full key set.
    # Routed through __call__ so FSDP2's hooks on a separately-wrapped indexer unit fire.
    q_idx, k_idx = attn.indexer(hidden_states, cos, sin, rope_fn=rope_fn)  # [b,H_kv,T,d], [b,1,T,d]

    z = lambda: query_states.new_zeros((), dtype=torch.float32)  # noqa: E731
    total_kl, total_cnt = z(), z()
    acc = {k: z() for k in ("cap", "oracle", "forced", "learned_coverage", "brecall", "srecall", "gdiv", "gcnt",
                            "ssum", "ssq", "scnt", "nan", "entf", "aentf", "rows")}

    def _tile_kl(q_t, k_all, q_states, k_states, pos_ids, kmask, q0, q1):
        """Grad-carrying per-tile KL, computing EVERYTHING per-tile internally.

        **Why the bias/mask/teacher are built in here and not passed in.**
        ``torch.utils.checkpoint`` must be able to replay this function in backward, so it saves every
        input TENSOR and holds it on the graph node until backward runs. Anything freshly allocated per
        tile and passed in is therefore retained for the whole layer:

            p_blk [1, H_kv, T_q, T] fp32   537 MB      <- the PER-GROUP teacher (Eq. 9)
            bias  [1, T_q, T]       fp32    67 MB
            allow [1, T_q, T]       bool    17 MB
            -> 621 MB x (T / T_q) tiles = 39.7 GB per layer at 32K/512, x33 layers = 1.31 TB.

        Crucially that total is ``n_tiles x per_tile``, i.e. **independent of ``kl_block_size``** — which
        is why shrinking the tile does not help, and why `kl_block_size` bounds only the transient working
        set, not this. Built inside instead, the saved inputs are ``q_states``/``k_states``/``k_all``/
        ``pos_ids`` — the SAME tensor objects for every tile, so stored once per layer — plus two ints.

        The price is that the teacher is recomputed in backward (one extra dense attention pass per tile).
        That is the trade checkpointing exists to make. The memory-optimal alternative is a manual
        per-tile backward using ``dL/dS_idx = P_idx - P`` in closed form, which avoids the recompute but
        hand-rolls autograd; only reach for it if measurement demands it.

        ``where`` zeroes masked keys, so an all-masked (pad) row sums to a finite 0 and its NaN teacher
        entries are discarded. Returns ``[bsz, H_kv, T_q]``.
        """
        bias_t = _causal_doc_bias_block(pos_ids, q0, q1, T, q_t.device, key_mask=kmask)
        allow_t = bias_t == 0.0
        with torch.no_grad():
            p_t = _group_teacher(q_states, k_states, q0, q1, bias_t, H_kv, scaling)
        s = attn.indexer.scores(q_t, k_all, attn_bias=bias_t)  # [bsz, H_kv, T_q, T]
        log_q = torch.log_softmax(s, dim=-1)
        term = p_t * (torch.log(p_t.clamp_min(_KL_EPS)) - log_q)
        return torch.where(allow_t.unsqueeze(1), term, torch.zeros_like(term)).sum(dim=-1)

    for q0 in range(0, T, block):
        q1 = min(q0 + block, T)
        qv = key_mask[:, q0:q1].to(torch.float32) if key_mask is not None else None  # [bsz, B]
        # Only shared/small tensors cross the checkpoint boundary (see _tile_kl's docstring).
        args = (q_idx[:, :, q0:q1], k_idx, query_states, key_states, position_ids, key_mask, q0, q1)
        if tile_ckpt:
            kl_blk = torch.utils.checkpoint.checkpoint(_tile_kl, *args, use_reentrant=False)
        else:
            kl_blk = _tile_kl(*args)  # [bsz, H_kv, B]

        if qv is not None:
            total_kl = total_kl + (kl_blk * qv.unsqueeze(1)).sum()
            total_cnt = total_cnt + qv.sum() * H_kv  # valid (query, group) pairs -> Eq. 10's 1/(N*H_kv)
        else:
            total_kl = total_kl + kl_blk.sum()
            total_cnt = total_cnt + kl_blk.numel()

        if do_diag or getattr(cfg, "_capture_p", False):
            # Rebuild bias/teacher here too. Under no_grad nothing is retained, and diagnostics are gated
            # to every `diag_interval` forwards, so the extra teacher pass is amortised.
            with torch.no_grad():
                dbias = _causal_doc_bias_block(position_ids, q0, q1, T, device, key_mask=key_mask)
                dp = _group_teacher(query_states, key_states, q0, q1, dbias, H_kv, scaling)
                if getattr(cfg, "_capture_p", False):  # test hook: validate against eager attention
                    attn._msa_p = dp.detach()
                if do_diag:
                    _accumulate_block_diag(attn, q_idx, k_idx, dbias, dbias == 0.0, dp, q0, q1, qv, acc)
                del dbias, dp

    attn._msa_diag = _finalize_diag(acc) if do_diag else None
    return (total_kl / total_cnt.clamp_min(1)).to(compute_dtype)


def block_selection_metrics(
    P_b: torch.Tensor,
    sel: torch.Tensor,
    valid_blocks: torch.Tensor,
    top_k: int,
    local_blocks: int = 0,
    init_blocks: int = 0,
) -> dict:
    """Block-level selection quality for one query tile. Pure function — unit-tested directly against the
    worked example in docs/qwen3_4b_msa/kl_loss.md §1.2 step 7.

    Args:
        P_b: teacher block mass `[bsz, H_kv, B, n_blocks]` (sums to 1 over blocks for a valid row).
        sel: index-branch selection `[bsz, H_kv, B, k]`, block ids, `-1` in unused slots.
        valid_blocks: `[B]` number of causally-visible blocks per query row, `(pos + B_k) // B_k`.
        top_k: `k`.

    Returns per-(batch, group, row) tensors `[bsz, H_kv, B]` for the mass metrics and `[bsz, B]` for
    `group_disagreement`.

    Definitions — the two paper metrics are VERBATIM from §5.2 (p.9): *"let I* be the corresponding
    Top-k block set induced by the Main Branch scores and let Î be the Index Branch selection. Block
    recall is |I* ∩ Î|/|I*|, while score recall is sum_{b in I*∩Î} P_b / sum_{b in I*} P_b, where P_b is
    the Main Branch attention probability summed over tokens in block b."*

        block_recall  = |I* ∩ Î| / |I*|                     set identity
        score_recall  = mass(I* ∩ Î) / mass(I*)             mass of the AGREED blocks, oracle-relative
        main_attn_covered = mass(Î)                             ours: mass of EVERYTHING selected (absolute)
        coverage_ceiling  = mass(I*)                            the ceiling at this budget

    Note `main_attn_covered >= mass(I* ∩ Î)` by construction, so `main_attn_covered/coverage_ceiling >= score_recall`:
    the gap is mass the index branch found in blocks the oracle did not rank top-k. Neither paper metric
    can detect a budget that is simply too small (both are oracle-relative), which is why the absolute
    `main_attn_covered` exists and why the Phase-1 gate is `main_attn_covered / coverage_ceiling` (plan §5, §12).
    """
    bsz, H_kv, B, n_blocks = P_b.shape
    dev = P_b.device
    visible = torch.arange(n_blocks, device=dev)[None, None, None, :] < valid_blocks.to(dev)[None, None, :, None]

    # I* must be restricted to visible blocks BEFORE the top-k: on a row with fewer than k visible
    # blocks, `topk` would otherwise pad I* with arbitrary zero-mass invisible blocks, inflating |I*| and
    # under-reporting block recall at every early query position. With this mask,
    # |I*| == |Î| == min(k, valid_blocks), so the paper's |I*| denominator is literally correct.
    mass = P_b.nan_to_num(0.0) * visible
    star = mass.masked_fill(~visible, -1.0).topk(min(top_k, n_blocks), dim=-1).indices
    hit_star = torch.zeros_like(P_b, dtype=torch.bool).scatter_(-1, star, True) & visible
    # Î: a `-1` slot must not mark block 0, hence clamp + the visible mask.
    hit_sel = torch.zeros_like(P_b, dtype=torch.bool).scatter_(-1, sel.clamp_min(0), True) & visible

    inter = hit_sel & hit_star
    oracle = (mass * hit_star).sum(-1)
    # group_disagreement: 0 when all H_kv groups pick identical blocks, 1 when fully disjoint. The paper
    # reports that groups DO diverge (Appendix A/Fig. 5: "different groups attend to different long-range
    # stripes while sharing the common local and sink patterns"), so a value pinned at 0 means the
    # per-group capacity MSA is built around is being wasted.
    distinct = hit_sel.any(dim=1).sum(-1).float()  # [bsz, B] union over groups
    per_group = hit_sel.sum(-1).float().mean(dim=1)  # [bsz, B] mean |Î| per group
    gdiv = (
        ((distinct - per_group) / (per_group * (H_kv - 1)).clamp_min(_KL_EPS)).clamp(0, 1)
        if H_kv > 1
        else torch.zeros_like(distinct)
    )
    # --- forced-block baseline and the SKILL of the learned slots -----------------------------------
    # MSA always selects the local block (paper §3.2), and on a causal LM that block carries most of the
    # attention mass — measured on Qwen3-0.6B at k=16, the forced block ALONE captures 0.783 while the
    # oracle is 0.979. So `main_attn_covered` and `coverage_vs_ceiling` are dominated by the forcing, and an
    # UNTRAINED indexer already reads coverage_vs_ceiling = 0.903. A 0.90 gate would pass at init.
    #
    #     learned_coverage = (main_attn_covered - coverage_from_forced)
    #                        / (coverage_ceiling  - coverage_from_forced)          in [0, 1]
    #
    # 0.0 = the learned slots add nothing over the forced block; 1.0 = they reach the ceiling. This is the
    # quantity to gate Phase 1 on; `coverage_vs_ceiling` stays as the absolute deployment number.
    captured = (mass * hit_sel).sum(-1)
    blk_ids = torch.arange(n_blocks, device=dev)[None, None, None, :]
    forced = torch.zeros_like(hit_sel)
    if local_blocks:
        forced |= blk_ids >= (valid_blocks.to(dev)[None, None, :, None] - local_blocks).clamp_min(0)
    if init_blocks:
        forced |= blk_ids < init_blocks
    forced &= visible
    coverage_from_forced = (mass * forced).sum(-1)
    headroom = (oracle - coverage_from_forced).clamp_min(_KL_EPS)
    return {
        "main_attn_covered": captured,
        "coverage_ceiling": oracle,
        "coverage_from_forced": coverage_from_forced,
        "learned_coverage": ((captured - coverage_from_forced) / headroom).clamp(0, 1),
        "block_recall": inter.sum(-1).float() / hit_star.sum(-1).clamp_min(1).float(),
        "score_recall": (mass * inter).sum(-1) / oracle.clamp_min(_KL_EPS),
        "group_disagreement": gdiv,
    }


def _accumulate_block_diag(attn, q_idx, k_idx, bias, allow, p_blk, q0, q1, qv, acc) -> None:
    """Block-level selection diagnostics for one query tile (no_grad, `diag_interval`-gated).

    All five quality numbers are BLOCK-level, unlike the loss: the paper's `score recall` / `block recall`
    (§5.2) plus our absolute `main_attn_covered` and its oracle-relative ratio. Neither paper metric can
    detect a budget that is simply too small — that is what `main_attn_covered` is for — while an absolute
    floor is unachievable on this model, which is why the gate is the ratio (plan §5 item 1, §12).
    """
    cfg = attn.msa
    bsz, H_kv, B, T = p_blk.shape
    bk, k = cfg.block_size, cfg.top_k
    dev = p_blk.device

    # Score ONE GROUP AT A TIME. The full [bsz, H_kv, B, T] score tensor is 512 MiB at 32K/512, and the
    # health/entropy reductions below would each allocate another one (a boolean-mask gather is worse: it
    # materialises a variable-size copy AND an expanded bool mask). Per group the working set is
    # [bsz, 1, B, T] = 64 MiB, while everything retained across groups is block-level (4 MiB). Diagnostics
    # must never be the memory peak — the loss tiles are.
    query_pos = torch.arange(q0, q1, device=dev)
    m_parts, ent_parts, aent_parts = [], [], []
    valid_log = allow.sum(-1).clamp_min(2).float().log()  # [bsz, B]
    for r in range(H_kv):
        s_r = attn.indexer.scores(q_idx[:, r : r + 1, q0:q1], k_idx, attn_bias=bias)  # [bsz,1,B,T]
        m_parts.append(attn.indexer.block_scores(s_r))
        keep = allow.unsqueeze(1)  # [bsz,1,B,T]
        s_keep = s_r.masked_fill(~keep, 0.0)
        acc["ssum"] += s_keep.sum()
        acc["ssq"] += (s_keep * s_keep).sum()
        acc["scnt"] += keep.sum()
        acc["nan"] += (s_r.isnan() & keep).sum()
        ent_parts.append(torch.special.entr(torch.softmax(s_r, dim=-1)).sum(-1) / valid_log.unsqueeze(1))
        aent_parts.append(
            torch.special.entr(p_blk[:, r : r + 1].nan_to_num(0.0)).sum(-1) / valid_log.unsqueeze(1)
        )
        del s_r, s_keep
    M = torch.cat(m_parts, dim=1)  # [bsz,H_kv,B,n_blocks]
    sel = attn.indexer.select_blocks(M, query_pos)  # [bsz,H_kv,B,k] (may contain -1)

    # Teacher block mass P_b (sums to 1 over blocks for a valid row).
    n_blocks = M.shape[-1]
    pad = n_blocks * bk - T
    p_pad = torch.nn.functional.pad(p_blk, (0, pad)) if pad else p_blk
    P_b = p_pad.view(bsz, H_kv, B, n_blocks, bk).sum(-1)  # [bsz,H_kv,B,n_blocks]

    valid_blocks = (query_pos + bk) // bk
    md = block_selection_metrics(
        P_b, sel, valid_blocks, top_k=k, local_blocks=cfg.local_blocks, init_blocks=cfg.init_blocks
    )
    captured, oracle = md["main_attn_covered"], md["coverage_ceiling"]
    block_recall, score_recall, gdiv = md["block_recall"], md["score_recall"], md["group_disagreement"]

    rows = qv if qv is not None else torch.ones(bsz, B, device=dev)
    rows_g = rows.unsqueeze(1)  # [bsz,1,B]
    acc["cap"] += (captured * rows_g).sum()
    acc["oracle"] += (oracle * rows_g).sum()
    acc["forced"] += (md["coverage_from_forced"] * rows_g).sum()
    acc["learned_coverage"] += (md["learned_coverage"] * rows_g).sum()
    acc["brecall"] += (block_recall * rows_g).sum()
    acc["srecall"] += (score_recall * rows_g).sum()
    acc["rows"] += rows.sum() * H_kv
    acc["gdiv"] += (gdiv * rows).sum()
    acc["gcnt"] += rows.sum()

    # Entropy health, token-level: entropy_norm ~1.0 means near-uniform (a healthy distillation start);
    # a falling value flags an over-committed or saturated index branch. attn/* is the teacher's.
    # (Both were accumulated per group above; score mean/std/nan likewise.)
    acc["entf"] += (torch.cat(ent_parts, dim=1) * rows_g).sum()
    acc["aentf"] += (torch.cat(aent_parts, dim=1) * rows_g).sum()


def _finalize_diag(acc, full: bool = True) -> dict:
    """Reduce the accumulators to scalars.

    ``full=False`` is the Phase-2 *training* path, where only ``group_disagreement`` was computed (the
    coverage family needs the dense teacher — see `_accumulate_sparse_diag`). Emit ONLY what was actually
    measured: logging un-computed metrics as 0.0 would draw a flat zero line in wandb that reads exactly
    like a collapsed indexer.
    """
    if not full:
        return {"group_disagreement": (acc["gdiv"] / acc["gcnt"].clamp_min(1)).detach()}
    rows = acc["rows"].clamp_min(1)
    mean = acc["ssum"] / acc["scnt"].clamp_min(1)
    var = (acc["ssq"] / acc["scnt"].clamp_min(1) - mean * mean).clamp_min(0)
    cap, oracle = acc["cap"] / rows, acc["oracle"] / rows
    return {
        "main_attn_covered": cap.detach(),
        "coverage_ceiling": oracle.detach(),
        "coverage_from_forced": (acc["forced"] / rows).detach(),
        "learned_coverage": (acc["learned_coverage"] / rows).detach(),
        "coverage_vs_ceiling": (cap / oracle.clamp_min(_KL_EPS)).detach(),
        "block_recall": (acc["brecall"] / rows).detach(),
        "score_recall": (acc["srecall"] / rows).detach(),
        "group_disagreement": (acc["gdiv"] / acc["gcnt"].clamp_min(1)).detach(),
        "score_mean": mean.detach(),
        "score_std": var.sqrt().detach(),
        "nan_frac": (acc["nan"] / acc["scnt"].clamp_min(1)).detach(),
        "entropy_norm": (acc["entf"] / rows).detach(),
        "attn_entropy_norm": (acc["aentf"] / rows).detach(),
    }


def _selected_token_index(indexer, sel: torch.Tensor, seq_len: int):
    """Expand selected BLOCK ids to token positions (Phase-2 step 10).

    ``sel`` ``[b, H_kv, T_q, k]`` block ids with ``-1`` in unused slots → ``(tok, slot_ok)``:
      * ``tok`` ``[b, H_kv, T_q, k*B_k]`` int64 token positions, clamped into ``[0, seq_len)``
      * ``slot_ok`` marks slots that came from a real block.

    ``slot_ok`` is load-bearing, not defensive: a ``-1`` slot clamps to block 0, whose tokens may be
    perfectly legal for this query, so it would otherwise be silently attended to.
    """
    bk = indexer.cfg.block_size
    b, h, tq, k = sel.shape
    off = torch.arange(bk, device=sel.device)
    tok = (sel.clamp_min(0).unsqueeze(-1) * bk + off).reshape(b, h, tq, k * bk).clamp_max(seq_len - 1)
    slot_ok = (sel >= 0).unsqueeze(-1).expand(b, h, tq, k, bk).reshape(b, h, tq, k * bk)
    return tok, slot_ok


def _sparse_tile(attn, q_tile, key_states, value_states, q_idx_tile, k_idx, position_ids, key_mask, q0, q1):
    """One query tile of the Phase-2 sparse path: select, attend, and build the KL. Checkpointable.

    Returns ``(out [b, H_q, T_q, d], kl_rows [b, H_kv, T_q], sel [b, H_kv, T_q, k])``.

    Everything grad-carrying lives in here, so under ``torch.utils.checkpoint`` only ONE tile's graph is
    live — in particular the ``[b, H_kv, T_q, N]`` index scores (512 MiB at 32K) and the
    ``[b, H_kv, T_q, k*B_k, d]`` gathers (~2.1 GB each). The attention and the KL must be computed in ONE
    checkpointed function rather than two: a KL stashed as a side effect during a ``no_grad`` first pass
    would carry no graph and silently contribute zero gradient (phase2_plan §1.1).
    """
    ix = attn.indexer
    b, h_q, tq, d = q_tile.shape
    h_kv = key_states.shape[1]
    g = h_q // h_kv
    t = key_states.shape[2]

    # Build the additive causal/document/padding bias HERE rather than accepting it as an argument.
    # `torch.utils.checkpoint` saves its input TENSORS for recompute, and this bias is [1, T_q, T] fp32 =
    # 67 MB at 32K/512 — freshly allocated per tile. Passing it in retained 64 tiles x 33 layers =
    # ~142 GB. Rebuilding it from `position_ids` (one shared [b, T] int64 tensor, 262 KB, saved once
    # across all tiles) costs a few comparisons and retains nothing.
    device = q_tile.device
    bias = _causal_doc_bias_block(position_ids, q0, q1, t, device, key_mask=key_mask)
    query_pos = torch.arange(q0, q1, device=device)

    # --- C. selection (steps 6-10) -------------------------------------------------------------------
    s_idx = ix.scores(q_idx_tile, k_idx, attn_bias=bias)  # [b, H_kv, T_q, T] -- the KL student's logits
    m_blk = ix.block_scores(s_idx)  # [b, H_kv, T_q, n_blocks]
    sel = ix.select_blocks(m_blk, query_pos)  # [b, H_kv, T_q, k]; top-k is detached inside
    tok, slot_ok = _selected_token_index(ix, sel, t)  # [b, H_kv, T_q, M]

    # Gather the additive bias at the selected positions (the DSA trick): a selected key that is
    # non-causal, cross-document or padding already carries -inf, so validity needs no extra bookkeeping.
    bias_sel = torch.gather(bias.unsqueeze(1).expand(b, h_kv, tq, t), 3, tok)
    allow = (bias_sel == 0.0) & slot_ok
    # A row with no valid slot (a pad query) would softmax over all -inf -> NaN, poisoning the LM path.
    # Force slot 0 open; such rows are dropped from the KL and masked out of the LM loss anyway.
    first = torch.arange(allow.shape[-1], device=allow.device) == 0
    allow = allow | (~allow.any(dim=-1, keepdim=True) & first)
    neg = torch.zeros_like(bias_sel).masked_fill(~allow, float("-inf"))  # [b, H_kv, T_q, M]

    # --- D. sparse attention, the LM path (steps 11-16) ----------------------------------------------
    m = tok.shape[-1]
    gidx = tok.reshape(b, h_kv, tq * m, 1).expand(b, h_kv, tq * m, d)
    k_g = torch.gather(key_states, 2, gidx).reshape(b, h_kv, tq, m, d)
    v_g = torch.gather(value_states, 2, gidx).reshape(b, h_kv, tq, m, d)
    qg = q_tile.view(b, h_kv, g, tq, d)
    scores = torch.einsum("bhgqd,bhqmd->bhgqm", qg, k_g) * attn.scaling + neg.unsqueeze(2)
    attn_f32 = torch.softmax(scores.float(), dim=-1)  # [b, H_kv, G, T_q, M]
    out = torch.einsum("bhgqm,bhqmd->bhgqd", attn_f32.to(v_g.dtype), v_g).reshape(b, h_q, tq, d)

    # --- E. the KL (steps 17-20) ---------------------------------------------------------------------
    # Teacher: taken from the fp32 softmax (not the bf16 cast, so the target is not quantised), detached
    # FIRST so no graph node is built, then averaged over the group's G heads at the PROBABILITY level
    # (Eq. 9). This is the entire "free teacher" property -- no second attention computation.
    p = attn_f32.detach().mean(dim=2)  # [b, H_kv, T_q, M]
    student = torch.gather(s_idx, 3, tok) + neg  # identical support to the teacher
    log_q = torch.log_softmax(student, dim=-1)
    term = p * (torch.log(p.clamp_min(_KL_EPS)) - log_q)
    kl_rows = torch.where(allow, term, torch.zeros_like(term)).sum(dim=-1)  # [b, H_kv, T_q]
    return out, kl_rows, sel


def _sparse_attn_and_kl(attn, hidden_states, query_states, key_states, value_states, cos, sin, rope_fn):
    """Phase-2: block-sparse attention over the selected blocks + the restricted-support KL, query-tiled.

    Returns ``(attn_output [b, T, H_q*d], kl_scalar)``. ``L_LM`` reaches the base through the gathered
    K/V; ``L_KL`` reaches only the index branch (block ids detached in `select_blocks`, index projections
    read detached hidden states per Eq. 11).
    """
    cfg = attn.msa
    b, h_q, t, d = query_states.shape
    h_kv = key_states.shape[1]
    device, compute_dtype = query_states.device, query_states.dtype
    block = getattr(cfg, "kl_block_size", 0) or t
    tile_ckpt = bool(getattr(cfg, "kl_checkpoint", False)) and attn.training
    do_diag = bool(getattr(cfg, "_do_diag", False))

    position_ids = getattr(cfg, "_position_ids", None)
    if position_ids is None:
        position_ids = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
    am = getattr(cfg, "_attention_mask", None)
    key_mask = am.bool() if (am is not None and am.dim() == 2) else None

    # Index projections once for the whole sequence, routed through __call__ (FSDP2 hooks).
    q_idx, k_idx = attn.indexer(hidden_states, cos, sin, rope_fn=rope_fn)

    z = lambda: query_states.new_zeros((), dtype=torch.float32)  # noqa: E731
    total_kl, total_cnt = z(), z()
    acc = {k: z() for k in ("cap", "oracle", "forced", "learned_coverage", "brecall", "srecall", "gdiv",
                            "gcnt", "ssum", "ssq", "scnt", "nan", "entf", "aentf", "rows")}
    outs = []

    for q0 in range(0, t, block):
        q1 = min(q0 + block, t)
        qv = key_mask[:, q0:q1].to(torch.float32) if key_mask is not None else None
        # Pass position_ids/key_mask, NOT the materialised bias: see the note in _sparse_tile. The bias is
        # rebuilt inside so the checkpoint retains only small shared tensors.
        args = (attn, query_states[:, :, q0:q1], key_states, value_states, q_idx[:, :, q0:q1], k_idx,
                position_ids, key_mask, q0, q1)
        if tile_ckpt:
            out_t, kl_rows, sel = torch.utils.checkpoint.checkpoint(_sparse_tile, *args, use_reentrant=False)
        else:
            out_t, kl_rows, sel = _sparse_tile(*args)
        outs.append(out_t)

        if qv is not None:
            total_kl = total_kl + (kl_rows * qv.unsqueeze(1)).sum()
            total_cnt = total_cnt + qv.sum() * h_kv  # valid (query, group) pairs -> Eq. 10's 1/(N*H_kv)
        else:
            total_kl = total_kl + kl_rows.sum()
            total_cnt = total_cnt + kl_rows.numel()

        if do_diag:
            with torch.no_grad():  # rebuilt here too; under no_grad nothing is retained
                dbias = _causal_doc_bias_block(position_ids, q0, q1, t, device, key_mask=key_mask)
                _accumulate_sparse_diag(attn, sel, q0, q1, qv, acc, full=not attn.training,
                                        query_states=query_states, key_states=key_states, bias=dbias)
                del dbias

    attn._msa_diag = _finalize_diag(acc, full=not attn.training) if do_diag else None
    kl = (total_kl / total_cnt.clamp_min(1)).to(compute_dtype)
    o = torch.cat(outs, dim=2).transpose(1, 2).reshape(b, t, h_q * d)
    return o, kl


def _accumulate_sparse_diag(attn, sel, q0, q1, qv, acc, full, query_states, key_states, bias) -> None:
    """Phase-2 diagnostics for one query tile.

    ``group_disagreement`` needs only ``sel``, so it is free on every diag step. The COVERAGE family needs
    the FULL-support teacher, which Phase 2 deliberately does not compute — a dense pass costs
    8.80 TFLOP/layer at 32K (phase2_plan §3.2). So it is gated on ``full`` (eval mode), where the two
    forwards that the sparsity-cost metric needs are already being paid for. Consequence: in *training*
    the Phase-1 gate metrics are unavailable by design; read them off validation.
    """
    cfg = attn.msa
    b, h_kv, tq, _ = sel.shape
    dev = sel.device
    bk = cfg.block_size
    t = key_states.shape[2]
    n_blocks = (t + bk - 1) // bk
    query_pos = torch.arange(q0, q1, device=dev)
    vb = (query_pos + bk) // bk
    rows = qv if qv is not None else torch.ones(b, tq, device=dev)

    if full:
        p_blk = _group_teacher(query_states, key_states, q0, q1, bias, h_kv, attn.scaling)
        pad = n_blocks * bk - t
        p_pad = torch.nn.functional.pad(p_blk, (0, pad)) if pad else p_blk
        p_b = p_pad.view(b, h_kv, tq, n_blocks, bk).sum(-1)
        md = block_selection_metrics(p_b, sel, vb, top_k=cfg.top_k, local_blocks=cfg.local_blocks,
                                     init_blocks=cfg.init_blocks)
        rows_g = rows.unsqueeze(1)
        for src, dst in (("main_attn_covered", "cap"), ("coverage_ceiling", "oracle"),
                         ("coverage_from_forced", "forced"), ("learned_coverage", "learned_coverage"),
                         ("block_recall", "brecall"), ("score_recall", "srecall")):
            acc[dst] += (md[src] * rows_g).sum()
        acc["gdiv"] += (md["group_disagreement"] * rows).sum()
        acc["gcnt"] += rows.sum()
        acc["rows"] += rows.sum() * h_kv
        return

    # Cheap path: how much the groups' selections differ, from `sel` alone.
    hit = torch.zeros(b, h_kv, tq, n_blocks, dtype=torch.bool, device=dev).scatter_(-1, sel.clamp_min(0), True)
    hit &= torch.arange(n_blocks, device=dev)[None, None, None, :] < vb[None, None, :, None]
    distinct = hit.any(dim=1).sum(-1).float()
    per_group = hit.sum(-1).float().mean(dim=1)
    gdiv = (((distinct - per_group) / (per_group * (h_kv - 1)).clamp_min(_KL_EPS)).clamp(0, 1)
            if h_kv > 1 else torch.zeros_like(distinct))
    acc["gdiv"] += (gdiv * rows).sum()
    acc["gcnt"] += rows.sum()
    acc["rows"] += rows.sum() * h_kv


def qwen3_msa_attn_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Patched `Qwen3Attention.forward`.

    The LM path is a line-for-line mirror of stock `Qwen3Attention.forward` (transformers 5.x), so in
    `dense_warmup` the output is bit-identical to the stock model — asserted by
    `tests/msa/test_qwen3_msa_dense_equivalence.py`. We inline the body rather than delegating so q/k/v
    are computed ONCE and shared with the KL path; the equivalence test is what guards against
    transformers drift in those lines.
    """
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    cfg = getattr(self, "msa", None)
    if cfg is not None and getattr(self, "indexer", None) is not None:
        if cfg.mode == "dense_warmup":
            # Phase 1: attention stays dense (below); the index branch only produces a KL term. The index
            # branch has no output path (paper C.3 drops the value head), so the LM forward is untouched.
            self._msa_kl = _dense_warmup_kl(
                self, hidden_states, query_states, key_states, cos, sin, apply_rotary_pos_emb
            )
        elif cfg.mode == "sparse":
            # Phase 2: the sparse path REPLACES the dense attention below and returns early — the LM
            # output now comes from attending over the selected blocks only, and the KL teacher is that
            # same attention's own weights (phase2_plan §3).
            attn_output, self._msa_kl = _sparse_attn_and_kl(
                self, hidden_states, query_states, key_states, value_states, cos, sin, apply_rotary_pos_emb
            )
            return self.o_proj(attn_output), None
        else:
            raise ValueError(f"unknown MSA mode {cfg.mode!r}")

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
