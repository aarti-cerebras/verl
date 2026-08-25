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
"""DSA integration for Qwen3: token-granular sparse attention on a **GQA** backbone, no MLA.

Wires `Qwen3DSAIndexer` (see qwen3_dsa_indexer.py) into `Qwen3Attention` via verl's monkey-patch path.

**Phase 1 (`dense_warmup`).** Attention stays dense and the LM output is **bit-identical** to the stock
model, so the phase carries zero capability risk. The indexer runs as a pure side-channel KL:

    p[t, s] = (1/H_q) * sum_h softmax_s( q_h[t] . k_{g(h)}[s] * scaling + bias )    # detached teacher
    L_KL    = KL( p[t, :] || softmax(I[t, :]) )

That teacher is DeepSeek §2.1.1 verbatim — *"aggregate the main attention scores by summing across all
attention heads[, then] L1-normalized along the sequence dimension"*: each head's softmax sums to 1, so the
head-sum has L1 norm `H_q` and dividing by it IS the L1 normalization. It is also gradient-equivalent to
Keye's per-group form at 1/G the memory, because cross-entropy is linear in the target:

    sum_g KL(p_g || q) = G * KL(p_bar || q) + G * JSD(p_1..p_G),   p_bar = mean_g p_g

with the JSD term constant in `q`. Three consequences, all of which shape the code below: `q = p_bar` is the
exact minimizer (so the shared-across-heads selection is not a compromise, it is what the objective asks
for); one `[b, T_q, T]` teacher replaces eight; and the loss has an irreducible floor of `G*JSD`, so it is
NOT a progress metric — read `indexer/topk_recall`, and note that `indexer/group_jsd` below measures that
floor directly. See docs/qwen3_4b_dsa/plan_v2.md §3.

**Phase 2 (`sparse`).** The sparse path REPLACES dense attention. Per query tile it selects the top-k
tokens, attends only over them, and takes the KL over that same set — with the teacher read straight off
the sparse attention's own softmax. That is free, and it is the published recipe, not an optimization: in
the sparse stage the main attention *is* sparse, so DeepSeek's `p` construction applied to it yields
exactly this (plan_v2.md §4.1). `minicpm_dsa._sparse_indexer_kl` instead runs an extra dense pass — 8.8
TFLOP/layer at 32K, 8x the attention it supervises, and a *different* target (heads weighted by how much
mass they hold inside the selected set, rather than one vote each). **Do not port it.** The structural
reason it happened is worth remembering: its `_sparse_attn` returns before the KL is built, so the softmax
is out of scope and there is nothing to reuse. Hence the fused `_sparse_tile` here.

A forward hook on the root model reduces the per-layer KLs into `model._dsa_indexer_kl` and builds
`model._dsa_metrics`; the custom FSDP engine reads them via `workers/utils/losses.py`. The `_dsa_*` names
and the `config.dsa_enabled` flag are reused deliberately — that naming IS the integration with
`fsdp_utils`, the engine, the losses and the SFT trainer, none of which need changes (plan_v2.md §6).

See plan_v2.md §2 (architecture), §2.4 (init), §3 (Phase 1), §4 (Phase 2).
"""

import random
from typing import Optional

import torch
import torch.utils.checkpoint

from verl.models.transformers.qwen3_dsa_indexer import NO_DECAY_SUFFIXES, Qwen3DSAConfig, Qwen3DSAIndexer

_KL_EPS = 1e-12

# Overridable knobs collected from flat `dsa_<field>` attributes on the HF config. Flat scalars are
# required because verl's `override_config` -> `update_model_config` recurses into nested dict values, so a
# nested dict cannot be injected but scalars `setattr` fine. Mirrors the minicpm3/MSA convention.
_OVERRIDE_FIELDS = (
    "n_heads",
    "head_dim",
    "rope_head_dim",
    "top_k",
    "dense_prefix",
    "sparse_layers",
    "mode",
    "sigma_target",
    "kl_block_size",
    "kl_reduction",
    "kl_checkpoint",
    "compile_teacher",
    "diag_interval",
    "log_per_layer",
    "diag_overlap_sample",
    "full_support_kl_prob",
    "fp8",
    "fp8_ue8m0",
    "rotate_activation",
    "serving_compat",
    "warmstart_path",
)


def dsa_overrides_from_config(model_config) -> dict:
    """Collect DSA overrides from the HF config: a `dsa_overrides` dict (tests) or flat `dsa_<field>`."""
    ov = getattr(model_config, "dsa_overrides", None)
    if isinstance(ov, dict):
        return dict(ov)
    out = {}
    for field in _OVERRIDE_FIELDS:
        val = getattr(model_config, f"dsa_{field}", None)
        if val is not None:
            out[field] = val
    return out


def build_dsa_config(model_config, **overrides) -> Qwen3DSAConfig:
    """Build a `Qwen3DSAConfig` from a live Qwen3 config.

    Geometry is FORCED from the model — an override that disagreed with the backbone would produce a
    checkpoint that cannot be served. `rope_theta` matters especially: the indexer builds its own rotary at
    the base theta, and a mismatch there would silently decouple the indexer's positional encoding from the
    attention it distills.
    """
    head_dim = getattr(model_config, "head_dim", None) or (
        model_config.hidden_size // model_config.num_attention_heads
    )
    # transformers 5.x normalizes rope config into `rope_scaling` and MOVES `rope_theta` inside it --
    # `Qwen3Config.rope_theta` does not exist on 5.3, and `rope_scaling` is populated as
    # `{"rope_theta": ..., "rope_type": "default"}` even when the checkpoint's config.json has
    # `rope_scaling: null`. So test the rope TYPE (presence means nothing) and read theta from either place.
    rs = getattr(model_config, "rope_scaling", None) or {}
    rope_type = rs.get("rope_type", rs.get("type", "default"))
    if rope_type != "default":
        # The indexer builds its own rotary from theta alone. If the base model gains YaRN/linear/mrope
        # scaling it must be mirrored in `_IndexerRotary`, or the indexer's positional encoding silently
        # decouples from the attention it distills.
        raise ValueError(
            f"base model has rope_type={rope_type!r} (rope_scaling={rs}); the DSA indexer's own rotary "
            "implements plain RoPE only — mirror the scaling in _IndexerRotary before training"
        )
    theta = rs.get("rope_theta") or getattr(model_config, "rope_theta", None)
    if theta is None:
        raise ValueError("cannot determine rope_theta from the model config; the indexer rotary needs it")
    kw = dict(
        enabled=True,
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        num_kv_heads=model_config.num_key_value_heads,
        rope_theta=float(theta),
    )
    kw.update(overrides)
    cfg = Qwen3DSAConfig(**kw)
    if cfg.head_dim != head_dim:
        # Not fatal — the indexer head dim is independent of the attention head dim (16x64 vs 32x128) —
        # but flag it so a typo'd override does not pass silently.
        print(f"DSA: indexer head_dim={cfg.head_dim} differs from attention head_dim={head_dim} (expected)")
    return cfg


def attach_indexers(model, dsa_cfg: Qwen3DSAConfig) -> None:
    """Attach the shared config to every decoder layer's `self_attn`, and a `Qwen3DSAIndexer` to the
    SPARSE ones (`dsa_cfg.layer_is_sparse(i)`).

    Layers in `[0, dense_prefix)` get the config but no indexer and no KL term. The patched forward
    branches on `hasattr(self, "indexer")`, so those layers run the stock dense path untouched — which is
    also why `dsa_layer_idx` must be the TRUE layer index and not a count of indexers attached so far.

    **The per-layer `hidden_rms` is the whole point of this function.** Each indexer is initialised against
    `rms(layer.input_layernorm.weight)`, which is (approximately) the RMS of the hidden states that layer's
    indexer will read. On Qwen3-4B-Thinking that spans 186x (0.025 at layer 0, 4.709 at layer 34), and since
    the initial score scale is proportional to it, a constant init would leave the early layers with scores
    of essentially zero — and therefore ~100x less gradient into `wq`/`wk`, which are fed only through the
    gate — while the last layers start already committed (`entropy_frac` 0.91 vs 1.00). It would also defeat
    `kl_reduction="mean"`, whose purpose is a readable `grad_norm`: the layer-mean would be dominated by the
    few late layers. See plan_v2.md §2.4; verified in tests/dsa/test_qwen3_dsa_indexer.py.
    """
    n = 0
    n_layers = len(model.model.layers)
    rms_lo, rms_hi = float("inf"), 0.0
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.dsa = dsa_cfg
        attn.dsa_layer_idx = i
        if not dsa_cfg.layer_is_sparse(i):
            continue
        with torch.no_grad():
            hidden_rms = layer.input_layernorm.weight.float().pow(2).mean().sqrt().item()
        rms_lo, rms_hi = min(rms_lo, hidden_rms), max(rms_hi, hidden_rms)
        idx = Qwen3DSAIndexer(dsa_cfg, hidden_rms=hidden_rms)
        ref = next(attn.parameters())
        attn.indexer = idx.to(device=ref.device, dtype=ref.dtype)
        n += 1
    assert n > 0, (
        f"DSA: no sparse layers — dense_prefix={dsa_cfg.dense_prefix} covers all {n_layers} layers "
        f"(sparse_layers={dsa_cfg.sparse_layers!r})"
    )
    print(
        f"DSA: attached {n} indexers ({dsa_cfg.n_heads}x{dsa_cfg.head_dim}, top_k={dsa_cfg.top_k}); "
        f"{n_layers - n} layer(s) stay dense (dense_prefix={dsa_cfg.dense_prefix}, "
        f"sparse_layers={dsa_cfg.sparse_layers!r}); "
        f"input_layernorm rms spans {rms_lo:.3f}..{rms_hi:.3f} ({rms_hi / max(rms_lo, 1e-9):.0f}x) and the "
        f"weights_proj init absorbs it"
    )
    path = getattr(dsa_cfg, "warmstart_path", None)
    if path:
        _warmstart_from_consolidated(model, path)


def _warmstart_from_consolidated(model, path: str) -> None:
    """Warm-start from a CONSOLIDATED (world-size-agnostic) state dict — indexer-only (from a Phase-1 ckpt;
    base stays stock) or full base+indexer (from a Phase-2 ckpt). Runs BEFORE the FSDP wrap, so it is
    GPU-count-agnostic and yields a FRESH optimizer at step 0. Loaded with `strict=False`, but we assert the
    file HAS indexer keys (guards a typo'd path silently no-op'ing) and NO unexpected keys (guards a name
    mismatch, e.g. loading a MiniCPM3 indexer into this one)."""
    sd = torch.load(path, weights_only=False, map_location="cpu")
    ref = next(model.parameters())
    sd = {k: v.to(device=ref.device, dtype=ref.dtype) for k, v in sd.items()}
    n_idx = sum(1 for k in sd if ".indexer." in k)
    assert n_idx > 0, f"warmstart_path {path} has no *.indexer.* keys — is it a consolidated dict?"
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"warmstart_path {path} has keys absent from the model: {list(unexpected)[:5]}"
    print(
        f"DSA warm-start: loaded {len(sd)} tensors ({n_idx} indexer + {len(sd) - n_idx} base) from {path}; "
        f"{len(missing)} params kept from base init (fresh optimizer, step 0)"
    )


def freeze_base_train_indexer(model) -> list:
    """Phase-1 freeze: `requires_grad=False` everywhere except `*.indexer.*`.

    Returns the trainable (indexer) params. With the base frozen, autograd retains no base activation graph
    and AdamW no-ops the frozen params. Call BEFORE the FSDP wrap; the wrap must use `use_orig_params=True`
    (FSDP1) or FSDP2 for mixed `requires_grad` in one flat parameter to be legal.
    """
    trainable = []
    for name, p in model.named_parameters():
        is_indexer = ".indexer." in name
        p.requires_grad_(is_indexer)
        if is_indexer:
            trainable.append(p)
    return trainable


def indexer_param_groups(model, weight_decay: float) -> list:
    """Optimizer param groups that exclude the indexer norms and gate from weight decay.

    Decay pulls the norm gains toward 0, which suppresses the whole branch (strictly worse than MSA's
    Gemma-style parameterization, where decay pulls the gain toward 1 and is harmless), and decaying
    `weights_proj` toward 0 severs the ONLY gradient path into `wq`/`wk`
    (docs/dsa_grad_norm_debugging.md issue #2). See plan_v2.md §2.4.

    **NOT WIRED IN, deliberately.** The FSDP engine owns optimizer construction
    (`workers/engine/fsdp/transformer_impl.py::_build_optimizer`), and reaching in there would change
    behaviour for the MiniCPM3 Phase-2 path that shares that branch — which the isolation rule forbids. The
    launch scripts instead set `optim.weight_decay=0.0` globally, which is not a workaround for Phase 1:
    every trainable parameter in that phase is either a norm gain, the gate, or a projection whose scale is
    forward-invisible (a norm follows it), so decay has nothing useful to act on. Phase 2 trains the base as
    well, so if base decay is ever wanted, this is the helper `_build_optimizer` should call — and it is
    tested (`tests/dsa/test_qwen3_dsa.py::test_param_groups_exclude_norms_and_gate_from_decay`) so it stays
    correct until then.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith(NO_DECAY_SUFFIXES) else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def install_kl_accumulation(model) -> None:
    """Wire per-layer KL reduction + monitoring into model attributes on every forward.

    Pre-hook: reset per-layer state, stash `position_ids`/`attention_mask` on the shared config, bump the
    forward counter, set `_do_diag`, and draw the `full_support_kl_prob` coin.

    The `position_ids` side channel is not a convenience: transformers 5.x `Qwen3Attention.forward` receives
    pre-gathered `position_embeddings`, NOT `position_ids`, so the document/padding mask and the indexer's
    own rotary have no other way to reach the attention module. Without it the causal/document support is
    wrong and the teacher normalizes over a different key set than the real attention.

    Post-hook: reduce the per-layer KLs into `model._dsa_indexer_kl` and build `model._dsa_metrics` as plain
    floats — the SFT trainer logs the metrics dict raw and does not unwrap `Metric` objects.
    """
    layers = model.model.layers

    def _pre_hook(module, args, kwargs):
        cfg = None
        for layer in layers:
            layer.self_attn._dsa_kl = None
            layer.self_attn._dsa_diag = None
            cfg = layer.self_attn.dsa
        cfg._position_ids = kwargs.get("position_ids", None)
        cfg._attention_mask = kwargs.get("attention_mask", None)
        cnt = getattr(model, "_dsa_fwd_count", 0) + 1
        model._dsa_fwd_count = cnt
        if torch.cuda.is_available():
            # Peak memory over the PREVIOUS forward+backward window, read here and then reset.
            #
            # It has to be read in the PRE-hook, not the post-hook: backward runs after the post-hook, and
            # backward is where the peak is (the checkpoint recompute materialises the gathered K/V). And it
            # has to be read in-process: sampling `nvidia-smi` externally misses it entirely — a 15 s poll
            # reported 53.2 GiB for a step that OOM'd at 78.25 GiB, because the spike lives inside one
            # backward and lasts well under a second. So the values reported at step N describe step N-1.
            model._dsa_prev_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            model._dsa_prev_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
            torch.cuda.reset_peak_memory_stats()
        interval = max(1, getattr(cfg, "diag_interval", 1))
        # eval/validation: always diagnose, so every held-out batch reports recall. training: gate on the
        # interval (diagnostics cost extra top-ks and a per-group teacher). Always diagnose on forward #1 —
        # that is where the §2.4 init-entropy gate is read.
        cfg._do_diag = (not model.training) or ((cnt - 1) % interval == 0)
        # One coin per forward, shared by all layers, so the whole model sees the same support this step.
        p_full = float(getattr(cfg, "full_support_kl_prob", 0.0) or 0.0)
        cfg._full_support = bool(p_full > 0.0 and random.random() < p_full)
        return None

    def _post_hook(module, args, output):
        attns = [layer.self_attn for layer in layers]
        kl_items = [(i, a._dsa_kl) for i, a in enumerate(attns) if getattr(a, "_dsa_kl", None) is not None]
        if not kl_items:
            model._dsa_indexer_kl = None
            model._dsa_metrics = {}
            return output
        cfg = attns[0].dsa
        kl_stack = torch.stack([kl for _, kl in kl_items])
        # Pure gradient scale (disjoint per-layer params), so the optimum is identical; "mean" keeps
        # grad_norm readable rather than pinned against clip_grad. See plan_v2.md §3.
        model._dsa_indexer_kl = kl_stack.sum() if cfg.kl_reduction == "sum" else kl_stack.mean()
        kl_det = kl_stack.detach()  # detach before .item(): the KL carries grad, and .item() on a
        # grad-requiring tensor warns on every step, which buries real warnings in the training log
        metrics = {
            "indexer/kl_layer_mean": kl_det.mean().item(),
            "indexer/kl_layer_min": kl_det.min().item(),
            "indexer/kl_layer_max": kl_det.max().item(),
            # SPARSE layers, i.e. those that actually have an indexer -- 32, not 36, at dense_prefix=4.
            # Watch it on step 1: it is the cheapest confirmation that dense_prefix took effect.
            "indexer/n_layers": float(len(kl_items)),
        }
        # Peak of the previous step's forward+backward (see the pre-hook). Emitted every step because "does
        # it fit" is not the question -- the headroom is, and it is what sizes SEQ_LEN, kl_block_size and the
        # long-context bands.
        if getattr(model, "_dsa_prev_peak_gb", None):
            metrics["mem/peak_alloc_gb"] = model._dsa_prev_peak_gb
            metrics["mem/peak_reserved_gb"] = model._dsa_prev_reserved_gb
        diag_items = [(i, a._dsa_diag) for i, a in enumerate(attns) if getattr(a, "_dsa_diag", None) is not None]
        if diag_items:
            diags = [d for _, d in diag_items]
            present = set(diags[0].keys())

            def _layer_mean(key):
                return torch.stack([d[key] for d in diags]).mean().item()

            # `indexer/` = student health. Emit ONLY what was measured: logging an un-computed metric as 0.0
            # draws a flat zero line in wandb that reads exactly like a collapsed indexer.
            for key in (
                "topk_recall",
                "topk_overlap",
                "entropy_frac",
                "score_mean",
                "score_std",
                "nan_frac",
                "local_mass",
                "group_recall_min",
                "group_jsd",
            ):
                if key in present:
                    metrics[f"indexer/{key}"] = _layer_mean(key)
            # the teacher's own entropy goes in its own wandb section
            if "attn_entropy_frac" in present:
                metrics["attn/entropy_frac"] = _layer_mean("attn_entropy_frac")
        if getattr(cfg, "log_per_layer", False):
            # Separate scalar keys so the logger cannot collapse them to a mean. The Phase-1 gate is per
            # layer (recall >= 0.95, and init entropy_frac in [0.99, 1.0]), so this is what it is read off.
            for i, kl in kl_items:
                metrics[f"kl_by_layer/L{i:02d}"] = kl.detach().item()
            for i, d in diag_items:
                for key in ("topk_recall", "entropy_frac", "group_recall_min"):
                    if key in d:
                        metrics[f"indexer/{key}_by_layer/L{i:02d}"] = d[key].item()
        model._dsa_metrics = metrics
        return output

    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model.register_forward_hook(_post_hook)


# ---------------------------------------------------------------------------------------------------
# masks and the teacher
# ---------------------------------------------------------------------------------------------------


def _causal_doc_bias_block(position_ids, q0: int, q1: int, T: int, device, key_mask=None) -> torch.Tensor:
    """Additive mask `[bsz, q1-q0, T]` — 0 where key `j` is attendable by query `i` in `[q0, q1)`, else
    `-inf`: causal (`j <= i`) AND same-document (boundary = `position_ids == 0`) AND, if `key_mask` is
    given, the key is a real (non-pad) token.

    This must mirror exactly what the base attention attends over, or the recomputed teacher is a
    distribution over a different support than the real one.
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


def _head_avg_teacher_impl(q_block, key_states, bias, n_kv_heads, scaling) -> torch.Tensor:
    """The Phase-1 teacher for one query tile: `[bsz, T_q, T]`, fp32, to be detached by the caller.

    Per head: softmax over the (masked) support. THEN the average over all `H_q` heads — the softmax is
    INSIDE the sum. Averaging logits first and normalizing after is a different distribution whenever heads
    disagree; on the full support of Phase 1 the difference is invisible in aggregate, but writing it the
    wrong way here would silently become wrong the moment Phase 2 restricts the support.

    Accumulates one GQA GROUP at a time (`G` heads per matmul against that group's single KV head), summing
    the group's heads immediately into a `[bsz, T_q, T]` fp32 buffer. Two reasons: the peak extra allocation
    is one group's `[bsz, G, T_q, T]` (268 MB bf16 at 32K/512) rather than all `H_q` at once (2.0 GB), and
    the retained result is a single `[bsz, T_q, T]` (67 MB) rather than MSA's per-group `[bsz, H_kv, T_q, T]`
    (537 MB) — the 1/G saving that the `sum_g -> G*mean_g` identity buys.

    `q_block` arrives ALREADY SLICED, and that is a `torch.compile` requirement rather than a style choice:
    `q0`/`q1` as Python ints make Dynamo guard on their VALUES and recompile per tile, which at
    `kl_block_size=512` and T=32768 blows `recompile_limit` at tile 8 and silently runs the remaining 56
    tiles EAGER. With the tile pre-sliced, every tile presents identical shapes and one compiled kernel
    serves all of them. `T` therefore comes from `key_states`, not from the (shorter) query tensor.
    """
    bsz, H, T_q, _ = q_block.shape
    T = key_states.shape[2]
    G = H // n_kv_heads
    p = q_block.new_zeros(bsz, T_q, T, dtype=torch.float32)
    for r in range(n_kv_heads):
        qg = q_block[:, r * G : (r + 1) * G]  # [bsz, G, T_q, d]
        s = torch.matmul(qg, key_states[:, r : r + 1].transpose(-1, -2)) * scaling  # [bsz, G, T_q, T]
        p += torch.softmax(s.float() + bias.unsqueeze(1), dim=-1).sum(dim=1)
    return p / H  # rows with no valid key (pad queries) are NaN here; the caller masks them out


_TEACHER_COMPILED = None


def _teacher_from_block(q_block, key_states, bias, n_kv_heads, scaling, compile_it=True):
    """The teacher for an ALREADY-SLICED query block, routed through a lazily-compiled implementation.

    It is the largest single cost in Phase 1 and is bandwidth-bound (the upcast, the softmax and the
    group-sum each round-trip a large fp32 tensor through HBM), so fusing that chain is the biggest
    available win; MSA measured 2.70x on the equivalent function, *and* lower error against an fp64
    reference because Inductor keeps more of the reduction in registers. Compilation is lazy so imports stay
    cheap and CPU-only tests are unaffected, and it falls back to eager if compilation fails.
    """
    global _TEACHER_COMPILED
    if not compile_it or not q_block.is_cuda:
        return _head_avg_teacher_impl(q_block, key_states, bias, n_kv_heads, scaling)
    if _TEACHER_COMPILED is None:
        try:
            _TEACHER_COMPILED = torch.compile(_head_avg_teacher_impl, dynamic=False)
        except Exception as e:  # never let a compile failure break training
            print(f"DSA: torch.compile of the teacher failed ({type(e).__name__}), using eager: {e}")
            _TEACHER_COMPILED = _head_avg_teacher_impl
    return _TEACHER_COMPILED(q_block, key_states, bias, n_kv_heads, scaling)


def _head_avg_teacher(query_states, key_states, q0, q1, bias, n_kv_heads, scaling, compile_it=True):
    """Convenience wrapper for call sites holding the FULL `query_states`: slice, then route.

    The slice happens HERE, outside the compiled region, which is the `torch.compile` requirement documented
    in `_head_avg_teacher_impl` — `q0`/`q1` must never reach a compiled function as int arguments.
    """
    return _teacher_from_block(query_states[:, :, q0:q1, :], key_states, bias, n_kv_heads, scaling, compile_it)


# ---------------------------------------------------------------------------------------------------
# Phase 1 — dense warm-up
# ---------------------------------------------------------------------------------------------------


def _dense_warmup_kl(attn, hidden_states, query_states, key_states):
    """Per-layer indexer KL for Phase 1, tiled over query blocks to bound memory at 32K.

    Args:
        query_states: `[bsz, H_q, T, d_h]`, post-QK-norm and post-RoPE — exactly what the attention uses.
        key_states:   `[bsz, H_kv, T, d_h]`, NOT repeat_kv'd; group `r` is `key_states[:, r]`.

    Returns the scalar KL averaged over valid (non-pad) query rows, in the compute dtype.
    """
    cfg = attn.dsa
    bsz, H, T, _ = query_states.shape
    device, compute_dtype = query_states.device, query_states.dtype
    scaling = attn.scaling
    block = getattr(cfg, "kl_block_size", 0) or T
    tile_ckpt = bool(getattr(cfg, "kl_checkpoint", False)) and attn.training
    do_diag = bool(getattr(cfg, "_do_diag", False))
    compile_it = bool(getattr(cfg, "compile_teacher", True))

    position_ids = getattr(cfg, "_position_ids", None)
    if position_ids is None:
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(bsz, T)
    am = getattr(cfg, "_attention_mask", None)
    key_mask = am.bool() if (am is not None and am.dim() == 2) else None

    # Index projections once for the whole sequence, scored per query tile against the full key set.
    # Routed through __call__ (not `.project()`) so FSDP2's hooks on the separately-wrapped indexer unit
    # fire: the pre-forward all-gather and the pre-backward gate that triggers the grad reduce-scatter.
    q_idx, k_idx, weights = attn.indexer(hidden_states, position_ids, return_projection=True)

    z = lambda: query_states.new_zeros((), dtype=torch.float32)  # noqa: E731
    total_kl, total_cnt = z(), z()
    acc = {k: z() for k in ("recall", "overlap", "ovcnt", "ssum", "ssq", "scnt", "nan", "ent", "aent",
                            "local", "grec", "gjsd", "rows")}

    def _tile_kl(q_t, k_all, w_t, q_states, k_states, pos_ids, kmask, q0, q1):
        """Grad-carrying per-tile KL, building the bias and teacher INTERNALLY.

        `torch.utils.checkpoint` saves every input TENSOR for the backward replay and holds it on the graph
        node, so anything freshly allocated per tile and passed IN is retained for the whole layer: a
        `[1, T_q, T]` fp32 bias is 67 MB, times `T/T_q` tiles times 36 layers is ~155 GB. Crucially that
        total is `n_tiles * per_tile`, i.e. INDEPENDENT of `kl_block_size` — which is why shrinking the tile
        does not help. Built inside instead, the saved inputs are the same shared tensors for every tile
        (stored once per layer) plus two ints. The price is recomputing the teacher in backward, which is
        the trade checkpointing exists to make.
        """
        bias_t = _causal_doc_bias_block(pos_ids, q0, q1, k_states.shape[2], q_t.device, key_mask=kmask)
        allow_t = bias_t == 0.0
        with torch.no_grad():
            p_t = _head_avg_teacher(q_states, k_states, q0, q1, bias_t, cfg.num_kv_heads, scaling, compile_it)
        s = attn.indexer.scores(q_t, k_all, w_t, attn_bias=bias_t)  # [bsz, T_q, T]
        log_q = torch.log_softmax(s.float(), dim=-1)
        term = p_t * (torch.log(p_t.clamp_min(_KL_EPS)) - log_q)
        # `where` zeroes masked keys, so an all-masked (pad-query) row sums to a finite 0 and its NaN
        # teacher entries are discarded rather than poisoning the loss.
        return torch.where(allow_t, term, torch.zeros_like(term)).sum(dim=-1)  # [bsz, T_q]

    for q0 in range(0, T, block):
        q1 = min(q0 + block, T)
        qv = key_mask[:, q0:q1].to(torch.float32) if key_mask is not None else None  # [bsz, B]
        args = (q_idx[:, q0:q1], k_idx, weights[:, q0:q1], query_states, key_states, position_ids, key_mask, q0, q1)
        if tile_ckpt:
            kl_blk = torch.utils.checkpoint.checkpoint(_tile_kl, *args, use_reentrant=False)
        else:
            kl_blk = _tile_kl(*args)

        if qv is not None:
            total_kl = total_kl + (kl_blk * qv).sum()  # count only real (non-pad) query rows
            total_cnt = total_cnt + qv.sum()
        else:
            total_kl = total_kl + kl_blk.sum()
            total_cnt = total_cnt + kl_blk.numel()

        if do_diag or getattr(cfg, "_capture_p", False):
            with torch.no_grad():  # nothing is retained; diagnostics are `diag_interval`-gated
                dbias = _causal_doc_bias_block(position_ids, q0, q1, T, device, key_mask=key_mask)
                dp = _head_avg_teacher(query_states, key_states, q0, q1, dbias, cfg.num_kv_heads, scaling,
                                       compile_it)
                if getattr(cfg, "_capture_p", False):
                    # test hook: stash the full [bsz, T, T] teacher for comparison against eager attention.
                    # Accumulated ACROSS tiles rather than overwritten, so the hook works at any
                    # `kl_block_size` (the MiniCPM3 version only worked when the tile covered the sequence).
                    if q0 == 0 or getattr(attn, "_dsa_p", None) is None or attn._dsa_p.shape[1] != T:
                        attn._dsa_p = query_states.new_zeros(bsz, T, T, dtype=torch.float32)
                    attn._dsa_p[:, q0:q1] = dp.detach()
                if do_diag:
                    _accumulate_diag(attn, q_idx, k_idx, weights, dbias, dbias == 0.0, dp, q0, q1, qv, acc,
                                     query_states, key_states, scaling)
                del dbias, dp

    attn._dsa_diag = _finalize_diag(acc) if do_diag else None
    return (total_kl / total_cnt.clamp_min(1)).to(compute_dtype)


def _accumulate_diag(attn, q_idx, k_idx, weights, bias, allow, p_blk, q0, q1, qv, acc,
                     query_states, key_states, scaling) -> None:
    """Monitoring diagnostics for one query tile (no_grad, `diag_interval`-gated).

    `topk_recall` is *the* Phase-1 number: the teacher mass that falls inside the indexer's top-k. Note it
    is a MASS metric and mass is dominated by local tokens — an indexer can sit at 0.96 recall while
    systematically dropping low-mass needles, which is why the plan also gates on a separate needle-rank
    probe. `local_mass` says whether recency is being learned at all.

    `group_recall_min` and `group_jsd` interrogate the one architectural assumption with no fallback: that a
    single shared top-k can serve all `H_kv` groups (Qwen3-4B has 8, twice Keye's 4). `group_jsd` is exactly
    the `JSD` term in the `sum_g KL = G*KL(p_bar||q) + G*JSD` decomposition — i.e. the irreducible floor of
    the paper's per-group loss, and the principled measure of how much the groups disagree. Both are built
    from per-group teachers computed ONE GROUP AT A TIME so the peak stays at one `[bsz, T_q, T]` (67 MB)
    rather than the `[bsz, H_kv, T_q, T]` (537 MB) that Phase 1 deliberately avoids.
    """
    cfg = attn.dsa
    bsz, B, T = p_blk.shape
    device = p_blk.device
    k = min(cfg.top_k, T)
    rows = qv if qv is not None else torch.ones(bsz, B, device=device)
    rows_bool = rows.bool()

    idf = attn.indexer.scores(q_idx[:, q0:q1], k_idx, weights[:, q0:q1], attn_bias=bias).float()
    topk_i = idf.topk(k, dim=-1).indices  # [bsz, B, k]

    recall = torch.where(rows_bool, p_blk.gather(-1, topk_i).sum(-1), torch.zeros(1, device=device))
    acc["recall"] += recall.sum()
    acc["rows"] += rows.sum()

    # overlap with the teacher's own top-k, via a boolean membership mask (O(B*T)) rather than the O(B*k^2)
    # pairwise compare, so the [.., k, k] tensor never forms.
    sample = int(getattr(cfg, "diag_overlap_sample", 0) or 0)
    sel = (torch.linspace(0, B - 1, sample, device=device).round().long()
           if (sample and B > sample) else slice(None))
    p_sel = p_blk[:, sel]
    mask = torch.zeros_like(p_sel, dtype=torch.bool).scatter_(-1, p_sel.topk(k, dim=-1).indices, True)
    overlap = mask.gather(-1, topk_i[:, sel]).float().sum(-1) / k
    ov_rows = rows[:, sel]
    acc["overlap"] += (overlap * ov_rows).sum()
    acc["ovcnt"] += ov_rows.sum()

    fin = idf[allow]  # finite valid scores only
    acc["ssum"] += fin.sum()
    acc["ssq"] += (fin * fin).sum()
    acc["scnt"] += fin.numel()
    acc["nan"] += torch.isnan(fin).float().sum()

    valid_log = allow.sum(-1).clamp_min(2).float().log()  # [bsz, B]
    q_dist = torch.softmax(idf, dim=-1)  # masked keys carry -inf -> 0
    # A row with NO attendable key (a pad query at the very start) softmaxes over all -inf and yields NaN.
    # `* rows` would not clear it (NaN * 0 == NaN), so drop those rows with `where` before summing.
    ent = torch.special.entr(q_dist).sum(-1).nan_to_num(0.0) / valid_log
    aent = torch.special.entr(p_blk.nan_to_num(0.0)).sum(-1) / valid_log
    acc["ent"] += torch.where(rows_bool, ent, torch.zeros_like(ent)).sum()
    acc["aent"] += torch.where(rows_bool, aent, torch.zeros_like(aent)).sum()

    # local_mass: teacher mass in the 128 keys immediately preceding (and including) the query
    q_pos = torch.arange(q0, q1, device=device)
    keys = torch.arange(T, device=device)
    local = ((keys[None, :] <= q_pos[:, None]) & (keys[None, :] > q_pos[:, None] - 128))[None]
    local_mass = (p_blk.nan_to_num(0.0) * local).sum(-1)
    acc["local"] += torch.where(rows_bool, local_mass, torch.zeros_like(local_mass)).sum()

    # per-group teachers, one group at a time
    G = cfg.group_size
    grec_min = torch.full((bsz, B), float("inf"), device=device)
    jsd = torch.zeros(bsz, B, device=device)
    logp_bar = p_blk.clamp_min(_KL_EPS).log()
    for r in range(cfg.num_kv_heads):
        qg = query_states[:, r * G : (r + 1) * G, q0:q1]
        s = torch.matmul(qg, key_states[:, r : r + 1].transpose(-1, -2)) * scaling
        p_g = torch.softmax(s.float() + bias.unsqueeze(1), dim=-1).mean(dim=1)  # [bsz, B, T]
        grec_min = torch.minimum(grec_min, p_g.gather(-1, topk_i).sum(-1))
        term = p_g * (p_g.clamp_min(_KL_EPS).log() - logp_bar)
        jsd += torch.where(allow, term, torch.zeros_like(term)).sum(-1)
        del s, p_g
    acc["grec"] += (torch.where(rows_bool, grec_min, torch.zeros(1, device=device))).sum()
    acc["gjsd"] += ((jsd / cfg.num_kv_heads) * rows).sum()


def _finalize_diag(acc) -> dict:
    rows = acc["rows"].clamp_min(1)
    mean = acc["ssum"] / acc["scnt"].clamp_min(1)
    var = (acc["ssq"] / acc["scnt"].clamp_min(1) - mean * mean).clamp_min(0)
    return {
        "topk_recall": (acc["recall"] / rows).detach(),
        "topk_overlap": (acc["overlap"] / acc["ovcnt"].clamp_min(1)).detach(),
        "entropy_frac": (acc["ent"] / rows).detach(),
        "attn_entropy_frac": (acc["aent"] / rows).detach(),
        "score_mean": mean.detach(),
        "score_std": var.sqrt().detach(),
        "nan_frac": (acc["nan"] / acc["scnt"].clamp_min(1)).detach(),
        "local_mass": (acc["local"] / rows).detach(),
        "group_recall_min": (acc["grec"] / rows).detach(),
        "group_jsd": (acc["gjsd"] / rows).detach(),
    }


# ---------------------------------------------------------------------------------------------------
# Phase 2 — sparse
# ---------------------------------------------------------------------------------------------------


def _sparse_tile(attn, q_tile, key_states, value_states, q_idx_t, k_idx, w_t, position_ids, key_mask, q0, q1,
                 full_support=False):
    """One query tile of the Phase-2 sparse path: select, attend, and build the KL. Checkpointable.

    Returns `(out [b, H_q, T_q, d], kl_rows [b, T_q], idx [b, T_q, k])`.

    Everything grad-carrying lives in here, so under `torch.utils.checkpoint` only ONE tile's graph is live
    — in particular the `[b, T_q, T]` index scores and the `[b, H_kv, T_q, k, d]` gathers (~2.1 GB each at
    32K/512, which is why Phase 2 wants `kl_block_size=256`). The attention and the KL MUST be computed in
    one function rather than two: a KL stashed as a side effect during a `no_grad` first pass would carry no
    graph and contribute exactly zero gradient, silently. It is also what keeps the teacher free — split
    them and the softmax goes out of scope, which is precisely how `minicpm_dsa` ended up recomputing a
    dense pass (plan_v2.md §4.1).

    The bias is built here rather than passed in, for the retention reason in `_tile_kl`.
    """
    cfg = attn.dsa
    b, h_q, tq, d = q_tile.shape
    h_kv = key_states.shape[1]
    g = h_q // h_kv
    t = key_states.shape[2]
    top_k = min(cfg.top_k, t)

    bias = _causal_doc_bias_block(position_ids, q0, q1, t, q_tile.device, key_mask=key_mask)  # [b, tq, t]

    # --- selection. Scores carry gradient into the indexer's params only (the projection read detached
    # hidden states); `idx` is detached because top-k is non-differentiable, which is what keeps the LM
    # loss from reaching the indexer.
    s_idx = attn.indexer.scores(q_idx_t, k_idx, w_t, attn_bias=bias)  # [b, tq, t] — the KL student's logits
    idx = s_idx.detach().topk(top_k, dim=-1).indices  # [b, tq, k] (stop-grad: top-k is non-differentiable)

    # Gather the additive bias at the selected positions (the DSA trick): a selected key that is
    # non-causal, cross-document or padding already carries -inf, so validity needs no extra bookkeeping.
    bias_sel = torch.gather(bias, 2, idx)  # [b, tq, k]
    allow = bias_sel == 0.0
    # A row with no valid slot (a pad query) would softmax over all -inf -> NaN, poisoning the LM path.
    # Force slot 0 open; such rows are dropped from the KL and masked out of the LM loss anyway.
    first = torch.arange(top_k, device=allow.device) == 0
    allow = allow | (~allow.any(dim=-1, keepdim=True) & first)
    neg = torch.zeros_like(bias_sel).masked_fill(~allow, float("-inf"))  # [b, tq, k]

    # --- sparse attention, the LM path. Grad flows to the base through the gathered K/V.
    gidx = idx.reshape(b, 1, tq * top_k, 1).expand(b, h_kv, tq * top_k, d)
    k_g = torch.gather(key_states, 2, gidx).reshape(b, h_kv, tq, top_k, d)
    v_g = torch.gather(value_states, 2, gidx).reshape(b, h_kv, tq, top_k, d)
    qg = q_tile.view(b, h_kv, g, tq, d)
    scores = torch.einsum("bhgqd,bhqkd->bhgqk", qg, k_g) * attn.scaling + neg[:, None, None]
    attn_f32 = torch.softmax(scores.float(), dim=-1)  # [b, h_kv, g, tq, k]
    out = torch.einsum("bhgqk,bhqkd->bhgqd", attn_f32.to(v_g.dtype), v_g).reshape(b, h_q, tq, d)

    # --- the KL. The teacher is the sparse attention's OWN fp32 softmax, detached FIRST so no graph node
    # is built, then averaged over all H_q heads. This is DeepSeek's `p` construction applied to a sparse
    # attention — no second attention pass, and no dense normalizer to compute.
    if full_support:
        # `full_support_kl_prob` fired: supervise over the FULL causal support instead of the selected set,
        # while the LM path above still runs sparse. Computed HERE, inside the checkpointed tile, so the
        # dense teacher and the [b, tq, t] student logits are recomputed in backward rather than retained
        # across every tile of every layer (which would be ~4 GB at 32K/256 before the layer even finishes).
        with torch.no_grad():
            # q_tile is already the sliced block, so route straight to the pre-sliced entry point.
            p_full = _teacher_from_block(
                q_tile, key_states, bias, h_kv, attn.scaling, bool(getattr(cfg, "compile_teacher", True))
            )
        allow_full = bias == 0.0
        log_q_full = torch.log_softmax(s_idx.float(), dim=-1)
        term_full = p_full * (torch.log(p_full.clamp_min(_KL_EPS)) - log_q_full)
        kl_rows = torch.where(allow_full, term_full, torch.zeros_like(term_full)).sum(dim=-1)
        return out, kl_rows, idx

    p = attn_f32.detach().mean(dim=(1, 2))  # [b, tq, k]
    student = torch.gather(s_idx, 2, idx) + neg  # identical support to the teacher
    log_q = torch.log_softmax(student.float(), dim=-1)
    term = p * (torch.log(p.clamp_min(_KL_EPS)) - log_q)
    kl_rows = torch.where(allow, term, torch.zeros_like(term)).sum(dim=-1)  # [b, tq]
    return out, kl_rows, idx


def _sparse_attn_and_kl(attn, hidden_states, query_states, key_states, value_states):
    """Phase 2: token-sparse attention over the selected set + the selected-set KL, query-tiled.

    Returns `(attn_output [b, T, H_q*d], kl_scalar)`. `L_LM` reaches the base through the gathered K/V;
    `L_KL` reaches only the indexer (the projection reads detached hidden states, the top-k is detached, and
    the teacher is detached). All three edges are load-bearing (arXiv 2512.02556 §2.1.1).
    """
    cfg = attn.dsa
    b, h_q, t, d = query_states.shape
    device, compute_dtype = query_states.device, query_states.dtype
    block = getattr(cfg, "kl_block_size", 0) or t
    tile_ckpt = bool(getattr(cfg, "kl_checkpoint", False)) and attn.training
    do_diag = bool(getattr(cfg, "_do_diag", False))
    full_support = bool(getattr(cfg, "_full_support", False))

    position_ids = getattr(cfg, "_position_ids", None)
    if position_ids is None:
        position_ids = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
    am = getattr(cfg, "_attention_mask", None)
    key_mask = am.bool() if (am is not None and am.dim() == 2) else None

    q_idx, k_idx, weights = attn.indexer(hidden_states, position_ids, return_projection=True)

    z = lambda: query_states.new_zeros((), dtype=torch.float32)  # noqa: E731
    total_kl, total_cnt = z(), z()
    acc_mass, acc_rows = z(), z()
    outs = []

    for q0 in range(0, t, block):
        q1 = min(q0 + block, t)
        qv = key_mask[:, q0:q1].to(torch.float32) if key_mask is not None else None
        args = (attn, query_states[:, :, q0:q1], key_states, value_states, q_idx[:, q0:q1], k_idx,
                weights[:, q0:q1], position_ids, key_mask, q0, q1, full_support)
        if tile_ckpt:
            out_t, kl_rows, idx = torch.utils.checkpoint.checkpoint(_sparse_tile, *args, use_reentrant=False)
        else:
            out_t, kl_rows, idx = _sparse_tile(*args)
        outs.append(out_t)

        if qv is not None:
            total_kl = total_kl + (kl_rows * qv).sum()
            total_cnt = total_cnt + qv.sum()
        else:
            total_kl = total_kl + kl_rows.sum()
            total_cnt = total_cnt + kl_rows.numel()

        if do_diag:
            with torch.no_grad():
                # The coverage family needs the full-support teacher, which Phase 2 deliberately does not
                # compute (a dense pass is 8.80 TFLOP/layer at 32K). What IS free is the per-head share of
                # attention mass landing inside the selected set — review item: its spread across heads is
                # what says whether the free teacher's equal-vote weighting differs measurably from the
                # dense-normalized alternative, and it converges to 1 as recall rises.
                bias_d = _causal_doc_bias_block(position_ids, q0, q1, t, device, key_mask=key_mask)
                allow_d = torch.gather(bias_d, 2, idx) == 0.0
                rows = qv if qv is not None else torch.ones(b, q1 - q0, device=device)
                acc_mass += (allow_d.float().mean(-1) * rows).sum()
                acc_rows += rows.sum()
                del bias_d, allow_d

    attn._dsa_diag = {"selected_valid_frac": (acc_mass / acc_rows.clamp_min(1)).detach()} if do_diag else None
    kl = (total_kl / total_cnt.clamp_min(1)).to(compute_dtype)
    o = torch.cat(outs, dim=2).transpose(1, 2).reshape(b, t, h_q * d)
    return o, kl


# ---------------------------------------------------------------------------------------------------
# the patched forward
# ---------------------------------------------------------------------------------------------------


def qwen3_dsa_attn_forward(
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
    `dense_warmup` the output is bit-identical to the stock model. We inline the body rather than delegating
    so q/k/v are computed ONCE and shared with the KL path; `tests/dsa/test_qwen3_dsa_dense_equivalence.py`
    is what guards those lines against transformers drift.
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

    cfg = getattr(self, "dsa", None)
    # The `indexer is not None` test is what makes `dense_prefix` work, in BOTH modes: `attach_indexers`
    # gives every layer the shared config but only the sparse ones an indexer, so a dense-prefix layer
    # falls through to the stock attention below and contributes no KL — including in `sparse` mode,
    # where it must keep running full attention. Do not weaken this to `cfg.enabled` alone.
    if cfg is not None and cfg.enabled and getattr(self, "indexer", None) is not None:
        if cfg.mode == "dense_warmup":
            # Phase 1: attention stays dense (below); the indexer only produces a KL term, and it has no
            # output path, so the LM forward cannot change. Checkpoint the KL when training: recompute the
            # O(T^2) score graph in backward instead of holding it across all 36 layers. use_reentrant=False
            # is required — the inputs (frozen base) do not require grad; the grad-carrying tensors are the
            # indexer params referenced inside.
            if getattr(cfg, "kl_checkpoint", False) and self.training:
                self._dsa_kl = torch.utils.checkpoint.checkpoint(
                    _dense_warmup_kl, self, hidden_states, query_states, key_states, use_reentrant=False
                )
            else:
                self._dsa_kl = _dense_warmup_kl(self, hidden_states, query_states, key_states)
        elif cfg.mode == "sparse":
            # Phase 2: the sparse path REPLACES the dense attention below and returns early.
            attn_output, self._dsa_kl = _sparse_attn_and_kl(
                self, hidden_states, query_states, key_states, value_states
            )
            return self.o_proj(attn_output), None
        else:
            raise ValueError(f"unknown DSA mode {cfg.mode!r}")

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
