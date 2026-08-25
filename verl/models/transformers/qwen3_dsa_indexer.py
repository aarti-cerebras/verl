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
"""DSA "lightning indexer" for a **GQA** backbone (Qwen3), with no MLA anywhere.

Per layer this produces, for query token ``t`` and key token ``s``::

    I[t, s] = sum_j  w[t, j] * softmax_scale * ReLU( q_idx[t, j] . k_idx[s] )

and ``select_topk`` takes the top-k **tokens** per query from those raw scores. The score is raw: no
softmax and no L1 normalization live here. Both consumers are elsewhere — the Phase-1/2 KL softmaxes ``I``
(``qwen3_dsa.py``), and inference takes an exp-free top-k.

**Deliberately self-contained.** This module duplicates the FP8/Hadamard numerics from
``dsa_indexer.py`` instead of importing them, so that MiniCPM3-DSA and Qwen3-MSA keep running
byte-identically no matter what happens here (docs/qwen3_4b_dsa/plan_v2.md §6). The duplicated pieces are
``FP8_DTYPE``/``FP8_MAX``, ``_fake_quant_fp8``, ``_fwht``, ``_rotate_activation`` and ``_rotate_half``.
**They are exactly the numerics train/serve parity depends on — if you change one copy, change the other.**

Three ways this differs from the MLA port in ``dsa_indexer.py``, all forced by the absence of MLA:

1. **Query source.** MLA hands its indexer a free compressed query latent
   ``qr = q_a_layernorm(q_a_proj(x))``; GQA has no such latent, so ``wq`` projects straight from hidden
   states. (Kwai Keye-VL-2.0, the only published DSA-on-GQA system, does the same: *"q^I_{t,j} and
   w^I_{t,j} are derived from h_t"*, with no bottleneck rank in its shipped ``sa_config``.)
2. **A query norm.** Because MLA's ``q_a_layernorm`` used to sit upstream of the indexer query and now
   nothing does, ``q_norm`` (RMSNorm, gain 1) is added — a *relocated* norm, not a new idea. Keye has one
   at ``keye_indexer.py:156``.
3. **Its own rotary, over the whole indexer head.** MLA lends its indexer the decoupled rotary tables, so
   DeepSeek ropes only the ``qk_rope_head_dim`` slice — because MLA's *attention* is itself only partly
   position-dependent. Qwen3's attention ropes all 128 dims, so the indexer ropes all 64 of its own. The
   base model's 128-wide ``cos``/``sin`` **cannot** be sliced down to 64: ``cos[..., :64]`` holds 64
   *distinct* frequencies while ``rotate_half`` on a 64-dim vector needs 32 duplicated. A dedicated 64-dim
   rotary at the same ``theta`` recovers exactly the base's even-indexed frequencies.

Sizing follows Keye's shipped ``sa_config`` (``indexer_num_heads=16``, ``indexer_head_dim=64``,
``indexer_num_kv_heads=1``, ``topk=2048``), which also happens to be the MiniCPM3 port's default — two
independent DSA implementations landing on ``16 x 64``.

**The class name is load-bearing.** ``verl/utils/fsdp_utils.py`` keys the FSDP2 Option-B2 indexer wrap on
the class *name* (``_indexer_cls_names``). ``"Qwen3DSAIndexer"`` must appear in that tuple, or Phase-1
training silently produces a nonzero-but-fake ``grad_norm`` and a flat loss at ``world_size > 1``
(docs/dsa_fsdp_sharding_notes.md §3b/§4). The class must also expose ``.cfg.mode``, which the same line
reads.

See docs/qwen3_4b_dsa/plan_v2.md §2 (architecture), §2.4 (initialization) and §5.1 (serving constraints).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- copied from dsa_indexer.py (see module docstring) -----------------------------------------------
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0

# Parameters that must be excluded from weight decay, as suffixes of their qualified names. Decay pulls the
# norm gains toward 0, which suppresses the whole branch (strictly worse than MSA's Gemma-style
# parameterization, where decay pulls the gain toward 1 and is harmless); and decaying `weights_proj` toward
# 0 severs the ONLY gradient path into `wq`/`wk` (docs/dsa_grad_norm_debugging.md issue #2). The optimizer
# param groups have to be built from this — it is not a config value. See plan_v2.md §2.4.
NO_DECAY_SUFFIXES = ("q_norm.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight")

# Target for the across-key standard deviation of the raw scores at init, `std_s(I)`. Sets the initial
# entropy via `entropy_frac ~= 1 - sigma^2 / (2 ln N)`; 0.3 gives ~0.996 at N=32768. Anything in [0.2, 0.5]
# is fine — EQUALITY ACROSS LAYERS is the property that matters, not the value (plan_v2.md §2.4).
SIGMA_TARGET = 0.3

# `std_s(I) = SCORE_STD_COEFF * w_rms` at init with unit norm gains. Derivation: q rows have unit RMS
# (RMSNorm) so ||q_j|| = sqrt(head_dim); k has unit per-element variance (LayerNorm) so
# std(q_j . k) = sqrt(head_dim); std_s(ReLU(.)) = sqrt(head_dim) * sqrt(1/2 - 1/(2 pi)); summing over
# n_heads independent-ish heads multiplies by sqrt(n_heads) * w_rms, and `softmax_scale = head_dim**-0.5`
# cancels the sqrt(head_dim). So the coefficient is `sqrt(n_heads) * sqrt(1/2 - 1/(2 pi))` -- head_dim drops
# out entirely. Verified numerically in tests/dsa/test_qwen3_dsa_indexer.py.
_RELU_STD = (0.5 - 1.0 / (2.0 * torch.pi)) ** 0.5  # 0.58383...


@dataclass
class Qwen3DSAConfig:
    """Configuration for the Qwen3 GQA DSA indexer.

    Geometry (``hidden_size``, ``num_heads``, ``num_kv_heads``, ``rope_theta``) is FORCED from the live
    Qwen3 config by ``qwen3_dsa.build_dsa_config`` — an override that disagreed with the backbone would
    produce a checkpoint that cannot be served. Everything else comes from training-side ``dsa_*``
    overrides on the HF config.
    """

    enabled: bool = False

    # --- geometry, forced from the base model --------------------------------------------------------
    hidden_size: int = 2560
    num_heads: int = 32  # H_q (base attention) -- the teacher averages over all of these
    num_kv_heads: int = 8  # H_kv (base attention); NOT the indexer's head count
    rope_theta: float = 5e6

    # --- indexer geometry ---------------------------------------------------------------------------
    n_heads: int = 16  # indexer query heads; Keye `indexer_num_heads`, also the MiniCPM3 default
    head_dim: int = 64  # d_idx; Keye `indexer_head_dim`
    rope_head_dim: int = 64  # rope width; == head_dim (all dims roped), see module docstring point 3.
    # 32 (DeepSeek's *fraction*) is the documented ablation and needs only this value changed.

    # --- selection -----------------------------------------------------------------------------------
    top_k: int = 2048  # selected TOKENS per query

    # --- which layers get an indexer at all ---------------------------------------------------------
    # Layers [0, dense_prefix) stay DENSE: no indexer, no KL term, stock attention forward. The default
    # is 0 (every layer sparse) and must STAY 0 -- DeepSeek-V3.2 sparsifies all layers, and the two
    # in-flight Phase-1 checkpoints hold 36 indexers, so a nonzero default would make `resume_mode=auto`
    # load them into a model with fewer. 4 is the value the launch scripts pass, motivated by the
    # PER-LAYER topk_recall of the lr1e-4 run at step ~4150: L00-L03 = 0.79/0.79/0.82/0.88 while every
    # layer from L07 up is >= 0.92 and the top third reaches 0.95-0.97. That is the same shape MSA/M3
    # found (msa_indexer.dense_prefix=3, M3's shipped `sparse_attention_freq = [0]*3 + [1]*57`).
    #
    # This is an ARCHITECTURE field, not a training preference: it must be identical at train and serve
    # time or the served model runs dense where it was trained sparse. It is therefore written into the
    # serving dir's config.json (build_qwen3_dsa_serving_dir.py) and read back by the vLLM plugin.
    dense_prefix: int = 0
    sparse_layers: Optional[str] = None  # explicit comma-separated layer ids that get an indexer, e.g.
    # "4,5,6,...". Overrides `dense_prefix` entirely when set (an arbitrary set, not just a prefix).

    # --- FP8 numerics (must match serve time exactly) -----------------------------------------------
    fp8: bool = True  # run the score matmul in fake-quantized E4M3, as the reference does
    fp8_ue8m0: bool = True  # power-of-2 (UE8M0) per-row scale, matching the serving kernels. Default True
    # here (unlike DSAConfig, which defaults False for backward compat): leaving it off cost the MiniCPM3
    # run ~2% train/serve selection drift. See docs/dsa_eval_report.md §5.
    block_size: int = 128  # FP8 quant block. `head_dim <= block_size` => exactly one scale per row, which
    # is what makes the serve-time zero-pad from 64 to 128 lossless (the row amax is unchanged).
    rotate_activation: bool = True  # Hadamard pre-quant rotation (orthonormal, so it preserves the dot)

    # --- loss / training -----------------------------------------------------------------------------
    mode: str = "dense_warmup"  # "dense_warmup" (Phase 1) | "sparse" (Phase 2). Read by fsdp_utils.
    sigma_target: float = SIGMA_TARGET  # init score scale (see §2.4)
    kl_block_size: int = 512  # query tile for the teacher/KL. Phase 1: 512 (a retained fp32 [1,512,32768]
    # teacher is 67 MB). Phase 2: drop to 256 — the gathered K/V are [1,8,T_q,2048,128] bf16, 2.15 GB EACH.
    kl_reduction: str = "mean"  # per-layer KL -> loss reduction over layers: "mean" (default) | "sum".
    # Per-layer indexer params are disjoint, so this is a pure gradient SCALE (sum == mean * n_layers), same
    # optimum. "mean" keeps grad_norm in a range where it is still a usable diagnostic rather than pinned
    # against clip_grad=1.0 — the DSA Phase-1 runs sat at grad_norm ~330. See plan_v2.md §3.
    kl_checkpoint: bool = False  # recompute the per-tile score graph in backward instead of retaining it
    # across all layers. REQUIRED at 32K. No effect on numerics.
    compile_teacher: bool = True  # torch.compile the head-averaged teacher. It is the largest single cost
    # and is bandwidth-bound; MSA measured 2.70x on the equivalent function, with LOWER error against an
    # fp64 reference than eager. Set False to bisect an Inductor problem.
    diag_interval: int = 10  # compute monitoring diagnostics every N forwards (they cost extra top-ks)
    log_per_layer: bool = False  # also emit per-layer scalar keys — the Phase-1 gate is per layer, so this
    # is what you read it off
    diag_overlap_sample: int = 0  # cap query rows used for the O(k^2)-ish overlap diagnostic; 0 = all rows
    full_support_kl_prob: float = 0.0  # Phase-2 escape hatch: with this probability per forward, compute
    # the KL over the FULL causal support instead of the selected set, while the LM path still runs sparse.
    # The restricted KL gives the indexer no gradient about tokens it failed to select — a token ranked
    # 2049th never enters the loss — so the ranking outside the top-k can decalibrate. Costs one dense
    # attention pass (8.80 TFLOP/layer at 32K) when it fires. See plan_v2.md §4 and review item 7.
    warmstart_path: Optional[str] = None  # consolidated (world-size-agnostic) state dict to warm-start
    # from; loaded before the FSDP wrap, so it is GPU-count-agnostic and yields a fresh optimizer at step 0.

    # --- guards --------------------------------------------------------------------------------------
    serving_compat: bool = True  # enforce the dims the serving kernels require, at CONSTRUCTION time, so a
    # config that cannot be served fails immediately rather than after a multi-day run. Unit tests that
    # exercise the algebra on toy shapes set this False.

    def __post_init__(self):
        if self.mode not in ("dense_warmup", "sparse"):
            raise ValueError(f"mode must be 'dense_warmup' or 'sparse', got {self.mode!r}")
        if self.kl_reduction not in ("sum", "mean"):
            raise ValueError(f"kl_reduction must be 'sum' or 'mean', got {self.kl_reduction!r}")
        if self.dense_prefix < 0:
            raise ValueError(f"dense_prefix ({self.dense_prefix}) must be >= 0")
        if self.sparse_layers and self.dense_prefix:
            # Both set is always a mistake: sparse_layers silently wins and the dense_prefix in the run
            # name / config.json then describes a model that was never built.
            raise ValueError(
                f"set either dense_prefix ({self.dense_prefix}) or sparse_layers "
                f"({self.sparse_layers!r}), not both — sparse_layers overrides dense_prefix"
            )
        if self.rope_head_dim > self.head_dim:
            raise ValueError(f"rope_head_dim ({self.rope_head_dim}) must be <= head_dim ({self.head_dim})")
        if self.rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim ({self.rope_head_dim}) must be even for RoPE")
        if self.head_dim > self.block_size:
            # One FP8 scale per row is assumed throughout, and is what makes the serve-time pad lossless.
            raise ValueError(f"head_dim ({self.head_dim}) must be <= block_size ({self.block_size})")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        if self.head_dim & (self.head_dim - 1):
            # The Hadamard transform needs a power-of-2 width; both codebases assert this.
            raise ValueError(f"head_dim ({self.head_dim}) must be a power of 2 for the Hadamard rotation")
        if self.serving_compat:
            # Not preferences. KeyeIndexer asserts `head_dim in (32, 64, 128)`; its prefill path computes
            # `block_q = 128 // num_heads`, so the head count must divide 128; and stock DeepGEMM's
            # mqa_logits accepts head counts in {32, 64, 128} at head_dim 128, which is why 16x64 is
            # zero-padded to 32x128 at serve time (plan_v2.md §2.3, §5.1).
            if self.head_dim not in (32, 64, 128):
                raise ValueError(f"head_dim must be 32, 64 or 128 to serve, got {self.head_dim}")
            if 128 % self.n_heads != 0:
                raise ValueError(f"n_heads must divide 128 to serve, got {self.n_heads}")
            if not self.fp8_ue8m0:
                raise ValueError(
                    "fp8_ue8m0 must be True: the serving kernels quantize with a UE8M0 (power-of-2) scale, "
                    "and training against a continuous scale reintroduces train/serve selection drift"
                )

    @property
    def group_size(self) -> int:
        """G = H_q / H_kv — query heads sharing one KV head in the base attention."""
        return self.num_heads // self.num_kv_heads

    def layer_is_sparse(self, layer_idx: int) -> bool:
        """Whether this layer gets an indexer (and therefore a KL term). Same contract as MSA's."""
        if self.sparse_layers:
            ids = {int(s) for s in str(self.sparse_layers).replace(" ", "").split(",") if s != ""}
            return layer_idx in ids
        return layer_idx >= self.dense_prefix

    def sparse_layer_ids(self, n_layers: int) -> list[int]:
        """The resolved sparse-layer set. The single source of truth shared by training, the serving-dir
        builder and the vLLM plugin — a second reimplementation of this predicate is exactly how a model
        gets served dense where it was trained sparse."""
        return [i for i in range(n_layers) if self.layer_is_sparse(i)]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims (llama / Qwen3 "neox" convention: ``[-x2, x1]``)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized fast Walsh-Hadamard transform over the last dim (must be a power of 2)."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"Hadamard transform requires a power-of-2 last dim, got {n}"
    y = x.clone()
    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.cat([a + b, a - b], dim=-1).reshape(*x.shape[:-1], n)
        h *= 2
    return y


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Hadamard transform over the last dim scaled by ``n**-0.5`` (DeepSeek-V3.2 ``rotate_activation``).

    Orthonormal, so it leaves ``q.k`` unchanged when applied to both; its purpose is to spread magnitude
    across dims so FP8 per-row quantization has fewer outliers. Uses the ``fast_hadamard_transform`` CUDA
    kernel when available (as both serving stacks do), else a pure-torch FWHT — any orthonormal Hadamard
    ordering preserves the dot product, so the fallback is equivalent for scores.
    """
    scale = x.shape[-1] ** -0.5
    if x.is_cuda:
        try:
            from fast_hadamard_transform import hadamard_transform

            in_dtype = x.dtype
            return hadamard_transform(x.to(torch.bfloat16), scale=scale).to(in_dtype)
        except ImportError:
            pass
    return _fwht(x) * scale


def _fake_quant_fp8(x: torch.Tensor, use_ue8m0: bool = True) -> torch.Tensor:
    """Row-wise FP8 (E4M3) fake-quantization with a straight-through estimator (QAT).

    Forward returns the dequantized round-trip ``round_fp8(x / scale) * scale`` with
    ``scale = amax / FP8_MAX`` over the last dim, so the indexer trains against the numerics it will serve
    with. Backward is the identity via the STE (``x + (x_q - x).detach()``): a bare ``.to(float8)`` cast is
    non-differentiable, so without the STE *no* gradient reaches ``wq``/``wk``.

    ``use_ue8m0`` rounds the scale up to a power of two and clamps before the cast, matching the serving
    kernels' UE8M0 quantization exactly. ``head_dim <= block_size`` (enforced by the config) means one
    block per row, hence one fp32 scale per row — which is also why zero-padding a 64-dim row out to 128 at
    serve time cannot change the scale.

    See Jacob et al. 2018 (arXiv:1712.05877) for the fake-quant + STE pattern.
    """
    with torch.no_grad():  # x_q is a value-only reference point; no graph is kept for it
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = amax / FP8_MAX
        if use_ue8m0:
            scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))  # UE8M0: power-of-2 scale (match serve)
            x_q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float() * scale
        else:
            x_q = (x / scale).to(FP8_DTYPE).float() * scale
    return x + (x_q - x).detach()  # STE: forward == x_q, backward == identity into x


class _IndexerRMSNorm(nn.Module):
    """Standard RMSNorm, ``x * rsqrt(mean(x^2) + eps) * weight``, gain initialised to 1.

    Matches Keye's indexer query norm (SGLang ``RMSNorm(head_dim, eps=1e-6)``). Deliberately *not* the
    Gemma-style ``(1 + w)`` form used by ``msa_indexer.py`` — that parameterization exists because the
    MiniMax serving kernel hardcodes it; the DSA serving path does not.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(in_dtype)


class _IndexerRotary(nn.Module):
    """The indexer's OWN rotary embedding, ``rope_head_dim`` wide, at the base model's ``theta``.

    Not a slice of the base model's tables: with ``rotate_half``, a ``d``-wide rope needs ``d/2``
    frequencies duplicated, whereas ``base_cos[..., :d]`` holds ``d`` *distinct* frequencies. Building it
    fresh at the same ``theta`` yields exactly the base rope's even-indexed frequencies, so it spans the
    same spectrum as the attention it distills.

    Frequencies are computed in fp64 and cached in fp32; ``forward`` returns fp32 ``cos``/``sin`` and the
    caller casts. If the base model ever gains rope scaling (YaRN etc.), it must be applied here too.
    """

    def __init__(self, dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``position_ids`` ``[b, s]`` (int) -> ``(cos, sin)`` each ``[b, s, rope_head_dim]`` fp32."""
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.to(position_ids.device)  # [b, s, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [b, s, d]
        return emb.cos(), emb.sin()


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the leading ``rope_head_dim`` slice of ``t``, in fp32.

    Args:
        t: ``[b, s, h, r]`` (the rope slice only; ``h`` may be 1 for the MQA key).
        cos, sin: ``[b, s, r]``, broadcast over the head axis.
    """
    orig_dtype = t.dtype
    cos = cos.float().unsqueeze(2)  # [b, s, 1, r]
    sin = sin.float().unsqueeze(2)
    t = t.float()
    return (t * cos + _rotate_half(t) * sin).to(orig_dtype)


class Qwen3DSAIndexer(nn.Module):
    """DSA lightning indexer for one Qwen3 decoder layer.

    Parameters (per layer, at ``hidden=2560``, ``16 x 64``)::

        wq            2560 x 1024 = 2.62M      query, direct from hidden states
        wk            2560 x   64 = 0.16M      SINGLE shared (MQA) key head
        weights_proj  2560 x   16 = 0.04M      per-head gate, kept in fp32
        q_norm/k_norm         64 x 3           norms
                                    ~2.83M/layer  ->  102M over 36 layers (2.5% of 4.02B)

    ``hidden_rms`` is this layer's ``rms(input_layernorm.weight)``, and is required: it is the only thing
    that makes the initial score scale equal across layers. Measured on Qwen3-4B-Thinking it spans 186x
    (0.025 at layer 0 to 4.709 at layer 34), so a constant ``weights_proj`` init leaves the first layers
    with scores of essentially zero — and therefore ~100x less gradient into ``wq``/``wk``, which receive
    gradient only through the gate — while the last layers start already committed
    (``entropy_frac`` 0.91). See plan_v2.md §2.4.
    """

    def __init__(self, cfg: Qwen3DSAConfig, hidden_rms: float = 1.0):
        super().__init__()
        self.cfg = cfg  # `.cfg.mode` is read by fsdp_utils' Option-B2 wrap -- keep the attribute name
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = cfg.head_dim**-0.5

        self.wq = nn.Linear(cfg.hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.q_norm = _IndexerRMSNorm(cfg.head_dim)
        self.wk = nn.Linear(cfg.hidden_size, cfg.head_dim, bias=False)  # single MQA key head
        self.k_norm = nn.LayerNorm(cfg.head_dim, eps=1e-6)
        # Kept in fp32: the reference does, and it is the only unnormalized path in the branch. Calling it
        # via F.linear with an explicit .float() means a blanket module.bfloat16() cannot silently drop it.
        self.weights_proj = nn.Linear(cfg.hidden_size, cfg.n_heads, bias=False).float()

        self.rotary = _IndexerRotary(cfg.rope_head_dim, cfg.rope_theta)
        self.reset_parameters(hidden_rms)

    # ------------------------------------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------------------------------------

    def weights_proj_std(self, hidden_rms: float) -> float:
        """The ``weights_proj`` init std that makes ``std_s(I) == cfg.sigma_target`` for this layer.

        From ``std_s(I) = sqrt(n_heads) * _RELU_STD * w_rms`` and
        ``w_rms = std_W * sqrt(hidden_size) * hidden_rms * n_heads**-0.5``, so the ``n_heads`` factors
        cancel and ``head_dim`` drops out (the ``softmax_scale`` cancels the score's ``sqrt(head_dim)``)::

            std_W = sigma_target / (_RELU_STD * sqrt(hidden_size) * hidden_rms)

        At ``sigma_target=0.3``, ``hidden=2560``: ``0.0102 / hidden_rms`` — i.e. the inherited constant
        ``0.5/sqrt(hidden) = 0.00988`` divided by this layer's gain RMS, which is exactly why the inherited
        init looked right at mid-depth (``hidden_rms ~ 1``) and drifted either side of it.
        """
        cfg = self.cfg
        denom = _RELU_STD * cfg.hidden_size**0.5 * max(float(hidden_rms), 1e-6)
        return cfg.sigma_target / denom

    def reset_parameters(self, hidden_rms: float = 1.0) -> None:
        """Explicit, reproducible init. Do NOT rely on framework defaults: only rank-0's init survives the
        FSDP2 broadcast, so it must be deterministic and seedable.

        ``wq``/``wk`` are followed by a norm, so their std does **not** affect the forward score scale at
        all (RMSNorm and LayerNorm are invariant to row scaling) — it sets only the gradient geometry, i.e.
        the effective LR on those matrices. The entire initial score scale, and hence the initial entropy,
        is controlled by ``weights_proj`` alone. This is why the inherited ``dsa_indexer.py`` rationale
        ("each linear starts at half-unit variance so scores are small") does not carry over: it describes
        a network without a normalized query.

        ``weights_proj`` is small but must stay NONZERO — zeroing it severs the gradient to ``wq``/``wk``,
        which receive gradient only through the gate (docs/dsa_grad_norm_debugging.md issue #2).

        Watch ``indexer/entropy_frac`` on the first forward: it should be in ``[0.99, 1.0]`` for **every**
        layer. That is a unit test and a smoke-run gate, not a hope.
        """
        nn.init.normal_(self.wq.weight, std=0.5 * self.cfg.hidden_size**-0.5)  # grad geometry only
        nn.init.normal_(self.wk.weight, std=0.5 * self.cfg.hidden_size**-0.5)  # grad geometry only
        nn.init.ones_(self.q_norm.weight)
        nn.init.ones_(self.k_norm.weight)
        nn.init.zeros_(self.k_norm.bias)
        nn.init.normal_(self.weights_proj.weight, std=self.weights_proj_std(hidden_rms))

    # ------------------------------------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------------------------------------

    def project(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q_idx [b, s, n_heads, d], k_idx [b, s, d], weights [b, s, n_heads] fp32)``.

        Order is project -> norm -> RoPE, matching the fused serving kernel (Keye's
        ``_get_q_k_w_bf16``); the Hadamard rotation and FP8 quantization happen in ``scores`` so that a
        checkpointed KL can recompute the score matrix without re-projecting.

        ``hidden_states`` is **detached** here, which is DSA's ``stopgrad`` (arXiv 2512.02556 §2.1.1: *"we
        detach the indexer input from the computational graph for separate optimization"*). Doing it inside
        the module rather than at the call site makes it impossible to forget in Phase 2, where it is what
        keeps the KL out of the backbone. In Phase 1 the base is frozen, so it is a no-op.
        """
        x = hidden_states.detach()
        b, s, _ = x.shape
        r = self.rope_head_dim

        cos, sin = self.rotary(position_ids)  # [b, s, r] fp32

        q = self.q_norm(self.wq(x).view(b, s, self.n_heads, self.head_dim))
        q = torch.cat([_apply_rope(q[..., :r], cos, sin), q[..., r:]], dim=-1) if r < self.head_dim \
            else _apply_rope(q, cos, sin)

        k = self.k_norm(self.wk(x)).unsqueeze(2)  # [b, s, 1, d] for the rope broadcast
        k = torch.cat([_apply_rope(k[..., :r], cos, sin), k[..., r:]], dim=-1) if r < self.head_dim \
            else _apply_rope(k, cos, sin)
        k = k.squeeze(2)  # [b, s, d]

        # fp32 regardless of the module's stored dtype (the reference keeps this projection in float32).
        # `n_heads**-0.5` is folded in here, as in DeepSeek and our MiniCPM3 port. Keye omits it; since it
        # is a uniform positive scalar it cannot change the top-k, only the KL temperature.
        weights = F.linear(x.float(), self.weights_proj.weight.float()) * (self.n_heads**-0.5)
        return q, k, weights

    def scores(
        self,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
        weights: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw scores ``I[b, s_q, s_k] = sum_h (w_h * softmax_scale) * ReLU(<q_h, k>)``.

        The single key head broadcasts across all indexer query heads (MQA). With ``cfg.fp8``, q/k are
        Hadamard-rotated and fake-quantized to E4M3 and the ReLU dot is taken in the *dequantized* domain —
        the same numerics the serving kernel computes (the per-row positive scales factor out of the ReLU),
        but differentiable so ``wq``/``wk`` train. The bf16 branch is the algebraically identical
        full-precision reference. No softmax and no L1 norm here.

        ``q_idx`` may be a query tile; ``k_idx`` is always the full key sequence. Peak transient is the
        ``[b, s_q, n_heads, s_k]`` ``dots`` tensor (537 MB at 32K with a 512-query tile and 16 heads), which
        is why the caller tiles over queries and checkpoints this call.
        """
        if self.cfg.fp8:
            if self.cfg.rotate_activation:
                q_idx = _rotate_activation(q_idx)
                k_idx = _rotate_activation(k_idx)
            q_idx = _fake_quant_fp8(q_idx, self.cfg.fp8_ue8m0)
            k_idx = _fake_quant_fp8(k_idx, self.cfg.fp8_ue8m0)
        dots = torch.relu(torch.einsum("bqhd,bkd->bqhk", q_idx, k_idx))
        eff_w = (weights * self.softmax_scale).to(dots.dtype)
        out = torch.einsum("bqhk,bqh->bqk", dots, eff_w)
        if attn_bias is not None:
            out = out + attn_bias
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        return_projection: bool = False,
    ):
        """Default: raw scores ``I``. With ``return_projection``, return ``(q_idx, k_idx, weights)``.

        The projection carries ALL indexer parameters; ``scores`` is parameter-free. Reaching the
        projection through ``__call__`` (rather than calling ``.project()`` directly) is what lets FSDP2's
        forward hooks fire when this module is wrapped as its own ``fully_shard`` unit: the pre-forward
        all-gather, and the pre-backward gate whose backward triggers the gradient reduce-scatter onto the
        sharded optimizer master. Bypass it and Phase 1 trains nothing while still reporting a plausible
        ``grad_norm`` (docs/dsa_fsdp_sharding_notes.md, Option B2).
        """
        q_idx, k_idx, weights = self.project(hidden_states, position_ids)
        if return_projection:
            return q_idx, k_idx, weights
        return self.scores(q_idx, k_idx, weights, attn_bias)

    def select_topk(
        self, scores: torch.Tensor, top_k: Optional[int] = None, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Top-k key indices per query from the raw scores: ``[b, s_q, k]``, clamped to ``s_k``.

        NOTE for serving parity: ``torch.topk``'s tie-breaking is not guaranteed deterministic, and with
        2048 of 32768 selected there are near-ties in every row. Keye replaced it with ``flashinfer.topk``
        for exactly this reason (*"to avoid mismatch between training and inference Top-k results"*), so the
        parity ladder must assert that selection is bit-identical across repeated runs and between train
        and serve. Third instance of this failure class after the FP8 scale format and the ``index_topk``
        config gate — each of which presented as a fluent, fully-dense-looking model.
        """
        if attn_bias is not None:
            scores = scores + attn_bias
        k = min(top_k if top_k is not None else self.cfg.top_k, scores.shape[-1])
        return scores.topk(k, dim=-1).indices
