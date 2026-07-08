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
"""DeepSeek Sparse Attention (DSA) "lightning indexer" module.

Faithful port of the DeepSeek-V3.2 reference indexer (`inference/model.py::Indexer`), grafted onto an MLA
backbone (e.g. MiniCPM3-4B). Per layer it produces

    I[t, s] = sum_j  w[t, j] * ReLU( q_idx[t, j] . k_idx[s] )

over query token ``t`` and key token ``s``; ``j`` indexes the indexer query heads. The indexer query comes
from the MLA *compressed query latent* ``qr = q_a_layernorm(q_a_proj(x))`` (a single ``wq_b``, no ``wq_a``);
the key and per-head weights come from hidden states ``x``. The key is single-head (MQA) and shared across
all query heads. The score path runs in FP8 exactly as the reference (blockwise ``act_quant`` -> per-head
``ReLU`` dot products -> fp32 ``weights`` carrying ``q_scale * softmax_scale`` -> sum over heads).

What this module does NOT do (by design, matching the reference): no normalization of ``I``. The score is
raw. Both downstream uses live elsewhere:
  * Training (Phase 1/2): the loss softmaxes ``I`` and matches it to the main-attention target
    ``p`` (main attention summed over heads, **L1-normalized**). The L1-norm and Softmax are in the loss,
    NOT here.
  * Inference: ``select_topk`` takes the top-k keys per query from the raw scores.

RoPE matches MiniCPM3 (plain ``rotate_half`` computed in fp32). At integration the patched MLA forward
passes the base model's own ``cos``/``sin`` (the same rotary tables, LongRoPE scaling included) so the
indexer's RoPE is identical to the attention it distills.

Step status: the FP8 numerics here are a pure-torch reference of DeepSeek's ``act_quant``/``fp8_index``
(swap in the fused Triton/`_scaled_mm` kernels for speed later). The pre-quant ``rotate_activation``
(Hadamard transform, V3.2) is implemented faithfully. See ``docs/dsa_indexer_module_plan.md`` /
``docs/dsa_minicpm3_plan.md``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# E4M3 constants, shared with verl's FP8 kernels (verl/utils/kernel/fp8_kernel.py).
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0


@dataclass
class DSAConfig:
    """Configuration for the DSA lightning indexer.

    Defaults are the recommended MiniCPM3-4B sizing (see docs/dsa_minicpm3_plan.md). ``rope_head_dim`` MUST
    equal the base model's ``qk_rope_head_dim`` and ``q_lora_rank`` MUST equal the base ``q_lora_rank`` so
    the indexer reuses the MLA query latent and the base RoPE.
    """

    enabled: bool = False
    n_heads: int = 16  # indexer query heads (sweep {8, 16, 24, 32}); reference DeepSeek-V3.2 uses 64
    head_dim: int = 64  # per-head dim = rope_head_dim + nope; DeepSeek-V3.2 uses 128
    rope_head_dim: int = 32  # must == base qk_rope_head_dim (MiniCPM3: 32)
    q_lora_rank: int = 768  # must == base q_lora_rank (indexer query input width)
    hidden_size: int = 2560
    top_k: int = 2048  # attention top-k (used by select_topk / the DSA attention layer)
    mode: str = "dense_warmup"  # "dense_warmup" (Phase 1) | "sparse" (Phase 2)
    kl_block_size: int = 1024  # query tiling block for the target/KL recompute (used later)
    kl_reduction: str = "mean"  # per-layer KL -> loss reduction over layers: "mean" (default; interpretable
    # per-layer scale, loss ~= per-layer KL) | "sum" (reference; each layer weighted equally, loss ~=
    # n_layers * per-layer KL). Same optimum; "sum" scales the gradient by ~n_layers vs "mean" at a given LR.
    fp8: bool = True  # run the indexer score matmul in FP8 (E4M3), as in the reference
    block_size: int = 128  # FP8 act_quant block size (reference default)
    rotate_activation: bool = True  # Hadamard pre-quant rotation (V3.2), only in the FP8 path
    diag_interval: int = 10  # compute monitoring diagnostics (topk recall/overlap/score) every N forwards
    log_per_layer: bool = False  # also emit per-layer kl_by_layer/L## (+ entropy/{indexer,attn}_frac_by_layer/L##
    # on diag forwards) — a debug breakdown (~3*n_layers extra scalar keys); the mean/min/max always log

    def __post_init__(self):
        if self.kl_reduction not in ("sum", "mean"):
            raise ValueError(f"kl_reduction must be 'sum' or 'mean', got {self.kl_reduction!r}")
        if self.rope_head_dim > self.head_dim:
            raise ValueError(f"rope_head_dim ({self.rope_head_dim}) must be <= head_dim ({self.head_dim})")
        if self.rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim ({self.rope_head_dim}) must be even for RoPE")
        if self.head_dim > self.block_size:
            # single-block-per-head quantization assumes head_dim <= block_size (true for MiniCPM3=64,
            # DeepSeek-V3.2=128). Larger head dims would need multi-block scaling.
            raise ValueError(f"head_dim ({self.head_dim}) must be <= block_size ({self.block_size})")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (MiniCPM3 / llama convention: [-x2, x1])."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the rope-slice of ``t``, matching MiniCPM3's ``apply_rotary_pos_emb`` (fp32 compute).

    Args:
        t: ``[b, s, h, rope_head_dim]`` (the rope slice only; ``h`` may be 1 for the MQA key).
        cos, sin: ``[b, s, rope_head_dim]`` (the base model's rotary tables, already gathered by
            position_ids; LongRoPE scaling already applied). Broadcast over the head dim.
    """
    orig_dtype = t.dtype
    cos = cos.float().unsqueeze(2)  # [b, s, 1, rope_head_dim]
    sin = sin.float().unsqueeze(2)
    t = t.float()
    out = t * cos + _rotate_half(t) * sin
    return out.to(orig_dtype)


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
    """Hadamard transform over the last dim, scaled by ``n**-0.5`` (DeepSeek-V3.2 ``rotate_activation``).

    Orthonormal, so it preserves dot products (q.k is unchanged when applied to both); its purpose is to
    spread magnitude across dims for better FP8 per-block quantization. Uses the ``fast_hadamard_transform``
    CUDA kernel when available, else a pure-torch FWHT (any orthonormal Hadamard ordering preserves the
    dot, so the fallback is equivalent for scores).
    """
    scale = x.shape[-1] ** -0.5
    # The fused kernel is CUDA-only; fall back to the pure-torch FWHT on CPU or when the lib is absent
    # (any orthonormal Hadamard ordering preserves the dot, so the fallback is equivalent for scores).
    if x.is_cuda:
        try:
            from fast_hadamard_transform import hadamard_transform

            in_dtype = x.dtype
            return hadamard_transform(x.to(torch.bfloat16), scale=scale).to(in_dtype)
        except ImportError:
            pass
    return _fwht(x) * scale


def _fake_quant_fp8(x: torch.Tensor) -> torch.Tensor:
    """Blockwise FP8 (E4M3) fake-quantization with a straight-through estimator (QAT).

    Forward returns the dequantized E4M3 round-trip ``round_fp8(x / scale) * scale`` (``scale = amax /
    FP8_MAX`` over the last dim), so the indexer sees exactly the fp8 numerics it will use at inference.
    Backward is the identity via the straight-through estimator (``x + (x_q - x).detach()``): a bare
    ``.to(float8)`` cast is non-differentiable (its rounding derivative is 0 a.e. and float8 has no autograd
    support), so without the STE *no* gradient reaches ``wq_b``/``wk``. ``head_dim <= block_size`` (enforced
    by DSAConfig) => a single block per head/token, i.e. one fp32 scale per row.

    See Jacob et al. 2018 (arXiv:1712.05877) and PyTorch ``FakeQuantize`` for the fake-quant + STE pattern.
    """
    with torch.no_grad():  # x_q is a value-only reference point; no graph is kept for it
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = amax / FP8_MAX
        x_q = (x / scale).to(FP8_DTYPE).float() * scale
    return x + (x_q - x).detach()  # STE: forward == x_q, backward == identity into x


class LightningIndexer(nn.Module):
    """DSA lightning indexer for one decoder layer (DeepSeek-V3.2 faithful).

    forward inputs:
        x:   hidden states ``[b, s, hidden_size]`` — source of the key and per-head weights.
        qr:  MLA compressed query latent ``[b, s, q_lora_rank]`` = ``q_a_layernorm(q_a_proj(x))`` — source
             of the indexer query (no ``wq_a``).
        cos, sin: base-model RoPE tables ``[b, s, rope_head_dim]`` (use MiniCPM3's own ``rotary_emb``).
        attn_bias: optional additive mask ``[b, s_q, s_k]`` (causal + per-document ``-inf``).

    Returns raw scores ``I[b, s_q, s_k]`` (no softmax / no L1-norm — those are in the loss). Use
    ``select_topk`` for the inference selection path.
    """

    def __init__(self, cfg: DSAConfig, softmax_scale: Optional[float] = None):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.softmax_scale = softmax_scale if softmax_scale is not None else cfg.head_dim**-0.5

        # query from the compressed latent qr -> n_heads * head_dim (single proj; no wq_a)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.head_dim, bias=False)
        # single (MQA) key head from hidden states
        self.wk = nn.Linear(cfg.hidden_size, cfg.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(cfg.head_dim)
        # per-head weights from hidden states, kept in fp32 (reference: Linear(dim, n_heads, dtype=float32))
        self.weights_proj = nn.Linear(cfg.hidden_size, cfg.n_heads, bias=False).float()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Explicit, reproducible init for the indexer (do NOT rely on framework defaults — only rank-0's
        init survives the FSDP2 broadcast, so it must be deterministic and seedable).

        Per-fan-in width-scaled normal, ``std = 0.5 / sqrt(fan_in)`` (the DeepSeek-family ``0.5/sqrt(d)``
        convention applied per projection): each linear's output starts at ~half-unit variance, so the raw
        scores ``I`` are small and ``softmax(I)`` starts near-uniform — the right high-entropy start for the
        KL distillation (unbiased, gentle gradients, maximal softmax responsiveness). ``k_norm`` at identity
        (the key is renormalized anyway). ``weights_proj`` is kept SMALL-BUT-NONZERO: zeroing it severs the
        gradient to ``wq_b``/``wk`` (they only receive grad through ``w`` — see docs/dsa_grad_norm_debugging.md
        issue #2). Watch ``entropy/indexer_frac`` (~1.0 = near-uniform) to confirm the start is healthy.
        """
        nn.init.normal_(self.wq_b.weight, std=0.5 * self.cfg.q_lora_rank**-0.5)
        nn.init.normal_(self.wk.weight, std=0.5 * self.cfg.hidden_size**-0.5)
        nn.init.normal_(self.weights_proj.weight, std=0.5 * self.cfg.hidden_size**-0.5)
        nn.init.ones_(self.k_norm.weight)
        nn.init.zeros_(self.k_norm.bias)

    def project(
        self, x: torch.Tensor, qr: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute (q_idx, k_idx, weights) with RoPE on the rope-slice of q and k (pe-first layout,
        matching the reference indexer ``split([rope_head_dim, head_dim - rope_head_dim])``).

        Returns:
            q_idx: ``[b, s, n_heads, head_dim]``
            k_idx: ``[b, s, head_dim]`` (single MQA head)
            weights: ``[b, s, n_heads]`` (fp32, scaled by n_heads**-0.5)
        """
        b, s, _ = x.shape
        r = self.rope_head_dim

        q = self.wq_b(qr).view(b, s, self.n_heads, self.head_dim)
        q = torch.cat([_apply_rope(q[..., :r], cos, sin), q[..., r:]], dim=-1)

        k = self.k_norm(self.wk(x))  # [b, s, head_dim]
        k_he = k.unsqueeze(2)  # [b, s, 1, head_dim] for rope broadcast
        k = torch.cat([_apply_rope(k_he[..., :r], cos, sin), k_he[..., r:]], dim=-1).squeeze(2)

        # weights_proj runs in fp32 regardless of the module's stored dtype (reference keeps it float32 and
        # calls it on x.float()); upcast explicitly so a blanket .bfloat16() cast does not drop it to bf16.
        weights = F.linear(x.float(), self.weights_proj.weight.float()) * (self.n_heads**-0.5)
        return q, k, weights

    def scores(
        self,
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
        weights: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw indexer scores ``I[b, s_q, s_k] = sum_h (w_h * softmax_scale) * ReLU(<q_h, k>)``.

        Single key broadcast across all query heads (MQA). When ``cfg.fp8`` is set, q/k are fake-quantized
        to E4M3 (blockwise, straight-through) and the ReLU dot product is taken in the *dequantized* domain
        — the same fp8 numerics DeepSeek's ``fp8_index`` computes (the per-row positive scales ``q_scale``/
        ``k_scale`` factor out of the ReLU), but differentiable so ``wq_b``/``wk`` train. The bf16 path is
        the algebraically-identical full-precision reference. No softmax / L1-norm here.
        """
        if self.cfg.fp8:
            # V3.2 applies a Hadamard rotation to q/k before quantization (orthonormal -> preserves the
            # dot product; spreads magnitude so FP8 per-block scaling has fewer outliers).
            if self.cfg.rotate_activation:
                q_idx = _rotate_activation(q_idx)
                k_idx = _rotate_activation(k_idx)
            q_dq = _fake_quant_fp8(q_idx)  # dequantized E4M3 (with STE): [b, s_q, n_heads, head_dim]
            k_dq = _fake_quant_fp8(k_idx)  # dequantized E4M3 (with STE): [b, s_k, head_dim]
            dots = torch.relu(torch.einsum("bqhd,bkd->bqhk", q_dq, k_dq))
            eff_w = (weights * self.softmax_scale).to(dots.dtype)
            scores = torch.einsum("bqhk,bqh->bqk", dots, eff_w)
        else:
            dots = torch.relu(torch.einsum("bqhd,bkd->bqhk", q_idx, k_idx))
            eff_w = (weights * self.softmax_scale).to(dots.dtype)
            scores = torch.einsum("bqhk,bqh->bqk", dots, eff_w)

        if attn_bias is not None:
            scores = scores + attn_bias
        return scores

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        return_projection: bool = False,
    ):
        """Default: raw scores ``I`` (project -> scores). When ``return_projection`` is set, return the
        ``(q_idx, k_idx, weights)`` projection tuple instead.

        The projection carries ALL indexer parameters (``wq_b``/``wk``/``k_norm``/``weights_proj``); ``scores``
        is param-free. Reaching the projection through ``__call__`` (not a direct ``.project()`` method call) is
        what lets FSDP2's forward hooks fire when the indexer is wrapped as its own ``fully_shard`` unit — the
        pre-forward all-gather and the pre-backward EXIT gate whose backward triggers the gradient
        reduce-scatter onto the sharded (optimizer) master. See docs/dsa_fsdp_sharding_notes.md (Option B2).
        """
        q_idx, k_idx, weights = self.project(x, qr, cos, sin)
        if return_projection:
            return q_idx, k_idx, weights
        return self.scores(q_idx, k_idx, weights, attn_bias)

    def select_topk(
        self, scores: torch.Tensor, top_k: Optional[int] = None, attn_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Inference selection: top-k key indices per query from the raw scores, matching the reference
        ``index_score.topk(min(index_topk, end_pos))[1]``.

        Args:
            scores: ``[b, s_q, s_k]`` raw indexer scores (from ``forward``/``scores``).
            top_k: number of keys to keep (defaults to ``cfg.top_k``); clamped to ``s_k``.
            attn_bias: optional additive mask applied before top-k (if not already in ``scores``).

        Returns:
            indices ``[b, s_q, k]`` of the selected keys.
        """
        if attn_bias is not None:
            scores = scores + attn_bias
        k = min(top_k if top_k is not None else self.cfg.top_k, scores.shape[-1])
        return scores.topk(k, dim=-1).indices
