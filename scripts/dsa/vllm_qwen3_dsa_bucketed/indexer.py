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
"""Serving twin of ``verl.models.transformers.qwen3_dsa_indexer.Qwen3DSAIndexer``.

Adapted from ``scripts/dsa/vllm_minicpm3_dsa/indexer.py`` (already parity-tested against its own
training module). Four deltas, all forced by Qwen3 being GQA rather than MLA:

1. **The query comes from the hidden states, not an MLA latent.** MLA hands its indexer a free
   compressed query ``qr``; GQA has none, so ``wq`` projects straight from ``hidden_states`` and the
   parameter is ``wq`` (not ``wq_b``).
2. **A ``q_norm`` (RMSNorm) after ``wq``**, replacing the normalization MLA's ``q_a_layernorm``
   provided upstream. This is why we cannot instantiate vLLM's own ``deepseek_v2.Indexer``: it has no
   query norm, and a norm cannot be folded into a GEMM. It costs nothing here — see (4).
3. **``rope_head_dim == head_dim == 64``**: all indexer dims are roped (MiniCPM3 roped 32 of 64).
4. **Own rotary tables**, built at the base model's ``rope_theta`` in fp64, rather than vLLM's
   ``rotary_emb``. The training module does the same and the reason matters: with ``rotate_half``, a
   64-wide RoPE needs 32 frequencies *duplicated*, whereas slicing a 128-wide table's first 64 entries
   gives 64 *distinct* frequencies -- a different function. Recomputing at the same theta reproduces
   the base rope's even-indexed frequencies exactly.

**Kernel reuse is total.** The fp8 paged logits, the paged top-k and the side-cache insert are
``SparseAttnIndexer`` (``vllm/model_executor/layers/sparse_attn_indexer.py:706``) writing into
``DeepseekV32IndexerCache``, both taken verbatim. We only prepare its inputs.

**The 16x64 -> 32x128 pad is mandatory, not an optimization.** vLLM sizes the fp8 side cache as
``head_dim + head_dim // quant_block_size * 4`` with ``quant_block_size = 128``
(``deepseek_v2.py:694-697``): at ``head_dim=64`` that reserves **zero** scale bytes. Padding is also
lossless -- norms, RoPE and the Hadamard run at the real 64 dims, and zeros are appended immediately
before quantization, so the dot is unchanged and the row amax (hence the UE8M0 scale) is unchanged.
Padded heads carry zero ``q`` rows and zero gate weights, contributing ``w * ReLU(0) = 0``.

Note also that ``use_fused_indexer_q`` requires ``head_dim == 128 and rope_dim == 64``
(``deepseek_v2.py:723-729``), so at ``d_idx = 64`` the fused query path is unavailable regardless --
which is exactly what makes inserting ``q_norm`` free. The two mismatches cancel.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
# DeepGEMM's mqa_logits accepts only these head counts, at head_dim 128.
SUPPORTED_KERNEL_HEADS = (32, 64, 128)
PADDED_HEAD_DIM = 128
QUANT_BLOCK_SIZE = 128
# Empirical ceiling of vLLM 0.26.0's compiled top-k (`_C.top_k_per_row_prefill`, reached through
# `sparse_attn_indexer`): top_k 4096 runs, 8192 dies with an ASYNC `CUDA error: invalid argument`
# whose traceback points at the *preceding* DeepGEMM `fp8_fp4_mqa_logits` launch and therefore reads
# as a logits-kernel bug. Measured 2026-08-20 on H100. It does not constrain the real config
# (DeepSeek, Keye and this checkpoint all use 2048), but it DOES cap the dense-equivalence control
# (`top_k >= T`) to sequences of at most 4096 tokens. Asserted here so that lands as one clear
# message at construction instead of an inscrutable CUDA error 90 s into warmup.
MAX_KERNEL_TOP_K = 4096

_LOGGED_HADAMARD = False


# --------------------------------------------------------------------------------------------- #
# Math helpers. Copied from verl/models/transformers/qwen3_dsa_indexer.py rather than imported, so
# the serving module has no training-package dependency -- and kept bit-identical to it, which is
# what the parity test asserts.
# --------------------------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[-x2, x1]`` (llama / Qwen3 non-interleaved "neox" convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE in fp32. ``t`` ``[T, h, r]``; ``cos``/``sin`` ``[T, r]``, broadcast over heads."""
    orig_dtype = t.dtype
    cos = cos.float().unsqueeze(1)  # [T, 1, r]
    sin = sin.float().unsqueeze(1)
    t = t.float()
    return (t * cos + _rotate_half(t) * sin).to(orig_dtype)


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized fast Walsh-Hadamard transform over the last dim (power of 2)."""
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


# Resolved ONCE at import, never inside the function. An `import` statement inside a Dynamo-traced
# region is an unconditional graph break -- `torch._dynamo.exc.Unsupported: Import failure`, which
# kills engine startup outright when cudagraphs/torch.compile are on (i.e. everything except
# --enforce-eager). Found the hard way on 2026-08-20.
try:  # pragma: no cover - environment dependent
    from fast_hadamard_transform import hadamard_transform as _HADAMARD_CUDA
except ImportError:
    _HADAMARD_CUDA = None

HADAMARD_IMPL = "fast_hadamard_transform" if _HADAMARD_CUDA is not None else "torch_fwht"


@torch.library.custom_op("qwen3_dsa_bucketed::rotate_activation", mutates_args=())
def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """DeepSeek-V3.2 ``rotate_activation``: orthonormal Hadamard, so ``q.k`` is preserved.

    Registered as a **custom op** so Dynamo treats it as opaque. Two weaker approaches both fail
    with torch.compile on (i.e. anything but ``--enforce-eager``), and each cost a bring-up cycle:
    an `import` inside the function is an unconditional graph break
    (``torch._dynamo.exc.Unsupported: Import failure``), and ``@torch._dynamo.disable`` raises
    ``Unsupported: Skip calling torch.compiler.disable()'d function`` because vLLM's piecewise
    region tolerates only registered splitting ops. A custom op stays IN the graph, so cudagraph
    capture still covers it -- the underlying kernel is an autograd.Function over a C extension,
    which Dynamo cannot trace but CUDA can happily record.

    Two implementations, and WHICH ONE RUNS IS PART OF THE TRAIN/SERVE CONTRACT. Both compute the
    standard unnormalized FWHT scaled by ``n**-0.5``, so they agree up to floating-point summation
    order -- but the rotation is applied immediately BEFORE fp8 quantization, so a per-element
    difference can flip a near-tie in the top-k. The training module resolves the same two options
    the same way (``verl/models/transformers/qwen3_dsa_indexer.py``), so the environments agree as
    long as the package is present (or absent) in both. ``HADAMARD_IMPL`` is logged at construction
    for exactly this reason.
    """
    scale = x.shape[-1] ** -0.5
    if x.is_cuda and _HADAMARD_CUDA is not None:
        in_dtype = x.dtype
        with torch.no_grad():  # the kernel is an autograd.Function; we never want its graph
            return _HADAMARD_CUDA(x.to(torch.bfloat16), scale=scale).to(in_dtype).clone()
    return (_fwht(x) * scale).clone()  # custom_op must not return an alias/view of its input


@_rotate_activation.register_fake
def _rotate_activation_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _quant_fp8_rows(x: torch.Tensor, use_ue8m0: bool = True, eps: float = 1e-10):
    """UE8M0 per-row FP8 E4M3 quant -> ``(x_fp8, scale)``. Pure-torch fallback."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / FP8_MAX
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    x_q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_q, scale.squeeze(-1)


# Resolved at import for the same Dynamo reason as the Hadamard above: an `import` inside a traced
# region is an unconditional graph break.
try:  # pragma: no cover - environment dependent
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8 as _VLLM_GROUP_QUANT,
    )
except ImportError:
    _VLLM_GROUP_QUANT = None


def _per_token_group_quant(x: torch.Tensor, group_size: int = QUANT_BLOCK_SIZE, use_ue8m0: bool = True):
    """Prefer vLLM's ``per_token_group_quant_fp8`` (the exact serve numerics); else pure torch."""
    if _VLLM_GROUP_QUANT is not None:
        x_fp8, x_scale = _VLLM_GROUP_QUANT(
            x, group_size, column_major_scales=False, use_ue8m0=use_ue8m0
        )
        return x_fp8, x_scale.squeeze(-1)
    return _quant_fp8_rows(x, use_ue8m0=use_ue8m0)


def _fake_quant_fp8(x: torch.Tensor, use_ue8m0: bool = True) -> torch.Tensor:
    """Dequantized round trip, matching the training module's forward value exactly (no STE needed
    at serve). Used only by ``torch_scores`` -- the reference path for the parity test."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        return (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).float() * scale
    return (x / scale).to(FP8_DTYPE).float() * scale


class _IndexerRMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) * weight``, computed in fp32. NOT the Gemma ``(1 + w)`` form."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(in_dtype)


class Qwen3DSARotary(nn.Module):
    """The indexer's own ``rope_head_dim``-wide rotary at the base model's theta (fp64 -> fp32)."""

    def __init__(self, dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``positions [T]`` -> ``(cos, sin)`` each ``[T, dim]`` fp32."""
        freqs = positions.float().unsqueeze(-1) * self.inv_freq.to(positions.device)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class Qwen3DSABucketedServingIndexer(nn.Module):
    """One layer's lightning indexer, serving side.

    Parameter names match the checkpoint 1:1 (``self_attn.indexer.{wq, q_norm, wk, k_norm,
    weights_proj}``), so weights load without a mapper. Nothing is fused: the training module keeps
    ``wk`` and ``weights_proj`` separate, unlike stock vLLM's ``wk_weights_proj``.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 2560,
        n_heads: int = 16,
        head_dim: int = 64,
        rope_head_dim: int = 64,
        rope_theta: float = 5e6,
        top_k: int = 2048,
        fp8: bool = True,
        fp8_ue8m0: bool = True,
        rotate_activation: bool = True,
        # Serve-path wiring; omit for the metadata-free (test) paths.
        vllm_config=None,
        cache_config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert head_dim in (32, 64, 128), f"head_dim must be 32/64/128 to serve, got {head_dim}"
        assert 128 % n_heads == 0, f"n_heads must divide 128 to serve, got {n_heads}"
        assert top_k <= MAX_KERNEL_TOP_K, (
            f"top_k={top_k} exceeds MAX_KERNEL_TOP_K={MAX_KERNEL_TOP_K}: vLLM 0.26.0's compiled "
            "top-k kernel fails above ~4096 with an async 'CUDA error: invalid argument' during "
            "warmup. For the dense-equivalence control use top_k=4096 with a prompt shorter than "
            "that, rather than a larger top_k."
        )
        assert fp8_ue8m0 or not fp8, (
            "the serving kernels quantize with a UE8M0 (power-of-2) scale; a checkpoint trained "
            "against a continuous scale would drift in its selection (memory dsa-fp8-ue8m0-fix)"
        )
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.top_k = top_k
        self.fp8 = fp8
        self.fp8_ue8m0 = fp8_ue8m0
        self.rotate_activation = rotate_activation
        self.softmax_scale = head_dim**-0.5
        self.padded_n_heads = next(h for h in SUPPORTED_KERNEL_HEADS if n_heads <= h)
        self.layer_name = prefix or "metadata_free"

        self.wq = nn.Linear(hidden_size, n_heads * head_dim, bias=False)
        self.q_norm = _IndexerRMSNorm(head_dim, eps=1e-6)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(hidden_size, n_heads, bias=False)
        self.rotary = Qwen3DSARotary(rope_head_dim, rope_theta)
        # HARD GATE, not a warning. Measured 2026-08-20 on Qwen3 shapes: the CUDA kernel and the
        # torch FWHT differ by ~6.7e-3 relative in bf16, and because the rotation feeds fp8
        # quantization the resulting fp8 BYTES and even the UE8M0 row scales differ -- so running
        # the fallback against a checkpoint trained on the kernel perturbs the selected set for
        # every near-tie. The training runs used /usr/bin/python3, whose user site-packages carries
        # fast_hadamard_transform 1.1.0 (see the launch manifests under the ckpt's logs/), so the
        # kernel is the contract. This is the fourth instance of this failure class in the project,
        # after fp8_ue8m0, the missing index_topk gate and nondeterministic top-k ties -- every one
        # of which presented as a fluent, plausible model rather than an error.
        if rotate_activation and fp8 and _HADAMARD_CUDA is None:
            if os.environ.get("DSA_ALLOW_TORCH_HADAMARD", "0") in ("0", "", "false", "False"):
                raise RuntimeError(
                    "fast_hadamard_transform is not importable in this environment, but the "
                    "checkpoint was trained with it (dsa_rotate_activation=True). The torch FWHT "
                    "fallback changes the fp8 quantization of q/k and therefore the selected "
                    "token set. Install/symlink the package into the serving venv (see "
                    "docs/qwen3_4b_dsa/serving_eval_plan.md), or set DSA_ALLOW_TORCH_HADAMARD=1 to "
                    "accept the drift deliberately."
                )
            print("[Qwen3DSA] WARNING: DSA_ALLOW_TORCH_HADAMARD=1 -- serving on the torch FWHT "
                  "fallback while the checkpoint trained on the CUDA kernel. Selection will drift.")
        global _LOGGED_HADAMARD
        if rotate_activation and fp8 and not _LOGGED_HADAMARD:
            _LOGGED_HADAMARD = True
            print(f"[Qwen3DSA] Hadamard rotate_activation impl: {HADAMARD_IMPL} "
                  f"(part of the train/serve numerics contract -- see _rotate_activation)")

        # vLLM's sparse MLA impl reads `indexer.topk_indices_buffer` / `indexer.topk_tokens`; ours
        # reads them off the impl, but keep the attribute names for anything that introspects.
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = top_k
        self.indexer_op = None
        self.k_cache = None
        if vllm_config is not None:
            self._build_runtime_op(vllm_config, cache_config, topk_indices_buffer, prefix)

    # ----------------------------------------------------------------------------------------- #
    def _build_runtime_op(self, vllm_config, cache_config, topk_indices_buffer, prefix) -> None:
        """Build ``DeepseekV32IndexerCache`` + ``SparseAttnIndexer`` at the PADDED head_dim."""
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        assert topk_indices_buffer is not None, "serve path needs the shared top-k buffer"
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=PADDED_HEAD_DIM + PADDED_HEAD_DIM // QUANT_BLOCK_SIZE * 4,  # 128 + 4
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            QUANT_BLOCK_SIZE,
            "ue8m0",
            self.top_k,
            PADDED_HEAD_DIM,
            vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config),
            topk_indices_buffer,
        )

    # ----------------------------------------------------------------------------------------- #
    # Shared projection: wq -> q_norm -> RoPE ; wk -> k_norm -> RoPE ; fp32 gate.
    # Order is project -> norm -> RoPE, matching the training module.
    # ----------------------------------------------------------------------------------------- #
    def project(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[T, hidden]`` -> ``(q [T, n_heads, d], k [T, d], weights [T, n_heads] fp32)``."""
        x = hidden_states if hidden_states.dim() == 2 else hidden_states.reshape(-1, self.hidden_size)
        T = x.shape[0]
        r = self.rope_head_dim
        cos, sin = self.rotary(positions.view(-1))

        q = self.q_norm(self.wq(x).view(T, self.n_heads, self.head_dim))
        q = (
            _apply_rope(q, cos, sin)
            if r == self.head_dim
            else torch.cat([_apply_rope(q[..., :r], cos, sin), q[..., r:]], dim=-1)
        )

        k = self.k_norm(self.wk(x)).unsqueeze(1)  # [T, 1, d] for the rope broadcast
        k = (
            _apply_rope(k, cos, sin)
            if r == self.head_dim
            else torch.cat([_apply_rope(k[..., :r], cos, sin), k[..., r:]], dim=-1)
        )
        k = k.squeeze(1)

        # fp32, and carrying n_heads**-0.5 with the REAL head count -- as the training module does.
        weights = F.linear(x.float(), self.weights_proj.weight.float()) * (self.n_heads**-0.5)
        return q, k, weights

    def _pad_for_kernel(self, q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor):
        """Hadamard at the real dims, then zero-pad ``64 -> 128`` and ``16 -> 32`` heads."""
        if self.fp8 and self.rotate_activation:
            q = _rotate_activation(q)
            k = _rotate_activation(k)
        pad_d = PADDED_HEAD_DIM - self.head_dim
        if pad_d > 0:
            q = F.pad(q, (0, pad_d))
            k = F.pad(k, (0, pad_d))
        pad_h = self.padded_n_heads - self.n_heads
        if pad_h > 0:
            q = F.pad(q, (0, 0, 0, pad_h))  # zero q rows
            weights = F.pad(weights, (0, pad_h))  # zero gates -> w * ReLU(0) = 0
        return q, k, weights

    # ----------------------------------------------------------------------------------------- #
    # Metadata-free reference path (parity tests): full causal logits, in torch.
    # ----------------------------------------------------------------------------------------- #
    @torch.no_grad()
    def torch_scores(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``[T, T]`` raw indexer scores, computed exactly as the training module's ``scores`` does.

        This is the reference the paged kernels are checked against; it needs no engine, no KV cache
        and no DeepGEMM, which is what lets the parity test run anywhere.
        """
        q, k, weights = self.project(hidden_states, positions)
        if self.fp8:
            if self.rotate_activation:
                q = _rotate_activation(q)
                k = _rotate_activation(k)
            q = _fake_quant_fp8(q, self.fp8_ue8m0)
            k = _fake_quant_fp8(k, self.fp8_ue8m0)
        dots = torch.relu(torch.einsum("qhd,kd->qhk", q.float(), k.float()))
        eff_w = weights.float() * self.softmax_scale
        return torch.einsum("qhk,qh->qk", dots, eff_w)

    @torch.no_grad()
    def select_topk(self, scores: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
        k = min(top_k if top_k is not None else self.top_k, scores.shape[-1])
        return scores.topk(k, dim=-1).indices

    # ----------------------------------------------------------------------------------------- #
    # Serve path: hand the padded fp8 query, the padded bf16 key and the fused gate to the op.
    # ----------------------------------------------------------------------------------------- #
    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.indexer_op is None:
            raise RuntimeError(
                "serve path needs a live vLLM engine: construct with vllm_config=... . For "
                "engine-free testing use torch_scores()."
            )
        q, k, weights = self.project(hidden_states, positions)
        q, k, weights = self._pad_for_kernel(q, k, weights)

        Hs = self.padded_n_heads
        q_fp8, q_scale = _per_token_group_quant(q.reshape(-1, PADDED_HEAD_DIM).contiguous())
        q_fp8 = q_fp8.view(-1, Hs, PADDED_HEAD_DIM)
        q_scale = q_scale.view(-1, Hs)

        # Fold the per-row q scale and softmax_scale into the gate: both are positive scalars per
        # (token, head), so they pull straight out of the ReLU. `weights` already carries
        # n_heads**-0.5 from project(). The kernel itself omits the score scale
        # (`common/ops/index_topk.py`), so absolute serve-side scores are NOT comparable to
        # training scores -- only the selected SET is.
        fused_w = (weights * q_scale * self.softmax_scale).float().contiguous()

        # k is inserted into the fp8 side cache by the op itself (bf16 in, fp8+scale out).
        # A plain-attribute scope is Dynamo-traceable and gives eager telemetry its layer name.
        # Decode graph capture recovers the same static prefix from vLLM's custom-op frame because
        # Python scopes do not execute on replay.
        from .bucket_selector_runtime import RUNTIME

        if not RUNTIME.telemetry_active:
            return self.indexer_op(hidden_states, q_fp8, k.contiguous(), fused_w)
        with RUNTIME.layer(self.layer_name):
            return self.indexer_op(hidden_states, q_fp8, k.contiguous(), fused_w)
