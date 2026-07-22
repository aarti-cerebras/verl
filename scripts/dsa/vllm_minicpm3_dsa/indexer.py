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
"""vLLM-side MiniCPM3 DSA "lightning indexer" (serving module).

This is the *serving* twin of the training-side
``verl.models.transformers.dsa_indexer.LightningIndexer``. It carries the SAME
parameters under the SAME names (``indexer.{wq_b, wk, k_norm, weights_proj}``)
so a Phase-2 checkpoint loads 1:1 (NO fusion into a ``wk_weights_proj`` GEMM —
unlike stock vLLM ``deepseek_v2.Indexer`` — because our checkpoint keeps ``wk``
and ``weights_proj`` separate).

Two forward paths:

* ``project_and_score(hidden_states, qr, cos, sin) -> logits [T, T]`` — the
  TESTABLE path. It runs the full padded FP8 forward and calls the DeepGEMM
  ``fp8_fp4_mqa_logits`` kernel directly (as tests/dsa/probe_deepgemm_indexer.py
  does), needing NO vLLM attention metadata. This is what the parity test checks.

* ``forward(hidden_states, qr, positions, rotary_emb, ...)`` — the SERVE path.
  Structured like vLLM ``deepseek_v2.Indexer.forward`` but with our separate
  projections + head/dim padding, delegating the paged top-k selection to the
  real ``SparseAttnIndexer`` runtime custom op. Requires a live vLLM engine
  context (KV cache + attn metadata), so it is only wired when a ``vllm_config``
  is supplied at construction; standalone (test) construction leaves it stubbed
  and raises a clear error if called. See the class docstring for status.

Math must match the training reference exactly:
  * RoPE pe-first on the ``rope_head_dim`` (=32) slice, non-interleaved
    ``rotate_half`` in fp32 (MiniCPM3 / llama convention).
  * ``k_norm`` LayerNorm over the real head_dim (=64).
  * Hadamard ``rotate_activation`` over the real head_dim (=64) on q and k.
  * per-head ``ReLU(<q_h, k>)`` weighted by ``w * softmax_scale``,
    ``softmax_scale = head_dim**-0.5 = 64**-0.5``, and ``w`` itself carries the
    ``n_heads**-0.5`` factor (with the REAL n_heads=16, not the padded 32).
  * FP8 UE8M0 per-token(row) quant with ``q_scale`` folded into ``weights``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure the vendored-deep_gemm shim is installed before anything imports the
# kernel wrapper (importing this module implies the package __init__ already ran,
# but call again — it is idempotent — in case indexer.py is imported standalone).
from . import install_deep_gemm_shim

install_deep_gemm_shim()

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
SUPPORTED_KERNEL_HEADS = (32, 64, 128)  # DeepGEMM mqa_logits accepts only these head counts


# --------------------------------------------------------------------------- #
# Math helpers (copied verbatim from verl.models.transformers.dsa_indexer so the
# serve module is self-contained at inference time — no dependency on the training
# package — while remaining bit-for-bit identical).
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[-x2, x1] (MiniCPM3 / llama non-interleaved convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the rope-slice of ``t`` in fp32.

    Args:
        t: ``[b, s, h, rope_head_dim]`` (rope slice only; ``h`` may be 1 for the MQA key).
        cos, sin: ``[b, s, rope_head_dim]`` (broadcast over the head dim).
    """
    orig_dtype = t.dtype
    cos = cos.float().unsqueeze(2)  # [b, s, 1, r]
    sin = sin.float().unsqueeze(2)
    t = t.float()
    out = t * cos + _rotate_half(t) * sin
    return out.to(orig_dtype)


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


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Hadamard transform over the last dim scaled by ``n**-0.5`` (DeepSeek-V3.2).

    Uses the ``fast_hadamard_transform`` CUDA kernel when available (identical to
    the training path), else a pure-torch FWHT (any orthonormal Hadamard ordering
    preserves the dot, so the fallback is equivalent for scores).
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


# --------------------------------------------------------------------------- #
# DeepGEMM MQA-logits kernel resolution (vendored, post-shim).
# --------------------------------------------------------------------------- #
_KERNEL_CACHE = None  # (flavor, callable); flavor in {"fp8_fp4", "legacy"}


def _resolve_mqa_logits_kernel():
    """Resolve the FP8 MQA-logits kernel, preferring vLLM's unified public wrapper
    (which — thanks to the shim — resolves to the vendored ``fp8_fp4_mqa_logits``),
    then the vendored module directly, then the legacy ``fp8_mqa_logits``."""
    global _KERNEL_CACHE
    if _KERNEL_CACHE is not None:
        return _KERNEL_CACHE

    install_deep_gemm_shim()  # make sure external top-level deep_gemm is blocked

    # 1) vLLM public wrapper (resolves to vendored fp8_fp4_mqa_logits after the shim).
    try:
        from vllm.utils import deep_gemm as _dg

        for name in list(vars(_dg)):  # reset any stale external/_missing resolution
            if name.endswith("_impl"):
                setattr(_dg, name, None)
        _dg._lazy_init()
        if getattr(_dg, "_fp8_fp4_mqa_logits_impl", None) is not None:
            _KERNEL_CACHE = ("fp8_fp4", _dg.fp8_fp4_mqa_logits)
            return _KERNEL_CACHE
    except Exception:
        pass

    # 2) vendored module directly.
    try:
        import vllm.third_party.deep_gemm as vdg

        if hasattr(vdg, "fp8_fp4_mqa_logits"):
            _KERNEL_CACHE = ("fp8_fp4", vdg.fp8_fp4_mqa_logits)
            return _KERNEL_CACHE
        if hasattr(vdg, "fp8_mqa_logits"):
            _KERNEL_CACHE = ("legacy", vdg.fp8_mqa_logits)
            return _KERNEL_CACHE
    except Exception:
        pass

    raise RuntimeError(
        "No DeepGEMM MQA-logits kernel available. Expected vendored "
        "vllm.third_party.deep_gemm.fp8_fp4_mqa_logits (or legacy fp8_mqa_logits)."
    )


def _quant_fp8_rows(x: torch.Tensor, use_ue8m0: bool = True, eps: float = 1e-10):
    """UE8M0 per-row (last-dim) FP8 E4M3 quant. Returns (x_fp8, scale[...]).

    Pure-torch fallback identical to probe_deepgemm_indexer.quant_fp8_rows; used
    only when vLLM's ``per_token_group_quant_fp8`` is unavailable."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / FP8_MAX
    if use_ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    x_q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_q, scale.squeeze(-1)


def _per_token_group_quant(x: torch.Tensor, group_size: int, use_ue8m0: bool = True):
    """Prefer vLLM's ``per_token_group_quant_fp8`` (exact serve numerics); fall back
    to the pure-torch row quant. ``x`` last dim must be a multiple of ``group_size``;
    here head_dim(128) == group_size so there is a single group per row.

    Returns (x_fp8 [..., D], scale [...]) with the trailing singleton group dim removed.
    """
    try:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

        x_fp8, x_scale = per_token_group_quant_fp8(
            x, group_size, column_major_scales=False, use_ue8m0=use_ue8m0
        )
        return x_fp8, x_scale.squeeze(-1)
    except Exception:
        return _quant_fp8_rows(x, use_ue8m0=use_ue8m0)


class MiniCPM3DSAIndexer(nn.Module):
    """DSA lightning indexer for MiniCPM3, vLLM serving side.

    Parameters match the checkpoint 1:1 (``indexer.{wq_b, wk, k_norm, weights_proj}``);
    NOT fused, unlike stock vLLM. The FP8 logit matmul runs through DeepGEMM.
    """

    def __init__(
        self,
        *,
        n_heads: int = 16,
        head_dim: int = 64,
        rope_head_dim: int = 32,
        top_k: int = 256,
        q_lora_rank: int = 768,
        hidden_size: int = 2560,
        fp8: bool = True,
        rotate_activation: bool = True,
        dtype: torch.dtype | None = None,
        # Serve-path (runtime) wiring — only built when a live vLLM engine config is
        # supplied; standalone/test construction leaves the serve path stubbed.
        vllm_config=None,
        cache_config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.top_k = top_k
        self.q_lora_rank = q_lora_rank
        self.hidden_size = hidden_size
        self.fp8 = fp8
        self.rotate_activation = rotate_activation
        self.softmax_scale = head_dim**-0.5  # 64**-0.5

        # Padded dims required by the DeepGEMM kernel (dim 64->128, heads 16->32).
        self.padded_head_dim = 128
        self.quant_block_size = 128
        self.scale_fmt = "ue8m0"
        self.padded_n_heads = next(hs for hs in SUPPORTED_KERNEL_HEADS if n_heads <= hs)

        # SEPARATE projections — names match the checkpoint so weights load 1:1.
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim)
        self.weights_proj = nn.Linear(hidden_size, n_heads, bias=False)
        if dtype is not None:
            self.to(dtype)

        # ---- serve-path runtime op (optional) ----
        # ``topk_indices_buffer`` / ``topk_tokens`` are read by vLLM's sparse MLA
        # backend (FlashMLASparseImpl reads ``indexer.topk_indices_buffer``) and by
        # the MLA wrapper (``indexer.topk_tokens``), so expose them here.
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = top_k
        self.indexer_op = None
        self.k_cache = None
        self._serve_ready = False
        if vllm_config is not None:
            self._build_runtime_op(vllm_config, cache_config, topk_indices_buffer, prefix)

    # ------------------------------------------------------------------ #
    # Serve-path runtime wiring
    # ------------------------------------------------------------------ #
    def _build_runtime_op(self, vllm_config, cache_config, topk_indices_buffer, prefix):
        """Build the ``DeepseekV32IndexerCache`` + ``SparseAttnIndexer`` runtime op.

        Requires a live vLLM engine context (``get_current_vllm_config()`` /
        compilation config / KV-cache manager). Structured exactly like stock vLLM
        ``deepseek_v2.Indexer.__init__`` but with our PADDED head_dim (128).
        """
        from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        # fp8 naive cache: store value in fp8 + a fp32 scale per quant_block_size elems.
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.padded_head_dim
            + self.padded_head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.top_k,
            self.padded_head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            topk_indices_buffer,
        )
        self._serve_ready = True

    # ------------------------------------------------------------------ #
    # Shared projection (pe-first RoPE, k_norm, weights)
    # ------------------------------------------------------------------ #
    def _project(self, hidden_states, qr, cos, sin):
        """Compute (q_idx [T,H,64], k_idx [T,64], weights [T,H]) with RoPE + k_norm.

        Accepts hidden_states/qr as [T,*] or [1,T,*]; cos/sin as [T,r] or [1,T,r].
        Returns token-flattened tensors (batch dim removed).
        """
        # normalize to [1, T, *] for the rope helper, then squeeze batch at the end.
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if qr.dim() == 2:
            qr = qr.unsqueeze(0)
        if cos.dim() == 2:
            cos = cos.unsqueeze(0)
        if sin.dim() == 2:
            sin = sin.unsqueeze(0)
        b, s, _ = hidden_states.shape
        r = self.rope_head_dim

        q = self.wq_b(qr).view(b, s, self.n_heads, self.head_dim)
        q = torch.cat([_apply_rope(q[..., :r], cos, sin), q[..., r:]], dim=-1)

        k = self.k_norm(self.wk(hidden_states))  # [b, s, head_dim]
        k_he = k.unsqueeze(2)  # [b, s, 1, head_dim]
        k = torch.cat([_apply_rope(k_he[..., :r], cos, sin), k_he[..., r:]], dim=-1).squeeze(2)

        # weights_proj in fp32 (reference keeps it float32), carrying n_heads**-0.5
        # with the REAL n_heads (not the padded head count).
        weights = F.linear(hidden_states.float(), self.weights_proj.weight.float()) * (
            self.n_heads**-0.5
        )
        return q[0], k[0], weights[0]  # [T,H,64], [T,64], [T,H]

    # ------------------------------------------------------------------ #
    # TESTABLE path: full padded FP8 forward -> DeepGEMM logits [T, T]
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def project_and_score(self, hidden_states, qr, cos, sin) -> torch.Tensor:
        """Return causal indexer logits ``[T, T]`` via the DeepGEMM FP8 MQA kernel.

        Steps (matching probe_deepgemm_indexer + vLLM Indexer.forward):
          1. project -> q [T,16,64], k [T,64], weights [T,16]
          2. Hadamard rotate_activation over the REAL 64 dims (q and k) if enabled
          3. pad dim 64->128 (zeros), pad heads 16->32 (zero q rows + zero weights)
          4. UE8M0 per-token(row) FP8 quant of q and k
          5. fold q_scale into weights: w * softmax_scale * n_heads**-0.5 * q_scale
             (n_heads**-0.5 already applied in _project; here fold softmax_scale+q_scale)
          6. causal cu_seqlens (query m sees keys 0..m), call the kernel
        """
        q, k, weights = self._project(hidden_states, qr, cos, sin)  # [T,16,64],[T,64],[T,16]
        T = q.shape[0]
        H = self.n_heads

        # 2) Hadamard over the real 64 (orthonormal: preserves the dot, spreads FP8 mass)
        if self.fp8 and self.rotate_activation:
            q = _rotate_activation(q)
            k = _rotate_activation(k)

        # 3) pad dim 64 -> 128 (zeros; dot unchanged, amax unchanged so scale unchanged)
        pad_d = self.padded_head_dim - self.head_dim
        if pad_d > 0:
            q = F.pad(q, (0, pad_d))
            k = F.pad(k, (0, pad_d))
        # pad heads 16 -> 32 with ZERO q rows and ZERO weights (contribute w*relu()=0)
        Hs = self.padded_n_heads
        if H < Hs:
            q = F.pad(q, (0, 0, 0, Hs - H))  # [T, Hs, 128]
            weights = F.pad(weights, (0, Hs - H))  # [T, Hs]

        # 4) FP8 UE8M0 quant. q flattened to [T*Hs, 128] (one group), k over [T,128].
        q_flat = q.reshape(-1, self.padded_head_dim).contiguous()
        q_fp8, q_scale = _per_token_group_quant(q_flat, self.quant_block_size, use_ue8m0=True)
        q_fp8 = q_fp8.view(T, Hs, self.padded_head_dim)
        q_scale = q_scale.view(T, Hs)
        k_fp8, k_scale = _per_token_group_quant(k.contiguous(), self.quant_block_size, use_ue8m0=True)

        # 5) fold q_scale + softmax_scale into weights (q_scale>0 pulls out of the ReLU).
        #    weights already carries n_heads**-0.5 from _project.
        fused_w = (weights * self.softmax_scale * q_scale).float().contiguous()  # [T, Hs]

        # 6) causal cu_seqlens: query m sees keys [0, m].
        cu_ks = torch.zeros(T, dtype=torch.int32, device=q.device)
        cu_ke = torch.arange(1, T + 1, dtype=torch.int32, device=q.device).clamp(max=T)

        flavor, kernel = _resolve_mqa_logits_kernel()
        q_fp8 = q_fp8.contiguous()
        k_fp8 = k_fp8.contiguous()
        k_scale = k_scale.float().contiguous()
        if flavor == "fp8_fp4":
            # FP8 path: q as (values, None) — per-token scale folded into weights.
            logits = kernel((q_fp8, None), (k_fp8, k_scale), fused_w, cu_ks, cu_ke, clean_logits=True)
        else:  # legacy fp8_mqa_logits(q[M,H,D], (k,k_scale), weights, cu_ks, cu_ke, clean)
            logits = kernel(q_fp8, (k_fp8, k_scale), fused_w, cu_ks, cu_ke, True)
        return logits  # [T, T], fp32, -inf off-causal

    def select_topk(self, logits: torch.Tensor, top_k: int | None = None) -> torch.Tensor:
        """Top-k key indices per query from raw logits ``[T, T]`` (clamped to #keys)."""
        k = min(top_k if top_k is not None else self.top_k, logits.shape[-1])
        return logits.topk(k, dim=-1).indices

    # ------------------------------------------------------------------ #
    # SERVE path (paged runtime). Structurally complete; requires a live vLLM
    # engine (KV cache + attn metadata via the SparseAttnIndexer custom op).
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states, qr, positions, rotary_emb) -> torch.Tensor:
        """Serve-path forward: mirrors vLLM ``deepseek_v2.Indexer.forward`` but with
        our SEPARATE wk / weights_proj + head/dim padding, delegating paged top-k
        selection to the ``SparseAttnIndexer`` runtime op.

        NOTE: this path needs a live vLLM engine context (the SparseAttnIndexer op
        reads KV cache + per-batch attention metadata from the forward context).
        It is only wired when ``vllm_config`` was supplied at construction. When it
        is not (e.g. standalone tests), this raises — use ``project_and_score`` for
        the metadata-free, testable path.
        """
        if not self._serve_ready or self.indexer_op is None:
            raise RuntimeError(
                "Serve-path forward requires a live vLLM engine: construct with "
                "vllm_config=... to build the DeepseekV32IndexerCache + "
                "SparseAttnIndexer. For standalone GPU testing use project_and_score()."
            )

        # q from the compressed latent qr, pe-first split.
        q = self.wq_b(qr).view(-1, self.n_heads, self.head_dim)
        r = self.rope_head_dim
        q_pe, q_nope = torch.split(q, [r, self.head_dim - r], dim=-1)

        # SEPARATE wk + weights_proj (checkpoint-native; NOT fused).
        k = self.k_norm(self.wk(hidden_states))  # [T, head_dim]
        weights = F.linear(hidden_states.float(), self.weights_proj.weight.float())  # [T, n_heads]
        k_pe, k_nope = torch.split(k, [r, self.head_dim - r], dim=-1)

        # RoPE (vLLM rotary_emb, NeoX) on the pe slice; k as MQA single head.
        # vLLM's rotary_emb expects the rope input flattened to
        # ``[T, n_heads * rope_dim]`` (query) / ``[T, rope_dim]`` (key), exactly as
        # stock MiniCPM3 does (minicpm3.py:156-159) — passing the unflattened
        # ``[T, n_heads, rope_dim]`` mis-indexes the cos/sin gather. Flatten first,
        # then reshape back.
        T = q_pe.shape[0]
        q_pe = q_pe.reshape(T, self.n_heads * r)
        k_pe = k_pe.reshape(T, r)
        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
        q_pe = q_pe.reshape(-1, self.n_heads, r)
        k_pe = k_pe.reshape(-1, 1, r)
        q = torch.cat([q_pe, q_nope], dim=-1)  # [T, n_heads, 64]
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)  # [T, 64]

        # Hadamard rotate_activation over the real 64 (q and k).
        if self.fp8 and self.rotate_activation:
            q = _rotate_activation(q)
            k = _rotate_activation(k)

        # pad dim 64->128 and heads 16->32 (zero q / zero weights on padded heads).
        pad_d = self.padded_head_dim - self.head_dim
        if pad_d > 0:
            q = F.pad(q, (0, pad_d))
            k = F.pad(k, (0, pad_d))
        Hs = self.padded_n_heads
        if self.n_heads < Hs:
            q = F.pad(q, (0, 0, 0, Hs - self.n_heads))
            weights = F.pad(weights, (0, Hs - self.n_heads))

        # quant q (k quant is fused into cache insertion inside the op).
        q_flat = q.view(-1, self.padded_head_dim)
        q_fp8, q_scale = _per_token_group_quant(q_flat, self.quant_block_size, use_ue8m0=True)
        q_fp8 = q_fp8.view(-1, Hs, self.padded_head_dim)
        q_scale = q_scale.view(-1, Hs, 1)

        # fold q_scale + softmax_scale + n_heads**-0.5 (real n_heads) into weights.
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_heads**-0.5
        weights = weights.squeeze(-1)  # [T, Hs]

        return self.indexer_op(hidden_states, q_fp8, k, weights)
