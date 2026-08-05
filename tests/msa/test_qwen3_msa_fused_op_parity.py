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
"""P1 / S1 gate (docs/qwen3_4b_msa/serving_plan.md §6.2): is MiniMax-M3's fused
QK-norm+RoPE+KV-insert kernel numerically correct at **Qwen3** shapes?

Why this is the first thing we run: the kernel is mandatory inside
``MiniMaxM3SparseAttention.forward`` (no branch, ``nvidia/model.py:604``), and its own
guard admits ``rotary_dim == 128`` (``rotary_dim > 0 && %8 == 0 && <= kHeadDim``,
``fused.cu:678-680``) -- but M3 itself runs *partial* rotary and therefore never
exercises 128. Qwen3 is full-rotary, so we would be the first caller at that setting.
The outcome picks the port's implementation route (serving_plan §4.3):

  PASS -> Route A: subclass ``MiniMaxM3SparseAttention``, reuse the fused kernel,
          and export main ``q_norm``/``k_norm`` as ``w - 1`` (serving_plan §2.3).
  FAIL -> Route B: override ``forward``, use Qwen3's own norm modules + vLLM RoPE,
          and skip the ``w - 1`` conversion entirely.

Semantics implemented by the reference below, all read from
``csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu`` at v0.26.0:

* ``qkv`` row layout ``[q: nq*128 | k: nkv*128 | v: nkv*128 | index_q: niq*128 |
  index_k: 1*128]``, width ``(nq + 2*nkv + niq + 1) * 128``  (``:360``)
* Gemma RMSNorm ``x * rsqrt(mean(x^2)+eps) * (1 + w)``  (``:135-149``, the
  ``1.0f + weight[dim]`` at ``:145``) -- ONE helper, applied to q, k, index_q and
  index_k alike. V gets neither norm nor RoPE (``norm_w = nullptr``, ``:376-380``).
* Partial NeoX RoPE on dims ``[0, rotary_dim)``, ``half = rotary_dim/2``, pairing
  ``(i, i+half)``, with ``cos_ptr = cos_sin_cache + pos*rotary_dim`` so cos occupies
  ``[0, half)`` and sin ``[half, rotary_dim)``  (``:150-158``, ``:432``)
* norm THEN rope, in that order (``normAndRope``)
* ``q``/``index_q`` are de-interleaved into contiguous ``q_out``/``index_q_out``;
  ``k``/``v``/``index_k`` are rewritten in place in ``qkv`` and scatter-inserted into
  the paged caches -- main ``[num_blocks, nkv, block_size, 2*head_dim]`` (``:467-471``),
  index ``[num_blocks, block_size, head_dim]``.

Run (inside the serving venv, which has vLLM 0.26.0 and the compiled op):
  cd <repo> && .devlibs/vllm026/bin/python tests/msa/test_qwen3_msa_fused_op_parity.py
"""

import sys

import torch

# Qwen3-4B-Thinking-2507 geometry (docs/qwen3_4b_msa/serving_plan.md §3.1).
NQ = 32  # num_attention_heads
NKV = 8  # num_key_value_heads
NIQ = 8  # index heads == num_key_value_heads (vLLM asserts the equality)
HEAD_DIM = 128
ROTARY_DIM = 128  # <-- the setting under test; M3 ships partial rotary
EPS = 1e-6
ROPE_THETA = 5_000_000.0  # Qwen3-4B-Thinking-2507
BLOCK_SIZE = 128  # == sparse block size == page size
N_TOKENS = 96
NUM_BLOCKS = 8
DTYPE = torch.bfloat16


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2)+eps) * (1 + w), computed in fp32 like the kernel."""
    xf = x.float()
    rms_rcp = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms_rcp * (1.0 + w.float())).to(x.dtype)


def neox_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Partial NeoX RoPE on dims [0, rotary_dim); trailing dims pass through.

    half = rotary_dim/2, pairing (i, i+half) -- kernel fused.cu:150-158.
    """
    xf = x.float()
    out = xf.clone()
    half = rotary_dim // 2
    # NB: read x1/x2 from `xf`, write into `out`. Slicing `out` here would alias --
    # the first assignment would clobber x1 before the second line reads it.
    x1 = xf[..., :half]
    x2 = xf[..., half:rotary_dim]
    c = cos.float().unsqueeze(-2)  # [N, 1, half]
    s = sin.float().unsqueeze(-2)
    out[..., :half] = x1 * c - x2 * s
    out[..., half:rotary_dim] = x2 * c + x1 * s
    return out.to(x.dtype)


def build_cos_sin_cache(max_pos: int, rotary_dim: int, theta: float, device, dtype):
    """vLLM layout: [max_pos, rotary_dim] with cos in [0, half) and sin in [half, rotary_dim)."""
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, half]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype)


def reference(qkv, positions, q_nw, k_nw, iq_nw, ik_nw, cos_sin_cache):
    """Torch reference for the whole fused op. Returns (q_out, k, v, index_q_out, index_k)."""
    n = qkv.shape[0]
    half = ROTARY_DIM // 2
    cs = cos_sin_cache[positions]  # [N, rotary_dim]
    cos, sin = cs[:, :half], cs[:, half:]

    off_q = 0
    off_k = NQ * HEAD_DIM
    off_v = (NQ + NKV) * HEAD_DIM
    off_iq = (NQ + 2 * NKV) * HEAD_DIM
    off_ik = (NQ + 2 * NKV + NIQ) * HEAD_DIM

    q = qkv[:, off_q:off_k].view(n, NQ, HEAD_DIM)
    k = qkv[:, off_k:off_v].view(n, NKV, HEAD_DIM)
    v = qkv[:, off_v:off_iq].view(n, NKV, HEAD_DIM)
    iq = qkv[:, off_iq:off_ik].view(n, NIQ, HEAD_DIM)
    ik = qkv[:, off_ik:].view(n, 1, HEAD_DIM)

    q = neox_rope(gemma_rmsnorm(q, q_nw, EPS), cos, sin, ROTARY_DIM)
    k = neox_rope(gemma_rmsnorm(k, k_nw, EPS), cos, sin, ROTARY_DIM)
    # V: no norm, no rope.
    iq = neox_rope(gemma_rmsnorm(iq, iq_nw, EPS), cos, sin, ROTARY_DIM)
    ik = neox_rope(gemma_rmsnorm(ik, ik_nw, EPS), cos, sin, ROTARY_DIM)
    return q, k, v.clone(), iq, ik


def report(name: str, got: torch.Tensor, want: torch.Tensor, max_ulp: float = 2.0) -> bool:
    """Compare max absolute error against the tensor's PEAK magnitude, in bf16 ULP.

    Two wrong gates, both tried and rejected: a flat absolute tolerance (meaningless
    without knowing the data scale — 1 ULP near 4.0 is already 3.1e-2), and per-element
    ULP (RoPE computes x1*c - x2*s, which CANCELS: a near-zero output carries absolute
    error inherited from its O(1) inputs, so error in ULP-of-output reads in the
    thousands and says nothing). The kernel accumulates in fp32 and rounds once; the
    meaningful question is error relative to the tensor's dynamic range.

    Selection is what actually has to match downstream, not scores — plan.md §4.2 #2
    makes the same point about the index kernel ("near-threshold blocks can flip …
    the target is set overlap ≈ 1.0000").
    """
    g, w = got.float(), want.float()
    d = (g - w).abs()
    scale = w.abs().max().clamp_min(1e-6)  # peak magnitude of the tensor
    rel = (d.max() / scale).item()
    thresh = max_ulp * 2.0**-8  # max_ulp ULP at the peak magnitude
    ok = rel <= thresh
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name:12s} max|Δ|={d.max().item():.3e}  "
        f"mean|Δ|={d.mean().item():.3e}  peak={scale.item():.3f}  "
        f"max|Δ|/peak={rel:.2e} ({rel * 256:.2f} ULP, gate {max_ulp:g})"
    )
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 2
    try:
        from vllm import _custom_ops as ops
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot import vllm._custom_ops: {e!r}")
        return 1
    if not hasattr(ops, "fused_minimax_m3_qknorm_rope_kv_insert"):
        print(
            "FAIL: vllm._custom_ops has no fused_minimax_m3_qknorm_rope_kv_insert "
            "-- the wheel does not ship the compiled op (serving_plan §9 P0.3). "
            "=> Route B."
        )
        return 1

    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"shapes: nq={NQ} nkv={NKV} niq={NIQ} head_dim={HEAD_DIM} rotary_dim={ROTARY_DIM}")

    qkv_row = (NQ + 2 * NKV + NIQ + 1) * HEAD_DIM
    qkv = torch.randn(N_TOKENS, qkv_row, device=dev, dtype=DTYPE)
    qkv_ref_in = qkv.clone()  # the kernel mutates qkv in place
    positions = torch.arange(N_TOKENS, device=dev, dtype=torch.int64)

    # Gemma norms are zero-init in training (gain 1.0); perturb so a dropped
    # (1 + w) would actually show up rather than cancelling at w = 0.
    q_nw = (torch.randn(HEAD_DIM, device=dev, dtype=DTYPE) * 0.05)
    k_nw = (torch.randn(HEAD_DIM, device=dev, dtype=DTYPE) * 0.05)
    iq_nw = (torch.randn(HEAD_DIM, device=dev, dtype=DTYPE) * 0.05)
    ik_nw = (torch.randn(HEAD_DIM, device=dev, dtype=DTYPE) * 0.05)

    cos_sin_cache = build_cos_sin_cache(N_TOKENS + 8, ROTARY_DIM, ROPE_THETA, dev, DTYPE)

    kv_cache = torch.zeros(NUM_BLOCKS, NKV, BLOCK_SIZE, 2 * HEAD_DIM, device=dev, dtype=DTYPE)
    index_cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM, device=dev, dtype=DTYPE)
    slot_mapping = torch.arange(N_TOKENS, device=dev, dtype=torch.int64)
    index_slot_mapping = slot_mapping.clone()

    q_out = torch.empty(N_TOKENS, NQ * HEAD_DIM, device=dev, dtype=DTYPE)
    index_q_out = torch.empty(N_TOKENS, NIQ * HEAD_DIM, device=dev, dtype=DTYPE)

    try:
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, q_nw, k_nw, cos_sin_cache, positions,
            NQ, NKV, ROTARY_DIM, EPS,
            iq_nw, ik_nw, NIQ,
            slot_mapping, index_slot_mapping,
            kv_cache, index_cache, BLOCK_SIZE,
            q_out, index_q_out,
            "auto", False,
        )
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: kernel raised at rotary_dim={ROTARY_DIM}: {e!r}\n  => Route B.")
        return 1

    rq, rk, rv, riq, rik = reference(
        qkv_ref_in, positions, q_nw, k_nw, iq_nw, ik_nw, cos_sin_cache
    )

    print("\nfused op vs torch reference (gate: max|Δ|/peak <= 2 bf16 ULP):")
    ok = True
    ok &= report("q_out", q_out.view(N_TOKENS, NQ, HEAD_DIM), rq)
    ok &= report("index_q_out", index_q_out.view(N_TOKENS, NIQ, HEAD_DIM), riq)

    off_k = NQ * HEAD_DIM
    off_v = (NQ + NKV) * HEAD_DIM
    off_iq = (NQ + 2 * NKV) * HEAD_DIM
    off_ik = (NQ + 2 * NKV + NIQ) * HEAD_DIM
    ok &= report("k (in-place)", qkv[:, off_k:off_v].view(N_TOKENS, NKV, HEAD_DIM), rk)
    ok &= report("v (untouched)", qkv[:, off_v:off_iq].view(N_TOKENS, NKV, HEAD_DIM), rv, max_ulp=0.0)
    ok &= report("index_k", qkv[:, off_ik:].view(N_TOKENS, 1, HEAD_DIM), rik)

    # Cache inserts: slot s -> block s//block_size, token s%block_size.
    blk, tok = slot_mapping // BLOCK_SIZE, slot_mapping % BLOCK_SIZE
    got_k = kv_cache[blk, :, tok, :HEAD_DIM]
    got_v = kv_cache[blk, :, tok, HEAD_DIM:]
    ok &= report("kv_cache K", got_k, rk)
    ok &= report("kv_cache V", got_v, rv, max_ulp=0.0)
    ok &= report("index_cache", index_cache[blk, tok].unsqueeze(1), rik)

    print()
    if ok:
        print(f"S1 PASS at rotary_dim={ROTARY_DIM} => Route A (reuse the fused kernel, export w-1).")
        return 0
    print(f"S1 FAIL at rotary_dim={ROTARY_DIM} => Route B (own forward, Qwen3 norms, no conversion).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
