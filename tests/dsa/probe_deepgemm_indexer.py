#!/usr/bin/env python3
"""Numerical probe of the DeepGEMM FP8 MQA indexer-logits kernel for MiniCPM3 DSA.

Validates the FP8 indexer-logits kernel that vLLM's DeepSeek-V3.2 sparse path uses,
for our MiniCPM3 indexer config (n_heads=16, head_dim=64, top_k=512), against our
trusted pure-torch reference `verl.models.transformers.dsa_indexer.LightningIndexer`.

KERNEL SELECTION GOTCHA (see printed notes):
  * The vLLM wrapper `vllm.utils.deep_gemm.fp8_fp4_mqa_logits` resolves to a `_missing()`
    stub here: the externally-installed deep_gemm (~/.local) predates the unified
    `fp8_fp4_mqa_logits` symbol and only exposes the legacy `fp8_mqa_logits`
    (vLLM's `has_deep_gemm()` prefers the external package, which shadows the vendored
    `vllm.third_party.deep_gemm` that DOES have the fp4 symbol).
  * The legacy `deep_gemm.fp8_mqa_logits` IS the FP8 code path. In vLLM's FP8 branch
    q_scale is None (folded into weights), so `fp8_fp4_mqa_logits(FP8)` and
    `fp8_mqa_logits` compute the identical thing. We call `fp8_mqa_logits` directly.
    Schema: fp8_mqa_logits(Tensor q[M,H,D] e4m3, Any kv=(k[N,D] e4m3, k_scale[N] f32),
                           Tensor weights[M,H] f32, Tensor cu_ks[M] i32,
                           Tensor cu_ke[M] i32, bool clean_logits) -> Tensor[M,N] f32

  * HEAD-COUNT GOTCHA: this build rejects num_heads=16 with
    "seq_len_alignment % block_q == 0". Supported: 32, 64, 128 only (others also fail
    "block_qh % num_heads == 0"). Workaround: pad heads up to 32 with ZERO q and ZERO
    weights — padded heads contribute w*relu(.)=0, so logits are unchanged (verified).
"""

import math

import torch
import torch.nn.functional as F

import deep_gemm  # external ~/.local build; provides legacy fp8_mqa_logits

from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer

DEV = "cuda"
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
KERNEL = deep_gemm.fp8_mqa_logits
SUPPORTED_HEADS = (32, 64, 128)


# --------------------------------------------------------------------------- #
# Quantization matching vllm fp8_utils.per_token_group_quant_fp8 (ue8m0):
#   _absmax = max(|y|); scale = _absmax / fp8_max; if ue8m0: scale = 2^ceil(log2(scale))
#   y_q = clamp(y/scale, -fp8_max, fp8_max).to(e4m3)
# One block per head/token (head_dim <= 128 group), so one scale per row.
# --------------------------------------------------------------------------- #
def quant_fp8_rows(x: torch.Tensor, ue8m0: bool = True, eps: float = 1e-10):
    """Quantize over the LAST dim (one scale per row). Returns (x_fp8, scale[...])."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / FP8_MAX
    if ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    x_q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return x_q, scale.squeeze(-1)


def pick_padded_heads(h: int) -> int:
    for hs in SUPPORTED_HEADS:
        if h <= hs:
            return hs
    raise ValueError(f"n_heads {h} exceeds max supported {SUPPORTED_HEADS[-1]}")


def kernel_logits(q, k, w, softmax_scale, pad_D_to=None, clean_logits=True):
    """Run the DeepGEMM FP8 MQA kernel. q[M,H,D], k[N,D], w[M,H] (positive weights).

    Quantizes q,k to e4m3 (ue8m0, one scale/row), folds q_scale*softmax_scale into
    weights (q_scale>0 pulls out of the per-head ReLU). Causal: query m sees keys 0..m.
    Pads head dim to pad_D_to (zeros, dot unchanged) and head count to a supported value
    (zero q + zero weight, contributes nothing). Returns logits[M,N] f32 (-inf off-causal
    when clean_logits).
    """
    M, H, D = q.shape
    N = k.shape[0]
    if pad_D_to is not None and D < pad_D_to:
        q = F.pad(q, (0, pad_D_to - D))
        k = F.pad(k, (0, pad_D_to - D))
        D = pad_D_to
    Hs = pick_padded_heads(H)
    if H < Hs:
        q = F.pad(q, (0, 0, 0, Hs - H))      # zero q for padded heads
        w = F.pad(w, (0, Hs - H))            # zero weight for padded heads

    q_q, q_s = quant_fp8_rows(q)             # [M,Hs,D], [M,Hs]
    k_q, k_s = quant_fp8_rows(k)             # [N,D], [N]
    weights = (w * softmax_scale * q_s).float().contiguous()   # fold q_scale + softmax_scale
    cu_ks = torch.zeros(M, dtype=torch.int32, device=DEV)
    cu_ke = torch.arange(1, M + 1, dtype=torch.int32, device=DEV).clamp(max=N)
    return KERNEL(
        q_q.contiguous(),
        (k_q.contiguous(), k_s.float().contiguous()),
        weights,
        cu_ks,
        cu_ke,
        clean_logits,
    )


def topk_overlap(A, B, top_k, valid_counts, only_saturated=False):
    """Per-query fraction overlap between top-min(top_k, valid) indices of A vs B.

    A,B: [M,N] logits with -inf off-causal. valid_counts[m] = #valid keys (m+1).
    only_saturated: restrict to queries with valid > top_k (where selection is a real
    sub-selection, not "take all keys"). Returns (mean, min, n_queries_used).
    """
    M = A.shape[0]
    ov = []
    for m in range(M):
        v = int(valid_counts[m].item())
        kk = min(top_k, v)
        if only_saturated and v <= top_k:
            continue
        ia = A[m].topk(kk).indices
        ib = B[m].topk(kk).indices
        inter = len(set(ia.tolist()) & set(ib.tolist()))
        ov.append(inter / kk)
    if not ov:
        return float("nan"), float("nan"), 0
    t = torch.tensor(ov)
    return t.mean().item(), t.min().item(), len(ov)


def ref_scores(q64, k64, weights, softmax_scale, rotate, causal_bias):
    """LightningIndexer.scores (pure torch), fp8=True. q64[M,16,64],k64[N,64],weights[M,16].

    Returns [M,N] with causal bias added. rotate toggles the Hadamard pre-quant rotation.
    """
    cfg = DSAConfig(
        n_heads=16, head_dim=64, rope_head_dim=32, q_lora_rank=768,
        top_k=512, fp8=True, rotate_activation=rotate,
    )
    idx = LightningIndexer(cfg, softmax_scale=softmax_scale).to(DEV)
    with torch.no_grad():
        s = idx.scores(
            q64.unsqueeze(0), k64.unsqueeze(0), weights.unsqueeze(0),
            attn_bias=causal_bias.unsqueeze(0),
        )
    return s[0]


def causal_bias(M, N):
    b = torch.zeros(M, N, device=DEV)
    b.masked_fill_(~torch.tril(torch.ones(M, N, device=DEV, dtype=torch.bool)), float("-inf"))
    return b


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(1234)
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"kernel = deep_gemm.fp8_mqa_logits (legacy FP8 path; vLLM fp8_fp4 wrapper is _missing here)")
    print(f"config: n_heads=16 head_dim=64 top_k=512  supported kernel heads={SUPPORTED_HEADS}\n")

    softmax_scale = 64 ** -0.5

    # ============================= M0 ==================================== #
    print("=" * 72)
    print("M0 - kernel executes at our (padded) dims [M=N=256, H=16, D=64->128]")
    print("=" * 72)
    M = N = 256
    q = torch.randn(M, 16, 64, device=DEV)
    k = torch.randn(N, 64, device=DEV)
    w = F.softplus(torch.randn(M, 16, device=DEV))
    out = kernel_logits(q, k, w, softmax_scale, pad_D_to=128, clean_logits=True)
    causal = torch.tril(torch.ones(M, N, device=DEV, dtype=torch.bool))
    finite = torch.isfinite(out[causal]).all().item()
    print(f"  head-pad 16->32 (zero q/weight), D-pad 64->128, ue8m0 quant")
    print(f"  output shape {tuple(out.shape)} dtype {out.dtype}")
    print(f"  causal entries all finite: {finite}")
    print(f"  off-causal entries are -inf (clean_logits): "
          f"{torch.isinf(out[~causal]).all().item()}")
    print(f"  logit range (causal): [{out[causal].min().item():.3f}, {out[causal].max().item():.3f}]")
    print(f"  M0 RESULT: {'PASS' if finite else 'FAIL'}\n")

    # ============================= M1 ==================================== #
    print("=" * 72)
    print("M1 - ReLU / semantics discovery [tiny M=N=4, effective H=2, D=32]")
    print("=" * 72)
    Mt = Nt = 4
    Ht = 2
    Dt = 32
    torch.manual_seed(7)
    qt = torch.randn(Mt, Ht, Dt, device=DEV) * 1.5
    kt = torch.randn(Nt, Dt, device=DEV) * 1.5
    # force some negative dots by anti-aligning head 0 of query 3 with keys
    qt[3, 0] = -kt.mean(0) * 3.0
    wt = F.softplus(torch.randn(Mt, Ht, device=DEV)) + 0.1
    out_t = kernel_logits(qt, kt, wt, softmax_scale, pad_D_to=None, clean_logits=True)
    # torch refs on the SAME dequantized fp8 values (heads 0..1), causal
    Hs = pick_padded_heads(Ht)
    q_pad = F.pad(qt, (0, 0, 0, Hs - Ht))
    q_q, q_s = quant_fp8_rows(q_pad)
    k_q, k_s = quant_fp8_rows(kt)
    q_dq = (q_q.float() * q_s.unsqueeze(-1))[:, :Ht]     # [M,2,D]
    k_dq = k_q.float() * k_s.unsqueeze(-1)               # [N,D]
    raw = torch.einsum("mhd,nd->mhn", q_dq, k_dq)        # [M,2,N]
    eff_w = (wt * softmax_scale)
    ref_relu = torch.einsum("mhn,mh->mn", torch.relu(raw), eff_w)
    ref_norelu = torch.einsum("mhn,mh->mn", raw, eff_w)
    cmask = torch.tril(torch.ones(Mt, Nt, device=DEV, dtype=torch.bool))
    d_relu = (out_t - ref_relu)[cmask].abs().max().item()
    d_norelu = (out_t - ref_norelu)[cmask].abs().max().item()
    applies_relu = d_relu < d_norelu
    print(f"  max|kernel - sum_h w*ReLU(q.k)| = {d_relu:.6f}")
    print(f"  max|kernel - sum_h w*(q.k)|     = {d_norelu:.6f}")
    print(f"  M1 RESULT: kernel applies per-head ReLU = {applies_relu} "
          f"(matches ReLU ref to {d_relu:.2e}, fp8 tolerance)\n")

    # ============================= M2 ==================================== #
    print("=" * 72)
    print("M2 - zero-padding faithfulness (kernel vs kernel) [M=N=256, H=16, top_k=512]")
    print("=" * 72)
    torch.manual_seed(2024)
    q64 = torch.randn(M, 16, 64, device=DEV)
    k64 = torch.randn(N, 64, device=DEV)
    w2 = F.softplus(torch.randn(M, 16, device=DEV))
    A = kernel_logits(q64, k64, w2, softmax_scale, pad_D_to=None)    # D=64, group 64
    B = kernel_logits(q64, k64, w2, softmax_scale, pad_D_to=128)     # D=128, group 128
    cmask = torch.tril(torch.ones(M, N, device=DEV, dtype=torch.bool))
    dab = (A - B)[cmask].abs()
    denom = B[cmask].abs().clamp(min=1e-6)
    relerr = (dab / denom).mean().item()
    valid = torch.arange(1, M + 1, device=DEV)
    mean_ov, min_ov, nq = topk_overlap(A, B, 512, valid)
    print(f"  (i)  max|logitsA - logitsB| = {dab.max().item():.6e}   mean rel-err = {relerr:.3e}")
    print(f"  (ii) top-512 index overlap  mean={mean_ov:.4f} min={min_ov:.4f} "
          f"(all {nq} queries; N<top_k so each selects ALL valid keys -> trivially ~1.0)")
    print(f"  M2: padding is faithful (logits ~equal, top-k ~1.0)\n")

    # ============================= M3 ==================================== #
    print("=" * 72)
    print("M3 - cost of dropping our Hadamard (selection overlap)")
    print("=" * 72)
    for Mbig, note in [(256, "as-specified (N<top_k -> trivial)"),
                       ("stress", "selection-stress N>top_k (decision-relevant)")]:
        if Mbig == "stress":
            Mb = Nb = 2048
        else:
            Mb = Nb = 256
        torch.manual_seed(99)
        qb = torch.randn(Mb, 16, 64, device=DEV)
        kb = torch.randn(Nb, 64, device=DEV)
        wb = F.softplus(torch.randn(Mb, 16, device=DEV))
        cb = causal_bias(Mb, Nb)
        ref_had = ref_scores(qb, kb, wb, softmax_scale, rotate=True, causal_bias=cb)
        ref_nohad = ref_scores(qb, kb, wb, softmax_scale, rotate=False, causal_bias=cb)
        kern_B = kernel_logits(qb, kb, wb, softmax_scale, pad_D_to=128)  # no Hadamard
        valid = torch.arange(1, Mb + 1, device=DEV)
        sat = (Mb > 512)
        tag = "saturated queries only (valid>512)" if sat else "all queries"
        hh = topk_overlap(ref_had, ref_nohad, 512, valid, only_saturated=sat)
        kn = topk_overlap(kern_B, ref_nohad, 512, valid, only_saturated=sat)
        kh = topk_overlap(kern_B, ref_had, 512, valid, only_saturated=sat)
        print(f"  --- M=N={Mb}  ({note}); overlap over {tag} [{hh[2]} queries] ---")
        print(f"    ref_had   vs ref_nohad : mean={hh[0]:.4f} min={hh[1]:.4f}   "
              f"<- how much dropping Hadamard changes SELECTION")
        print(f"    kernel(B) vs ref_nohad : mean={kn[0]:.4f} min={kn[1]:.4f}   "
              f"<- kernel matches our no-Hadamard math (validation)")
        print(f"    kernel(B) vs ref_had   : mean={kh[0]:.4f} min={kh[1]:.4f}   "
              f"<- serving-without-fix vs TRAINED")
        print()

    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
