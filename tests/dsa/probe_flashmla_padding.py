#!/usr/bin/env python3
"""
Feasibility probe: can we REUSE vLLM 0.20.2's FlashMLA *sparse* kernel for
MiniCPM3 by ZERO-PADDING MiniCPM3's MLA dims up to DeepSeek's (576 / 512),
instead of writing a custom kernel?

Run:
  cd <repo>
  export PYTHONPATH=$(pwd)/.devlibs/tf457lib:$(pwd)
  /usr/bin/python3 tests/dsa/probe_flashmla_padding.py

The most-isolated real callable for the BF16 sparse path is
`flash_mla_sparse_fwd` (see
  vllm/v1/attention/backends/mla/flashmla_sparse.py:983-1012  (_bf16_flash_mla_kernel)
  vllm/third_party/flashmla/flash_mla_interface.py:180-217   (flash_mla_sparse_fwd)
). It is the "absorbed MQA 576/512" form:
    q  : [s_q,  h_q, d_qk]  bf16
    kv : [s_kv, h_kv=1, d_qk] bf16   (single shared latent = [kv_c ; k_pe])
    indices : [s_q, h_kv=1, topk] int32   (direct offsets into s_kv)
    sm_scale : float (runtime arg)
    d_v : 512  (value = first d_v dims of kv)
No paged cache / block table / fp8 needed -> ideal for an exactness probe.

MiniCPM3 MLA (native):
    kv_lora_rank (NoPE / value latent) = 256
    qk_rope_head_dim (RoPE)            = 32
    -> d_qk_native = 288, d_v_native = 256
    softmax_scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5 = 288 ** -0.5

Padding scheme (all appended dims are ZEROS):
    kv latent 576 = [ kv_c(256) | zeros(256) | k_pe(32) | zeros(32) ]
    q     per head = [ q_nope(256)| zeros(256) | q_pe(32) | zeros(32) ]
    value = first 512 dims -> [ weighted_sum(kv_c)(256) | zeros(256) ]
Take first 256 dims of the 512-d kernel output and compare to a pure-torch
MiniCPM3-dim reference.
"""

import torch

from vllm.third_party.flashmla.flash_mla_interface import flash_mla_sparse_fwd

torch.manual_seed(0)

# ----- MiniCPM3 MLA dims -----
KV_LORA = 256          # NoPE latent == value latent (d_v native)
ROPE = 32              # qk_rope_head_dim
D_QK_NATIVE = KV_LORA + ROPE          # 288
NATIVE_SCALE = (KV_LORA + ROPE) ** -0.5  # 288**-0.5  (MiniCPM3 real per-head scale)

# ----- DeepSeek / kernel-native dims -----
D_QK_PAD = 576         # 512 NoPE + 64 RoPE
D_V_PAD = 512
NOPE_PAD = 512
ROPE_PAD = 64

DEV = "cuda"
DT = torch.bfloat16


def build_padded_kv(kv_c, k_pe):
    """[N,256]+[N,32] -> [N,576] = [kv_c | 0*256 | k_pe | 0*32]."""
    N = kv_c.shape[0]
    out = torch.zeros(N, D_QK_PAD, device=DEV, dtype=DT)
    out[:, 0:KV_LORA] = kv_c
    out[:, NOPE_PAD:NOPE_PAD + ROPE] = k_pe
    return out


def build_padded_q(q_nope, q_pe):
    """[H,256]+[H,32] -> [H,576]."""
    H = q_nope.shape[0]
    out = torch.zeros(H, D_QK_PAD, device=DEV, dtype=DT)
    out[:, 0:KV_LORA] = q_nope
    out[:, NOPE_PAD:NOPE_PAD + ROPE] = q_pe
    return out


def native_reference(q_nope, q_pe, kv_c, k_pe, topk_idx, scale):
    """Pure-torch MiniCPM3-dim absorbed MLA MQA over the SELECTED topk keys.

    q_nope [H,256], q_pe [H,32], kv_c [N,256], k_pe [N,32],
    topk_idx [K] int, scale float.  Done in fp32 for a clean ground truth.
    Returns [H,256] attention output (value latent).
    """
    q_nope = q_nope.float()
    q_pe = q_pe.float()
    kv_c = kv_c.float()
    k_pe = k_pe.float()
    sel_kv_c = kv_c[topk_idx]   # [K,256]
    sel_k_pe = k_pe[topk_idx]   # [K,32]
    # scores [H,K] = q_nope . kv_c + q_pe . k_pe
    scores = q_nope @ sel_kv_c.T + q_pe @ sel_k_pe.T
    scores = scores * scale
    w = torch.softmax(scores, dim=-1)          # [H,K]
    out = w @ sel_kv_c                          # [H,256]
    return out


def run_kernel(q_pad, kv_pad, topk_idx, scale, h_pad=None):
    """q_pad [H,576], kv_pad [N,576], topk_idx [K] -> [H,512] (first h heads).

    Pads head count to h_pad (multiple of 64 on Hopper) with ZERO q rows if
    requested, then slices the real heads back off.
    """
    H = q_pad.shape[0]
    K = topk_idx.shape[0]
    if h_pad is not None and h_pad > H:
        qp = torch.zeros(h_pad, D_QK_PAD, device=DEV, dtype=DT)
        qp[:H] = q_pad
        q_pad = qp
        Heff = h_pad
    else:
        Heff = H
    q = q_pad.view(1, Heff, D_QK_PAD)                       # [s_q=1, h_q, d_qk]
    kv = kv_pad.view(-1, 1, D_QK_PAD)                       # [s_kv, h_kv=1, d_qk]
    idx = topk_idx.to(torch.int32).view(1, 1, K)            # [s_q, h_kv=1, topk]
    out, max_logits, lse = flash_mla_sparse_fwd(q, kv, idx, float(scale), d_v=D_V_PAD)
    out = out[0, :H, :]                                    # [H,512]
    return out


def summarize(tag, ref, got):
    ref = ref.float()
    got = got.float()
    abs_err = (ref - got).abs()
    max_abs = abs_err.max().item()
    denom = ref.abs().max().clamp_min(1e-6)
    rel = (max_abs / denom).item()
    # argmax agreement across the value-latent dim, per head
    arg_ref = ref.argmax(dim=-1)
    arg_got = got.argmax(dim=-1)
    top1 = (arg_ref == arg_got).float().mean().item()
    print(f"  [{tag}] max_abs_err={max_abs:.4e}  rel_err={rel:.4e}  "
          f"argmax_agreement={top1*100:.1f}%")
    return max_abs, rel, top1


def main():
    print("=" * 78)
    print("FlashMLA pad-to-reuse probe  (BF16 sparse path: flash_mla_sparse_fwd)")
    print(f"device={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}")
    print("=" * 78)

    N_KV = 512      # past keys
    H = 40          # MiniCPM3 num attention heads
    K = 128         # topk selected keys (kernel needs topk % (2*B_TOPK) == 0)

    # random MiniCPM3-dim inputs (bf16, like the real model activations)
    q_nope = torch.randn(H, KV_LORA, device=DEV, dtype=DT) * 0.5
    q_pe = torch.randn(H, ROPE, device=DEV, dtype=DT) * 0.5
    kv_c = torch.randn(N_KV, KV_LORA, device=DEV, dtype=DT) * 0.5
    k_pe = torch.randn(N_KV, ROPE, device=DEV, dtype=DT) * 0.5

    # pick K distinct topk indices in [0, N_KV)
    topk_idx = torch.randperm(N_KV, device=DEV)[:K].sort().values

    # ---- reference ----
    ref = native_reference(q_nope, q_pe, kv_c, k_pe, topk_idx, NATIVE_SCALE)  # [H,256]

    # ---- padded kernel ----
    q_pad = build_padded_q(q_nope, q_pe)
    kv_pad = build_padded_kv(kv_c, k_pe)

    results = {}

    print("\n[TEST 1] Exactness: padded kernel (h padded 40->64) vs native ref")
    got = run_kernel(q_pad, kv_pad, topk_idx, NATIVE_SCALE, h_pad=64)
    got256 = got[:, :KV_LORA]
    tail = got[:, KV_LORA:].abs().max().item()
    print(f"  output tail (dims 256:512) max_abs = {tail:.4e}  "
          f"(should be ~0: zero-padded value dims)")
    results["exact"] = summarize("exact", ref, got256)

    # ---- GOTCHA A: is softmax_scale a real runtime arg? ----
    print("\n[GOTCHA A] softmax_scale runtime arg? (vary scale, expect ref to track)")
    alt_scale = NATIVE_SCALE * 2.0
    ref_alt = native_reference(q_nope, q_pe, kv_c, k_pe, topk_idx, alt_scale)
    got_alt = run_kernel(q_pad, kv_pad, topk_idx, alt_scale, h_pad=64)[:, :KV_LORA]
    ma_default, _, _ = summarize("scale=native", ref, run_kernel(q_pad, kv_pad, topk_idx, NATIVE_SCALE, h_pad=64)[:, :KV_LORA])
    ma_alt, _, _ = summarize("scale=2x -> ref(2x)", ref_alt, got_alt)
    # cross-check: kernel with 2x scale should DISAGREE with ref at 1x scale
    _, _, _ = summarize("scale=2x -> ref(1x) [expect BIG]", ref, got_alt)
    scale_is_runtime = ma_alt < 5e-2 and ma_alt < 10 * ma_default
    print(f"  => softmax_scale is a used runtime arg: {scale_is_runtime}")
    results["scale_runtime"] = scale_is_runtime

    # ---- GOTCHA B: does kernel rotate/misuse the padded rope dims? ----
    # If the kernel applied RoPE internally or treated the last 64 dims
    # specially, zeros in dims [544:576] and [256:512] would leak into the
    # output. Test: zero out q_pe entirely -> score should reduce to pure
    # q_nope.kv_c; compare against a native ref computed the same way.
    print("\n[GOTCHA B] internal RoPE / positional split abuse of padded dims?")
    q_pe0 = torch.zeros_like(q_pe)
    ref_norope = native_reference(q_nope, q_pe0, kv_c, k_pe, topk_idx, NATIVE_SCALE)
    q_pad_norope = build_padded_q(q_nope, q_pe0)
    got_norope = run_kernel(q_pad_norope, kv_pad, topk_idx, NATIVE_SCALE, h_pad=64)[:, :KV_LORA]
    ma_b, _, _ = summarize("q_pe=0 matches pure-nope ref", ref_norope, got_norope)
    no_internal_rope = ma_b < 5e-2
    print(f"  => no internal RoPE / no positional abuse of padded dims: {no_internal_rope}")
    results["no_internal_rope"] = no_internal_rope

    # ---- GOTCHA C: head padding with zero q affects real heads? ----
    print("\n[GOTCHA C] head-count padding 40->64 disturbs real heads?")
    # compare real-head output with vs without padding (need h multiple of 64)
    got_pad64 = run_kernel(q_pad, kv_pad, topk_idx, NATIVE_SCALE, h_pad=64)[:, :KV_LORA]
    # also run at exactly 64 real heads to confirm kernel accepts 64 natively
    ma_c, _, _ = summarize("padded-heads real output vs ref", ref, got_pad64)
    head_pad_ok = ma_c < 5e-2
    print(f"  => zero-q head padding leaves real heads correct: {head_pad_ok}")
    results["head_pad_ok"] = head_pad_ok

    # ---- GOTCHA D: does kernel REQUIRE h_q multiple of 64? ----
    print("\n[GOTCHA D] does kernel reject non-multiple-of-64 head counts?")
    try:
        _ = run_kernel(q_pad, kv_pad, topk_idx, NATIVE_SCALE, h_pad=None)  # H=40
        print("  h_q=40 (not mult of 64) ACCEPTED by kernel")
        results["requires_h64"] = False
    except Exception as e:
        msg = str(e).splitlines()[0][:160]
        print(f"  h_q=40 REJECTED -> must pad to 64: {msg}")
        results["requires_h64"] = True

    # ---- GOTCHA E: does kernel REQUIRE d_qk == 576 / d_v == 512? ----
    print("\n[GOTCHA E] does kernel reject native d_qk=288 (no pad)?")
    try:
        qn = q_nope  # [H,256] -> build [H,288]
        q_native = torch.zeros(64, D_QK_NATIVE, device=DEV, dtype=DT)
        q_native[:H, :KV_LORA] = q_nope
        q_native[:H, KV_LORA:] = q_pe
        kv_native = torch.zeros(N_KV, 1, D_QK_NATIVE, device=DEV, dtype=DT)
        kv_native[:, 0, :KV_LORA] = kv_c
        kv_native[:, 0, KV_LORA:] = k_pe
        idx = topk_idx.to(torch.int32).view(1, 1, K)
        _ = flash_mla_sparse_fwd(q_native.view(1, 64, D_QK_NATIVE), kv_native,
                                 idx, float(NATIVE_SCALE), d_v=KV_LORA)
        print("  native d_qk=288/d_v=256 ACCEPTED (kernel is dim-flexible!)")
        results["requires_576"] = False
    except Exception as e:
        msg = str(e).splitlines()[-1][:200]
        print(f"  native dims REJECTED -> kernel is 576/512-locked, pad required:")
        print(f"    {msg}")
        results["requires_576"] = True

    # ---- cost note ----
    print("\n[COST] padded latent 576 vs native 288 = "
          f"{576/288:.2f}x d_qk;  value 512 vs 256 = {512/256:.2f}x d_v")
    print("  => ~2x KV-cache bytes per token and ~2x per-selected-key MAC compute.")

    # ---- verdict ----
    print("\n" + "=" * 78)
    max_abs, rel, top1 = results["exact"]
    # NOTE: argmax-agreement (top1) is over the 256-d value latent, which has
    # many near-tied dims; a single bf16 near-tie flip drops it to 97.5%. It is
    # informational only. The load-bearing criterion is the error magnitude.
    exact_pass = max_abs < 2e-2 and rel < 2e-2
    print(f"EXACTNESS: max_abs_err={max_abs:.4e} rel_err={rel:.4e} "
          f"argmax={top1*100:.1f}%  -> {'PASS' if exact_pass else 'FAIL'}")
    print("GOTCHAS:")
    print(f"  A scale is runtime arg (MiniCPM3 scale usable): {results['scale_runtime']}")
    print(f"  B no internal RoPE / no positional abuse:       {results['no_internal_rope']}")
    print(f"  C zero-q head padding safe for real heads:      {results['head_pad_ok']}")
    print(f"  D kernel requires h_q multiple of 64:           {results['requires_h64']}")
    print(f"  E kernel locked to d_qk=576/d_v=512:            {results['requires_576']}")
    viable = exact_pass and results["scale_runtime"] and results["no_internal_rope"] and results["head_pad_ok"]
    print("-" * 78)
    print(f"PAD-TO-REUSE VERDICT: {'VIABLE' if viable else 'NOT VIABLE'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
