#!/usr/bin/env python3
"""Phase 0.5 - verify whether forcing vLLM's VENDORED deep_gemm is a viable,
non-destructive fix for the missing `fp8_fp4_mqa_logits` FP8 indexer kernel.

Run in the REAL serve env:
    cd /net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl
    export PYTHONPATH=$(pwd)/.devlibs/tf457lib:$(pwd)
    /usr/bin/python3 tests/dsa/check_vendored_deepgemm.py

Read-only: no uninstall / rm / move / rename, no PYTHONNOUSERSITE. In-process only.

Background: `vllm.utils.deep_gemm._import_deep_gemm()` tries top-level `import deep_gemm`
FIRST (external ~/.local, which lacks `fp8_fp4_mqa_logits`), then falls back to the
vendored `vllm.third_party.deep_gemm`. So the external copy shadows the vendored one and
vLLM's `fp8_fp4_mqa_logits` wrapper resolves to a `_missing()` stub.
"""

import sys
import traceback

import torch
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
DEV = "cuda"
SUPPORTED_HEADS = (32, 64, 128)


def quant_fp8_rows(x, ue8m0=True, eps=1e-10):
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scale = absmax / FP8_MAX
    if ue8m0:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    x_q = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return x_q, scale.squeeze(-1)


def pick_padded_heads(h):
    for hs in SUPPORTED_HEADS:
        if h <= hs:
            return hs
    raise ValueError(h)


def make_inputs(M=256, N=256, H=16, D=64, pad_D_to=128, seed=1234):
    """Return kernel-ready tensors (head-padded to 32, dim-padded to 128, ue8m0 quant).

    weights already fold q_scale * softmax_scale (q_scale > 0 pulls out of the per-head
    ReLU). Also returns a torch ReLU reference on the same dequantized fp8 values.
    """
    torch.manual_seed(seed)
    softmax_scale = D ** -0.5
    q = torch.randn(M, H, D, device=DEV)
    k = torch.randn(N, D, device=DEV)
    w = F.softplus(torch.randn(M, H, device=DEV))

    if pad_D_to and D < pad_D_to:
        q = F.pad(q, (0, pad_D_to - D))
        k = F.pad(k, (0, pad_D_to - D))
    Hs = pick_padded_heads(H)
    q = F.pad(q, (0, 0, 0, Hs - H))   # zero q for padded heads
    w = F.pad(w, (0, Hs - H))         # zero weight for padded heads

    q_q, q_s = quant_fp8_rows(q)      # [M,Hs,Dp], [M,Hs]
    k_q, k_s = quant_fp8_rows(k)      # [N,Dp], [N]
    weights = (w * softmax_scale * q_s).float().contiguous()
    cu_ks = torch.zeros(M, dtype=torch.int32, device=DEV)
    cu_ke = torch.arange(1, M + 1, dtype=torch.int32, device=DEV).clamp(max=N)

    # torch ReLU reference (heads 0..H-1) on the dequantized fp8 values
    q_dq = (q_q.float() * q_s.unsqueeze(-1))[:, :H]
    k_dq = k_q.float() * k_s.unsqueeze(-1)
    raw = torch.einsum("mhd,nd->mhn", q_dq, k_dq)
    ref = torch.einsum("mhn,mh->mn", torch.relu(raw), (w[:, :H] * softmax_scale))
    cmask = torch.tril(torch.ones(M, N, device=DEV, dtype=torch.bool))
    return q_q.contiguous(), k_q.contiguous(), k_s.float().contiguous(), weights, cu_ks, cu_ke, ref, cmask


def q1_vendored_import():
    print("=" * 72)
    print("Q1 - vendored import + symbol")
    print("=" * 72)
    try:
        import vllm.third_party.deep_gemm as vdg
    except Exception:
        print("  IMPORT RAISED (decisive negative):")
        traceback.print_exc()
        return None
    print(f"  vdg.__file__ = {vdg.__file__}")
    for sym in ("fp8_fp4_mqa_logits", "fp8_mqa_logits", "get_paged_mqa_logits_metadata"):
        print(f"  hasattr(vdg, {sym!r}) = {hasattr(vdg, sym)}")
    return vdg


def q2_vendored_runs(vdg):
    print("=" * 72)
    print("Q2 - vendored kernel actually runs (tuple signature)")
    print("=" * 72)
    if vdg is None or not hasattr(vdg, "fp8_fp4_mqa_logits"):
        print("  SKIP: no vendored fp8_fp4_mqa_logits")
        return
    q_q, k_q, k_s, weights, cu_ks, cu_ke, ref, cmask = make_inputs()
    try:
        out = vdg.fp8_fp4_mqa_logits(
            (q_q, None),           # FP8 path: q_scale None (folded into weights)
            (k_q, k_s),
            weights, cu_ks, cu_ke,
            clean_logits=True,
        )
    except Exception:
        print("  KERNEL CALL RAISED:")
        traceback.print_exc()
        return
    finite = torch.isfinite(out[cmask]).all().item()
    dmax = (out - ref)[cmask].abs().max().item()
    print(f"  out shape {tuple(out.shape)} dtype {out.dtype}")
    print(f"  causal entries finite: {finite}")
    print(f"  logit range (causal): [{out[cmask].min().item():.3f}, {out[cmask].max().item():.3f}]")
    print(f"  max|vendored - torch ReLU ref| = {dmax:.6f} (fp8 tolerance)")
    print(f"  Q2 RESULT: {'PASS - vendored copy is complete & runs' if finite and dmax < 0.05 else 'FAIL'}")


class _BlockExternalDeepGemm:
    """meta_path finder that makes ONLY top-level `import deep_gemm` fail.

    Leaves sys.path and ~/.local untouched, so flashinfer / fast_hadamard_transform /
    cupy / sgl_kernel remain importable. `vllm.third_party.deep_gemm` (dotted) is not
    matched, so the vendored copy still imports.
    """
    def find_spec(self, name, path, target=None):
        if name == "deep_gemm" or name.startswith("deep_gemm."):
            raise ImportError(f"blocked external {name} (Phase 0.5 vendored-fallback test)")
        return None


def q3_wrapper_resolves():
    print("=" * 72)
    print("Q3 - vLLM wrapper resolves to vendored when external is hidden (in-process)")
    print("=" * 72)
    import vllm.utils.deep_gemm as dg

    dg._lazy_init()
    print(f"  BEFORE fix: _fp8_fp4_mqa_logits_impl is None = "
          f"{dg._fp8_fp4_mqa_logits_impl is None}")

    # purge cached external deep_gemm, install blocker, reset vLLM cached globals
    for m in [m for m in sys.modules if m == "deep_gemm" or m.startswith("deep_gemm.")]:
        del sys.modules[m]
    sys.meta_path.insert(0, _BlockExternalDeepGemm())
    try:
        import importlib
        importlib.import_module("deep_gemm")
        print("  WARNING: external deep_gemm still importable (blocker ineffective)")
    except ImportError as e:
        print(f"  external `import deep_gemm` now raises: {e}")

    # reset every cached *_impl global to force re-resolution
    for name in list(vars(dg)):
        if name.endswith("_impl"):
            setattr(dg, name, None)
    dg._lazy_init()
    resolved = dg._fp8_fp4_mqa_logits_impl is not None
    print(f"  AFTER fix:  _fp8_fp4_mqa_logits_impl is None = {not resolved}")
    if resolved:
        print(f"  resolved impl module = {getattr(dg._fp8_fp4_mqa_logits_impl, '__module__', '?')}")

    # prove the public wrapper now runs instead of _missing()
    try:
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
        q_q, k_q, k_s, weights, cu_ks, cu_ke, ref, cmask = make_inputs()
        out = fp8_fp4_mqa_logits((q_q, None), (k_q, k_s), weights, cu_ks, cu_ke, clean_logits=True)
        dmax = (out - ref)[cmask].abs().max().item()
        ok = torch.isfinite(out[cmask]).all().item() and dmax < 0.05
        print(f"  public wrapper call: shape {tuple(out.shape)}  "
              f"max|-ReLU ref|={dmax:.6f}  {'OK' if ok else 'BAD'}")
    except Exception:
        print("  public wrapper call RAISED:")
        traceback.print_exc()
        resolved = False
    print(f"  Q3 RESULT: {'PASS - blocking external -> vLLM uses vendored end-to-end' if resolved else 'FAIL'}")


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  "
          f"{torch.cuda.get_device_name(0)}")
    import vllm
    print(f"vllm {vllm.__version__}\n")
    vdg = q1_vendored_import()
    print()
    q2_vendored_runs(vdg)
    print()
    q3_wrapper_resolves()


if __name__ == "__main__":
    main()
