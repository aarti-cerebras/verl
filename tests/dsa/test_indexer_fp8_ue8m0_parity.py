#!/usr/bin/env python3
"""CPU parity test: training-side UE8M0 fake-quant == serving-side UE8M0 quant.

The training indexer's ``_fake_quant_fp8`` (verl/models/transformers/dsa_indexer.py) historically used a
*continuous* per-row absmax scale, while the serving FLASHMLA_SPARSE kernel quantizes with a *power-of-2*
(UE8M0) scale (scripts/dsa/vllm_minicpm3_dsa/indexer.py::_quant_fp8_rows). That mismatch is the ~2%
train/serve indexer-selection drift documented in docs/dsa_eval_report.md §5.

With ``DSAConfig.fp8_ue8m0=True`` the training fake-quant rounds the scale to a power of two, so the
dequantized E4M3 round-trip it trains against must be *bit-identical* to the serve quant's dequantization.
This test asserts exactly that against the REAL serve function (imported, not re-implemented), plus:
  * legacy (use_ue8m0=False) still uses the continuous scale (regression guard — old runs reproduce);
  * the straight-through estimator passes identity gradient in both modes.

Pure torch, CPU, deterministic. Run:
    PYTHONPATH=.devlibs/tf457lib:. python3 tests/dsa/test_indexer_fp8_ue8m0_parity.py
or: pytest tests/dsa/test_indexer_fp8_ue8m0_parity.py
"""

import torch

from scripts.dsa.vllm_minicpm3_dsa.indexer import _quant_fp8_rows  # serve UE8M0 quant (real)
from verl.models.transformers.dsa_indexer import _fake_quant_fp8  # train fake-quant (STE)

SEED = 1234


def _serve_dequant(x, use_ue8m0=True):
    """Dequantized round-trip of the serve quant: x_fp8.float() * scale (matches what the kernel consumes)."""
    x_fp8, scale = _quant_fp8_rows(x, use_ue8m0=use_ue8m0)  # scale has the last (group) dim squeezed off
    return x_fp8.float() * scale.unsqueeze(-1)


def test_ue8m0_matches_serve_exactly():
    """train _fake_quant_fp8(use_ue8m0=True) forward == serve _quant_fp8_rows(use_ue8m0=True) dequant."""
    torch.manual_seed(SEED)
    # indexer rows: [b, s, head_dim]; quant is per-row over the last dim. Varied magnitudes stress the scale.
    for shape in [(2, 2048, 64), (1, 512, 128), (4, 33, 64)]:
        for mag in (0.01, 1.0, 137.0):
            x = torch.randn(*shape) * mag
            train = _fake_quant_fp8(x, use_ue8m0=True)  # STE forward value == x_q (dequantized)
            serve = _serve_dequant(x, use_ue8m0=True)
            assert torch.equal(train, serve), (
                f"UE8M0 mismatch shape={shape} mag={mag}: "
                f"max|Δ|={(train - serve).abs().max().item():.3e}"
            )
    print("[ok] train UE8M0 fake-quant is bit-identical to serve UE8M0 quant")


def test_legacy_is_continuous_scale():
    """use_ue8m0=False must keep the legacy continuous absmax scale (so prior Phase-2 runs reproduce)."""
    torch.manual_seed(SEED)
    x = torch.randn(2, 256, 64) * 3.3  # amax/448 generically NOT a power of two
    legacy = _fake_quant_fp8(x, use_ue8m0=False)
    ue8m0 = _fake_quant_fp8(x, use_ue8m0=True)
    # explicit legacy reference: round(x/scale)*scale with scale = amax/FP8_MAX (no pow2 rounding)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / float(torch.finfo(torch.float8_e4m3fn).max)
    ref = (x / scale).to(torch.float8_e4m3fn).float() * scale
    assert torch.equal(legacy, ref), "legacy path changed — it must stay the continuous-scale round-trip"
    assert not torch.equal(legacy, ue8m0), "UE8M0 must differ from legacy on non-power-of-2 scales"
    print("[ok] legacy (use_ue8m0=False) unchanged; distinct from UE8M0")


def test_ste_identity_gradient():
    """STE: gradient of the fake-quant output w.r.t. input is identity in both modes."""
    for use_ue8m0 in (False, True):
        x = (torch.randn(3, 64) * 2.0).requires_grad_(True)
        y = _fake_quant_fp8(x, use_ue8m0=use_ue8m0)
        y.sum().backward()
        assert torch.equal(x.grad, torch.ones_like(x)), f"STE broken (use_ue8m0={use_ue8m0})"
    print("[ok] STE passes identity gradient in both modes")


def main():
    print(f"torch {torch.__version__}  (CPU parity)")
    test_ue8m0_matches_serve_exactly()
    test_legacy_is_continuous_scale()
    test_ste_identity_gradient()
    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
