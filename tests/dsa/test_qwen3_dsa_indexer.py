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
"""Unit tests for the Qwen3 GQA DSA lightning indexer (CPU only, no model download).

These are the step-1 gates from docs/qwen3_4b_dsa/plan_v2.md §7. The load-bearing ones:

* ``test_init_entropy_is_equal_across_layers`` — the §2.4 gate. Qwen3-4B's ``input_layernorm`` gain RMS
  spans 186x across its 36 layers, so a *constant* ``weights_proj`` init leaves early layers with scores of
  essentially zero (hence ~100x less gradient into ``wq``/``wk``, which are fed only through the gate) and
  late layers already committed. This asserts the per-layer init actually equalizes them.
* ``test_wq_wk_scale_is_forward_invisible`` — pins the claim the init recipe rests on: with a norm on both
  q and k, those two matrices' init std controls gradient geometry, NOT the forward score scale.
* ``test_serve_side_zero_pad_is_lossless`` — stock DeepGEMM takes ``head_dim=128`` and head counts in
  {32, 64, 128}, so a 16x64 indexer is zero-padded at serve time. This is the invariant that makes that
  padding safe, and it is exactly where train/serve drift would hide.
"""

import math

import pytest
import torch

from verl.models.transformers.qwen3_dsa_indexer import (
    _RELU_STD,
    NO_DECAY_SUFFIXES,
    Qwen3DSAConfig,
    Qwen3DSAIndexer,
    _fake_quant_fp8,
    _rotate_activation,
)

HIDDEN = 2560

# Measured on /cb/ml-eng/aarti/models/qwen3_4b_thinking_2507: rms(input_layernorm.weight) per layer, which
# is (approximately) the RMS of the hidden states the indexer reads. Min 0.025 (layer 0), max 4.709
# (layer 34) -> a 186x spread that the init has to absorb.
QWEN3_4B_GAIN_RMS = (0.025, 0.122, 0.331, 0.623, 1.095, 2.298, 4.709)


def _cfg(**kw) -> Qwen3DSAConfig:
    base = dict(enabled=True, hidden_size=HIDDEN, num_heads=32, num_kv_heads=8, rope_theta=5e6)
    base.update(kw)
    return Qwen3DSAConfig(**base)


def _causal_bias(t: int) -> torch.Tensor:
    keys = torch.arange(t)
    return torch.zeros(1, t, t).masked_fill(keys[None, None, :] > keys[None, :, None], float("-inf"))


# ---------------------------------------------------------------------------------------------------
# initialization (plan_v2.md §2.4)
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("hidden_rms", QWEN3_4B_GAIN_RMS)
def test_init_entropy_is_equal_across_layers(hidden_rms):
    """Per-layer init must put every layer at ``entropy_frac`` in [0.99, 1.0] and ``std_s(I) ~ sigma_target``.

    This is the Phase-1 smoke gate. Without the ``1/hidden_rms`` factor the same assertion fails at both
    ends of the model: ``entropy_frac`` would be 1.0000 at layer 0 (scores ~0, so no gradient reaches
    ``wq``/``wk``) and 0.909 at layer 34 (already committed).
    """
    torch.manual_seed(0)
    seq = 2048
    cfg = _cfg()
    ix = Qwen3DSAIndexer(cfg, hidden_rms=hidden_rms)
    x = torch.randn(1, seq, HIDDEN) * hidden_rms
    with torch.no_grad():
        scores = ix(x, torch.arange(seq).unsqueeze(0))

    sigmas, fracs = [], []
    for t in (seq // 4, seq // 2, 3 * seq // 4, seq - 1):
        row = scores[0, t, : t + 1].float()  # causal support only
        sigmas.append(row.std().item())
        probs = torch.softmax(row, dim=-1)
        fracs.append((torch.special.entr(probs).sum() / math.log(t + 1)).item())

    sigma, frac = sum(sigmas) / len(sigmas), sum(fracs) / len(fracs)
    # sigma_target=0.3 with generous slack: the estimate assumes isotropic hidden states and Gaussian
    # scores, and FP8 quantization adds noise. What must hold is that it does not drift WITH hidden_rms.
    assert 0.5 * cfg.sigma_target < sigma < 2.0 * cfg.sigma_target, f"sigma_I={sigma:.3f} at rms={hidden_rms}"
    assert 0.99 <= frac <= 1.0, f"entropy_frac={frac:.4f} at rms={hidden_rms}"


def test_weights_proj_std_matches_the_closed_form():
    """``std_W = sigma_target / (_RELU_STD * sqrt(hidden) * hidden_rms)``, i.e. the inherited constant
    ``0.5/sqrt(hidden)`` divided by the layer's gain RMS. head_dim and n_heads cancel out."""
    cfg = _cfg(sigma_target=0.3)
    ix = Qwen3DSAIndexer(cfg, hidden_rms=1.0)
    for rms in QWEN3_4B_GAIN_RMS:
        expected = 0.3 / (_RELU_STD * HIDDEN**0.5 * rms)
        assert ix.weights_proj_std(rms) == pytest.approx(expected, rel=1e-6)
    # head_dim must not enter the formula (the softmax_scale cancels the score's sqrt(head_dim))
    a = Qwen3DSAIndexer(_cfg(head_dim=64, rope_head_dim=64), hidden_rms=1.0).weights_proj_std(1.0)
    b = Qwen3DSAIndexer(_cfg(head_dim=128, rope_head_dim=128), hidden_rms=1.0).weights_proj_std(1.0)
    assert a == pytest.approx(b, rel=1e-9)


def test_wq_wk_scale_is_forward_invisible():
    """Scaling ``wq``/``wk`` must not change the scores: RMSNorm and LayerNorm are scale-invariant.

    This is *why* those two inits control only gradient geometry. The residual is not zero for two
    understood reasons, both bounded here: the norms' ``eps`` becomes non-negligible once the pre-norm
    variance is shrunk (``k_norm`` eps 1e-6 against a pre-norm variance of ~0.0025 is 4e-4 relative), and
    FP8's power-of-2 scale realigns the quantization grid when the row magnitude moves.
    """
    torch.manual_seed(0)
    x, pos = torch.randn(1, 256, HIDDEN), torch.arange(256).unsqueeze(0)

    for fp8, tol in ((False, 2e-3), (True, 5e-2)):
        ix = Qwen3DSAIndexer(_cfg(fp8=fp8, serving_compat=fp8), hidden_rms=1.0)
        with torch.no_grad():
            before = ix(x, pos)
            ix.wq.weight.mul_(10.0)
            ix.wk.weight.mul_(0.1)
            after = ix(x, pos)
        rel = ((before - after).abs().max() / before.abs().max()).item()
        assert rel < tol, f"fp8={fp8}: relative change {rel:.2e} exceeds {tol}"


def test_gradient_reaches_every_parameter():
    """``wq``/``wk`` are fed gradient ONLY through the gate, so a zero ``weights_proj`` would silently
    train nothing (docs/dsa_grad_norm_debugging.md issue #2). Also covers the FP8 straight-through
    estimator: without it the ``.to(float8)`` cast would block gradient entirely."""
    torch.manual_seed(0)
    ix = Qwen3DSAIndexer(_cfg(), hidden_rms=1.0)
    ix(torch.randn(1, 128, HIDDEN), torch.arange(128).unsqueeze(0)).square().mean().backward()
    for name, p in ix.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad has non-finite values"
        assert p.grad.norm().item() > 0.0, f"{name} grad is exactly zero"


def test_no_decay_suffixes_cover_the_right_parameters():
    """The norms and the gate must be excluded from weight decay: decay drives the gains toward 0 (killing
    the branch) and ``weights_proj`` toward 0 (severing the only path to ``wq``/``wk``)."""
    ix = Qwen3DSAIndexer(_cfg(), hidden_rms=1.0)
    names = {n for n, _ in ix.named_parameters()}
    excluded = {n for n in names if n.endswith(NO_DECAY_SUFFIXES)}
    assert excluded == {"q_norm.weight", "k_norm.weight", "k_norm.bias", "weights_proj.weight"}
    assert names - excluded == {"wq.weight", "wk.weight"}  # only the two projections keep decay


# ---------------------------------------------------------------------------------------------------
# rotary
# ---------------------------------------------------------------------------------------------------


def test_own_rotary_equals_base_even_indexed_frequencies():
    """The indexer's own ``rope_head_dim``-wide rotary at the base ``theta`` recovers exactly the base
    128-dim rope's even-indexed frequencies — so it spans the same spectrum as the attention it distills,
    which is the justification for building it fresh instead of reusing the base tables."""
    theta = 5e6
    base = 1.0 / (theta ** (torch.arange(0, 128, 2, dtype=torch.float64) / 128))  # 64 freqs
    ours = Qwen3DSAIndexer(_cfg(head_dim=64, rope_head_dim=64), hidden_rms=1.0).rotary.inv_freq.double()
    assert len(ours) == 32
    torch.testing.assert_close(ours, base[::2], rtol=1e-6, atol=0)


def test_base_cos_slice_would_be_wrong():
    """Guards the trap the docstring warns about: slicing the base model's 128-wide tables down to 64 is
    NOT the same as a 64-wide rope, because ``rotate_half`` needs the frequencies duplicated. If this ever
    starts passing, someone has 'simplified' the rotary into a correctness bug."""
    theta = 5e6
    base = 1.0 / (theta ** (torch.arange(0, 128, 2, dtype=torch.float64) / 128))
    ours = Qwen3DSAIndexer(_cfg(head_dim=64, rope_head_dim=64), hidden_rms=1.0).rotary.inv_freq.double()
    assert not torch.allclose(ours, base[:32], rtol=1e-3)


def test_rope_preserves_norms():
    """RoPE is orthogonal, so it must not change ``|q|`` or ``|k|`` — the premise of the init derivation."""
    torch.manual_seed(0)
    ix = Qwen3DSAIndexer(_cfg(fp8=False, serving_compat=False), hidden_rms=1.0)
    x, pos = torch.randn(1, 64, HIDDEN), torch.arange(64).unsqueeze(0)
    q, k, _ = ix(x, pos, return_projection=True)
    # post-norm, pre-rope magnitudes: RMSNorm gives unit RMS per row, LayerNorm unit variance
    assert q.norm(dim=-1).mean().item() == pytest.approx(ix.head_dim**0.5, rel=0.05)
    assert k.norm(dim=-1).mean().item() == pytest.approx(ix.head_dim**0.5, rel=0.05)


# ---------------------------------------------------------------------------------------------------
# scores and selection
# ---------------------------------------------------------------------------------------------------


def test_topk_equals_causal_set_when_k_covers_the_sequence():
    """Precondition for the Phase-2 M0 parity test: at ``top_k >= T`` the selected set is the whole causal
    support, so sparse attention must reduce exactly to dense attention."""
    torch.manual_seed(0)
    seq = 64
    ix = Qwen3DSAIndexer(_cfg(top_k=4096), hidden_rms=1.0)
    with torch.no_grad():
        scores = ix(torch.randn(1, seq, HIDDEN), torch.arange(seq).unsqueeze(0), attn_bias=_causal_bias(seq))
        idx = ix.select_topk(scores, top_k=seq)
    for t in range(seq):
        assert set(idx[0, t, : t + 1].tolist()) == set(range(t + 1)), f"row {t} is not the causal set"


def test_topk_selection_is_deterministic():
    """Selection must be bit-identical across repeated runs on the same input. Keye replaced ``torch.topk``
    with ``flashinfer.topk`` precisely because tie-breaking drift shows up as train/serve mismatch, and with
    2048 of 32768 selected there are near-ties in every row."""
    torch.manual_seed(0)
    seq = 512
    ix = Qwen3DSAIndexer(_cfg(top_k=128), hidden_rms=1.0)
    x, pos, bias = torch.randn(1, seq, HIDDEN), torch.arange(seq).unsqueeze(0), _causal_bias(seq)
    with torch.no_grad():
        first = ix.select_topk(ix(x, pos, attn_bias=bias))
        for _ in range(3):
            assert torch.equal(first, ix.select_topk(ix(x, pos, attn_bias=bias)))


def test_scores_are_nonnegative_before_the_gate_and_bias_is_additive():
    torch.manual_seed(0)
    seq = 32
    ix = Qwen3DSAIndexer(_cfg(), hidden_rms=1.0)
    x, pos = torch.randn(1, seq, HIDDEN), torch.arange(seq).unsqueeze(0)
    with torch.no_grad():
        q, k, w = ix(x, pos, return_projection=True)
        plain = ix.scores(q, k, w)
        biased = ix.scores(q, k, w, attn_bias=_causal_bias(seq))
    allowed = _causal_bias(seq) == 0.0
    torch.testing.assert_close(biased[allowed], plain[allowed], rtol=1e-5, atol=1e-6)
    assert torch.isneginf(biased[~allowed]).all()


def test_fp8_path_tracks_the_full_precision_path():
    """FP8 fake-quant must be a perturbation of the bf16 algebra, not a different function: the per-row
    positive scales factor out of the ReLU. A large divergence here means the quantization or the Hadamard
    is being applied inconsistently between q and k."""
    torch.manual_seed(0)
    seq = 128
    x, pos = torch.randn(1, seq, HIDDEN), torch.arange(seq).unsqueeze(0)
    q, k, w = Qwen3DSAIndexer(_cfg(), hidden_rms=1.0).eval()(x, pos, return_projection=True)
    ix_fp8 = Qwen3DSAIndexer(_cfg(fp8=True), hidden_rms=1.0)
    ix_bf16 = Qwen3DSAIndexer(_cfg(fp8=False, serving_compat=False), hidden_rms=1.0)
    with torch.no_grad():
        a, b = ix_fp8.scores(q, k, w), ix_bf16.scores(q, k, w)
    rel = ((a - b).norm() / b.norm()).item()
    assert rel < 0.05, f"fp8 vs bf16 relative deviation {rel:.3f} is too large to be quantization noise"


def test_serve_side_zero_pad_is_lossless():
    """Stock DeepGEMM's ``mqa_logits`` takes ``head_dim=128`` and head counts in {32, 64, 128}, so serving a
    16x64 indexer means zero-padding to 32x128 (``vllm_minicpm3_dsa/indexer.py``). Two properties make that
    safe, and both are asserted here: zeros do not change the row ``amax`` (so the FP8 scale is identical),
    and padded heads carry zero gate weights (so ``w * ReLU(.) = 0``).
    """
    torch.manual_seed(0)
    seq, cfg = 96, _cfg()
    ix = Qwen3DSAIndexer(cfg, hidden_rms=1.0)
    x, pos = torch.randn(1, seq, HIDDEN), torch.arange(seq).unsqueeze(0)
    with torch.no_grad():
        q, k, w = ix(x, pos, return_projection=True)
        ours = ix.scores(q, k, w)

        # replicate the serving path: Hadamard over the REAL head_dim, then pad dim 64->128 and heads
        # 16->32 with zeros, then quantize over the padded rows.
        qr, kr = _rotate_activation(q), _rotate_activation(k)
        pad_d = 128 - cfg.head_dim
        qp = torch.nn.functional.pad(qr, (0, pad_d))
        kp = torch.nn.functional.pad(kr, (0, pad_d))
        qp = torch.nn.functional.pad(qp, (0, 0, 0, 32 - cfg.n_heads))  # zero q rows for padded heads
        wp = torch.nn.functional.pad(w, (0, 32 - cfg.n_heads))  # zero gate for padded heads
        qq, kq = _fake_quant_fp8(qp, True), _fake_quant_fp8(kp, True)
        dots = torch.relu(torch.einsum("bqhd,bkd->bqhk", qq, kq))
        padded = torch.einsum("bqhk,bqh->bqk", dots, (wp * ix.softmax_scale).to(dots.dtype))

    torch.testing.assert_close(padded, ours, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------------------------------
# config guards (plan_v2.md §5.1)
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(head_dim=48, rope_head_dim=48),  # not in {32, 64, 128}: KeyeIndexer asserts this
        dict(n_heads=12),  # must divide 128: Keye's block_q = 128 // num_heads
        dict(fp8_ue8m0=False),  # serving quantizes with a UE8M0 scale
        dict(head_dim=256, rope_head_dim=256),  # > block_size: breaks one-scale-per-row
        dict(rope_head_dim=65),  # odd rope width
        dict(mode="warmup"),  # typo'd mode
        dict(num_kv_heads=7),  # H_q not divisible by H_kv
    ],
)
def test_serving_compat_rejects_unservable_configs(kw):
    """A config that cannot be served must fail at construction, not after a multi-day run."""
    with pytest.raises(ValueError):
        _cfg(**kw)


def test_toy_shapes_allowed_when_serving_compat_is_off():
    cfg = _cfg(n_heads=2, head_dim=8, rope_head_dim=4, top_k=4, fp8=False, serving_compat=False)
    ix = Qwen3DSAIndexer(cfg, hidden_rms=1.0)
    out = ix(torch.randn(1, 6, HIDDEN), torch.arange(6).unsqueeze(0))
    assert out.shape == (1, 6, 6)
