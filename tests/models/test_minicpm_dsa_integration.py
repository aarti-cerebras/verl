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
"""Part B integration tests: DSA indexer grafted onto a tiny MiniCPM3 via the monkey patch.

Requires a CUDA device (MiniCPMFlashAttention2 + flash_attn) and the transformers-4.57.1 env with the
`get_usable_length` shim (applied by the patch). Run with:
    PYTHONPATH=/tmp/tf457lib pytest tests/models/test_minicpm_dsa_integration.py -v
"""

import math

import pytest
import torch
import torch.nn as nn

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MiniCPM3 flash-attention integration test requires CUDA"
)

MODEL = "openbmb/MiniCPM3-4B"
DSA_OVERRIDES = {"n_heads": 4, "head_dim": 64, "top_k": 8, "mode": "dense_warmup", "fp8": False}


@pytest.fixture(autouse=True)
def _cuda_cleanup():
    # free each test's model so the suite footprint stays at one tiny model (the GPU may be shared/busy)
    import gc

    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_tiny_minicpm3(dsa_enabled):
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.num_hidden_layers = 2
    cfg.vocab_size = 1000
    cfg._attn_implementation = "flash_attention_2"
    if dsa_enabled:
        cfg.dsa_enabled = True
        cfg.dsa_overrides = dict(DSA_OVERRIDES)
    model = AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    )
    return model.to("cuda").to(torch.bfloat16).eval()


def _patch(model, **kw):
    from verl.models.transformers.monkey_patch import apply_monkey_patch

    apply_monkey_patch(model=model, use_remove_padding=False, ulysses_sp_size=1, **kw)


def _ids(bsz=1, T=16):
    return torch.randint(0, 1000, (bsz, T), device="cuda")


@requires_cuda
def test_patch_attaches_indexer_and_swaps_forward():
    from verl.models.transformers.dsa_indexer import LightningIndexer
    from verl.models.transformers.minicpm_dsa import minicpm3_dsa_attn_forward

    model = _build_tiny_minicpm3(dsa_enabled=True)
    _patch(model)
    for layer in model.model.layers:
        assert isinstance(layer.self_attn.indexer, LightningIndexer)
        assert layer.self_attn.dsa.enabled and layer.self_attn.dsa.n_heads == 4
    # the flash class forward was swapped to the DSA forward
    assert type(model.model.layers[0].self_attn).forward is minicpm3_dsa_attn_forward


@requires_cuda
def test_dense_warmup_produces_indexer_kl():
    model = _build_tiny_minicpm3(dsa_enabled=True)
    _patch(model)
    ids = _ids()
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        model(input_ids=ids, position_ids=pos)
    kl = model._dsa_indexer_kl
    assert kl is not None and torch.isfinite(kl) and kl.item() > 0
    for layer in model.model.layers:
        assert layer.self_attn._dsa_kl is not None and torch.isfinite(layer.self_attn._dsa_kl)


@requires_cuda
def test_base_lm_output_unchanged_by_dense_warmup():
    """The DSA dense_warmup path is a side computation -> LM logits must match the un-patched model."""
    model = _build_tiny_minicpm3(dsa_enabled=True)
    ids = _ids()
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        before = model(input_ids=ids, position_ids=pos).logits.clone()
    _patch(model)
    with torch.no_grad():
        after = model(input_ids=ids, position_ids=pos).logits
    torch.testing.assert_close(after, before, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_qr_fed_to_indexer_is_mla_latent():
    """The forward must feed the indexer qr = q_a_layernorm(q_a_proj(x)), not raw hidden states."""
    model = _build_tiny_minicpm3(dsa_enabled=True)
    _patch(model)
    attn = model.model.layers[0].self_attn

    class Capture(nn.Module):
        # _dense_warmup_kl drives the indexer via project()/scores(), so capture at project().
        def __init__(self, real):
            super().__init__()
            self.real = real
            self.x = None
            self.qr = None

        def project(self, x, qr, cos, sin):
            self.x, self.qr = x.detach(), qr.detach()
            return self.real.project(x, qr, cos, sin)

        def scores(self, *args, **kwargs):
            return self.real.scores(*args, **kwargs)

    cap = Capture(attn.indexer)
    attn.indexer = cap
    ids = _ids()
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        model(input_ids=ids, position_ids=pos)
    expected_qr = attn.q_a_layernorm(attn.q_a_proj(cap.x))
    torch.testing.assert_close(cap.qr, expected_qr, rtol=2e-2, atol=2e-2)


@requires_cuda
def test_sparse_mode_stub_raises():
    model = _build_tiny_minicpm3(dsa_enabled=True)
    _patch(model)
    for layer in model.model.layers:
        layer.self_attn.dsa.mode = "sparse"
    with pytest.raises(NotImplementedError):
        with torch.no_grad():
            model(input_ids=_ids(), position_ids=torch.arange(16, device="cuda").unsqueeze(0))


@requires_cuda
def test_monitoring_diagnostics_populated():
    """After a forward, model._dsa_metrics carries plain-float KL + top-k recall/overlap/score diagnostics."""
    model = _build_tiny_minicpm3(dsa_enabled=True)  # diag fires on the 1st forward regardless of interval
    _patch(model)
    ids = _ids()
    pos = torch.arange(ids.shape[1], device="cuda").unsqueeze(0)
    with torch.no_grad():
        model(input_ids=ids, position_ids=pos)
    m = model._dsa_metrics
    expected = [
        "indexer/kl_layer_mean", "indexer/kl_layer_min", "indexer/kl_layer_max",
        "indexer/topk_recall", "indexer/topk_overlap", "indexer/score_mean", "indexer/score_std",
        "indexer/nan_frac",
        # entropy metrics split by student/teacher into SEPARATE sections: indexer softmax(I) under
        # `indexer/`, base-model attention under `attn/` (so the two never share a wandb pane group)
        "indexer/entropy", "indexer/entropy_frac", "attn/entropy", "attn/entropy_frac",
    ]
    for key in expected:
        assert key in m, f"missing metric {key}"
        assert isinstance(m[key], float) and math.isfinite(m[key]), f"{key}={m[key]!r} not a finite float"
    assert 0.0 <= m["indexer/topk_recall"] <= 1.0 + 1e-4
    assert 0.0 <= m["indexer/topk_overlap"] <= 1.0 + 1e-4
    assert m["indexer/nan_frac"] == 0.0
    assert 0.0 <= m["indexer/entropy_frac"] <= 1.0 + 1e-4  # normalized softmax(I) entropy
    assert 0.0 <= m["attn/entropy_frac"] <= 1.0 + 1e-4  # normalized base-model attention entropy
    assert m["indexer/kl_layer_min"] <= m["indexer/kl_layer_mean"] + 1e-6 <= m["indexer/kl_layer_max"] + 1e-6


@requires_cuda
def test_per_layer_metrics_logged_when_enabled():
    """log_per_layer emits SEPARATE per-layer scalars, each in its own wandb section: kl_by_layer/L## every
    step; indexer/entropy_frac_by_layer/L## and attn/entropy_frac_by_layer/L## on diag forwards. Off by
    default (would be ~3*n_layers keys)."""
    import statistics

    model = _build_tiny_minicpm3(dsa_enabled=True)
    model.config.dsa_overrides = {**DSA_OVERRIDES, "log_per_layer": True}
    _patch(model)
    ids = _ids(bsz=1, T=16)
    pos = torch.arange(16, device="cuda").unsqueeze(0)
    with torch.no_grad():
        model(input_ids=ids, position_ids=pos)  # forward #1 -> diag always runs
    m = model._dsa_metrics

    per_layer_kl = []
    for i in range(2):  # tiny model has 2 layers -> L00, L01
        kk = f"kl_by_layer/L0{i}"
        assert kk in m and isinstance(m[kk], float) and math.isfinite(m[kk]), f"missing/bad {kk}: {list(m)}"
        for ek in (f"indexer/entropy_frac_by_layer/L0{i}", f"attn/entropy_frac_by_layer/L0{i}"):
            assert ek in m and 0.0 <= m[ek] <= 1.0 + 1e-4, f"missing/bad {ek}"
        per_layer_kl.append(m[kk])
    # the aggregate mean must equal the mean of the per-layer values (bf16 tol — per-layer KL is bf16)
    assert m["indexer/kl_layer_mean"] == pytest.approx(statistics.mean(per_layer_kl), abs=5e-3)
    assert min(per_layer_kl) == pytest.approx(m["indexer/kl_layer_min"], abs=5e-3)


def test_per_layer_metrics_off_by_default():
    """Sanity (CPU): with log_per_layer unset, no per-layer keys are emitted (dashboards stay clean)."""
    from verl.models.transformers.dsa_indexer import DSAConfig

    assert DSAConfig().log_per_layer is False


def test_dense_warmup_kl_tiling_matches_full():
    """The query-block-tiled KL recompute must equal the single-block (full) recompute (CPU, no model)."""
    import types

    from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer
    from verl.models.transformers.minicpm_dsa import _dense_warmup_kl

    torch.manual_seed(0)
    bsz, H, T, Dh = 2, 2, 12, 6  # attention heads/dim (target side)
    cfg = DSAConfig(enabled=True, n_heads=4, head_dim=8, rope_head_dim=4, q_lora_rank=16, hidden_size=32, fp8=False)
    attn = types.SimpleNamespace(softmax_scale=Dh**-0.5, indexer=LightningIndexer(cfg), dsa=cfg)

    hidden = torch.randn(bsz, T, 32)
    qr = torch.randn(bsz, T, 16)
    q_states = torch.randn(bsz, H, T, Dh)
    k_states = torch.randn(bsz, H, T, Dh)
    cos = torch.randn(T, 4)
    sin = torch.randn(T, 4)
    pos = torch.arange(T).unsqueeze(0).expand(bsz, T)

    cfg.kl_block_size = T  # full (one block)
    kl_full = _dense_warmup_kl(attn, hidden, qr, q_states, k_states, cos, sin, pos)
    cfg.kl_block_size = 3  # tiled
    kl_tiled = _dense_warmup_kl(attn, hidden, qr, q_states, k_states, cos, sin, pos)
    torch.testing.assert_close(kl_tiled, kl_full, rtol=1e-5, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("T", [16, 512])
def test_recompute_matches_minicpm_eager_attention(T):
    """Ground-truth check: our recomputed target `p` must equal MiniCPM's OWN attention distribution.

    Flash hides the weights, so we use MiniCPM's *eager* attention (output_attentions=True) as the
    reference and compare its head-averaged attn_weights to our recomputed `p` (from the flash-path q/k),
    on the same weights + input. Catches any mismatch in softmax_scale, RoPE, nope/pe split, or causal mask.

    T=512 is the positional-confidence case (9b): eager at 32K is infeasible (~172 GB attn matrix), but
    MiniCPM3's RoPE is regime-independent (long_factor == short_factor, original_max == max == 32768) and
    `p` is built from the base's own post-RoPE q/k, so a mid-length single-doc match is representative of
    32K. Larger T would just cost eager memory without adding coverage.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from verl.models.transformers.minicpm_dsa import apply_get_usable_length_shim

    apply_get_usable_length_shim()
    ids = _ids(bsz=1, T=T)
    pos = torch.arange(T, device="cuda").unsqueeze(0)  # single doc, no padding -> strictly causal

    # --- reference: MiniCPM eager attention weights (the real distribution) ---
    cfg_e = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg_e.num_hidden_layers = 2
    cfg_e.vocab_size = 1000
    cfg_e._attn_implementation = "eager"
    eager = AutoModelForCausalLM.from_config(cfg_e, trust_remote_code=True).to("cuda").to(torch.bfloat16).eval()
    with torch.no_grad():
        out = eager(input_ids=ids, position_ids=pos, output_attentions=True)
    ref_p = [a.float().mean(dim=1) for a in out.attentions]  # per layer: [bsz, T, T] head-averaged

    # --- ours: DSA flash path, SAME weights, capture the recomputed p ---
    cfg_f = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg_f.num_hidden_layers = 2
    cfg_f.vocab_size = 1000
    cfg_f._attn_implementation = "flash_attention_2"
    flash = AutoModelForCausalLM.from_config(
        cfg_f, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    ).to("cuda").to(torch.bfloat16)
    flash.load_state_dict(eager.state_dict())  # identical base weights (no indexer yet)
    flash.config.dsa_enabled = True
    flash.config.dsa_overrides = {"n_heads": 4, "head_dim": 64, "mode": "dense_warmup", "fp8": False, "kl_block_size": 4096}
    _patch(flash)
    for layer in flash.model.layers:
        layer.self_attn.dsa._capture_p = True
    with torch.no_grad():
        flash(input_ids=ids, position_ids=pos)
    ours_p = [layer.self_attn._dsa_p.float() for layer in flash.model.layers]

    for li, (o, r) in enumerate(zip(ours_p, ref_p)):
        assert o.shape == r.shape, f"layer {li}: {o.shape} vs {r.shape}"
        torch.testing.assert_close(o, r, rtol=2e-2, atol=2e-2)


def test_dense_warmup_kl_ignores_padding():
    """With an attention_mask, pad tokens contribute nothing: KL(masked length-8) == KL(real length-5)."""
    import types

    from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer
    from verl.models.transformers.minicpm_dsa import _dense_warmup_kl

    torch.manual_seed(0)
    bsz, H, Dh = 1, 2, 6
    R, L = 5, 8  # 5 real tokens, padded to length 8
    cfg = DSAConfig(enabled=True, n_heads=4, head_dim=8, rope_head_dim=4, q_lora_rank=16, hidden_size=32, fp8=False)
    attn = types.SimpleNamespace(softmax_scale=Dh**-0.5, indexer=LightningIndexer(cfg), dsa=cfg)

    hidden = torch.randn(bsz, L, 32)
    qr = torch.randn(bsz, L, 16)
    q = torch.randn(bsz, H, L, Dh)
    k = torch.randn(bsz, H, L, Dh)
    cos = torch.randn(L, 4)
    sin = torch.randn(L, 4)
    # right-padded: position_ids reset to 0 in the pad region (as torch.nested.to_padded_tensor does)
    pos8 = torch.tensor([[0, 1, 2, 3, 4, 0, 0, 0]])
    mask8 = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])

    kl_masked = _dense_warmup_kl(attn, hidden, qr, q, k, cos, sin, pos8, attention_mask=mask8)
    assert torch.isfinite(kl_masked)  # pad-query NaNs must not leak

    # real-only equivalent: first R positions, no padding
    pos5 = torch.tensor([[0, 1, 2, 3, 4]])
    kl_real = _dense_warmup_kl(attn, hidden[:, :R], qr[:, :R], q[:, :, :R], k[:, :, :R], cos, sin, pos5)
    torch.testing.assert_close(kl_masked, kl_real, rtol=1e-5, atol=1e-5)

    # diagnostics must also stay finite under padding (pad-query rows give NaN recall pre-guard)
    cfg._do_diag = True
    _dense_warmup_kl(attn, hidden, qr, q, k, cos, sin, pos8, attention_mask=mask8)
    diag = attn._dsa_diag
    assert diag is not None and all(torch.isfinite(v) for v in diag.values()), f"non-finite diag: {diag}"
    assert 0.0 <= diag["recall"].item() <= 1.0 + 1e-4 and diag["nan_frac"].item() == 0.0


def test_causal_doc_mask_helper():
    """Pure-CPU unit test of the per-document causal mask (no model needed)."""
    from verl.models.transformers.minicpm_dsa import _build_causal_doc_bias

    # two packed docs of length 4 each; position_ids reset at each doc start
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    bias = _build_causal_doc_bias(pos, T=8, device="cpu", dtype=torch.float32)[0]  # [8,8]
    allow = bias == 0.0
    # query 5 (2nd token of doc 2) may see keys 4,5 only (in-doc, causal); not 0-3 (other doc) or 6,7 (future)
    assert allow[5].tolist() == [False, False, False, False, True, True, False, False]
    # query 2 (doc 1) sees keys 0,1,2; never crosses into doc 2
    assert allow[2].tolist() == [True, True, True, False, False, False, False, False]
    # no cross-document attention anywhere
    assert not allow[:4, 4:].any() and not allow[4:, :4].any()


def test_dense_warmup_target_is_blockdiagonal_multidoc():
    """9a: with multi-document `position_ids` (per-doc resets), the recomputed target `p` must be exactly
    the block-diagonal per-document causal head-averaged softmax — cross-doc/future entries zero, within-doc
    a renormalized causal softmax over the SAME q/k. This validates the doc-mask math that real multi-doc
    packing (#5) will depend on. NOTE: the base flash attention must ALSO be made per-doc (varlen/cu_seqlens)
    when real packing lands, or `p` (per-doc masked) != the true base attention (which would attend
    cross-doc). This test is the acceptance oracle for that #5 wiring. CPU-only (no model needed).
    """
    import types

    from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer
    from verl.models.transformers.minicpm_dsa import _dense_warmup_kl

    torch.manual_seed(0)
    bsz, H, Dh = 1, 3, 8
    doc_lens = [7, 5]  # two packed docs, position_ids reset at the 2nd
    T = sum(doc_lens)
    pos = torch.tensor([[*range(doc_lens[0]), *range(doc_lens[1])]])  # [0..6, 0..4]
    scale = Dh**-0.5
    # kl_block_size >= T => a single query block, so `_capture_p` stashes the full [bsz, T, T] target
    cfg = DSAConfig(
        enabled=True, n_heads=4, head_dim=8, rope_head_dim=4, q_lora_rank=16, hidden_size=32, fp8=False, kl_block_size=T
    )
    attn = types.SimpleNamespace(softmax_scale=scale, indexer=LightningIndexer(cfg), dsa=cfg)

    hidden = torch.randn(bsz, T, 32)
    qr = torch.randn(bsz, T, 16)
    q = torch.randn(bsz, H, T, Dh)
    k = torch.randn(bsz, H, T, Dh)
    cos, sin = torch.randn(T, 4), torch.randn(T, 4)

    cfg._capture_p = True
    _dense_warmup_kl(attn, hidden, qr, q, k, cos, sin, pos)
    p = attn._dsa_p[0]  # [T, T] head-averaged target

    # --- independent block-diagonal causal reference over the SAME q/k ---
    doc_id = torch.tensor([d for d, n in enumerate(doc_lens) for _ in range(n)])  # [0]*7 + [1]*5
    qi = torch.arange(T)
    allow = (qi[:, None] >= qi[None, :]) & (doc_id[:, None] == doc_id[None, :])  # causal AND same-document
    logits = torch.einsum("hid,hjd->hij", q[0], k[0]) * scale  # [H, T, T]
    logits = logits.masked_fill(~allow[None], float("-inf"))
    per_head = torch.softmax(logits, dim=-1)  # [H, T, T], masked entries exactly 0
    ref = per_head.mean(0)  # head-averaged [T, T]

    torch.testing.assert_close(p, ref, rtol=1e-4, atol=1e-4)
    # structural guarantees the doc-mask must give
    assert torch.count_nonzero(p[~allow]) == 0, "cross-document / future keys must be exactly 0"
    torch.testing.assert_close(p.sum(-1), torch.ones(T), atol=1e-4, rtol=0)  # each query row normalized

    # 9c: head aggregation — mean-over-heads == L1-normalize(sum-over-heads) (the DeepSeek-V3.2 target form)
    summed = per_head.sum(0)
    l1 = summed / summed.sum(-1, keepdim=True)
    torch.testing.assert_close(ref, l1, rtol=1e-5, atol=1e-5)

