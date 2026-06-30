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

