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
"""Integration tests for Qwen3 + DSA (CPU, tiny randomly-initialised Qwen3 — no model download).

These are the step-2 and step-5 gates from docs/qwen3_4b_dsa/plan_v2.md §7. The load-bearing ones:

* ``test_phase1_lm_output_is_bit_identical`` — Phase 1's entire safety argument. The index branch has no
  output path, so the LM forward must be *exactly* the stock model's. This is also the guard against
  transformers drift in the inlined ``Qwen3Attention.forward`` body.
* ``test_teacher_matches_eager_attention`` — the teacher is a *recomputation* of the base attention, so if
  the mask, the RoPE application or the head-averaging order is wrong, everything downstream distills
  against the wrong target while still looking healthy.
* ``test_gradient_isolation`` — the three-way wiring from arXiv 2512.02556 §2.1.1: LM loss trains only the
  base, the KL trains only the indexer. A leak in either direction is silent and changes what is learned.
* ``test_phase2_free_teacher_is_the_head_averaged_sparse_softmax`` — pins the §4.1 claim, i.e. that the
  target is the mean of the per-head softmaxes over the selected set (equal vote per head), NOT the
  dense-normalized-then-restricted alternative that ``minicpm_dsa`` computes.
"""

import math

import pytest
import torch

from verl.models.transformers.qwen3_dsa import (
    _causal_doc_bias_block,
    _sparse_tile,
    attach_indexers,
    build_dsa_config,
    freeze_base_train_indexer,
    indexer_param_groups,
    install_kl_accumulation,
    qwen3_dsa_attn_forward,
)
from verl.models.transformers.qwen3_dsa_indexer import Qwen3DSAConfig, Qwen3DSAIndexer

SEQ = 24
N_Q, N_KV, D_H, HID, LAYERS = 4, 2, 16, 64, 2


@pytest.fixture()
def restore_qwen3_forward():
    """The monkey patch is class-level, so it must be undone or it leaks into every later test."""
    from transformers.models.qwen3 import modeling_qwen3

    original = modeling_qwen3.Qwen3Attention.forward
    yield
    modeling_qwen3.Qwen3Attention.forward = original


def _build(**dsa_kw):
    """A tiny random Qwen3 + DSA. Returns (model, cfg, batch, stock_logits) with stock computed BEFORE the
    patch is applied, so the dense-equivalence comparison is against the genuine stock forward."""
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(0)
    hf = Qwen3Config(
        vocab_size=256, hidden_size=HID, intermediate_size=128, num_hidden_layers=LAYERS,
        num_attention_heads=N_Q, num_key_value_heads=N_KV, head_dim=D_H,
        max_position_embeddings=512, rope_theta=10000.0, attn_implementation="eager",
    )
    model = modeling_qwen3.Qwen3ForCausalLM(hf).eval()
    batch = dict(
        input_ids=torch.randint(0, 256, (1, SEQ)),
        attention_mask=torch.ones(1, SEQ, dtype=torch.long),
        position_ids=torch.arange(SEQ).unsqueeze(0),
    )
    stock = model(**batch).logits.clone()

    kw = dict(n_heads=2, head_dim=8, rope_head_dim=8, top_k=8, fp8=False, serving_compat=False,
              kl_block_size=8, diag_interval=1)
    kw.update(dsa_kw)
    cfg = build_dsa_config(hf, **kw)
    attach_indexers(model, cfg)
    modeling_qwen3.Qwen3Attention.forward = qwen3_dsa_attn_forward
    install_kl_accumulation(model)
    return model, cfg, batch, stock


def _eager_head_averaged_teacher(model, batch, layer_idx=0):
    """Recompute the head-averaged attention distribution independently, from the layer's own weights."""
    from transformers.models.qwen3 import modeling_qwen3

    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    h = model.model.embed_tokens(batch["input_ids"])
    hs = layer.input_layernorm(h)
    q = attn.q_norm(attn.q_proj(hs).view(1, SEQ, N_Q, D_H)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hs).view(1, SEQ, N_KV, D_H)).transpose(1, 2)
    cos, sin = model.model.rotary_emb(h, batch["position_ids"])
    q, k = modeling_qwen3.apply_rotary_pos_emb(q, k, cos, sin)
    keys = torch.arange(SEQ)
    mask = torch.zeros(1, SEQ, SEQ).masked_fill(keys[None, None, :] > keys[None, :, None], float("-inf"))
    p = torch.zeros(1, SEQ, SEQ)
    for head in range(N_Q):
        logits = (q[:, head] @ k[:, head // (N_Q // N_KV)].transpose(-1, -2)) * attn.scaling
        p += torch.softmax(logits.float() + mask, dim=-1)
    return p / N_Q


# ---------------------------------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------------------------------


def test_phase1_lm_output_is_bit_identical(restore_qwen3_forward):
    """Phase 1's safety argument: attention stays dense and the index branch has no output path, so the LM
    logits must be EXACTLY the stock model's — not merely close."""
    model, cfg, batch, stock = _build(mode="dense_warmup")
    out = model(**batch)
    assert torch.equal(out.logits, stock)
    assert model._dsa_indexer_kl is not None and torch.isfinite(model._dsa_indexer_kl)


def test_teacher_matches_eager_attention(restore_qwen3_forward):
    """The teacher must reproduce the base attention exactly, with tiling active (kl_block_size < SEQ)."""
    model, cfg, batch, _ = _build(kl_block_size=8)
    cfg._capture_p = True
    model(**batch)
    cfg._capture_p = False
    captured = model.model.layers[0].self_attn._dsa_p
    assert captured.shape == (1, SEQ, SEQ)
    torch.testing.assert_close(captured, _eager_head_averaged_teacher(model, batch), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("block", [SEQ, 7, 3])
def test_kl_is_invariant_to_the_query_tile_size(restore_qwen3_forward, block):
    """`kl_block_size` is pure tiling granularity — it must not move the loss."""
    model, cfg, batch, _ = _build(kl_block_size=SEQ)
    model(**batch)
    reference = model._dsa_indexer_kl.item()
    cfg.kl_block_size = block
    model(**batch)
    assert model._dsa_indexer_kl.item() == pytest.approx(reference, abs=1e-6)


def test_kl_checkpoint_changes_neither_loss_nor_gradient(restore_qwen3_forward):
    """`kl_checkpoint` trades memory for a backward recompute and must be numerically invisible. It only
    engages in training mode, and the gradient comparison is the part that matters: a checkpoint that
    silently dropped the graph would still produce the right loss."""
    model, cfg, batch, _ = _build(kl_block_size=5)
    freeze_base_train_indexer(model)
    model.train()
    results = []
    for ckpt in (False, True):
        cfg.kl_checkpoint = ckpt
        for p in model.parameters():
            p.grad = None
        model(**batch)
        model._dsa_indexer_kl.backward()
        grad = model.model.layers[0].self_attn.indexer.weights_proj.weight.grad.clone()
        results.append((model._dsa_indexer_kl.item(), grad))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-7)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)


def test_diagnostics_are_populated_and_finite(restore_qwen3_forward):
    model, cfg, batch, _ = _build(diag_interval=1)
    model(**batch)
    metrics = model._dsa_metrics
    for key in ("indexer/topk_recall", "indexer/topk_overlap", "indexer/entropy_frac",
                "indexer/local_mass", "indexer/group_recall_min", "indexer/group_jsd",
                "indexer/nan_frac", "attn/entropy_frac"):
        assert key in metrics, f"{key} missing"
        assert math.isfinite(metrics[key]), f"{key} is not finite"
    assert metrics["indexer/nan_frac"] == 0.0
    assert 0.0 <= metrics["indexer/topk_recall"] <= 1.0
    assert metrics["indexer/group_jsd"] >= 0.0  # a divergence; negative means the decomposition is wrong


def test_attach_uses_each_layers_own_hidden_rms(restore_qwen3_forward):
    """The per-layer init is the §2.4 fix. Give two layers very different `input_layernorm` gains and the
    attached indexers must come out with correspondingly different `weights_proj` scales."""
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    torch.manual_seed(0)
    hf = Qwen3Config(vocab_size=256, hidden_size=HID, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=N_Q, num_key_value_heads=N_KV, head_dim=D_H,
                     max_position_embeddings=512, rope_theta=10000.0)
    model = modeling_qwen3.Qwen3ForCausalLM(hf)
    with torch.no_grad():
        model.model.layers[0].input_layernorm.weight.fill_(0.05)
        model.model.layers[1].input_layernorm.weight.fill_(5.0)
    attach_indexers(model, build_dsa_config(hf, n_heads=2, head_dim=8, rope_head_dim=8, serving_compat=False))
    lo = model.model.layers[0].self_attn.indexer.weights_proj.weight.std().item()
    hi = model.model.layers[1].self_attn.indexer.weights_proj.weight.std().item()
    # 100x apart in gain => ~100x apart in init std, in the opposite direction
    assert lo / hi == pytest.approx(100.0, rel=0.2), f"lo={lo:.4g} hi={hi:.4g}"


def test_param_groups_exclude_norms_and_gate_from_decay(restore_qwen3_forward):
    model, cfg, batch, _ = _build()
    freeze_base_train_indexer(model)
    groups = indexer_param_groups(model, weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1 and groups[1]["weight_decay"] == 0.0
    # per layer: wq + wk decay; q_norm + k_norm.weight + k_norm.bias + weights_proj do not
    assert len(groups[0]["params"]) == 2 * LAYERS
    assert len(groups[1]["params"]) == 4 * LAYERS


# ---------------------------------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------------------------------


def test_phase2_reduces_to_dense_when_k_covers_the_sequence(restore_qwen3_forward):
    """M0 parity. At `top_k >= T` the selected set is the whole causal set, and softmax is
    permutation-invariant with V gathered in the same order as K, so the sparse path must reproduce dense
    attention. If this fails, the gather, the bias or the group reshape is wrong."""
    model, cfg, batch, stock = _build(mode="sparse", top_k=SEQ, kl_block_size=7)
    out = model(**batch)
    torch.testing.assert_close(out.logits, stock, rtol=1e-4, atol=1e-5)


def test_phase2_sparsity_actually_bites(restore_qwen3_forward):
    """Guard against a silently-dense sparse path — the failure mode that has bitten this project twice on
    the serving side (the missing `index_topk` gate, the FP8 scale format). A small `top_k` MUST change the
    output."""
    model, cfg, batch, stock = _build(mode="sparse", top_k=SEQ)
    torch.testing.assert_close(model(**batch).logits, stock, rtol=1e-4, atol=1e-5)
    cfg.top_k = 4
    assert (model(**batch).logits - stock).abs().max().item() > 1e-2


def test_gradient_isolation(restore_qwen3_forward):
    """arXiv 2512.02556 §2.1.1: *"the training signal of the indexer is from only L^I, while the
    optimization of the main model is according to only the language modeling loss."*

    Enforced by three stop-gradients: the indexer reads detached hidden states, the top-k is detached, and
    the teacher is detached. A leak either way is silent — the run trains, just not the thing you think.
    """
    model, cfg, batch, _ = _build(mode="sparse", top_k=6)
    model.train()

    def _who_has_grad(loss):
        for p in model.parameters():
            p.grad = None
        loss.backward()
        idx = [n for n, p in model.named_parameters()
               if ".indexer." in n and p.grad is not None and p.grad.abs().sum() > 0]
        base = [n for n, p in model.named_parameters()
                if ".indexer." not in n and p.grad is not None and p.grad.abs().sum() > 0]
        return idx, base

    idx, base = _who_has_grad(model(**batch).logits.float().pow(2).mean())
    assert idx == [], f"LM loss leaked into the indexer: {idx}"
    assert base, "LM loss did not reach the base at all"

    model(**batch)
    idx, base = _who_has_grad(model._dsa_indexer_kl)
    assert base == [], f"the indexer KL leaked into the base: {base}"
    assert len(idx) == 6 * LAYERS, f"expected every indexer param to get grad, got {len(idx)}"


def test_phase2_free_teacher_is_the_head_averaged_sparse_softmax(restore_qwen3_forward):
    """Pins the §4.1 target definition by rebuilding `_sparse_tile`'s KL independently.

    The teacher is the mean over ALL query heads of each head's softmax **over the selected set** — an equal
    vote per head. It is NOT the dense-normalized distribution restricted to the selected set and rescaled,
    which is what `minicpm_dsa._sparse_indexer_kl` computes and which weights heads by how much of their
    mass happens to fall inside the selection. The two differ whenever heads disagree, so this test is what
    stops the cheaper-and-correct version from being 'fixed' into the expensive-and-different one.
    """
    torch.manual_seed(0)
    b, t, top_k = 1, 16, 5
    cfg = Qwen3DSAConfig(enabled=True, hidden_size=HID, num_heads=N_Q, num_kv_heads=N_KV,
                         rope_theta=10000.0, n_heads=2, head_dim=8, rope_head_dim=8, top_k=top_k,
                         fp8=False, serving_compat=False, mode="sparse")
    indexer = Qwen3DSAIndexer(cfg, hidden_rms=1.0)

    class _Attn:  # minimal stand-in for the attention module `_sparse_tile` reads
        scaling = D_H**-0.5
        dsa = cfg
        training = False

    attn = _Attn()
    attn.indexer = indexer
    hidden = torch.randn(b, t, HID)
    q = torch.randn(b, N_Q, t, D_H)
    k = torch.randn(b, N_KV, t, D_H)
    v = torch.randn(b, N_KV, t, D_H)
    pos = torch.arange(t).unsqueeze(0)
    q_idx, k_idx, w = indexer(hidden, pos, return_projection=True)

    out, kl_rows, idx = _sparse_tile(attn, q, k, v, q_idx, k_idx, w, pos, None, 0, t)

    # rebuild the expected teacher and KL from scratch
    bias = _causal_doc_bias_block(pos, 0, t, t, q.device)
    bias_sel = torch.gather(bias, 2, idx)
    allow = bias_sel == 0.0
    allow = allow | (~allow.any(-1, keepdim=True) & (torch.arange(top_k) == 0))
    neg = torch.zeros_like(bias_sel).masked_fill(~allow, float("-inf"))
    per_head = []
    for head in range(N_Q):
        kg = torch.gather(k[:, head // (N_Q // N_KV)], 1,
                          idx.reshape(b, t * top_k, 1).expand(b, t * top_k, D_H)).reshape(b, t, top_k, D_H)
        logits = torch.einsum("bqd,bqkd->bqk", q[:, head], kg) * attn.scaling + neg
        per_head.append(torch.softmax(logits.float(), dim=-1))
    expected_p = torch.stack(per_head).mean(0)  # equal vote per head, softmax INSIDE the average

    s_sel = torch.gather(indexer.scores(q_idx, k_idx, w, attn_bias=bias), 2, idx) + neg
    log_q = torch.log_softmax(s_sel.float(), dim=-1)
    term = expected_p * (torch.log(expected_p.clamp_min(1e-12)) - log_q)
    expected_kl = torch.where(allow, term, torch.zeros_like(term)).sum(-1)

    torch.testing.assert_close(kl_rows, expected_kl, rtol=1e-4, atol=1e-6)
    assert out.shape == (b, N_Q, t, D_H)


def test_full_support_kl_runs_and_differs(restore_qwen3_forward):
    """The escape hatch for the selected-set blind spot: with `full_support_kl_prob=1` the KL is taken over
    the whole causal support while the LM path still runs sparse. It must produce a different (and finite)
    loss, and must not disturb the LM output."""
    model, cfg, batch, _ = _build(mode="sparse", top_k=6)
    restricted_logits = model(**batch).logits.clone()
    restricted = model._dsa_indexer_kl.item()
    cfg.full_support_kl_prob = 1.0
    full_logits = model(**batch).logits
    full = model._dsa_indexer_kl.item()
    assert math.isfinite(full) and full != pytest.approx(restricted, abs=1e-6)
    torch.testing.assert_close(full_logits, restricted_logits, rtol=1e-5, atol=1e-6)


def test_padding_is_excluded_from_the_loss(restore_qwen3_forward):
    """Pad keys must be masked out of the teacher's support and pad query rows dropped from the average, or
    padding silently changes the target. Compare a right-padded batch against the unpadded prefix."""
    model, cfg, batch, _ = _build(kl_block_size=SEQ)
    keep = SEQ - 6
    model(**batch)
    padded_mask = batch["attention_mask"].clone()
    padded_mask[:, keep:] = 0
    model(input_ids=batch["input_ids"], attention_mask=padded_mask, position_ids=batch["position_ids"])
    padded_kl = model._dsa_indexer_kl.item()
    model(input_ids=batch["input_ids"][:, :keep], attention_mask=batch["attention_mask"][:, :keep],
          position_ids=batch["position_ids"][:, :keep])
    trimmed_kl = model._dsa_indexer_kl.item()
    assert padded_kl == pytest.approx(trimmed_kl, rel=1e-4)
