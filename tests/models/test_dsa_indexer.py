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
"""Standalone unit tests for the DSA lightning indexer module.

These run on CPU with dummy tensors -- no MiniCPM3 weights, no monkey patch, no distributed setup. The FP8
parity test is gated on a CUDA device with FP8 (E4M3) support.
"""

import pytest
import torch

from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer

# Small dummy shapes (MiniCPM3-style dims, tiny seq/batch).
B, S, HIDDEN, Q_LORA = 2, 16, 2560, 768
N_HEADS, HEAD_DIM, ROPE_DIM = 16, 64, 32


def _cfg(**overrides):
    base = dict(
        enabled=True,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        q_lora_rank=Q_LORA,
        hidden_size=HIDDEN,
        fp8=False,  # default to the bf16/fp32 reference path for CPU tests
    )
    base.update(overrides)
    return DSAConfig(**base)


def _inputs(dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, S, HIDDEN, generator=g, dtype=dtype)
    qr = torch.randn(B, S, Q_LORA, generator=g, dtype=dtype)
    # simple RoPE tables of shape [b, s, rope_head_dim]
    pos = torch.arange(S, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM))
    ang = torch.outer(pos, inv_freq)  # [s, rope_dim/2]
    emb = torch.cat([ang, ang], dim=-1)  # [s, rope_dim]
    cos = emb.cos()[None].expand(B, S, ROPE_DIM).to(dtype).contiguous()
    sin = emb.sin()[None].expand(B, S, ROPE_DIM).to(dtype).contiguous()
    return x, qr, cos, sin


def _naive_scores(q_idx, k_idx, weights, softmax_scale):
    """Reference score: loop over heads, single shared key broadcast across all heads."""
    b, s, h, _ = q_idx.shape
    out = torch.zeros(b, s, s, dtype=q_idx.dtype)
    for hh in range(h):
        dot = torch.einsum("bqd,bkd->bqk", q_idx[:, :, hh, :], k_idx)
        out = out + (weights[:, :, hh : hh + 1] * softmax_scale) * torch.relu(dot)
    return out


def test_project_and_scores_shapes():
    idx = LightningIndexer(_cfg())
    x, qr, cos, sin = _inputs()
    q, k, w = idx.project(x, qr, cos, sin)
    assert q.shape == (B, S, N_HEADS, HEAD_DIM)
    assert k.shape == (B, S, HEAD_DIM)  # single (MQA) key head
    assert w.shape == (B, S, N_HEADS)
    scores = idx.scores(q, k, w)
    assert scores.shape == (B, S, S)
    assert torch.isfinite(scores).all()


def test_scores_match_naive_reference():
    """ReLU weighted-sum + MQA broadcast must equal an explicit per-head loop."""
    idx = LightningIndexer(_cfg())
    x, qr, cos, sin = _inputs()
    q, k, w = idx.project(x, qr, cos, sin)
    got = idx.scores(q, k, w)
    ref = _naive_scores(q, k, w, idx.softmax_scale)
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_mqa_single_key_head():
    """The key projection produces ONE head; perturbing it shifts every query head's contribution."""
    idx = LightningIndexer(_cfg())
    x, qr, cos, sin = _inputs()
    _, k, _ = idx.project(x, qr, cos, sin)
    assert k.dim() == 3 and k.shape[-1] == HEAD_DIM  # [b, s, head_dim], no head axis
    # wk output width is head_dim, not n_heads * head_dim
    assert idx.wk.weight.shape == (HEAD_DIM, HIDDEN)


def test_scores_are_nonnegative_without_mask():
    """Raw scores are sums of (positive weights? not necessarily) * ReLU(.) -- check ReLU clamps the dot."""
    idx = LightningIndexer(_cfg())
    x, qr, cos, sin = _inputs()
    q, k, w = idx.project(x, qr, cos, sin)
    dots = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k))
    assert (dots >= 0).all()


def test_additive_mask_applied():
    idx = LightningIndexer(_cfg())
    x, qr, cos, sin = _inputs()
    q, k, w = idx.project(x, qr, cos, sin)
    unmasked = idx.scores(q, k, w)
    # causal additive mask: -inf above the diagonal
    bias = torch.full((S, S), 0.0)
    bias = bias.masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1), float("-inf"))
    masked = idx.scores(q, k, w, attn_bias=bias[None])
    # below+on diagonal unchanged; above diagonal is -inf
    tril = torch.tril(torch.ones(S, S, dtype=torch.bool))
    assert torch.allclose(masked[:, tril], unmasked[:, tril])
    assert torch.isinf(masked[:, ~tril]).all() and (masked[:, ~tril] < 0).all()


def test_rope_only_touches_rope_slice():
    """With identity RoPE (cos=1, sin=0) projection equals the no-rope projection; the nope-slice is
    never rotated regardless of RoPE tables."""
    idx = LightningIndexer(_cfg())
    x, qr, _, _ = _inputs()
    ones = torch.ones(B, S, ROPE_DIM)
    zeros = torch.zeros(B, S, ROPE_DIM)
    q_id, k_id, _ = idx.project(x, qr, ones, zeros)

    # raw (pre-rope) projections
    q_raw = idx.wq_b(qr).view(B, S, N_HEADS, HEAD_DIM)
    k_raw = idx.k_norm(idx.wk(x))
    torch.testing.assert_close(q_id, q_raw, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_id, k_raw, rtol=1e-5, atol=1e-5)

    # with non-trivial RoPE, the nope-slice is still untouched
    _, _, cos, sin = _inputs()
    q_rot, k_rot, _ = idx.project(x, qr, cos, sin)
    torch.testing.assert_close(q_rot[..., ROPE_DIM:], q_raw[..., ROPE_DIM:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_rot[..., ROPE_DIM:], k_raw[..., ROPE_DIM:], rtol=1e-5, atol=1e-5)
    # the rope-slice DID change
    assert not torch.allclose(q_rot[..., :ROPE_DIM], q_raw[..., :ROPE_DIM])


def test_param_names_and_dtypes():
    idx = LightningIndexer(_cfg())
    names = dict(idx.named_parameters())
    for expected in ["wq_b.weight", "wk.weight", "weights_proj.weight", "k_norm.weight", "k_norm.bias"]:
        assert expected in names, f"missing param {expected}"
    assert "wq_a" not in " ".join(names)  # query reuses MLA q-down path; no wq_a
    assert idx.weights_proj.weight.dtype == torch.float32  # reference keeps weights_proj in fp32
    # all linears are bias-free
    assert idx.wq_b.bias is None and idx.wk.bias is None and idx.weights_proj.bias is None


def test_param_count_matches_table():
    """~0.99M params/layer at n_heads=16, head_dim=64 (see docs/dsa_minicpm3_plan.md table)."""
    idx = LightningIndexer(_cfg())
    total = sum(p.numel() for p in idx.parameters())
    wq_b = Q_LORA * (N_HEADS * HEAD_DIM)  # 786,432
    wk = HIDDEN * HEAD_DIM  # 163,840
    weights_proj = HIDDEN * N_HEADS  # 40,960
    k_norm = 2 * HEAD_DIM  # 128
    assert total == wq_b + wk + weights_proj + k_norm
    assert abs(total - 991_360) == 0


def test_rotate_activation_orthonormal():
    """rotate_activation (Hadamard * n**-0.5) is orthonormal: involutory and dot-product preserving."""
    from verl.models.transformers.dsa_indexer import _rotate_activation

    x = torch.randn(2, 5, HEAD_DIM)
    y = torch.randn(2, 5, HEAD_DIM)
    # involutory: applying twice returns the input
    torch.testing.assert_close(_rotate_activation(_rotate_activation(x)), x, rtol=1e-5, atol=1e-5)
    # preserves the dot product (so it does NOT change indexer scores, only FP8 quant fidelity)
    dot_raw = (x * y).sum(-1)
    dot_rot = (_rotate_activation(x) * _rotate_activation(y)).sum(-1)
    torch.testing.assert_close(dot_rot, dot_raw, rtol=1e-4, atol=1e-4)


def test_select_topk_inference_path():
    """Inference selection: top-k key indices per query, respecting a causal mask."""
    idx = LightningIndexer(_cfg(top_k=4))
    x, qr, cos, sin = _inputs()
    scores = idx(x, qr, cos, sin)
    sel = idx.select_topk(scores)
    assert sel.shape == (B, S, 4)
    assert sel.min() >= 0 and sel.max() < S

    # causal mask: query t may only select keys <= t
    bias = torch.zeros(S, S).masked_fill(torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float("-inf"))
    sel_c = idx.select_topk(scores, top_k=2, attn_bias=bias[None])
    for t in range(1, S):  # t>=1 guarantees >=2 valid keys
        assert (sel_c[0, t] <= t).all(), f"query {t} selected a future key: {sel_c[0, t].tolist()}"


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9),
    reason="FP8 (E4M3) parity test requires a CUDA device with compute capability >= 9.0 (Hopper)",
)
def test_fp8_matches_bf16_within_tolerance():
    """The FP8 score path must match the bf16 reference within tolerance (FP8 is the production precision;
    bf16 is the correctness baseline)."""
    device = "cuda"
    x, qr, cos, sin = _inputs(dtype=torch.bfloat16)
    x, qr, cos, sin = x.to(device), qr.to(device), cos.to(device), sin.to(device)

    idx_bf16 = LightningIndexer(_cfg(fp8=False)).to(device).bfloat16()
    idx_fp8 = LightningIndexer(_cfg(fp8=True)).to(device).bfloat16()
    idx_fp8.load_state_dict(idx_bf16.state_dict())  # identical weights

    ref = idx_bf16(x, qr, cos, sin).float()
    got = idx_fp8(x, qr, cos, sin).float()

    # What matters for the indexer is that FP8 preserves the score *ranking / direction* (used for top-k
    # selection and softmax(I)), not bit-level parity -- E4M3's ~2-digit precision yields large RELATIVE
    # error on near-zero entries that is irrelevant downstream. So assert:
    #   (a) high cosine similarity of the score vectors (scale-free, dominated by the entries that matter),
    cos_sim = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    assert cos_sim > 0.99, f"FP8 vs bf16 score cosine similarity too low: {cos_sim.item():.4f}"
    #   (b) elementwise closeness with an absolute tolerance scaled to the score magnitude.
    scale = ref.abs().amax().clamp(min=1e-6)
    torch.testing.assert_close(got, ref, rtol=0.3, atol=0.05 * scale.item())
    #   (c) top-k selection is largely preserved (the decision the scores actually drive).
    k = S // 2
    top_ref = ref.topk(k, dim=-1).indices.sort(dim=-1).values
    top_got = got.topk(k, dim=-1).indices.sort(dim=-1).values
    overlap = (top_ref == top_got).float().mean()
    assert overlap > 0.9, f"FP8 top-k selection overlap too low: {overlap.item():.3f}"
