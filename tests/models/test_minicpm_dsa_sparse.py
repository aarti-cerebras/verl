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
"""M0 correctness tests for DSA Phase-2 sparse attention (T1/T2), CPU-only, tiny dims.

Covers:
  * parity     — with top_k >= T the gathered-KV sparse attention equals dense causal attention;
  * grad(LM)   — the LM path trains the BASE (q/k/v) but NOT the indexer (top-k is stop-grad);
  * grad(KL)   — the selected-set KL trains the INDEXER but NOT the base (detached target + input);
  * masking    — top-k selection never attends across the causal / per-document boundary.
"""

import types

import torch

from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer
from verl.models.transformers.minicpm_dsa import (
    _causal_doc_bias_block,
    _sparse_attn,
    _sparse_attn_and_kl,
    _sparse_indexer_kl,
)

torch.manual_seed(0)
B, H, T = 1, 4, 12
QK, DV, ROPE, QLORA, HID = 16, 16, 8, 32, 64
DTYPE = torch.float32


def _make_attn(top_k, kl_block=T):
    cfg = DSAConfig(enabled=True, n_heads=H, head_dim=16, rope_head_dim=ROPE, q_lora_rank=QLORA,
                    hidden_size=HID, top_k=top_k, mode="sparse", kl_block_size=kl_block, fp8=False)
    attn = types.SimpleNamespace()
    attn.dsa = cfg
    attn.indexer = LightningIndexer(cfg).to(DTYPE)
    attn.o_proj = torch.nn.Linear(H * DV, H * DV, bias=False).to(DTYPE)
    attn.softmax_scale = QK ** -0.5
    attn.training = True
    return attn


def _inputs():
    hidden = torch.randn(B, T, HID, dtype=DTYPE)
    qr = torch.randn(B, T, QLORA, dtype=DTYPE)
    q = torch.randn(B, H, T, QK, dtype=DTYPE)
    k = torch.randn(B, H, T, QK, dtype=DTYPE)
    v = torch.randn(B, H, T, DV, dtype=DTYPE)
    pos = torch.arange(T).unsqueeze(0)  # single document
    ang = torch.arange(T).float().unsqueeze(1) * torch.arange(ROPE).float().unsqueeze(0) * 0.1
    cos, sin = torch.cos(ang), torch.sin(ang)  # [T, ROPE]
    return hidden, qr, q, k, v, cos, sin, pos


def _dense_ref(attn, q, k, v, pos):
    bias = _causal_doc_bias_block(pos, 0, T, T, q.device).to(DTYPE)  # [B,T,T] causal (single doc)
    s = torch.einsum("bhtd,bhsd->bhts", q, k) * attn.softmax_scale + bias[:, None]
    a = torch.softmax(s.float(), dim=-1).to(v.dtype)
    o = torch.einsum("bhts,bhsd->bhtd", a, v)
    return attn.o_proj(o.transpose(1, 2).reshape(B, T, H * DV))


def test_parity_topk_ge_T_equals_dense():
    attn = _make_attn(top_k=T)  # select all -> must equal dense
    hidden, qr, q, k, v, cos, sin, pos = _inputs()
    out, idx, _ = _sparse_attn(attn, hidden, qr, q, k, v, cos, sin, pos, None)
    ref = _dense_ref(attn, q, k, v, pos)
    err = (out - ref).abs().max().item()
    assert err < 1e-4, f"parity failed: max abs err {err}"
    print(f"[parity] top_k>=T sparse == dense (max abs err {err:.2e})  OK")


def test_grad_lm_trains_base_not_indexer():
    attn = _make_attn(top_k=6)
    hidden, qr, q, k, v, cos, sin, pos = _inputs()
    q.requires_grad_(True)
    attn.indexer.zero_grad(set_to_none=True)
    out, _, _ = _sparse_attn(attn, hidden, qr, q, k, v, cos, sin, pos, None)
    out.sum().backward()
    assert q.grad is not None and q.grad.abs().sum() > 0, "LM path must send grad to base q"
    assert attn.indexer.wq_b.weight.grad is None, "LM path must NOT grad the indexer (top-k is stop-grad)"
    print("[grad-LM] base q gets grad; indexer gets none  OK")


def test_grad_kl_trains_indexer_not_base():
    attn = _make_attn(top_k=6)
    hidden, qr, q, k, v, cos, sin, pos = _inputs()
    q.requires_grad_(True)
    attn.indexer.zero_grad(set_to_none=True)
    out, idx, I = _sparse_attn(attn, hidden, qr, q, k, v, cos, sin, pos, None)
    kl = _sparse_indexer_kl(attn, q, k, I, idx, pos, None)
    kl.backward()
    assert attn.indexer.wq_b.weight.grad is not None and attn.indexer.wq_b.weight.grad.abs().sum() > 0, \
        "selected-set KL must train the indexer"
    assert q.grad is None, "selected-set KL must NOT grad the base (target + indexer input are detached)"
    assert torch.isfinite(kl), f"KL not finite: {kl}"
    print(f"[grad-KL] indexer gets grad; base gets none; kl={kl.item():.4f}  OK")


def test_topk_respects_causal():
    attn = _make_attn(top_k=T)  # even selecting all, masked (future) keys must get ~0 weight
    hidden, qr, q, k, v, cos, sin, pos = _inputs()
    out, idx, _ = _sparse_attn(attn, hidden, qr, q, k, v, cos, sin, pos, None)
    # query 0 can attend only to key 0; its output must equal v[...,0,:] @ o_proj (single-key softmax)
    ref = _dense_ref(attn, q, k, v, pos)
    assert (out[:, 0] - ref[:, 0]).abs().max().item() < 1e-4
    print("[mask] causal respected (query 0 attends only to key 0)  OK")


def test_kl_checkpoint_parity():
    """Activation-checkpointing the fused sparse graph (kl_checkpoint) must be numerically identical — same
    attn_output, same kl, same grads to BOTH base (q/k/v) and indexer — as the non-checkpointed path."""
    import torch.utils.checkpoint as ckpt

    def run(use_ckpt):
        torch.manual_seed(1)
        attn = _make_attn(top_k=6, kl_block=5)  # multi-tile (block < T) to exercise the tiled loops
        attn.training = True
        hidden, qr, q, k, v, cos, sin, pos = _inputs()
        q, k, v = q.clone().requires_grad_(True), k.clone().requires_grad_(True), v.clone().requires_grad_(True)
        attn.indexer.zero_grad(set_to_none=True)
        args = (attn, hidden, qr, q, k, v, cos, sin, pos, None)
        if use_ckpt:
            out, kl = ckpt.checkpoint(_sparse_attn_and_kl, *args, use_reentrant=False)
        else:
            out, kl = _sparse_attn_and_kl(*args)
        (out.sum() + kl).backward()
        return out, kl, q.grad, k.grad, v.grad, attn.indexer.wq_b.weight.grad

    o0, kl0, qg0, kg0, vg0, ig0 = run(False)
    o1, kl1, qg1, kg1, vg1, ig1 = run(True)
    assert torch.allclose(o0, o1, atol=1e-5), "attn_output differs under checkpoint"
    assert torch.allclose(kl0, kl1, atol=1e-5), "kl differs under checkpoint"
    for name, a, b in [("q", qg0, qg1), ("k", kg0, kg1), ("v", vg0, vg1), ("indexer", ig0, ig1)]:
        assert a is not None and b is not None, f"{name} grad missing"
        assert torch.allclose(a, b, atol=1e-5), f"{name} grad differs under checkpoint"
    print(f"[kl-ckpt] checkpointed == eager (out/kl/grads to base+indexer)  kl={kl0.item():.4f}  OK")


def test_padding_invariance():
    """no_padding engine path right-pads jagged input to [bsz,T] + a mask (transformer_impl.py:1099-1133).
    Invariant #1: the LM attn output and the selected-set KL computed on the REAL tokens must be IDENTICAL
    whether or not masked pad tokens are appended. If the mask/causal handling were wrong, pads would leak
    into real-token attention / the KL average and this would drift."""
    torch.manual_seed(3)
    attn = _make_attn(top_k=6, kl_block=5)  # multi-tile
    attn.training = True
    hidden, qr, q, k, v, cos, sin, pos = _inputs()  # T real tokens, single doc

    # (i) real sequence alone (all valid)
    out_r, idx_r, I_r = _sparse_attn(attn, hidden, qr, q, k, v, cos, sin, pos, None)
    kl_r = _sparse_indexer_kl(attn, q, k, I_r, idx_r, pos, None)

    # (ii) same sequence + P masked pad tokens appended (right-pad, attention_mask=0 on pads)
    P = 5
    def _pad(x, dim):
        shape = list(x.shape); shape[dim] = P
        return torch.cat([x, torch.randn(shape, dtype=x.dtype)], dim=dim)
    Tp = T + P
    angP = torch.arange(Tp).float().unsqueeze(1) * torch.arange(ROPE).float().unsqueeze(0) * 0.1
    cosP, sinP = torch.cos(angP), torch.sin(angP)
    posP = torch.arange(Tp).unsqueeze(0)
    maskP = torch.cat([torch.ones(1, T), torch.zeros(1, P)], dim=1)  # [1, Tp]
    out_p, idx_p, I_p = _sparse_attn(attn, _pad(hidden, 1), _pad(qr, 1), _pad(q, 2), _pad(k, 2), _pad(v, 2),
                                     cosP, sinP, posP, maskP)
    kl_p = _sparse_indexer_kl(attn, _pad(q, 2), _pad(k, 2), I_p, idx_p, posP, maskP)

    # NOTE: idx may CONTAIN pad/future indices when a query has fewer valid keys than top_k — top-k fills the
    # remaining slots with masked keys, but they carry -inf bias => zero attention weight and are zeroed out of
    # the KL (allow_sel). So the meaningful invariant is that the OUTPUT and KL on real tokens are unchanged,
    # NOT that idx excludes pads.
    err_out = (out_p[:, :T] - out_r).abs().max().item()
    err_kl = (kl_p - kl_r).abs().item()
    assert err_out < 1e-4, f"LM output on real tokens changed by pads: {err_out}"
    assert err_kl < 1e-4, f"selected-set KL changed by pads: {err_kl}"
    print(f"[pad-inv] real-token output & KL invariant to appended masked pads "
          f"(out err {err_out:.2e}, kl err {err_kl:.2e})  OK")


def test_full_attention_below_topk():
    """A query at position i attends over i+1 causal keys. When i+1 <= top_k it must use FULL (dense)
    attention (all valid keys selected; masked filler carries -inf => zero weight). When i+1 > top_k it is
    genuinely sparse. Verifies both halves with top_k in the middle of the sequence."""
    tk = 6
    attn = _make_attn(top_k=tk)
    hidden, qr, q, k, v, cos, sin, pos = _inputs()  # single doc, T=12 > tk
    out, idx, _ = _sparse_attn(attn, hidden, qr, q, k, v, cos, sin, pos, None)
    ref = _dense_ref(attn, q, k, v, pos)

    # positions with (i+1) <= top_k  ->  i <= tk-1  ->  must equal dense exactly
    dense_err = (out[:, :tk] - ref[:, :tk]).abs().max().item()
    assert dense_err < 1e-4, f"positions < top_k are not full attention: max err {dense_err}"

    # positions with (i+1) > top_k should actually be sparse for at least one row (non-vacuous check)
    sparse_gap = (out[:, tk:] - ref[:, tk:]).abs().max().item()
    assert sparse_gap > 1e-4, "expected genuine sparsity for positions >= top_k, but output matched dense"
    print(f"[below-topk] positions < top_k == dense (err {dense_err:.2e}); positions >= top_k sparse "
          f"(gap {sparse_gap:.2e})  OK")


if __name__ == "__main__":
    test_parity_topk_ge_T_equals_dense()
    test_grad_lm_trains_base_not_indexer()
    test_grad_kl_trains_indexer_not_base()
    test_topk_respects_causal()
    test_kl_checkpoint_parity()
    test_padding_invariance()
    test_full_attention_below_topk()
    print("\nALL M0 SPARSE TESTS PASSED")
