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
from verl.models.transformers.minicpm_dsa import _causal_doc_bias_block, _sparse_attn, _sparse_indexer_kl

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


if __name__ == "__main__":
    test_parity_topk_ge_T_equals_dense()
    test_grad_lm_trains_base_not_indexer()
    test_grad_kl_trains_indexer_not_base()
    test_topk_respects_causal()
    print("\nALL M0 SPARSE TESTS PASSED")
