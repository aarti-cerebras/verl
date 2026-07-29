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
"""Phase-2 (sparse) exit criteria for Qwen3 + MSA. See docs/qwen3_4b_msa/phase2_plan.md §9 step 2-3.

Covers:
  D1 dense equivalence — with k >= n_blocks the sparse path reproduces dense-attention logits
  D2 teacher identity — the KL matches an INDEPENDENT reimplementation of steps D-E (which pins the
     teacher to the group-mean of the forward's own weights, Eq. 9)
  D3 the KL gradient survives per-tile checkpointing (the graph-less-side-effect trap, §1.1)
  D4 dL/dS = P_idx - P on the restricted support
  D5 gradient wiring: L_LM -> base only, L_KL -> index branch only
  D6 no NaN when the sequence is shorter than k*B_k, or with padding

Run:
  cd <repo> && PYTHONPATH=$(pwd) python3 tests/msa/test_qwen3_msa_phase2.py
"""

import argparse
import sys

import torch

from verl.models.transformers.qwen3_msa import (
    _causal_doc_bias_block,
    _selected_token_index,
    _sparse_tile,
    attach_indexers,
    build_msa_config,
    install_kl_accumulation,
    qwen3_msa_attn_forward,
)

OK, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{OK if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/cb/ml-eng/aarti/models/qwen3_0p6b")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3 import modeling_qwen3

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager")
    model = model.to(a.device).eval()
    cfg_hf = model.config
    T, bk = a.seq_len, 128
    n_blocks = T // bk
    ids = torch.randint(0, cfg_hf.vocab_size, (1, T), device=a.device)

    with torch.no_grad():
        dense_ref = model(input_ids=ids).logits.clone()

    cfg = build_msa_config(cfg_hf, mode="sparse", top_k=n_blocks, kl_block_size=128, dense_prefix=3,
                           diag_interval=0)
    attach_indexers(model, cfg)
    modeling_qwen3.Qwen3Attention.forward = qwen3_msa_attn_forward
    install_kl_accumulation(model)

    print("\n-- D1. dense equivalence: k >= n_blocks reproduces dense attention --")
    with torch.no_grad():
        out = model(input_ids=ids)
    d = (out.logits - dense_ref).abs().max().item()
    check("sparse path == dense logits (fp32 accumulation noise)", d / dense_ref.abs().max().item() < 1e-4,
          f"max|delta| = {d:.3e}, relative = {d / dense_ref.abs().max().item():.2e}")
    check("KL finite and positive", torch.isfinite(model._msa_indexer_kl) and model._msa_indexer_kl > 0,
          f"{model._msa_indexer_kl.item():.4f}")
    cfg.top_k = 1
    with torch.no_grad():
        out1 = model(input_ids=ids)
    check("k=1 genuinely restricts (logits move a lot)",
          (out1.logits - dense_ref).abs().max().item() > 1.0,
          f"max|delta| = {(out1.logits - dense_ref).abs().max().item():.3e}")

    print("\n-- D2. the KL matches an independent reimplementation of steps D-E --")
    cfg.top_k = 2
    attn = model.model.layers[10].self_attn
    torch.manual_seed(1)
    b, h_kv, g, tq, dh = 1, cfg.num_kv_heads, cfg.group_size, 8, cfg.head_dim
    h_q = h_kv * g
    q_tile = torch.randn(b, h_q, tq, dh, device=a.device)
    k_st = torch.randn(b, h_kv, T, dh, device=a.device)
    v_st = torch.randn(b, h_kv, T, dh, device=a.device)
    q_ix = torch.randn(b, h_kv, tq, cfg.index_dim, device=a.device)
    k_ix = torch.randn(b, 1, T, cfg.index_dim, device=a.device)
    q0 = 200
    qpos = torch.arange(q0, q0 + tq, device=a.device)
    # `_sparse_tile` now rebuilds the bias internally from position_ids (it must not be passed in: as a
    # checkpoint input it was retained 64 tiles x 33 layers = ~142 GB at 32K). Build the reference with
    # the same helper so the comparison is exact.
    position_ids = torch.arange(T, device=a.device).unsqueeze(0)
    bias = _causal_doc_bias_block(position_ids, q0, q0 + tq, T, a.device, key_mask=None)
    out_t, kl_rows, sel = _sparse_tile(attn, q_tile, k_st, v_st, q_ix, k_ix, position_ids, None, q0, q0 + tq)

    # independent recomputation from `sel` alone
    tok, slot_ok = _selected_token_index(attn.indexer, sel, T)
    bias_sel = torch.gather(bias.unsqueeze(1).expand(b, h_kv, tq, T), 3, tok)
    allow = (bias_sel == 0.0) & slot_ok
    neg = torch.zeros_like(bias_sel).masked_fill(~allow, float("-inf"))
    m = tok.shape[-1]
    gidx = tok.reshape(b, h_kv, tq * m, 1).expand(b, h_kv, tq * m, dh)
    kg = torch.gather(k_st, 2, gidx).reshape(b, h_kv, tq, m, dh)
    vg = torch.gather(v_st, 2, gidx).reshape(b, h_kv, tq, m, dh)
    s = torch.einsum("bhgqd,bhqmd->bhgqm", q_tile.view(b, h_kv, g, tq, dh), kg) * attn.scaling
    aw = torch.softmax(s + neg.unsqueeze(2), dim=-1)
    ref_out = torch.einsum("bhgqm,bhqmd->bhgqd", aw, vg).reshape(b, h_q, tq, dh)
    check("output matches the independent gather+attend", torch.allclose(out_t, ref_out, atol=1e-5),
          f"max|delta| = {(out_t - ref_out).abs().max().item():.2e}")
    # teacher = mean over G of the per-head restricted softmax (Eq. 9), then KL vs the index student
    P = aw.mean(dim=2)
    stu = torch.log_softmax(torch.gather(attn.indexer.scores(q_ix, k_ix, attn_bias=bias), 3, tok) + neg, -1)
    term = P * (P.clamp_min(1e-12).log() - stu)
    ref_kl = torch.where(allow, term, torch.zeros_like(term)).sum(-1)
    check("KL matches the independent Eq.-9 teacher (group-mean of the forward's own weights)",
          torch.allclose(kl_rows, ref_kl, atol=1e-5),
          f"max|delta| = {(kl_rows - ref_kl).abs().max().item():.2e}")

    print("\n-- D3/D5. gradient wiring and checkpointing --")
    for ckpt in (False, True):
        cfg.kl_checkpoint = ckpt
        cfg.top_k = 2
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)  # Phase 2b: base unfrozen
        model.zero_grad(set_to_none=True)
        o = model(input_ids=ids)
        model._msa_indexer_kl.backward()
        idx_g = {n: p.grad for n, p in model.named_parameters() if p.grad is not None and ".indexer." in n}
        base_g = {n: p.grad for n, p in model.named_parameters() if p.grad is not None and ".indexer." not in n}
        tag = "kl_checkpoint=True" if ckpt else "kl_checkpoint=False"
        check(f"L_KL reaches the index branch [{tag}]",
              len(idx_g) > 0 and all(torch.isfinite(v).all() and v.abs().sum() > 0 for v in idx_g.values()),
              f"{len(idx_g)} tensors")
        check(f"L_KL does NOT reach the base [{tag}]", len(base_g) == 0,
              f"{len(base_g)} base tensors got grad")
        model.zero_grad(set_to_none=True)

    o = model(input_ids=ids)
    o.logits.square().mean().backward()  # an LM-like loss
    idx_g = [n for n, p in model.named_parameters() if p.grad is not None and ".indexer." in n]
    base_g = [n for n, p in model.named_parameters() if p.grad is not None and ".indexer." not in n]
    check("L_LM reaches the base", len(base_g) > 0, f"{len(base_g)} tensors")
    check("L_LM does NOT reach the index branch (top-k detached)", len(idx_g) == 0,
          f"{len(idx_g)} index tensors got grad")
    model.zero_grad(set_to_none=True)
    model.eval()

    print("\n-- D4. dL/dS_idx = P_idx - P on the RESTRICTED support --")
    torch.manual_seed(2)
    P = torch.softmax(torch.randn(1, 4, 3, 16), dim=-1)
    S = torch.randn(1, 4, 3, 16, requires_grad=True)
    (P * (P.clamp_min(1e-12).log() - torch.log_softmax(S, -1))).sum(-1).sum().backward()
    check("gradient identity holds", torch.allclose(S.grad, torch.softmax(S, -1) - P, atol=1e-6),
          f"max|delta| = {(S.grad - (torch.softmax(S, -1) - P)).abs().max().item():.2e}")

    print("\n-- D6. NaN safety: short sequence and padding --")
    cfg.kl_checkpoint = False
    cfg.top_k = 16  # k*B_k = 2048 >> T, so every row has fewer visible blocks than slots
    short = torch.randint(0, cfg_hf.vocab_size, (1, 300), device=a.device)
    with torch.no_grad():
        os_ = model(input_ids=short)
    check("T < k*B_k: logits and KL finite", torch.isfinite(os_.logits).all() and
          torch.isfinite(model._msa_indexer_kl), f"kl = {model._msa_indexer_kl.item():.4f}")
    am = torch.ones(1, T, dtype=torch.long, device=a.device)
    am[:, T // 2:] = 0  # right padding -> pad queries have no valid key
    cfg.top_k = 2
    with torch.no_grad():
        op = model(input_ids=ids, attention_mask=am)
    check("padded batch: logits and KL finite (all-masked-row guard)",
          torch.isfinite(op.logits).all() and torch.isfinite(model._msa_indexer_kl),
          f"kl = {model._msa_indexer_kl.item():.4f}")

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
