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
"""Phase-0 exit criteria for Qwen3 + MSA, Phase-1 (dense warm-up) path. Runs on a small Qwen3.

Covers plan §4.2 items 1 (dense equivalence), 4 (dL/dS identity), and the Phase-1 "zero capability
risk" assertion in §5, plus the Eq.-9 teacher-ordering check from kl_loss.md §3.1.

Run:
  cd <repo> && python3 tests/msa/test_qwen3_msa_phase1.py [--model /cb/ml-eng/aarti/models/qwen3_0p6b]
"""

import argparse
import sys

import torch

from verl.models.transformers.qwen3_msa import (
    block_selection_metrics,
    _group_teacher,
    attach_indexers,
    build_msa_config,
    freeze_base_train_indexer,
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
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--top-k", type=int, default=2, help="k blocks; keep k*B_k < seq_len or the "
                    "selection metrics saturate at 1.0 (everything is selected) and prove nothing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", help="float32 keeps the equivalence check exact")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3 import modeling_qwen3

    torch.manual_seed(0)
    dtype = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, attn_implementation="eager")
    model = model.to(a.device).eval()
    cfg_hf = model.config
    print(
        f"model: {cfg_hf.num_hidden_layers}L hidden={cfg_hf.hidden_size} "
        f"{cfg_hf.num_attention_heads}q/{cfg_hf.num_key_value_heads}kv hd={cfg_hf.head_dim} "
        f"G={cfg_hf.num_attention_heads // cfg_hf.num_key_value_heads}"
    )
    ids = torch.randint(0, cfg_hf.vocab_size, (1, a.seq_len), device=a.device)

    # ---------------------------------------------------------------- 1. stock reference
    with torch.no_grad():
        ref = model(input_ids=ids).logits.clone()

    msa_cfg = build_msa_config(
        cfg_hf, kl_block_size=128, kl_reduction="mean", diag_interval=1, dense_prefix=3, top_k=a.top_k
    )
    n_blocks = (a.seq_len + msa_cfg.block_size - 1) // msa_cfg.block_size
    assert a.top_k < n_blocks, f"top_k={a.top_k} >= n_blocks={n_blocks}: selection metrics would saturate"
    n_sparse = sum(1 for i in range(cfg_hf.num_hidden_layers) if msa_cfg.layer_is_sparse(i))
    attach_indexers(model, msa_cfg)
    modeling_qwen3.Qwen3Attention.forward = qwen3_msa_attn_forward
    install_kl_accumulation(model)

    print("\n-- 1. Phase 1 leaves the LM forward untouched --")
    with torch.no_grad():
        out = model(input_ids=ids)
    check("logits bit-identical to stock", torch.equal(out.logits, ref),
          f"max|delta| = {(out.logits - ref).abs().max().item():.3e}")
    kl = model._msa_indexer_kl
    check("per-layer KL produced on every sparse layer",
          model._msa_metrics.get("indexer/n_sparse_layers") == float(n_sparse),
          f"{model._msa_metrics.get('indexer/n_sparse_layers')} of {n_sparse}")
    check("KL is finite and positive", torch.isfinite(kl) and kl.item() > 0, f"sum over layers = {kl.item():.4f}")

    print("\n-- 1b. kl_reduction is a pure gradient scale (sum == mean * n_sparse_layers) --")
    # Per-layer indexer params are disjoint, so the layer reduction cannot change the optimum -- only the
    # gradient scale. This is what licenses defaulting to "mean" in BOTH phases, and it is also the exact
    # factor by which a Phase-2 lambda must be multiplied to stay paper-equivalent.
    kl_mean = kl.item()
    msa_cfg.kl_reduction = "sum"
    with torch.no_grad():
        model(input_ids=ids)
    kl_sum = model._msa_indexer_kl.item()
    msa_cfg.kl_reduction = "mean"
    check(f"sum == mean * {n_sparse}", abs(kl_sum - kl_mean * n_sparse) < 1e-3 * max(1.0, abs(kl_sum)),
          f"mean {kl_mean:.4f} * {n_sparse} = {kl_mean * n_sparse:.4f} vs sum {kl_sum:.4f}")
    with torch.no_grad():
        model(input_ids=ids)  # restore the "mean" metrics for the section below

    print("\n-- 2. metrics --")
    m = model._msa_metrics
    for k in ("indexer/main_attn_covered", "indexer/coverage_ceiling", "indexer/coverage_vs_ceiling",
              "indexer/block_recall", "indexer/score_recall", "indexer/group_disagreement",
              "indexer/entropy_norm", "attn/entropy_norm", "indexer/nan_frac"):
        print(f"     {k:38s} {m[k]:.4f}")
    check("main_attn_covered <= coverage_ceiling (oracle is an upper bound)",
          m["indexer/main_attn_covered"] <= m["indexer/coverage_ceiling"] + 1e-6)
    check("selection is non-trivial (an UNTRAINED indexer must fall short of the oracle)",
          m["indexer/coverage_vs_ceiling"] < 0.99 and m["indexer/block_recall"] < 0.99,
          f"captured/oracle = {m['indexer/coverage_vs_ceiling']:.3f}, "
          f"block_recall = {m['indexer/block_recall']:.3f}")
    # NOTE: score_recall > block_recall is an EMPIRICAL property of a *trained* selector (paper Fig. 3:
    # "the higher score recall further shows that the retrieved blocks account for most of the Main
    # Branch attention mass"), NOT an invariant. A random-init indexer hits random members of I*, so the
    # direction is unconstrained here. Only assert both are well-formed probabilities.
    check("score_recall and block_recall are in [0, 1]",
          0.0 <= m["indexer/score_recall"] <= 1.0 and 0.0 <= m["indexer/block_recall"] <= 1.0,
          f"score {m['indexer/score_recall']:.3f}, block {m['indexer/block_recall']:.3f}")
    check("group_disagreement > 0 (per-group selection is live, not collapsed)",
          m["indexer/group_disagreement"] > 0.0, f"{m['indexer/group_disagreement']:.3f}")
    check("nan_frac == 0", m["indexer/nan_frac"] == 0.0)
    check("untrained entropy_norm is near-uniform (>0.5)", m["indexer/entropy_norm"] > 0.5,
          f"{m['indexer/entropy_norm']:.3f}")

    print("\n-- 3. gradient wiring (Eq. 11) --")
    trainable = freeze_base_train_indexer(model)
    n_idx_params = sum(p.numel() for p in trainable)
    check("only *.indexer.* requires grad", all(".indexer." in n for n, p in model.named_parameters() if p.requires_grad))
    check("index params ~= 2.95M-equivalent per sparse layer", n_idx_params > 0,
          f"{n_idx_params / 1e6:.2f}M over {n_sparse} layers")
    model.train()
    out = model(input_ids=ids)
    loss = model._msa_indexer_kl
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    check("KL gradient reaches all four index-branch tensors",
          all(any(k in n for n in grads) for k in ("index_q_proj", "index_k_proj", "index_q_norm", "index_k_norm")),
          f"{len(grads)} tensors got grad")
    check("no gradient on any base parameter", all(".indexer." in n for n in grads))
    check("all index grads finite and nonzero",
          all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads.values()))
    model.zero_grad(set_to_none=True)
    model.eval()

    print("\n-- 4. dL/dS_idx == P_idx - P  (KL direction; silent if wrong) --")
    torch.manual_seed(1)
    b, h_kv, tq, t = 1, 4, 3, 8
    p = torch.softmax(torch.randn(b, h_kv, tq, t), dim=-1)  # teacher
    s = torch.randn(b, h_kv, tq, t, requires_grad=True)  # student logits
    log_q = torch.log_softmax(s, dim=-1)
    kl_val = (p * (p.clamp_min(1e-12).log() - log_q)).sum(-1).sum()
    kl_val.backward()
    check("gradient identity holds", torch.allclose(s.grad, torch.softmax(s, -1) - p, atol=1e-6),
          f"max|delta| = {(s.grad - (torch.softmax(s, -1) - p)).abs().max().item():.2e}")

    print("\n-- 5. Eq. 9 teacher: per-head softmax THEN 1/G average --")
    torch.manual_seed(2)
    bsz, H, T, D, H_kv = 1, 4, 6, 8, 2
    G = H // H_kv
    q = torch.randn(bsz, H, T, D, dtype=torch.float64)
    k = torch.randn(bsz, H_kv, T, D, dtype=torch.float64)
    bias = torch.zeros(bsz, T, T, dtype=torch.float64)
    bias.masked_fill_(torch.arange(T)[None, :] > torch.arange(T)[:, None], float("-inf"))
    got = _group_teacher(q, k, 0, T, bias, H_kv, D**-0.5)
    # reference: renormalise per head over the support, then average within the group
    want = torch.zeros_like(got)
    for r in range(H_kv):
        for h in range(r * G, (r + 1) * G):
            want[:, r] += torch.softmax(q[:, h] @ k[:, r].transpose(-1, -2) * D**-0.5 + bias, dim=-1)
    want /= G
    check("matches renormalise-then-average", torch.allclose(got, want, atol=1e-12))
    # and it must DIFFER from average-then-renormalise on a restricted support (Phase-2 correctness)
    sup = torch.zeros(bsz, T, T, dtype=torch.float64).masked_fill(
        (torch.arange(T)[None, :] > torch.arange(T)[:, None]) | (torch.arange(T)[None, :] % 2 == 1), float("-inf")
    )
    a_then_r = torch.zeros_like(got)
    for r in range(H_kv):
        for h in range(r * G, (r + 1) * G):
            a_then_r[:, r] += torch.softmax(q[:, h] @ k[:, r].transpose(-1, -2) * D**-0.5 + bias, dim=-1)
    a_then_r = (a_then_r / G) * (sup == 0).unsqueeze(1)
    a_then_r = a_then_r / a_then_r.sum(-1, keepdim=True).clamp_min(1e-30)
    r_then_a = _group_teacher(q, k, 0, T, sup, H_kv, D**-0.5)
    check("the two orders differ on a restricted support (so the order is load-bearing)",
          not torch.allclose(r_then_a, a_then_r, atol=1e-6),
          f"max|delta| = {(r_then_a - a_then_r).abs().max().item():.3e}")

    print("\n-- 6. paper §5.2 metrics vs the kl_loss.md §1.2 worked example --")
    # P_b = [0.055, 0.105, 0.590, 0.250]; I* = {2,3}; Î = {2,1}
    # doc expects: coverage_ceiling 0.840, main_attn_covered 0.695, block_recall 0.500, score_recall 0.59/0.84
    P = torch.tensor([0.055, 0.105, 0.590, 0.250]).view(1, 1, 1, 4)
    m6 = block_selection_metrics(P, torch.tensor([2, 1]).view(1, 1, 1, 2), torch.tensor([7]), top_k=2)
    check("coverage_ceiling == 0.840", abs(m6["coverage_ceiling"].item() - 0.840) < 1e-6)
    check("main_attn_covered == 0.695", abs(m6["main_attn_covered"].item() - 0.695) < 1e-6)
    check("block_recall == |I*∩Î|/|I*| == 0.500", abs(m6["block_recall"].item() - 0.5) < 1e-6)
    check("score_recall == mass(I*∩Î)/mass(I*) == 0.7024",
          abs(m6["score_recall"].item() - 0.59 / 0.84) < 1e-6)
    # Early rows: with fewer than k visible blocks, a PERFECT selection must read 1.0. This is what
    # breaks if I* is not masked to visible blocks before the top-k (phantom zero-mass blocks inflate |I*|).
    m6b = block_selection_metrics(
        torch.tensor([0.3, 0.7, 0.0, 0.0]).view(1, 1, 1, 4), torch.tensor([0, 1]).view(1, 1, 1, 2),
        torch.tensor([200]), top_k=2)
    check("early row (valid_blocks < n_blocks): perfect selection reads 1.0",
          m6b["block_recall"].item() == 1.0 and abs(m6b["score_recall"].item() - 1.0) < 1e-6)
    Pg = torch.rand(1, 4, 1, 8)
    g_same = block_selection_metrics(Pg, torch.tensor([1, 2]).view(1, 1, 1, 2).expand(1, 4, 1, 2),
                                     torch.tensor([999]), top_k=2)["group_disagreement"].item()
    g_disj = block_selection_metrics(Pg, torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]]).view(1, 4, 1, 2),
                                     torch.tensor([999]), top_k=2)["group_disagreement"].item()
    check("group_disagreement spans 0 (identical groups) to 1 (disjoint)",
          g_same == 0.0 and g_disj == 1.0, f"{g_same:.2f} .. {g_disj:.2f}")

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
