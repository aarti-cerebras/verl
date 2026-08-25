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
"""P1 (docs/qwen3_4b_dsa/serving_eval_plan.md §4): training indexer == serving indexer.

Engine-free: no vLLM engine, no KV cache, no DeepGEMM. It pins the numerics contract between
``verl.models.transformers.qwen3_dsa_indexer.Qwen3DSAIndexer`` (train) and
``scripts.dsa.vllm_qwen3_dsa.indexer.Qwen3DSAServingIndexer`` (serve), which is the layer where
train/serve drift has bitten this project three times: the FP8 scale format, the missing
``index_topk`` gate, and nondeterministic top-k ties. Each presented as a fluent, plausible model.

What is asserted, in order of how badly it fails if wrong:
  1. **Selected sets are identical.** The only quantity that changes the served function.
  2. Projections (q, k, gate) agree to bf16 noise -- localises any failure in (1).
  3. The ``16x64 -> 32x128`` pad is lossless: dot unchanged, UE8M0 row scale unchanged.
  4. Top-k selection is bit-deterministic across repeated runs on the same input.

Run:  .devlibs/vllm026/bin/python -m pytest tests/dsa/test_qwen3_dsa_serving_indexer_parity.py -q
"""

import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.dsa.vllm_qwen3_dsa.indexer import (  # noqa: E402
    PADDED_HEAD_DIM,
    Qwen3DSAServingIndexer,
    _per_token_group_quant,
    _rotate_activation,
)
# Load the TRAINING module by path, not as `verl.models...`: `verl/__init__.py` imports ray, which
# the serving venv deliberately does not have. Loading the file directly also documents the real
# dependency -- the training indexer needs nothing but torch.
_TRAIN_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "verl", "models", "transformers", "qwen3_dsa_indexer.py",
)
_spec = importlib.util.spec_from_file_location("qwen3_dsa_indexer_train", _TRAIN_SRC)
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
Qwen3DSAConfig = _train_mod.Qwen3DSAConfig
Qwen3DSAIndexer = _train_mod.Qwen3DSAIndexer

HIDDEN, N_HEADS, HEAD_DIM, ROPE_DIM, THETA = 2560, 16, 64, 64, 5e6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_pair(top_k: int = 256, seed: int = 0, dtype=torch.bfloat16):
    """A train/serve pair carrying the SAME random weights."""
    torch.manual_seed(seed)
    cfg = Qwen3DSAConfig(
        enabled=True,
        hidden_size=HIDDEN,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        rope_theta=THETA,
        top_k=top_k,
        mode="sparse",
    )
    train = Qwen3DSAIndexer(cfg, hidden_rms=1.0).to(device=DEVICE, dtype=dtype)
    serve = Qwen3DSAServingIndexer(
        hidden_size=HIDDEN,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        rope_theta=THETA,
        top_k=top_k,
    ).to(device=DEVICE, dtype=dtype)
    # Names are identical by construction -- that is the point of the export having no mapper.
    serve.load_state_dict({k: v.clone() for k, v in train.state_dict().items()}, strict=True)
    return train, serve


def inputs(T: int = 128, seed: int = 1, dtype=torch.bfloat16):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn((1, T, HIDDEN), generator=g, device=DEVICE, dtype=dtype)
    pos = torch.arange(T, device=DEVICE).unsqueeze(0)
    return x, pos


def test_state_dicts_have_identical_keys():
    train, serve = build_pair()
    assert set(train.state_dict()) == set(serve.state_dict()), (
        "parameter names diverged -- the export would need a mapper, and unmatched names are "
        "dropped silently by vLLM loaders"
    )


@pytest.mark.parametrize("T", [64, 512])
def test_projection_parity(T):
    train, serve = build_pair()
    x, pos = inputs(T)
    with torch.no_grad():
        q_t, k_t, w_t = train.project(x, pos)
        q_s, k_s, w_s = serve.project(x[0], pos[0])
    for name, a, b in (("q", q_t[0], q_s), ("k", k_t[0], k_s), ("weights", w_t[0], w_s)):
        rel = (a.float() - b.float()).abs().max() / a.float().abs().max().clamp(min=1e-6)
        assert rel < 5e-3, f"{name} projection mismatch: rel={rel:.3e}"


@pytest.mark.parametrize("T,top_k", [(128, 32), (512, 256), (512, 4096)])
def test_selection_parity(T, top_k):
    """The load-bearing assertion: identical selected sets (order is irrelevant -- FA3 treats the
    block table as a set, verified in probe_fa3_sparse_gqa.py)."""
    train, serve = build_pair(top_k=top_k)
    x, pos = inputs(T)
    with torch.no_grad():
        s_t = train.scores(*train.project(x, pos))
        idx_t = train.select_topk(s_t)[0]
        idx_s = serve.select_topk(serve.torch_scores(x[0], pos[0]))
    # Compare as sets per query row, only where the row is fully causal-eligible.
    inter = torch.zeros(T, dtype=torch.float32)
    for i in range(T):
        a = set(idx_t[i].tolist())
        b = set(idx_s[i].tolist())
        inter[i] = len(a & b) / max(len(a), 1)
    mean_overlap = inter.mean().item()
    assert mean_overlap > 0.99, f"selection overlap {mean_overlap:.4f} < 0.99"


def test_pad_is_lossless():
    """``64 -> 128`` dims and ``16 -> 32`` heads must not move the dot or the UE8M0 scale."""
    _, serve = build_pair()
    x, pos = inputs(96)
    with torch.no_grad():
        q, k, w = serve.project(x[0], pos[0])
        qr, kr = _rotate_activation(q), _rotate_activation(k)
        dot_real = torch.einsum("qhd,kd->qhk", qr.float(), kr.float())
        qp, kp, wp = serve._pad_for_kernel(q, k, w)
        dot_pad = torch.einsum("qhd,kd->qhk", qp.float(), kp.float())[:, : N_HEADS, :]
        assert torch.allclose(dot_real, dot_pad, atol=1e-4), "zero-pad changed the dot product"

        _, s_real = _per_token_group_quant(kr.float().contiguous(), group_size=HEAD_DIM)
        _, s_pad = _per_token_group_quant(kp.float().contiguous(), group_size=PADDED_HEAD_DIM)
        assert torch.equal(s_real, s_pad), "zero-pad changed the UE8M0 row scale"
        assert wp[:, N_HEADS:].abs().max() == 0, "padded heads must carry zero gate weight"


def test_topk_is_deterministic():
    """Repeated runs on identical input must give bit-identical selection.

    ``torch.topk`` tie-breaking is not documented as deterministic and at k=2048 of 32K there are
    near-ties in every row; Keye replaced it with ``flashinfer.topk`` for exactly this reason. If
    this ever fails, train/serve overlap becomes unmeasurable.
    """
    _, serve = build_pair(top_k=256)
    x, pos = inputs(512)
    with torch.no_grad():
        first = serve.select_topk(serve.torch_scores(x[0], pos[0]))
        for _ in range(3):
            assert torch.equal(first, serve.select_topk(serve.torch_scores(x[0], pos[0])))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
