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
"""Regression test for DSA Phase-1 Option B2 (docs/dsa_fsdp_sharding_notes.md §3b/§4).

The indexer KL is a side-channel loss: it is computed from indexer params during the forward and stashed on
the model, and does NOT flow through any wrapped module's *returned* output (the base is frozen, so the
logits/hidden output has requires_grad=False). Under FSDP2 the gradient reduce-scatter that writes the
**sharded (optimizer) master's** ``.grad`` is triggered only by autograd crossing a wrapped unit's
requires-grad boundary tensor. So without B2 the indexer's masters never get a grad even though autograd
fills the unsharded compute copy (nonzero ``grad_norm``, flat loss).

B2 fixes this by wrapping each indexer as its own ``fully_shard`` unit whose OUTPUT (the projection) requires
grad, so ITS pre-backward gate fires and ``post_backward``/reduce-scatter runs. This test asserts the
optimizer masters actually receive a grad, and (as a sensitivity check) that they do NOT without B2.
"""

import os
import types

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 reduce-scatter path needs CUDA")

_CFG = dict(
    enabled=True,
    n_heads=4,
    head_dim=32,
    rope_head_dim=16,
    q_lora_rank=48,
    hidden_size=32,
    fp8=True,
    mode="dense_warmup",
    kl_block_size=0,
)


def _build_model(dsa_enabled):
    from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(32, 32)  # stands in for the frozen base
            self.self_attn = nn.Module()
            self.self_attn.indexer = LightningIndexer(DSAConfig(**_CFG))

        def forward(self, hidden, qr, cos, sin):
            # route the projection through __call__ (Change 1) so a separately-wrapped indexer's FSDP hooks fire
            q, k, w = self.self_attn.indexer(hidden, qr, cos, sin, return_projection=True)
            self._kl = self.self_attn.indexer.scores(q, k, w).float().pow(2).mean()  # side-channel loss
            return self.mlp(hidden)  # frozen output: requires_grad=False

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._no_split_modules = ["DecoderLayer"]
            self.config = types.SimpleNamespace(dsa_enabled=dsa_enabled, tie_word_embeddings=False)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([DecoderLayer() for _ in range(2)])

        def forward(self, hidden, qr, cos, sin):
            for layer in self.model.layers:
                hidden = layer(hidden, qr, cos, sin)
            self._kl_total = sum(layer._kl for layer in self.model.layers)  # stashed; read after forward
            return hidden

    return Model()


def _grad_norm(p):
    if p.grad is None:
        return None
    g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
    return float(g.float().norm())


def _run(dsa_enabled, base_reshard):
    """Wrap with FSDP2, run one side-channel-KL backward, return #indexer masters that got a grad."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from verl.utils.fsdp_utils import apply_fsdp2

    torch.manual_seed(0)
    m = _build_model(dsa_enabled).cuda().bfloat16()
    for name, p in m.named_parameters():  # freeze base, train indexer (mirror freeze_base_train_indexer)
        p.requires_grad_(".indexer." in name)

    mesh = init_device_mesh("cuda", (1,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
    apply_fsdp2(m, dict(mesh=mesh, mp_policy=mp, offload_policy=None, reshard_after_forward=base_reshard), {})

    # Capture the SHARDED masters at BUILD time (before any forward) — exactly the objects the optimizer holds.
    # After a forward with reshard=False, module.named_parameters() would instead return the unsharded compute
    # copies (which autograd fills regardless), so reading those would mask the bug.
    masters = [(n, p) for n, p in m.named_parameters() if ".indexer." in n and p.requires_grad]
    assert masters, "no trainable indexer masters captured"

    B, S = 1, 6
    hidden = torch.randn(B, S, 32, device="cuda", dtype=torch.bfloat16)
    qr = torch.randn(B, S, 48, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(B, S, 16, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)

    out = m(hidden, qr, cos, sin)
    assert out.requires_grad is False, "frozen base should yield a non-grad module output (side-channel setup)"
    m._kl_total.backward()

    n_with_grad = sum(1 for _, p in masters if _grad_norm(p))
    return n_with_grad, len(masters)


@pytest.fixture(scope="module")
def _dist():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.cuda.set_device(0)
    torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    yield
    torch.distributed.destroy_process_group()


def test_b2_indexer_masters_receive_grad(_dist):
    """With B2 (indexer as own fully_shard unit), every indexer optimizer-master gets a grad from the
    side-channel KL — even with the frozen base resharded (reshard_after_forward=True)."""
    n_with_grad, total = _run(dsa_enabled=True, base_reshard=True)
    assert n_with_grad == total, f"B2: only {n_with_grad}/{total} indexer masters got a grad (expected all)"


def test_without_b2_masters_are_starved(_dist):
    """Sensitivity check: without B2 (indexer absorbed in the frozen decoder-layer unit), the side-channel KL
    never triggers the reduce-scatter, so the masters get NO grad — the bug B2 fixes. If this ever starts
    passing with grads, the test above is no longer proving anything."""
    n_with_grad, total = _run(dsa_enabled=False, base_reshard=False)
    assert n_with_grad == 0, f"expected 0/{total} indexer masters to have grad without B2, got {n_with_grad}"
