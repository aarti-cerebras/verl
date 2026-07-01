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
"""DSA Phase-1 overfit-a-batch test: prove the indexer KL loss actually TRAINS the indexer.

On a tiny MiniCPM3 with the base frozen, repeatedly minimizing the indexer KL on ONE fixed batch must:
  (1) drive the KL down substantially (the indexer learns to match the frozen model's attention),
  (2) update ONLY the indexer params (base bit-identical; base grads stay None).

Requires CUDA (MiniCPMFlashAttention2 + flash_attn) and the transformers-4.57.1 env. Run with:
    PYTHONPATH=/tmp/fht_clean:/tmp/tf457lib pytest tests/models/test_minicpm_dsa_overfit.py -v -s
"""

import gc

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA (flash attention)")

MODEL = "openbmb/MiniCPM3-4B"
# fp8=False for a clean overfit demonstration (fp8 fake-quant adds a small noise floor; its parity with
# bf16 is covered by the Part-A tests).
DSA_OVERRIDES = {"n_heads": 4, "head_dim": 64, "top_k": 8, "mode": "dense_warmup", "fp8": False}


@pytest.fixture(autouse=True)
def _cuda_cleanup():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_and_patch():
    from transformers import AutoConfig, AutoModelForCausalLM

    from verl.models.transformers.monkey_patch import apply_monkey_patch

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.num_hidden_layers = 2
    cfg.vocab_size = 1000
    cfg._attn_implementation = "flash_attention_2"
    cfg.dsa_enabled = True
    cfg.dsa_overrides = dict(DSA_OVERRIDES)
    model = (
        AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
        )
        .to("cuda")
        .to(torch.bfloat16)
    )
    apply_monkey_patch(model=model, use_remove_padding=False, ulysses_sp_size=1)  # attaches indexer + freezes base
    return model


@requires_cuda
def test_freeze_only_indexer_trainable():
    model = _build_and_patch()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable params"
    assert all(".indexer." in n for n in trainable), f"non-indexer params are trainable: {trainable[:3]}"
    # a representative base param and the LM head are frozen
    assert not model.model.embed_tokens.weight.requires_grad
    assert not model.model.layers[0].self_attn.q_a_proj.weight.requires_grad
    # the indexer projections are trainable
    assert model.model.layers[0].self_attn.indexer.wq_b.weight.requires_grad


@requires_cuda
def test_overfit_one_batch_drives_kl_down_and_only_indexer_moves():
    torch.manual_seed(0)
    model = _build_and_patch()
    model.train()

    ids = torch.randint(0, 1000, (2, 16), device="cuda")
    pos = torch.arange(16, device="cuda").unsqueeze(0).expand(2, 16)

    # snapshots: a frozen base weight must not move; an indexer weight must move
    base_w = model.model.layers[0].self_attn.q_a_proj.weight.detach().clone()
    idx_w = model.model.layers[0].self_attn.indexer.wq_b.weight.detach().clone()

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)

    kls = []
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        model(input_ids=ids, position_ids=pos)  # hooks set model._dsa_indexer_kl
        loss = model._dsa_indexer_kl
        loss.backward()
        opt.step()
        kls.append(loss.item())

    initial = sum(kls[:5]) / 5
    final = sum(kls[-5:]) / 5
    print(f"\n[overfit] indexer KL: initial={initial:.4f} -> final={final:.4f} ({final / initial:.2%})")

    # (1) the KL dropped substantially on the fixed batch
    assert all(k == k for k in kls), "NaN in KL"  # no NaNs
    assert final < 0.6 * initial, f"indexer KL did not drop enough: {initial:.4f} -> {final:.4f}"

    # (2) only the indexer moved: frozen base weight unchanged, base grads None, indexer weight changed
    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_a_proj.weight.detach(), base_w, rtol=0, atol=0
    )
    assert model.model.layers[0].self_attn.q_a_proj.weight.grad is None
    assert not torch.allclose(model.model.layers[0].self_attn.indexer.wq_b.weight.detach(), idx_w)


@requires_cuda
def test_indexer_kl_loss_reads_module():
    """The indexer_kl_loss (bound to the model) returns the module's KL + metrics."""
    from verl.workers.utils.losses import indexer_kl_loss

    model = _build_and_patch()
    ids = torch.randint(0, 1000, (1, 16), device="cuda")
    pos = torch.arange(16, device="cuda").unsqueeze(0)
    with torch.no_grad():
        model(input_ids=ids, position_ids=pos)
    loss, metrics = indexer_kl_loss(config=None, model_output={}, data=None, model=model)
    assert torch.isfinite(loss) and loss.item() > 0
    assert "indexer/kl" in metrics
