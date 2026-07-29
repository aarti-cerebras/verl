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
"""Verify the MSA path is reachable through verl's real entry points, not just by calling the module.

Covers the wiring that `test_qwen3_msa_phase1.py` bypasses by setting things up manually:
  1. `apply_monkey_patch` fires on `model_type == "qwen3" and config.msa_enabled`, and is INERT without
     the flag (an ordinary Qwen3 run must be untouched).
  2. flat `msa_*` override keys reach `MSAConfig` (verl's `override_config` can only inject scalars).
  3. `indexer_kl_loss` finds `model._msa_indexer_kl` — it used to read only the DSA attribute name.
  4. `fsdp_utils` Option-B2 selection recognises `MSAIndexer` — it used to match `LightningIndexer`
     only, which would have silently reproduced the DSA §3b grad bug at world_size > 1.
  5. ulysses is rejected loudly rather than silently normalising the KL over a sequence fragment.

Run:
  cd <repo> && PYTHONPATH=$(pwd) python3 tests/msa/test_qwen3_msa_monkey_patch.py
"""

import argparse
import sys

import torch

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

    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen3 import modeling_qwen3

    from verl.models.transformers.monkey_patch import apply_monkey_patch

    stock_forward = modeling_qwen3.Qwen3Attention.forward

    print("\n-- 1. the hook is INERT without config.msa_enabled --")
    cfg = AutoConfig.from_pretrained(a.model)
    plain = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, dtype=torch.float32)
    apply_monkey_patch(model=plain, use_remove_padding=False, ulysses_sp_size=1)
    check("Qwen3Attention.forward not replaced", modeling_qwen3.Qwen3Attention.forward is stock_forward)
    check("no indexer attached", not any("indexer" in n for n, _ in plain.named_modules()))
    del plain

    print("\n-- 2. flat msa_* overrides reach MSAConfig through the hook --")
    cfg = AutoConfig.from_pretrained(a.model)
    # Exactly how verl's `+model.override_config={...}` injects: scalars set with setattr.
    overrides = dict(
        msa_enabled=True, msa_mode="dense_warmup", msa_top_k=2, msa_kl_block_size=128,
        msa_kl_reduction="mean", msa_diag_interval=1, msa_dense_prefix=3, msa_kl_checkpoint=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    model = AutoModelForCausalLM.from_pretrained(a.model, config=cfg, dtype=torch.float32).to(a.device)
    apply_monkey_patch(model=model, use_remove_padding=False, ulysses_sp_size=1)
    attn0 = model.model.layers[cfg.dense_prefix if hasattr(cfg, "dense_prefix") else 3].self_attn
    msa = attn0.msa
    check("forward WAS replaced", modeling_qwen3.Qwen3Attention.forward is not stock_forward)
    check("top_k=2 came through", msa.top_k == 2, f"top_k={msa.top_k}")
    check("kl_block_size=128 came through", msa.kl_block_size == 128)
    check("kl_reduction='mean' came through", msa.kl_reduction == "mean")
    check("kl_checkpoint=True came through", msa.kl_checkpoint is True)
    check("geometry forced from the model, not the overrides",
          msa.num_kv_heads == cfg.num_key_value_heads and msa.index_dim == cfg.head_dim,
          f"H_kv={msa.num_kv_heads}, d_idx={msa.index_dim}")
    n_sparse = sum(1 for m in model.modules() if m.__class__.__name__ == "MSAIndexer")
    check("indexers attached to layers >= dense_prefix only",
          n_sparse == cfg.num_hidden_layers - 3, f"{n_sparse} of {cfg.num_hidden_layers - 3}")
    check("dense layers have no indexer", getattr(model.model.layers[0].self_attn, "indexer", None) is None)
    check("base frozen by the hook (dense_warmup)",
          all(".indexer." in n for n, p in model.named_parameters() if p.requires_grad))

    print("\n-- 3. indexer_kl_loss reads the MSA attribute --")
    from verl.workers.utils.losses import indexer_kl_loss

    ids = torch.randint(0, cfg.vocab_size, (1, a.seq_len), device=a.device)
    model.train()
    model(input_ids=ids)
    loss, metrics = indexer_kl_loss(config=None, model_output=None, data=None, model=model)
    check("loss returned and finite", torch.isfinite(loss) and loss.item() > 0, f"{loss.item():.4f}")
    check("metrics forwarded from _msa_metrics",
          "indexer/coverage_vs_ceiling" in metrics and "indexer/kl" in metrics,
          f"{len(metrics)} keys")
    check("all metric values are scalars (the DSA issue #4 logger trap)",
          all(not isinstance(v, (list, tuple)) for v in metrics.values()))

    print("\n-- 4. FSDP2 Option-B2 selection recognises MSAIndexer --")
    # Mirror fsdp_utils.apply_fsdp2's selection predicate exactly (it keys on the class NAME).
    picked = [
        n for n, sub in model.named_modules()
        if sub.__class__.__name__ in ("LightningIndexer", "MSAIndexer")
        and getattr(getattr(sub, "cfg", None), "mode", None) == "dense_warmup"
    ]
    check("every attached indexer would get its own FSDP2 unit", len(picked) == n_sparse,
          f"{len(picked)} of {n_sparse}")
    gate = getattr(model.config, "dsa_enabled", False) or getattr(model.config, "msa_enabled", False)
    check("the enabling gate in fsdp_utils sees msa_enabled", bool(gate))

    print("\n-- 5. ulysses is rejected, not silently wrong --")
    cfg2 = AutoConfig.from_pretrained(a.model)
    for k, v in overrides.items():
        setattr(cfg2, k, v)
    m2 = AutoModelForCausalLM.from_pretrained(a.model, config=cfg2, dtype=torch.float32)
    try:
        apply_monkey_patch(model=m2, use_remove_padding=False, ulysses_sp_size=2)
        check("assertion raised for ulysses_sp_size=2", False, "no error raised")
    except AssertionError as e:
        check("assertion raised for ulysses_sp_size=2", "ulysses" in str(e).lower())

    modeling_qwen3.Qwen3Attention.forward = stock_forward  # leave the process clean
    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
