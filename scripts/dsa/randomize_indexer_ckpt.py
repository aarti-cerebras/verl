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
"""DSA inference sanity-check: take a CONSOLIDATED full base+indexer checkpoint (the
``consolidated_model_stepN.pt`` from ``consolidate_indexer_ckpt.py --key-substr ""``) and overwrite ONLY the
``*.indexer.*`` weights with a FRESH, untrained ``LightningIndexer`` init (``reset_parameters``), leaving every
base weight bit-identical.

Why: a negative control for the serving path. The base model is unchanged, so if — and only if — vLLM's
``MiniCPM3DSAForCausalLM`` truly routes token selection through the indexer, replacing the trained indexer with
a random one should COLLAPSE accuracy on any benchmark whose prompt exceeds ``top_k`` (GSM8K few-shot ~1000
tokens >> top_k=128, so sparsity bites on 100% of items). If accuracy holds, the indexer is silently not
driving selection at inference — a bug. The trained k128 indexer scores 78.54 on GSM8K; a random indexer
should be far below that.

The random init is the model's OWN ``LightningIndexer.reset_parameters()`` (width-scaled normal, ~half-unit
variance) — i.e. exactly the untrained state the indexer had at the start of Phase-1 — cast to the checkpoint's
dtype. Deterministic under ``--seed`` (default 0).

Reproducible + logged (argv/cwd/host/git/env) via _dsa_log.
"""

import argparse
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for _dsa_log
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from _dsa_log import setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="overwrite indexer weights with fresh random init (inference sanity check)")
    ap.add_argument("--consolidated", required=True, help="consolidated_model_stepN.pt (full base+indexer)")
    ap.add_argument("--out", required=True, help="output .pt path (base weights unchanged, indexer randomized)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the fresh indexer init (deterministic)")
    ap.add_argument("--log-dir", default=None)
    # indexer dims (MiniCPM3-4B DSA sizing; must match the trained checkpoint's dsa_* config)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--rope-head-dim", type=int, default=32)
    ap.add_argument("--q-lora-rank", type=int, default=768)
    ap.add_argument("--hidden-size", type=int, default=2560)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger, _ = setup_logging("randomize_indexer_ckpt", args.log_dir or out_dir)
    logger.info("config: %s", vars(args))

    from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer

    logger.info("loading consolidated weights %s ...", args.consolidated)
    sd = torch.load(args.consolidated, weights_only=False, map_location="cpu")
    idx_keys = [k for k in sd if ".indexer." in k]
    assert idx_keys, "no *.indexer.* keys — is this a FULL (--key-substr '') consolidation?"
    layers = sorted({int(re.search(r"layers\.(\d+)\.", k).group(1)) for k in idx_keys})
    logger.info("found %d indexer keys across %d layers (%d..%d)", len(idx_keys), len(layers), layers[0], layers[-1])

    # one fresh LightningIndexer per layer, seeded deterministically. top_k is irrelevant to the weights.
    cfg = DSAConfig(enabled=True, n_heads=args.n_heads, head_dim=args.head_dim, rope_head_dim=args.rope_head_dim,
                    q_lora_rank=args.q_lora_rank, hidden_size=args.hidden_size, top_k=128)
    ref_keys = set(LightningIndexer(cfg).state_dict())

    torch.manual_seed(args.seed)
    n_replaced = 0
    for li in layers:
        prefix = f"model.layers.{li}.self_attn.indexer."
        present = {k[len(prefix):] for k in sd if k.startswith(prefix)}
        assert present == ref_keys, f"layer {li} indexer keys {present} != LightningIndexer keys {ref_keys}"
        fresh = LightningIndexer(cfg)  # calls reset_parameters() -> fresh random init
        fresh.reset_parameters()  # explicit re-draw (belt-and-suspenders; advances the seeded RNG per layer)
        fresh_sd = fresh.state_dict()
        for name, tensor in fresh_sd.items():
            key = prefix + name
            old = sd[key]
            sd[key] = tensor.to(dtype=old.dtype).contiguous()
            assert sd[key].shape == old.shape, f"{key}: {sd[key].shape} != {old.shape}"
            n_replaced += 1
    logger.info("replaced %d indexer tensors with fresh random init (seed=%d)", n_replaced, args.seed)

    torch.save(sd, args.out)
    logger.info("wrote %s", args.out)

    # verification: base bit-identical, indexer changed, all finite
    orig = torch.load(args.consolidated, weights_only=False, map_location="cpu")
    base_changed = [k for k in orig if ".indexer." not in k and not torch.equal(orig[k], sd[k])]
    assert not base_changed, f"base weights changed (must be identical): {base_changed[:5]}"
    idx_unchanged = [k for k in idx_keys if torch.equal(orig[k], sd[k])]
    assert not idx_unchanged, f"{len(idx_unchanged)} indexer tensors did NOT change: {idx_unchanged[:5]}"
    nonfinite = [k for k in sd if not torch.isfinite(sd[k]).all()]
    assert not nonfinite, f"non-finite tensors: {nonfinite[:5]}"
    logger.info("verify: %d base tensors bit-identical, all %d indexer tensors changed, all finite ✓",
                len(orig) - len(idx_keys), len(idx_keys))
    logger.info("verify: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
