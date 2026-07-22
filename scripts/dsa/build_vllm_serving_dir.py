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
"""DSA Phase-2 -> vLLM serving dir. Convert a CONSOLIDATED full base+indexer checkpoint (the
``consolidated_model_stepN.pt`` produced by ``consolidate_indexer_ckpt.py --key-substr ""``) into an
HF-loadable directory that vLLM's ``MiniCPM3DSAForCausalLM`` (Tier-3 plugin) can serve.

The naming maps 1:1 (verified): base weights are standard MiniCPM3/vLLM names, and the indexer weights use the
``LightningIndexer`` names (``self_attn.indexer.{wq_b,wk,k_norm,weights_proj}``) that our custom
``MiniCPM3DSAIndexer`` keeps separate (no ``wk_weights_proj`` fusion). So this is a pure format conversion +
config/tokenizer assembly — no tensor renaming.

Output dir contains: ``model.safetensors`` (bf16), ``config.json`` (architectures ->
``MiniCPM3DSAForCausalLM``, auto_map trimmed to AutoConfig, existing ``dsa_*`` fields preserved), and the
tokenizer / chat-template / ``configuration_minicpm.py`` copied from the training ckpt's ``huggingface/`` dir.

Reproducible + logged (argv/cwd/host/git/env) via _dsa_log.
"""

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for _dsa_log
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from _dsa_log import setup_logging  # noqa: E402

# tokenizer / config aux files copied verbatim from the training ckpt's huggingface/ dir
AUX_FILES = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "generation_config.json", "chat_template.jinja", "configuration_minicpm.py",
]


def main():
    ap = argparse.ArgumentParser(description="consolidated .pt -> vLLM HF serving dir (MiniCPM3DSA)")
    ap.add_argument("--consolidated", required=True, help="consolidated_model_stepN.pt (full base+indexer)")
    ap.add_argument("--hf-src", required=True, help="the training ckpt's huggingface/ dir (config + tokenizer)")
    ap.add_argument("--out", required=True, help="output serving dir")
    ap.add_argument("--arch", default="MiniCPM3DSAForCausalLM", help="architectures[0] the vLLM plugin registers")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    logger, _ = setup_logging("build_vllm_serving_dir", args.log_dir or args.out)
    logger.info("config: %s", vars(args))

    # 1) weights: load consolidated .pt (dict of bf16 cpu tensors), write a single model.safetensors.
    logger.info("loading consolidated weights %s ...", args.consolidated)
    sd = torch.load(args.consolidated, weights_only=False, map_location="cpu")
    sd = {k: v.contiguous() for k, v in sd.items()}  # safetensors requires contiguous, no shared storage
    n_idx = sum(1 for k in sd if ".indexer." in k)
    logger.info("loaded %d tensors (%d indexer), dtypes=%s", len(sd), n_idx, {str(v.dtype) for v in sd.values()})
    assert n_idx > 0, "no *.indexer.* keys — is this a FULL (--key-substr '') consolidation?"
    st_path = os.path.join(args.out, "model.safetensors")
    save_file(sd, st_path, metadata={"format": "pt"})
    logger.info("wrote %s (%.2f GB)", st_path, os.path.getsize(st_path) / 1e9)

    # 2) config.json: base config + arch swap + trimmed auto_map (dsa_* fields already present, kept as-is).
    with open(os.path.join(args.hf_src, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["architectures"] = [args.arch]
    cfg["auto_map"] = {"AutoConfig": "configuration_minicpm.MiniCPM3Config"}  # drop verl FSDP modeling entries
    assert cfg.get("dsa_enabled"), "source config has no dsa_enabled — not a DSA checkpoint?"
    logger.info("config: arch->%s dsa_mode=%s dsa_top_k=%s dsa_head_dim=%s dsa_n_heads=%s",
                args.arch, cfg.get("dsa_mode"), cfg.get("dsa_top_k"), cfg.get("dsa_head_dim"), cfg.get("dsa_n_heads"))
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # 3) tokenizer / chat template / configuration_minicpm.py
    copied = []
    for f in AUX_FILES:
        src = os.path.join(args.hf_src, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, f))
            copied.append(f)
    logger.info("copied aux files: %s", copied)

    if not args.no_verify:
        _verify(args, logger, sd, st_path)
    logger.info("DONE. serve with: PYTHONPATH=<plugin>:.devlibs/tf457lib python3 -m vllm ... --model %s", args.out)


def _verify(args, logger, sd, st_path):
    from safetensors import safe_open
    with safe_open(st_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
    assert keys == set(sd), f"safetensors keys != source | missing={set(sd) - keys} extra={keys - set(sd)}"
    # spot-check a base + an indexer tensor round-trips bit-identically
    with safe_open(st_path, framework="pt", device="cpu") as f:
        for k in ["model.embed_tokens.weight", "model.layers.0.self_attn.indexer.wq_b.weight"]:
            assert torch.equal(f.get_tensor(k), sd[k]), f"{k}: safetensors round-trip mismatch"
    for f in ["config.json", "tokenizer.json", "configuration_minicpm.py"]:
        assert os.path.exists(os.path.join(args.out, f)), f"missing {f} in serving dir"
    logger.info("verify: %d safetensors keys == source, round-trip OK, config+tokenizer present ✓", len(keys))


if __name__ == "__main__":
    main()
