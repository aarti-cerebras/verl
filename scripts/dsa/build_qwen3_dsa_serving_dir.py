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
"""Qwen3-DSA Phase-2 FSDP2 checkpoint -> vLLM serving dir (serving_eval_plan.md §2, P2).

Reads the per-rank DTensor shards directly (reusing the verified reconstruction in
``consolidate_indexer_ckpt._full_from_shards``, so no ~8 GB intermediate ``.pt``) and writes a
directory ``scripts/dsa/vllm_qwen3_dsa`` can serve.

**Only two transforms, and neither renames a tensor.** Unlike the MSA exporter -- which has to strip
an ``.indexer.`` level and shift the main q/k norms by ``-1`` for M3's Gemma-convention fused kernel --
our serving module was written with the training module's exact parameter names and norm conventions,
so the weights go across verbatim:

1. **Consolidate** the 8 rank shards into full tensors and write safetensors.
2. **Write ``config.json``** with ``architectures = ["Qwen3DSAForCausalLM"]``, the ``dsa_*`` geometry,
   and **``index_topk``**. That last key is the documented sparse gate; the plugin hard-asserts it
   (memory ``dsa-serving-index-topk-gate`` -- its absence once cost a whole misread MiniCPM3 eval,
   because vLLM's own use of the key is a soft gate that silently falls back to dense).

**The `dsa_*` geometry must come from somewhere trustworthy.** Normally it is in the checkpoint's own
``huggingface/config.json``. Some runs leave that directory empty (the p2 run of 2026-08-20 did), and
guessing the geometry would produce a model that loads, generates fluent text and selects with the
wrong budget. So the CLI path is opt-in and loud: pass ``--dsa-from-cli`` plus the flags, and the
values are echoed into the manifest.

Example:
  python3 scripts/dsa/build_qwen3_dsa_serving_dir.py \
    --ckpt-dir /cb/ml-eng/aarti/dsa_qwen3/sparse/_ckpt/<run>/global_step_1200 \
    --base-model /cb/ml-eng/aarti/models/qwen3_4b_thinking_2507 \
    --out /cb/ml-eng/aarti/dsa_qwen3/serving/<run>_step1200 \
    --dsa-from-cli --top-k 2048 --n-heads 16 --head-dim 64
"""

import argparse
import glob
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # _dsa_log, consolidate_indexer_ckpt
from _dsa_log import setup_logging  # noqa: E402
from consolidate_indexer_ckpt import _full_from_shards, _resolve_step_dir  # noqa: E402

ARCH = "Qwen3DSAForCausalLM"
AUX_FILES = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "generation_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
]
# Per-layer indexer parameters, exactly as `Qwen3DSAIndexer` names them.
INDEXER_LEAVES = ("wq.weight", "q_norm.weight", "wk.weight", "k_norm.weight", "k_norm.bias",
                  "weights_proj.weight")


def load_full_state(step_dir: str, logger) -> dict[str, torch.Tensor]:
    """Consolidate ``model_world_size_W_rank_*.pt`` into full CPU tensors."""
    shards = sorted(
        glob.glob(os.path.join(step_dir, "model_world_size_*_rank_*.pt")),
        key=lambda p: int(p.rsplit("rank_", 1)[1].split(".")[0]),
    )
    assert shards, f"no model_world_size_*_rank_*.pt under {step_dir}"
    logger.info("reading %d rank shards", len(shards))
    per_rank = []
    for s in shards:
        per_rank.append(torch.load(s, map_location="cpu", weights_only=False))
        logger.info("  loaded %s (%d keys)", os.path.basename(s), len(per_rank[-1]))
    keys = list(per_rank[0].keys())
    for r, sd in enumerate(per_rank[1:], start=1):
        assert list(sd.keys()) == keys, f"rank {r} key set differs from rank 0"
    full = {}
    for k in keys:
        full[k] = _full_from_shards([sd[k] for sd in per_rank], k).to(torch.bfloat16).contiguous()
    return full


def check_indexer_complete(state: dict[str, torch.Tensor], n_layers: int, logger) -> None:
    """Every layer must carry every indexer tensor. A partial branch is the silent-dense failure."""
    missing = [
        f"model.layers.{i}.self_attn.indexer.{leaf}"
        for i in range(n_layers)
        for leaf in INDEXER_LEAVES
        if f"model.layers.{i}.self_attn.indexer.{leaf}" not in state
    ]
    assert not missing, (
        f"{len(missing)} indexer tensors missing from the checkpoint, e.g. {missing[:3]}. "
        "This is not a DSA Phase-2 checkpoint, or the copy is incomplete."
    )
    logger.info("indexer branch complete: %d layers x %d tensors", n_layers, len(INDEXER_LEAVES))


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-DSA FSDP2 ckpt -> vLLM serving dir")
    ap.add_argument("--ckpt-dir", required=True, help="a global_step_N dir, or a ckpt root (latest used)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", required=True,
                    help="stock Qwen3-4B-Thinking dir: source of config.json and the tokenizer")
    ap.add_argument("--arch", default=ARCH)
    ap.add_argument("--dsa-from-cli", action="store_true",
                    help="take the dsa_* geometry from the flags below instead of the checkpoint's "
                         "huggingface/config.json (required when that file is absent)")
    ap.add_argument("--top-k", type=int, default=2048)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--rope-head-dim", type=int, default=64)
    ap.add_argument("--no-fp8", action="store_true", help="checkpoint was trained with dsa_fp8=False")
    ap.add_argument("--no-rotate-activation", action="store_true")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    step_dir = _resolve_step_dir(args.ckpt_dir)
    os.makedirs(args.out, exist_ok=True)
    logger, _ = setup_logging("build_qwen3_dsa_serving_dir", args.log_dir or args.out)
    logger.info("config: %s", vars(args))
    logger.info("step_dir=%s", step_dir)

    # ---- base config -------------------------------------------------------------------------
    with open(os.path.join(args.base_model, "config.json")) as fh:
        cfg = json.load(fh)
    n_layers = int(cfg["num_hidden_layers"])

    # ---- dsa_* geometry ----------------------------------------------------------------------
    train_cfg_path = os.path.join(step_dir, "huggingface", "config.json")
    if os.path.exists(train_cfg_path):
        with open(train_cfg_path) as fh:
            train_cfg = json.load(fh)
        assert train_cfg.get("dsa_enabled"), f"{train_cfg_path} has no dsa_enabled"
        dsa = {k: v for k, v in train_cfg.items() if k.startswith("dsa_")}
        logger.info("dsa_* geometry taken from the checkpoint's own config: %s", dsa)
        if args.dsa_from_cli:
            for key, val in (("dsa_top_k", args.top_k), ("dsa_n_heads", args.n_heads),
                             ("dsa_head_dim", args.head_dim), ("dsa_rope_head_dim", args.rope_head_dim)):
                assert int(dsa.get(key, val)) == int(val), (
                    f"--dsa-from-cli disagrees with the checkpoint config: {key} "
                    f"{dsa.get(key)} != {val}. Refusing to guess which is right."
                )
    else:
        assert args.dsa_from_cli, (
            f"{train_cfg_path} is absent (the run did not save it, or the copy is incomplete), so "
            "the indexer geometry cannot be read from the checkpoint. Re-run with --dsa-from-cli "
            "and the explicit flags -- guessing it silently would serve the wrong selection budget."
        )
        dsa = {
            "dsa_enabled": True,
            "dsa_mode": "sparse",
            "dsa_top_k": args.top_k,
            "dsa_n_heads": args.n_heads,
            "dsa_head_dim": args.head_dim,
            "dsa_rope_head_dim": args.rope_head_dim,
            "dsa_fp8": not args.no_fp8,
            "dsa_fp8_ue8m0": True,
            "dsa_rotate_activation": not args.no_rotate_activation,
        }
        logger.warning("dsa_* geometry taken from the CLI (no checkpoint config): %s", dsa)

    assert dsa.get("dsa_mode") == "sparse", (
        f"dsa_mode={dsa.get('dsa_mode')!r}: only a Phase-2 (sparse) checkpoint is servable. A "
        "dense_warmup checkpoint computes the stock dense function and would score like the baseline."
    )

    # ---- weights ------------------------------------------------------------------------------
    state = load_full_state(step_dir, logger)
    check_indexer_complete(state, n_layers, logger)
    if cfg.get("tie_word_embeddings") and "lm_head.weight" in state:
        logger.info("dropping lm_head.weight (tied embeddings)")
        state.pop("lm_head.weight")
    n_params = sum(t.numel() for t in state.values())
    logger.info("writing %d tensors, %.3f B params", len(state), n_params / 1e9)
    save_file(state, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})

    # ---- config.json --------------------------------------------------------------------------
    cfg["architectures"] = [args.arch]
    cfg.update(dsa)
    # THE GATE. Mirrors dsa_top_k; the plugin asserts both presence and agreement.
    cfg["index_topk"] = int(dsa["dsa_top_k"])
    # Self-describing: tf5 keeps rope settings only under rope_parameters, so the flat key is
    # informational here (the plugin reads either).
    rope = cfg.get("rope_parameters") or {}
    if "rope_theta" in rope:
        cfg["rope_theta"] = rope["rope_theta"]
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    # ---- tokenizer / aux ----------------------------------------------------------------------
    copied = []
    for name in AUX_FILES:
        src = os.path.join(step_dir, "huggingface", name)
        if not os.path.exists(src):
            src = os.path.join(args.base_model, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, name))
            copied.append(name)
    logger.info("copied aux files: %s", copied)

    manifest = {
        "step_dir": step_dir,
        "base_model": args.base_model,
        "arch": args.arch,
        "dsa": dsa,
        "index_topk": cfg["index_topk"],
        "geometry_source": "checkpoint" if os.path.exists(train_cfg_path) else "cli",
        "n_tensors": len(state),
        "n_params": n_params,
    }
    with open(os.path.join(args.out, "BUILD_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("done: %s", args.out)
    logger.info("serve with: --block-size 64  (both the indexer kernel and the slot conversion assume it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
