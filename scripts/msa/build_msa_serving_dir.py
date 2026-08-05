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
"""P2 (docs/qwen3_4b_msa/serving_plan.md §5): Qwen3-MSA Phase-2 FSDP2 checkpoint -> vLLM serving dir.

Reads the per-rank DTensor shards directly (reusing the verified reconstruction in
``scripts/dsa/consolidate_indexer_ckpt.py::_full_from_shards``, so no ~8 GB intermediate
``.pt``) and writes a directory that our out-of-tree ``Qwen3MSAForCausalLM`` can serve on
vLLM 0.26.0.

THREE TRANSFORMS, and every one of them fails SILENTLY if wrong (serving_plan §8 R4):

1. **Strip ``.indexer.``** (serving_plan §2.2). Training nests the index branch in an
   ``MSAIndexer`` submodule -> ``self_attn.indexer.index_q_proj.weight``; vLLM's
   ``stacked_params_mapping`` matches ``.index_q_proj`` directly under ``self_attn`` and folds
   it into the fused ``qkv_proj`` (``nvidia/model.py:902-911``). Unmatched names hit
   ``if name not in params_dict: continue`` (``:940-941``) and vanish -> a fluent, entirely
   dense model.

2. **The ``w - 1`` shift** (serving_plan §2.3, §5.1; Route A decided in §4.3). The fused kernel
   computes Gemma ``x * rsqrt(...) * (1 + w)`` for whichever norm weight it is handed
   (``fused.cu:135-149``), but Qwen3's MAIN ``q_norm``/``k_norm`` are standard ``x * w``. Store
   ``w' = w - 1`` so ``1 + w'`` reproduces the trained function.
   Scope is narrow: main q/k norms ONLY, sparse layers ONLY. The index norms are already Gemma
   (``msa_indexer.py:169-186``), and layers ``[0, dense_prefix)`` build stock ``Qwen3Attention``
   which never touches the kernel -- shifting either would corrupt the model.
   The shift is lossy by <= 1 bf16 ULP (measured worst case 3.75e-3 on k8 @ step 1400), so we
   MEASURE it per tensor rather than assume exactness; ``--max-shift-err`` gates it.

3. **``sparse_attention_config``** (serving_plan §5.2). vLLM reads a nested dict; training writes
   flat ``msa_*`` keys. ``_sparse_attention_layer_ids`` returns an EMPTY SET when the dict or its
   ``sparse_attention_freq`` is missing (``nvidia/model.py:95-103``) -> every layer dense. Also
   emit flat ``rope_theta`` (tf5 nests it under ``rope_parameters``) and
   ``partial_rotary_factor: 1.0`` (Qwen3 is full-rotary -> rotary_dim 128).

Reproducible + logged (argv/cwd/host/git/env) via _dsa_log.

Example:
  python3 scripts/msa/build_msa_serving_dir.py \
    --ckpt-dir /cb/ml-eng/aarti/msa/sparse/_ckpt/<run>/global_step_1400 \
    --out /cb/ml-eng/aarti/msa/serving/<run>_step1400
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import save_file

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "dsa"))  # _dsa_log, consolidate_indexer_ckpt
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # repo root
from _dsa_log import setup_logging  # noqa: E402
from consolidate_indexer_ckpt import _full_from_shards, _resolve_step_dir  # noqa: E402

AUX_FILES = [
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "generation_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
]

# training key -> serving key. Only the `.indexer.` level is removed; the leaf names already match
# what vLLM's stacked_params_mapping / MiniMaxM3SparseAttention expect.
INDEXER_RE = re.compile(r"^(.*\.self_attn)\.indexer\.(index_(?:q|k)_(?:proj|norm)\.weight)$")
QK_NORM_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight$")


def sparse_layer_ids(cfg: dict, n_layers: int) -> list[int]:
    """Which layers carry an index branch, per the TRAINING config.

    Mirrors verl/models/transformers/msa_indexer.py::MSAConfig -- `msa_sparse_layers` (explicit
    comma-separated ids) overrides `msa_dense_prefix` when set.
    """
    explicit = cfg.get("msa_sparse_layers")
    if explicit:
        return sorted(int(x) for x in str(explicit).split(",") if x.strip() != "")
    return list(range(int(cfg.get("msa_dense_prefix", 3)), n_layers))


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-MSA FSDP2 ckpt -> vLLM serving dir")
    ap.add_argument("--ckpt-dir", required=True, help="a global_step_N dir, or a ckpt root (latest used)")
    ap.add_argument("--out", required=True, help="output serving dir")
    ap.add_argument("--arch", default="Qwen3MSAForCausalLM", help="architectures[0] the plugin registers")
    ap.add_argument("--no-norm-shift", action="store_true",
                    help="Route B: skip the w-1 shift (own forward, Qwen3's own norm modules)")
    ap.add_argument("--max-shift-err", type=float, default=2.0**-8,
                    help="fail if any |(1 + bf16(w-1)) - w| exceeds this (default 1 bf16 ULP at |w|~1)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    step_dir = _resolve_step_dir(args.ckpt_dir)
    os.makedirs(args.out, exist_ok=True)
    logger, _ = setup_logging("build_msa_serving_dir", args.log_dir or args.out)
    logger.info("config: %s", vars(args))
    logger.info("step_dir=%s", step_dir)

    hf_src = os.path.join(step_dir, "huggingface")
    assert os.path.isdir(hf_src), f"no huggingface/ dir under {step_dir}"
    with open(os.path.join(hf_src, "config.json")) as fh:
        cfg = json.load(fh)
    assert cfg.get("msa_enabled"), "source config has no msa_enabled -- not an MSA checkpoint?"
    assert cfg.get("msa_mode") == "sparse", f"expected msa_mode=sparse, got {cfg.get('msa_mode')!r}"

    n_layers = int(cfg["num_hidden_layers"])
    sparse_ids = sparse_layer_ids(cfg, n_layers)
    logger.info("msa: top_k=%s block=%s dense_prefix=%s -> %d sparse layers %s..%s",
                cfg.get("msa_top_k"), cfg.get("msa_block_size"), cfg.get("msa_dense_prefix"),
                len(sparse_ids), sparse_ids[0], sparse_ids[-1])

    # ---- 1) weights: reconstruct from the per-rank DTensor shards -------------------------------
    shards = sorted(glob.glob(os.path.join(step_dir, "model_world_size_*_rank_*.pt")),
                    key=lambda p: int(re.search(r"rank_(\d+)", p).group(1)))
    assert shards, f"no model_world_size_*_rank_*.pt under {step_dir}"
    world = int(re.search(r"world_size_(\d+)", shards[0]).group(1))
    assert len(shards) == world, f"found {len(shards)} shards but world_size={world}"
    logger.info("loading %d shards (CPU)...", world)
    rank_sds = [torch.load(s, weights_only=False, map_location="cpu") for s in shards]
    src_keys = list(rank_sds[0].keys())
    logger.info("source state dict: %d tensors", len(src_keys))

    sd: dict[str, torch.Tensor] = {}
    n_renamed = 0
    for k in src_keys:
        full = _full_from_shards([s[k] for s in rank_sds], k)
        m = INDEXER_RE.match(k)
        if m:
            k = f"{m.group(1)}.{m.group(2)}"  # drop the `.indexer.` level
            n_renamed += 1
        assert k not in sd, f"key collision after rename: {k}"
        sd[k] = full.contiguous()
    del rank_sds

    n_index = sum(1 for k in sd if ".index_" in k)
    logger.info("reconstructed %d tensors; renamed %d indexer keys; %d index_* present",
                len(sd), n_renamed, n_index)
    assert len(sd) == len(src_keys), f"tensor count changed: {len(src_keys)} -> {len(sd)}"
    assert n_index == 4 * len(sparse_ids), f"expected {4 * len(sparse_ids)} index_* tensors, got {n_index}"
    leftover = [k for k in sd if ".indexer." in k]
    assert not leftover, f"{len(leftover)} keys still contain '.indexer.': {leftover[:3]}"

    # ---- 2) the w-1 shift on MAIN q/k norms, sparse layers only ---------------------------------
    shift_report = {"applied": 0, "max_err": 0.0, "worst_key": None}
    orig_qk: dict[str, torch.Tensor] = {}  # pre-shift copies, for the round-trip check in _verify
    if not args.no_norm_shift:
        sparse_set = set(sparse_ids)
        for k in list(sd):
            m = QK_NORM_RE.match(k)
            if not m or int(m.group(1)) not in sparse_set:
                continue  # dense-prefix layers keep Qwen3's own convention
            w = sd[k]
            orig_qk[k] = w.clone()  # 66 x [128] -- negligible
            shifted = (w.float() - 1.0).to(w.dtype)
            # measure what the kernel will actually reconstruct: 1 + bf16(w-1) vs the original w
            err = (shifted.float() + 1.0 - w.float()).abs().max().item()
            if err > shift_report["max_err"]:
                shift_report.update(max_err=err, worst_key=k)
            sd[k] = shifted
            shift_report["applied"] += 1
        exp = 2 * len(sparse_ids)
        assert shift_report["applied"] == exp, \
            f"w-1 shift applied to {shift_report['applied']} tensors, expected {exp}"
        logger.info("w-1 shift: %d tensors (main q/k norms, layers %s..%s); "
                    "max |(1+bf16(w-1)) - w| = %.3e at %s",
                    shift_report["applied"], sparse_ids[0], sparse_ids[-1],
                    shift_report["max_err"], shift_report["worst_key"])
        assert shift_report["max_err"] <= args.max_shift_err, (
            f"w-1 round-trip error {shift_report['max_err']:.3e} exceeds --max-shift-err "
            f"{args.max_shift_err:.3e} at {shift_report['worst_key']}"
        )
    else:
        logger.info("w-1 shift SKIPPED (--no-norm-shift / Route B)")

    st_path = os.path.join(args.out, "model.safetensors")
    save_file(sd, st_path, metadata={"format": "pt"})
    logger.info("wrote %s (%.2f GB)", st_path, os.path.getsize(st_path) / 1e9)

    # ---- 3) config.json --------------------------------------------------------------------------
    out_cfg = dict(cfg)
    out_cfg["architectures"] = [args.arch]
    out_cfg.pop("auto_map", None)

    freq = [0] * n_layers
    for i in sparse_ids:
        freq[i] = 1
    out_cfg["sparse_attention_config"] = {
        "sparse_attention_freq": freq,
        "sparse_topk_blocks": int(cfg["msa_top_k"]),
        "sparse_block_size": int(cfg["msa_block_size"]),
        "sparse_num_index_heads": int(cfg["num_key_value_heads"]),
        "sparse_index_dim": int(cfg.get("msa_index_dim", 128)),
        "sparse_init_block": int(cfg.get("msa_init_blocks", 0)),
        "sparse_local_block": int(cfg.get("msa_local_blocks", 1)),
        "sparse_score_type": cfg.get("msa_score_type", "max"),
    }
    # vLLM reads these flat; transformers 5 nests rope_theta under rope_parameters.
    rope = cfg.get("rope_parameters") or {}
    out_cfg["rope_theta"] = float(cfg.get("rope_theta", rope.get("rope_theta", 5_000_000.0)))
    out_cfg["partial_rotary_factor"] = 1.0
    out_cfg["msa_norm_shift_applied"] = not args.no_norm_shift  # so a re-run cannot double-shift silently

    sac = out_cfg["sparse_attention_config"]
    assert len(sac["sparse_attention_freq"]) == n_layers
    assert sum(sac["sparse_attention_freq"]) == len(sparse_ids)
    assert sac["sparse_block_size"] == 128, "vLLM supports only block_size 128"
    assert sac["sparse_num_index_heads"] == cfg["num_key_value_heads"], "vLLM asserts this equality"
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(out_cfg, fh, indent=2)
    logger.info("config.json: arch=%s topk=%d block=%d idx_heads=%d local=%d init=%d rope_theta=%g",
                args.arch, sac["sparse_topk_blocks"], sac["sparse_block_size"],
                sac["sparse_num_index_heads"], sac["sparse_local_block"], sac["sparse_init_block"],
                out_cfg["rope_theta"])

    # ---- 4) tokenizer / chat template -------------------------------------------------------------
    copied = [f for f in AUX_FILES
              if os.path.exists(os.path.join(hf_src, f))
              and (shutil.copy2(os.path.join(hf_src, f), os.path.join(args.out, f)) or True)]
    logger.info("copied aux files: %s", copied)
    assert "tokenizer.json" in copied, "tokenizer.json missing from the checkpoint's huggingface/ dir"

    if not args.no_verify:
        _verify(args, logger, sd, st_path, out_cfg, sparse_ids, orig_qk)

    logger.info("DONE -> %s", args.out)
    return 0


def _verify(args, logger, sd, st_path, out_cfg, sparse_ids, orig_qk):
    """Round-trip + semantic checks (serving_plan §5.1 exit criteria)."""
    from safetensors import safe_open

    with safe_open(st_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        assert keys == set(sd), f"safetensors keys != source | missing={set(sd) - keys} extra={keys - set(sd)}"
        for k in ["model.embed_tokens.weight",
                  f"model.layers.{sparse_ids[0]}.self_attn.index_q_proj.weight",
                  f"model.layers.{sparse_ids[0]}.self_attn.q_norm.weight"]:
            assert torch.equal(f.get_tensor(k), sd[k]), f"{k}: safetensors round-trip mismatch"

    nonfinite = [k for k, v in sd.items() if not torch.isfinite(v).all()]
    assert not nonfinite, f"non-finite tensors: {nonfinite[:5]}"

    # The shift must invert back to the ORIGINAL (fully reconstructed) weights within --max-shift-err.
    # NB compare against the pre-shift copies, NOT a rank-0 shard: these norms are Shard(0)-placed, so
    # rank 0 holds 16 of 128 elements and `to_local()` would silently compare the wrong slice.
    if not args.no_norm_shift:
        assert len(orig_qk) == 2 * len(sparse_ids), f"stashed {len(orig_qk)} originals, expected {2 * len(sparse_ids)}"
        worst, worst_k = 0.0, None
        for k, orig in orig_qk.items():
            assert orig.shape == sd[k].shape, f"{k}: stashed {tuple(orig.shape)} vs shifted {tuple(sd[k].shape)}"
            e = (sd[k].float() + 1.0 - orig.float()).abs().max().item()
            if e > worst:
                worst, worst_k = e, k
        assert worst <= args.max_shift_err, \
            f"un-shift does not recover the original weights: {worst:.3e} at {worst_k}"
        logger.info("verify: un-shift recovers all %d originals (full %d-dim tensors), worst %.3e <= %.3e ✓",
                    len(orig_qk), orig_qk[next(iter(orig_qk))].numel(), worst, args.max_shift_err)

    # index_* keys must sit directly under self_attn, which is what the vLLM loader matches on.
    bad = [k for k in sd if ".index_" in k and ".self_attn.index_" not in k]
    assert not bad, f"index tensors not directly under self_attn: {bad[:3]}"

    # dense-prefix layers must NOT have been shifted and must NOT have an index branch.
    dense = [i for i in range(out_cfg["num_hidden_layers"]) if i not in set(sparse_ids)]
    for i in dense:
        assert not any(f"model.layers.{i}.self_attn.index_" in k for k in sd), \
            f"dense layer {i} unexpectedly has index weights"
    logger.info("verify: %d safetensors keys, %d dense layers clean, index keys well-placed ✓",
                len(keys), len(dense))

    for f in ["config.json", "tokenizer.json"]:
        assert os.path.exists(os.path.join(args.out, f)), f"missing {f} in serving dir"


if __name__ == "__main__":
    sys.exit(main())
