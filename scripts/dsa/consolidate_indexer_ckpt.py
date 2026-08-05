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
"""DSA Phase-2 warm-start prep: consolidate the trained lightning-indexer weights from a Phase-1 (dense-warmup)
verl FSDP checkpoint into a SINGLE, world-size-AGNOSTIC file (``indexer_full.pt``).

Why: a verl FSDP checkpoint is stored as per-rank DTensor shards (``model_world_size_{W}_rank_{r}.pt``), so a
native resume is locked to W GPUs. Phase-1 froze the base and trained only the indexer, so the only weights
worth carrying into Phase-2 are the ``*.indexer.*`` tensors. This script reconstructs the full (unsharded)
indexer tensors from the W shards WITHOUT a GPU / distributed context (it reads each DTensor's local shard and
reassembles per its placement), so the result can be warm-loaded into a Phase-2 run on ANY number of GPUs
(see the ``dsa_indexer_init_path`` hook in verl/models/transformers/minicpm_dsa.py::attach_indexers).

Reproducible + logged (argv/cwd/host/git/env) via _dsa_log.
"""

import argparse
import glob
import os
import re
import sys

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for _dsa_log
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root, for `verl`
from _dsa_log import setup_logging  # noqa: E402


def _resolve_step_dir(ckpt_dir: str) -> str:
    """Accept either a global_step_N dir directly, or a 'checkpoints' dir (then pick the latest step)."""
    if os.path.basename(ckpt_dir.rstrip("/")).startswith("global_step_"):
        return ckpt_dir
    latest = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    if os.path.exists(latest):
        with open(latest) as fh:
            step = fh.read().strip()
        cand = os.path.join(ckpt_dir, f"global_step_{step}")
        if os.path.isdir(cand):
            return cand
    steps = sorted(glob.glob(os.path.join(ckpt_dir, "global_step_*")),
                   key=lambda p: int(re.search(r"global_step_(\d+)", p).group(1)))
    assert steps, f"no global_step_* under {ckpt_dir}"
    return steps[-1]


def _full_from_shards(dtensors, key):
    """Reconstruct the full (unsharded) tensor from the W per-rank DTensor shards of one param.

    Generic over the placement: Replicate -> take any rank's local; Shard(d) -> concat the local shards
    (in rank order) along dim d and trim FSDP's padding back to the global shape. Asserts the result matches
    the DTensor's advertised global shape.
    """
    d0 = dtensors[0]
    if not isinstance(d0, DTensor):  # already a plain tensor (unsharded) -> use as-is
        return d0
    assert len(d0.placements) == 1, f"{key}: expected 1D mesh, got placements={d0.placements}"
    global_shape = tuple(d0.shape)
    p = d0.placements[0]
    if isinstance(p, Replicate):
        full = d0._local_tensor.clone()
    elif isinstance(p, Shard):
        dim = p.dim
        full = torch.cat([d._local_tensor for d in dtensors], dim=dim)
        full = full.narrow(dim, 0, global_shape[dim]).clone()  # drop FSDP even-shard padding
    else:
        raise NotImplementedError(f"{key}: unsupported placement {p}")
    assert tuple(full.shape) == global_shape, f"{key}: reconstructed {tuple(full.shape)} != global {global_shape}"
    return full


def main():
    ap = argparse.ArgumentParser(description="consolidate Phase-1 indexer weights -> world-size-agnostic file")
    ap.add_argument("--ckpt-dir", required=True, help="a global_step_N dir, or a 'checkpoints' dir (latest used)")
    ap.add_argument("--out", default=None, help="output .pt path (dict of {param_name: full cpu tensor}). "
                    "Default: written INTO the resolved global_step_N shard folder as "
                    "consolidated_indexer_stepN.pt (indexer-only) / consolidated_model_stepN.pt (full). "
                    "The run log is written to the same folder.")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--key-substr", default=".indexer.",
                    help="only keep params whose name contains this. Default '.indexer.' => Phase-1 warm-start "
                         "(base stays stock). Pass '' to consolidate the FULL model => Phase-2 branch/restart "
                         "(base+indexer). Either output loads via dsa_warmstart_path (strict=False).")
    ap.add_argument("--no-verify", action="store_true", help="skip the post-write verification pass")
    ap.add_argument("--arch", choices=("auto", "dsa", "msa"), default="auto",
                    help="which indexer the checkpoint holds, for the key-match check. 'auto' (default) "
                         "infers it by matching the first sparse layer's key set against both "
                         "LightningIndexer (DSA) and MSAIndexer (MSA).")
    # dims for the LightningIndexer key-match check (defaults = MiniCPM3-4B DSA sizing; MSA ignores these,
    # since only key NAMES are compared and those are geometry-independent)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--rope-head-dim", type=int, default=32)
    ap.add_argument("--q-lora-rank", type=int, default=768)
    ap.add_argument("--hidden-size", type=int, default=2560)
    args = ap.parse_args()

    step_dir = _resolve_step_dir(args.ckpt_dir)
    m = re.search(r"global_step_(\d+)", step_dir)
    step_num = m.group(1) if m else "unknown"
    if args.out is None:  # default: co-locate with the source shards (the global_step_* folder), tagged with step
        tag = "indexer" if args.key_substr == ".indexer." else "model"
        args.out = os.path.join(step_dir, f"consolidated_{tag}_step{step_num}.pt")
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger, _ = setup_logging("consolidate_indexer_ckpt", args.log_dir or out_dir)  # log to the same folder as --out
    logger.info("config: %s", vars(args))

    shards = sorted(glob.glob(os.path.join(step_dir, "model_world_size_*_rank_*.pt")),
                    key=lambda p: int(re.search(r"rank_(\d+)", p).group(1)))
    assert shards, f"no model_world_size_*_rank_*.pt under {step_dir}"
    world = int(re.search(r"world_size_(\d+)", shards[0]).group(1))
    assert len(shards) == world, f"found {len(shards)} shards but world_size={world}"
    logger.info("step_dir=%s world_size=%d (%d shards)", step_dir, world, len(shards))

    logger.info("loading %d shards (CPU)...", world)
    rank_sds = [torch.load(s, weights_only=False, map_location="cpu") for s in shards]

    keys = [k for k in rank_sds[0] if args.key_substr in k]
    logger.info("consolidating %d params matching %r (of %d total)", len(keys), args.key_substr, len(rank_sds[0]))

    out = {}
    for k in keys:
        out[k] = _full_from_shards([sd[k] for sd in rank_sds], k)
    total = sum(v.numel() for v in out.values())
    logger.info("reconstructed %d params, %.2fM elements; sample: %s %s",
                len(out), total / 1e6, keys[0], tuple(out[keys[0]].shape))

    torch.save(out, args.out)
    logger.info("wrote %s (world-size-agnostic; load via dsa_warmstart_path)", args.out)

    if not args.no_verify:
        _verify(args, logger, out, rank_sds, keys)


def _verify(args, logger, out, rank_sds, keys):
    """Post-write verification (logged, reproducible). Fails loudly on any problem:
      1. save/load round-trip — reload the file and confirm identical to what we wrote;
      2. sanity — nonzero count, every key matches --key-substr, all tensors finite;
      3. reconstruction — for a sample of keys, re-concat the raw rank shards and confirm == the saved tensor
         (also confirms the saved shape == the DTensor's global shape);
      4. key-match — for an indexer-only consolidation, the keys of the FIRST layer that actually has an
         indexer must exactly equal a fresh indexer's state_dict, so the warmstart_path strict=False load
         maps with no unexpected keys. The reference class is chosen per architecture: DSA's
         LightningIndexer or MSA's MSAIndexer. Note MSA leaves layers [0, dense_prefix) dense (M3 ships
         dense_prefix=3), so layer 0 has no indexer and must not be assumed to.
    """
    from torch.distributed.tensor.placement_types import Shard

    logger.info("verify: reloading %s ...", args.out)
    reloaded = torch.load(args.out, weights_only=False, map_location="cpu")
    assert set(reloaded) == set(out) and all(torch.equal(reloaded[k], out[k]) for k in out), "save/load round-trip mismatch"
    assert len(out) > 0, "empty consolidation"
    assert all(args.key_substr in k for k in out), "a key does not contain --key-substr"
    nonfinite = [k for k, v in out.items() if not torch.isfinite(v).all()]
    assert not nonfinite, f"non-finite tensors: {nonfinite[:5]}"
    logger.info("verify: %d keys, all match %r, dtypes=%s, all finite ✓",
                len(out), args.key_substr, {str(v.dtype) for v in out.values()})

    sample = keys[:6] + keys[-3:]
    for k in sample:
        d0 = rank_sds[0][k]
        p = d0.placements[0]
        if isinstance(p, Shard):
            recon = torch.cat([sd[k]._local_tensor for sd in rank_sds], dim=p.dim).narrow(p.dim, 0, d0.shape[p.dim])
        else:
            recon = rank_sds[0][k]._local_tensor
        assert tuple(out[k].shape) == tuple(d0.shape), f"{k}: shape {tuple(out[k].shape)} != global {tuple(d0.shape)}"
        assert torch.equal(recon, out[k]), f"{k}: reconstruction != re-concat of shards"
    logger.info("verify: reconstruction spot-check %d/%d keys match raw shards ✓", len(sample), len(sample))

    if args.key_substr == ".indexer.":
        # Reference key sets for both architectures. Only NAMES are compared here, and those do not depend
        # on the geometry, so default configs are sufficient for the DSA side's tunable dims.
        from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer
        from verl.models.transformers.msa_indexer import MSAConfig, MSAIndexer

        refs = {
            "dsa": set(LightningIndexer(DSAConfig(
                enabled=True, n_heads=args.n_heads, head_dim=args.head_dim, rope_head_dim=args.rope_head_dim,
                q_lora_rank=args.q_lora_rank, hidden_size=args.hidden_size, top_k=512)).state_dict()),
            "msa": set(MSAIndexer(MSAConfig(enabled=True)).state_dict()),
        }

        # MSA keeps layers [0, dense_prefix) dense, so layer 0 need not have an indexer. Use the LOWEST
        # layer that actually does -- assuming layer 0 is how this check silently fails on an MSA ckpt.
        layer_re = re.compile(r"model\.layers\.(\d+)\.self_attn\.indexer\.")
        by_layer = {}
        for k in out:
            m = layer_re.match(k)
            if m:
                by_layer.setdefault(int(m.group(1)), set()).add(k.split("self_attn.indexer.")[-1])
        assert by_layer, f"no keys matched 'model.layers.<i>.self_attn.indexer.' (sample: {keys[:3]})"
        first = min(by_layer)
        suffixes = by_layer[first]

        arch = args.arch
        if arch == "auto":
            hits = [a for a, ref in refs.items() if suffixes == ref]
            assert len(hits) == 1, (
                f"layer-{first} keys match no known indexer (or both). got={sorted(suffixes)}\n"
                + "\n".join(f"  vs {a}: missing={sorted(r - suffixes)} extra={sorted(suffixes - r)}"
                            for a, r in refs.items())
            )
            arch = hits[0]
        ref = refs[arch]
        assert suffixes == ref, (
            f"layer-{first} keys != {arch} indexer keys | missing={sorted(ref - suffixes)} "
            f"extra={sorted(suffixes - ref)}"
        )
        # Every sparse layer must carry the SAME key set, else the strict=False load would silently skip some.
        ragged = {i: sorted(s ^ ref) for i, s in by_layer.items() if s != ref}
        assert not ragged, f"layers with a different indexer key set than layer {first}: {ragged}"
        logger.info("verify: arch=%s -- layers %s each expose exactly the %d %s indexer keys ✓",
                    arch, f"{first}..{max(by_layer)} ({len(by_layer)} sparse)", len(ref), arch)
    logger.info("verify: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
