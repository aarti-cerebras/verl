#!/usr/bin/env python3
"""Derive an isolated modulo-bucket serving directory from an exact Qwen3 DSA directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

EXACT_ARCH = "Qwen3DSAForCausalLM"
BUCKETED_ARCH = "Qwen3DSABucketedForCausalLM"
BACKENDS = ("torch_reference", "vllm_stock_per_bucket")
DECODE_GRAPH_VALIDATED_GEOMETRY = {
    "backend": "vllm_stock_per_bucket",
    "bucket_count": 8,
    "bucket_top_k": 256,
    "total_k": 2048,
}
DECODE_GRAPH_EVIDENCE = ".agents/gpu_jobs/20260828T210745Z-qwen3-dsa-bucketed-decode-graph/result.md"


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_source(source: Path) -> tuple[dict[str, Any], bytes]:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"source exact serving dir has no config.json: {config_path}")
    raw = config_path.read_bytes()
    config = json.loads(raw)
    architectures = config.get("architectures") or []
    if EXACT_ARCH not in architectures:
        raise SystemExit(f"source architectures={architectures!r}; expected exact architecture {EXACT_ARCH!r}")
    if not config.get("dsa_enabled") or config.get("dsa_mode") != "sparse":
        raise SystemExit("source config is not a servable sparse Qwen3 DSA checkpoint")
    top_k = int(config.get("dsa_top_k", 0))
    if top_k <= 0 or int(config.get("index_topk", -1)) != top_k:
        raise SystemExit("source exact config must have positive dsa_top_k == index_topk before derivation")
    return config, raw


def _link_assets(source: Path, output: Path) -> list[str]:
    linked: list[str] = []
    excluded = {"config.json", "BUILD_MANIFEST.json", "BUCKETED_BUILD_MANIFEST.json"}
    for asset in sorted(source.iterdir(), key=lambda path: path.name):
        if asset.name in excluded:
            continue
        destination = output / asset.name
        destination.symlink_to(os.path.relpath(asset, start=output), target_is_directory=asset.is_dir())
        linked.append(asset.name)
    return linked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bucket-count", type=int, required=True)
    parser.add_argument(
        "--bucket-top-k",
        type=int,
        default=None,
        help="local k; omitted means dsa_top_k // bucket_count",
    )
    parser.add_argument("--backend", choices=BACKENDS, default="vllm_stock_per_bucket")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.out.resolve()
    if source == output:
        raise SystemExit("--out must differ from --source")
    if output.is_relative_to(source):
        raise SystemExit(f"--out must not be inside --source: {output} is under {source}")
    if source.is_relative_to(output):
        raise SystemExit(f"--source must not be inside --out: {source} is under {output}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    if args.bucket_count <= 0:
        raise SystemExit("--bucket-count must be positive")

    config, raw_config = _load_source(source)
    total_k = int(config["dsa_top_k"])
    if args.bucket_top_k is None:
        if total_k % args.bucket_count:
            raise SystemExit(
                f"dsa_top_k={total_k} is not divisible by bucket_count={args.bucket_count}; "
                "pass an explicit geometry whose product preserves the total budget"
            )
        bucket_top_k = total_k // args.bucket_count
    else:
        bucket_top_k = args.bucket_top_k
    if bucket_top_k <= 0 or args.bucket_count * bucket_top_k != total_k:
        raise SystemExit(
            f"fixed budget requires bucket_count * bucket_top_k == dsa_top_k: "
            f"{args.bucket_count} * {bucket_top_k} != {total_k}"
        )

    derived = dict(config)
    derived.update(
        {
            "architectures": [BUCKETED_ARCH],
            "dsa_selector": "modulo_bucket_topk",
            "dsa_selector_backend": args.backend,
            "dsa_bucket_count": args.bucket_count,
            "dsa_bucket_top_k": bucket_top_k,
            "dsa_bucket_telemetry": "off",
            "index_topk": total_k,
        }
    )

    output.mkdir(parents=True, exist_ok=True)
    linked = _link_assets(source, output)
    (output / "config.json").write_text(json.dumps(derived, indent=2) + "\n")

    repo = Path(__file__).resolve().parents[2]
    decode_graph_validated = {
        "backend": args.backend,
        "bucket_count": args.bucket_count,
        "bucket_top_k": bucket_top_k,
        "total_k": total_k,
    } == DECODE_GRAPH_VALIDATED_GEOMETRY
    manifest = {
        "artifact_kind": "qwen3_dsa_bucketed_serving_dir",
        "source_exact_serving_dir": str(source),
        "source_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "source_architecture": EXACT_ARCH,
        "architecture": BUCKETED_ARCH,
        "selector": "modulo_bucket_topk",
        "selector_backend": args.backend,
        "selector_speed_claim_valid": False,
        "dsa_top_k": total_k,
        "index_topk": total_k,
        "bucket_count": args.bucket_count,
        "bucket_top_k": bucket_top_k,
        "stock_topk_launches_per_selection": (args.bucket_count if args.backend == "vllm_stock_per_bucket" else 0),
        "bucket_score_materialization": (
            "none_strided_per_bucket" if args.backend == "vllm_stock_per_bucket" else "reference_tensor_view"
        ),
        "linked_assets": linked,
        "verl_revision": _git_revision(repo),
        "exact_source_modified": False,
        "cuda_graph_modes_supported": (
            ["NONE", "FULL_DECODE_ONLY"] if args.backend == "vllm_stock_per_bucket" else ["NONE"]
        ),
        "decode_cuda_graph_validated": decode_graph_validated,
        "prefill_cuda_graph_validated": False,
        "cuda_graph_validation_evidence": DECODE_GRAPH_EVIDENCE if decode_graph_validated else None,
    }
    (output / "BUCKETED_BUILD_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
