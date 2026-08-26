#!/usr/bin/env python3
"""Derive an isolated approximate-selector serving dir from an exact one.

The source directory is read-only. ``config.json`` and a build manifest are
new files; unchanged weights, tokenizer data, and auxiliary assets are relative
symlinks back to the exact serving directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


EXACT_ARCH = "Qwen3DSAForCausalLM"
APPROX_ARCH = "Qwen3DSAApproxForCausalLM"
SELECTORS = ("topk", "exact_ge", "radix_floor", "radix_midpoint", "radix_ceil")
TELEMETRY = ("off", "summary", "verify_exact", "graph_safety")


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _git_revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
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
        raise SystemExit(
            f"source architectures={architectures!r}; expected the exact {EXACT_ARCH!r}"
        )
    if not config.get("dsa_enabled") or config.get("dsa_mode") != "sparse":
        raise SystemExit("source config is not a servable sparse Qwen3 DSA checkpoint")
    top_k = int(config.get("dsa_top_k", 0))
    if top_k <= 0 or int(config.get("index_topk", -1)) != top_k:
        raise SystemExit(
            "source exact config must have positive dsa_top_k == index_topk before derivation"
        )
    return config, raw


def _capacity(selector: str, k: int, margin: int, alignment: int) -> int:
    if margin < 0:
        raise SystemExit("--selector-margin must be non-negative")
    if selector in ("topk", "radix_ceil"):
        if margin:
            raise SystemExit(f"{selector} has fixed capacity k; --selector-margin must be zero")
        return k
    return _round_up(k + margin, alignment)


def _link_assets(source: Path, output: Path) -> list[str]:
    linked: list[str] = []
    excluded = {"config.json", "BUILD_MANIFEST.json", "APPROX_BUILD_MANIFEST.json"}
    for asset in sorted(source.iterdir(), key=lambda path: path.name):
        if asset.name in excluded:
            continue
        destination = output / asset.name
        target = os.path.relpath(asset, start=output)
        destination.symlink_to(target, target_is_directory=asset.is_dir())
        linked.append(asset.name)
    return linked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="working exact serving directory")
    parser.add_argument("--out", type=Path, required=True, help="new approximate serving directory")
    parser.add_argument("--selector", choices=SELECTORS, default="topk")
    parser.add_argument("--selector-margin", type=int, default=0)
    parser.add_argument("--capacity-alignment", type=int, default=128)
    parser.add_argument("--radix-omit-bits", type=int, default=4)
    parser.add_argument("--telemetry", choices=TELEMETRY, default="off")
    parser.add_argument("--telemetry-artifact", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.out.resolve()
    if source == output:
        raise SystemExit("--out must differ from --source")
    # Containment, not just equality: `output.mkdir` runs before `_link_assets` iterates
    # `source.iterdir()`, so an --out nested under --source is itself an entry by the time the walk
    # starts and gets symlinked into itself (--source exact --out exact/approx yields
    # exact/approx/approx -> ..). The result is a serving directory with a recursive asset.
    if output.is_relative_to(source):
        raise SystemExit(
            f"--out must not be inside --source: {output} is under {source}. The output directory "
            f"would be linked into itself as one of the source assets."
        )
    if source.is_relative_to(output):
        raise SystemExit(f"--source must not be inside --out: {source} is under {output}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    if args.capacity_alignment <= 0:
        raise SystemExit("--capacity-alignment must be positive")
    if args.radix_omit_bits != 4:
        raise SystemExit("the reused dsa-csx reference selector currently requires 4 omitted bits")

    config, raw_config = _load_source(source)
    k = int(config["dsa_top_k"])
    capacity = _capacity(args.selector, k, args.selector_margin, args.capacity_alignment)
    if capacity > 4096:
        raise SystemExit(
            f"derived index_topk={capacity} exceeds the validated vLLM 0.26 selector limit 4096"
        )
    backend = "vllm_stock" if args.selector == "topk" else "dsa_csx_reference"
    derived = dict(config)
    derived.update(
        {
            "architectures": [APPROX_ARCH],
            "dsa_selector": args.selector,
            "dsa_selector_backend": backend,
            "dsa_radix_omit_bits": args.radix_omit_bits,
            "dsa_selector_margin": args.selector_margin,
            "dsa_telemetry": args.telemetry,
            "index_topk": capacity,
        }
    )
    if args.telemetry_artifact:
        derived["dsa_telemetry_artifact"] = args.telemetry_artifact
    else:
        derived.pop("dsa_telemetry_artifact", None)

    output.mkdir(parents=True, exist_ok=True)
    linked = _link_assets(source, output)
    (output / "config.json").write_text(json.dumps(derived, indent=2) + "\n")

    repo = Path(__file__).resolve().parents[2]
    manifest = {
        "artifact_kind": "qwen3_dsa_approx_serving_dir",
        "source_exact_serving_dir": str(source),
        "source_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "source_architecture": EXACT_ARCH,
        "architecture": APPROX_ARCH,
        "selector": args.selector,
        "selector_backend": backend,
        "selector_speed_claim_valid": False,
        "dsa_top_k": k,
        "index_topk": capacity,
        "selector_margin": args.selector_margin,
        "radix_omit_bits": args.radix_omit_bits,
        "telemetry": args.telemetry,
        "linked_assets": linked,
        "verl_revision": _git_revision(repo),
        "exact_source_modified": False,
    }
    (output / "APPROX_BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
