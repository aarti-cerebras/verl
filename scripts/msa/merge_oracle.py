#!/usr/bin/env python3
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
"""Merge per-GPU shard outputs of ``probe_block_oracle.py`` into one scorecard.

Merges the ``raw`` sums/counts, so the result is **exactly** what a single-process run over all
documents would have produced — averaging the shards' means would be wrong whenever shards see
different numbers of valid (position, group) samples.

Usage:
    python3 scripts/msa/merge_oracle.py --out merged.json shard_*.json
"""

import argparse
import glob
import json
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="shard JSON paths (globs allowed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = sorted({p for pat in a.shards for p in (glob.glob(pat) or [pat])})
    if not paths:
        sys.exit("no shard files matched")

    combined, meta, total_docs, seen = {}, None, 0, []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if "raw" not in d:
            sys.exit(f"{p}: no 'raw' section — regenerate with the current probe_block_oracle.py")
        if meta is None:
            meta = {k: d.get(k) for k in ("args", "model_config", "query_positions")}
        else:
            for k in ("seq_len", "ks", "block_sizes", "num_queries", "min_pos", "model"):
                if d["args"].get(k) != meta["args"].get(k):
                    sys.exit(f"{p}: arg '{k}' differs from the first shard — refusing to merge")
        total_docs += d.get("num_docs", 0)
        seen.append({"path": p, "shard": d.get("shard"), "num_docs": d.get("num_docs"),
                     "elapsed_s": d.get("elapsed_s")})
        for layer, cells in d["raw"].items():
            tgt = combined.setdefault(layer, {})
            for cfg, c in cells.items():
                t = tgt.setdefault(cfg, {"blk": 0.0, "tok": 0.0, "n": 0})
                t["blk"] += c["blk"]
                t["tok"] += c["tok"]
                t["n"] += c["n"]

    per_layer, agg = {}, {}
    for layer, cells in sorted(combined.items(), key=lambda kv: int(kv[0])):
        per_layer[layer] = {}
        for cfg, c in sorted(cells.items()):
            if c["n"] == 0:
                continue
            blk, tokm = c["blk"] / c["n"], c["tok"] / c["n"]
            per_layer[layer][cfg] = {
                "oracle_block": blk, "oracle_token": tokm,
                "granularity_cost": tokm - blk, "unreachable": 1.0 - tokm, "n": c["n"],
            }
            s = agg.setdefault(cfg, {"blk": [], "tok": []})
            s["blk"].append(blk)
            s["tok"].append(tokm)

    summary = {}
    for cfg, s in sorted(agg.items()):
        bm, tm = sum(s["blk"]) / len(s["blk"]), sum(s["tok"]) / len(s["tok"])
        summary[cfg] = {
            "oracle_block_mean": bm,
            "oracle_block_min_layer": min(s["blk"]),
            "oracle_token_mean": tm,
            "granularity_cost_mean": tm - bm,
            "layers": len(s["blk"]),
        }

    print("\n" + "=" * 96)
    print(f"MSA BLOCK-ORACLE PROBE (merged {len(paths)} shards, {total_docs} docs)")
    print(f"model: {meta['args'].get('model')}   seq_len={meta['args'].get('seq_len')}")
    print("=" * 96)
    print(f"{'config':>12} | {'oracle_block':>13} | {'min layer':>10} | {'oracle_token':>13} "
          f"| {'granularity':>11} | {'unreachable':>11}")
    print("-" * 96)
    for cfg, v in summary.items():
        print(f"{cfg:>12} | {v['oracle_block_mean']:13.4f} | {v['oracle_block_min_layer']:10.4f} "
              f"| {v['oracle_token_mean']:13.4f} | {v['granularity_cost_mean']:11.4f} "
              f"| {1.0 - v['oracle_token_mean']:11.4f}")
    print("-" * 96)
    print("Decision rule (docs/qwen3_4b_msa/plan.md §10.2):  >=0.95 proceed | 0.85-0.95 workable "
          "| <=~0.70 too coarse\nGate on oracle_block AND its per-layer min -- one bad layer matters.")
    print("=" * 96 + "\n")

    with open(a.out, "w") as f:
        json.dump({"merged_from": seen, "num_docs": total_docs, "args": meta["args"],
                   "model_config": meta["model_config"], "summary": summary,
                   "per_layer": per_layer, "raw": combined}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
