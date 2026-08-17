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
"""Aggregate `probe_needle_rank.py` shards into the near-miss vs blind verdict.

The single number that decides the remedy is the rank of the needle block in layers that FAIL to
select it. Ranks just past `k` mean the indexer nearly had it (more Phase-1 / `full_support_kl_prob`
should recover it); ranks in the hundreds mean it is not in contention at all, and no amount of
restricted-support Phase-2 KL can move it -- that loss gives zero gradient to unselected blocks.

Also splits correct vs incorrect items, which is the causal check: if selection were irrelevant to
the answer, the two groups would look the same.

  python3 scripts/msa/analyze_needle_rank.py <dir-with-shard-jsons>
"""

import glob
import json
import os
import statistics as st
import sys


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> int:
    root = sys.argv[1]
    by_task: dict[str, list] = {}
    top_k = block = None
    for f in sorted(glob.glob(os.path.join(root, "*.json"))):
        d = json.load(open(f))
        top_k, block = d["top_k"], d["block_size"]
        task = os.path.basename(f).rsplit("_", 2)[0]
        by_task.setdefault(task, []).extend(d["results"])

    print(f"top_k={top_k} block_size={block}  (selected tokens = {top_k * block})\n")
    for task in sorted(by_task):
        rs = by_task[task]
        if not rs:
            continue
        n_layers = len(rs[0]["layers"])
        acc = sum(r["correct"] for r in rs) / len(rs)
        print(f"=== {task}   n={len(rs)}  probe-accuracy={acc:.0%} "
              f"({sum(r['correct'] for r in rs)}/{len(rs)}) ===")

        # Per-layer selection rate and rank, pooled over (item, layer).
        allr = [p["rank_min"] for r in rs for p in r["layers"]]
        miss = [x for x in allr if x >= top_k]
        sel_frac = [r["layers_selected"] / n_layers for r in rs]
        print(f"  layers selecting the needle : mean {st.mean(sel_frac):.0%} of {n_layers} "
              f"(min {min(sel_frac):.0%}, max {max(sel_frac):.0%})")
        print(f"  item-level recall (>=1 layer): "
              f"{sum(r['layers_selected'] > 0 for r in rs) / len(rs):.0%}")
        print(f"  rank of needle block, pooled over item x layer  "
              f"p10={pct(allr, 10)} p50={pct(allr, 50)} p90={pct(allr, 90)} max={max(allr)}")
        if miss:
            print(f"  ... restricted to the {len(miss)}/{len(allr)} ({len(miss) / len(allr):.0%}) "
                  f"layer-misses: p50={pct(miss, 50)} p90={pct(miss, 90)} max={max(miss)}")

        # THE VERDICT BUCKETS. `k`..4k is "near miss"; beyond that the block is not in contention.
        buckets = [(0, top_k, "selected"), (top_k, 2 * top_k, f"near-miss {top_k}-{2 * top_k}"),
                   (2 * top_k, 4 * top_k, f"{2 * top_k}-{4 * top_k}"),
                   (4 * top_k, 10**9, f">={4 * top_k} (blind)")]
        print("  rank buckets:", "  ".join(
            f"{lab} {sum(lo <= x < hi for x in allr) / len(allr):.0%}" for lo, hi, lab in buckets))

        for label, sub in (("correct", [r for r in rs if r["correct"]]),
                           ("WRONG", [r for r in rs if not r["correct"]])):
            if not sub:
                continue
            sr = [r["layers_selected"] / n_layers for r in sub]
            pr = [p["rank_min"] for r in sub for p in r["layers"]]
            print(f"    {label:<8} n={len(sub):3d}  layers-selecting {st.mean(sr):.0%}  "
                  f"rank p50={pct(pr, 50)} p90={pct(pr, 90)}  "
                  f"blind(>={4 * top_k}) {sum(x >= 4 * top_k for x in pr) / len(pr):.0%}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
