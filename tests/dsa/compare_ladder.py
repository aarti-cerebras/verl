#!/usr/bin/env python3
"""Token-exact comparison of the ladder rows written by run_qwen3_dsa_ladder.sh.

Prints, per row, agreement against its dense reference. The gates (serving_eval_plan.md §5):
``B2``/``F2`` must be EXACT against ``A2_dense``; ``D``/``E`` must diverge; ``E`` must collapse.
"""

import glob
import json
import sys

PAIRS = [
    ("A2_dense", "B2_topk_ge_T", "EXACT"),
    ("A2_dense", "F2_rand_ge_T", "EXACT"),
    ("A_dense", "C_topk2048", "close"),
    ("A_dense", "D_topk256", "differs"),
    ("A_dense", "E_rand256", "collapses"),
]


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dsa_ladder"
    rows = {}
    for f in glob.glob(f"{root}/*.json"):
        d = json.load(open(f))
        rows[d["label"]] = d
    ok = True
    print(f"{'comparison':<32} {'prompt0':>16} {'prompt1':>16}  expectation")
    for a, b, expect in PAIRS:
        if a not in rows or b not in rows:
            print(f"{a} vs {b}: MISSING")
            continue
        cells = []
        for i in ("0", "1"):
            x = rows[a]["outputs"][i]["token_ids"]
            y = rows[b]["outputs"][i]["token_ids"]
            n = min(len(x), len(y))
            first = next((j for j in range(n) if x[j] != y[j]), None)
            same = sum(1 for j in range(n) if x[j] == y[j])
            cells.append("EXACT" if first is None and len(x) == len(y) else f"{same}/{n} div@{first}")
        print(f"{a + ' vs ' + b:<32} {cells[0]:>16} {cells[1]:>16}  {expect}")
        if expect == "EXACT" and not all(c == "EXACT" for c in cells):
            ok = False
            print(f"   ^^ GATE FAILED: {b} must match {a} token-exactly")
    print(f"\nladder gates: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
