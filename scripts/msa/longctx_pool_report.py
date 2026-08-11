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
"""Pool report + survey regression for the long-context prompt banks.

Runbook: ``docs/qwen3_4b_msa/phase2_long_context_gen.md`` §9 test 2. Reads a ``prompts.jsonl`` produced
by ``select_prompts.py --source longctx`` and reports, per source:

* prefill-token percentiles, band histogram, and total prefill tokens (the generation-cost driver),
* language split and how it was decided (``lang_basis``),
* generation budget under the recorded window, and the ``skipped_nofit`` tail,
* **a comparison against the measured survey table** in ``phase2_long_context_data.md`` §1/§2b.

The survey comparison is the point: a preprocessing bug that silently changes what we are buying shows
up here as a row count or a max that has moved, and nowhere else. It is reported as WARN, never as a
hard failure, because three known effects legitimately move the numbers and their directions are
predictable (see ``EXPECTED_DRIFT``).

    python3 scripts/msa/longctx_pool_report.py --prompts $RUN/prompts.jsonl --out-report $RUN/pool.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsa"))
from _dsa_log import setup_logging  # noqa: E402

# Measured survey reference: rows with prefill >=16K, and max prefill, over the WHOLE dataset
# (all languages). From phase2_long_context_data.md §1 and §2b. `rows_ge16k` for LoongRL is the sum of
# its three _distractor_ configs.
SURVEY = {
    "longcite":   {"rows_ge16k": 23321, "max": 138127, "frac_ge16k": 0.588},
    "loongrl":    {"rows_ge16k": 7458,  "max": 20668,  "frac_ge16k": 0.994},
    "longreward": {"rows_ge16k": 4958,  "max": 65029,  "frac_ge16k": 0.496},
    "longalpaca": {"rows_ge16k": 3000,  "max": 27403,  "frac_ge16k": 0.250},
    "longalign":  {"rows_ge16k": 2925,  "max": 53176,  "frac_ge16k": 0.296},
    "docqarl":    {"rows_ge16k": 338,   "max": 61992,  "frac_ge16k": 0.212},
    "chatqa2":    {"rows_ge16k": 56447, "max": 137401, "frac_ge16k": None},
}

# Known effects that legitimately move our numbers away from the survey. Printed next to any WARN so a
# real regression is not confused with an expected shift.
EXPECTED_DRIFT = {
    "longcite": "stripping removes ~9.7% of prompt tokens, so max and percentiles should come in LOWER; "
                "row count should be close",
    "chatqa2":  "window/floor rejects the pile at NVIDIA's 131,072-Llama-3-token truncation wall, and "
                "--max-per-document caps repeats per book — both cut rows legitimately",
    "loongrl":  "exact-prompt dedup across three configs sharing a Wikipedia base may cut a few rows",
    "longalpaca": "the survey's 25%>=16K came from a 240-row STRIDED sample of an ORDERED dataset "
                  "(long-QA rows first, short Alpaca rows after); the full census measures 16.3%",
    "longalign": "survey figures are a 240-row strided sample, not a census",
}

BANDS = [(0, 16384), (16384, 32768), (32768, 65536), (65536, 131072), (131072, 1 << 30)]


def _pctl(xs, p):
    if not xs:
        return 0
    return xs[min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--skipped", default=None, help="default: skipped_nofit.jsonl beside --prompts")
    ap.add_argument("--out-report", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--warn-frac", type=float, default=0.25,
                    help="relative deviation from the survey that triggers a WARN")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.prompts)) or "."
    logger, _ = setup_logging("longctx_pool_report", args.log_dir or os.path.join(out_dir, "logs"))

    by = {}
    for line in open(args.prompts):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by.setdefault(r["domain"], []).append(r)

    skipped_path = args.skipped or os.path.join(out_dir, "skipped_nofit.jsonl")
    skipped = {}
    if os.path.exists(skipped_path):
        for line in open(skipped_path):
            line = line.strip()
            if line:
                s = json.loads(line)
                skipped.setdefault(s["source"], []).append(s)

    report = {"prompts_file": os.path.abspath(args.prompts), "sources": {}, "warnings": []}
    grand_rows = grand_tok = 0

    logger.info("%-12s %7s %9s %9s %9s %9s %13s  %s", "source", "rows", "p50", "p90", "p99", "max",
                "prefill_tok", "lang")
    for name in sorted(by):
        rows = by[name]
        toks = sorted(r["prompt_tokens"] for r in rows)
        total = sum(toks)
        langs, bases, tiers, windows = {}, {}, {}, set()
        for r in rows:
            langs[r["lang"]] = langs.get(r["lang"], 0) + 1
            bases[r.get("lang_basis", "?")] = bases.get(r.get("lang_basis", "?"), 0) + 1
            tiers[r.get("licence_tier", "?")] = tiers.get(r.get("licence_tier", "?"), 0) + 1
            windows.add(r.get("window"))
        window = next(iter(windows)) if len(windows) == 1 else sorted(windows)
        budgets = [(window if isinstance(window, int) else 131072) - (t + 10) - 1 for t in toks]

        entry = {
            "rows": len(rows), "prefill_tokens_total": total, "window": window,
            "p50": _pctl(toks, 50), "p90": _pctl(toks, 90), "p99": _pctl(toks, 99),
            "min": toks[0], "max": toks[-1], "mean": total // max(1, len(toks)),
            "lang": langs, "lang_basis": bases, "licence_tier": tiers,
            "gen_budget_min": min(budgets), "gen_budget_p50": _pctl(sorted(budgets), 50),
            "skipped_nofit": len(skipped.get(name, [])),
            "bands": {f"{lo}-{hi}": sum(1 for t in toks if lo <= t < hi) for lo, hi in BANDS},
        }
        report["sources"][name] = entry
        grand_rows += len(rows)
        grand_tok += total
        logger.info("%-12s %7d %9d %9d %9d %9d %13s  %s", name, len(rows), entry["p50"], entry["p90"],
                    entry["p99"], entry["max"], f"{total/1e6:.0f}M", langs)

        ref = SURVEY.get(name)
        if ref:
            n_seen = len(rows) + entry["skipped_nofit"]
            dev = abs(n_seen - ref["rows_ge16k"]) / max(1, ref["rows_ge16k"])
            if dev > args.warn_frac:
                w = (f"{name}: rows>=16K measured {n_seen} vs survey {ref['rows_ge16k']} "
                     f"({dev*100:.0f}% off). {EXPECTED_DRIFT.get(name, 'no expected drift on record')}")
                report["warnings"].append(w)
            # The survey's max is the max of a 240-row STRIDED SAMPLE, not a census, so a full
            # extraction legitimately finds a longer tail -- verified on LongAlpaca (78,696 vs 27,403:
            # a real 140K-char paper with the source's normal framing) and LongAlign (73,114 vs
            # 53,176). Only a wild overshoot suggests the extractor is concatenating the wrong fields.
            if entry["max"] > ref["max"] * 5:
                report["warnings"].append(
                    f"{name}: max prefill {entry['max']:,} is >5x the survey's sampled max "
                    f"{ref['max']:,} — check the extractor is not concatenating extra fields")
            elif entry["max"] > ref["max"] * 1.02:
                report.setdefault("notes", []).append(
                    f"{name}: max prefill {entry['max']:,} > survey's SAMPLED max {ref['max']:,} "
                    "(expected: the survey measured n=240 strided, this is a full census)")

    report["total_rows"] = grand_rows
    report["total_prefill_tokens"] = grand_tok
    report["total_skipped_nofit"] = sum(len(v) for v in skipped.values())
    logger.info("%-12s %7d %9s %9s %9s %9s %13s", "TOTAL", grand_rows, "", "", "", "",
                f"{grand_tok/1e9:.2f}B")

    logger.info("")
    logger.info("band histogram (prefill tokens):")
    hdr = "  %-12s" % "source" + "".join(f"{lo//1024}-{hi//1024 if hi < (1<<30) else '+'}K".rjust(12)
                                         for lo, hi in BANDS)
    logger.info(hdr)
    for name in sorted(by):
        b = report["sources"][name]["bands"]
        logger.info("  %-12s" % name + "".join(str(b[f"{lo}-{hi}"]).rjust(12) for lo, hi in BANDS))

    for n_ in report.get("notes", []):
        logger.info("note: %s", n_)
    if report["warnings"]:
        logger.warning("")
        for w in report["warnings"]:
            logger.warning("WARN %s", w)
    else:
        logger.info("")
        logger.info("no survey deviations beyond %.0f%%", args.warn_frac * 100)

    if args.out_report:
        with open(args.out_report, "w") as fh:
            json.dump(report, fh, indent=1)
        logger.info("report -> %s", args.out_report)


if __name__ == "__main__":
    main()
