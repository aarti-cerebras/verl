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
"""DSA Phase-2 — LENGTH ANALYSIS of generated trajectories (see docs/dsa_phase2_plan.md, T5 / M2 probe).

Reads a trajectories JSONL (from gen_trajectories.py) and reports, per domain and overall: input/response/
total token-length distributions, DSA length-bucket yields (the ``top_k=512`` signal), truncation rate
(``finish_reason == length`` = the cap was hit), and a **recommended per-domain ``max_new_tokens``**
(measured p99 + margin). Uses the token counts recorded at generation time; falls back to re-tokenizing with
the MiniCPM3 tokenizer if a row lacks them. Writes a JSON report + logs everything (incl. the exact command).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsa_log import setup_logging  # noqa: E402

BUCKETS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
PCTLS = [10, 25, 50, 75, 90, 95, 99, 99.9]


def _extract(messages, role_is_assistant):
    parts = [m.get("content") or "" for m in messages
             if isinstance(m, dict) and ((m.get("role") == "assistant") == role_is_assistant)]
    return "\n".join(parts)


def _stats(name, arr, logger):
    a = np.asarray(arr)
    q = {f"p{p}": int(np.percentile(a, p)) for p in PCTLS}
    logger.info("  %-9s n=%d mean=%.0f max=%d  %s", name, len(a), a.mean(), a.max(),
                " ".join(f"{k}={v}" for k, v in q.items()))
    return {"n": len(a), "mean": float(a.mean()), "max": int(a.max()), **q}


def main():
    ap = argparse.ArgumentParser(description="Analyze trajectory length distributions.")
    ap.add_argument("--trajectories", required=True, help="input JSONL from gen_trajectories.py")
    ap.add_argument("--out-report", required=True, help="output JSON report path")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--tokenizer", default="openbmb/MiniCPM3-4B", help="only used if token counts are missing")
    ap.add_argument("--cap-percentile", type=float, default=99, help="percentile for recommended max_new_tokens")
    ap.add_argument("--cap-margin", type=float, default=1.15, help="multiply the percentile by this for the cap")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_report)) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(out_dir, "logs")
    logger, _ = setup_logging("analyze_lengths", log_dir)
    logger.info("config: %s", vars(args))

    rows = []
    with open(args.trajectories) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("loaded %d trajectories", len(rows))

    need_tok = any("resp_tokens" not in r for r in rows)
    tok = None
    if need_tok:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        logger.info("tokenizing (some rows missing recorded token counts)")

    def tlen(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    by_dom = {}
    for r in rows:
        if "resp_tokens" in r:
            rt, it = r["resp_tokens"], r.get("prompt_tokens", 0)
        else:
            rt = tlen(_extract(r["messages"], True))
            it = tlen(_extract(r["messages"], False))
        r["_rt"], r["_it"], r["_tt"] = rt, it, rt + it
        by_dom.setdefault(r["domain"], []).append(r)

    report = {"overall": {}, "by_domain": {}}

    def summarize(name, subset):
        rt = [r["_rt"] for r in subset]
        it = [r["_it"] for r in subset]
        tt = [r["_tt"] for r in subset]
        logger.info("[%s] n=%d", name, len(subset))
        s = {"input": _stats("INPUT", it, logger),
             "response": _stats("RESPONSE", rt, logger),
             "total": _stats("TOTAL", tt, logger)}
        tt_a = np.asarray(tt)
        s["total_bucket_frac"] = {str(b): round(float(np.mean(tt_a >= b)), 4) for b in BUCKETS}
        logger.info("  total >= : %s", " ".join(f"{b}:{100 * s['total_bucket_frac'][str(b)]:.0f}%" for b in BUCKETS))
        trunc = np.mean([r.get("finish_reason") == "length" for r in subset]) if subset else 0.0
        s["truncated_frac"] = round(float(trunc), 4)
        rec = int(np.percentile(rt, args.cap_percentile) * args.cap_margin)
        s["recommended_max_new_tokens"] = rec
        logger.info("  truncated(@cap)=%.1f%%  recommended max_new_tokens(p%.0f*%.2f)=%d",
                    100 * trunc, args.cap_percentile, args.cap_margin, rec)
        return s

    report["overall"] = summarize("ALL", rows)
    for dom in sorted(by_dom):
        report["by_domain"][dom] = summarize(dom, by_dom[dom])

    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("wrote report -> %s", args.out_report)
    logger.info("DONE")


if __name__ == "__main__":
    main()
