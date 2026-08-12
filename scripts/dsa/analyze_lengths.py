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
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsa_log import setup_logging  # noqa: E402

# Cumulative ">= N" thresholds. The ladder is EXTENDED PAST 32K and clipped to the run's --window:
# these constants were written for the 32,768 Dolci run, and at a 131,072 window every sequence above
# 32K collapsed into one number (0.45), erasing all resolution in exactly the band sparse attention is
# trained for. 1.5x midpoints are included above 32K because power-of-two steps alone are too coarse
# there. docs/qwen3_4b_msa/phase2_long_context_gen.md §6.1.
_BUCKETS_BASE = [512, 1024, 2048, 4096, 8192, 16384, 32768,
                 49152, 65536, 98304, 131072, 163840, 196608, 262144]


def buckets_for(window):
    b = [x for x in _BUCKETS_BASE if x <= window]
    return b or [_BUCKETS_BASE[0]]


BUCKETS = buckets_for(32768)  # back-compat default; main() recomputes from --window
PCTLS = [10, 25, 50, 75, 90, 95, 99, 99.9]
# histogram bin edges (open-ended top bin appended at runtime)
_HIST_PROMPT_BASE = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                     49152, 65536, 98304, 131072, 163840, 262144]
_HIST_RESP_BASE = [0, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768,
                   49152, 65536, 98304, 131072, 163840, 262144]


def hist_edges_for(window):
    """Histogram edges clipped to the window, always keeping the window itself as the last edge so the
    top bin is closed rather than an open-ended catch-all."""
    p = [x for x in _HIST_PROMPT_BASE if x < window] + [window]
    r = [x for x in _HIST_RESP_BASE if x < window] + [window]
    return p, r


HIST_EDGES_PROMPT, HIST_EDGES_RESP = hist_edges_for(32768)
# Fixed chat wrapper around the prompt, in tokens. Qwen3-Thinking: <|im_start|>user \n … <|im_end|> \n
# <|im_start|> assistant \n <think> \n = 10 (docs/qwen3_4b_msa/phase2_data_gen.md §5.1). gpt-oss/harmony
# prepends a whole system message instead and is **67** (docs/gpt_oss_20b_msa/phase2_data_gen.md §3), so
# this is a default, not a constant -- override with --chat-wrapper-tokens.
CHAT_WRAPPER_TOKENS = 10


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


def _hist(name, arr, edges, logger, width=44):
    """Text histogram over `edges` (+ an open-ended top bin). Logged and returned for the JSON report."""
    a = np.asarray(arr)
    counts, _ = np.histogram(a, bins=edges + [np.inf])
    peak = max(int(counts.max()), 1)
    n = max(len(a), 1)
    logger.info("  %s histogram (n=%d):", name, len(a))
    out = {}
    for i, c in enumerate(counts):
        lo = edges[i]
        label = f">={lo}" if i == len(counts) - 1 else f"{lo}-{edges[i + 1] - 1}"
        pct = 100.0 * int(c) / n
        logger.info("    %-13s %7d (%5.1f%%) %s", label, int(c), pct, "#" * int(width * c / peak))
        out[label] = {"n": int(c), "frac": round(pct / 100.0, 5)}
    return out


def main():
    ap = argparse.ArgumentParser(description="Analyze trajectory length distributions.")
    ap.add_argument("--trajectories", default=None, help="input JSONL from gen_trajectories.py")
    ap.add_argument("--prompts", default=None,
                    help="input JSONL from select_prompts.py: PROMPT-ONLY mode (no responses yet). Reports "
                         "the prompt histogram per subset plus the remaining generation budget in --window")
    ap.add_argument("--out-report", required=True, help="output JSON report path")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--tokenizer", default="openbmb/MiniCPM3-4B", help="only used if token counts are missing")
    ap.add_argument("--group-by", default="domain",
                    help="row key to group by (e.g. domain, source_config, original_dataset)")
    ap.add_argument("--window", type=int, default=32768,
                    help="training/generation window; the per-row generation budget is "
                         "window - prompt_tokens - wrapper (see phase2_data_gen.md §5.1)")
    ap.add_argument("--chat-wrapper-tokens", type=int, default=CHAT_WRAPPER_TOKENS,
                    help="fixed chat-template overhead around the prompt. 10 for Qwen3-Thinking (default), "
                         "**67 for gpt-oss/harmony** — pass it or the generation-budget report is off by 57")
    ap.add_argument("--cap-percentile", type=float, default=99, help="percentile for recommended max_new_tokens")
    ap.add_argument("--cap-margin", type=float, default=1.15, help="multiply the percentile by this for the cap")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out_report)) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(out_dir, "logs")
    logger, _ = setup_logging("analyze_lengths", log_dir)
    logger.info("config: %s", vars(args))

    assert bool(args.trajectories) ^ bool(args.prompts), "pass exactly one of --trajectories / --prompts"
    prompts_only = bool(args.prompts)
    src = args.prompts or args.trajectories
    # Accept a glob so the un-merged '<out>.part*' files can be read directly (gen_trajectories --no-merge).
    files = sorted(glob.glob(src)) or ([src] if os.path.exists(src) else [])
    assert files, f"no input files matched {src}"
    rows, n_bad = [], 0
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    n_bad += 1  # a kill mid-write can truncate the last line of a part file
    logger.info("loaded %d rows from %d file(s) matching %s (%s mode)%s", len(rows), len(files), src,
                "prompts-only" if prompts_only else "trajectories",
                f" — skipped {n_bad} unparseable lines" if n_bad else "")

    count_key = "prompt_tokens" if prompts_only else "resp_tokens"
    need_tok = any(count_key not in r for r in rows)
    tok = None
    if need_tok:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        logger.info("tokenizing (some rows missing recorded token counts)")

    def tlen(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    by_dom = {}
    for r in rows:
        if prompts_only:
            it = r.get("prompt_tokens")
            if it is None:
                it = tlen(_extract(r["messages"], False))
            rt = None
        elif "resp_tokens" in r:
            rt, it = r["resp_tokens"], r.get("prompt_tokens", 0)
        else:
            rt = tlen(_extract(r["messages"], True))
            it = tlen(_extract(r["messages"], False))
        r["_it"] = it
        r["_rt"] = rt
        # The trained sequence is prefix + response + 1 (closing <|im_end|>). gen_trajectories.py records
        # `prefix_tokens` = the served prefix (wrapper INCLUDED); older rows only have the bare prompt count,
        # so add the wrapper ourselves in that case.
        if rt is None:
            r["_tt"] = None
        elif r.get("prefix_tokens") is not None:
            r["_tt"] = int(r["prefix_tokens"]) + rt + 1
        else:
            r["_tt"] = args.chat_wrapper_tokens + it + rt + 1
        by_dom.setdefault(r.get(args.group_by, "?"), []).append(r)

    # Rebind the ladders to THIS run's window. Without this the module-level defaults (built for the
    # 32,768 Dolci run) apply, and every sequence above 32K lands in one open-ended top bin.
    global BUCKETS, HIST_EDGES_PROMPT, HIST_EDGES_RESP
    BUCKETS = buckets_for(args.window)
    HIST_EDGES_PROMPT, HIST_EDGES_RESP = hist_edges_for(args.window)
    logger.info("length ladders for window=%d: buckets=%s", args.window, BUCKETS)

    report = {"window": args.window, "group_by": args.group_by, "prompts_only": prompts_only,
              "overall": {}, "by_domain": {}}

    def summarize(name, subset):
        it = [r["_it"] for r in subset]
        logger.info("[%s] n=%d", name, len(subset))
        s = {"input": _stats("INPUT", it, logger)}
        s["input_hist"] = _hist("PROMPT tokens", it, HIST_EDGES_PROMPT, logger)

        # generation budget left inside the window for each prompt (§5.1)
        budget = np.asarray([args.window - args.chat_wrapper_tokens - 1 - x for x in it])
        s["gen_budget"] = _stats("GEN_BUDGET", budget, logger)
        s["gen_budget_lt_8192_frac"] = round(float(np.mean(budget < 8192)), 5)
        s["prompt_exceeds_window_frac"] = round(float(np.mean(budget <= 0)), 5)
        logger.info("  budget<8192 (cannot reach the decode-long bucket): %.3f%%   prompt>=window: %.3f%%",
                    100 * s["gen_budget_lt_8192_frac"], 100 * s["prompt_exceeds_window_frac"])

        if prompts_only:
            return s

        rt = [r["_rt"] for r in subset]
        tt = [r["_tt"] for r in subset]
        s["response"] = _stats("RESPONSE", rt, logger)
        s["response_hist"] = _hist("RESPONSE tokens", rt, HIST_EDGES_RESP, logger)
        s["total"] = _stats("TOTAL", tt, logger)
        s["total_hist"] = _hist("TOTAL tokens", tt, HIST_EDGES_RESP, logger)
        tt_a = np.asarray(tt)
        s["total_bucket_frac"] = {str(b): round(float(np.mean(tt_a >= b)), 4) for b in BUCKETS}
        logger.info("  total >= : %s", " ".join(f"{b}:{100 * s['total_bucket_frac'][str(b)]:.0f}%" for b in BUCKETS))
        trunc = np.mean([r.get("finish_reason") == "length" for r in subset]) if subset else 0.0
        s["truncated_frac"] = round(float(trunc), 4)
        rec = int(np.percentile(rt, args.cap_percentile) * args.cap_margin)
        s["recommended_max_new_tokens"] = rec
        logger.info("  truncated(@window)=%.1f%%  recommended max_new_tokens(p%.0f*%.2f)=%d",
                    100 * trunc, args.cap_percentile, args.cap_margin, rec)
        # realized mixture buckets (phase2_data_gen.md §9.1)
        # NOTE: the old "decode_long_8k_32k" key was computed as `tt_a >= 8192` with NO upper bound,
        # so the name said 8k-32k while the number meant ">=8k" -- it read 1.0 on long-context data and
        # told you nothing. These bands are disjoint and each one is what its name says.
        def _band(lo, hi):
            m = (tt_a >= lo) if hi is None else ((tt_a >= lo) & (tt_a < hi))
            return round(float(np.mean(m)), 4)

        s["bucket_frac"] = {
            "lt_4k": _band(0, 4096), "4k_8k": _band(4096, 8192), "8k_16k": _band(8192, 16384),
            "16k_32k": _band(16384, 32768), "32k_64k": _band(32768, 65536),
            "64k_128k": _band(65536, 131072), "ge_128k": _band(131072, None),
        }
        logger.info("  mixture buckets: %s", s["bucket_frac"])
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
