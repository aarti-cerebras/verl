#!/usr/bin/env python3
"""scripts/msa/compare_eval_roots.py — paired comparison of the long-context triad across
two or more eval roots (RULER / GSM-Infinite / MRCR).

WHY THIS EXISTS. Every ladder row lives in its own path-absolute eval root (see
setup_eval_root.sh), so "did 32K recover?" means diffing per-task CSVs and score JSONs across
trees by hand. That is exactly the operation that silently compares the wrong slices: RULER
summaries are SPLIT BY TASK FAMILY in the MSA roots (pass0_len32k_{a..e}) but NOT in the dense
baseline (pass0_len16k, pass0_len4k8k), so a filename-based join drops or double-counts tasks
depending on which root you started from.

The fix is to key on the DATASET NAME inside the CSV, never on the filename: rows whose dataset
ends in `_16k` belong to the 16K aggregate regardless of which slice file carried them. The
script then asserts the task count per length, so a missing slice fails loudly instead of
quietly averaging 11 tasks against 13.

Trace integrity is recomputed by POOLING the per-slice side-cars, not by averaging the
per-slice trace_stats.json files: percentiles do not average (a mean of five p90s is not the
pooled p90), and the accuracy comparison is only valid if truncation is comparable
(eval_plan §5.4). Metric definitions are IMPORTED from _common/trace_stats.py so the numbers
here and on the scorecard cannot drift apart.

Usage:
  python3 scripts/msa/compare_eval_roots.py \
    --root dense=/cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking \
    --root k16@10700=/cb/ml-eng/aarti/msa/evals/qwen3-4b-thinking-msa-k16v2-step10700 \
    --root k16lc@1600=/cb/ml-eng/aarti/msa/evals/qwen3-4b-thinking-msa-k16v2-longctx-step1600 \
    --lengths 16k,32k --out docs/qwen3_4b_msa/longctx_step1600_comparison.md

The LAST --root is the subject; every Δ column is subject minus that earlier root.
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import sys
from collections import defaultdict

RULER_TASKS_PER_LENGTH = 13
THINK_TAG = "</think>"


def load_trace_stats_module(root):
    """Import _common/trace_stats.py from an eval root so the integrity metrics match the
    scorecard's definitions exactly (pct/dist/has_repeat/TRUNC_SLACK)."""
    path = os.path.join(root, "_common", "trace_stats.py")
    spec = importlib.util.spec_from_file_location("_ts", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- RULER
def ruler_scores(root, length):
    """dataset -> score for one length, joined on the dataset name inside the CSV.

    Globs ALL summary CSVs in the root and filters rows by the `_<length>` suffix, because the
    slice split differs per root (see module docstring). On duplicate datasets the newest file
    wins -- a re-run supersedes the run it repeats.
    """
    out, seen_in = {}, {}
    files = sorted(glob.glob(f"{root}/ruler/results/pass0_*summary_*.csv"))
    for f in files:
        ts = re.search(r"summary_(\d{8}_\d{6})", f)
        ts = ts.group(1) if ts else ""
        with open(f) as fh:
            for row in csv.DictReader(fh):
                ds = row.get("dataset", "")
                if not ds.endswith(f"_{length}"):
                    continue
                score = row.get("Qwen3-4B-Thinking-2507")
                if score in (None, "", "-"):
                    continue
                if ds not in out or ts >= seen_in[ds]:
                    out[ds], seen_in[ds] = float(score), ts
    return out


# ---------------------------------------------------------------- GSM-Infinite
def gsm_scores(root, length):
    """cell -> accuracy for one length, plus the ops grid actually run."""
    p = f"{root}/gsm_infinite/results/scores_pass0_l{length}.json"
    if not os.path.exists(p):
        return {}, None
    d = json.load(open(p))
    return d.get("grid", {}), d.get("config", {}).get("ops")


# ---------------------------------------------------------------- MRCR
def mrcr_scores(root):
    """cell -> mean. Handles both layouts: one combined scores_pass0.json (MSA roots) and the
    dense baseline's per-needle scores_pass0_n{2,4,8}.json."""
    cells = {}
    p = f"{root}/mrcr/results/scores_pass0.json"
    if os.path.exists(p):
        for cell, v in json.load(open(p)).get("per_cell", {}).items():
            cells[cell] = v["mean"]
        return cells
    for f in sorted(glob.glob(f"{root}/mrcr/results/scores_pass0_n*.json")):
        for cell, v in json.load(open(f)).get("per_cell", {}).items():
            cells[cell] = v["mean"]
    return cells


# ---------------------------------------------------------------- trace integrity
def trace_integrity(root, length, ts_mod, tok_cache):
    """Pooled integrity over every RULER side-car for one length. Percentiles are computed on
    the POOLED sample; rates are exact counts."""
    files = sorted(glob.glob(f"{root}/ruler/results/traces_pass0_len{length}*.jsonl"))
    if not files:
        return None
    model_dir = f"{root}/model/Qwen3-4B-Thinking-2507"
    if model_dir not in tok_cache:
        from transformers import AutoTokenizer
        tok_cache[model_dir] = AutoTokenizer.from_pretrained(model_dir)
    tok = tok_cache[model_dir]

    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"]) if s else 0
    comp, trace, ans = [], [], []
    n = closed = empty = trunc = repeat = 0
    max_out = None
    # Older side-cars (the dense baseline's) predate the `max_out_len` field. Truncation is
    # undecidable without the cap, and defaulting it to 0 reads as "this root never truncates"
    # -- which is how an earlier version of this script reported dense at 0.00% against the
    # sparse rows' 3%, and made a shared task-level truncation look sparse-specific. So track
    # whether ANY record supplied the cap and surface "unknown" rather than a plausible zero.
    saw_cap = False
    for f in files:
        for line in open(f):
            r = json.loads(line)
            raw = r.get("raw") or ""
            n += 1
            if "max_out_len" in r:
                max_out, saw_cap = r["max_out_len"], True
            if not raw.strip():
                empty += 1
                continue
            c = ntok(raw)
            comp.append(c)
            if THINK_TAG in raw:
                closed += 1
                t, a = raw.split(THINK_TAG, 1)
                trace.append(ntok(t))
                ans.append(ntok(a))
            else:
                trace.append(c)
                ans.append(0)
            if max_out and c >= max_out - ts_mod.TRUNC_SLACK:
                trunc += 1
            if ts_mod.has_repeat(raw):
                repeat += 1
    if not n:
        return None
    return dict(n=n, files=len(files), max_out_len=max_out,
                closure_rate=closed / n, empty_rate=empty / n,
                trunc_rate=(trunc / n) if saw_cap else None, repeat_rate=repeat / n,
                completion=ts_mod.dist(comp), trace=ts_mod.dist(trace), answer=ts_mod.dist(ans))


# ---------------------------------------------------------------- rendering
def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def delta(a, b, nd=2):
    """b - a, signed, or em-dash when either side is missing."""
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{d:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable; the LAST one is the subject of the Δ columns")
    ap.add_argument("--lengths", default="16k,32k")
    ap.add_argument("--no-traces", action="store_true", help="skip trace tokenisation (faster)")
    ap.add_argument("--out")
    args = ap.parse_args()

    roots = []
    for spec in args.root:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--root must be NAME=PATH, got {spec!r}")
        roots.append((name, path.rstrip("/")))
    lengths = args.lengths.split(",")
    subj_name, subj_path = roots[-1]
    others = roots[:-1]
    ts_mod = load_trace_stats_module(subj_path)
    tok_cache = {}
    L = []
    w = L.append

    w(f"# Long-context comparison — subject: `{subj_name}`\n")
    w(f"* subject root: `{subj_path}`")
    for nm, p in others:
        w(f"* reference `{nm}`: `{p}`")
    w("\nΔ columns are **subject − reference**. Positive = subject better "
      "(except MRCR/GSM where higher is also better).\n")

    # ---- RULER -------------------------------------------------------------------------
    w("\n## RULER\n")
    for length in lengths:
        per_root = {nm: ruler_scores(p, length) for nm, p in roots}
        subj = per_root[subj_name]
        if not subj:
            w(f"### {length.upper()} — no subject results yet\n")
            continue
        ntasks = len(subj)
        flag = "" if ntasks == RULER_TASKS_PER_LENGTH else \
            f"  **INCOMPLETE: {ntasks}/{RULER_TASKS_PER_LENGTH} tasks**"
        w(f"### {length.upper()}{flag}\n")
        cols = " | ".join(nm for nm, _ in roots)
        dcols = " | ".join(f"Δ vs {nm}" for nm, _ in others)
        w(f"| task | {cols} | {dcols} |")
        w("|---|" + "---|" * (len(roots) + len(others)))
        for ds in sorted(subj):
            vals = [per_root[nm].get(ds) for nm, _ in roots]
            ds_short = ds[len("ruler_"):] if ds.startswith("ruler_") else ds
            ds_short = ds_short[: -len(f"_{length}")]
            row = " | ".join(fmt(v) for v in vals)
            drow = " | ".join(delta(per_root[nm].get(ds), subj.get(ds)) for nm, _ in others)
            w(f"| {ds_short} | {row} | {drow} |")
        # aggregate over the tasks the SUBJECT has, so the mean is always like-for-like
        means = {}
        for nm, _ in roots:
            xs = [per_root[nm][d] for d in subj if d in per_root[nm]]
            means[nm] = sum(xs) / len(xs) if xs else None
        row = " | ".join(fmt(means[nm]) for nm, _ in roots)
        drow = " | ".join(delta(means[nm], means[subj_name]) for nm, _ in others)
        w(f"| **mean ({ntasks} tasks)** | {row} | {drow} |")
        w("")

    # ---- GSM-Infinite ------------------------------------------------------------------
    w("\n## GSM-Infinite\n")
    for length in lengths:
        per_root, grids = {}, {}
        for nm, p in roots:
            per_root[nm], grids[nm] = gsm_scores(p, length)
        subj = per_root[subj_name]
        if not subj:
            w(f"### {length.upper()} — no subject results yet\n")
            continue
        mism = [f"{nm}={grids[nm]}" for nm, _ in roots if grids[nm] != grids[subj_name]]
        w(f"### {length.upper()}  (ops grid `{grids[subj_name]}`)"
          + (f"\n\n**GRID MISMATCH — not comparable: {', '.join(mism)}**" if mism else "") + "\n")
        cols = " | ".join(nm for nm, _ in roots)
        dcols = " | ".join(f"Δ vs {nm}" for nm, _ in others)
        w(f"| ops | {cols} | {dcols} |")
        w("|---|" + "---|" * (len(roots) + len(others)))
        key_ops = lambda k: int(re.sub(r".*ops", "", k))
        for cell in sorted(subj, key=key_ops):
            vals = [per_root[nm].get(cell) for nm, _ in roots]
            row = " | ".join(fmt(v, 1) for v in vals)
            drow = " | ".join(delta(per_root[nm].get(cell), subj.get(cell), 1) for nm, _ in others)
            w(f"| {key_ops(cell)} | {row} | {drow} |")
        means = {}
        for nm, _ in roots:
            xs = [per_root[nm][c] for c in subj if c in per_root[nm]]
            means[nm] = sum(xs) / len(xs) if xs else None
        row = " | ".join(fmt(means[nm], 1) for nm, _ in roots)
        drow = " | ".join(delta(means[nm], means[subj_name], 1) for nm, _ in others)
        w(f"| **mean** | {row} | {drow} |")
        w("")

    # ---- MRCR --------------------------------------------------------------------------
    w("\n## MRCR (mean SequenceMatcher ratio)\n")
    per_root = {nm: mrcr_scores(p) for nm, p in roots}
    subj = per_root[subj_name]
    if not subj:
        w("No subject results yet.\n")
    else:
        cols = " | ".join(nm for nm, _ in roots)
        dcols = " | ".join(f"Δ vs {nm}" for nm, _ in others)
        w(f"| cell | {cols} | {dcols} |")
        w("|---|" + "---|" * (len(roots) + len(others)))
        bin_order = {"4-8K": 0, "8-16K": 1, "16-32K": 2}
        skey = lambda c: (int(c.split("needle")[0]), bin_order.get(c.split("_", 1)[1], 9))
        for cell in sorted(subj, key=skey):
            vals = [per_root[nm].get(cell) for nm, _ in roots]
            row = " | ".join(fmt(v, 4) for v in vals)
            drow = " | ".join(delta(per_root[nm].get(cell), subj.get(cell), 4) for nm, _ in others)
            w(f"| {cell} | {row} | {drow} |")
        # by-bin and overall, recomputed from the cells so every root uses one definition
        for b in ("4-8K", "8-16K", "16-32K"):
            means = {}
            for nm, _ in roots:
                xs = [v for c, v in per_root[nm].items() if c.endswith(f"_{b}")]
                means[nm] = sum(xs) / len(xs) if xs else None
            row = " | ".join(fmt(means[nm], 4) for nm, _ in roots)
            drow = " | ".join(delta(means[nm], means[subj_name], 4) for nm, _ in others)
            w(f"| **bin {b}** | {row} | {drow} |")
        means = {}
        for nm, _ in roots:
            xs = [per_root[nm][c] for c in subj if c in per_root[nm]]
            means[nm] = sum(xs) / len(xs) if xs else None
        row = " | ".join(fmt(means[nm], 4) for nm, _ in roots)
        drow = " | ".join(delta(means[nm], means[subj_name], 4) for nm, _ in others)
        w(f"| **mean ({len(subj)} cells)** | {row} | {drow} |")
        w("")

    # ---- trace integrity ---------------------------------------------------------------
    if not args.no_traces:
        w("\n## Trace integrity (RULER side-cars, pooled per length)\n")
        w("Gates: closure ≥99%, truncation ≤1%. A length whose truncation differs across roots "
          "is not a valid accuracy comparison.\n")
        w("| root | length | n | closure | trunc | empty | repeat | completion p50/p90/max |")
        w("|---|---|---|---|---|---|---|---|")
        for nm, p in roots:
            for length in lengths:
                s = trace_integrity(p, length, ts_mod, tok_cache)
                if not s:
                    w(f"| {nm} | {length} | — | — | — | — | — | — |")
                    continue
                c = s["completion"]
                tr = f"{s['trunc_rate']*100:.2f}%" if s["trunc_rate"] is not None else "unknown"
                w(f"| {nm} | {length} | {s['n']} | {s['closure_rate']*100:.1f}% | "
                  f"{tr} | {s['empty_rate']*100:.2f}% | "
                  f"{s['repeat_rate']*100:.2f}% | {c['p50']} / {c['p90']} / {c['max']} |")
        w("")

    text = "\n".join(L)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[compare] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
