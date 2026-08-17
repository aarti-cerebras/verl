#!/usr/bin/env python3
"""scripts/msa/length_histograms.py — prompt/response length histograms, several eval roots at once.

`_common/length_report.py` already does this for ONE root and emits ASCII. This does it for N roots
and emits both (a) the same ASCII, per root, and (b) an overlaid PNG per benchmark so the dense and
sparse distributions can be read against each other. The comparison is the point: a sparse model's
realized selection ratio is k/(prompt+generated), so a *shift in the response distribution* between
dense and sparse changes the effective sparsity mid-generation, and that is invisible in a scorecard.

Deliberate differences from `_common/length_report.py`:
  * tokenizes with `tokenizers` (tokenizer.json) instead of `transformers` — the eval venvs' python
    symlinks are broken outside the container, and this needs no torch.
  * caches per-item token counts to `<out>/cache/<root>__<bench>.json`, so re-rendering the charts
    does not re-tokenize ~1 GB of traces.
  * covers LiveCodeBench, which has no `traces_pass*.jsonl` side-car — its lengths come from the
    harness's own `output/**/Scenario.codegeneration_*.json`. See LCB_NOTE: those prompts are the
    raw problem text, NOT the formatted chat prompt, so LCB prompt lengths are a lower bound and
    are labelled as such rather than being silently plotted next to exact ones.

Usage:
  python3 scripts/msa/length_histograms.py                    # all three default roots
  python3 scripts/msa/length_histograms.py --bench gpqa ruler
  python3 scripts/msa/length_histograms.py --no-tokenize      # charts only, from cache
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

DSA = "/cb/ml-eng/aarti/dsa/evals"
MSA = "/cb/ml-eng/aarti/msa/evals"

# label -> eval root. Order is the categorical slot order, so it is also the legend order.
ROOTS = [
    ("dense baseline", f"{DSA}/qwen3-4b-thinking"),
    ("MSA k8 s11214", f"{MSA}/qwen3-4b-thinking-msa-k8v2-step11214"),
    ("MSA k16 s10700", f"{MSA}/qwen3-4b-thinking-msa-k16v2-step10700"),
]
TOKENIZER = f"{MSA}/qwen3-4b-thinking-msa-k16v2-step10700/model/Qwen3-4B-Thinking-2507/tokenizer.json"

# Reference-palette categorical slots 1-3 (blue / orange / aqua). Documented all-pairs safe in both
# modes; not re-derived here. Slot order is the CVD-safety mechanism -- do not reorder or cycle.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd6"
SURFACE = "#fcfcfb"

# same edges as _common/length_report.py, so the two reports bin identically
BINS = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
THINK_TAG = "</think>"

BENCHES = ["ifeval", "gpqa", "mmlu_pro", "aime25", "aime25_cap32k",
           "livecodebench_v6", "livecodebench_v6_cap32k", "ruler", "mrcr", "gsm_infinite"]

LCB_NOTE = ("prompt = raw problem text (`question_content` + `starter_code`); the LCB harness "
            "builds the actual chat prompt at request time and does not persist it, so this is a "
            "lower bound, not the served prompt")

# `run_gsm_infinite.py` writes only completions -- prompts live in the source parquets and
# reconstructing which 20 rows per cell were sampled would mean re-deriving the sampler. Left
# unavailable rather than approximated. The nominal bands are in the cell names (0/8K/16K/32K).
BENCH_NOTES = {
    "gsm_infinite": "prompts not persisted by run_gsm_infinite.py; nominal context bands are "
                    "0 / 8K / 16K / 32K by construction",
}


# ---------------------------------------------------------------- collection

def bin_label(lo, hi):
    def f(v):
        return f"{v // 1024}K" if v >= 1024 else str(v)
    return f"{f(lo)}-{f(hi)}" if hi is not None else f">{f(lo)}"


LABELS = [bin_label(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)] + [bin_label(BINS[-1], None)]


def histogram(xs):
    counts = [0] * len(BINS)
    for x in xs:
        for i in range(len(BINS) - 1):
            if BINS[i] <= x < BINS[i + 1]:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    return counts


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1))))]


def stat_line(xs):
    if not xs:
        return "n=0"
    return (f"n={len(xs)} min={min(xs)} p50={pct(xs,50)} p90={pct(xs,90)} "
            f"p99={pct(xs,99)} max={max(xs)}")


class Tok:
    """Batched tokenizer. encode_batch_fast skips offset/word bookkeeping we never read."""

    def __init__(self, path):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(path)
        self.fast = getattr(self.t, "encode_batch_fast", None) or self.t.encode_batch

    def counts(self, texts, chunk=512):
        out = []
        for i in range(0, len(texts), chunk):
            out.extend(len(e.ids) for e in self.fast(texts[i:i + chunk], add_special_tokens=False))
        return out


def newest_artifact(root, bench):
    """mtime of the most recently written artifact for this bench, or 0.

    An eval suite can be RUNNING while this script reads it -- that actually happened: the k16
    s10700 AIME25 @81920 run started mid-session and a snapshot caught 115 of 480 completions,
    which would have been charted as a finished distribution. Every cached entry therefore records
    the source mtime it was built from, so a stale cache is detected and a still-growing bench is
    held back rather than plotted.
    """
    newest = 0.0
    for pat in ("results/traces_pass*.jsonl", "results/scores_pass*.json",
                "output/*/Scenario.codegeneration_*.json"):
        for f in glob.glob(os.path.join(root, bench, pat)):
            try:
                newest = max(newest, os.path.getmtime(f))
            except OSError:
                pass
    return newest


def collect_bench(root, bench, tok):
    """-> {'prompt': [...], 'response': [...], 'total': [...], 'caps': [...], 'per_dataset': {...}}

    `total` is only populated where an item's prompt and response are joined by the side-car's
    `prompt_tokens` field. Runs predating that field get `total: []` rather than an estimate --
    p90(a)+p90(b) != p90(a+b) unless the two are perfectly rank-correlated.
    """
    bdir = os.path.join(root, bench)
    resp, prm_paired, totals, caps = [], [], [], set()
    per_dataset = defaultdict(list)

    if bench.startswith("livecodebench"):
        fs = glob.glob(os.path.join(bdir, "output", "*", "Scenario.codegeneration_*.json"))
        fs = [f for f in fs if "_eval" not in os.path.basename(f)]
        if not fs:
            return None
        probs = json.load(open(sorted(fs)[-1]))
        gens = [g for p in probs for g in (p.get("output_list") or []) if g]
        prompts = [(p.get("question_content") or "") + (p.get("starter_code") or "") for p in probs]
        return {"prompt": tok.counts(prompts), "response": tok.counts(gens), "total": [],
                "caps": [], "per_dataset": {}, "note": LCB_NOTE}

    tfiles = sorted(glob.glob(os.path.join(bdir, "results", "traces_pass*.jsonl")))
    if not tfiles:
        return None
    join = {"paired_by_field": 0, "paired_by_answer": 0, "unjoined": 0, "ambiguous": 0}

    for tf in tfiles:
        # traces_<tag>.jsonl  <->  oc_workdir/<tag>/<ts>/predictions/...   (verified exact on
        # ifeval/gpqa/aime25/ruler/mmlu_pro). Joining PER PASS matters: pooling all passes would
        # let one pass's completion match another pass's prompt.
        tag = re.sub(r"^traces_|\.jsonl$", "", os.path.basename(tf))
        recs = []
        for line in open(tf):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("raw"):
                recs.append(rec)
        if not recs:
            continue
        counts = tok.counts([r["raw"] for r in recs])
        for r, c in zip(recs, counts):
            if r.get("max_out_len"):
                caps.add(r["max_out_len"])
        resp.extend(counts)

        # 1. the cheap path: the side-car carries the prompt length (runs after 2026-07-30)
        need = []
        for r, c in zip(recs, counts):
            p = r.get("prompt_tokens")
            if isinstance(p, int) and p > 0:
                prm_paired.append(p)
                totals.append(p + c)
                join["paired_by_field"] += 1
            else:
                need.append((r, c))
        if not need:
            continue

        # 2. the recovery path for older runs: the scored prediction IS the completion with the
        # trace stripped, so `raw.split('</think>')[-1]` is a content key into the predictions,
        # which carry `origin_prompt`. Verified on the dense baseline's ifeval: 541/541 joined.
        pdir = os.path.join(bdir, "oc_workdir", tag)
        by_answer = defaultdict(list)
        for f in glob.glob(os.path.join(pdir, "**", "predictions", "*", "*.json"), recursive=True):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            items = [it for it in data.values() if isinstance(it, dict)]
            texts = []
            for it in items:
                o = it.get("origin_prompt", "")
                if isinstance(o, list):
                    o = "\n".join((m.get("prompt") or m.get("content") or "")
                                  if isinstance(m, dict) else str(m) for m in o)
                texts.append(str(o))
            if not texts:
                continue
            ptoks = tok.counts(texts)
            for it, pt in zip(items, ptoks):
                by_answer[(it.get("prediction") or "").strip()].append(pt)
        for r, c in need:
            key = r["raw"].split(THINK_TAG)[-1].strip()
            bucket = by_answer.get(key)
            if bucket:
                if len(bucket) > 1:
                    join["ambiguous"] += 1
                p = bucket.pop()
                prm_paired.append(p)
                totals.append(p + c)
                join["paired_by_answer"] += 1
            else:
                join["unjoined"] += 1

    # prompts per dataset, from the scored predictions -- the only place they are broken out by
    # dataset name, which the side-car cannot give
    for f in glob.glob(os.path.join(bdir, "oc_workdir", "**", "predictions", "*", "*.json"),
                       recursive=True):
        abbr = re.sub(r"_\d+$", "", os.path.splitext(os.path.basename(f))[0])
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        texts = []
        for item in data.values():
            if not isinstance(item, dict):
                continue
            o = item.get("origin_prompt", "")
            if isinstance(o, list):
                o = "\n".join((m.get("prompt") or m.get("content") or "")
                              if isinstance(m, dict) else str(m) for m in o)
            texts.append(str(o))
        if texts:
            per_dataset[abbr].extend(tok.counts(texts))

    prompt = prm_paired or [x for v in per_dataset.values() for x in v]
    return {"prompt": prompt, "response": resp, "total": totals, "caps": sorted(caps),
            "per_dataset": {k: v for k, v in per_dataset.items()}, "join": join}


# ---------------------------------------------------------------- ascii

def bar(count, mx, width=32):
    return "#" * (0 if not mx else int(round(width * count / mx)))


def render_ascii(title, xs):
    if not xs:
        return [f"**{title}** — no data", ""]
    out = [f"**{title}** — {stat_line(xs)}", "", "```"]
    counts = histogram(xs)
    mx = max(counts)
    for lab, c in zip(LABELS, counts):
        if c:
            out.append(f"  {lab:>12} | {bar(c, mx):<32} {c:>6}  {100*c/len(xs):5.1f}%")
    out += ["```", ""]
    return out


def write_ascii(label, root, data, path):
    doc = [f"# Realized length distributions — {label}", "",
           f"Eval root: `{root}`", "",
           "`response` = the FULL completion (thinking trace + `</think>` + answer) — what the model",
           "actually generates and attends over, not the scored answer alone. `total` = prompt +",
           "response = the sequence length at end-of-generation, i.e. the length that sets the",
           "realized selection ratio for a sparse model.", "",
           "Bin edges match `_common/length_report.py` so the two reports are directly comparable.", ""]
    for bench in BENCHES:
        d = data.get(bench)
        if not d:
            continue
        doc += [f"## {bench}", ""]
        note = d.get("note") or BENCH_NOTES.get(bench, "")
        if note:
            doc += [f"> {note}", ""]
        doc += render_ascii(f"{bench} — prompt tokens", d["prompt"])
        doc += render_ascii(f"{bench} — response tokens (full completion)", d["response"])
        if d["total"]:
            doc += render_ascii(f"{bench} — total tokens (prompt + response)", d["total"])
            j = d.get("join") or {}
            if j.get("paired_by_answer"):
                doc += [f"pairing: {j['paired_by_field']} by `prompt_tokens` side-car, "
                        f"{j['paired_by_answer']} recovered by answer-text join, "
                        f"{j['unjoined']} unjoined. Of the recovered, **{j['ambiguous']} "
                        f"({100*j['ambiguous']/max(1,j['paired_by_answer']):.1f}%)** matched an "
                        "answer text shared by more than one item, so their prompt is assigned "
                        "arbitrarily within that group — an error term on this histogram only.", ""]
        else:
            doc += [f"**{bench} — total tokens** — `unpaired`: no key joins a completion to its "
                    "prompt in this run (traces predate the `prompt_tokens` side-car). Not estimated "
                    "by summing percentiles — that is only valid under perfect rank correlation.", ""]
        if d["caps"]:
            cap = min(d["caps"])
            over = [r for r in d["response"] if r >= cap - 16]
            doc += [f"max_out_len in effect: {d['caps']} — {len(over)} of {len(d['response'])} "
                    f"completions at/near the cap "
                    f"({100*len(over)/max(1,len(d['response'])):.2f}% truncated)", ""]
        if len(d["per_dataset"]) > 1:
            doc += ["<details><summary>prompt tokens per dataset</summary>", ""]
            for abbr, xs in sorted(d["per_dataset"].items()):
                doc.append(f"- `{abbr}`: {stat_line(xs)}")
            doc += ["", "</details>", ""]
    open(path, "w").write("\n".join(doc))
    return path


# ---------------------------------------------------------------- charts

PANELS = [("prompt", "prompt"), ("response", "response (full completion)"),
          ("total", "overall (prompt + response)")]


def tickfmt(v):
    if v >= 1000:
        return f"{v/1000:.0f}K" if v % 1000 == 0 or v >= 10000 else f"{v/1000:.1f}K"
    return f"{v:.0f}"


def chart(bench, series, out_png, nbins=60):
    """series = [(slot, label, {'prompt':[], 'response':[], 'total':[]}, note)], ROOTS order.

    Small multiples: one ROW per checkpoint, one COLUMN per length kind, each a filled histogram
    over fine LINEAR bins. Rows rather than overlays because at 60 bins three filled distributions
    on one axis are unreadable, and the alternative (outlines) hides the shape that fine bins exist
    to show. A column shares its bin edges AND its y-scale across rows, so the rows are directly
    comparable by eye -- that is what makes small multiples work here.

    `slot` is the model's fixed index into the categorical palette -- NOT its position among the
    series present. A benchmark where one checkpoint has no run (aime25 @81920 for k16) must not
    repaint the survivors.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    live = [s for s in series if any(s[2].get(k) for k, _ in PANELS)]
    if not live:
        return None

    # shared linear edges per column, spanning the union of that column's realized range
    cols, edges = [], {}
    for key, kind in PANELS:
        allv = [v for s in live for v in (s[2].get(key) or [])]
        if not allv:
            continue
        lo, hi = min(allv), max(allv)
        if hi <= lo:
            hi = lo + 1
        cols.append((key, kind))
        edges[key] = np.linspace(lo, hi, nbins + 1)
    if not cols:
        return None

    nrow, ncol = len(live), len(cols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 1.85 * nrow + 0.9),
                             facecolor=SURFACE, squeeze=False, sharex="col", sharey="col")
    absent = []
    for r, (slot, label, vals, _) in enumerate(live):
        for c, (key, kind) in enumerate(cols):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            xs = vals.get(key) or []
            e = edges[key]
            if xs:
                counts, _ = np.histogram(xs, bins=e)
                # PMF: bar heights are P(length in bin) and sum to 1 over the panel. Not a density
                # -- bins are equal-width within a column, so the two differ by a constant, and a
                # mass is what you actually want to read off ("23% of items pile up at the cap").
                pmf = counts / len(xs)
                ax.bar(e[:-1], pmf, width=(e[1] - e[0]) * 0.9, align="edge",
                       color=SERIES_LIGHT[slot], linewidth=0, zorder=3)
                # mean / p50 / p90 as rules in TEXT ink, not the series colour -- they are
                # annotation, and colouring them would read as a fourth series. Distinguished by
                # line style (solid / dashed / dotted), spelled out in the figure footer, with the
                # values in a corner box so labels can never collide with each other or the bars.
                mu, q50, q90, q99 = (sum(xs) / len(xs), pct(xs, 50), pct(xs, 90), pct(xs, 99))
                for v, ls, lw in ((mu, "-", 1.4), (q50, "--", 1.1), (q90, ":", 1.1),
                                  ((q99), (0, (3, 1, 1, 1)), 1.1)):
                    ax.axvline(v, color=INK, linestyle=ls, linewidth=lw, alpha=0.75, zorder=6)
                ax.text(0.985, 0.94,
                        f"μ {tickfmt(mu)}\np50 {tickfmt(q50)}\np90 {tickfmt(q90)}"
                        f"\np99 {tickfmt(q99)}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7,
                        color=INK_2, linespacing=1.35)
            else:
                absent.append(f"{label} has no {key}")
                ax.text(0.5, 0.5, "not run / not persisted", ha="center", va="center",
                        transform=ax.transAxes, color=INK_2, fontsize=8.5)
            if r == 0:
                ax.set_title(f"{bench} — {kind} tokens", color=INK, fontsize=10, pad=8, loc="left")
            if c == 0:
                # direct row label in text ink -- identity is never colour-alone, so no legend
                ax.set_ylabel(f"{label}\nprobability", color=INK, fontsize=8.5)
            ax.tick_params(axis="both", labelsize=7.5, colors=INK_2, length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            if r == nrow - 1:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: tickfmt(v)))
                ax.set_xlabel("tokens", color=INK_2, fontsize=8)

    for c, (key, _) in enumerate(cols):
        e = edges[key]
        axes[0][c].set_xlim(e[0], e[-1])

    notes = sorted({s[3] for s in live if s[3]}) + sorted(set(absent))
    notes.append(f"{nbins} linear bins per column; column shares bin edges and y-scale across rows")
    notes.append("bar heights are a PMF (sum to 1 per panel); rules = mean (solid), "
                 "p50 (dashed), p90 (dotted), p99 (dash-dot)")
    fig.text(0.006, 0.006, "note: " + "; ".join(notes), fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/qwen3_4b_msa/lengths")
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--no-tokenize", action="store_true",
                    help="render from cache only; fail loudly on a cache miss")
    ap.add_argument("--bins", type=int, default=60, help="linear bins per column")
    ap.add_argument("--quiet-window", type=float, default=1800,
                    help="seconds an artifact must be untouched before it counts as finished")
    a = ap.parse_args()

    benches = a.bench or BENCHES
    cache_dir = os.path.join(a.out, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    tok = None
    now = time.time()

    collected = {}
    for label, root in ROOTS:
        slug = os.path.basename(root)
        collected[label] = {}
        for bench in benches:
            cf = os.path.join(cache_dir, f"{slug}__{bench}.json")
            mt = newest_artifact(root, bench)
            # a bench whose newest artifact is younger than the quiet window is still being
            # written; hold it back rather than charting a partial distribution as a finished one
            if mt and (now - mt) < a.quiet_window:
                age = int(now - mt)
                print(f"[in-flight] {slug}/{bench}: last write {age}s ago "
                      f"(< {a.quiet_window}s quiet window) — excluded", file=sys.stderr)
                if os.path.exists(cf):
                    os.remove(cf)
                continue
            if os.path.exists(cf):
                cached = json.load(open(cf))
                if mt and cached.get("src_mtime", 0) + 1 < mt:
                    print(f"[stale] {slug}/{bench}: artifacts newer than cache — recollecting",
                          file=sys.stderr)
                else:
                    collected[label][bench] = cached
                    continue
            if a.no_tokenize:
                print(f"[skip] no cache for {slug}/{bench}", file=sys.stderr)
                continue
            if not os.path.isdir(os.path.join(root, bench)):
                continue
            if tok is None:
                tok = Tok(TOKENIZER)
            print(f"[tokenize] {slug}/{bench} …", flush=True)
            d = collect_bench(root, bench, tok)
            if d is None:
                print(f"[tokenize] {slug}/{bench}: no artifacts", flush=True)
                continue
            d["src_mtime"] = mt
            json.dump(d, open(cf, "w"))
            collected[label][bench] = d
            print(f"[tokenize] {slug}/{bench}: prompt n={len(d['prompt'])} "
                  f"response n={len(d['response'])}", flush=True)

    for label, root in ROOTS:
        slug = os.path.basename(root)
        p = write_ascii(label, root, collected[label], os.path.join(a.out, f"LENGTHS_{slug}.md"))
        print(f"[ascii] -> {p}")

    for bench in benches:
        # slot = index into ROOTS = index into the categorical palette, fixed per model
        series = [(slot, label,
                   {k: collected[label].get(bench, {}).get(k, []) for k in ("prompt", "response",
                                                                            "total")},
                   collected[label].get(bench, {}).get("note", "") or BENCH_NOTES.get(bench, ""))
                  for slot, (label, _) in enumerate(ROOTS)]
        p = chart(bench, series, os.path.join(a.out, f"{bench}.png"), nbins=a.bins)
        print(f"[chart] -> {p}" if p else f"[chart] {bench}: no data")


if __name__ == "__main__":
    main()
