#!/usr/bin/env python3
"""scripts/msa/score_bars.py — benchmark scores across checkpoints, as bar charts.

Numbers are NOT hand-copied from any scorecard: this imports the eval tree's own
`_common/collect_scorecard.py` and recomputes every value from the raw per-bench artifacts, so a
chart cannot drift from the data the way a transcribed table does.

Outputs:
  scores.png                 headline number per benchmark, one small-multiple panel each
  scores_ruler.png           per-length task breakdown; each row = that length's 13 tasks + the
                             length mean, and a final row = the four lengths + the RULER mean
  scores_mmlu_pro.png        14 categories + the macro-average
  scores_mrcr.png            9 cells + the mean, arranged one row per needle count
  scores_gsm_infinite.png    one row per context length (6 ops + that row's mean), then a final
                             row of the four lengths + the grand mean
  scores_livecodebench.png   per problem difficulty (easy 31 / medium 39 / hard 61) + the
                             131-problem mean, one row per output budget
  scores_aime.png            per pass + the average, both AIME budgets -- avg@16 over 30 items,
                             where one pass carries ~+/-9 pts, so the spread is the whole story
  scores_gpqa.png            per pass + the avg@4
  scores_ifeval.png          the four IFEval scorings; the last group RESTATES prompt-strict (the
                             reported headline) because the four are not a partition

Every subset row ends with its OVERALL bar, offset by a gap, so the aggregate is always readable
against the parts it is made of -- an aggregate that moves while its cells do not (or vice versa)
is the whole diagnostic.

Only the best checkpoint per geometry is charted (see MODELS). The full checkpoint ladder lives in
docs/qwen3_4b_msa/eval_scorecard_all_ckpts.md.

Usage:
  python3 scripts/msa/score_bars.py [--outdir docs/qwen3_4b_msa]
"""
import argparse
import importlib.util
import os
import re
import statistics as st
import sys

DSA = "/cb/ml-eng/aarti/dsa/evals"
MSA = "/cb/ml-eng/aarti/msa/evals"
COLLECTOR = f"{MSA}/qwen3-4b-thinking-msa-k16v2-step10700/_common/collect_scorecard.py"

# Best checkpoint per geometry. k8 s11214 and k16 s10700 are the last and strongest of their runs
# (s11214 leads its siblings on MMLU-Pro / LCB / RULER 16K; s10700 leads everything at k16). The
# intermediate checkpoints are in the ladder doc, not here.
MODELS = [
    ("dense", f"{DSA}/qwen3-4b-thinking", "#2a78d6"),
    ("MSA k8 s11214", f"{MSA}/qwen3-4b-thinking-msa-k8v2-step11214", "#eb6834"),
    ("MSA k16 s10700", f"{MSA}/qwen3-4b-thinking-msa-k16v2-step10700", "#1baf7a"),
]
# Reference-palette categorical slots 1-3 -- the three documented as clearing the ALL-PAIRS gates
# in both modes. Same model, same colour as the length histograms; a reader should never have to
# re-learn the palette between two figures in one report.

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dedcd6", "#fcfcfb"

CARD = {  # Qwen3-4B-Thinking-2507 model card; no published number exists for RULER/MRCR/GSM-Inf
    "IFEval": 87.4,
    "GPQA-diamond": 65.8,
    "MMLU-Pro": 74.0,
    "AIME25 @81920": 81.3,
    "LiveCodeBench v6 @81920": 55.2,
}

# A suite can be RUNNING while this reads it. `collect()` happily averages however many passes
# exist, so a 2-of-16-pass AIME returns a real-looking number with a fat stderr -- that actually
# happened here (k16 s10700 AIME @81920 came back 83.34 +/- 3.34 at n=2 mid-session). Any
# pass-averaged metric therefore declares the pass count it needs, and short results are withheld
# as n/a rather than charted as finished.
EXPECT_N = {"gpqa": 4, "aime_81920": 16, "aime_32768": 16}

HEADLINE = [
    ("IFEval", "ifeval", 1),
    ("GPQA-diamond", "gpqa", 1),
    ("MMLU-Pro", "mmlu_pro", 1),
    ("AIME25 @81920", "aime_81920", 1),
    ("AIME25 @32768", "aime_32768", 1),
    ("LiveCodeBench v6 @81920", "lcb_81920", 1),
    ("LiveCodeBench v6 @32768", "lcb_32768", 1),
    ("RULER 4K", "ruler:4k", 1),
    ("RULER 8K", "ruler:8k", 1),
    ("RULER 16K", "ruler:16k", 1),
    ("RULER 32K", "ruler:32k", 1),
    ("MRCR (mean ratio)", "mrcr", 3),
    ("GSM-Infinite (mean)", "gsm", 1),
]


def headline_value(r, key):
    """-> (value, stderr) with incomplete pass-averaged results withheld as None."""
    if r is None:
        return None, 0.0
    if key.startswith("ruler:"):
        cell = (r.get("ruler") or {}).get(key.split(":", 1)[1])
        return (cell[0], 0.0) if cell and cell[0] is not None else (None, 0.0)
    raw = r.get(key)
    if isinstance(raw, (tuple, list)) and len(raw) == 3:
        mean, se, n = raw
        need = EXPECT_N.get(key)
        if need is not None and n < need:
            print(f"[incomplete] {key}: {n}/{need} passes — withheld", file=sys.stderr)
            return None, 0.0
        return float(mean), float(se)
    return value_err(raw)

RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
               "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
               "vt", "cwe", "fwe", "qa_squad", "qa_hotpotqa"]
MMLU_CATS = ["math", "physics", "chemistry", "law", "engineering", "other", "history",
             "economics", "health", "psychology", "business", "biology", "philosophy",
             "computer_science"]
GSM_OPS = ["ops2", "ops5", "ops10", "ops15", "ops20", "ops30"]
GSM_LENS = ["len0", "len8k", "len16k", "len32k"]
MRCR_NEEDLES = ["2needle", "4needle", "8needle"]
MRCR_BANDS = ["4-8K", "8-16K", "16-32K"]


def load_collector():
    spec = importlib.util.spec_from_file_location("collect_scorecard", COLLECTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def value_err(v):
    """collect() returns a bare float, (mean, stderr, n), (score, n), or None."""
    if v is None:
        return None, 0.0
    if isinstance(v, (tuple, list)):
        if len(v) == 3:
            return float(v[0]), float(v[1])
        return (float(v[0]), 0.0) if v else (None, 0.0)
    return float(v), 0.0


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="both", labelsize=7.5, colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def zoom(ax, values, card=None, headroom=0.12):
    """Zoom to the data band. A 0-100 axis hides a 3-pt gap, which is the question being asked.

    `headroom` is a fraction of the VISIBLE span reserved above the tallest bar for value labels.
    It must not scale with the data range: LCB spans 26-96, and a range-proportional pad drove the
    axis to 175 while a 96-pt bar sat at half height.
    """
    good = [v for v in values if v is not None]
    if not good:
        return
    lo, hi = min(good), max(good)
    if card is not None:
        lo, hi = min(lo, card), max(hi, card)
    bottom = max(0, lo - max((hi - lo) * 0.32, abs(hi) * 0.02, 1e-6))
    span = max(hi - bottom, 1e-6)
    ax.set_ylim(bottom, hi + span * headroom)


def grouped_row(ax, title, subset_labels, series_vals, nd=1, card=None, errs=None):
    """One axes: grouped bars over subsets, then a gap, then the OVERALL group.

    series_vals[i] is the list of values for model i, length len(subset_labels); the caller has
    already appended the overall value as the last entry.
    """
    import numpy as np

    n = len(MODELS)
    # a visual gap before the last group so 'overall' reads as an aggregate, not another subset
    x = np.arange(len(subset_labels), dtype=float)
    x[-1] += 0.7
    w = 0.8 / n
    flat = []
    for si, vals in enumerate(series_vals):
        for xi, v in zip(x, vals):
            if v is None:
                continue
            flat.append(v)
            ax.bar(xi + (si - (n - 1) / 2) * w, v, width=w * 0.88, color=MODELS[si][2],
                   linewidth=0, zorder=3, label=MODELS[si][0])
            if v == 0:
                # a genuine 0.0 draws no bar and would read as "missing"; GSM-Infinite
                # len32k x ops30 is exactly this for all three models
                ax.text(xi + (si - (n - 1) / 2) * w, 0.015, "0", ha="center", va="bottom",
                        fontsize=6.5, color=MODELS[si][2], zorder=6, rotation=90,
                        transform=ax.get_xaxis_transform())
            else:
                # rotated 90 deg: at 3 bars x up to 14 groups a horizontal label overlaps its
                # neighbours, and the reader needs the exact cell value more than the silhouette
                ax.text(xi + (si - (n - 1) / 2) * w, v, f" {v:.{nd}f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK, zorder=6, rotation=90)
        if errs:
            for xi, v, e in zip(x, vals, errs[si]):
                if v is not None and e:
                    ax.errorbar(xi + (si - (n - 1) / 2) * w, v, yerr=e, fmt="none",
                                ecolor=INK_2, elinewidth=1.0, capsize=2, zorder=5)
    # mark the cells a model is individually missing, so a gap never reads as a low score
    for i in range(len(subset_labels)):
        if all(vals[i] is None for vals in series_vals):
            continue
        for si, vals in enumerate(series_vals):
            if vals[i] is None:
                ax.text(x[i] + (si - (n - 1) / 2) * w, 0.02, "n/a", ha="center", va="bottom",
                        fontsize=6, color=INK_2, rotation=90, zorder=6,
                        transform=ax.get_xaxis_transform())
    if card is not None:
        ax.axhline(card, color=INK, linestyle="--", linewidth=1.3, alpha=0.8, zorder=7)
    if len(x) > 1:
        ax.axvline((x[-2] + x[-1]) / 2, color=GRID, linewidth=1.2, zorder=1)
    ax.set_title(title + (f"   ·   card {card} (dashed)" if card is not None else ""),
                 color=INK, fontsize=10, loc="left", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(subset_labels, rotation=40, ha="right", fontsize=7, color=INK_2)
    style(ax)
    zoom(ax, flat, card, headroom=0.30)  # room for the rotated value labels


def finish(fig, out, extra_notes=()):
    notes = ["y-axes are ZOOMED to the data band (not 0-based) so few-point gaps stay visible — "
             "compare within a panel, never bar heights across panels.",
             "the last group in each row, past the divider, is the OVERALL/aggregate for that "
             "row. n/a = not run or not comparable — never drawn as a zero."] + list(extra_notes)
    # reserve a fixed PIXEL strip for the footer: these figures range from 13 to 22 inches tall,
    # so a fractional reserve is a hairline on the tall ones and a canyon on the short ones
    h = fig.get_size_inches()[1]
    strip = (0.20 + 0.145 * len(notes) + 0.30) / h  # +0.30in: the legend gets its own line
    fig.get_layout_engine().set(rect=(0, strip, 1, 1))

    seen, handles, names = set(), [], []
    for ax in fig.axes:
        for hnd, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(hnd)
                names.append(l)
    order = [names.index(m) for m, _, _ in MODELS if m in names]
    if order:
        # top-right, above the plots -- at the bottom it collided with the last row's rotated
        # x tick labels on the tall figures
        fig.legend([handles[i] for i in order], [names[i] for i in order], loc="lower right",
                   ncol=len(order), frameon=False, fontsize=9, labelcolor=INK_2,
                   bbox_to_anchor=(0.995, 0.02 / h))
    for i, t in enumerate(notes):
        fig.text(0.006, (0.36 + 0.145 * (len(notes) - 1 - i)) / h, t, fontsize=7.5, color=INK_2)
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"[chart] -> {out}")


def fig_headline(results, out):
    import matplotlib.pyplot as plt
    import numpy as np

    ncol, n = 4, len(MODELS)
    nrow = (len(HEADLINE) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.2 * nrow), facecolor=SURFACE,
                             squeeze=False, constrained_layout=True)
    for pi, (title, key, nd) in enumerate(HEADLINE):
        ax = axes[pi // ncol][pi % ncol]
        vals, errs = [], []
        for label, _, _ in MODELS:
            v, e = headline_value(results.get(label), key)
            vals.append(v)
            errs.append(e)
        x = np.arange(n)
        for i, v in enumerate(vals):
            if v is None:
                ax.text(x[i], 0.02, "n/a", ha="center", va="bottom", fontsize=7, color=INK_2,
                        rotation=90, zorder=6, transform=ax.get_xaxis_transform())
                continue
            ax.bar(x[i], v, width=0.62, color=MODELS[i][2], linewidth=0, zorder=3,
                   label=MODELS[i][0])
            if errs[i]:
                ax.errorbar(x[i], v, yerr=errs[i], fmt="none", ecolor=INK_2, elinewidth=1.1,
                            capsize=3, zorder=5)
            ax.text(x[i], v + (errs[i] or 0), f"{v:.{nd}f}", ha="center", va="bottom",
                    fontsize=8, color=INK, zorder=6)
        card = CARD.get(title)
        if card is not None:
            ax.axhline(card, color=INK, linestyle="--", linewidth=1.3, alpha=0.8, zorder=7)
        ax.set_title(title + (f"   ·   card {card} (dashed)" if card is not None else ""),
                     color=INK, fontsize=10, loc="left", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([m for m, _, _ in MODELS], rotation=30, ha="right", fontsize=8,
                           color=INK_2)
        style(ax)
        zoom(ax, vals, card, headroom=0.16)
    for pi in range(len(HEADLINE), nrow * ncol):
        axes[pi // ncol][pi % ncol].axis("off")
    finish(fig, out, ["error bars = stderr over passes (AIME avg@16, GPQA avg@4); every other "
                      "bench is single-pass and has none. A pass-averaged result with fewer "
                      "passes than it needs is withheld as n/a, not charted as finished."])


def fig_ruler(results, out):
    import matplotlib.pyplot as plt

    lens = ["4k", "8k", "16k", "32k"]
    fig, axes = plt.subplots(len(lens) + 1, 1, figsize=(13.5, 3.1 * (len(lens) + 1)),
                             facecolor=SURFACE, squeeze=False, constrained_layout=True)
    for li, ln in enumerate(lens):
        series = []
        for label, _, _ in MODELS:
            r = results.get(label) or {}
            cell = (r.get("ruler") or {}).get(ln)
            tasks = cell[1] if cell else {}
            vals = [tasks.get(t) for t in RULER_TASKS]
            got = [v for v in vals if v is not None]
            # the aggregate is the mean over ALL 13 tasks; if a model is missing some, its mean is
            # withheld rather than computed over a different task set than the other rows
            vals.append(st.mean(got) if len(got) == len(RULER_TASKS) else None)
            series.append(vals)
        grouped_row(axes[li][0], f"RULER {ln.upper()} — per task",
                    RULER_TASKS + ["OVERALL (mean of 13)"], series)
    series = []
    for label, _, _ in MODELS:
        r = results.get(label) or {}
        vals = [((r.get("ruler") or {}).get(l_) or [None])[0] for l_ in lens]
        got = [v for v in vals if v is not None]
        vals.append(st.mean(got) if len(got) == len(lens) else None)
        series.append(vals)
    grouped_row(axes[len(lens)][0], "RULER — per context length",
                [l_.upper() for l_ in lens] + ["OVERALL (mean of 4)"], series)
    finish(fig, out, ["a withheld OVERALL means that model is missing at least one task in the "
                      "row — averaging a different task set would not be comparable."])


def fig_mmlu(results, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1, figsize=(14.5, 3.6), facecolor=SURFACE, squeeze=False,
                             constrained_layout=True)
    series = []
    for label, _, _ in MODELS:
        r = results.get(label) or {}
        cats = r.get("mmlu_cats") or {}
        vals = [cats.get(c) for c in MMLU_CATS]
        v, _ = value_err(r.get("mmlu_pro"))
        vals.append(v)
        series.append(vals)
    grouped_row(axes[0][0], "MMLU-Pro — per category",
                MMLU_CATS + ["OVERALL (macro of 14)"], series, card=CARD["MMLU-Pro"])
    finish(fig, out)


def fig_mrcr(results, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(MRCR_NEEDLES) + 1, 1, figsize=(9.0, 3.0 * (len(MRCR_NEEDLES) + 1)),
                             facecolor=SURFACE, squeeze=False, constrained_layout=True)
    for ni, needle in enumerate(MRCR_NEEDLES):
        series = []
        for label, _, _ in MODELS:
            cells = (results.get(label) or {}).get("mrcr_cells") or {}
            vals = [(cells.get(f"{needle}_{b}") or {}).get("mean") for b in MRCR_BANDS]
            got = [v for v in vals if v is not None]
            vals.append(st.mean(got) if len(got) == len(MRCR_BANDS) else None)
            series.append(vals)
        grouped_row(axes[ni][0], f"MRCR {needle} — per context band",
                    MRCR_BANDS + [f"OVERALL ({needle})"], series, nd=3)
    series = []
    for label, _, _ in MODELS:
        r = results.get(label) or {}
        cells = r.get("mrcr_cells") or {}
        vals = []
        for needle in MRCR_NEEDLES:
            got = [(cells.get(f"{needle}_{b}") or {}).get("mean") for b in MRCR_BANDS]
            got = [g for g in got if g is not None]
            vals.append(st.mean(got) if len(got) == len(MRCR_BANDS) else None)
        v = r.get("mrcr")
        vals.append(v if isinstance(v, float) else None)
        series.append(vals)
    grouped_row(axes[len(MRCR_NEEDLES)][0], "MRCR — per needle count",
                MRCR_NEEDLES + ["OVERALL (mean of 9)"], series, nd=3)
    finish(fig, out, ["MRCR is a mean SequenceMatcher ratio on 0-1, not a percentage; n=25 per "
                      "cell, so a single cell carries roughly +/-0.1."])


def fig_gsm(results, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(GSM_LENS) + 1, 1, figsize=(10.0, 3.0 * (len(GSM_LENS) + 1)),
                             facecolor=SURFACE, squeeze=False, constrained_layout=True)
    for li, ln in enumerate(GSM_LENS):
        series = []
        for label, _, _ in MODELS:
            cells = (results.get(label) or {}).get("gsm_cells") or {}
            vals = [cells.get(f"{ln}_{o}") for o in GSM_OPS]
            got = [v for v in vals if v is not None]
            vals.append(st.mean(got) if len(got) == len(GSM_OPS) else None)
            series.append(vals)
        grouped_row(axes[li][0], f"GSM-Infinite {ln} — per reasoning-op count",
                    GSM_OPS + [f"OVERALL ({ln})"], series)
    series = []
    for label, _, _ in MODELS:
        r = results.get(label) or {}
        cells = r.get("gsm_cells") or {}
        vals = []
        for ln in GSM_LENS:
            got = [cells.get(f"{ln}_{o}") for o in GSM_OPS]
            got = [g for g in got if g is not None]
            vals.append(st.mean(got) if len(got) == len(GSM_OPS) else None)
        v = r.get("gsm")
        vals.append(v if isinstance(v, float) else None)
        series.append(vals)
    grouped_row(axes[len(GSM_LENS)][0], "GSM-Infinite — per context length",
                GSM_LENS + ["OVERALL (mean of 24)"], series)
    finish(fig, out, ["n=20 per cell. The 16K-32K x high-ops cells are near the floor on the "
                      "DENSE row too, so a low sparse bar there proves nothing."])


def fig_lcb(results, roots, cs, out):
    """LiveCodeBench by problem difficulty. The 131 problems are 31 easy / 39 medium / 61 hard, so
    a flat pass@1 can move purely by which tier the model lost -- the tiers are the diagnosis."""
    import json
    import matplotlib.pyplot as plt

    tiers = ["easy", "medium", "hard"]
    variants = [("livecodebench_v6", "@81920"), ("livecodebench_v6_cap32k", "@32768")]
    fig, axes = plt.subplots(len(variants), 1, figsize=(9.0, 3.2 * len(variants)),
                             facecolor=SURFACE, squeeze=False, constrained_layout=True)
    for vi, (bench, tag) in enumerate(variants):
        series = []
        for label, root, _ in MODELS:
            fs = glob_eval_all(root, bench)
            if not fs:
                series.append([None] * (len(tiers) + 1))
                continue
            d = json.load(open(fs))
            by = {t: [x["pass@1"] for x in d if x.get("difficulty") == t] for t in tiers}
            vals = [100 * st.mean(by[t]) if by[t] else None for t in tiers]
            vals.append(100 * st.mean(x["pass@1"] for x in d) if d else None)
            series.append(vals)
        grouped_row(axes[vi][0], f"LiveCodeBench v6 {tag} — by problem difficulty",
                    [f"{t} (n={c})" for t, c in zip(tiers, (31, 39, 61))]
                    + ["OVERALL (131 problems)"], series,
                    card=CARD.get(f"LiveCodeBench v6 {tag}"))
    finish(fig, out, ["OVERALL is the mean over all 131 problems, so it is weighted toward hard "
                      "(61 of 131) — it is not the mean of the three tier bars."])


def glob_eval_all(root, bench):
    import glob as _g
    fs = _g.glob(os.path.join(root, bench, "results", "*_eval_all.json"))
    return sorted(fs)[-1] if fs else None


def per_pass(cs, root, bench, dataset_sub, metric):
    """-> {pass_index: value} for a bench whose headline is an average over passes."""
    out = {}
    for key, rows in cs.read_csvs(os.path.join(root, bench)):
        m = re.match(r"(\d+)", str(key))
        if not m:
            continue
        for ds, mets in rows.items():
            if dataset_sub in ds and metric in mets:
                out[int(m.group(1))] = mets[metric]
    return out


AIME_ROWS = [("aime25", "aime2025", "accuracy", 16, "AIME25 @81920", "AIME25 @81920"),
             ("aime25_cap32k", "aime2025", "accuracy", 16, "AIME25 @32768", None)]
GPQA_ROWS = [("gpqa", "GPQA", "accuracy", 4, "GPQA-diamond", "GPQA-diamond")]


def fig_passes(results, cs, out, rows):
    """Pass-by-pass scores for a bench whose headline is an average over passes.

    Split one file per benchmark rather than one shared AIME+GPQA figure, so each benchmark
    section of the write-up can put its length figure and its score figure side by side.

    The headline is a mean over passes, and with 30 AIME items one pass is +/-9 pts -- seeing the
    spread is the only way to know whether a 4-pt gap between two models is signal.
    """
    import matplotlib.pyplot as plt

    rows = [(b, d, m, n, t, CARD.get(c) if c else None) for b, d, m, n, t, c in rows]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12.0, 3.2 * len(rows)), facecolor=SURFACE,
                             squeeze=False, constrained_layout=True)
    for ri, (bench, ds, metric, npass, title, card) in enumerate(rows):
        series = []
        for label, root, _ in MODELS:
            pp = per_pass(cs, root, bench, ds, metric)
            vals = [pp.get(i) for i in range(npass)]
            got = [v for v in vals if v is not None]
            # withhold the mean unless every pass is in -- a mean over 4 of 16 passes is not the
            # same estimator as a mean over 16 and must not sit in the same row
            vals.append(st.mean(got) if len(got) == npass else None)
            series.append(vals)
        grouped_row(axes[ri][0], f"{title} — per pass",
                    [f"p{i}" for i in range(npass)] + [f"OVERALL (avg@{npass})"], series,
                    card=card)
    finish(fig, out, ["a withheld OVERALL means that model has not finished all its passes; the "
                      "individual passes it HAS finished are still shown."])


def fig_ifeval(results, cs, out):
    import matplotlib.pyplot as plt

    metrics = [("Prompt-level-strict-accuracy", "prompt-strict"),
               ("Inst-level-strict-accuracy", "inst-strict"),
               ("Prompt-level-loose-accuracy", "prompt-loose"),
               ("Inst-level-loose-accuracy", "inst-loose")]
    fig, axes = plt.subplots(1, 1, figsize=(9.0, 3.4), facecolor=SURFACE, squeeze=False,
                            constrained_layout=True)
    series = []
    for label, root, _ in MODELS:
        vals = []
        for key, _ in metrics:
            v, _e = value_err(cs.scalar(root, "ifeval", "IFEval", key))
            vals.append(v)
        vals.append(vals[0])  # the headline IS prompt-strict, restated past the divider
        series.append(vals)
    grouped_row(axes[0][0], "IFEval — per metric",
                [m for _, m in metrics] + ["HEADLINE (prompt-strict)"], series,
                card=CARD["IFEval"])
    finish(fig, out, ["IFEval's four metrics are different scorings of the same run, not a "
                      "partition, so the last group RESTATES prompt-strict (the reported "
                      "headline) rather than aggregating the four."])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/qwen3_4b_msa")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    cs = load_collector()
    results = {}
    for label, root, _ in MODELS:
        if not os.path.isdir(root):
            print(f"[warn] missing root {root}", file=sys.stderr)
            continue
        results[label] = cs.collect(root)
        print(f"[collect] {label}")

    os.makedirs(a.outdir, exist_ok=True)
    roots = {label: root for label, root, _ in MODELS}
    fig_headline(results, os.path.join(a.outdir, "scores.png"))
    fig_ruler(results, os.path.join(a.outdir, "scores_ruler.png"))
    fig_mmlu(results, os.path.join(a.outdir, "scores_mmlu_pro.png"))
    fig_mrcr(results, os.path.join(a.outdir, "scores_mrcr.png"))
    fig_gsm(results, os.path.join(a.outdir, "scores_gsm_infinite.png"))
    fig_lcb(results, roots, cs, os.path.join(a.outdir, "scores_livecodebench.png"))
    fig_passes(results, cs, os.path.join(a.outdir, "scores_aime.png"), AIME_ROWS)
    fig_passes(results, cs, os.path.join(a.outdir, "scores_gpqa.png"), GPQA_ROWS)
    fig_ifeval(results, cs, os.path.join(a.outdir, "scores_ifeval.png"))


if __name__ == "__main__":
    main()
