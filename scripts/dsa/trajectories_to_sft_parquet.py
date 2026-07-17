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
"""DSA Phase-2 — convert self-gen trajectories (JSONL from gen_trajectories.py) into the SFT `messages`
parquet that `MultiTurnSFTDataset` (the default SFT dataset) consumes, applying the DSA length + health
filters (see docs/dsa_phase2_impl.md T5, docs/dsa_phase2_plan.md).

Filters:
  * total_tokens >= --min-total  (default 512 = top_k; below this a doc runs dense -> no DSA signal)
  * drop finish_reason == "length"  (the runaways that rode the cap; not behavior worth cloning)
  * optional --domains subset (e.g. Math for the math-only validation)
Keeps ``messages`` (the SFT target) + provenance/metadata columns. Accepts one or more inputs (merged
``trajectories.jsonl`` or the un-merged ``*.partN`` files via a glob).
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsa_log import setup_logging  # noqa: E402

KEEP = ["messages", "domain", "lang", "source_uid", "source_config", "prompt_sha256",
        "prompt_tokens", "resp_tokens", "total_tokens", "finish_reason"]


def main():
    ap = argparse.ArgumentParser(description="trajectories JSONL -> SFT messages parquet (filtered).")
    ap.add_argument("--input", nargs="+", required=True, help="jsonl file(s) or glob(s) (merged or .partN)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--min-total", type=int, default=512, help="drop trajectories with total_tokens < this")
    ap.add_argument("--keep-truncated", action="store_true", help="keep finish_reason==length (default: drop)")
    ap.add_argument("--domains", nargs="+", default=None, help="keep only these domains (e.g. Math)")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger, _ = setup_logging("trajectories_to_sft_parquet", args.log_dir or os.path.join(out_dir, "logs"))
    logger.info("config: %s", vars(args))

    files = []
    for pat in args.input:
        files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    assert files, f"no input files matched {args.input}"
    logger.info("reading %d file(s): %s", len(files), [os.path.basename(f) for f in files])

    rows, n_read = [], 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_read += 1
                r = json.loads(line)
                rows.append(r)
    logger.info("read %d trajectories", n_read)

    df = pd.DataFrame(rows)
    n0 = len(df)
    if args.domains:
        df = df[df["domain"].isin(args.domains)]
    if not args.keep_truncated and "finish_reason" in df.columns:
        df = df[df["finish_reason"] != "length"]
    if "total_tokens" in df.columns:
        df = df[df["total_tokens"] >= args.min_total]
    df = df[[c for c in KEEP if c in df.columns]].reset_index(drop=True)

    logger.info("kept %d / %d after filters (min_total=%d, drop_truncated=%s, domains=%s)",
                len(df), n0, args.min_total, not args.keep_truncated, args.domains or "all")
    if "domain" in df.columns:
        logger.info("by domain: %s", df["domain"].value_counts().to_dict())
    if "lang" in df.columns:
        logger.info("by lang: %s", df["lang"].value_counts().to_dict())
    if "total_tokens" in df.columns and len(df):
        q = df["total_tokens"].quantile([0.5, 0.9, 0.99]).astype(int).to_dict()
        logger.info("total_tokens p50/p90/p99 = %s", q)
        _log_seqlen_histogram(logger, df["total_tokens"], top_k=512)

    df.to_parquet(args.out, index=False)
    logger.info("wrote %d rows -> %s", len(df), args.out)


def _log_seqlen_histogram(logger, series, top_k=512):
    """Log a text histogram of sequence lengths (total_tokens) + the fraction that will engage sparse
    attention (>= top_k; shorter sequences run dense since top_k covers the whole causal set)."""
    import numpy as np

    edges = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 1 << 30]
    vals = series.to_numpy()
    n = len(vals)
    counts, _ = np.histogram(vals, bins=edges)
    peak = max(int(counts.max()), 1)
    logger.info("sequence-length (total_tokens) histogram over %d samples:", n)
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        label = f">={lo}" if hi >= (1 << 30) else f"{lo}-{hi - 1}"
        bar = "#" * int(40 * c / peak)
        logger.info("  %-12s %8d (%5.1f%%) %s", label, int(c), 100.0 * c / n, bar)
    for thr in (top_k, 1024, 2048, 4096):
        logger.info("  >= %-6d : %5.1f%% (%d samples)", thr, 100.0 * (vals >= thr).mean(), int((vals >= thr).sum()))
    logger.info("  sparse-active fraction (>= top_k=%d): %.1f%%  | dense (< top_k): %.1f%%",
                top_k, 100.0 * (vals >= top_k).mean(), 100.0 * (vals < top_k).mean())


if __name__ == "__main__":
    main()
