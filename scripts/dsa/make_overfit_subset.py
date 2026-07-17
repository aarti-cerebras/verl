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
"""DSA Phase-2 — build a TINY fixed OVERFIT subset parquet from the self-gen trajectories JSONL.

Unlike ``trajectories_to_sft_parquet.py`` (full-corpus filter), this selects a small, deterministic set of
N trajectories for the overfit sanity run (see docs/dsa_phase2_implementation.md): filter by domain /
finish_reason / a [min,max] token window, then take the top-N by ``total_tokens`` (or the first N). The
window keeps every sample above ``top_k`` (so the sparse path is exercised) and below ``max_length`` (so
nothing truncates). Emits the same ``messages`` schema ``MultiTurnSFTDataset`` consumes.

Reproducible + logged: like the other DSA scripts it records the exact argv/cwd/host/git/env via _dsa_log,
so the parquet's provenance is captured in ``<log-dir>/make_overfit_subset_<ts>.log``.
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
    ap = argparse.ArgumentParser(description="tiny fixed overfit subset -> SFT messages parquet.")
    ap.add_argument("--input", nargs="+", required=True, help="trajectories jsonl file(s) or glob(s)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--n", type=int, default=24, help="number of trajectories to keep")
    ap.add_argument("--domains", nargs="+", default=["Code"], help="keep only these domains")
    ap.add_argument("--finish-reason", default="stop", help="keep only this finish_reason (\"\" = any)")
    ap.add_argument("--min-total", type=int, default=1024, help="drop total_tokens < this (>= top_k => sparse)")
    ap.add_argument("--max-total", type=int, default=4096, help="drop total_tokens > this (<= max_length => no trunc)")
    ap.add_argument("--sort", choices=["desc", "asc", "none"], default="desc",
                    help="order before taking the first --n (desc = longest first)")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger, _ = setup_logging("make_overfit_subset", args.log_dir or os.path.join(out_dir, "logs"))
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
                if args.domains and r.get("domain") not in args.domains:
                    continue
                if args.finish_reason and r.get("finish_reason") != args.finish_reason:
                    continue
                if not (args.min_total <= r.get("total_tokens", 0) <= args.max_total):
                    continue
                rows.append({k: r[k] for k in KEEP if k in r})
    logger.info("read %d trajectories; %d pass filters", n_read, len(rows))
    assert len(rows) >= args.n, f"only {len(rows)} pass filters, need >= {args.n}"

    df = pd.DataFrame(rows)
    if args.sort != "none":
        df = df.sort_values("total_tokens", ascending=(args.sort == "asc"))
    df = df.head(args.n)[[c for c in KEEP if c in df.columns]].reset_index(drop=True)

    logger.info("kept %d rows | total_tokens min/max = %d/%d | by domain: %s | by lang: %s",
                len(df), int(df.total_tokens.min()), int(df.total_tokens.max()),
                df.domain.value_counts().to_dict(), df.lang.value_counts().to_dict())
    df.to_parquet(args.out, index=False)
    logger.info("wrote %d rows -> %s", len(df), args.out)


if __name__ == "__main__":
    main()
