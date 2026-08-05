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
"""Verify a spliced Qwen3-Thinking BC parquet (``--emit-input-ids`` output of trajectories_to_sft_parquet).

Checks the invariants that make the data trainable, on token IDs. Exits non-zero if ANY row fails, because
every one of these failures is silent at training time:

  1. ``loss_mask == 0`` over exactly the prefix, ``== 1`` over the whole target
  2. the prefix ends with ``<think>\\n`` (151667, 198) -- i.e. the served generation prompt
  3. exactly one ``</think>`` (151668) in the target, and none in the prefix
  4. no ``<think>`` re-opened inside the target
  5. the row ends with ``<|im_end|>`` (151645) and contains no ``<|im_start|>`` in the target
  6. a non-empty answer after ``</think>``
  7. ``len(input_ids) == len(loss_mask) == length <= --max-length``

The headline number it prints is the **trained-token fraction**: for a thinking model this should be very
high (~98%), because the reasoning trace is nearly all of the sequence. A low value means the trace was
dropped somewhere -- exactly the failure mode of per-message chat templating
(docs/qwen3_4b_msa/phase2_data_gen.md §6).
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsa_log import setup_logging  # noqa: E402

TOK_IM_START, TOK_IM_END, TOK_THINK_OPEN, TOK_THINK_CLOSE = 151644, 151645, 151667, 151668
WHITESPACE_IDS = {198, 271}  # "\n", "\n\n"


def check_row(ids, mask, npre, max_length):
    """Return a list of failed invariant names for one row."""
    bad = []
    if not (len(ids) == len(mask) <= max_length):
        bad.append("length")
    if sum(mask[:npre]) != 0:
        bad.append("mask_nonzero_over_prefix")
    if not all(x == 1 for x in mask[npre:]):
        bad.append("mask_not_one_over_target")
    if ids[npre - 2 : npre] != [TOK_THINK_OPEN, 198]:
        bad.append("prefix_not_ending_in_think_nl")
    if TOK_THINK_CLOSE in ids[:npre]:
        bad.append("think_close_in_prefix")
    tgt = ids[npre:]
    n_close = tgt.count(TOK_THINK_CLOSE)
    if n_close != 1:
        bad.append(f"think_close_count_{n_close}")
    if TOK_THINK_OPEN in tgt:
        bad.append("think_reopened")
    if ids[-1] != TOK_IM_END:
        bad.append("no_terminal_im_end")
    if TOK_IM_START in tgt:
        bad.append("im_start_in_target")
    if n_close == 1:
        answer = tgt[tgt.index(TOK_THINK_CLOSE) + 1 :]
        if not [t for t in answer if t not in WHITESPACE_IDS and t != TOK_IM_END]:
            bad.append("empty_answer")
    return bad


def main():
    ap = argparse.ArgumentParser(description="Verify a spliced BC parquet's mask/marker invariants.")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--max-report", type=int, default=20, help="how many failing rows to list")
    args = ap.parse_args()

    log_dir = args.log_dir or os.path.join(os.path.dirname(os.path.abspath(args.parquet)), "logs")
    logger, _ = setup_logging("verify_sft_parquet", log_dir)
    logger.info("config: %s", vars(args))

    df = pd.read_parquet(args.parquet)
    logger.info("loaded %d rows from %s", len(df), args.parquet)

    failures, per_check = [], {}
    trained = total = 0
    for i, r in enumerate(df.itertuples()):
        ids, mask, npre = list(r.input_ids), list(r.loss_mask), int(r.prefix_tokens)
        bad = check_row(ids, mask, npre, args.max_length)
        for b in bad:
            per_check[b] = per_check.get(b, 0) + 1
        if bad:
            failures.append((i, getattr(r, "prompt_sha256", None), bad))
        trained += len(ids) - npre
        total += len(ids)

    logger.info("trained tokens: %d / %d (%.1f%% — prompt masked)", trained, total, 100.0 * trained / max(total, 1))
    for col in ("domain", "bucket", "finish_reason"):
        if col in df.columns:
            logger.info("by %s: %s", col, df[col].value_counts().to_dict())
    if "length" in df.columns:
        logger.info("length p50/p90/p99 = %s  max=%d",
                    df["length"].quantile([0.5, 0.9, 0.99]).astype(int).to_dict(), int(df["length"].max()))

    if failures:
        logger.error("FAILED: %d / %d rows violate invariants: %s", len(failures), len(df), per_check)
        for i, sha, bad in failures[: args.max_report]:
            logger.error("  row %d (sha=%s): %s", i, sha, bad)
        sys.exit(1)
    logger.info("ALL %d ROWS PASS every invariant", len(df))
    logger.info("DONE")


if __name__ == "__main__":
    main()
