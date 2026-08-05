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
from _dsa_tok import chat_prefix_ids  # noqa: E402

KEEP = ["messages", "domain", "lang", "source_uid", "source_config", "prompt_sha256",
        "prompt_tokens", "resp_tokens", "total_tokens", "finish_reason"]
KEEP_IDS = ["input_ids", "loss_mask", "length", "bucket", "domain", "lang", "source_uid", "source_config",
            "original_dataset", "prompt_sha256", "prefix_tokens", "resp_tokens", "finish_reason", "sample_idx"]

# Qwen3-Thinking marker token ids (verified against the checkpoint; see phase2_data_gen.md §5.1/§7)
TOK_IM_START, TOK_IM_END, TOK_THINK_OPEN, TOK_THINK_CLOSE = 151644, 151645, 151667, 151668


def _bucket(total):
    if total < 4096:
        return "short"
    if total < 8192:
        return "mid"
    return "decode_long"


def _repetition_frac(ids, ngram=32, min_repeats=8):
    """Fraction of the sequence covered by an n-gram that repeats >= min_repeats times (loop detector)."""
    if len(ids) < ngram * min_repeats:
        return 0.0
    counts = {}
    for i in range(len(ids) - ngram + 1):
        key = tuple(ids[i : i + ngram])
        counts[key] = counts.get(key, 0) + 1
    worst = max(counts.values())
    if worst < min_repeats:
        return 0.0
    return min(1.0, worst * ngram / len(ids))


def _malformed(resp_ids, args):
    """§7 structural health checks, on token IDs. Returns a class name, or None if the row is clean."""
    if len(resp_ids) < 8:
        return "stub"
    if TOK_IM_START in resp_ids:
        return "role_marker_leak"
    if TOK_IM_END in resp_ids[:-1]:
        return "role_marker_leak"
    if TOK_THINK_OPEN in resp_ids:
        return "reopened_think"
    n_close = resp_ids.count(TOK_THINK_CLOSE)
    if n_close == 0:
        return "no_think_close"
    if n_close > 1:
        return "multiple_think_close"
    tail = resp_ids[resp_ids.index(TOK_THINK_CLOSE) + 1 :]
    if len([t for t in tail if t not in (198, 271, TOK_IM_END)]) == 0:
        return "empty_answer"
    if _repetition_frac(resp_ids) > args.max_repetition_frac:
        return "repetition_loop"
    return None


def _prefix_ids(tok, messages):
    return chat_prefix_ids(tok, [m for m in messages if m.get("role") != "assistant"])


def to_input_ids_rows(rows, args, logger):
    """Splice served prefix + sampled completion tokens -> (input_ids, loss_mask). phase2_data_gen.md §6."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    out, counters = [], {}

    def bump(k):
        counters[k] = counters.get(k, 0) + 1

    for r in rows:
        resp_ids = r.get("resp_token_ids")
        if resp_ids is None:
            bump("missing_resp_token_ids")
            continue
        resp_ids = [int(t) for t in resp_ids]
        if r.get("finish_reason") == "length":
            if not args.keep_truncated:
                bump("truncated")
                continue
            bump("truncated_kept")
        else:
            cls = _malformed(resp_ids, args)
            if cls:
                bump(cls)
                continue
        # Prefer the prefix the ENGINE conditioned on (recorded by gen_trajectories.py); rebuilding from
        # `messages` is the fallback + cross-check. Using the served ids makes the mask exact regardless of
        # tokenizer/template version drift between generation and conversion.
        served = r.get("prefix_token_ids")
        rebuilt = _prefix_ids(tok, r["messages"])
        if served:
            pre = [int(t) for t in served]
            if list(pre) != list(rebuilt):
                bump("prefix_rebuild_differs")  # not fatal: the served prefix is authoritative
        else:
            pre = rebuilt
            rec = r.get("prefix_tokens")
            if rec is not None and len(pre) != int(rec):
                # no served ids AND the rebuild disagrees -> the mask would be wrong. Never silent.
                bump("prefix_len_mismatch")
                logger.error("prefix mismatch for %s: rebuilt=%d recorded=%d",
                             r.get("prompt_sha256"), len(pre), rec)
                continue
        ids = pre + resp_ids
        if ids[-1] != TOK_IM_END:
            ids = ids + [TOK_IM_END]
        if len(ids) > args.max_length:
            bump("over_window")
            logger.error("row exceeds --max-length %d (len=%d) — check --fit-window at generation time",
                         args.max_length, len(ids))
            continue
        mask = [0] * len(pre) + [1] * (len(ids) - len(pre))
        out.append({
            "input_ids": ids, "loss_mask": mask, "length": len(ids), "bucket": _bucket(len(ids)),
            "domain": r.get("domain"), "lang": r.get("lang"), "source_uid": r.get("source_uid"),
            "source_config": r.get("source_config"), "original_dataset": r.get("original_dataset"),
            "prompt_sha256": r.get("prompt_sha256"), "prefix_tokens": len(pre),
            "resp_tokens": len(resp_ids), "finish_reason": r.get("finish_reason"),
            "sample_idx": r.get("sample_idx", 0),
        })
        bump("kept")
    logger.info("splice counters: %s", dict(sorted(counters.items())))
    n_in = max(len(rows), 1)
    for k, v in sorted(counters.items()):
        logger.info("  %-24s %6d (%5.2f%% of input)", k, v, 100.0 * v / n_in)
    return out, counters


def main():
    ap = argparse.ArgumentParser(description="trajectories JSONL -> SFT messages parquet (filtered).")
    ap.add_argument("--input", nargs="+", required=True, help="jsonl file(s) or glob(s) (merged or .partN)")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--min-total", type=int, default=512, help="drop trajectories with total_tokens < this")
    ap.add_argument("--keep-truncated", action="store_true", help="keep finish_reason==length (default: drop)")
    ap.add_argument("--domains", nargs="+", default=None, help="keep only these domains (e.g. Math)")
    ap.add_argument("--emit-input-ids", action="store_true",
                    help="emit pre-tokenized input_ids + loss_mask by splicing the served prefix with the "
                         "sampled completion tokens, and apply the §7 malformed-response filters. Required "
                         "for Qwen3-Thinking: per-message chat templating DELETES <think> traces "
                         "(docs/qwen3_4b_msa/phase2_data_gen.md §6)")
    ap.add_argument("--tokenizer", default=None, help="model dir (required with --emit-input-ids)")
    ap.add_argument("--max-length", type=int, default=32768, help="training window; rows longer than this fail loudly")
    ap.add_argument("--max-repetition-frac", type=float, default=0.30,
                    help="drop if a 32-token n-gram repeating >=8x covers more than this fraction of the trace")
    args = ap.parse_args()
    assert not (args.emit_input_ids and not args.tokenizer), "--emit-input-ids requires --tokenizer"

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    logger, _ = setup_logging("trajectories_to_sft_parquet", args.log_dir or os.path.join(out_dir, "logs"))
    logger.info("config: %s", vars(args))

    files = []
    for pat in args.input:
        files.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    assert files, f"no input files matched {args.input}"
    logger.info("reading %d file(s): %s", len(files), [os.path.basename(f) for f in files])

    rows, n_read, n_bad = [], 0, 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n_read += 1
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A kill/pause mid-write can truncate the last line of a .partN file. The generator's
                    # resume path tolerates that (it re-generates the prompt), so the converter must too --
                    # crashing here would make a paused run unreadable until hand-edited.
                    n_bad += 1
                    logger.warning("skipping unparseable line %d in %s (truncated write?)", n_read, f)
    logger.info("read %d trajectories (%d unparseable lines skipped)", n_read - n_bad, n_bad)

    if args.domains:
        rows = [r for r in rows if r.get("domain") in args.domains]
        logger.info("domain filter %s -> %d rows", args.domains, len(rows))

    if args.emit_input_ids:
        spliced, counters = to_input_ids_rows(rows, args, logger)
        assert spliced, "no rows survived the §7 filters"
        df = pd.DataFrame(spliced)[[c for c in KEEP_IDS if c in spliced[0]]]
        logger.info("kept %d / %d after splice + health filters", len(df), len(rows))
        logger.info("by domain: %s", df["domain"].value_counts().to_dict())
        logger.info("by bucket: %s", df["bucket"].value_counts().to_dict())
        q = df["length"].quantile([0.5, 0.9, 0.99]).astype(int).to_dict()
        logger.info("length p50/p90/p99 = %s   max=%d", q, int(df["length"].max()))
        _log_seqlen_histogram(logger, df["length"], top_k=2048)
        df.to_parquet(args.out, index=False)
        logger.info("wrote %d rows -> %s", len(df), args.out)
        with open(args.out + ".COUNTERS.json", "w") as f:
            json.dump(counters, f, indent=2)
        logger.info("DONE")
        return

    df = pd.DataFrame(rows)
    n0 = len(df)
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
