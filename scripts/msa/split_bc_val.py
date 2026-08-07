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
"""Carve a held-out validation split out of the Phase-2 behaviour-cloning parquet.

The generation pipeline emits a single `bc_2b.parquet` with no val split, so a Phase-2 run has no
regression signal. This writes a disjoint train/val pair to a NEW directory; the source parquet is read
only and never modified.

Three properties that matter:

* **Stratified by domain.** The source is written in domain order (the first thousands of rows are all
  Math), so a contiguous or naively-sampled slice would be single-domain and useless as a val set. Rows
  are sampled per domain in proportion to that domain's share.
* **Disjoint by `prompt_sha256`, not just by row.** The generator can emit several samples per prompt
  (`sample_idx`), so holding out a ROW could leave a sibling sample of the SAME prompt in train --
  leakage that would make val optimistic. Whole prompts move to val.
* **Deterministic.** Fixed seed, and the outputs are frozen files. Row order within the training file is
  preserved from the source, so the training row order remains a pure function of the artifact (which is
  what makes resume safe -- verl stores dataloader state as a bare batch counter).

**Sharing one split across generator models (`--val-sha-file`).** Re-running the sampler on a different
model's parquet does NOT reproduce the same split: the seed is fixed, but the per-domain row counts and
the surviving `prompt_sha256` universe both differ (each generator loses a different ~4 % of prompts to
the health filters), so `want` and `np.unique(...)` differ and a *different* prompt set is drawn. Since
`prompt_sha256` is sha256 of the cleaned prompt text and therefore generator-independent, the fix is to
freeze the val prompt set once and select by it thereafter:

    --val-sha-file /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__ph2b_split_v1/val_prompt_shas.txt

Rows are then routed by membership, not sampled. The val set may come out slightly smaller than the
frozen list (a held-out prompt whose trace this generator filtered out has no row to contribute); the
count is logged and recorded in the manifest as `val_shas_missing`. What transfers is the prompt-level
boundary -- the rows cannot match, since each model's BC val has to be that model's own traces.

Example:
    # original (sampling) -- Qwen3
    python3 scripts/msa/split_bc_val.py \
      --src /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__dolci-think-rl-32b__ph2b_full93889_L32768_20260730_231930/bc_2b.parquet \
      --out-dir /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__ph2b_split_v1 --val-rows 512

    # reuse that exact prompt-level split -- gpt-oss-20b
    python3 scripts/msa/split_bc_val.py \
      --src <gpt-oss run>/bc_2b.parquet --out-dir <gpt-oss run>__ph2b_split_v1 \
      --val-sha-file /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__ph2b_split_v1/val_prompt_shas.txt
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsa"))
from _dsa_log import setup_logging  # noqa: E402


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-rows", type=int, default=512)
    ap.add_argument("--val-sha-file", default=None,
                    help="file of prompt_sha256 (one per line) defining the val set. Selects by membership "
                         "instead of sampling, so a different generator model reproduces the SAME "
                         "prompt-level split. Makes --val-rows/--seed unused.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rows-per-shard", type=int, default=8192)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    log, log_path = setup_logging("split_bc_val", a.out_dir)
    log.info("args: %s", vars(a))
    t0 = time.time()

    meta = pq.read_table(a.src, columns=["domain", "prompt_sha256", "length"]).to_pydict()
    dom = np.asarray(meta["domain"])
    sha = np.asarray(meta["prompt_sha256"])
    lens = np.asarray(meta["length"], dtype=np.int64)
    n = len(dom)
    log.info("source: %d rows, %.1fM tokens, domains=%s", n, lens.sum() / 1e6, sorted(set(dom.tolist())))

    frozen = None
    if a.val_sha_file:
        # Reuse a previously frozen prompt-level split so two generator models are directly comparable.
        raw = open(a.val_sha_file, "rb").read()
        frozen = {"path": os.path.abspath(a.val_sha_file), "sha256": hashlib.sha256(raw).hexdigest()}
        val_shas = {ln for ln in raw.decode().split() if ln}
        assert val_shas, f"--val-sha-file {a.val_sha_file} is empty"
        log.info("frozen val split: %d prompt_sha256 from %s (file sha256=%s)",
                 len(val_shas), a.val_sha_file, frozen["sha256"][:16])
        present = val_shas & set(sha.tolist())
        frozen["n_shas"] = len(val_shas)
        frozen["n_missing"] = len(val_shas) - len(present)
        # A held-out prompt whose trace THIS generator filtered out simply has no row to contribute. Never
        # silently backfill from train -- that would break comparability, which is the point of the flag.
        log.info("  %d/%d frozen prompts present in this parquet (%d missing -> smaller val, not backfilled)",
                 len(present), len(val_shas), frozen["n_missing"])
        assert present, "no frozen val prompt_sha256 found in --src; wrong split file for this parquet?"
        for d in sorted(set(dom.tolist())):
            idx = np.where(dom == d)[0]
            log.info("  %-8s %6d rows -> %d held-out prompt(s)", d, len(idx),
                     len(present & set(np.unique(sha[idx]).tolist())))
    else:
        # Pick whole PROMPTS per domain, proportional to the domain's share of rows.
        rng = np.random.default_rng(a.seed)
        val_shas = set()
        for d in sorted(set(dom.tolist())):
            idx = np.where(dom == d)[0]
            want = max(1, round(a.val_rows * len(idx) / n))
            uniq = np.unique(sha[idx])
            take = rng.choice(uniq, size=min(want, len(uniq)), replace=False)
            val_shas.update(take.tolist())
            log.info("  %-8s %6d rows -> holding out %d prompt(s)", d, len(idx), len(take))

    is_val = np.fromiter((s in val_shas for s in sha), dtype=bool, count=n)
    log.info("val: %d rows / %.2fM tokens | train: %d rows / %.1fM tokens",
             is_val.sum(), lens[is_val].sum() / 1e6, (~is_val).sum(), lens[~is_val].sum() / 1e6)
    assert is_val.any() and (~is_val).any(), "degenerate split"
    # The leakage check the whole exercise exists for.
    assert not (set(sha[is_val].tolist()) & set(sha[~is_val].tolist())), "prompt_sha256 overlap across splits"

    # Stream the source and route each row to its split, preserving source order within each.
    writers, counts, files = {}, {"train": 0, "val": 0}, {"train": [], "val": []}
    schema = pq.ParquetFile(a.src).schema_arrow
    buf = {"train": [], "val": []}

    def flush(split, force=False):
        if not buf[split] or (len(buf[split]) < a.rows_per_shard and not force):
            return
        path = os.path.join(a.out_dir, f"{split}-{len(files[split]):05d}.parquet")
        pq.write_table(pa.Table.from_batches(buf[split], schema=schema), path, compression="zstd")
        files[split].append(os.path.basename(path))
        log.info("  wrote %s (%d rows)", os.path.basename(path), sum(b.num_rows for b in buf[split]))
        buf[split] = []

    off = 0
    for batch in pq.ParquetFile(a.src).iter_batches(batch_size=2048):
        m = is_val[off: off + batch.num_rows]
        for split, sel in (("val", m), ("train", ~m)):
            if sel.any():
                sub = batch.filter(pa.array(sel))
                buf[split].append(sub)
                counts[split] += sub.num_rows
                if sum(b.num_rows for b in buf[split]) >= a.rows_per_shard:
                    flush(split)
        off += batch.num_rows
    for s in ("train", "val"):
        flush(s, force=True)
    assert counts["train"] + counts["val"] == n, f"row count mismatch: {counts} vs {n}"

    manifest = dict(
        kind="msa_phase2b_bc_split", source=a.src, val_rows=int(counts["val"]), train_rows=int(counts["train"]),
        val_tokens=int(lens[is_val].sum()), train_tokens=int(lens[~is_val].sum()),
        stratified_by=("frozen-sha-file" if frozen else "domain"), disjoint_by="prompt_sha256",
        seed=(None if frozen else a.seed), frozen_val_split=frozen,
        val_shas_missing=(frozen["n_missing"] if frozen else 0),
        source_row_order_preserved=True,
        train_files=files["train"], val_files=files["val"],
        steps_at_bsz8=counts["train"] // 8,
        git_commit=_git("rev-parse", "HEAD"), invocation=" ".join([sys.executable] + sys.argv),
        host=os.uname().nodename, elapsed_s=round(time.time() - t0, 1), log_file=os.path.basename(log_path),
    )
    with open(os.path.join(a.out_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log.info("DONE in %.1f min: train=%d val=%d -> %s", (time.time() - t0) / 60,
             counts["train"], counts["val"], a.out_dir)
    print(f"\nTRAIN_FILES={a.out_dir}\nVAL_FILES={a.out_dir}\nSTEPS={counts['train'] // 8}")


if __name__ == "__main__":
    sys.exit(main())
