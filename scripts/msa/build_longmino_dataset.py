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
"""Build the MSA Phase-1 long-context dataset from OLMo 3's `dolma3_longmino_mix-100B-1125`.

Why this source rather than InfLLM-V2-data-5B (which the MiniCPM3-DSA runs used):

    corpus                                        yield of >=32768-token docs      total available
    openbmb/InfLLM-V2-data-5B                     0.295% (7,757 of 2.6M scanned)   ~250M tokens
    longmino `olmocr_science_pdfs-*-length_2e15`  100%   (60 of 60 sampled)         ~18.3B tokens

The `length_2eNN` suffix is a token-count FLOOR (2^13/2^14/2^15 = 8192/16384/32768) and it survives
re-tokenisation: measured on a `length_2e15` shard with the Qwen3 tokenizer, 60/60 docs were >= 32768
tokens (min 32,837, median 47,309, max 67,587; 3.48 chars/token). So no length filtering is needed and
the build is I/O bound rather than scan bound.

Design notes:
  * **Text, not token_ids.** We re-tokenize with the TARGET model's tokenizer. Never reuse another
    model's ids (the InfLLM corpus ships CPM-5 ids; longmino ships `metadata.len_cl100k_base`).
  * **One window per document**, truncated to exactly ``seq_len`` — the plan's one-doc-per-row rule, no
    cross-document packing. Documents average ~48.9k tokens, so the tail past 32,768 is dropped.
  * **Sharded, streaming output.** Each output parquet holds ``--windows-per-shard`` windows and is
    written as soon as it fills, so (a) peak RSS is one shard, not the corpus, and (b) the build is
    usable and resumable at any point — kill it and train on what exists.
  * **int32** ids (Qwen3 vocab is 151,936): halves both disk and the loader's RAM vs int64.
  * Val comes from input shards DISJOINT from train (held out by shard, not by row), so no document
    leaks across the split.

Load-time RAM caveat: ``PackedPretrainDataset`` reads every parquet into pandas and concatenates, PER
RANK. At int32 that is ~4 GB/rank per 1B tokens. Past ~1B, add a memmap path instead.

Example (1B tokens):
    python3 scripts/msa/build_longmino_dataset.py \
      --model /cb/ml-eng/aarti/models/qwen3_4b_thinking_2507 --seq-len 32768 \
      --target-tokens 1e9 --out-dir /cb/ml-eng/aarti/msa/data/longmino_qwen3_32768
"""

import argparse
import io
import json
import logging
import os
import random
import subprocess
import sys
import time

for _n in ("filelock", "httpx", "httpcore", "huggingface_hub", "fsspec", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)  # one 300-char signed URL per shard otherwise
log = logging.getLogger("longmino")

REPO = "allenai/dolma3_longmino_mix-100B-1125"
# The >=32768-token buckets, largest first. `denyagain` variants are the 2e13/2e14 (shorter) buckets and
# are deliberately excluded here; they are the right source for a Phase-2b LENGTH MIXTURE, not Phase 1.
SUBSETS_32K = [
    "olmocr_science_pdfs-high_quality-science_tech-length_2e15",
    "olmocr_science_pdfs-high_quality-education_jobs-length_2e15",
    "olmocr_science_pdfs-high_quality-health-length_2e15",
    "olmocr_science_pdfs-high_quality-finance_business-length_2e15",
    "olmocr_science_pdfs-high_quality-crime_law-length_2e15",
    "olmocr_science_pdfs-high_quality-politics-length_2e15",
]


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="tokenizer to build with (the TARGET model)")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--target-tokens", type=float, default=1e9, help="stop once this many TRAIN tokens are written")
    ap.add_argument("--val-tokens", type=float, default=16e6, help="held-out tokens (from disjoint input shards)")
    ap.add_argument("--windows-per-shard", type=int, default=2048, help="windows per output parquet (~268 MB at 32K int32)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS_32K)
    ap.add_argument("--revision", default=None, help="pin; default = resolve current HEAD of the repo")
    ap.add_argument("--seed", type=int, default=1234, help="input-shard shuffle seed (mixes domains)")
    ap.add_argument("--min-tokens", type=int, default=None, help="drop docs shorter than this (default: seq_len)")
    ap.add_argument("--batch-docs", type=int, default=32, help="docs per tokenizer batch call")
    a = ap.parse_args()
    a.min_tokens = a.min_tokens or a.seq_len

    os.makedirs(a.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(a.out_dir, "build.log"))])
    log.info("CMD: %s", " ".join([sys.executable] + sys.argv))
    log.info("CWD: %s", os.getcwd())
    log.info("ENV: %s", {k: os.environ.get(k) for k in ("HF_HOME", "HF_TOKEN", "HF_HUB_OFFLINE")})
    log.info("ARGS: %s", vars(a))

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard as zstd
    from huggingface_hub import HfApi, HfFileSystem
    from transformers import AutoTokenizer

    api = HfApi()
    rev = a.revision or api.dataset_info(REPO).sha
    log.info("repo=%s revision=%s", REPO, rev)
    fs = HfFileSystem()
    root = f"datasets/{REPO}@{rev}"

    # Enumerate input shards across all subsets, then shuffle so the output interleaves domains
    # (otherwise the first N output shards would be pure science_tech).
    shards = []
    for sub in a.subsets:
        got = sorted(fs.glob(f"{root}/data/{sub}/*.jsonl.zst"))
        log.info("  %-58s %5d shards", sub, len(got))
        shards.extend(got)
    if not shards:
        raise FileNotFoundError("no input shards matched")
    random.Random(a.seed).shuffle(shards)
    log.info("total input shards: %d (shuffled, seed=%d)", len(shards), a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    dec = zstd.ZstdDecompressor()
    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])

    state = dict(split="val", buf=[], out_idx=0, tok_written={"train": 0, "val": 0},
                 win_written={"train": 0, "val": 0}, shard_files=[], docs_seen=0, docs_short=0)

    def flush(force=False):
        """Write the buffer to a numbered parquet shard for the current split."""
        if not state["buf"] or (len(state["buf"]) < a.windows_per_shard and not force):
            return
        split = state["split"]
        path = os.path.join(a.out_dir, f"{split}-{state['out_idx']:05d}.parquet")
        arr = pa.array(np.asarray(state["buf"], dtype=np.int32).tolist(), type=pa.list_(pa.int32()))
        pq.write_table(pa.table({"input_ids": arr}, schema=schema), path, compression="zstd")
        state["win_written"][split] += len(state["buf"])
        state["tok_written"][split] += len(state["buf"]) * a.seq_len
        state["shard_files"].append(os.path.basename(path))
        log.info("wrote %s (%d windows) | %s total: %d windows / %.3fB tokens",
                 os.path.basename(path), len(state["buf"]), split,
                 state["win_written"][split], state["tok_written"][split] / 1e9)
        state["buf"] = []
        state["out_idx"] += 1

    t0 = time.time()
    done = False
    for si, path in enumerate(shards):
        if done:
            break
        try:
            with fs.open(path, "rb") as fh:
                rdr = dec.stream_reader(fh)
                batch = []
                for line in io.TextIOWrapper(rdr, encoding="utf-8"):
                    batch.append(json.loads(line).get("text", ""))
                    if len(batch) < a.batch_docs:
                        continue
                    _consume(batch, tok, a, state, flush)
                    batch = []
                    if state["split"] == "val" and state["tok_written"]["val"] >= a.val_tokens:
                        flush(force=True)          # switch to train on a shard boundary of the buffer
                        state["split"], state["out_idx"] = "train", 0
                        log.info("val target reached; switching to train")
                    if state["tok_written"]["train"] >= a.target_tokens:
                        done = True
                        break
                if batch and not done:
                    _consume(batch, tok, a, state, flush)
        except Exception as e:  # a bad shard must not kill a multi-hour build
            log.warning("shard %s failed (%s: %s) — skipping", path.split("/")[-1], type(e).__name__, e)
            continue
        if (si + 1) % 20 == 0:
            el = time.time() - t0
            tw = state["tok_written"]["train"]
            log.info("input shard %d/%d | %.3fB train tokens | %.1f min | %.1fM tok/min",
                     si + 1, len(shards), tw / 1e9, el / 60, tw / 1e6 / max(el / 60, 1e-9))
    flush(force=True)

    manifest = dict(
        kind="longmino_long_docs", repo=REPO, revision=rev, subsets=a.subsets,
        seq_len=a.seq_len, min_tokens=a.min_tokens, tokenizer=a.model, seed=a.seed,
        one_window_per_doc=True, dtype="int32",
        input_shards_available=len(shards), docs_seen=state["docs_seen"], docs_too_short=state["docs_short"],
        train_windows=state["win_written"]["train"], train_tokens=state["tok_written"]["train"],
        val_windows=state["win_written"]["val"], val_tokens=state["tok_written"]["val"],
        out_files=state["shard_files"], elapsed_s=round(time.time() - t0, 1),
        git_commit=_git_commit(), invocation=" ".join([sys.executable] + sys.argv),
    )
    mp = os.path.join(a.out_dir, "MANIFEST.json")
    with open(mp, "w") as f:
        json.dump(manifest, f, indent=1)
    log.info("wrote %s", mp)
    log.info("DONE: train %.3fB tokens (%d windows), val %.3fB (%d windows) in %.1f min",
             state["tok_written"]["train"] / 1e9, state["win_written"]["train"],
             state["tok_written"]["val"] / 1e9, state["win_written"]["val"], (time.time() - t0) / 60)
    print(f"\nTRAIN_FILES={a.out_dir}/train-*.parquet\nVAL_FILES={a.out_dir}/val-*.parquet")


def _consume(texts, tok, a, state, flush):
    """Tokenize a batch, keep docs >= min_tokens truncated to seq_len, buffer, flush when full."""
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    for ids in enc:
        state["docs_seen"] += 1
        if len(ids) < a.min_tokens:
            state["docs_short"] += 1
            continue
        state["buf"].append(ids[: a.seq_len])
        if len(state["buf"]) >= a.windows_per_shard:
            flush()


if __name__ == "__main__":
    sys.exit(main())
