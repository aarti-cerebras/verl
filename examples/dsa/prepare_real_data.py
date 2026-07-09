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
"""Build the REAL long-context TRAIN (+ optional in-distribution VAL) parquet for DSA Phase-1 warm-up.

Source: ``openbmb/InfLLM-V2-data-5B``. We IGNORE the pre-tokenized ``token_ids`` (CPM-5 vocab, incompatible
with MiniCPM3-4B) and re-tokenize the raw ``text`` with the MiniCPM3-4B tokenizer. ONE document per row
(item 6a): each kept doc is filtered to >= ``seq_len`` tokens and truncated to exactly ``seq_len``.

TRAIN/VAL SPLIT (reproducible): with ``--val_out`` + ``--val_windows`` we collect a pool of
``num_windows + val_windows`` eligible windows, shuffle it with ``--seed``, and split into a TRAIN file and
a document-DISJOINT in-distribution VAL file. Fixed ``--revision`` (dataset commit SHA) + ``--seed`` =>
byte-identical splits; both are recorded in ``MANIFEST.json`` next to the outputs. Without ``--val_out`` it
behaves like the original single-file tool (sequential when ``--seed`` is omitted).

Requires the transformers-4.57.1 env for the MiniCPM3 tokenizer (trust_remote_code):
    DEVLIBS=/cb/home/aarti/ws/code/ws_repos/dsa/verl/.devlibs/tf457lib
    PYTHONPATH=$DEVLIBS python examples/dsa/prepare_real_data.py \
        --out data/dsa/infllm_minicpm3_4k_train.parquet --num_windows 2048 \
        --val_out data/dsa/infllm_minicpm3_4k_val.parquet --val_windows 256 \
        --seq_len 4096 --seed 1234
"""

import argparse
import math
import random

from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

from _dsa_data_utils import SOURCES, collect_windows, iter_docs, resolve_revision, write_manifest, write_windows


def main():
    src = SOURCES["infllm"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="TRAIN output parquet (single input_ids column)")
    ap.add_argument("--num_windows", type=int, default=2048, help="how many TRAIN one-doc windows to collect")
    ap.add_argument("--val_out", default=None, help="optional VAL output parquet (disjoint docs); enables split")
    ap.add_argument("--val_windows", type=int, default=0, help="how many in-distribution VAL windows")
    ap.add_argument("--seq_len", type=int, default=4096, help="window length; each row is exactly this many tokens")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed; omit for legacy sequential selection")
    ap.add_argument("--revision", default=None, help="HF dataset commit SHA to pin (default: current main, recorded)")
    ap.add_argument("--oversample", type=float, default=1.0, help="collect this x (train+val) before shuffle/split")
    ap.add_argument("--model", default="openbmb/MiniCPM3-4B", help="tokenizer (must match the trained model)")
    ap.add_argument("--min_len", type=int, default=None, help="min tokens to keep a doc (default: seq_len)")
    ap.add_argument("--repo", default=src["repo"], help="HF dataset repo")
    args = ap.parse_args()

    revision = resolve_revision(args.repo, args.revision)  # pin + record the exact commit
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer={args.model} vocab={tok.vocab_size} bos={tok.bos_token_id} | repo={args.repo}@{revision[:12]}")

    n_train = args.num_windows
    n_val = args.val_windows if args.val_out else 0
    pool_needed = math.ceil((n_train + n_val) * max(1.0, args.oversample))

    fs = HfFileSystem()
    # file_seed shuffles which shards we scan first, so a bounded pool samples across the corpus (not just shard 0).
    docs = iter_docs(fs, args.repo, src["glob"], src["text_col"], revision=revision, file_seed=args.seed)
    pool, scanned = collect_windows(
        tok, docs, args.seq_len, pool_needed, strip_markers=src["strip_markers"], min_len=args.min_len
    )
    if len(pool) < n_train + n_val:
        raise RuntimeError(
            f"collected {len(pool)} windows < requested {n_train + n_val}; scan more docs or lower the counts"
        )

    if args.seed is not None:
        random.Random(args.seed).shuffle(pool)  # decorrelate train vs val membership

    train = pool[:n_train]
    val = pool[n_train : n_train + n_val]

    out = write_windows(train, args.out)
    print(f"TRAIN: {len(train)} x {args.seq_len} tok -> {out}  ({len(train) * args.seq_len:,} tokens)")
    outputs = {"train": out, "train_windows": len(train)}
    if n_val:
        vout = write_windows(val, args.val_out)
        print(f"VAL(in-dist): {len(val)} x {args.seq_len} tok -> {vout}  ({len(val) * args.seq_len:,} tokens)")
        outputs.update(val=vout, val_windows=len(val))

    write_manifest(
        f"{args.out}.MANIFEST.json",
        {
            "kind": "infllm_train_val",
            "repo": args.repo,
            "revision": revision,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "min_len": args.min_len or args.seq_len,
            "oversample": args.oversample,
            "tokenizer": args.model,
            "scanned_docs": scanned,
            "pool_size": len(pool),
            **outputs,
        },
    )


if __name__ == "__main__":
    main()
