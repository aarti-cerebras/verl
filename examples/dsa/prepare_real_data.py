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

from _dsa_data_utils import (
    SOURCES,
    collect_doc_windows,
    collect_windows,
    iter_docs,
    resolve_revision,
    split_doc_groups,
    write_manifest,
    write_windows,
)


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
    ap.add_argument(
        "--windows_per_doc",
        type=int,
        default=0,
        help="0/1 = one window per doc (default, byte-identical to prior sets); >1 = slice each long doc "
        "into up to this many consecutive windows, with a document-level disjoint train/val split",
    )
    ap.add_argument("--repo", default=src["repo"], help="HF dataset repo")
    ap.add_argument(
        "--best_effort",
        action="store_true",
        help="if the corpus can't supply the requested windows after a full scan, write whatever was "
        "collected (val kept at <=25%% of the pool, rest to train) instead of erroring",
    )
    args = ap.parse_args()

    revision = resolve_revision(args.repo, args.revision)  # pin + record the exact commit
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer={args.model} vocab={tok.vocab_size} bos={tok.bos_token_id} | repo={args.repo}@{revision[:12]}")

    n_train = args.num_windows
    n_val = args.val_windows if args.val_out else 0
    multi = args.windows_per_doc and args.windows_per_doc > 1

    fs = HfFileSystem()
    # file_seed shuffles which shards we scan first, so a bounded pool samples across the corpus (not just shard 0).
    docs = iter_docs(fs, args.repo, src["glob"], src["text_col"], revision=revision, file_seed=args.seed)

    if multi:
        # Option B: multiple windows per doc; over-collect by one full group so a val doc straddling the
        # boundary can't starve train, then split at the DOCUMENT level (leak-free).
        pool_needed = math.ceil((n_train + n_val) * max(1.0, args.oversample)) + args.windows_per_doc
        groups, scanned = collect_doc_windows(
            tok, docs, args.seq_len, pool_needed,
            strip_markers=src["strip_markers"], min_len=args.min_len, max_per_doc=args.windows_per_doc,
        )
        pool_size = sum(len(g) for g in groups)
        if pool_size < n_train + n_val:
            if args.best_effort:
                n_val = min(n_val, pool_size // 4)  # cap val at 25% of the shortfall pool
                n_train = pool_size - n_val
                print(f"[best-effort] collected {pool_size} windows < requested; writing ~{n_train} train + {n_val} val")
            else:
                raise RuntimeError(
                    f"collected {pool_size} windows (from {len(groups)} docs) < requested {n_train + n_val}; "
                    "scan more docs or lower the counts"
                )
        train, val = split_doc_groups(groups, n_train, n_val, args.seed)
        if not args.best_effort and (len(train) < n_train or len(val) < n_val):
            raise RuntimeError(
                f"after doc-level split have train={len(train)}/{n_train} val={len(val)}/{n_val}; "
                "raise --oversample or scan more docs"
            )
    else:
        pool_needed = math.ceil((n_train + n_val) * max(1.0, args.oversample))
        pool, scanned = collect_windows(
            tok, docs, args.seq_len, pool_needed, strip_markers=src["strip_markers"], min_len=args.min_len
        )
        if len(pool) < n_train + n_val:
            if args.best_effort:
                n_val = min(n_val, len(pool) // 4)  # cap val at 25% of the shortfall pool
                n_train = len(pool) - n_val
                print(f"[best-effort] collected {len(pool)} windows < requested; writing {n_train} train + {n_val} val")
            else:
                raise RuntimeError(
                    f"collected {len(pool)} windows < requested {n_train + n_val}; scan more docs or lower the counts"
                )
        if args.seed is not None:
            random.Random(args.seed).shuffle(pool)  # decorrelate train vs val membership
        pool_size = len(pool)
        train = pool[:n_train]
        val = pool[n_train : n_train + n_val]

    # Dedup (leak-free guarantee): the corpus contains exact-duplicate documents, so identical windows can
    # appear twice and straddle the train/val split (val leakage) or repeat within train. Drop duplicate
    # train rows (keep first, order-preserving) and any val row whose content also appears in train. Applies
    # to BOTH paths; content key is the token tuple.
    seen = set()
    deduped_train = []
    for w in train:
        k = tuple(w)
        if k in seen:
            continue
        seen.add(k)
        deduped_train.append(w)
    dup_train = len(train) - len(deduped_train)
    train = deduped_train
    val_before = len(val)
    val = [w for w in val if tuple(w) not in seen]
    leaked_val = val_before - len(val)
    if dup_train or leaked_val:
        print(f"[dedup] removed {dup_train} duplicate train rows + {leaked_val} val rows overlapping train")

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
            "kind": "infllm_train_val_multi" if multi else "infllm_train_val",
            "repo": args.repo,
            "revision": revision,
            "seed": args.seed,
            "seq_len": args.seq_len,
            "min_len": args.min_len or args.seq_len,
            "windows_per_doc": args.windows_per_doc if multi else 1,
            "split_level": ("document" if multi else "window") + "+dedup",
            "dedup": True,
            "train_dups_removed": dup_train,
            "val_leaked_removed": leaked_val,
            "best_effort": args.best_effort,
            "oversample": args.oversample,
            "tokenizer": args.model,
            "scanned_docs": scanned,
            "pool_size": pool_size,
            **outputs,
        },
    )


if __name__ == "__main__":
    main()
