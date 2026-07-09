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
"""Build the OUT-OF-DISTRIBUTION VAL set for DSA Phase-1 from ``openbmb/Ultra-FineWeb`` (multiple lengths).

Ultra-FineWeb is a FineWeb-style web corpus (``content`` column) — a distinct distribution from the
InfLLM-V2 training corpus, so it measures the indexer's cross-domain generalization. Because web docs are
mostly SHORT, we collect windows at SEVERAL target lengths (``--lengths``), each doc used at most once and
assigned to the largest length bucket it can fill. This yields a recall-vs-length eval: how well the
indexer's top-k selection holds as context grows, on unseen-domain text.

Output: ONE parquet **per length** (drop-in for ``PackedPretrainDataset`` with ``data.max_length=<L>``),
named ``<out_prefix>_L{L}.parquet``, plus a shared ``MANIFEST.json``. Reproducible: pinned ``--revision``
(dataset commit SHA) + ``--seed`` => identical windows.

Requires the transformers-4.57.1 env for the MiniCPM3 tokenizer (trust_remote_code):
    DEVLIBS=/cb/home/aarti/ws/code/ws_repos/dsa/verl/.devlibs/tf457lib
    PYTHONPATH=$DEVLIBS python examples/dsa/prepare_ood_data.py \
        --out_prefix data/dsa/ood_ultrafineweb_minicpm3 \
        --lengths 1024,2048,4096 --per_len 128 --seed 1234 --max_files 64
"""

import argparse
import random

from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

from _dsa_data_utils import SOURCES, collect_multilen_windows, iter_docs, resolve_revision, write_manifest, write_windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", required=True, help="output path prefix; writes <prefix>_L{len}.parquet per length")
    ap.add_argument("--lengths", default="1024,2048,4096", help="comma-separated target window lengths")
    ap.add_argument("--per_len", type=int, default=128, help="how many VAL windows to collect per length")
    ap.add_argument("--seed", type=int, default=1234, help="shuffle seed (file order + per-bucket shuffle)")
    ap.add_argument("--revision", default=None, help="HF dataset commit SHA to pin (default: current main, recorded)")
    ap.add_argument("--source", default="ultrafineweb_en", choices=[k for k in SOURCES if k != "infllm"])
    ap.add_argument("--max_files", type=int, default=64, help="cap shards scanned (corpus has 2048; a sample suffices)")
    ap.add_argument(
        "--min_score",
        type=float,
        default=0.9,
        help="quality filter: keep only docs with classifier score >= this (Ultra-FineWeb score in [0.5,1.0]; "
        "median ~0.78, p75 ~0.93). Set 0 to disable.",
    )
    ap.add_argument("--model", default="openbmb/MiniCPM3-4B", help="tokenizer (must match the trained model)")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    src = SOURCES[args.source]
    revision = resolve_revision(src["repo"], args.revision)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    min_score = args.min_score if (args.min_score and src["score_col"]) else None
    print(
        f"tokenizer={args.model} | OOD repo={src['repo']}@{revision[:12]} lengths={lengths} "
        f"per_len={args.per_len} min_score={min_score}"
    )

    fs = HfFileSystem()
    stats = {}  # populated live by iter_docs: rows_read, score_dropped
    docs = iter_docs(
        fs,
        src["repo"],
        src["glob"],
        src["text_col"],
        revision=revision,
        file_seed=args.seed,
        max_files=args.max_files,
        score_col=src["score_col"],
        min_score=min_score,
        stats=stats,
    )
    buckets, scanned = collect_multilen_windows(
        tok, docs, lengths, args.per_len, strip_markers=src["strip_markers"]
    )
    if min_score is not None:
        print(f"quality filter: read {stats.get('rows_read', 0)} rows, dropped {stats.get('score_dropped', 0)} below {min_score}")

    outputs = {}
    for L in sorted(buckets):
        wins = buckets[L]
        if not wins:
            print(f"  L={L}: 0 windows (corpus too short at this length; raise --max_files or lower L)")
            continue
        random.Random(args.seed + L).shuffle(wins)
        path = write_windows(wins, f"{args.out_prefix}_L{L}.parquet")
        print(f"  L={L}: {len(wins)}/{args.per_len} windows -> {path}  ({len(wins) * L:,} tokens)")
        outputs[f"L{L}"] = {"path": path, "windows": len(wins), "requested": args.per_len}

    write_manifest(
        f"{args.out_prefix}.MANIFEST.json",
        {
            "kind": "ood_multilen_val",
            "repo": src["repo"],
            "source": args.source,
            "revision": revision,
            "seed": args.seed,
            "lengths": lengths,
            "per_len": args.per_len,
            "max_files": args.max_files,
            "min_score": min_score,
            "rows_read": stats.get("rows_read"),
            "score_dropped": stats.get("score_dropped"),
            "tokenizer": args.model,
            "scanned_docs": scanned,
            "outputs": outputs,
        },
    )


if __name__ == "__main__":
    main()
