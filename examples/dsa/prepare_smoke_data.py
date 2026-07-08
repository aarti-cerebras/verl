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
"""Make a tiny parquet with a `text` column for the DSA Phase-1 SMOKE run.

Synthetic text (no downloads) — enough tokens to form the handful of fixed-length windows a smoke needs.
For a *meaningful* warm-up, replace this with real long-context data (e.g. openbmb/InfLLM-V2-data-5B).

Usage:  python examples/dsa/prepare_smoke_data.py --out ~/data/dsa_smoke/train.parquet --rows 600 --words 500
"""

import argparse
import os
import random

import pandas as pd

# A small vocabulary so the tokenizer produces varied (non-degenerate) token streams.
_WORDS = (
    "model attention indexer sparse token sequence context query key value layer head softmax kernel "
    "gradient tensor matrix vector distribution kl divergence warmup dense flash rope position embedding "
    "the of and to in a is for on with as by from that this these those long document window packed "
    "compute memory throughput latency select topk score weight normalize scale bias mask causal".split()
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--rows", type=int, default=600, help="number of documents")
    ap.add_argument("--words", type=int, default=500, help="approx words per document")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    texts = []
    for _ in range(args.rows):
        n = max(1, int(rng.gauss(args.words, args.words * 0.2)))
        texts.append(" ".join(rng.choice(_WORDS) for _ in range(n)))

    os.makedirs(os.path.dirname(os.path.expanduser(args.out)) or ".", exist_ok=True)
    pd.DataFrame({"text": texts}).to_parquet(os.path.expanduser(args.out))
    total_words = sum(len(t.split()) for t in texts)
    print(f"wrote {args.rows} docs (~{total_words} words) -> {args.out}")


if __name__ == "__main__":
    main()
