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
"""Prepare a REAL long-context parquet for DSA Phase-1 indexer warm-up (one-doc-per-row).

Source: ``openbmb/InfLLM-V2-data-5B`` (20 parquet shards, columns ``token_ids`` + ``text``).
We IGNORE the pre-tokenized ``token_ids`` (they are CPM-5 vocab, incompatible with MiniCPM3-4B) and
re-tokenize the raw ``text`` with the MiniCPM3-4B tokenizer, so the ids match the model we train.

MVP shape (see docs/dsa_train_indexer_plan.md item 6a): ONE document per row, no cross-doc packing.
Each kept doc is tokenized, filtered to >= ``seq_len`` tokens, and truncated to exactly ``seq_len`` so
every row is a clean single-doc window. The base attention stays full-causal (correct for one doc) and
the KL target ``p`` equals the true base attention -- no varlen wiring needed.

The raw text carries literal ``<s>``/``</s>`` markers from the source's decode. ``<s>`` maps to the BOS
id (1), so tokenizing it raw with ``add_special_tokens=True`` yields a DOUBLE BOS ``[1, 1, ...]``. We
strip those markers and let the tokenizer add exactly one BOS -> ``[1, 6739, ...]``.

Reads only the parquet footer + the row groups it needs (HF filesystem range requests), so it does NOT
download whole shards. Writes a parquet with a single ``input_ids`` column (list<int>, length seq_len).

Requires the transformers-4.57.1 env for the MiniCPM3 tokenizer (trust_remote_code):
    DEVLIBS=/cb/home/aarti/ws/code/ws_repos/dsa/verl/.devlibs/tf457lib
    PYTHONPATH=$DEVLIBS python examples/dsa/prepare_real_data.py \
        --out data/dsa/infllm_minicpm3_4k.parquet --seq_len 4096 --num_windows 512
"""

import argparse
import os

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer

NUM_SHARDS = 20


def clean_text(text: str) -> str:
    """Strip the source's literal <s>/</s> markers so the tokenizer adds exactly one BOS."""
    t = text.strip()
    if t.startswith("<s>"):
        t = t[len("<s>") :].lstrip()
    if t.endswith("</s>"):
        t = t[: -len("</s>")].rstrip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output parquet path (single input_ids column)")
    ap.add_argument("--seq_len", type=int, default=4096, help="window length; each row is exactly this many tokens")
    ap.add_argument("--num_windows", type=int, default=512, help="how many one-doc windows to collect")
    ap.add_argument("--model", default="openbmb/MiniCPM3-4B", help="tokenizer to use (must match the trained model)")
    ap.add_argument("--repo", default="openbmb/InfLLM-V2-data-5B", help="HF dataset repo")
    ap.add_argument("--min_len", type=int, default=None, help="min tokens to keep a doc (default: seq_len)")
    args = ap.parse_args()
    min_len = args.min_len or args.seq_len

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer: {args.model}  vocab_size={tok.vocab_size}  bos={tok.bos_token_id}  eos={tok.eos_token_id}")
    fs = HfFileSystem()

    windows: list[list[int]] = []
    scanned = 0
    # Bound per-doc tokenization work: seq_len*8 chars is >= 2*seq_len tokens (~4 chars/token), enough to
    # confirm a doc reaches min_len and to slice seq_len, without tokenizing 60k-token docs in full.
    char_cap = args.seq_len * 8

    for shard in range(NUM_SHARDS):
        if len(windows) >= args.num_windows:
            break
        path = f"datasets/{args.repo}/train/data-{shard:05d}-of-{NUM_SHARDS:05d}.parquet"
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                if len(windows) >= args.num_windows:
                    break
                tbl = pf.read_row_group(rg, columns=["text"])
                for text in tbl.column("text").to_pylist():
                    scanned += 1
                    ids = tok(clean_text(text)[:char_cap], add_special_tokens=True)["input_ids"]
                    if len(ids) >= min_len:
                        windows.append(ids[: args.seq_len])
                        if len(windows) >= args.num_windows:
                            break
                print(f"  shard {shard} rg {rg}: kept {len(windows)}/{args.num_windows} (scanned {scanned})")

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame({"input_ids": windows}).to_parquet(out)
    yield_pct = 100 * len(windows) / max(scanned, 1)
    print(
        f"\nwrote {len(windows)} windows x {args.seq_len} tokens -> {out}\n"
        f"scanned {scanned} docs, yield {yield_pct:.1f}%, total tokens {len(windows) * args.seq_len:,}"
    )


if __name__ == "__main__":
    main()
