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
"""Dump long documents as RAW TEXT jsonl, filtered on a target model's own token count.

Why this exists: the existing InfLLM parquets (e.g.
``/cb/ml-eng/aarti/dsa/infllm_minicpm3_32768_250M_train.parquet``) contain **only** an ``input_ids``
column tokenized with ``openbmb/MiniCPM3-4B``. Those ids are useless for Qwen3 — different vocab —
and the raw text was not retained. Same gotcha the InfLLM pipeline already documents about the
upstream CPM-5 ``token_ids``: **always retokenize the text with the target model's tokenizer.**

So we re-stream the same source (``openbmb/InfLLM-V2-data-5B``, revision pinned) via the validated
``iter_docs`` reader and keep documents that reach ``--seq-len`` tokens **under the Qwen3
tokenizer**. Note this is a different bar than the MiniCPM3 run's: a larger vocab yields fewer tokens
for the same text, so the ≥32768-MiniCPM3-token pool is not the same set as the ≥32768-Qwen3-token
pool.

Consumed by ``scripts/msa/probe_block_oracle.py`` (which tokenizes the text itself and re-checks the
length, so this script's filter only needs to be approximately right).

Example
-------
    python3 scripts/msa/dump_long_docs.py \\
        --model /cb/ml-eng/aarti/models/qwen3_4b_thinking_2507 \\
        --seq-len 32768 --num-docs 128 \\
        --out /cb/ml-eng/aarti/msa/data/long_docs_qwen3_32768.jsonl
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples", "dsa"))

from _dsa_data_utils import SOURCES, iter_docs, resolve_revision  # noqa: E402

log = logging.getLogger("dump_long_docs")


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="tokenizer to filter with (the TARGET model)")
    ap.add_argument("--source", default="infllm", choices=sorted(SOURCES.keys()))
    ap.add_argument("--repo", default=None, help="override the source's HF dataset repo")
    ap.add_argument("--revision", default=None, help="pin; default = resolve current HEAD of the repo")
    ap.add_argument("--seq-len", type=int, default=32768, help="minimum token count to keep a doc")
    ap.add_argument("--num-docs", type=int, default=128, help="stop once this many docs are kept")
    ap.add_argument("--max-scanned", type=int, default=200_000, help="cap on docs scanned")
    ap.add_argument("--min-chars-per-token", type=float, default=3.0,
                    help="cheap pre-filter: skip docs shorter than seq_len * this, before tokenizing")
    ap.add_argument("--max-files", type=int, default=None, help="limit shards (debug)")
    ap.add_argument("--seed", type=int, default=1234, help="shard shuffle seed (matches the DSA pipeline)")
    ap.add_argument("--out", required=True, help="output .jsonl path (a .MANIFEST.json is written beside it)")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.out + ".log")],
    )
    log.info("CMD: %s", " ".join([sys.executable] + sys.argv))
    log.info("CWD: %s", os.getcwd())
    log.info("ENV: %s", {k: os.environ.get(k) for k in ("HF_HOME", "HF_TOKEN", "PYTHONPATH")})
    log.info("ARGS: %s", vars(a))

    src = SOURCES[a.source]
    repo = a.repo or src["repo"]
    revision = resolve_revision(repo, a.revision)
    log.info("source=%s repo=%s@%s glob=%s text_col=%s strip_markers=%s",
             a.source, repo, revision[:12], src["glob"], src["text_col"], src["strip_markers"])

    from huggingface_hub import HfFileSystem
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    log.info("tokenizer=%s vocab=%d", a.model, tok.vocab_size)

    min_chars = int(a.seq_len * a.min_chars_per_token)
    log.info("keeping docs with >= %d tokens; char pre-filter >= %d chars", a.seq_len, min_chars)

    fs = HfFileSystem()
    docs = iter_docs(fs, repo, src["glob"], src["text_col"],
                     revision=revision, file_seed=a.seed, max_files=a.max_files)

    kept, lens = 0, []
    scanned = short_chars = short_toks = 0
    t0 = time.time()
    with open(a.out, "w") as f:
        for text in docs:
            if kept >= a.num_docs or scanned >= a.max_scanned:
                break
            scanned += 1
            if not text:
                continue
            if src["strip_markers"]:
                text = text.replace("<s>", "").replace("</s>", "")
            if len(text) < min_chars:  # cheap reject before paying for tokenization
                short_chars += 1
                continue
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            if n < a.seq_len:
                short_toks += 1
                continue
            f.write(json.dumps({"text": text}) + "\n")
            lens.append(n)
            kept += 1
            if kept % 16 == 0:
                log.info("kept %d/%d  (scanned %d, %.0fs)", kept, a.num_docs, scanned, time.time() - t0)

    if kept == 0:
        raise RuntimeError(
            f"kept 0 docs after scanning {scanned}. Lower --seq-len, raise --max-scanned, or lower "
            f"--min-chars-per-token (currently rejecting {short_chars} docs on chars alone)."
        )
    if kept < a.num_docs:
        log.warning("kept only %d of the requested %d docs (scanned %d, cap %d)",
                    kept, a.num_docs, scanned, a.max_scanned)

    manifest = {
        "kind": "long_docs_text",
        "invocation": " ".join([sys.executable] + sys.argv),
        "git_commit": _git_commit(),
        "source": a.source,
        "repo": repo,
        "revision": revision,
        "glob": src["glob"],
        "text_col": src["text_col"],
        "strip_markers": src["strip_markers"],
        "tokenizer": a.model,
        "seq_len": a.seq_len,
        "seed": a.seed,
        "docs_scanned": scanned,
        "rejected_on_chars": short_chars,
        "rejected_on_tokens": short_toks,
        "docs_kept": kept,
        "token_len": {
            "min": min(lens), "median": int(statistics.median(lens)), "max": max(lens),
            "mean": int(statistics.fmean(lens)),
        },
        "out": os.path.abspath(a.out),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(a.out + ".MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1)

    log.info("kept %d docs | token_len min/med/max = %d/%d/%d | scanned %d in %.0fs",
             kept, manifest["token_len"]["min"], manifest["token_len"]["median"],
             manifest["token_len"]["max"], scanned, manifest["elapsed_s"])
    log.info("wrote %s (+ .MANIFEST.json)", a.out)


if __name__ == "__main__":
    main()
