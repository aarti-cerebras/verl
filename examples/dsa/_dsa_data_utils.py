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
"""Shared, reproducible helpers for building DSA Phase-1 indexer datasets (train / in-dist val / OOD val).

Reproducibility contract: **pinned HF dataset revision + fixed seed => byte-identical windows**. The prep
scripts (``prepare_real_data.py``, ``prepare_ood_data.py``) resolve and record the source commit SHA and
the seed in a MANIFEST.json next to the output, so a later run with the recorded SHA + seed regenerates the
exact same splits. Determinism comes from: sorted file list -> optional seeded file shuffle -> in-order row
scan -> seeded shuffle of the collected window pool. No ``Math.random``-style unseeded calls anywhere.

All windows are ONE document per row (item 6a shape): each doc is tokenized with the *training* tokenizer
(MiniCPM3-4B), filtered to >= the target length, and truncated to exactly that length. Reads only the
parquet row groups it needs via HF range requests (no full-shard downloads).
"""

import json
import os
import random
import subprocess

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

# Known sources: name -> (repo, glob under datasets/<repo>/, text column, strip <s></s> markers,
# score_col = quality-classifier score column for optional quality filtering (None if the source has none).
SOURCES = {
    # long-context training corpus; token_ids are CPM-5 vocab (incompatible) so we retokenize `text`.
    "infllm": dict(
        repo="openbmb/InfLLM-V2-data-5B",
        glob="train/data-*-of-*.parquet",
        text_col="text",
        strip_markers=True,
        score_col=None,  # no per-doc quality score in this corpus
    ),
    # OOD web corpus (FineWeb-style); shorter docs, distinct distribution from InfLLM. `score` is a
    # quality-classifier float in [0.5, 1.0] (corpus is pre-filtered to >= 0.5); higher = higher quality.
    "ultrafineweb_en": dict(
        repo="openbmb/Ultra-FineWeb",
        glob="data/ultrafineweb_en/ultrafineweb-en-part-*.parquet",
        text_col="content",
        strip_markers=False,
        score_col="score",
    ),
}


def clean_text(text: str, strip_markers: bool) -> str:
    """Strip the source's literal ``<s>``/``</s>`` markers so the tokenizer adds exactly one BOS."""
    t = text.strip()
    if strip_markers:
        if t.startswith("<s>"):
            t = t[len("<s>") :].lstrip()
        if t.endswith("</s>"):
            t = t[: -len("</s>")].rstrip()
    return t


def resolve_revision(repo: str, revision=None) -> str:
    """Resolve (and thus pin) the dataset commit SHA. Pass the returned SHA back via --revision to reproduce."""
    return HfApi().dataset_info(repo, revision=revision).sha


def _repo_root(repo: str, revision=None) -> str:
    # HfFileSystem supports the `datasets/<repo>@<revision>/...` path syntax for a pinned commit.
    return f"datasets/{repo}@{revision}" if revision else f"datasets/{repo}"


def iter_docs(
    fs: HfFileSystem,
    repo: str,
    glob: str,
    text_col: str,
    revision=None,
    file_seed=None,
    max_files=None,
    score_col=None,
    min_score=None,
    stats=None,
):
    """Yield raw doc strings in a deterministic order, optionally quality-filtered.

    Files are globbed and **sorted** (stable); if ``file_seed`` is given they are shuffled with that seed
    (so a small ``max_files`` samples across the whole corpus instead of just the first shard). Row groups
    and rows are read in file order.

    Quality filter: when ``score_col`` and ``min_score`` are both set, the score column is read alongside
    the text and any row with ``float(score) < min_score`` (or an unparseable score) is dropped. If
    ``stats`` (a dict) is passed, ``stats['rows_read']`` / ``stats['score_dropped']`` are updated live
    (before each yield) so counts are accurate even when the caller stops early.
    """
    root = _repo_root(repo, revision)
    files = sorted(fs.glob(f"{root}/{glob}"))
    if not files:
        raise FileNotFoundError(f"no files matched {root}/{glob}")
    if file_seed is not None:
        random.Random(file_seed).shuffle(files)
    if max_files:
        files = files[:max_files]
    apply_score = score_col is not None and min_score is not None
    cols = [text_col] + ([score_col] if apply_score else [])
    read = dropped = 0
    for path in files:
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=cols)
                texts = tbl.column(text_col).to_pylist()
                scores = tbl.column(score_col).to_pylist() if apply_score else [None] * len(texts)
                for text, sc in zip(texts, scores):
                    read += 1
                    if apply_score:
                        try:
                            keep = float(sc) >= min_score
                        except (TypeError, ValueError):
                            keep = False
                        if not keep:
                            dropped += 1
                            continue
                    if stats is not None:
                        stats["rows_read"] = read
                        stats["score_dropped"] = dropped
                    yield text


def collect_windows(tok, docs, seq_len: int, n_needed: int, *, strip_markers: bool, min_len=None, char_cap=None):
    """Tokenize streamed docs, keep those with >= ``min_len`` tokens, truncate to ``seq_len``.

    Returns ``(windows, scanned)``. ``char_cap`` bounds per-doc tokenization work (default seq_len*8 chars,
    ~2x seq_len tokens) so we never tokenize a 60k-token doc in full just to slice ``seq_len``.
    """
    min_len = min_len or seq_len
    char_cap = char_cap or seq_len * 8
    windows: list[list[int]] = []
    scanned = 0
    for text in docs:
        if len(windows) >= n_needed:
            break
        scanned += 1
        t = clean_text(text, strip_markers)[:char_cap]
        ids = tok(t, add_special_tokens=True)["input_ids"]
        if len(ids) >= min_len:
            windows.append([int(x) for x in ids[:seq_len]])
    return windows, scanned


def collect_multilen_windows(tok, docs, lengths, per_len: int, *, strip_markers: bool, char_cap=None):
    """Collect ``per_len`` windows for EACH target length, using every doc at most once.

    Each doc is tokenized once and assigned to the **largest** still-hungry length bucket it can fill
    (>= that length). Good for a short OOD corpus (Ultra-FineWeb): long buckets fill from the rare long
    docs, short buckets from the many short ones. Returns ``(buckets: dict[L]->windows, scanned)``.
    """
    targets = sorted(lengths, reverse=True)
    char_cap = char_cap or max(targets) * 8
    need = {L: per_len for L in targets}
    buckets = {L: [] for L in targets}
    scanned = 0
    for text in docs:
        if all(n <= 0 for n in need.values()):
            break
        scanned += 1
        t = clean_text(text, strip_markers)[:char_cap]
        ids = tok(t, add_special_tokens=True)["input_ids"]
        n = len(ids)
        for L in targets:  # largest first: use long docs where they're scarce
            if need[L] > 0 and n >= L:
                buckets[L].append([int(x) for x in ids[:L]])
                need[L] -= 1
                break
    return buckets, scanned


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_windows(windows, out_path: str):
    """Write a list of int-lists to a single-``input_ids``-column parquet (the PackedPretrainDataset format)."""
    out = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame({"input_ids": windows}).to_parquet(out)
    return out


def write_manifest(out_path: str, manifest: dict):
    """Write a MANIFEST.json capturing seed/revision/counts/tokenizer/git so the build is reproducible."""
    manifest = {**manifest, "git_commit": _git_commit()}
    out = os.path.expanduser(out_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"wrote manifest -> {out}")
    return out
