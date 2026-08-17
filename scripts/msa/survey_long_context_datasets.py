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
"""Measure the *prefill* token-length distribution of candidate long-context datasets.

Answers one question per dataset: **what fraction of rows have >= 16384 input tokens under the
TARGET model's tokenizer?** That is the bar `docs/qwen3_4b_dsa/data_plan.md` §5 sets for the
prefill-long bucket, and §11 records that no instruction-shaped source had been surveyed yet.

Why this script exists rather than reading dataset cards:

  * **Cards quote the wrong tokenizer.** `zai-org/LongAlign-10k` ships a `length` column measured
    with ChatGLM3; Qwen3's vocab is 3× larger so the same text yields fewer tokens. Same gotcha the
    InfLLM/longmino pipelines already document: never trust another model's token counts.
  * **`datasets-server /rows` silently truncates long cells.** Every long-context candidate reports
    ``truncated_cells: ['messages']``, so length measured through that API is meaningless. We read
    the parquet **row groups directly** over HTTP range requests instead.
  * **Prefill length != total length.** For sparse-attention training what matters is the context the
    model must attend *over* (everything up to the final assistant turn), not the response. A
    decode-long dataset (LongWriter) and a prefill-long one (LongAlign) can share a total length and
    be useless vs. useful respectively. We report both, split at the last assistant message.

Reads only as many row groups as needed for `--num-rows`, so a 3 GB shard costs a few MB of transfer.

Example
-------
    python3 scripts/msa/survey_long_context_datasets.py \
        --model /cb/ml-eng/aarti/models/qwen3_4b_thinking_2507 \
        --num-rows 300 --out /cb/ml-eng/aarti/msa/data/long_ctx_survey.json
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request

for _n in ("filelock", "httpx", "httpcore", "huggingface_hub", "fsspec", "urllib3"):
    logging.getLogger(_n).setLevel(logging.WARNING)

# (dataset, config, split, note). config=None -> first config the server reports.
CANDIDATES = [
    # --- instruction-shaped prefill-long (the data_plan.md §11 gap) ---
    ("HuggingFaceTB/smoltalk2", "SFT", "LongAlign_64k_Qwen3_32B_yarn_131k_think", "Qwen3-32B think traces"),
    ("HuggingFaceTB/smoltalk2", "SFT", "LongAlign_64k_context_lang_annotated_lang_6_no_think", "no-think variant"),
    ("zai-org/LongAlign-10k", "default", "train", "original LongAlign contexts"),
    ("zai-org/LongCite-45k", "default", "train", "long QA w/ citations"),
    ("zai-org/LongReward-10k", "default", "sft", "9-domain long SFT (NOT the dpo_* splits)"),
    ("Yukang/LongAlpaca-12k", "default", "train", "papers/books QA"),
    ("nvidia/ChatQA2-Long-SFT-data", "long_sft", "train", "CC-BY-NC-2.0 - noncommercial"),
    ("nvidia/ChatQA2-Long-SFT-data", "NarrativeQA_131072", "train", "CC-BY-NC-2.0 - noncommercial"),
    ("Tongyi-Zhiwen/DocQA-RL-1.6K", None, None, "RL prompts, math/logic over docs"),
    # --- LoongRL: multi-hop QA + injected distractors, built at max_seq 16384 ---
    ("OldKingMeister/LoongRL-Train-Data", "hotpotqa_distractor_2500_5000", "train", "SEMI-SYNTHETIC needles"),
    ("OldKingMeister/LoongRL-Train-Data", "hotpotqa_qwen_0_2500", "train", "SEMI-SYNTHETIC needles"),
    ("OldKingMeister/LoongRL-Train-Data", "2wikipedia_distractor_2500_5000", "train", "SEMI-SYNTHETIC needles"),
    ("OldKingMeister/LoongRL-Train-Data", "musique_distractor_2500_5000", "train", "SEMI-SYNTHETIC needles"),
    # --- code reasoning: expect SHORT prefill (decode-long), checked because code is the S1 gap ---
    ("nvidia/OpenCodeReasoning", "split_0", "split_0", "code, R1 traces"),
    ("nvidia/OpenCodeReasoning", "split_1", "split_1", "code, R1 traces"),
    ("nvidia/OpenCodeReasoning-2", "train", "python", "code, R1 traces"),
    ("nvidia/OpenCodeReasoning-2", "train", "cpp", "code, R1 traces"),
    # --- decode-long control: expect SHORT prefill, long output ---
    ("zai-org/LongWriter-6k", "default", "train", "control: decode-long, must show ~0% >=16K"),
]

# Raw .jsonl files the datasets-server never converted: (dataset, filename, note)
JSONL_CANDIDATES = [
    ("YeungNLP/LongQLoRA-Dataset", "LongQLoRA-SFT-Data-39k.jsonl", "LongQLoRA SFT (mixed short+long)"),
    ("YeungNLP/LongQLoRA-Dataset", "LongQLoRA-Pretrain-Data-54k.jsonl", "LongQLoRA pretrain docs"),
]

# Surveyed and REJECTED before measurement — recorded so nobody re-derives this.
#   princeton-nlp/prolong-data-{64K,512K}  MDS, columns [domain, indices, input_ids, length].
#   amd/Instella-Long (sft/ + pretrain-*)   MDS, columns [indices, input_ids, label_mask, length].
#     Both ship **only another model's token ids, no raw text** (Llama-3 / OLMo respectively) — the exact
#     trap `dump_long_docs.py` documents. Usable only via a decode->retokenize round trip through the
#     source tokenizer, and their rows are PACKED multi-document sequences (see the `indices` column),
#     so honouring the one-doc-per-row rule means splitting on those offsets first.
#   yuyijiong/Long-Instruction-with-Paraphrasing  ships only .zip archives (no parquet/jsonl); zh-heavy.
#   togethercomputer/Long-Data-Collections  no longer resolves on the Hub.

# Field groups we know how to read, tried in order. Each entry is (context_fields, response_fields).
FLAT_SCHEMAS = [
    (("instruction", "input"), ("output",)),
    (("context", "query"), ("answer",)),
    (("context", "question"), ("answer",)),
    # nvidia/ChatQA2: the document sits in `sub-paragraphs`, and `question` already carries the
    # instruction preamble. NOTE its text has Llama-3 `<|begin_of_text|>` markers baked in — those
    # must be stripped before use with any other tokenizer.
    (("sub-paragraphs", "question"), ("answer",)),
    (("prompt",), ("response",)),
    (("prompt",), ("completion",)),
    # nvidia/OpenCodeReasoning + YeungNLP/LongQLoRA: bare input/output pair.
    (("input",), ("output",)),
    # nvidia/OpenCodeReasoning-2. WARNING: `question` is the literal placeholder "-" for the
    # apps/taco/code_contests rows -- the prompt text is withheld for licensing and must be recovered
    # by joining `question_id` back to the upstream dataset. Measured prefill is meaningless there.
    (("question",), ("r1_generation",)),
    (("text",), ()),
    (("content",), ()),
]
# `prompt` is included because LoongRL ships a chat list under that name (with no assistant turn, so
# the whole thing is prefill — correct for RL prompt data). It stays a FLAT_SCHEMAS key too: the
# message path only fires when the value is actually a list.
MESSAGE_FIELDS = ("messages", "conversations", "conversation", "chat", "prompt")


def _get(url, retries=4):
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - transient 5xx from the datasets-server is normal
            if a == retries - 1:
                raise
            print(f"    retry {a + 1} after {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(5 * (a + 1))
    return None


def list_parquet(dataset, config, split):
    meta = _get(f"https://datasets-server.huggingface.co/parquet?dataset={dataset}")
    if "error" in meta:
        raise RuntimeError(meta["error"])
    files = meta["parquet_files"]
    if config:
        files = [f for f in files if f["config"] == config]
    else:
        first = files[0]["config"]
        files = [f for f in files if f["config"] == first]
    # The server names a split "partial-train" when it only converted the first ~5 GB of a large
    # dataset. Treat that as "train": the rows are real, only the tail is missing (flagged below).
    def norm(s):
        return s[len("partial-") :] if s.startswith("partial-") else s

    if split:
        files = [f for f in files if norm(f["split"]) == norm(split)]
    else:
        pref = [f for f in files if norm(f["split"]) == "train"] or files
        files = [f for f in pref if f["split"] == pref[0]["split"]]
    if not files:
        raise RuntimeError(f"no parquet files for {dataset} {config} {split} (not converted to parquet?)")
    return files


def fs_path(url):
    """HfFileSystem path from the datasets-server `url`, instead of rebuilding the layout by hand.

    The layout varies (``<cfg>/<split>/0000.parquet`` vs ``<cfg>/<split>-00000-of-0000N.parquet``, and
    ``partial-`` split prefixes), so deriving it from the URL the server actually gives us is the only
    reliable form.
    """
    marker = "/resolve/"
    repo, rest = url.split(marker, 1)
    repo = repo.split("huggingface.co/datasets/", 1)[1]
    rev, path = rest.split("/", 1)
    return f"datasets/{repo}@{rev}/{path}"



def measure_jsonl(dataset, filename, tok, num_rows, note, n_offsets=24):
    """Measure a plain .jsonl file the datasets-server refused to convert (LongQLoRA).

    Reads `n_offsets` chunks spread across the file by BYTE offset and discards the first (partial)
    line of each. Strided for the same reason the parquet path is: LongQLoRA-SFT-39k front-loads short
    Evol-Instruct rows, so a head read reports ~0 long rows and is simply wrong.
    """
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f"datasets/{dataset}/{filename}"
    size = fs.info(path)["size"]
    per = max(1, num_rows // n_offsets)
    chunk = 4 << 20
    ctx_lens, resp_lens, line_bytes = [], [], []

    with fs.open(path, "rb") as fh:
        for i in range(n_offsets):
            off = int(i * (size - chunk) / max(1, n_offsets - 1)) if n_offsets > 1 else 0
            fh.seek(max(0, off))
            blob = fh.read(chunk).decode("utf-8", "ignore")
            lines = blob.split("\n")
            if i or off:
                lines = lines[1:]          # partial first line
            taken = 0
            for line in lines[:-1]:        # last line is also partial
                if taken >= per:
                    break
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                ctx, resp = split_row(row, set(row.keys()))
                if ctx is None:
                    continue
                line_bytes.append(len(line))
                ctx_lens.append(len(tok(ctx, add_special_tokens=False)["input_ids"]))
                resp_lens.append(len(tok(resp, add_special_tokens=False)["input_ids"]) if resp else 0)
                taken += 1
    if not ctx_lens:
        return {"error": "no parseable jsonl rows"}
    # No row count in a jsonl; estimate from mean line length. Flagged as an estimate downstream.
    est_rows = round(size / (sum(line_bytes) / len(line_bytes)))
    return summarize(dataset, filename, "jsonl", note + f"; rows ESTIMATED from {size / 1e9:.2f} GB / mean line",
                     est_rows, ctx_lens, resp_lens, partial=False)


def split_row(row, columns):
    """Return (context_text, response_text). Context = everything the model must attend over."""
    for f in MESSAGE_FIELDS:
        if f in columns and row.get(f):
            msgs = row[f]
            if not isinstance(msgs, list) or not msgs or not isinstance(msgs[0], dict):
                continue
            def role(m):
                return (m.get("role") or m.get("from") or "").lower()

            def content(m):
                v = m.get("content", m.get("value", ""))
                return v if isinstance(v, str) else json.dumps(v)

            # prefill = up to (excluding) the LAST assistant/gpt turn
            last = -1
            for i, m in enumerate(msgs):
                if role(m) in ("assistant", "gpt", "model"):
                    last = i
            if last < 0:
                return "\n".join(content(m) for m in msgs), ""
            return "\n".join(content(m) for m in msgs[:last]), content(msgs[last])
    for ctx_f, resp_f in FLAT_SCHEMAS:
        if all(f in columns for f in ctx_f):
            def flat(f):
                v = row.get(f)
                if v is None:
                    return ""
                return "\n".join(str(x) for x in v) if isinstance(v, list) else str(v)

            ctx = "\n".join(flat(f) for f in ctx_f)
            resp = "\n".join(flat(f) for f in resp_f if f in columns)
            if ctx.strip():
                return ctx, resp
    return None, None


def measure(dataset, config, split, tok, num_rows, note):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    files = list_parquet(dataset, config, split)
    cfg, spl = files[0]["config"], files[0]["split"]
    fs = HfFileSystem()
    ctx_lens, resp_lens, schema_used = [], [], None

    # True row count up front, so we never have to read every shard to know the denominator.
    n_rows_total = 0
    info = _get(f"https://datasets-server.huggingface.co/size?dataset={dataset}")
    for s in info.get("size", {}).get("splits", []):
        if s["config"] == cfg and s["split"] == spl:
            n_rows_total = s["num_rows"]
    partial = spl.startswith("partial-")

    # Sample row groups STRIDED across shards, never the first-N rows. Several of these datasets are
    # ordered (LongAlpaca-12k is ~9k long QA rows followed by ~3k short Alpaca rows; ChatQA2's
    # long_sft concatenates its sub-corpora), so a sequential read reports whichever block happens to
    # be first and silently misses the rest of the distribution.
    shard_stride = max(1, len(files) // 4)
    shards = files[::shard_stride][:4]
    per_shard = max(1, num_rows // len(shards))

    for f in shards:
        path = fs_path(f["url"])
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            if not n_rows_total:
                n_rows_total = pf.metadata.num_rows * len(files)
            cols = set(pf.schema_arrow.names)
            want = [c for c in cols if c in set(MESSAGE_FIELDS) | {x for s in FLAT_SCHEMAS for x in s[0] + s[1]}]
            if not want:
                return {"error": f"unknown schema: {sorted(cols)}"}
            n_rg = pf.metadata.num_row_groups
            rg_stride = max(1, n_rg // 4)
            rgs = list(range(0, n_rg, rg_stride))[:4]
            per_rg = max(1, per_shard // len(rgs))
            got_shard = 0
            for rg in rgs:
                tbl = pf.read_row_group(rg, columns=want)
                rows = tbl.to_pylist()
                # spread within the row group too
                step = max(1, len(rows) // per_rg)
                taken = 0
                for row in rows[::step]:
                    if taken >= per_rg:
                        break
                    ctx, resp = split_row(row, cols)
                    if ctx is None:
                        continue
                    schema_used = schema_used or sorted(want)
                    ctx_lens.append(len(tok(ctx, add_special_tokens=False)["input_ids"]))
                    resp_lens.append(len(tok(resp, add_special_tokens=False)["input_ids"]) if resp else 0)
                    taken += 1
                    got_shard += 1
            del got_shard

    if not ctx_lens:
        return {"error": "no parseable rows"}
    return summarize(dataset, cfg, spl, note, n_rows_total, ctx_lens, resp_lens, partial,
                     schema=schema_used)


def summarize(dataset, cfg, spl, note, n_rows_total, ctx_lens, resp_lens, partial, schema=None):
    # Pair per row BEFORE sorting: sorting the two lists independently would add row i's context to
    # row j's response.
    tot = sorted(c + r for c, r in zip(ctx_lens, resp_lens, strict=True))
    ctx_lens = sorted(ctx_lens)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

    def frac(xs, thr):
        return sum(1 for x in xs if x >= thr) / len(xs)

    long_ones = [x for x in ctx_lens if x >= 16384]
    return {
        "dataset": dataset,
        "config": cfg,
        "split": spl,
        "note": note + ("; datasets-server converted only a PREFIX of this dataset" if partial else ""),
        "partial_conversion": partial,
        "rows_total": n_rows_total,
        "rows_sampled": len(ctx_lens),
        "schema": schema,
        "prefill_tokens": {
            "p10": pct(ctx_lens, 10), "p50": pct(ctx_lens, 50), "p90": pct(ctx_lens, 90),
            "p99": pct(ctx_lens, 99), "max": ctx_lens[-1], "mean": round(sum(ctx_lens) / len(ctx_lens)),
        },
        "total_tokens": {"p50": pct(tot, 50), "p90": pct(tot, 90), "max": tot[-1]},
        "resp_tokens": {"p50": pct(sorted(resp_lens), 50), "max": max(resp_lens)},
        "frac_prefill_ge_8k": round(frac(ctx_lens, 8192), 4),
        "frac_prefill_ge_16k": round(frac(ctx_lens, 16384), 4),
        "frac_prefill_ge_32k": round(frac(ctx_lens, 32768), 4),
        "frac_prefill_ge_64k": round(frac(ctx_lens, 65536), 4),
        "est_rows_ge_16k": round(n_rows_total * frac(ctx_lens, 16384)),
        "est_tokens_ge_16k": round(n_rows_total * frac(ctx_lens, 16384) * (sum(long_ones) / max(1, len(long_ones)))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="target tokenizer (Qwen3-4B-Thinking-2507)")
    ap.add_argument("--num-rows", type=int, default=300, help="rows sampled per dataset")
    ap.add_argument("--only", nargs="*", help="substring filter on dataset id")
    ap.add_argument("--out", help="write JSON report here")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    results = []
    for dataset, filename, note in JSONL_CANDIDATES:
        if args.only and not any(o.lower() in dataset.lower() or o.lower() in filename.lower() for o in args.only):
            continue
        print(f"\n>>> {dataset}::{filename}", flush=True)
        t0 = time.time()
        try:
            r = measure_jsonl(dataset, filename, tok, args.num_rows, note)
            r.setdefault("dataset", dataset)
            r.setdefault("config", filename)
            r.setdefault("split", "jsonl")
        except Exception as e:  # noqa: BLE001
            r = {"dataset": dataset, "config": filename, "split": "jsonl", "error": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        if "error" in r:
            print(f"    ERROR {r['error']}", flush=True)
        else:
            p = r["prefill_tokens"]
            print(f"    rows~{r['rows_total']} sampled={r['rows_sampled']} | prefill p50={p['p50']} "
                  f"p90={p['p90']} max={p['max']} | >=16K {r['frac_prefill_ge_16k']:.1%} "
                  f">=32K {r['frac_prefill_ge_32k']:.1%}", flush=True)

    for dataset, config, split, note in CANDIDATES:
        if args.only and not any(o.lower() in dataset.lower() for o in args.only):
            continue
        label = f"{dataset}[{config or '*'}/{split or '*'}]"
        print(f"\n>>> {label}", flush=True)
        t0 = time.time()
        try:
            r = measure(dataset, config, split, tok, args.num_rows, note)
            r.setdefault("dataset", dataset)
            r.setdefault("config", config)
            r.setdefault("split", split or "?")
        except Exception as e:  # noqa: BLE001 - one dead dataset must not kill the survey
            r = {"dataset": dataset, "config": config, "split": split, "error": f"{type(e).__name__}: {e}"}
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        if "error" in r:
            print(f"    ERROR {r['error']}", flush=True)
        else:
            p = r["prefill_tokens"]
            print(
                f"    rows={r['rows_total']} sampled={r['rows_sampled']} | prefill p50={p['p50']} "
                f"p90={p['p90']} max={p['max']} | >=16K {r['frac_prefill_ge_16k']:.1%} "
                f">=32K {r['frac_prefill_ge_32k']:.1%} | est rows>=16K {r['est_rows_ge_16k']}",
                flush=True,
            )

    print(f"\n{'dataset':<58} {'rows':>8} {'p50':>7} {'p90':>7} {'>=16K':>7} {'>=32K':>7} {'rows>=16K':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['dataset'] + '/' + str(r.get('config')):<58.57} {'ERROR':>8}  {r['error'][:60]}")
            continue
        print(
            f"{r['dataset'] + '/' + r['split']:<58.57} {r['rows_total']:>8} {r['prefill_tokens']['p50']:>7} "
            f"{r['prefill_tokens']['p90']:>7} {r['frac_prefill_ge_16k']:>6.1%} {r['frac_prefill_ge_32k']:>6.1%} "
            f"{r['est_rows_ge_16k']:>10}"
        )
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"model": args.model, "num_rows": args.num_rows, "results": results}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
