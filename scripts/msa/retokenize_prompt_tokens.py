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
"""Re-tokenize an existing ``prompts.jsonl`` for a different generator model.

Phase-2 prompt banks are **generator-independent** -- ``prompt_sha256`` is sha256 of the cleaned prompt
text, so the same bank is reused across models and that is what makes a shared train/val split possible
(``split_bc_val.py --val-sha-file``). The only tokenizer-dependent column is ``prompt_tokens``, which is
stale the moment the file is handed to a different model.

That column is *inert* at generation time -- ``gen_trajectories.py --fit-window`` recomputes the served
prefix live from whichever tokenizer is loaded -- but leaving a wrong number in an artifact is how wrong
numbers end up in a report. This script rewrites it, and adds ``prefix_tokens``: the length of the FULL
templated prefix (chat wrapper + prompt), which is what the generation budget is actually computed from.

What it does NOT do is re-run selection. Rows, order, and every other field are copied through untouched,
and the prompt_sha256 multiset is asserted identical -- so a re-tokenized bank cannot silently become a
different prompt set.

Example:
    python3 scripts/msa/retokenize_prompt_tokens.py \
      --src  <qwen3 run>/prompts.jsonl \
      --out  <gpt-oss run>/prompts.jsonl \
      --tokenizer /cb/ml-eng/aarti/models/gpt-oss-20b \
      --reasoning-effort medium --pin-date 2026-08-07 --window 32768
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsa"))
from _dsa_log import setup_logging  # noqa: E402
from _dsa_tok import chat_prefix_ids, pin_template_kwargs  # noqa: E402


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _pc(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="target model dir")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"])
    ap.add_argument("--pin-date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--chat-template-kwargs", default=None)
    ap.add_argument("--window", type=int, default=32768, help="report the per-row generation budget")
    ap.add_argument("--batch-size", type=int, default=2000)
    a = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    log, log_path = setup_logging("retokenize_prompt_tokens", os.path.join(out_dir, "logs"))
    log.info("args: %s", vars(a))
    t0 = time.time()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    extra = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else {}
    if a.reasoning_effort:
        extra["reasoning_effort"] = a.reasoning_effort
    tpl = pin_template_kwargs(extra, a.pin_date)
    log.info("template kwargs: %s (date pinned=%s)",
             {k: v for k, v in tpl.items() if k != "strftime_now"}, a.pin_date)

    rows = [json.loads(ln) for ln in open(a.src) if ln.strip()]
    log.info("read %d rows from %s", len(rows), a.src)
    assert rows, "empty --src"
    for r in rows:
        assert len(r["messages"]) == 1 and r["messages"][0]["role"] == "user", \
            f"expected single-user-turn rows, got {[m['role'] for m in r['messages']]}"

    # Fixed chat wrapper, measured once: prefix(prompt) - tokens(prompt) is constant for a single-user-turn
    # template, so the full prefix length is derivable without templating all 93,889 rows.
    wrap = len(chat_prefix_ids(tok, [{"role": "user", "content": ""}], **tpl)) \
        - len(tok("", add_special_tokens=False)["input_ids"])
    log.info("fixed chat wrapper = %d tokens", wrap)

    texts = [r["messages"][0]["content"] for r in rows]
    old = [r.get("prompt_tokens") for r in rows]
    new = []
    for s in range(0, len(texts), a.batch_size):
        new.extend(len(e) for e in tok(texts[s: s + a.batch_size], add_special_tokens=False)["input_ids"])
        if (s // a.batch_size) % 10 == 0:
            log.info("  tokenized %d/%d", min(s + a.batch_size, len(texts)), len(texts))
    assert len(new) == len(rows)

    # Spot-check the derived prefix length against a real template render, so a template whose wrapper is
    # NOT prompt-independent can never pass silently.
    for i in (0, len(rows) // 2, len(rows) - 1):
        real = len(chat_prefix_ids(tok, rows[i]["messages"], **tpl))
        assert real == wrap + new[i], f"row {i}: templated prefix {real} != wrapper {wrap} + prompt {new[i]}"
    log.info("prefix-length derivation verified on 3 rows")

    budget = [a.window - (wrap + n) - 1 for n in new]
    for r, n in zip(rows, new, strict=True):
        r["prompt_tokens"] = n
        r["prefix_tokens"] = wrap + n

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # The guarantee this script exists to make: same prompts, same order.
    src_sha = [json.loads(ln)["prompt_sha256"] for ln in open(a.src) if ln.strip()]
    out_sha = [json.loads(ln)["prompt_sha256"] for ln in open(a.out) if ln.strip()]
    assert src_sha == out_sha, "prompt_sha256 sequence changed -- this is a re-tokenize, not a re-select"
    log.info("prompt_sha256 sequence identical to source (%d rows, order preserved)", len(out_sha))

    by = {}
    for r, n in zip(rows, new, strict=True):
        by.setdefault(r.get("domain"), []).append(n)
    log.info("prompt_tokens  p50=%d p90=%d p99=%d max=%d   (was p50=%s max=%s)",
             _pc(new, 50), _pc(new, 90), _pc(new, 99), max(new),
             _pc([x for x in old if x is not None], 50) if any(old) else "n/a",
             max(x for x in old if x is not None) if any(old) else "n/a")
    for d in sorted(by):
        log.info("  %-9s n=%6d p50=%5d p99=%5d max=%5d", d, len(by[d]), _pc(by[d], 50), _pc(by[d], 99),
                 max(by[d]))
    log.info("generation budget @ window=%d: p50=%d min=%d | under 8192: %d | <=0: %d",
             a.window, _pc(budget, 50), min(budget), sum(b < 8192 for b in budget), sum(b <= 0 for b in budget))
    assert min(budget) > 0, "some prompt leaves no room in the window"

    manifest = dict(
        kind="msa_phase2_prompts_retokenized", source=os.path.abspath(a.src), tokenizer=a.tokenizer,
        reasoning_effort=a.reasoning_effort, pin_date=a.pin_date, window=a.window,
        n_rows=len(rows), chat_wrapper_tokens=wrap,
        prompt_tokens={f"p{q}": _pc(new, q) for q in (50, 90, 99)} | {"max": max(new)},
        budget_p50=_pc(budget, 50), budget_min=min(budget),
        prompt_sha256_identical_to_source=True,
        out_sha256=hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
        git_commit=_git("rev-parse", "HEAD"), invocation=" ".join([sys.executable] + sys.argv),
        host=os.uname().nodename, elapsed_s=round(time.time() - t0, 1),
        log_file=os.path.basename(log_path),
    )
    with open(a.out + ".MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1)
    log.info("DONE in %.1f min -> %s", (time.time() - t0) / 60, a.out)


if __name__ == "__main__":
    sys.exit(main())
