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
"""HARD GATE on a long-context ``prompts.jsonl``. Exits non-zero on any violation.

Runbook: ``docs/qwen3_4b_msa/phase2_long_context_gen.md`` §9 test 5. Run after every extraction, before
spending a GPU-hour on generation — a prompt set that violates the window budget produces truncated
traces at full prefill cost, and the failure is invisible until the length histogram comes back wrong.

Checks, all of them cheap and all of them things that have actually gone wrong somewhere in this
pipeline:

* **budget floor**  ``window - (prompt_tokens + wrapper) - 1 >= min_gen_budget`` for every kept row.
* **reconciliation** kept and skipped sets are disjoint; every skipped row really is below the floor.
* **sha integrity**  ``prompt_sha256 == sha256(content)``, and no duplicates.
* **provenance**     one window per file, tier matches the source registry, ``lang``/``lang_basis`` set.
* **shape**          exactly one user message, non-empty, no template/role markers leaked into the text.
* **LongCite**       no surviving ``<C{i}>`` markers or ``<statement>``/``<cite>`` scaffolding (§4.2).
* **ChatQA2**        no Llama-3 BOS, no trailing ``Assistant:`` role marker (§4.1b finding 2).

    python3 scripts/msa/verify_longctx_prompts.py --prompts $RUN/prompts.jsonl
"""

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsa"))
from _dsa_log import setup_logging  # noqa: E402
from longctx_sources import LLAMA3_BOS, SOURCES  # noqa: E402

CITE_RE = re.compile(r"<C_?\d+>")
TRAILING_ROLE_RE = re.compile(r"\n*(Assistant|User)\s*:\s*$")
TEMPLATE_MARKERS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", LLAMA3_BOS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--skipped", default=None, help="default: skipped_nofit.jsonl beside --prompts")
    ap.add_argument("--wrapper-tokens", type=int, default=10)
    ap.add_argument("--max-report", type=int, default=5, help="example violations to print per class")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.prompts)) or "."
    logger, _ = setup_logging("verify_longctx_prompts", args.log_dir or os.path.join(out_dir, "logs"))

    fail = {}       # class -> [examples]
    counts = {}

    def bad(cls, detail):
        counts[cls] = counts.get(cls, 0) + 1
        fail.setdefault(cls, [])
        if len(fail[cls]) < args.max_report:
            fail[cls].append(detail)

    shas, windows, n = set(), set(), 0
    for lineno, line in enumerate(open(args.prompts), 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        r = json.loads(line)
        dom = r.get("domain")
        msgs = r.get("messages") or []
        text = msgs[0].get("content", "") if msgs else ""

        if len(msgs) != 1 or msgs[0].get("role") != "user":
            bad("shape_not_single_user_turn", f"line {lineno} domain={dom} n_msgs={len(msgs)}")
        if not text.strip():
            bad("empty_prompt", f"line {lineno} domain={dom}")

        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha != r.get("prompt_sha256"):
            bad("sha_mismatch", f"line {lineno} domain={dom}")
        if sha in shas:
            bad("duplicate_sha", f"line {lineno} domain={dom} sha={sha[:12]}")
        shas.add(sha)

        w = r.get("window")
        windows.add(w)
        ntok, floor = r.get("prompt_tokens"), r.get("min_gen_budget")
        if ntok is None or w is None or floor is None:
            bad("missing_provenance", f"line {lineno} domain={dom} window={w} tok={ntok} floor={floor}")
        else:
            budget = w - (ntok + args.wrapper_tokens) - 1
            if budget < floor:
                bad("below_gen_budget", f"line {lineno} domain={dom} tok={ntok} budget={budget} < {floor}")

        spec = SOURCES.get(dom)
        if spec is None:
            bad("unknown_domain", f"line {lineno} domain={dom!r}")
        elif r.get("licence_tier") != spec["tier"]:
            bad("tier_mismatch", f"line {lineno} domain={dom} tier={r.get('licence_tier')} != {spec['tier']}")
        if r.get("lang") not in ("en", "zh"):
            bad("bad_lang", f"line {lineno} domain={dom} lang={r.get('lang')!r}")
        if r.get("lang_basis") not in ("question", "prompt"):
            bad("bad_lang_basis", f"line {lineno} domain={dom}")

        for m in TEMPLATE_MARKERS:
            if m in text:
                bad("template_marker_leak", f"line {lineno} domain={dom} marker={m}")
                break
        if TRAILING_ROLE_RE.search(text):
            bad("trailing_role_marker", f"line {lineno} domain={dom} tail={text[-40:]!r}")

        if dom == "longcite":
            if CITE_RE.search(text):
                bad("longcite_marker_survived", f"line {lineno}")
            if "<statement>" in text or "<cite>" in text:
                bad("longcite_scaffolding_survived", f"line {lineno}")

    skipped_path = args.skipped or os.path.join(out_dir, "skipped_nofit.jsonl")
    n_skip, skip_shas = 0, set()
    if os.path.exists(skipped_path):
        for line in open(skipped_path):
            line = line.strip()
            if not line:
                continue
            n_skip += 1
            s = json.loads(line)
            skip_shas.add(s.get("prompt_sha256"))
            if s.get("gen_budget") is not None and s["gen_budget"] >= 8192:
                bad("skipped_row_was_actually_fine",
                    f"sha={str(s.get('prompt_sha256'))[:12]} budget={s['gen_budget']}")
    overlap = shas & skip_shas
    if overlap:
        bad("kept_and_skipped_overlap", f"{len(overlap)} sha(s) in both files")

    # A row with no `window` puts None in this set; sorting it against ints raises TypeError and the
    # gate would die with a traceback BEFORE printing the violation report — i.e. a real
    # missing-provenance pool would look like a crashed tool rather than a failed gate.
    real_windows = sorted(w for w in windows if w is not None)
    shown = real_windows + (["<missing>"] if None in windows else [])
    logger.info("rows=%d  distinct_sha=%d  skipped=%d  window(s)=%s", n, len(shas), n_skip, shown)
    if len(real_windows) > 1:
        bad("mixed_windows", f"one file must carry ONE window, found {real_windows}")

    if fail:
        logger.error("FAILED — %d violation class(es):", len(fail))
        for cls in sorted(fail):
            logger.error("  %-32s %d occurrence(s)", cls, counts[cls])
            for ex in fail[cls]:
                logger.error("      %s", ex)
        sys.exit(1)
    logger.info("PASS — all §9 test-5 invariants hold over %d prompts", n)


if __name__ == "__main__":
    main()
