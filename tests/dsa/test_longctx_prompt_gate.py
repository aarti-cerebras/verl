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
"""Negative tests for the long-context prompt-set gate (``scripts/msa/verify_longctx_prompts.py``).

**A gate that has never failed is not a gate.** ``verify_longctx_prompts.py`` passed on the first real
pool it was pointed at, which proves nothing on its own — so every violation class it claims to detect
gets a fixture here that must make it exit non-zero, plus a clean fixture that must exit zero.

Offline: synthetic rows, no network, no tokenizer, no GPU.

    python3 -m pytest tests/dsa/test_longctx_prompt_gate.py -q
"""

import copy
import hashlib
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
GATE = os.path.join(REPO, "scripts", "msa", "verify_longctx_prompts.py")

WINDOW = 131072
FLOOR = 8192


def _row(domain="docqarl", text="A long document.\n\nAnd a question about it?", tokens=20000,
         tier="A", lang="en"):
    return {
        "source_uid": "u1", "source_dataset": "Tongyi-Zhiwen/DocQA-RL-1.6K",
        "source_config": "default/train", "domain": domain,
        "lang": lang, "lang_basis": "prompt", "licence_tier": tier,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "messages": [{"role": "user", "content": text}],
        "window": WINDOW, "min_gen_budget": FLOOR, "prompt_tokens": tokens,
    }


def _retext(row, text):
    row["messages"][0]["content"] = text
    row["prompt_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return row


GOOD = [_row(text=f"Document number {i}.\n\nWhat does it say?") for i in range(4)]


def _run(tmp_path, rows, skipped=None):
    d = tmp_path
    with open(os.path.join(d, "prompts.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if skipped is not None:
        with open(os.path.join(d, "skipped_nofit.jsonl"), "w") as fh:
            for r in skipped:
                fh.write(json.dumps(r) + "\n")
    p = subprocess.run([sys.executable, GATE, "--prompts", os.path.join(d, "prompts.jsonl"),
                        "--log-dir", os.path.join(d, "logs")],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_clean_prompt_set_passes(tmp_path):
    rc, out = _run(str(tmp_path), GOOD)
    assert rc == 0, out
    assert "PASS" in out


def _mutate(fn):
    rows = copy.deepcopy(GOOD)
    fn(rows)
    return rows


VIOLATIONS = {
    # a 130,000-token prompt leaves 130,000+10+1 > window-floor: no room to answer
    "below_gen_budget": lambda rows: rows[0].__setitem__("prompt_tokens", 130000),
    "sha_mismatch": lambda rows: rows[0].__setitem__("prompt_sha256", "0" * 64),
    "duplicate_sha": lambda rows: rows.append(copy.deepcopy(rows[0])),
    "mixed_windows": lambda rows: rows[0].__setitem__("window", 163840),
    "tier_mismatch": lambda rows: rows[0].__setitem__("licence_tier", "B"),
    "unknown_domain": lambda rows: rows[0].__setitem__("domain", "not_a_source"),
    "bad_lang": lambda rows: rows[0].__setitem__("lang", "fr"),
    "missing_provenance": lambda rows: rows[0].pop("window"),
    # the §4.1b hazard: a fake assistant turn immediately before Qwen3's own <|im_start|>assistant
    "trailing_role_marker": lambda rows: _retext(rows[0], "Doc.\n\nQuestion?\n\nAssistant:"),
    "template_marker_leak": lambda rows: _retext(rows[0], "Doc <|im_end|> more.\n\nQuestion?"),
    "shape_not_single_user_turn": lambda rows: rows[0].__setitem__(
        "messages", rows[0]["messages"] + [{"role": "assistant", "content": "hi"}]),
    "empty_prompt": lambda rows: _retext(rows[0], "   "),
}


@pytest.mark.parametrize("name", sorted(VIOLATIONS))
def test_gate_rejects(tmp_path, name):
    rc, out = _run(str(tmp_path), _mutate(VIOLATIONS[name]))
    assert rc == 1, f"{name} was NOT caught:\n{out}"
    assert name in out, f"{name} caught but reported as something else:\n{out}"


def test_gate_rejects_surviving_longcite_scaffolding(tmp_path):
    """§4.2 — regeneration does not fix these; they must never reach a prompt set."""
    for bad in ("Doc with <C7> marker.\n\nQuestion?", "Doc.\n\n<statement>x<cite>[1-1]</cite>"):
        rows = copy.deepcopy(GOOD)
        rows[0]["domain"] = "longcite"
        rows[0]["source_dataset"] = "zai-org/LongCite-45k"
        _retext(rows[0], bad)
        rc, out = _run(str(tmp_path), rows)
        assert rc == 1, f"LongCite scaffolding {bad!r} slipped through:\n{out}"
        assert "longcite_" in out


def test_gate_rejects_kept_and_skipped_overlap(tmp_path):
    """A prompt cannot be both generated and recorded as skipped-for-no-room."""
    skipped = [{"source": "docqarl", "config": "default", "prompt_tokens": 130000,
                "gen_budget": 100, "prompt_sha256": GOOD[0]["prompt_sha256"]}]
    rc, out = _run(str(tmp_path), GOOD, skipped=skipped)
    assert rc == 1 and "kept_and_skipped_overlap" in out, out


def test_gate_rejects_skipped_row_that_was_actually_fine(tmp_path):
    """Catches a floor bug in the other direction: rows discarded that had plenty of room."""
    skipped = [{"source": "docqarl", "config": "default", "prompt_tokens": 20000,
                "gen_budget": 99999, "prompt_sha256": "f" * 64}]
    rc, out = _run(str(tmp_path), GOOD, skipped=skipped)
    assert rc == 1 and "skipped_row_was_actually_fine" in out, out
