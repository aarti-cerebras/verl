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
"""Standing regression test for the Qwen3-4B-Thinking Phase-2 splice contract.

Listed as a to-write item in ``docs/qwen3_4b_msa/phase2_data_gen.md`` §10 since 2026-07-30 and never
written; required by ``phase2_long_context_gen.md`` §9 tests 6-7 before the long-context pilot.

**Why the production path is a splice, not a re-render.** Qwen3's chat template *deletes reasoning
traces* from any assistant turn that is not preceded by a user turn in the same render, and
``MultiTurnSFTDataset`` templates one message at a time — so routing BC data through it would train on
the final answer only, silently discarding >95% of the tokens and the entire capability being
preserved. ``test_template_deletes_the_trace_when_rendered_alone`` pins that landmine so the reason for
the splice cannot quietly stop being true.

The tests run at a small window by default and at the real 131,072 window where it matters (the long
prompt case is what the 32K-era pipeline never exercised).

    python3 -m pytest tests/dsa/test_qwen3_thinking_sft_roundtrip.py -q
"""

import argparse
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "dsa"))

from _dsa_tok import chat_prefix_ids  # noqa: E402
from trajectories_to_sft_parquet import (  # noqa: E402
    TOK_IM_END,
    TOK_IM_START,
    TOK_THINK_CLOSE,
    TOK_THINK_OPEN,
    _malformed,
    to_input_ids_rows,
)

MODEL = os.environ.get("QWEN3_4B_THINKING", "/home/aarti_cerebras/models/qwen3_4b_thinking_2507")
WINDOW = 131072

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL, "tokenizer_config.json")),
    reason=f"Qwen3-4B-Thinking-2507 tokenizer not found at {MODEL} (set $QWEN3_4B_THINKING)",
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def _args(**kw):
    base = dict(tokenizer=MODEL, chat_format="qwen3", chat_template_kwargs=None, pin_date=None,
                reasoning_effort=None, keep_truncated=False, max_length=WINDOW,
                max_repetition_frac=0.30)
    base.update(kw)
    return argparse.Namespace(**base)


TRACE = ("Let me work through this carefully. The document describes several points, and the "
         "question asks about the second one, so I should re-read that section before answering.")
ANSWER = "The second sentence adds supporting detail about the subject under discussion."


def _row(tok, prompt, trace=TRACE, answer=ANSWER, finish="stop"):
    """A trajectory row shaped exactly as gen_trajectories.py writes it.

    The trace/answer defaults are deliberately realistic: `_malformed` rejects anything under 8 tokens
    as a stub *before* any other check, so a toy fixture silently exercises the stub path instead of
    the one under test. The assert below turns that into an obvious fixture error.
    """
    messages = [{"role": "user", "content": prompt}]
    prefix = chat_prefix_ids(tok, messages)
    body = f"{trace}\n</think>\n\n{answer}"
    resp_ids = tok(body, add_special_tokens=False)["input_ids"]
    assert len(resp_ids) >= 8, f"fixture response is only {len(resp_ids)} tokens — _malformed calls it a stub"
    return {
        "messages": messages + [{"role": "assistant", "content": body}],
        "prefix_token_ids": list(prefix), "prefix_tokens": len(prefix),
        "resp_token_ids": list(resp_ids), "finish_reason": finish,
        "domain": "longcite", "prompt_sha256": "0" * 64, "source_uid": "t",
    }, list(prefix), list(resp_ids)


LOG = logging.getLogger("test")


# ---------------------------------------------------------------------------------------------------
# Marker ids and the wrapper arithmetic the window budget depends on
# ---------------------------------------------------------------------------------------------------
def test_marker_token_ids_match_the_documented_values(tok):
    """§5.1 quotes these ids; the per-row generation budget is computed from the wrapper they form."""
    assert tok.convert_tokens_to_ids("<|im_start|>") == TOK_IM_START == 151644
    assert tok.convert_tokens_to_ids("<|im_end|>") == TOK_IM_END == 151645
    assert tok.convert_tokens_to_ids("<think>") == TOK_THINK_OPEN == 151667
    assert tok.convert_tokens_to_ids("</think>") == TOK_THINK_CLOSE == 151668


def test_generation_prefix_is_ten_tokens_plus_the_prompt(tok):
    """The wrapper is 10 tokens and the closing <|im_end|> is 1 (§5.1). Code computes this live, but a
    silent template change would move every row's budget, so pin the documented value."""
    body = tok("X", add_special_tokens=False)["input_ids"]
    prefix = chat_prefix_ids(tok, [{"role": "user", "content": "X"}])
    assert len(prefix) - len(body) == 10
    # ... and the prefix must end INSIDE the think block: <think> then newline.
    assert prefix[-2:] == [TOK_THINK_OPEN, 198]


def test_template_deletes_the_trace_when_rendered_alone(tok):
    """THE LANDMINE (phase2_data_gen.md §6). Rendering the assistant turn on its own — which is exactly
    what MultiTurnSFTDataset._process_single_message does — drops the reasoning. If this test ever
    fails, the template changed and the splice rationale must be re-derived, not assumed."""
    asst = {"role": "assistant", "content": "<think>\nsecret reasoning\n</think>\n\nFinal answer: 42"}
    alone = tok.apply_chat_template([asst], tokenize=False)
    assert "secret reasoning" not in alone, "template no longer strips the trace — re-check §6"
    assert "Final answer: 42" in alone
    # Rendered WITH its user turn, the trace survives — the asymmetry is the whole point.
    both = tok.apply_chat_template([{"role": "user", "content": "q"}, asst], tokenize=False)
    assert "secret reasoning" in both


# ---------------------------------------------------------------------------------------------------
# The splice contract
# ---------------------------------------------------------------------------------------------------
def test_splice_is_exactly_prefix_plus_completion_plus_terminator(tok):
    row, prefix, resp = _row(tok, "Summarize the document.")
    out, counters = to_input_ids_rows([row], _args(), LOG)
    assert len(out) == 1, counters
    ids = list(out[0]["input_ids"])
    assert ids == prefix + resp + [TOK_IM_END]
    assert counters.get("prefix_rebuild_differs", 0) == 0, "rebuilt prefix != served prefix"


def test_loss_mask_covers_the_whole_trace_and_none_of_the_prompt(tok):
    row, prefix, resp = _row(tok, "Summarize the document.")
    out, _ = to_input_ids_rows([row], _args(), LOG)
    ids, mask = list(out[0]["input_ids"]), list(out[0]["loss_mask"])
    assert len(mask) == len(ids)
    assert set(mask[: len(prefix)]) == {0}, "prompt tokens are being trained on"
    assert set(mask[len(prefix):]) == {1}, "part of the trace is masked out"
    assert sum(mask) == len(resp) + 1  # completion + <|im_end|>


def test_think_markers_survive_into_the_spliced_ids(tok):
    row, prefix, _ = _row(tok, "Summarize the document.")
    out, _ = to_input_ids_rows([row], _args(), LOG)
    ids = list(out[0]["input_ids"])
    # exactly one <think> (in the prefix, opened by the template) and one </think> (in the completion)
    assert ids.count(TOK_THINK_OPEN) == 1
    assert ids.count(TOK_THINK_CLOSE) == 1
    assert ids.index(TOK_THINK_OPEN) < len(prefix) <= ids.index(TOK_THINK_CLOSE)
    assert ids[-1] == TOK_IM_END


def test_decoded_text_round_trips_the_reasoning_block(tok):
    row, _, _ = _row(tok, "q", trace="step one considers the premise\nstep two checks the corollary",
                     answer="final answer: the corollary holds")
    out, _ = to_input_ids_rows([row], _args(), LOG)
    text = tok.decode(list(out[0]["input_ids"]))
    assert "step one considers the premise" in text and "</think>" in text and "final answer" in text


def test_truncated_rows_are_dropped_unless_keep_truncated(tok):
    row, _, _ = _row(tok, "Summarize the document.", finish="length")
    out, _ = to_input_ids_rows([row], _args(), LOG)
    assert out == []
    out2, _ = to_input_ids_rows([row], _args(keep_truncated=True), LOG)
    assert len(out2) == 1


# ---------------------------------------------------------------------------------------------------
# Long-input behaviour — what the 32K-era pipeline never exercised
# ---------------------------------------------------------------------------------------------------
def test_template_has_no_length_dependent_behaviour_at_130k(tok):
    """The wrapper must still be 10 + 1 tokens for a ~130K-token prompt, or §5.1's per-row budget is
    wrong precisely where the window is tightest."""
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 12000
    body = tok(long_prompt, add_special_tokens=False)["input_ids"]
    assert len(body) > 100_000, f"fixture too short ({len(body)} tokens)"
    prefix = chat_prefix_ids(tok, [{"role": "user", "content": long_prompt}])
    assert len(prefix) - len(body) == 10
    assert prefix[-2:] == [TOK_THINK_OPEN, 198]


def test_splice_holds_and_stays_in_window_for_a_long_prompt(tok):
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 12000
    row, prefix, resp = _row(tok, long_prompt)
    out, counters = to_input_ids_rows([row], _args(), LOG)
    assert len(out) == 1, counters
    ids, mask = list(out[0]["input_ids"]), list(out[0]["loss_mask"])
    assert ids == prefix + resp + [TOK_IM_END]
    assert len(ids) <= WINDOW
    assert set(mask[: len(prefix)]) == {0} and set(mask[len(prefix):]) == {1}
    assert counters.get("prefix_rebuild_differs", 0) == 0


def test_chat_prefix_ids_returns_flat_ints_not_a_batchencoding(tok):
    """transformers 5.3 changed apply_chat_template(tokenize=True) to return a BatchEncoding; the old
    unwrap produced len==2 instead of the real length, which let a sequence reach window+1
    (phase2_data_gen.md §11.2). chat_prefix_ids exists to absorb that."""
    prefix = chat_prefix_ids(tok, [{"role": "user", "content": "hello"}])
    assert isinstance(prefix, list) and len(prefix) > 5
    assert all(isinstance(t, int) for t in prefix)


# ---------------------------------------------------------------------------------------------------
# §7 malformed-response classes, on token ids
# ---------------------------------------------------------------------------------------------------
def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def test_clean_response_is_not_flagged(tok):
    assert _malformed(_ids(tok, f"{TRACE}\n</think>\n\n{ANSWER}"), _args()) is None


@pytest.mark.parametrize("text,expected", [
    (TRACE + " and I never close the reasoning block at all",            "no_think_close"),
    (TRACE + "\n</think>\n\nfirst answer\n</think>\n\nsecond answer",  "multiple_think_close"),
    (TRACE + "\n<think>\nrestarted reasoning\n</think>\n\nanswer",      "reopened_think"),
    (TRACE + "\n</think>\n\n",                                          "empty_answer"),
])
def test_malformed_classes_are_detected(tok, text, expected):
    assert _malformed(_ids(tok, text), _args()) == expected


def test_role_marker_leak_is_detected(tok):
    ids = _ids(tok, f"{TRACE}\\n</think>\\n\\n{ANSWER}")
    assert _malformed(ids + [TOK_IM_START], _args()) == "role_marker_leak"
    assert _malformed([TOK_IM_END] + ids, _args()) == "role_marker_leak"
    # a terminal <|im_end|> is legitimate, not a leak
    assert _malformed(ids + [TOK_IM_END], _args()) is None


def test_stub_response_is_detected():
    assert _malformed([1, 2, 3], _args()) == "stub"


def test_repetition_loop_is_detected(tok):
    """The classic long-trace degeneracy. Thresholds were tuned on <=32K traces; runbook §6.3 flags
    re-validation at 100K, and this test is where a retune gets pinned."""
    looped = _ids(tok, "I need to check this again. " * 400 + "\n</think>\n\nanswer")
    assert _malformed(looped, _args()) == "repetition_loop"
    # a long but non-repetitive trace must NOT trip it
    varied = _ids(tok, " ".join(f"step {i} considers a different sub-problem." for i in range(400))
                  + "\n</think>\n\nanswer")
    assert _malformed(varied, _args()) is None
