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
"""Standing regression test for the gpt-oss (harmony) Phase-2 splice contract.

The Qwen3 analogue of this file exists because the chat template silently deletes reasoning traces; the
harmony template has the same hazard (it only renders the analysis channel when the final turn is an
assistant turn and ``add_generation_prompt`` is false, and it raises outright if you put channel tags in
``content``). So here too the production path is a SPLICE of the served prefix with the sampled
completion token ids, and this test is what keeps that path honest.

See docs/gpt_oss_20b_msa/phase2_data_gen.md §6/§7.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "dsa"))

from _dsa_tok import chat_prefix_ids, pin_template_kwargs  # noqa: E402
from trajectories_to_sft_parquet import (  # noqa: E402
    _harmony_channels,
    _harmony_markers,
    _malformed_harmony,
    to_input_ids_rows,
)

MODEL = os.environ.get("GPT_OSS_20B", "/cb/ml-eng/aarti/models/gpt-oss-20b")
PIN_DATE = "2026-01-15"
EFFORT = "medium"
WINDOW = 32768

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL, "tokenizer_config.json")),
    reason=f"gpt-oss-20b tokenizer not found at {MODEL} (set $GPT_OSS_20B)",
)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def mk(tok):
    return _harmony_markers(tok)


@pytest.fixture(scope="module")
def tpl():
    return pin_template_kwargs({"reasoning_effort": EFFORT}, PIN_DATE)


class _Args:
    """Stand-in for the argparse namespace that to_input_ids_rows consumes."""

    tokenizer = MODEL
    chat_format = "harmony"
    reasoning_effort = EFFORT
    pin_date = PIN_DATE
    chat_template_kwargs = None
    keep_truncated = False
    max_length = WINDOW
    max_repetition_frac = 0.30


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def _completion(tok, mk, cot="Let me add two and two. That is 4.", answer="The answer is **4**."):
    """Token ids of a well-formed harmony completion, as vLLM would return them."""
    return (
        [mk["<|channel|>"]] + _ids(tok, "analysis") + [mk["<|message|>"]] + _ids(tok, cot)
        + [mk["<|end|>"], mk["<|start|>"]] + _ids(tok, "assistant")
        + [mk["<|channel|>"]] + _ids(tok, "final") + [mk["<|message|>"]] + _ids(tok, answer)
        + [mk["<|return|>"]]
    )


def _row(tok, tpl, resp_ids, finish="stop"):
    msgs = [{"role": "user", "content": "2+2?"}]
    pre = chat_prefix_ids(tok, msgs, **tpl)
    return {
        "messages": msgs, "prefix_token_ids": pre, "prefix_tokens": len(pre),
        "resp_token_ids": resp_ids, "finish_reason": finish, "domain": "Math", "lang": "en",
        "source_uid": "t-1", "source_config": "c", "original_dataset": "d",
        "prompt_sha256": "0" * 64, "sample_idx": 0,
    }


# --------------------------------------------------------------------------------------------------
# The prefix: pinned, deterministic, and ending where generation begins.
# --------------------------------------------------------------------------------------------------

def test_prefix_is_date_pinned_and_deterministic(tok, tpl):
    a = chat_prefix_ids(tok, [{"role": "user", "content": "2+2?"}], **tpl)
    b = chat_prefix_ids(tok, [{"role": "user", "content": "2+2?"}], **tpl)
    assert a == b
    text = tok.decode(a)
    # The whole reason pin_template_kwargs exists: without it this line carries the wall-clock day.
    assert f"Current date: {PIN_DATE}" in text
    assert f"Reasoning: {EFFORT}" in text
    assert text.endswith("<|start|>assistant"), text[-40:]


def test_reasoning_effort_reaches_the_prefix(tok):
    seen = {}
    for eff in ("low", "medium", "high"):
        ids = chat_prefix_ids(tok, [{"role": "user", "content": "hi"}],
                              **pin_template_kwargs({"reasoning_effort": eff}, PIN_DATE))
        assert f"Reasoning: {eff}" in tok.decode(ids)
        seen[eff] = ids
    assert seen["low"] != seen["high"], "effort must change the served prefix"


# --------------------------------------------------------------------------------------------------
# The splice: mask boundary, terminator, window.
# --------------------------------------------------------------------------------------------------

def test_splice_mask_and_terminator(tok, mk, tpl, caplog):
    resp = _completion(tok, mk)
    rows, counters = to_input_ids_rows([_row(tok, tpl, resp)], _Args(), _logger())
    assert counters.get("kept") == 1, counters
    assert not counters.get("prefix_rebuild_differs"), "rebuilt prefix disagrees with the served prefix"
    r = rows[0]
    pre_len = r["prefix_tokens"]

    assert r["input_ids"][:pre_len] == chat_prefix_ids(tok, [{"role": "user", "content": "2+2?"}], **tpl)
    assert r["input_ids"][pre_len:] == resp, "the trace must be the sampled tokens, verbatim"
    assert set(r["loss_mask"][:pre_len]) == {0}, "prompt must not be trained on"
    assert set(r["loss_mask"][pre_len:]) == {1}, "the whole trace + answer must be trained on"
    assert len(r["loss_mask"]) == len(r["input_ids"]) == r["length"] <= WINDOW
    assert r["input_ids"][-1] == mk["<|return|>"], "harmony rows terminate on <|return|>, not <|end|>"
    # Qwen3's <|im_end|> must not leak in as the terminator.
    assert r["input_ids"][-1] != 151645


def test_terminator_appended_when_sample_lacks_one(tok, mk, tpl):
    resp = _completion(tok, mk)[:-1]  # model stopped without emitting <|return|>
    rows, counters = to_input_ids_rows([_row(tok, tpl, resp)], _Args(), _logger())
    assert counters.get("kept") == 1, counters
    assert rows[0]["input_ids"][-1] == mk["<|return|>"]
    assert rows[0]["loss_mask"][-1] == 1


def test_channels_parse(tok, mk):
    chans = _harmony_channels(_completion(tok, mk), mk, tok)
    assert [c for c, _ in chans] == ["analysis", "final"]


# --------------------------------------------------------------------------------------------------
# §7 health filters. The first of these is the one that matters most: the Qwen3 rules reject every
# harmony row, because a legal harmony completion contains an internal <|start|>assistant.
# --------------------------------------------------------------------------------------------------

def test_wellformed_row_is_not_malformed(tok, mk):
    assert _malformed_harmony(_completion(tok, mk), _Args(), mk, tok, {}) is None


def test_qwen3_filters_would_have_rejected_it(tok, mk):
    """Guards the regression that motivated --chat-format: <|start|> is legal mid-response on harmony."""
    from trajectories_to_sft_parquet import _malformed

    resp = _completion(tok, mk)
    assert mk["<|start|>"] in resp, "a legal harmony completion contains an internal <|start|>assistant"
    assert _malformed(resp, _Args()) is not None, "the qwen3-think path is expected to reject this row"


@pytest.mark.parametrize("case", ["no_final_channel", "multiple_final", "empty_answer",
                                  "tool_call_leak", "stub", "no_channel_header", "commentary_channel"])
def test_malformed_classes(tok, mk, case):
    a = _Args()
    if case == "stub":
        resp = [mk["<|channel|>"]]
    elif case == "no_channel_header":
        resp = _ids(tok, "just some prose with no channel header at all, at length")
    elif case == "no_final_channel":
        resp = ([mk["<|channel|>"]] + _ids(tok, "analysis") + [mk["<|message|>"]]
                + _ids(tok, "thinking and then stopping") + [mk["<|end|>"]])
    elif case == "multiple_final":
        resp = _completion(tok, mk)[:-1] + ([mk["<|start|>"]] + _ids(tok, "assistant")
                                            + [mk["<|channel|>"]] + _ids(tok, "final")
                                            + [mk["<|message|>"]] + _ids(tok, "again") + [mk["<|return|>"]])
    elif case == "empty_answer":
        resp = (_completion(tok, mk, answer="")[:-1] + [mk["<|return|>"]])
    elif case == "tool_call_leak":
        resp = _completion(tok, mk)[:-1] + [mk["<|call|>"]]
    else:  # commentary_channel
        resp = ([mk["<|channel|>"]] + _ids(tok, "commentary") + [mk["<|message|>"]]
                + _ids(tok, "let me call a tool here") + [mk["<|end|>"]] + _completion(tok, mk))
    assert _malformed_harmony(resp, a, mk, tok, {}) == case


def test_repetition_loop(tok, mk):
    loop = _ids(tok, "the same sentence over and over. ") * 40
    resp = _completion(tok, mk, cot=tok.decode(loop))
    assert _malformed_harmony(resp, _Args(), mk, tok, {}) == "repetition_loop"


def test_truncated_dropped_by_default(tok, mk, tpl):
    rows, counters = to_input_ids_rows([_row(tok, tpl, _completion(tok, mk), finish="length")],
                                       _Args(), _logger())
    assert rows == [] and counters.get("truncated") == 1, counters


# --------------------------------------------------------------------------------------------------
# verify_sft_parquet is the independent second opinion on the converter's output -- it must accept what
# the converter emits and reject what it should have caught.
# --------------------------------------------------------------------------------------------------

def test_verifier_accepts_converter_output(tok, mk, tpl):
    from verify_sft_parquet import check_row_harmony

    rows, _ = to_input_ids_rows([_row(tok, tpl, _completion(tok, mk))], _Args(), _logger())
    r = rows[0]
    bad = check_row_harmony(r["input_ids"], r["loss_mask"], r["prefix_tokens"], WINDOW, tok, {})
    assert bad == [], bad


def test_verifier_catches_qwen3_terminator(tok, mk, tpl):
    """A row terminated with <|end|> instead of <|return|> must not pass."""
    from verify_sft_parquet import check_row_harmony

    rows, _ = to_input_ids_rows([_row(tok, tpl, _completion(tok, mk))], _Args(), _logger())
    r = rows[0]
    ids = r["input_ids"][:-1] + [mk["<|end|>"]]
    assert "no_terminal_return" in check_row_harmony(ids, r["loss_mask"], r["prefix_tokens"], WINDOW, tok, {})


def test_verifier_catches_bad_mask_boundary(tok, mk, tpl):
    from verify_sft_parquet import check_row_harmony

    rows, _ = to_input_ids_rows([_row(tok, tpl, _completion(tok, mk))], _Args(), _logger())
    r = rows[0]
    mask = list(r["loss_mask"])
    mask[r["prefix_tokens"] - 1] = 1  # train on one token of the prompt
    bad = check_row_harmony(r["input_ids"], mask, r["prefix_tokens"], WINDOW, tok, {})
    assert "mask_nonzero_over_prefix" in bad


def test_qwen3_verifier_would_have_failed_every_harmony_row(tok, mk, tpl):
    """The gap this flag closes: verify_sft_parquet is a hard gate that exits non-zero."""
    from verify_sft_parquet import check_row

    rows, _ = to_input_ids_rows([_row(tok, tpl, _completion(tok, mk))], _Args(), _logger())
    r = rows[0]
    bad = check_row(r["input_ids"], r["loss_mask"], r["prefix_tokens"], WINDOW)
    assert "prefix_not_ending_in_think_nl" in bad and "no_terminal_im_end" in bad, bad


def test_markers_resolve_to_the_known_ids(tok):
    assert _harmony_markers(tok) == {"<|return|>": 200002, "<|constrain|>": 200003, "<|channel|>": 200005,
                                     "<|start|>": 200006, "<|end|>": 200007, "<|message|>": 200008,
                                     "<|call|>": 200012}


def _logger():
    import logging

    return logging.getLogger("test_harmony")
