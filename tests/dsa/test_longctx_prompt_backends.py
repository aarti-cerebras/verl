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
"""Offline tests for the long-context prompt backends (docs/qwen3_4b_msa/phase2_long_context_gen.md §9 A).

No network, no GPU, no tokenizer. Every fixture below is a **reduced copy of a real row** fetched from
the Hub on 2026-08-10; the shapes (and the traps) are measured, not invented.

    python3 -m pytest tests/dsa/test_longctx_prompt_backends.py -q
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "dsa"))

import longctx_sources as lcs  # noqa: E402


def _counters():
    return lcs._new_counters()


# ---------------------------------------------------------------------------------------------------
# LongCite — the only source with real preprocessing (runbook §4.2)
# ---------------------------------------------------------------------------------------------------
LONGCITE_PREAMBLE = (
    "Please answer the user's question based on the following document. When a sentence S in your "
    "response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-"
    '<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>'
    '[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the '
    "user's question.\n\n"
)
LONGCITE_ROW = {
    "prompt": LONGCITE_PREAMBLE
    + "[Document Start]\n<C0>The first sentence of the document is here and it is reasonably long.\n"
    "<C1>The second sentence adds more detail about the subject matter under discussion.\n"
    "<C2>A third sentence rounds out the document with a concluding thought for the reader.\n"
    "[Document End]\n\nWhat does the second sentence add?",
    "response": "<statement>It adds detail.<cite>[1-1]</cite></statement>",
}


def test_longcite_strips_preamble_and_markers_but_keeps_content():
    rec, reason = lcs.extract_longcite(LONGCITE_ROW, _counters(), "default")
    assert reason is None
    text = rec["text"]
    # the citation TASK is gone ...
    assert not re.search(r"<C_?\d+>", text), "chunk markers survived"
    assert "<statement>" not in text and "<cite>" not in text
    assert "Please answer the user's question based on" not in text
    # ... but the document and the question are not
    assert "The first sentence of the document" in text
    assert "The second sentence adds more detail" in text
    assert "A third sentence rounds out the document" in text
    assert text.rstrip().endswith("What does the second sentence add?")
    # the structural delimiters are deliberately KEPT (runbook §4.2)
    assert lcs.DOC_START in text and lcs.DOC_END in text
    assert rec["question"] == "What does the second sentence add?"


def test_longcite_rejects_when_stripping_would_eat_the_document():
    """Tripwire for an over-eager stripper: only a removal share so large that the regex must be
    deleting content (>60%) rejects the row."""
    row = {"prompt": LONGCITE_PREAMBLE + "[Document Start]\n" + "<C1>" * 500 + "x\n[Document End]\n\nQ?"}
    c = _counters()
    rec, reason = lcs.extract_longcite(row, c, "default")
    assert rec is None and reason == "cite_strip_out_of_band"
    assert c["cite_strip_out_of_band"] == 1


def test_longcite_keeps_sparsely_chunked_documents():
    """A 1-marker-in-49K-chars document is a form or a name list that LongCite's sentence chunker
    barely segmented — legitimate data. An earlier lower bound of 0.001 rejected 348 real rows."""
    body = "Name and address record line.\n" * 2000
    row = {"prompt": LONGCITE_PREAMBLE + f"[Document Start]\n<C0>{body}\n[Document End]\n\nWho is listed?"}
    c = _counters()
    rec, reason = lcs.extract_longcite(row, c, "default")
    assert reason is None and rec is not None
    assert "Name and address record line." in rec["text"]
    assert c["cite_low_marker_density"] == 1, "should be COUNTED as a diagnostic, not rejected"


def test_longcite_keeps_table_of_contents_documents():
    """ToC pages chunk every dot of a `. . . . .` leader as its own sentence, reaching ~29% markers.
    Stripping leaves the real ToC text — also legitimate, also previously rejected."""
    # Match the MEASURED shape (4,698 markers in 108,135 chars = 29%), not a marker-only string:
    # real ToC lines carry a dot leader between markers, so ~14 chars of text per ~5-char marker.
    toc = "About this manual " + "".join(f"<C{i}> . . . . . . " for i in range(1, 400)) + " 12\n"
    row = {"prompt": LONGCITE_PREAMBLE + f"[Document Start]\n<C0>{toc}\n[Document End]\n\nWhat is section 1?"}
    c = _counters()
    rec, reason = lcs.extract_longcite(row, c, "default")
    assert reason is None and rec is not None
    assert "About this manual" in rec["text"]
    assert not re.search(r"<C_?\d+>", rec["text"])


def test_longcite_rejects_question_that_references_chunk_ids():
    """0/20 real rows do this, but if it ever happens the question is meaningless once markers are
    stripped, so the row must not survive."""
    row = {"prompt": LONGCITE_PREAMBLE + "[Document Start]\n<C0>Body text here.\n[Document End]\n\n"
                                         "What does <C0> say?"}
    rec, reason = lcs.extract_longcite(row, _counters(), "default")
    assert rec is None and reason == "question_refs_chunks"


@pytest.mark.parametrize("prompt,expected", [
    ("no delimiters at all", "no_doc_delims"),
    (LONGCITE_PREAMBLE + "[Document Start]\nBody\n[Document End]\n\n   ", "empty_question"),
])
def test_longcite_rejects_malformed(prompt, expected):
    rec, reason = lcs.extract_longcite({"prompt": prompt}, _counters(), "default")
    assert rec is None and reason == expected


# ---------------------------------------------------------------------------------------------------
# LoongRL / DocQA-RL — shared verl RL schema
# ---------------------------------------------------------------------------------------------------
LOONGRL_ROW = {
    # `context` is the UN-injected passage; `prompt` is the KeyChain-injected one. Reading `context`
    # would silently discard the entire task -- module docstring hazard 2.
    "context": "Passage 1:\nNUS Business School\nNUS Business School is the business school of NUS.",
    "prompt": [{"role": "user", "content":
                'Please read the following text.\nPassage 1:\nNUS Business School\n'
                '{"652a79f3-5480-46cc-931f-3cd4469efe40": "473bc5be-eeba-43a0-b1f1-ca9b0dcb92b0"}.'
                "NUS Business School is the business school of NUS.\n"
                "...following the correct consecutive chain of key:value pairs encoded with UUID "
                "strings, starting from 64586b4d-0000-0000-0000-000000000000."}],
    "reward_model": {"ground_truth": ["Hergé"], "style": "rule"},
    "extra_info": {"index": 0, "input_question": "Which Belgian cartoonist?", "split": "train"},
    "data_source": "custom_longcontext_needle_qa_hotpotqa_qwen_filtered-max_seq_16384-distractor",
}


def test_verl_rl_uses_injected_prompt_not_context():
    rec, reason = lcs.extract_verl_rl(LOONGRL_ROW, _counters(), "hotpotqa_distractor_2500_5000")
    assert reason is None
    assert "652a79f3-5480-46cc-931f-3cd4469efe40" in rec["text"], "KeyChain injection was lost"
    assert rec["text"] != LOONGRL_ROW["context"]
    assert rec["question"] == "Which Belgian cartoonist?"
    assert rec["uid"] == "0"


def test_verl_rl_ground_truth_is_metadata_only_and_always_a_list():
    rec, _ = lcs.extract_verl_rl(LOONGRL_ROW, _counters(), "c")
    assert rec["meta"]["ground_truth"] == ["Hergé"]
    scalar = dict(LOONGRL_ROW, reward_model={"ground_truth": "42"})
    rec2, _ = lcs.extract_verl_rl(scalar, _counters(), "c")
    assert rec2["meta"]["ground_truth"] == ["42"]


def test_verl_rl_handles_json_encoded_structs():
    """Parquet hands struct columns back as dicts, but a jsonl path yields JSON strings."""
    import json as _json
    row = dict(LOONGRL_ROW, reward_model=_json.dumps({"ground_truth": ["x"]}),
               extra_info=_json.dumps({"index": 7, "input_question": "q?"}))
    rec, reason = lcs.extract_verl_rl(row, _counters(), "c")
    assert reason is None and rec["uid"] == "7" and rec["meta"]["ground_truth"] == ["x"]


# ---------------------------------------------------------------------------------------------------
# ChatQA2 — two configs, same column names, incompatible layouts
# ---------------------------------------------------------------------------------------------------
def test_chatqa2_narrativeqa_joins_doc_and_question_and_strips_llama3_bos():
    row = {"paragraph_id": "abc123", "question": "Who is Miss Delmer?",
           "sub-paragraphs": lcs.LLAMA3_BOS + "The Project Gutenberg EBook of Percival Keene."}
    c = _counters()
    rec, reason = lcs.extract_chatqa2(row, c, "NarrativeQA_131072")
    assert reason is None
    assert lcs.LLAMA3_BOS not in rec["text"]
    assert rec["text"].startswith("The Project Gutenberg EBook")
    assert rec["text"].endswith("Who is Miss Delmer?")
    assert rec["question"] == "Who is Miss Delmer?"
    assert rec["uid"] == "abc123"
    assert c["llama3_bos_stripped"] == 1


def test_chatqa2_long_sft_strips_chat_wrapper_and_does_not_duplicate_document():
    """`long_sft` has an EMPTY `sub-paragraphs` and puts the whole prompt in `question`, wrapped in
    `User: ... \\n\\nAssistant:`. Leaving that suffix in would end the user turn with a fake assistant
    role marker, immediately before Qwen3's own <|im_start|>assistant."""
    row = {"paragraph_id": "", "sub-paragraphs": "",
           "question": "User: Given an article and a question, respond concisely.\n\nArticle:\n"
                       "Greece has the largest merchant navy in the world.\n\n"
                       "Question:\nWhat other countries are middle powers?\n\nAssistant:"}
    c = _counters()
    rec, reason = lcs.extract_chatqa2(row, c, "long_sft")
    assert reason is None
    assert not rec["text"].startswith("User:")
    assert not re.search(r"Assistant:\s*$", rec["text"]), "trailing role marker survived"
    assert rec["text"].startswith("Given an article")
    assert rec["text"].count("Greece has the largest merchant navy") == 1, "document duplicated"
    assert rec["uid"] is None  # paragraph_id is empty in this config
    assert c["chatqa2_user_prefix_stripped"] == 1
    assert c["chatqa2_assistant_suffix_stripped"] == 1


def test_chatqa2_rejects_empty():
    assert lcs.extract_chatqa2({"question": "", "sub-paragraphs": "x"}, _counters(), "c")[1] == "empty_field"


# ---------------------------------------------------------------------------------------------------
# Mojibake repair — must fix the real case and refuse to mangle anything else
# ---------------------------------------------------------------------------------------------------
def test_fix_mojibake_repairs_latin1_decoded_utf8():
    original = "﻿The Café résumé — naïve"
    broken = original.encode("utf-8").decode("latin-1")
    fixed, repaired = lcs.fix_mojibake(broken)
    assert repaired is True
    assert fixed == original.replace("﻿", "")


def test_fix_mojibake_leaves_clean_text_untouched():
    for clean in ("Plain ASCII text.", "正常的中文文本", "Café résumé naïve"):
        fixed, repaired = lcs.fix_mojibake(clean)
        assert repaired is False and fixed == clean


def test_fix_mojibake_strips_bom_without_claiming_a_repair():
    fixed, repaired = lcs.fix_mojibake("﻿hello")
    assert fixed == "hello" and repaired is False


# ---------------------------------------------------------------------------------------------------
# LongAlign / LongAlpaca / LongReward
# ---------------------------------------------------------------------------------------------------
def test_longalign_drops_multi_turn():
    """Qwen3's template strips reasoning from every assistant turn before the last user query, so a
    multi-turn BC row would train on a trace-deleted history (phase2_data_gen.md §6)."""
    multi = {"id": "x", "dataset": "long", "messages": [
        {"role": "user", "content": "doc ... q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}]}
    c = _counters()
    rec, reason = lcs.extract_longalign(multi, c, "default")
    assert rec is None and reason == "multi_turn" and c["multi_turn"] == 1


def test_longalign_accepts_single_turn():
    rec, reason = lcs.extract_longalign(
        {"id": "x", "dataset": "long", "messages": [{"role": "user", "content": "long doc\n\nq?"}]},
        _counters(), "default")
    assert reason is None and rec["text"] == "long doc\n\nq?" and rec["uid"] == "x"


def test_longalpaca_uses_instruction_and_appends_nonnull_input():
    rec, reason = lcs.extract_longalpaca(
        {"instruction": "Below is a paper. ... Question: what?", "input": None, "file": "f.txt"},
        _counters(), "default")
    assert reason is None and rec["text"].endswith("Question: what?")
    c = _counters()
    rec2, _ = lcs.extract_longalpaca(
        {"instruction": "doc", "input": "extra", "file": None}, c, "default")
    assert rec2["text"] == "doc\n\nextra" and c["longalpaca_input_nonnull"] == 1


def test_longreward_lang_tag_follows_the_query_not_the_context():
    """LongReward is genuinely cross-lingual — English context, Chinese query — and it is the QUERY
    that determines what language the target will answer in."""
    # The context must be realistically long for this to mean anything: the real row is 42,397 chars
    # of English against a 60-char Chinese query, so a whole-prompt CJK ratio is ~0.1% and tags 'en'
    # while the model will answer in Chinese. That gap is the reason lang_basis exists.
    context = ("In 1963, the leaders of the two global superpowers stared each other down over the "
               "Cuban missile crisis, and one of them blinked. ") * 40
    rec, reason = lcs.extract_longreward(
        {"idx": 9, "context": context,
         "query": "文章中提到，美国支持南越的非共产主义政权，这反映了什么分歧？"}, _counters(), "default")
    assert reason is None
    assert len(context) > 5000, "fixture must be long enough for the ratio argument to hold"
    assert lcs.lang_of(rec["question"]) == "zh"
    assert lcs.lang_of(rec["text"]) == "en"  # the whole prompt is dominated by the English context


# ---------------------------------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------------------------------
def test_only_training_splits_are_registered():
    """ChatQA2 (both configs), DocQA-RL and LongReward all ship held-out splits. Drawing prompts from
    those is self-contamination — runbook §4.1 finding 2."""
    for name, spec in lcs.SOURCES.items():
        for _config, split in spec["splits"]:
            assert not re.search(r"(test|dev|valid|dpo)", split, re.I), f"{name} registers split {split!r}"


def test_loongrl_registers_only_distractor_configs():
    """The `_qwen_` configs are the un-injected originals at p50 14,217 (0% >=16K)."""
    cfgs = [c for c, _ in lcs.SOURCES["loongrl"]["splits"]]
    assert len(cfgs) == 3 and all("distractor" in c for c in cfgs)


def test_every_source_has_an_extractor_and_a_tier():
    assert set(lcs.SOURCES) == set(lcs.EXTRACTORS)
    assert set(lcs.TIER_A).isdisjoint(lcs.TIER_B)
    assert lcs.TIER_B == ["chatqa2"], "only the CC-BY-NC source belongs in tier B"
    for name, spec in lcs.SOURCES.items():
        assert spec["tier"] in ("A", "B") and spec["licence"], name


def test_cjk_language_tagging():
    assert lcs.lang_of("这是一个中文问题，关于房地产市场的表现。") == "zh"
    assert lcs.lang_of("What does the second sentence add?") == "en"
    assert lcs.lang_of("") == "en"
