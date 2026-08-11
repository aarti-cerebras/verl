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
"""Long-context PROMPT BANKS for Qwen3-4B MSA Phase-2b.

Runbook: ``docs/qwen3_4b_msa/phase2_long_context_gen.md`` §4. Survey that selected these sources:
``docs/qwen3_4b_msa/phase2_long_context_data.md``.

**Prompts only.** Every response / ground-truth column is discarded for training purposes;
``ground_truth`` survives as *metadata* for the diagnostic accuracy log and never gates selection
(``docs/qwen3_4b_dsa/data_plan.md`` §6).

Six backends over seven repos (LoongRL and DocQA-RL share one verl RL schema). Column names are
**measured**, not inferred — see §4.1 of the runbook for the step-0 dump that produced them.

Three source-specific hazards this module exists to handle, all verified against real rows:

1. **LongCite carries citation scaffolding in 100% of rows** — a 109-token preamble instructing the
   model to emit ``<statement>{S}<cite>[{s1}-{e1}]</cite></statement>``, plus ``<C{i}>`` chunk markers
   at **9.7% of prompt tokens**, one every ~35-70 tokens. Both are stripped (§4.2): the markers are
   periodic artificial attention landmarks with no analogue in the eval distribution, and the indexer
   is trained by KL against dense attention *on these very sequences*.
2. **LoongRL's ``context`` is the UN-injected passage; ``prompt`` is the KeyChain-injected one.**
   Reading ``context`` would silently discard the entire task.
3. **ChatQA2 ships a Llama-3 ``<|begin_of_text|>`` marker AND upstream mojibake** (``ï»¿`` = a UTF-8
   BOM mis-decoded as latin-1) at the head of Gutenberg-derived documents.

Plus one rule that applies to every source: **train split only.** ChatQA2 (both configs), DocQA-RL and
LongReward all ship held-out ``test``/``dpo_*`` splits, and drawing prompts from those is
self-contamination.
"""

import hashlib
import json
import re
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------------------------------
# licence_tier: "A" = permissive or undeclared, "B" = CC-BY-NC-2.0 (ChatQA2) -- segregated so the NC
# rows can be dropped at training-mix time without regenerating (runbook §1 decision 6).
SOURCES = {
    "longcite": {
        "repo": "zai-org/LongCite-45k",
        "splits": [("default", "train")],
        "columns": ["prompt"],
        # The datasets-server auto-conversion is PARTIAL (2.52 GB vs 5.73 GB native), and LongCite is
        # ~60% of the pool -- reading the parquet branch would silently drop about half of the largest
        # source. Read the native jsonl.
        "native": {("default", "train"): [("long.jsonl", "jsonl")]},
        "licence": "Apache-2.0",
        "tier": "A",
    },
    "loongrl": {
        "repo": "OldKingMeister/LoongRL-Train-Data",
        # ONLY the _distractor_ configs clear 16K; the _qwen_ configs are the un-injected originals
        # at p50 14,217 (0% >=16K) and are the wrong half to take.
        "splits": [
            ("hotpotqa_distractor_2500_5000", "train"),
            ("musique_distractor_2500_5000", "train"),
            ("2wikipedia_distractor_2500_5000", "train"),
        ],
        "columns": ["prompt", "reward_model", "extra_info", "data_source"],
        "licence": "undeclared (third-party mirror)",
        "tier": "A",
    },
    "longreward": {
        "repo": "zai-org/LongReward-10k",
        "splits": [("default", "sft")],  # NOT dpo_glm4_9b / dpo_llama3.1_8b
        "columns": ["idx", "context", "query"],
        "licence": "Apache-2.0",
        "tier": "A",
    },
    "longalign": {
        "repo": "zai-org/LongAlign-10k",
        "splits": [("default", "train")],
        "columns": ["dataset", "id", "messages"],  # NOT `length` -- that is a ChatGLM3 count (trap 1)
        "licence": "undeclared",
        "tier": "A",
    },
    "longalpaca": {
        "repo": "Yukang/LongAlpaca-12k",
        "splits": [("default", "train")],
        "columns": ["instruction", "input", "file"],
        "licence": "undeclared",
        "tier": "A",
    },
    "docqarl": {
        "repo": "Tongyi-Zhiwen/DocQA-RL-1.6K",
        "splits": [("default", "train")],  # NOT test (2,006 rows)
        "columns": ["prompt", "reward_model", "extra_info", "data_source"],
        "licence": "Apache-2.0",
        "tier": "A",
    },
    "chatqa2": {
        "repo": "nvidia/ChatQA2-Long-SFT-data",
        "splits": [("NarrativeQA_131072", "train"), ("long_sft", "train")],  # NOT test / dev
        "columns": ["paragraph_id", "question", "sub-paragraphs"],
        # Auto-conversion is PARTIAL (4.42 GB vs 14.3 GB of native train JSON). Native files are
        # top-level JSON arrays, streamed with ijson.
        "native": {
            ("NarrativeQA_131072", "train"):
                [("NarrativeQA_131072/NarrativeQA_131072_QA_train.json", "json")],
            ("long_sft", "train"): [("long_sft/long_sft_QA_train.json", "json")],
        },
        "licence": "CC-BY-NC-2.0",
        "tier": "B",
    },
}

TIER_A = [s for s, v in SOURCES.items() if v["tier"] == "A"]
TIER_B = [s for s, v in SOURCES.items() if v["tier"] == "B"]

# LongCite's own delimiters. Kept: they separate document from question, which is a real structural
# cue present in most long-doc QA formats (LongAlpaca's "The paper begins./ends." is the same idea).
# What is stripped is the citation *task* -- the preamble and the <C{i}> annotation of the body.
DOC_START = "[Document Start]"
DOC_END = "[Document End]"
CITE_MARKER_RE = re.compile(r"<C_?\d+>")
# Tripwire for a stripper that EATS THE DOCUMENT -- nothing else. Measured across 12,000 real rows,
# BOTH tails of the marker-share distribution are legitimate data, so the bound sits far out at 0.60
# and there is no lower bound at all:
#   * low share  (1 marker in 49,153 chars): forms, tables and name lists that LongCite's sentence
#     chunker barely segmented. Nothing is wrong with them.
#   * high share (4,698 markers in 108,135 chars = 29%): tables of contents, where every dot of a
#     ". . . . ." leader is chunked as its own sentence. Stripping leaves the real ToC text intact.
# An earlier revision rejected anything outside [0.001, 0.25] and threw away 348 LongCite rows (~1.4%),
# biased toward exactly those two document types. Only a share above 0.60 implies the regex is
# removing content rather than markers.
CITE_STRIP_MAX_FRAC = 0.60
CITE_LOW_DENSITY_FRAC = 0.001  # diagnostic only -- counted, never a rejection

LLAMA3_BOS = "<|begin_of_text|>"
_MOJIBAKE_SIGNS = ("ï»¿", "â€™", "â€œ", "â€\x9d", "Ã©", "Ã¨", "Ã¢", "Ãª", "Ã‚")


# ---------------------------------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------------------------------
def cjk_ratio(text: str, sample: int = 20_000) -> float:
    """CJK codepoint share. Sampled: these documents run to 500K+ chars and the ratio is stable."""
    t = text[:sample]
    if not t:
        return 0.0
    return sum(1 for ch in t if "一" <= ch <= "鿿") / len(t)


def lang_of(text: str) -> str:
    return "zh" if cjk_ratio(text) > 0.1 else "en"


def fix_mojibake(text: str):
    """Repair a latin-1-decoded UTF-8 document, conservatively.

    ChatQA2's Gutenberg rows begin ``ï»¿`` -- a UTF-8 BOM that was decoded as latin-1 upstream. The
    round trip is attempted **strictly in both directions** and abandoned on any error, so a document
    that merely *contains* one of the marker sequences legitimately is never mangled.

    Returns ``(text, repaired: bool)``.
    """
    text = text.replace("﻿", "")
    if not any(m in text for m in _MOJIBAKE_SIGNS):
        return text, False
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, False
    return fixed.replace("﻿", ""), True


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _msg_list_text(prompt_field) -> str | None:
    """verl RL format: ``prompt`` is a list of ``{role, content}``. Concatenate non-assistant turns."""
    if isinstance(prompt_field, str):
        return prompt_field
    if prompt_field is None:
        return None
    try:
        parts = [
            (m.get("content") or "")
            for m in prompt_field
            if isinstance(m, dict) and m.get("role") != "assistant"
        ]
    except (AttributeError, TypeError):
        return None
    return "\n".join(parts).strip() or None


def _as_obj(v):
    """Parquet may hand back a dict, a JSON string, or a numpy scalar for struct columns."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return {}
    return {}


# ---------------------------------------------------------------------------------------------------
# Per-source extractors
# ---------------------------------------------------------------------------------------------------
# Each returns (record | None, reject_reason | None) where record is
#   {"text": <prompt>, "question": <str|None>, "uid": <str|None>, "meta": {...}}
# `question` drives the `lang` tag when available: LongReward is genuinely cross-lingual (English
# context, Chinese query) and it is the QUERY that determines the response language.


def extract_longcite(row, counters, config):
    """Strip the citation preamble and the ``<C{i}>`` chunk markers. Runbook §4.2."""
    raw = row.get("prompt") or ""
    if DOC_START not in raw or DOC_END not in raw:
        return None, "no_doc_delims"
    _preamble, rest = raw.split(DOC_START, 1)
    body, _, question = rest.partition(DOC_END)

    n_markers = len(CITE_MARKER_RE.findall(body))
    stripped = CITE_MARKER_RE.sub("", body)
    removed = len(body) - len(stripped)
    frac = removed / len(body) if body else 0.0
    if frac > CITE_STRIP_MAX_FRAC:
        # Only a share this extreme means the regex is deleting content, not markers.
        counters["cite_strip_out_of_band"] += 1
        return None, "cite_strip_out_of_band"
    if n_markers and frac < CITE_LOW_DENSITY_FRAC:
        counters["cite_low_marker_density"] += 1  # diagnostic: sparsely-chunked forms/tables
    counters["cite_markers_removed"] += n_markers

    question = question.strip()
    if CITE_MARKER_RE.search(question):
        counters["question_refs_chunks"] += 1  # 0/20 in the step-0 sample; drop if it ever happens
        return None, "question_refs_chunks"
    if not question:
        return None, "empty_question"
    text = f"{DOC_START}\n{stripped.strip()}\n{DOC_END}\n\n{question}"
    return {"text": text, "question": question, "uid": None, "meta": {}}, None


def extract_verl_rl(row, counters, config):
    """LoongRL + DocQA-RL: identical verl RL schema.

    ``prompt`` (a message list) is the KeyChain-INJECTED text. ``context`` is the un-injected base
    passage and must not be used -- see the module docstring, hazard 2.
    """
    text = _msg_list_text(row.get("prompt"))
    if not text:
        return None, "empty_prompt"
    extra = _as_obj(row.get("extra_info"))
    rm = _as_obj(row.get("reward_model"))
    gt = rm.get("ground_truth")
    if gt is not None and not isinstance(gt, list):
        gt = [gt]
    return (
        {
            "text": text,
            "question": extra.get("input_question"),
            "uid": str(extra.get("index")) if extra.get("index") is not None else None,
            # metadata only -- NEVER a filter (data_plan.md §6)
            "meta": {"ground_truth": gt, "data_source": row.get("data_source")},
        },
        None,
    )


def extract_longreward(row, counters, config):
    """``(context, query)`` -> one user turn. ``win_response``/``lose_response`` are the literal string
    ``"none"`` in the sft split and are not read at all."""
    ctx = (row.get("context") or "").strip()
    query = (row.get("query") or "").strip()
    if not ctx or not query:
        return None, "empty_field"
    idx = row.get("idx")
    return (
        {
            "text": f"{ctx}\n\n{query}",
            "question": query,
            "uid": str(idx) if idx is not None else None,
            "meta": {},
        },
        None,
    )


def extract_longalign(row, counters, config):
    """``messages`` list. Multi-turn is DROPPED: Qwen3's template strips reasoning from every assistant
    turn before the last user query, so a multi-turn BC row trains on a trace-deleted history
    (phase2_data_gen.md §6)."""
    msgs = row.get("messages")
    if msgs is None:
        return None, "empty_prompt"
    try:
        users = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    except TypeError:
        return None, "empty_prompt"
    if len(users) != 1:
        counters["multi_turn"] += 1
        return None, "multi_turn"
    text = (users[0].get("content") or "").strip()
    if not text:
        return None, "empty_prompt"
    return {"text": text, "question": None, "uid": row.get("id"), "meta": {"subset": row.get("dataset")}}, None


def extract_longalpaca(row, counters, config):
    """``instruction`` already holds document + question; ``input``/``file`` are null in the sampled row.

    Its light preamble (*"Below is a paper. Memorize the paper and answer my question after the
    paper."*) is KEPT -- unlike LongCite's, it is ~20 tokens of plain framing and injects nothing into
    the document body (runbook §4.1 finding 4).
    """
    text = (row.get("instruction") or "").strip()
    if not text:
        return None, "empty_prompt"
    extra = (row.get("input") or "").strip()
    if extra:
        text = f"{text}\n\n{extra}"
        counters["longalpaca_input_nonnull"] += 1
    return {"text": text, "question": None, "uid": row.get("file"), "meta": {}}, None


CHATQA2_USER_PREFIX = "User: "
CHATQA2_ASSISTANT_SUFFIX_RE = re.compile(r"\n*Assistant:\s*$")


def extract_chatqa2(row, counters, config):
    """The two configs share column NAMES but not layout — verified against real rows of each:

    * ``NarrativeQA_131072``: ``sub-paragraphs`` is the document (up to ~550 KB), ``question`` is a
      bare question (*"Who is Miss Delmer?"*), no chat wrapper.
    * ``long_sft``: ``sub-paragraphs`` is **empty**; ``question`` carries the entire prompt —
      instruction + ``Article:`` + document + ``Question:`` — wrapped in ``User: … \\n\\nAssistant:``.

    Concatenating doc + question blindly would duplicate nothing for ``long_sft`` but would leave a
    trailing ``Assistant:`` role marker at the end of the user turn, i.e. a fake assistant turn
    immediately before Qwen3's own ``<|im_start|>assistant`` — the §7 "role-marker leakage" pathology,
    manufactured by us. Dispatch on the actual content, not on the config name, so a mixed row is
    handled correctly either way.
    """
    doc = row.get("sub-paragraphs") or ""
    question = (row.get("question") or "").strip()
    if not question:
        return None, "empty_field"

    if question.startswith(CHATQA2_USER_PREFIX):
        question = question[len(CHATQA2_USER_PREFIX):].lstrip()
        counters["chatqa2_user_prefix_stripped"] += 1
    m = CHATQA2_ASSISTANT_SUFFIX_RE.search(question)
    if m:
        question = question[: m.start()].rstrip()
        counters["chatqa2_assistant_suffix_stripped"] += 1
    if not question:
        return None, "empty_field"

    text = f"{doc.strip()}\n\n{question}" if doc.strip() else question
    if LLAMA3_BOS in text:
        text = text.replace(LLAMA3_BOS, "")
        counters["llama3_bos_stripped"] += 1
    text, repaired = fix_mojibake(text)
    if repaired:
        counters["mojibake_repaired"] += 1
    text = text.strip()
    if not text:
        return None, "empty_field"

    # Only NarrativeQA has a separable short question; for long_sft the "question" IS the prompt, so
    # the lang tag falls back to the whole text (which is what lang_basis records).
    short_q = question if (doc.strip() and len(question) < 2000) else None
    return {"text": text, "question": short_q, "uid": row.get("paragraph_id") or None, "meta": {}}, None


EXTRACTORS = {
    "longcite": extract_longcite,
    "loongrl": extract_verl_rl,
    "docqarl": extract_verl_rl,
    "longreward": extract_longreward,
    "longalign": extract_longalign,
    "longalpaca": extract_longalpaca,
    "chatqa2": extract_chatqa2,
}


# ---------------------------------------------------------------------------------------------------
# Parquet access
# ---------------------------------------------------------------------------------------------------
def list_parquet_files(repo, logger, timeout=90):
    """Resolve canonical parquet shards per (config, split) via the datasets-server /parquet endpoint.

    Uniform across all seven repos regardless of their native layout (jsonl, zip, custom dirs), because
    the endpoint reports the auto-converted ``refs/convert/parquet`` branch. Returns
    ``{(config, split): [filename, ...]}``.
    """
    url = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(repo)
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        payload = json.load(fh)
    out = {}
    partial = set()
    marker = "/resolve/refs%2Fconvert%2Fparquet/"
    for f in payload.get("parquet_files", []):
        # `filename` is only the leaf ("0000.parquet"); the in-branch path is what hf_hub_download
        # needs, and it is the tail of `url`. Note the path segment may read "partial-train" while
        # `f["split"]` has already been normalised to "train" -- so parse the URL, do not rebuild it.
        if marker not in f["url"]:
            logger.warning("[%s] unexpected parquet url %s — skipping", repo, f["url"])
            continue
        path = urllib.parse.unquote(f["url"].split(marker, 1)[1])
        if "/partial-" in f"/{path}":
            # datasets-server caps auto-conversion for large datasets. The rows are valid but the
            # split is TRUNCATED -- it must never be reported as the full source.
            partial.add((f["config"], f["split"]))
        out.setdefault((f["config"], f["split"]), []).append(path)
    for cfg, sp in sorted(partial):
        logger.warning("[%s] config=%s split=%s is a PARTIAL auto-conversion — the parquet branch is "
                       "TRUNCATED, so this is a lower bound on the real row count. Read the native "
                       "files instead if full coverage matters.", repo, cfg, sp)
    for k in out:
        out[k] = sorted(set(out[k]))
    return out


def _open_native(repo, path, args, logger):
    """Open a native repo file, streaming over HTTP when the run is capped.

    A capped run (``--limit-per-source``, i.e. every smoke test and the pilot) must not pay for a
    14 GB download to read 20 rows, so it streams and stops early. An uncapped run downloads instead:
    ``hf_hub_download`` is cached and resumable, which matters far more across a multi-hour full pass
    than the one-time transfer does.
    """
    if args.limit_per_source:
        from huggingface_hub import get_token, hf_hub_url

        url = hf_hub_url(repo, path, repo_type="dataset")
        headers = {}
        tokval = get_token()
        if tokval:
            headers["authorization"] = f"Bearer {tokval}"
        logger.info("[%s] STREAMING %s (capped run)", repo, path)
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)
    from huggingface_hub import hf_hub_download

    logger.info("[%s] downloading %s (uncapped run — cached + resumable)", repo, path)
    return open(hf_hub_download(repo_id=repo, filename=path, repo_type="dataset",
                                local_dir=args.local_dir), "rb")


def _iter_native(repo, files, args, logger):
    """Stream rows from native repo files. ``jsonl`` = one object per line; ``json`` = one top-level
    array, walked incrementally with ijson so an 8.7 GB file never lands in memory."""
    for path, fmt in files:
        fh = _open_native(repo, path, args, logger)
        try:
            if fmt == "jsonl":
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            elif fmt == "json":
                import ijson

                yield from ijson.items(fh, "item")
            else:
                raise ValueError(f"unknown native format {fmt!r} for {repo}/{path}")
        finally:
            fh.close()


def iter_rows(repo, config, split, columns, args, logger, native=None):
    """Stream rows of one (config, split).

    Prefers native repo files when the registry declares them — the datasets-server auto-conversion is
    PARTIAL for LongCite (2.52 of 5.73 GB) and ChatQA2 (4.42 of 17.34 GB), and those two are the
    largest sources in each tier, so silently reading a truncated branch would cost ~half of LongCite.
    Otherwise reads the converted parquet, row group at a time: ChatQA2 documents pass 500 KB each, so
    materialising a config as a DataFrame would need tens of GB.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if native and (config, split) in native:
        yield from _iter_native(repo, native[(config, split)], args, logger)
        return

    files = list_parquet_files(repo, logger).get((config, split))
    if not files:
        logger.error("[%s] no parquet for config=%s split=%s — skipping", repo, config, split)
        return
    logger.info("[%s] config=%s split=%s: %d parquet shard(s)", repo, config, split, len(files))
    for fname in files:
        local = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset",
                                revision="refs/convert/parquet", local_dir=args.local_dir)
        pf = pq.ParquetFile(local)
        have = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in have]
        missing = [c for c in columns if c not in have]
        if missing:
            logger.warning("[%s/%s] missing expected column(s) %s — schema drift?", repo, config, missing)
        for batch in pf.iter_batches(batch_size=args.batch_rows, columns=cols):
            yield from batch.to_pylist()


# ---------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------
def collect_longctx(args, rng, logger, exclude_shas=None, count_tokens_batch=None, emit=None):
    """Extract the long-context prompt pools. Returns ``({source: stats}, skipped_rows)``.

    **Rows are STREAMED to ``emit`` as they are accepted, never accumulated.** Buffering every source
    and writing at the end cost 8.9 GB of RSS on the Tier A run and would have thrown away ~50 minutes
    of tokenization on any crash before the last source; ChatQA2 (14.3 GB of source JSON) is worse.
    What stays resident is now only per-source scalars plus the token list for percentiles, and the
    cross-source dedup state -- which is hashes, not text.

    Dropping the old per-source ``rng.shuffle(pool)`` is what makes this possible, and it is safe
    *for this backend*: unlike the ultradata/dolci-rl paths there is no ``pool[:count]`` truncation
    downstream (every row is written), ``sample_prompts_stratified.py`` re-samples with its own RNG,
    and ``gen_trajectories``' ``prompts[rank::N]`` striding spreads each source across all replicas
    regardless of within-source order. Rows are therefore emitted in arrival order.

    Filters, in order: extractor-specific rejects -> char bounds -> ``--min-prefill-tokens`` ->
    ``--min-gen-budget`` (the window floor, runbook §5.2) -> sha256 dedup (global, across sources) ->
    ``--exclude-sha`` -> ``--max-per-document``.

    Tokenization is **batched** through ``count_tokens_batch``: one-at-a-time ``encode()`` runs at
    ~1.9 MB/s on these documents while ``encode_batch()`` reaches ~7.4 MB/s (measured, identical token
    counts), which is 50 -> 13 min for LongCite and 125 -> 32 min for ChatQA2. The filters themselves
    stay strictly in arrival order, because dedup keeps the first occurrence and ``--max-per-document``
    keeps the first N -- order is part of the result, so only the tokenizer call is reordered.
    """
    exclude_shas = exclude_shas or set()
    names = args.sources or (TIER_A if args.licence_tier == "A" else TIER_B)
    unknown = [n for n in names if n not in SOURCES]
    assert not unknown, f"unknown --sources {unknown}; known: {sorted(SOURCES)}"

    if args.min_gen_budget and count_tokens_batch is None:
        raise SystemExit("--min-gen-budget requires --tokenizer (the floor is a token budget)")

    if args.limit_per_source:
        logger.warning("--limit-per-source=%d reads each source SEQUENTIALLY from row 0, which is a "
                       "BIASED sample: LongAlpaca is ordered long-QA-first-then-short, ChatQA2/long_sft "
                       "concatenates sub-corpora, and NarrativeQA groups many questions per document "
                       "(phase2_long_context_data.md §0 trap 3). Use this for smoke tests only — draw "
                       "the pilot set from a FULL extraction with sample_prompts_stratified.py.",
                       args.limit_per_source)

    pools, skipped = {}, []
    seen = set()
    doc_prefix_seen = {}  # cross-source document-overlap DIAGNOSTIC (not a filter)
    doc_key_count = {}    # --max-per-document: NarrativeQA asks many questions per book
    batch_n = max(1, args.tokenize_batch)

    for name in names:
        spec = SOURCES[name]
        if args.licence_tier and spec["tier"] != args.licence_tier:
            logger.warning("[%s] tier %s != --licence-tier %s — skipping", name, spec["tier"], args.licence_tier)
            continue
        extract = EXTRACTORS[name]
        counters = _new_counters()
        stat = {"rows": 0, "lang": {}, "toks": [], "prefill_tokens_total": 0}

        def accept(rec, text, n_tok, config, spec=spec, name=name, stat=stat, counters=counters):
            """Every filter downstream of tokenization, for one row, in arrival order."""
            # --limit-per-source caps rows KEPT, not rows buffered. Enforcing it here rather than at
            # the read loop keeps the capped result identical to the unbatched path: the first N rows
            # that pass every filter, not the survivors of the first N candidates.
            if args.limit_per_source and stat["rows"] >= args.limit_per_source:
                return
            if n_tok is not None:
                if n_tok < args.min_prefill_tokens:
                    counters["below_min_prefill"] += 1
                    return
                # Runbook §5.1/§5.2: the wrapper is 10 tokens, the closing <|im_end|> is 1.
                budget = args.window - (n_tok + args.wrapper_tokens) - 1
                if budget < args.min_gen_budget:
                    counters["below_gen_budget"] += 1
                    skipped.append({"source": name, "config": config, "prompt_tokens": n_tok,
                                    "gen_budget": budget, "prompt_sha256": sha256_of(text)})
                    return

            sha = sha256_of(text)
            if sha in seen:
                counters["dup"] += 1
                return
            seen.add(sha)
            if sha in exclude_shas:
                counters["excluded"] += 1
                return

            # Document key = hash of the head of the prompt. Exact-prompt dedup cannot see that
            # NarrativeQA asks many DIFFERENT questions about the SAME book: those rows are distinct
            # prompts but share one document, and prefill is the dominant cost here, so each repeat
            # re-pays for ~130K tokens of the same text.
            dkey = sha256_of(text[:2000])
            if args.max_per_document:
                seen_n = doc_key_count.get(dkey, 0)
                if seen_n >= args.max_per_document:
                    counters["over_max_per_document"] += 1
                    return
                doc_key_count[dkey] = seen_n + 1
            # Diagnostic only: the same public document reaches several of these banks under
            # different framings.
            if dkey in doc_prefix_seen and doc_prefix_seen[dkey] != name:
                counters["doc_overlap_other_source"] += 1
            doc_prefix_seen.setdefault(dkey, name)

            basis = rec.get("question") or text
            row = {
                "source_uid": rec.get("uid"),
                "source_dataset": spec["repo"],
                "source_config": f"{config}/{rec['_split']}",
                "domain": name,
                "lang": lang_of(basis),
                "lang_basis": "question" if rec.get("question") else "prompt",
                "licence_tier": spec["tier"],
                "licence": spec["licence"],
                "prompt_sha256": sha,
                "messages": [{"role": "user", "content": text}],
                # Provenance: the tiers are generated under DIFFERENT windows (Tier B is raised to
                # 163,840 so ChatQA2's truncated-at-131,072-Llama-3-tokens documents still leave room
                # to answer -- runbook §2.2). A row must carry the window it was selected under, or a
                # merged parquet becomes impossible to interpret.
                "window": args.window,
                "min_gen_budget": args.min_gen_budget,
                **({"prompt_tokens": n_tok} if n_tok is not None else {}),
                **{k: v for k, v in rec["meta"].items() if v is not None},
            }
            emit(row)                       # <- streamed to disk here, not buffered
            stat["rows"] += 1
            stat["lang"][row["lang"]] = stat["lang"].get(row["lang"], 0) + 1
            if n_tok is not None:
                stat["toks"].append(n_tok)
                stat["prefill_tokens_total"] += n_tok

        def drain(buf):
            """Tokenize a buffer in ONE encode_batch call, then apply `accept` in order."""
            if not buf:
                return
            counts = (count_tokens_batch([t for _, t, _ in buf]) if count_tokens_batch
                      else [None] * len(buf))
            for (rec, text, config), n_tok in zip(buf, counts, strict=True):
                accept(rec, text, n_tok, config)
            buf.clear()

        for config, split in spec["splits"]:
            n_cfg = 0
            buf = []
            for row in iter_rows(spec["repo"], config, split, spec["columns"], args, logger,
                                 native=spec.get("native")):
                counters["raw"] += 1
                n_cfg += 1
                if args.limit_per_source and stat["rows"] >= args.limit_per_source:
                    break
                rec, reason = extract(row, counters, config)
                if rec is None:
                    counters[reason] = counters.get(reason, 0) + 1
                    continue
                text = rec["text"].strip()
                if not (args.min_prompt_chars <= len(text) <= args.max_prompt_chars):
                    counters["char_bounds"] += 1
                    continue
                rec["_split"] = split
                buf.append((rec, text, config))
                if len(buf) >= batch_n:
                    drain(buf)
            drain(buf)
            logger.info("[%s] %s/%s: %d raw rows scanned", name, config, split, n_cfg)
            if args.limit_per_source and stat["rows"] >= args.limit_per_source:
                break

        stat["counters"] = {k: v for k, v in counters.items() if v}
        pools[name] = stat
        logger.info("[%s] kept=%d lang=%s drops=%s", name, stat["rows"], stat["lang"],
                    {k: v for k, v in counters.items() if v and k != "raw"})
        if stat["toks"]:
            toks = sorted(stat["toks"])
            logger.info("[%s] prefill tokens: p50=%d p90=%d max=%d",
                        name, toks[len(toks) // 2], toks[int(0.9 * len(toks))], toks[-1])
        stat.pop("toks")  # percentiles are logged; the list itself is not needed downstream

    logger.info("pools: %s", {k: v["rows"] for k, v in sorted(pools.items())})
    if skipped:
        logger.warning("%d prompt(s) below --min-gen-budget=%d at window=%d — see skipped_nofit.jsonl",
                       len(skipped), args.min_gen_budget, args.window)
    return pools, skipped


def _new_counters():
    return dict.fromkeys(
        (
            "raw", "char_bounds", "dup", "excluded", "multi_turn", "empty_prompt", "empty_field",
            "empty_question", "no_doc_delims", "question_refs_chunks", "cite_strip_out_of_band",
            "cite_markers_removed", "cite_low_marker_density", "below_min_prefill", "below_gen_budget", "llama3_bos_stripped",
            "mojibake_repaired", "longalpaca_input_nonnull", "doc_overlap_other_source",
            "over_max_per_document", "chatqa2_user_prefix_stripped", "chatqa2_assistant_suffix_stripped",
        ),
        0,
    )
