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
"""DSA Phase-2 — pre-generation PROMPT SELECTION (see docs/dsa_phase2_plan.md, T5).

Subsets the (gated) ``openbmb/UltraData-SFT-2605`` ``no_think`` corpus into a per-config,
language-filtered prompt set for self-generation. For each kept prompt we record a back-reference to the
origin sample: the source ``uid`` and a ``prompt_sha256`` (sha256 of the prompt text — stable, source-
agnostic, dedup key). Gold responses are DISCARDED (we self-generate). Runs CPU-only (no torch).

Two modes:
  * **split spec** (``--split-json`` = JSON string or file): per-config target counts + per-config language
    filter, e.g. source ZH from ``Multi-lang-*`` and EN from base configs. Counts are absolute (``count``)
    or fractions of ``--total`` (``frac``). This is the M3a path.
        {"Code": {"frac":0.30,"lang":"en"}, "Multi-lang-Math": {"frac":0.15,"lang":"zh"}, ...}
  * **uniform** (fallback): ``--domains`` + ``--per-domain`` (equal count, no language filter). M2 probe path.

Per config we adaptively scan shards until the (language-filtered, deduped) pool reaches the target count
(capped by ``--max-shards-per-config``). Output JSONL row:
    {source_uid, source_dataset, source_config, domain, lang, prompt_sha256,
     messages: [{role:"user", content:<prompt>}]}

Requires a HF token with access to the gated repo (``~/.cache/huggingface/token`` or ``HF_TOKEN``).
"""

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsa_log import setup_logging  # noqa: E402

REPO = "openbmb/UltraData-SFT-2605"
DEFAULT_DOMAINS = ["Math", "Code", "IF", "Knowledge", "Chinese-general"]


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk / len(text)


def _lang_of(prompt: str) -> str:
    return "zh" if _cjk_ratio(prompt) > 0.1 else "en"


def _extract_prompt(messages) -> str:
    """Concatenate all non-assistant message contents = the prompt (user + optional system)."""
    parts = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") != "assistant":
            parts.append(m.get("content") or "")
    return "\n".join(parts).strip()


def load_exclude_shas(paths, logger):
    """Load a set of prompt_sha256 to skip. Each path may be a plain sha-per-line file OR a
    prompts.jsonl (rows carrying ``prompt_sha256``). Used to make a run generate *net-new* prompts
    relative to earlier run(s)."""
    excl = set()
    for p in paths or []:
        if not os.path.exists(p):
            logger.warning("--exclude-sha: %s does not exist — skipping", p)
            continue
        n0 = len(excl)
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line[:1] == "{":  # jsonl row
                    try:
                        sha = json.loads(line).get("prompt_sha256")
                    except json.JSONDecodeError:
                        sha = None
                    if sha:
                        excl.add(sha)
                elif len(line) == 64 and all(c in "0123456789abcdef" for c in line):
                    excl.add(line)  # bare sha256 hex
        logger.info("--exclude-sha: +%d shas from %s (total %d)", len(excl) - n0, p, len(excl))
    return excl


def resolve_specs(args, logger):
    """Return list of (config, target_count, lang_filter). lang_filter in {'en','zh','any'}."""
    if args.split_json:
        raw = open(args.split_json).read() if os.path.exists(args.split_json) else args.split_json
        spec = json.loads(raw)
        out = []
        for cfg, s in spec.items():
            if "count" in s:
                cnt = int(s["count"])
            elif "frac" in s:
                assert args.total, "--total is required when the split spec uses 'frac'"
                cnt = int(round(float(s["frac"]) * args.total))
            else:
                raise ValueError(f"config '{cfg}' needs 'count' or 'frac' in --split-json")
            out.append((cfg, cnt, s.get("lang", "any")))
        logger.info("split-spec mode: %d configs, total target=%d", len(out), sum(c for _, c, _ in out))
        return out
    logger.info("uniform mode: %d domains x %d", len(args.domains), args.per_domain)
    return [(d, args.per_domain, "any") for d in args.domains]


def collect_config(fs, hf_hub_download, cfg, count, lang_filter, args, rng, logger, exclude_shas=None):
    """Adaptively scan shards for one config until the filtered/deduped pool reaches `count`.

    ``exclude_shas`` (optional): prompt_sha256 seen in earlier runs — skipped so this run is net-new."""
    exclude_shas = exclude_shas or set()
    ds_prefix = f"datasets/{REPO}"
    ddir = f"{ds_prefix}/data/{args.split}/{cfg}"
    try:
        shards = sorted(f for f in fs.ls(ddir, detail=False) if f.endswith(".jsonl"))
    except Exception as e:
        logger.warning("[%s] cannot list %s: %s — skipping", cfg, ddir, e)
        return []
    seen = set()
    pool = []
    scanned = 0
    n_excluded = 0
    for sh in shards:
        if len(pool) >= count and scanned >= args.min_shards:
            break
        if scanned >= args.max_shards_per_config:
            logger.warning("[%s] hit max-shards-per-config=%d with pool=%d < target=%d",
                           cfg, args.max_shards_per_config, len(pool), count)
            break
        scanned += 1
        rel = sh.split(ds_prefix + "/", 1)[-1]
        lp = hf_hub_download(repo_id=REPO, filename=rel, repo_type="dataset", local_dir=args.local_dir)
        with open(lp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = _extract_prompt(row.get("messages", []))
                if not (args.min_prompt_chars <= len(prompt) <= args.max_prompt_chars):
                    continue
                lang = _lang_of(prompt)
                if lang_filter != "any" and lang != lang_filter:
                    continue
                sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                if sha in seen:
                    continue
                seen.add(sha)
                if sha in exclude_shas:
                    n_excluded += 1
                    continue
                pool.append(
                    {
                        "source_uid": row.get("uid"),
                        "source_dataset": REPO,
                        "source_config": f"{args.split}/{cfg}",
                        "domain": cfg,
                        "lang": lang,
                        "prompt_sha256": sha,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                )
    rng.shuffle(pool)
    kept = pool[:count]
    logger.info("[%s] scanned=%d shards pool=%d kept=%d/%d lang_filter=%s excluded=%d%s",
                cfg, scanned, len(pool), len(kept), count, lang_filter, n_excluded,
                "  ⚠SHORT" if len(kept) < count else "")
    return kept


def main():
    ap = argparse.ArgumentParser(description="Select prompts from UltraData-SFT-2605/no_think for self-gen.")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--log-dir", default=None, help="dir for the run log (default: <out dir>/logs)")
    # split-spec mode
    ap.add_argument("--split-json", default=None, help="JSON string or file: {config:{frac|count, lang}}")
    ap.add_argument("--total", type=int, default=0, help="total prompts (for 'frac' specs)")
    # uniform-mode fallback
    ap.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    ap.add_argument("--per-domain", type=int, default=2000)
    # shared
    ap.add_argument("--split", default="no_think", choices=["no_think", "think"])
    ap.add_argument("--min-shards", type=int, default=1, help="always scan at least this many shards/config")
    ap.add_argument("--max-shards-per-config", type=int, default=20, help="cap on adaptive shard scan")
    ap.add_argument("--max-prompt-chars", type=int, default=100_000, help="drop prompts longer than this")
    ap.add_argument("--min-prompt-chars", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--local-dir", default="/tmp/udsftcache", help="HF download cache dir")
    ap.add_argument("--exclude-sha", nargs="*", default=None,
                    help="file(s) of prompt_sha256 to SKIP (plain sha-per-line OR a prior prompts.jsonl); "
                         "makes this run net-new relative to earlier run(s)")
    # accepted for backward-compat with the uniform wrapper; treated as max-shards if set
    ap.add_argument("--shards-per-domain", type=int, default=0)
    args = ap.parse_args()
    if args.shards_per_domain and not args.split_json:
        args.max_shards_per_config = max(args.max_shards_per_config, args.shards_per_domain)
        args.min_shards = args.shards_per_domain

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(out_dir, "logs")
    logger, _ = setup_logging("select_prompts", log_dir)
    logger.info("config: %s", vars(args))

    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    rng = random.Random(args.seed)
    specs = resolve_specs(args, logger)
    exclude_shas = load_exclude_shas(args.exclude_sha, logger)

    total_kept = 0
    lang_counts = {}
    with open(args.out, "w") as fout:
        for cfg, count, lang_filter in specs:
            kept = collect_config(fs, hf_hub_download, cfg, count, lang_filter, args, rng, logger,
                                   exclude_shas=exclude_shas)
            for r in kept:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                lang_counts[r["lang"]] = lang_counts.get(r["lang"], 0) + 1
            total_kept += len(kept)

    logger.info("DONE: wrote %d prompts (lang %s) -> %s", total_kept, lang_counts, args.out)


if __name__ == "__main__":
    main()
