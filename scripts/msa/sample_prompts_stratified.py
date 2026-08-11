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
"""Carve a domain-stratified subset out of a ``prompts.jsonl`` -- the pilot prompt set.

**Why this exists rather than ``gen_trajectories.py --limit N``.** ``prompts.jsonl`` is written in domain
order: the first ~24,000 rows are all Math. ``--limit 100`` therefore yields a 100 % Math pilot, whose
truncation rate and length histogram describe the single most verbose slice in the bank and nothing else.
Since the truncation rate is the number the pilot exists to measure (and the one that decides whether the
32K window stands), a single-domain pilot is worse than no pilot: it looks like a measurement.

Optionally excludes the frozen val prompts, so pilot generation never touches held-out data.

Example:
    python3 scripts/msa/sample_prompts_stratified.py \
      --src <run>/prompts.jsonl --out <run>/prompts_pilot100.jsonl --per-domain 25 \
      --exclude-sha /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__ph2b_split_v1/val_prompt_shas.txt
"""

import argparse
import collections
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsa"))
from _dsa_log import setup_logging  # noqa: E402


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-domain", type=int, default=25)
    ap.add_argument("--domains", nargs="+", default=None, help="restrict to these domains (default: all)")
    # Length-band stratification. Domain alone is not enough for the long-context pilot: the numbers it
    # exists to measure -- decode length and truncation rate -- both vary with PREFILL length, and a
    # source like LongCite spans 16K to 138K. Sampling it flat would put almost everything in the
    # 16-32K band and leave the 64K+ band, where the window actually bites, unmeasured.
    # docs/qwen3_4b_msa/phase2_long_context_gen.md §10.
    ap.add_argument("--bands", default=None,
                    help="comma-separated prefill-token upper bounds, e.g. '32768,65536,131072,1e9'. "
                         "With --per-band, sample N from each (domain, band) cell")
    ap.add_argument("--per-band", type=int, default=0,
                    help="rows per (domain, band) cell; overrides --per-domain when --bands is set")
    ap.add_argument("--exclude-sha", nargs="*", default=[],
                    help="file(s) of prompt_sha256 to skip (e.g. the frozen val split)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(a.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    log, log_path = setup_logging("sample_prompts_stratified", os.path.join(out_dir, "logs"))
    log.info("args: %s", vars(a))
    t0 = time.time()

    excl = set()
    for p in a.exclude_sha:
        with open(p) as f:
            excl.update(ln.strip() for ln in f if ln.strip())
        log.info("exclude list %s -> %d sha total", p, len(excl))

    by = collections.defaultdict(list)
    n_read = n_excl = 0
    with open(a.src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_read += 1
            if r.get("prompt_sha256") in excl:
                n_excl += 1
                continue
            by[r.get("domain")].append(r)
    log.info("read %d rows; %d excluded; domains present: %s",
             n_read, n_excl, {d: len(v) for d, v in sorted(by.items())})

    want = sorted(a.domains) if a.domains else sorted(by)
    missing = [d for d in want if d not in by]
    assert not missing, f"requested domains absent from --src: {missing}"

    rng = random.Random(a.seed)
    picked = []
    if a.bands and a.per_band:
        bounds = [int(float(x)) for x in a.bands.split(",")]
        for d in want:
            lo = 0
            for hi in bounds:
                cell = [r for r in by[d] if lo <= (r.get("prompt_tokens") or 0) < hi]
                take = rng.sample(cell, min(a.per_band, len(cell)))
                langs = collections.Counter(r.get("lang") for r in take)
                # An empty cell is information, not an error: it says this source does not reach that
                # band. Log it rather than silently producing a short pilot.
                log.info("  %-11s %7s-%-7s %6d available -> sampled %-3d  lang=%s",
                         d, lo, hi if hi < 10**8 else "max", len(cell), len(take), dict(langs))
                if len(take) < a.per_band:
                    log.warning("  %-11s band %s-%s SHORT: %d < --per-band %d",
                                d, lo, hi, len(take), a.per_band)
                picked.extend(take)
                lo = hi
    else:
        for d in want:
            pool = by[d]
            take = rng.sample(pool, min(a.per_domain, len(pool)))
            if len(take) < a.per_domain:
                log.warning("  %-9s only %d available (< --per-domain %d)", d, len(take), a.per_domain)
            log.info("  %-9s %6d available -> sampled %d", d, len(pool), len(take))
            picked.extend(take)

    with open(a.out, "w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    shas = [r["prompt_sha256"] for r in picked]
    assert len(set(shas)) == len(shas), "duplicate prompt_sha256 in the sample"
    assert not (set(shas) & excl), "an excluded sha leaked into the sample"
    tokens = [r.get("prompt_tokens") for r in picked if r.get("prompt_tokens") is not None]
    log.info("wrote %d prompts -> %s (prompt_tokens min=%s max=%s)", len(picked), a.out,
             min(tokens) if tokens else "n/a", max(tokens) if tokens else "n/a")

    manifest = dict(
        kind="msa_phase2_prompts_stratified_subset", source=os.path.abspath(a.src),
        per_domain=a.per_domain, domains=want, n_rows=len(picked), seed=a.seed,
        excluded_sha_files=[os.path.abspath(p) for p in a.exclude_sha], n_excluded=n_excl,
        by_domain=dict(collections.Counter(r.get("domain") for r in picked)),
        git_commit=_git("rev-parse", "HEAD"), invocation=" ".join([sys.executable] + sys.argv),
        host=os.uname().nodename, elapsed_s=round(time.time() - t0, 1),
        log_file=os.path.basename(log_path),
    )
    with open(a.out + ".MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1)
    log.info("DONE: %s", manifest["by_domain"])


if __name__ == "__main__":
    sys.exit(main())
