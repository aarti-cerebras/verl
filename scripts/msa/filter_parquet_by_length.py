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
"""Cut a total-length band out of a BC parquet, without loading it into memory.

This is the operation ``phase2_long_context_gen.md`` decision 5 exists for: generation collects every
realized length once, and the Phase-2b mixture is sliced from it afterwards, as many times as needed,
without regenerating anything.

Streams row groups through a ``ParquetWriter``, so peak memory is one batch rather than the whole file
(the 43,491-row Tier A parquet is 2.6 GB on disk but ~150 GB as Python objects).

    python3 scripts/msa/filter_parquet_by_length.py \\
        --src  <run>/bc_2b_longctx.parquet \\
        --out  <run>/bc_2b_longctx_16k_45k.parquet \\
        --min-length 16384 --max-length 46080
"""

import argparse
import collections
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dsa"))
from _dsa_log import setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-length", type=int, required=True, help="inclusive lower bound on `length`")
    ap.add_argument("--max-length", type=int, required=True, help="inclusive upper bound on `length`")
    ap.add_argument("--length-column", default="length")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    logger, _ = setup_logging("filter_parquet_by_length", args.log_dir or os.path.join(out_dir, "logs"))
    assert args.min_length <= args.max_length, "--min-length must be <= --max-length"

    pf = pq.ParquetFile(args.src)
    total_in = pf.metadata.num_rows
    logger.info("src=%s rows=%d  band=[%d, %d] on `%s`",
                args.src, total_in, args.min_length, args.max_length, args.length_column)

    writer = None
    kept = 0
    tok_kept = 0
    by_domain = collections.Counter()
    by_lang = collections.Counter()
    lens = []
    for batch in pf.iter_batches(batch_size=args.batch_size):
        t = pa.Table.from_batches([batch])
        col = t.column(args.length_column).to_pylist()
        mask = [args.min_length <= v <= args.max_length for v in col]
        if not any(mask):
            continue
        sub = t.filter(pa.array(mask))
        if writer is None:
            writer = pq.ParquetWriter(args.out, sub.schema)
        writer.write_table(sub)
        kept += sub.num_rows
        d = sub.to_pydict()
        for i in range(sub.num_rows):
            n = d[args.length_column][i]
            lens.append(n)
            tok_kept += n
            by_domain[d.get("domain", ["?"] * sub.num_rows)[i]] += 1
            if "lang" in d:
                by_lang[d["lang"][i]] += 1
    if writer is not None:
        writer.close()
    else:
        logger.error("no rows in band [%d, %d] — nothing written", args.min_length, args.max_length)
        sys.exit(1)

    lens.sort()

    def p(q):
        return lens[min(len(lens) - 1, int(round(q / 100 * (len(lens) - 1))))]

    logger.info("kept %d / %d rows (%.2f%%)   tokens=%.3fB", kept, total_in, 100 * kept / total_in,
                tok_kept / 1e9)
    logger.info("  length: min=%d p50=%d p90=%d max=%d", lens[0], p(50), p(90), lens[-1])
    logger.info("  by domain: %s", dict(by_domain.most_common()))
    if by_lang:
        logger.info("  by lang:   %s", dict(by_lang))
    logger.info("  -> %s (%.2f GB)", args.out, os.path.getsize(args.out) / 1e9)

    meta = {"src": os.path.abspath(args.src), "out": os.path.abspath(args.out),
            "band": [args.min_length, args.max_length], "length_column": args.length_column,
            "rows_in": total_in, "rows_kept": kept, "tokens_kept": tok_kept,
            "length_p50": p(50), "length_p90": p(90), "length_min": lens[0], "length_max": lens[-1],
            "by_domain": dict(by_domain), "by_lang": dict(by_lang)}
    with open(args.out + ".BAND.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    logger.info("  band metadata -> %s.BAND.json", args.out)


if __name__ == "__main__":
    main()
