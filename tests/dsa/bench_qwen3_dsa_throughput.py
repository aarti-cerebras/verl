#!/usr/bin/env python3
"""P4-lite: engine-level decode throughput, sparse vs dense, at realistic concurrency.

`docs/qwen3_4b_dsa/serving_eval_plan.md` §4 P4. The kernel-level numbers in
``probe_fa3_sparse_gqa.py`` predict that sparse decode wins big at concurrency and loses at batch 1;
this measures the whole engine, which is what eval wall-clock actually depends on.

Both servers must already be running (``scripts/dsa/serving/serve_qwen3_dsa.sh``), one with
``DSA_SPARSE=1`` and one with ``DSA_SPARSE=0``, so the ONLY difference is the attention path.

Usage:
  .devlibs/vllm026/bin/python tests/dsa/bench_qwen3_dsa_throughput.py \
      --sparse-port 8001 --dense-port 8002 --concurrency 16 --prompt-facts 400 --max-tokens 128
"""

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one(port: int, prompt: str, max_tokens: int, idx: int) -> dict:
    body = {
        "model": "Qwen3-4B-Thinking-2507",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,  # equalise work across rows; otherwise EOS makes the comparison
        "temperature": 0.0,        # depend on which model happened to stop early
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return {"dt": time.time() - t0, "usage": r["usage"]}


def run(port: int, concurrency: int, prompt: str, max_tokens: int) -> dict:
    # warm one request so JIT/autotune is not billed to the measurement
    one(port, prompt[:2000], 8, -1)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        res = list(ex.map(lambda i: one(port, prompt, max_tokens, i), range(concurrency)))
    wall = time.time() - t0
    out_tok = sum(r["usage"]["completion_tokens"] for r in res)
    in_tok = sum(r["usage"]["prompt_tokens"] for r in res)
    return {
        "wall_s": wall,
        "prompt_tokens_total": in_tok,
        "output_tokens_total": out_tok,
        "output_tok_per_s": out_tok / wall,
        "per_request_s": sum(r["dt"] for r in res) / len(res),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse-port", type=int, default=8001)
    ap.add_argument("--dense-port", type=int, default=8002)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prompt-facts", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()

    prompt = " ".join(
        f"Fact {i}: the code word for item {i} is {(i * 7919) % 10007}." for i in range(args.prompt_facts)
    ) + "\n\nSummarise the list.\nA:"

    rows = {}
    for name, port in (("sparse", args.sparse_port), ("dense", args.dense_port)):
        rows[name] = run(port, args.concurrency, prompt, args.max_tokens)
        r = rows[name]
        print(f"{name:>7}: wall {r['wall_s']:7.2f}s  out {r['output_tok_per_s']:8.1f} tok/s  "
              f"per-req {r['per_request_s']:6.2f}s  prompt_tok/req "
              f"{r['prompt_tokens_total'] // args.concurrency}")
    s, d = rows["sparse"], rows["dense"]
    print(f"\nconcurrency={args.concurrency}  max_tokens={args.max_tokens}")
    print(f"sparse / dense output throughput: {s['output_tok_per_s'] / d['output_tok_per_s']:.2f}x")
    print(f"sparse / dense wall-clock:        {d['wall_s'] / s['wall_s']:.2f}x faster")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
