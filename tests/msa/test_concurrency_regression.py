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
"""Concurrency regression: the `topk_indices_buffer` layout bug (serving_plan §8 R6).

**The bug this exists for.** vLLM 0.26.0's `nvidia/model.py` allocates the shared top-k buffer
token-major `[tokens, heads, topk]`, but both consumers index it head-major -- `indexer.py`
writes `out=buf[:, nd:, :]` and `sparse_attention.py` reads `topk[:, :nd, :]`, where `nd` is a
DECODE TOKEN COUNT. Upstream fixed this after 0.26.0 by transposing in both places; our model
supplies the transposed view instead (`_consumers_transpose_topk_buffer`).

**Why every earlier test missed it.** With one request in flight the decode batch is 0 or 1
tokens, so `[:, :1, :]` stays inside the (mis-ordered) buffer -- wrong strides, no fault. The
kernels only walk off the end once the decode batch exceeds `num_index_heads`, which is **8**
here. Every P0-P4 test was single-request, so the entire parity suite passed against a build
that dies within seconds of real eval load: `CUDA error: an illegal memory access`.

So the property under test is not numerical -- it is "does the engine survive a decode batch
wider than num_index_heads". CONCURRENCY IS THE VARIABLE; keep it comfortably above 8.

A double-transpose (this fix left in place after upgrading to a fixed vLLM) fails identically,
which is why the fix is gated on source inspection, not a version string.

Run:
  cd <repo> && GPU=0 .devlibs/vllm026/bin/python tests/msa/test_concurrency_regression.py
"""

import argparse
import os
import subprocess
import sys

DEFAULT_MODEL = "/cb/ml-eng/aarti/msa/serving/k8v2_step100"

CHILD = r'''
import os, sys, json
sys.path.insert(0, os.environ["REPO"])
import scripts.msa.vllm_qwen3_msa  # noqa: F401
from scripts.msa.vllm_qwen3_msa.model import _consumers_transpose_topk_buffer
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

n_conc = int(os.environ["N_CONC"])
model = os.environ["MODEL"]
tok = AutoTokenizer.from_pretrained(model)

# Prompts must exceed k*B_k = 1024 tokens or selection is degenerate and the decode kernels
# never discard a block.
src = ("In a distant valley there lived a clockmaker who repaired instruments no one else could "
       "understand, and every evening he wrote down what the gears told him. ")
n = 1
while len(tok(src * n).input_ids) < int(os.environ.get('PROMPT_TOK', '8000')):
    n += 1
base = src * n
# Distinct prompts so nothing is served from a shared prefix.
prompts = [f"{base}\nObservation number {i}. The clockmaker's profession is" for i in range(n_conc)]

llm = LLM(model=model, tensor_parallel_size=1, block_size=128,
          max_model_len=int(os.environ.get("MAX_LEN", "40960")),
          gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
          dtype="bfloat16", enforce_eager=True)

# One generate() call with N prompts -> vLLM batches them, so the decode batch is ~N tokens/step.
outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=64, seed=0))
texts = [o.outputs[0].text for o in outs]
print("RESULT " + json.dumps({
    "n": len(texts),
    "n_empty": sum(1 for t in texts if not t.strip()),
    "sample": texts[0][:90],
    "fixed_upstream": _consumers_transpose_topk_buffer(),
}))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--concurrency", type=int, default=32,
                    help="must exceed num_index_heads (8); 32 gives margin")
    # The crash needs EVAL conditions, not just concurrency: 32 short prompts at max_model_len
    # 8192 survive the buggy layout, because slicing past dim 1 clamps rather than faulting.
    # Reproduced only at ~8K prompts with max_model_len 40960 (env MAX_LEN / PROMPT_TOK).
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env = dict(os.environ, MODEL=args.model, REPO=repo, N_CONC=str(args.concurrency),
               MSA_SPARSE="1", CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "0"),
               VLLM_NO_USAGE_STATS="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
               TOKENIZERS_PARALLELISM="false")
    print(f"driving {args.concurrency} concurrent requests (num_index_heads = 8) ...", flush=True)
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr

    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            import json

            res = json.loads(line[len("RESULT "):])
    if res is None:
        crashed = "illegal memory access" in out
        print("  FAIL  engine did not survive"
              + ("  (CUDA illegal memory access -- the topk buffer layout bug)" if crashed else ""))
        print("\n".join(out.splitlines()[-20:]))
        return 1

    ok = res["n"] == args.concurrency and res["n_empty"] == 0
    print(f"  upstream already transposes: {res['fixed_upstream']} "
          f"(so we pre-transpose: {not res['fixed_upstream']})")
    print(f"  {'PASS' if ok else 'FAIL'}  {res['n']}/{args.concurrency} completed, "
          f"{res['n_empty']} empty")
    print(f"  sample: {res['sample']!r}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
