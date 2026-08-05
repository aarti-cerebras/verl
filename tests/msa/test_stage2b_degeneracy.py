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
"""S2b degeneracy gate (docs/qwen3_4b_msa/serving_plan.md §6.3), P4.

**The idea.** With ``top_k = 8`` blocks of 128, any prompt shorter than ``k * B_k = 1024``
tokens has at most 8 visible blocks, so the top-k selects *every* one of them and the sparse
attention degenerates to exact dense attention over the same weights. Sparse and dense outputs
must then agree to bf16 noise. This exercises the real Triton kernels — the fused op, the index
score/top-k, the block-sparse attend — rather than a degenerate config, and needs no external
oracle: both sides are vLLM on the same serving directory.

**The control matters as much as the test.** "Sparse == dense on a short prompt" is only
evidence if the same comparison can *detect* a difference when selection is real. So we also run
a prompt LONGER than 1024 tokens, where the indexer must discard blocks, and require the two to
DIVERGE. Without this, a silently-dense sparse path would pass the short case and look correct —
which is the exact failure mode §6.6 exists to catch, arrived at from a different direction.

**The threshold is measured, not guessed.** A first version of this test used an absolute
|Δlogprob| <= 0.05 and "failed" the degenerate case at 6.4e-2 — but that is BELOW the setup's own
noise floor. On a Route-A export the ``w - 1`` round trip perturbs the norm gains by up to 3.8e-3
(§5.1), and that alone moves first-step logprobs by ~7e-2. So the test measures the floor itself,
from two DENSE runs that differ only by that round trip, plus a determinism control proving the
floor is the weights and not run-to-run variance. Everything is then judged in units of the floor.
Any later parity work on a Route-A export inherits this floor — do not compare against zero.

The dense side is ``MSA_SPARSE=0``, which loads through stock ``Qwen3Attention`` and un-shifts
the ``w - 1`` norms (§2.3); it is a bring-up reference, not a servable config.

Run:
  cd <repo> && GPU=0 .devlibs/vllm026/bin/python tests/msa/test_stage2b_degeneracy.py
"""

import argparse
import json
import math
import os
import subprocess
import sys

DEFAULT_MODEL = "/cb/ml-eng/aarti/msa/serving/k8_step1400"

CHILD = r'''
import os, sys, json
sys.path.insert(0, os.environ["REPO"])
import scripts.msa.vllm_qwen3_msa  # noqa: F401
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model = os.environ["MODEL"]
tok = AutoTokenizer.from_pretrained(model)

# SHORT: < k*B_k = 1024 tokens -> every block visible -> sparse must degenerate to dense.
# LONG : > 1024 tokens         -> selection is real -> sparse must differ from dense.
short_p = "The capital of France is"
long_src = ("In a distant valley there lived a clockmaker who repaired instruments no one "
            "else could understand, and every evening he wrote down what the gears told him. ")
n_rep = 1
while len(tok(long_src * n_rep).input_ids) < 1600:
    n_rep += 1
long_p = long_src * n_rep + "\nThe clockmaker's profession is"

lens = {"short": len(tok(short_p).input_ids), "long": len(tok(long_p).input_ids)}

llm = LLM(model=model, tensor_parallel_size=1, block_size=128, max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.30")),
          dtype="bfloat16", enforce_eager=True)

res = {"lens": lens}
for name, prompt in (("short", short_p), ("long", long_p)):
    o = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8, logprobs=20, seed=0))[0]
    step0 = o.outputs[0].logprobs[0]  # {token_id: Logprob} for the FIRST generated position
    res[name] = {
        "top": {str(t): lp.logprob for t, lp in sorted(step0.items(), key=lambda kv: -kv[1].logprob)},
        "tokens": list(o.outputs[0].token_ids),
        "text": o.outputs[0].text,
    }
print("RESULT " + json.dumps(res))
'''


def run(model: str, sparse: str, repo: str, gpu: str) -> dict:
    env = dict(os.environ, MODEL=model, MSA_SPARSE=sparse, REPO=repo,
               CUDA_VISIBLE_DEVICES=gpu, VLLM_NO_USAGE_STATS="1",
               VLLM_ENABLE_V1_MULTIPROCESSING="0", TOKENIZERS_PARALLELISM="false")
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print("\n".join((p.stdout + p.stderr).splitlines()[-25:]))
    raise SystemExit(f"child failed (MSA_SPARSE={sparse})")


def compare(a: dict, b: dict) -> tuple[float, int, int]:
    """Return (max |Δlogprob| over the shared top-K, #shared, #greedy tokens matching)."""
    shared = set(a["top"]) & set(b["top"])
    dmax = max((abs(a["top"][t] - b["top"][t]) for t in shared), default=math.inf)
    match = sum(1 for x, y in zip(a["tokens"], b["tokens"]) if x == y)
    return dmax, len(shared), match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--exact-model", default=DEFAULT_MODEL + "_noshift",
                    help="same checkpoint exported with --no-norm-shift; used to measure the "
                         "w-1 round-trip noise floor")
    ap.add_argument("--degenerate-margin", type=float, default=1.5,
                    help="short-prompt sparse-vs-dense must be <= margin * noise floor")
    ap.add_argument("--divergence-margin", type=float, default=3.0,
                    help="long-prompt sparse-vs-dense must be >= margin * noise floor")
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gpu = os.environ.get("GPU", "0")
    assert os.path.isdir(args.exact_model), (
        f"missing {args.exact_model}. Build it with:\n"
        f"  build_msa_serving_dir.py --ckpt-dir <ckpt> --out {args.exact_model} --no-norm-shift")

    print("1/4 dense  on shifted export ...", flush=True)
    dn = run(args.model, "0", repo, gpu)
    print("2/4 dense  on shifted export AGAIN (determinism control) ...", flush=True)
    dn2 = run(args.model, "0", repo, gpu)
    print("3/4 dense  on unshifted export (exact weights -> noise floor) ...", flush=True)
    ex = run(args.exact_model, "0", repo, gpu)
    print("4/4 sparse on shifted export ...", flush=True)
    sp = run(args.model, "1", repo, gpu)

    print(f"\nprompt lengths: short={sp['lens']['short']} tok, long={sp['lens']['long']} tok "
          f"(k*B_k = 1024)")
    rc = 0

    d_det, _, _ = compare(dn["short"], dn2["short"])
    print(f"\nDETERMINISM (same config twice, must be ~0):  max|Δlogprob|={d_det:.4e}")
    ok = d_det <= 1e-6
    print(f"  {'PASS' if ok else 'FAIL'}  decoding is deterministic, so the floor below is the weights")
    rc |= 0 if ok else 4

    for tag, expect_match in (("short", True), ("long", False)):
        floor, _, _ = compare(dn[tag], ex[tag])          # w-1 round trip alone
        delta, n, m = compare(sp[tag], dn[tag])          # sparse vs dense
        ratio = delta / floor if floor > 0 else float("inf")
        n_tok = len(sp[tag]["tokens"])
        print(f"\n{tag.upper()} ({sp['lens'][tag]} tok, "
              f"{'<' if expect_match else '>'} 1024 -> must {'MATCH' if expect_match else 'DIFFER'}):")
        print(f"  noise floor (w-1 round trip)  = {floor:.4e}")
        print(f"  sparse vs dense               = {delta:.4e}   ({ratio:.2f}x floor, "
              f"{n} shared top-20, greedy {m}/{n_tok})")
        if expect_match:
            ok = ratio <= args.degenerate_margin and m == n_tok
            print(f"  {'PASS' if ok else 'FAIL'}  sparse degenerates to dense "
                  f"(<= {args.degenerate_margin}x floor and all greedy tokens equal)")
            rc |= 0 if ok else 1
        else:
            print(f"  sparse: {sp[tag]['text']!r}")
            print(f"  dense : {dn[tag]['text']!r}")
            ok = ratio >= args.divergence_margin
            print(f"  {'PASS' if ok else 'FAIL'}  selection is load-bearing (>= "
                  f"{args.divergence_margin}x floor); if not, the sparse path is silently dense")
            rc |= 0 if ok else 2

    print("\nS2b DEGENERACY PASS" if rc == 0 else f"\nS2b DEGENERACY FAILED (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
