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
"""S3 decode parity (docs/qwen3_4b_msa/serving_plan.md §6.5), P4. Mirrors tests/dsa/test_stage3*.

**What the earlier tests could not reach.** S2b compares vLLM's *prefill* against the training
forward. But evals spend almost all their tokens in DECODE, which on the serving side is a
different set of kernels entirely — `_decode_index_score_kernel`, split-K partial top-k plus a
merge, and `_gqa_sparse_decode_kernel`. Prefill parity does not transfer to them. This test
exercises the decode path and checks it stayed on the trained function.

**Method: teacher forcing, not free-running comparison.** Greedy-generate G tokens from the vLLM
server (every one produced by the decode kernels), then push `prompt + generation` through the
training forward in ONE prefill and compare per position. Free-running equality is the wrong
gate — a single bf16 flip cascades, so divergence there is expected and says nothing; it is
reported as secondary. This is the finding recorded in DSA's Stage 3.

Teacher forcing is a full-sequence prefill, which is why a prefill-only oracle suffices even
though the training module has no decode path (§1).

**Judged against the same measured floors as S2b** — the `w - 1` round trip (~7e-2 on logprobs)
and ~1.5% near-tie selection flips. Zero is not reachable; see test_stage2b_hf_parity.py.

Run (needs both envs; does the vLLM half itself, then tells you the oracle command):
  GPU=0 .devlibs/vllm026/bin/python tests/msa/test_stage3_decode_parity.py --step generate
  /usr/bin/python3 scripts/msa/hf_msa_oracle.py --model <noshift> \
      --tokens /tmp/p4/s3_tokens.json --out /tmp/p4/s3_oracle.pt
  GPU=0 .devlibs/vllm026/bin/python tests/msa/test_stage3_decode_parity.py --step compare
"""

import argparse
import json
import os
import subprocess
import sys

import torch

MODEL = "/cb/ml-eng/aarti/msa/serving/k8_step1400"
NOSHIFT = MODEL + "_noshift"
TOKENS = "/tmp/p4/s3_tokens.json"
VOUT = "/tmp/p4/s3_vllm.json"
ORACLE = "/tmp/p4/s3_oracle.pt"

GEN = r'''
import os, sys, json
sys.path.insert(0, os.environ["REPO"])
import scripts.msa.vllm_qwen3_msa  # noqa: F401
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

n_gen = int(os.environ["N_GEN"])
tok = AutoTokenizer.from_pretrained(os.environ["MODEL"])
# Prompts LONGER than k*B_k = 1024 so selection is real during decode, and varied in kind:
# prose, code, math -- DSA Stage 3 used the same three.
src = {
 "prose": "In a distant valley there lived a clockmaker who repaired instruments no one else could "
          "understand, and every evening he wrote down what the gears told him. ",
 "code":  "def solve(xs):\n    total = 0\n    for x in xs:\n        if x % 3 == 0:\n            "
          "total += x * x\n    return total\n\n# The function above computes a sum. ",
 "math":  "Let a_1 = 1 and a_{n+1} = 2 a_n + 3 for all n >= 1. We compute the first several terms "
          "and look for a closed form. ",
}
prompts = {}
for k, v in src.items():
    n = 1
    while len(tok(v * n).input_ids) < 1200:
        n += 1
    prompts[k] = tok(v * n).input_ids

llm = LLM(model=os.environ["MODEL"], tensor_parallel_size=1, block_size=128, max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.30")),
          dtype="bfloat16", enforce_eager=True)

full, vll = {}, {}
for name, ids in prompts.items():
    # logprobs=0 returns the logprob of each SAMPLED token as computed by the DECODE kernels --
    # that is the quantity S3 is about. Re-scoring the finished sequence with prompt_logprobs would
    # run PREFILL over it and test nothing new (an earlier version of this file did exactly that).
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                     SamplingParams(temperature=0.0, max_tokens=n_gen, logprobs=0, seed=0))[0]
    gen = list(o.outputs[0].token_ids)
    decode_lp = [lp[t].logprob for lp, t in zip(o.outputs[0].logprobs, gen)]
    seq = list(ids) + gen
    full[name] = seq
    # Re-run the FULL sequence as a prompt to get vLLM's own per-position logprobs over it. The
    # generated part was produced by the decode kernels; scoring it back through prefill is how we
    # get a like-for-like teacher-forced number from the serving side.
    o2 = llm.generate([TokensPrompt(prompt_token_ids=seq)],
                      SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0, seed=0))[0]
    pl = [None if p is None else p[t].logprob for p, t in zip(o2.prompt_logprobs, seq)]
    vll[name] = {"n_prompt": len(ids), "n_gen": len(gen), "logprobs": pl[1:],
                 "decode_lp": decode_lp, "gen_text": o.outputs[0].text[:120]}

json.dump(full, open(os.environ["TOKENS"], "w"))
json.dump(vll, open(os.environ["VOUT"], "w"))
print("GEN_OK")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("generate", "compare"), required=True)
    ap.add_argument("--n-gen", type=int, default=256,
                    help="48 cannot resolve an outlier RATE: one outlier is 2.1 points, so a "
                         "5%% gate is inside the sampling error. 256 gives real power.")
    ap.add_argument("--floor", type=float, default=7e-2)
    ap.add_argument("--margin", type=float, default=3.0)
    a = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if a.step == "generate":
        env = dict(os.environ, MODEL=MODEL, TOKENS=TOKENS, VOUT=VOUT, REPO=repo,
                   N_GEN=str(a.n_gen), MSA_SPARSE="1",
                   CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "0"), VLLM_NO_USAGE_STATS="1",
                   VLLM_ENABLE_V1_MULTIPROCESSING="0", TOKENIZERS_PARALLELISM="false")
        p = subprocess.run([sys.executable, "-c", GEN], env=env, capture_output=True, text=True)
        if "GEN_OK" not in p.stdout:
            print("\n".join((p.stdout + p.stderr).splitlines()[-25:]))
            return 1
        v = json.load(open(VOUT))
        for k, d in v.items():
            print(f"  {k}: prompt {d['n_prompt']} + generated {d['n_gen']} tokens | "
                  f"{d['gen_text']!r}")
        print(f"\nwrote {TOKENS} and {VOUT}\nNow run the oracle in the TRAINING env:\n"
              f"  /usr/bin/python3 scripts/msa/hf_msa_oracle.py --model {NOSHIFT} \\\n"
              f"      --tokens {TOKENS} --out {ORACLE}\n"
              f"then: GPU=0 .devlibs/vllm026/bin/python {__file__} --step compare")
        return 0

    assert os.path.exists(ORACLE), f"missing {ORACLE}; run the oracle step (see --step generate)"
    ora = torch.load(ORACLE, weights_only=False)["data"]
    vll = json.load(open(VOUT))
    tol = a.floor * a.margin
    print(f"floor {a.floor:.2e}, outlier threshold {tol:.2e}\n")

    rc = 0
    for name in sorted(vll):
        d_v = vll[name]
        o = ora[name]
        n_p, n_g = d_v["n_prompt"], d_v["n_gen"]
        ot = o["tf_logprobs"].float()
        vt = torch.tensor(d_v["logprobs"], dtype=torch.float32)
        n = min(len(ot), len(vt))
        d = (ot[:n] - vt[:n]).abs()
        pre = d[: n_p - 1]                      # prefill control, vLLM prefill vs oracle
        # TRUE decode comparison: vLLM's per-step logprobs FROM THE DECODE KERNELS vs the oracle's
        # teacher-forced logprob for the same token. ot index i is the logprob of token i+1, so the
        # first generated token sits at ot[n_p-1].
        dlp = torch.tensor(d_v["decode_lp"], dtype=torch.float32)
        m_ = min(len(dlp), len(ot) - (n_p - 1))
        dec = (ot[n_p - 1: n_p - 1 + m_] - dlp[:m_]).abs()
        # Secondary: vLLM PREFILL re-scored over the same generated tokens. If `dec` is much worse
        # than this, the gap is the decode kernels; if both are bad, it is the tokens themselves.
        dec_pre = d[n_p - 1:]
        print(f"{name}  ({n_p} prompt + {n_g} generated):")
        print(f"  prefill positions : {len(pre):5d} | mean|Δ|={pre.mean():.4e} "
              f"p99={pre.quantile(0.99):.4e} | >{tol:.1e}: {(pre > tol).sum().item()}")
        print(f"  DECODE kernels    : {len(dec):5d} | mean|Δ|={dec.mean():.4e} "
              f"p99={dec.quantile(0.99):.4e} | >{tol:.1e}: {(dec > tol).sum().item()}")
        print(f"  (same tokens, vLLM PREFILL re-score: mean|Δ|={dec_pre.mean():.4e} "
              f"| >{tol:.1e}: {(dec_pre > tol).sum().item()})")
        # Gate decode against PREFILL ON THE SAME SEQUENCE, not against an absolute constant. The
        # question S3 asks is narrow: do the decode kernels (split-K score, partial top-k + merge,
        # sparse decode attend) degrade anything relative to the prefill kernels we already
        # validated in S2b? Same weights, same tokens, same floors -- so prefill is the control and
        # any absolute threshold would just re-litigate the w-1 and near-tie floors.
        f_dec = (dec > tol).float().mean().item()
        f_pre = (pre > tol).float().mean().item()
        ok = (dec.mean().item() <= max(a.floor, 2.0 * pre.mean().item())
              and f_dec <= max(3.0 * f_pre, 3.0 / max(len(dec), 1)))
        print(f"  gate: decode mean {dec.mean():.3e} <= max(floor {a.floor:.1e}, 2x prefill "
              f"{2 * pre.mean():.3e}) | outlier rate {f_dec:.1%} vs prefill {f_pre:.1%} "
              f"-> {'PASS' if ok else 'FAIL'}\n")
        rc |= 0 if ok else 1

    print("S3 DECODE PARITY PASS" if rc == 0 else "S3 DECODE PARITY FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
