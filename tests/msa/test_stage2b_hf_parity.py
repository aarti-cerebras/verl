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
"""S2b/S3 HF parity (docs/qwen3_4b_msa/serving_plan.md §6.3, §6.5), P4.

**The question P4 exists to answer:** does the function vLLM serves equal the function we
trained? S2b's degeneracy test showed sparse == dense on short prompts and != on long ones, but
both sides of that were vLLM — it cannot detect an error shared by both. This compares vLLM
against the *training* forward itself.

Two checks, both on prompts LONGER than ``k * B_k = 1024`` so selection is real:

  **first-token** (§6.3) — vLLM's step-0 top-20 logprobs vs the training forward's logits at the
  last prompt position.
  **teacher-forced** (§6.5) — the per-position logprob of every actual next token across the
  whole prompt, i.e. hundreds of independent comparisons rather than one. Teacher forcing is a
  single full-sequence prefill, which is why a prefill-only oracle suffices (§1).

**Judged against the measured noise floor, never against zero.** The oracle runs on the
``--no-norm-shift`` export (exact trained weights, standard RMSNorm) while vLLM serves the
shifted one, so this comparison *inherently* includes the ``w - 1`` round trip — which
`test_stage2b_degeneracy.py` measured at ~7e-2 on first-step logprobs. Different kernels and a
different transformers version add more. A first-token agreement at or near that floor is the
expected result; zero is not achievable and demanding it would manufacture a failure.

Prerequisite — run the oracle first, in the TRAINING env:
  /usr/bin/python3 scripts/msa/hf_msa_oracle.py \
      --model /cb/ml-eng/aarti/msa/serving/k8_step1400_noshift \
      --tokens /tmp/p4/tokens.json --out /tmp/p4/oracle_sparse.pt

Then, in the venv:
  GPU=0 .devlibs/vllm026/bin/python tests/msa/test_stage2b_hf_parity.py
"""

import argparse
import json
import os
import subprocess
import sys

import torch

DEFAULT_MODEL = "/cb/ml-eng/aarti/msa/serving/k8_step1400"

CHILD = r'''
import os, sys, json, torch
sys.path.insert(0, os.environ["REPO"])
import scripts.msa.vllm_qwen3_msa  # noqa: F401
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

toks = json.load(open(os.environ["TOKENS"]))
llm = LLM(model=os.environ["MODEL"], tensor_parallel_size=1, block_size=128, max_model_len=4096,
          gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.30")),
          dtype="bfloat16", enforce_eager=True)

out = {}
for name, ids in toks.items():
    # Feed token ids directly -- tokenisation must not be a variable between the two sides.
    o = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                     SamplingParams(temperature=0.0, max_tokens=1, logprobs=20,
                                    prompt_logprobs=0, seed=0))[0]
    step0 = o.outputs[0].logprobs[0]
    # prompt_logprobs[i] is the logprob of prompt token i given 0..i-1; index 0 is None.
    pl = [None if p is None else p[t].logprob for p, t in zip(o.prompt_logprobs, ids)]
    out[name] = {
        "top": {str(t): lp.logprob for t, lp in step0.items()},
        "prompt_logprobs": pl[1:],
    }
json.dump(out, open(os.environ["OUT"], "w"))
print("CHILD_OK")
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", default="/tmp/p4/tokens.json")
    ap.add_argument("--oracle", default="/tmp/p4/oracle_sparse.pt")
    ap.add_argument("--floor", type=float, default=7e-2,
                    help="first-step noise floor measured by test_stage2b_degeneracy.py")
    ap.add_argument("--margin", type=float, default=3.0, help="allowed multiple of the floor")
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    assert os.path.exists(args.oracle), (
        f"missing {args.oracle} -- run scripts/msa/hf_msa_oracle.py in the TRAINING env first "
        f"(see this file's docstring)")
    ora = torch.load(args.oracle, weights_only=False)
    assert ora["mode"] == "sparse", f"oracle is {ora['mode']}, expected sparse"
    print(f"oracle: {ora['model']} (mode={ora['mode']})")

    vout = "/tmp/p4/vllm_sparse.json"
    env = dict(os.environ, MODEL=args.model, TOKENS=args.tokens, OUT=vout, REPO=repo,
               MSA_SPARSE="1", CUDA_VISIBLE_DEVICES=os.environ.get("GPU", "0"),
               VLLM_NO_USAGE_STATS="1", VLLM_ENABLE_V1_MULTIPROCESSING="0",
               TOKENIZERS_PARALLELISM="false")
    print("running vLLM sparse on the same token ids ...", flush=True)
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    if "CHILD_OK" not in p.stdout:
        print("\n".join((p.stdout + p.stderr).splitlines()[-25:]))
        return 1
    vll = json.load(open(vout))

    tol = args.floor * args.margin
    print(f"\nnoise floor {args.floor:.2e} (w-1 round trip), tolerance {args.margin}x = {tol:.2e}\n")
    rc = 0
    for name in sorted(ora["data"]):
        o, v = ora["data"][name], vll[name]
        n_tok = o["n_tokens"]
        long_enough = n_tok > 1024
        # --- first token -----------------------------------------------------------------
        o_top = {str(int(i)): float(l) for i, l in zip(o["last_top_ids"], o["last_top_logprobs"])}
        shared = set(o_top) & set(v["top"])
        dmax = max((abs(o_top[t] - v["top"][t]) for t in shared), default=float("inf"))
        o_top1 = str(int(o["last_top_ids"][0]))
        v_top1 = max(v["top"], key=lambda t: v["top"][t])
        # --- teacher forced --------------------------------------------------------------
        ot = o["tf_logprobs"].float()
        vt = torch.tensor([x for x in v["prompt_logprobs"]], dtype=torch.float32)
        n = min(len(ot), len(vt))
        d = (ot[:n] - vt[:n]).abs()

        print(f"{name}  ({n_tok} tokens, selection {'REAL' if long_enough else 'degenerate'}):")
        print(f"  first token : top1 {'MATCH' if o_top1 == v_top1 else 'DIFFER'} "
              f"(oracle {o_top1}, vllm {v_top1}) | max|Δlogprob|={dmax:.4e} over {len(shared)} shared")
        print(f"  teacher-fcd : {n} positions | max|Δ|={d.max():.4e} mean|Δ|={d.mean():.4e} "
              f"p99={d.quantile(0.99):.4e} | >{tol:.2e}: {(d > tol).sum().item()}/{n}")
        # Gate on robust statistics, NOT on max. `dmax` is a max over ~20 low-probability tokens at a
        # single position; it is dominated by outliers and cannot be driven to zero. Two independent
        # reasons a floor exists on this comparison, both measured rather than assumed:
        #   * the w-1 round trip alone moves first-step logprobs ~7e-2 (test_stage2b_degeneracy.py);
        #   * ~1.5% of block selections flip on near-ties, since the training and serving score
        #     matmuls reduce in different orders (test_qwen3_msa_index_parity.py).
        # So the claim under test is "same function up to numerical noise and occasional near-tie
        # flips", and these are its observable consequences.
        frac_out = (d > tol).float().mean().item()
        ok = (o_top1 == v_top1) and d.mean().item() <= args.floor and frac_out < 0.02
        print(f"    gate: top1 match={o_top1 == v_top1} | mean|Δ|={d.mean():.3e} <= floor "
              f"{args.floor:.2e} | outliers {frac_out:.2%} < 2%   (max|Δ| {dmax:.3e} reported, "
              f"not gated)")
        print(f"  {'PASS' if ok else 'FAIL'}\n")
        rc |= 0 if ok else 1

    print("HF PARITY PASS" if rc == 0 else "HF PARITY FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
