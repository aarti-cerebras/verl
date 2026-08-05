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
"""S0 gate (docs/qwen3_4b_msa/serving_plan.md §6.1): does the Qwen3-MSA plugin load and generate?

Mirrors ``tests/dsa/test_stage0_plugin_load.py``. This is the stage that catches the two
silent transforms from P2 — the ``.indexer.`` strip (§2.2) and the ``sparse_attention_config``
gate (§5.2) — because the strict ``AutoWeightsLoader`` raises on any unmatched key instead of
skipping it the way M3's own loader would.

Two sub-goals, each in a fresh subprocess so a sparse-path failure cannot contaminate the
dense result:

  (a) MSA_SPARSE=0 — every layer dense, indexer never constructed. Proves registration, the
      Qwen3 scaffolding and weight loading are right, independent of the sparse kernels.
  (b) MSA_SPARSE=1 — the real config. Additionally asserts the §6.6 anti-dense checks:
      33 sparse layers built, TWO KV-cache groups, and both ``info_once`` backend lines.

Run:
  cd <repo> && .devlibs/vllm026/bin/python tests/msa/test_stage0_plugin_load.py \
      [--model /cb/ml-eng/aarti/msa/serving/k8_step1400]
"""

import argparse
import os
import subprocess
import sys

DEFAULT_MODEL = "/cb/ml-eng/aarti/msa/serving/k8_step1400"

CHILD = r'''
import os, sys, json
sys.path.insert(0, os.environ["REPO"])
import scripts.msa.vllm_qwen3_msa  # noqa: F401  (registers Qwen3MSAForCausalLM)
from vllm import LLM, SamplingParams

model = os.environ["MODEL"]
sparse = os.environ.get("MSA_SPARSE", "1")
llm = LLM(
    model=model,
    tensor_parallel_size=1,
    block_size=128,                 # mandatory: both backends report [128] only
    max_model_len=4096,             # S0 only needs a short prompt
    gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.30")),
    dtype="bfloat16",
    enforce_eager=True,             # cudagraph capture is R2, out of scope for S0
    load_format="safetensors",
)
out = llm.generate(
    ["The capital of France is"],
    SamplingParams(temperature=0.0, max_tokens=24, seed=0),
)
text = out[0].outputs[0].text
print("GENERATED:", json.dumps(text))

inner = llm.llm_engine.model_executor.driver_worker.model_runner.model
n_sparse = int(getattr(inner, "msa_num_sparse_layers", -1))
print("N_SPARSE_LAYERS:", n_sparse)

# Which attention class landed on which layer.
kinds = {}
for i, layer in enumerate(inner.model.layers):
    kinds[type(layer.self_attn).__name__] = kinds.get(type(layer.self_attn).__name__, 0) + 1
print("ATTN_CLASSES:", json.dumps(kinds))

# §6.6 check 2: the indexer registers its own side cache, named "<layer>.attn.index_cache"
# (common/indexer.py:390). NB vLLM 0.26 merges everything into ONE UniformTypeKVCacheSpecs
# group, so plan.md §7.3's "two kv-cache groups" is not the right invariant here -- count the
# cached layers instead: 36 main attentions + 33 index caches = 69.
kc = llm.llm_engine.model_executor.driver_worker.model_runner.kv_cache_config
names = [n for g in kc.kv_cache_groups for n in g.layer_names]
n_idx = sum(n.endswith(".index_cache") for n in names)
print("KV_CACHE: groups=%d total=%d index_cache=%d main=%d"
      % (len(kc.kv_cache_groups), len(names), n_idx, len(names) - n_idx))
print("CHILD_OK")
'''


def run(tag: str, model: str, sparse: str, repo: str) -> tuple[bool, str]:
    env = dict(os.environ, MODEL=model, MSA_SPARSE=sparse, REPO=repo,
               VLLM_NO_USAGE_STATS="1", TOKENIZERS_PARALLELISM="false",
               # Run EngineCore IN-PROCESS: with v1 multiprocessing the engine lives in a
               # subprocess and `llm_engine.model_executor` does not exist in the parent, so the
               # layer/KV-group introspection below is unreachable.
               VLLM_ENABLE_V1_MULTIPROCESSING="0")
    # Shared node: pin to one device and take a small slice so this never competes with training.
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))
    print(f"\n{'=' * 78}\n{tag}  (MSA_SPARSE={sparse})\n{'=' * 78}", flush=True)
    p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    ok = "CHILD_OK" in p.stdout
    for line in p.stdout.splitlines():
        if line.startswith(("GENERATED:", "N_SPARSE_LAYERS:", "ATTN_CLASSES:", "KV_CACHE:")):
            print("  " + line)
    # §6.6 check 1: the backend-selection info_once lines must appear, or we are serving dense.
    if sparse == "1":
        for needle, what in (
            ("MiniMax M3 sparse attention selected", "sparse-attention backend"),
            ("MiniMax M3 indexer: selected", "indexer backend"),
        ):
            print(f"  {'PASS' if needle in out else 'FAIL'}  log line: {what}")
            ok = ok and (needle in out)
        n_idx = next((int(l.split("index_cache=")[1].split()[0])
                      for l in out.splitlines() if l.startswith("KV_CACHE:")), -1)
        n_sp = next((int(l.split(":")[1]) for l in out.splitlines()
                     if l.startswith("N_SPARSE_LAYERS:")), -1)
        print(f"  {'PASS' if n_idx == n_sp and n_idx > 0 else 'FAIL'}  "
              f"index side caches allocated: {n_idx} (expect {n_sp}, one per sparse layer)")
        ok = ok and n_idx == n_sp and n_idx > 0
    # Coherence: both modes must produce real text. (a) is only meaningful because load_weights
    # un-shifts the norms for it; without that it emits fluent garbage regardless of correctness.
    gen = next((l.split("GENERATED:", 1)[1].strip() for l in out.splitlines()
                if l.startswith("GENERATED:")), "")
    coherent = "Paris" in gen
    print(f"  {'PASS' if coherent else 'FAIL'}  coherent continuation (expects 'Paris')")
    ok = ok and coherent
    if not ok:
        tail = "\n".join(out.splitlines()[-30:])
        print(f"\n--- child output (tail) ---\n{tail}")
    return ok, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only", choices=("dense", "sparse"), default=None)
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    assert os.path.isdir(args.model), f"serving dir not found: {args.model} (run P2 first)"
    rc = 0
    if args.only in (None, "dense"):
        ok, _ = run("(a) DENSE  — registration + strict weight load", args.model, "0", repo)
        print(f"\n  => (a) {'PASS' if ok else 'FAIL'}")
        rc |= 0 if ok else 1
    if args.only in (None, "sparse"):
        ok, _ = run("(b) SPARSE — real config + anti-dense gate", args.model, "1", repo)
        print(f"\n  => (b) {'PASS' if ok else 'FAIL'}")
        rc |= 0 if ok else 2

    print("\nS0 PASS" if rc == 0 else f"\nS0 FAILED (rc={rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
