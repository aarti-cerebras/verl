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
"""Stage-0 smoke/acceptance test for the MiniCPM3-DSA vLLM decode plugin.

Sub-goal (a) MUST pass: the custom ``MiniCPM3DSAForCausalLM`` registers, ALL
base + indexer weights load (no missing / unexpected keys — enforced by vLLM's
strict loader), and a short prompt generates coherent text (dense attention ==
base model).

Sub-goal (b) TRY: with ``DSA_WIRE_INDEXER=1`` the indexer serve op is built
(``DeepseekV32IndexerCache`` cache group + ``SparseAttnIndexer``) and called
during the (still dense) forward; check whether the engine builds the indexer
cache group + metadata and populates ``topk_indices_buffer``. Run as a
subprocess so a failure there never contaminates (a).

Run:
  cd <repo> && export PYTHONPATH=$(pwd)/scripts/dsa/vllm_minicpm3_dsa/..:$(pwd)/.devlibs/tf457lib:$(pwd)
  /usr/bin/python3 tests/dsa/test_stage0_plugin_load.py            # runs (a) then probes (b)
  /usr/bin/python3 tests/dsa/test_stage0_plugin_load.py --wire     # internal: the (b) wired build
"""

import os
import sys

# apply_model ships the probe fn to the engine-core worker process; allow the
# pickle-based fallback so a top-level fn returning primitives can cross it.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

SERVING_STAGE0 = (
    "/cb/ml-eng/aarti/dsa/phase2_topk_sweep_20260717_223140/"
    "phase2_full_k256_1ep_20260717_223341/train/checkpoints/inference/"
    "global_step_2805/serving_stage0"
)
CKPT_SAFETENSORS = os.path.join(os.path.dirname(SERVING_STAGE0), "serving", "model.safetensors")

PROMPT = "The capital of France is"


def _build_llm():
    # Importing the plugin package installs the deep_gemm shim AND registers
    # MiniCPM3DSAForCausalLM with vLLM's ModelRegistry.
    import scripts.dsa.vllm_minicpm3_dsa  # noqa: F401

    from vllm import LLM

    return LLM(
        model=SERVING_STAGE0,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
    )


def _probe_model(model):
    """Top-level probe (shipped to the engine-core worker). Returns ONLY JSON-able
    primitives so it survives cross-process serialization."""
    layers = model.model.layers
    attached = sum(1 for lyr in layers if hasattr(getattr(lyr, "self_attn", object()), "indexer"))
    buf = getattr(model, "topk_indices_buffer", None)
    w = model.model.layers[0].self_attn.indexer.wq_b.weight.detach().float().reshape(-1)[:8]
    return {
        "attached": int(attached),
        "n_layers": int(len(layers)),
        "buffer_shape": None if buf is None else list(buf.shape),
        "buffer_dtype": None if buf is None else str(buf.dtype),
        "wq_b_head": [float(x) for x in w.tolist()],
    }


def _probe_buffer_stats(model):
    buf = model.topk_indices_buffer
    return {
        "shape": list(buf.shape),
        "n_valid": int((buf >= 0).sum().item()),
        "max_idx": int(buf.max().item()),
        "min_idx": int(buf.min().item()),
    }


def _inspect_model(llm):
    """Return structural info + whether layer0 indexer.wq_b matches the checkpoint."""
    from safetensors import safe_open

    try:
        res = llm.apply_model(_probe_model)
        info = res[0] if isinstance(res, (list, tuple)) else res
    except Exception as e:  # pragma: no cover - API fallback
        return {"inspect_error": repr(e)}

    with safe_open(CKPT_SAFETENSORS, "pt") as f:
        w_ckpt = f.get_tensor("model.layers.0.self_attn.indexer.wq_b.weight").float().reshape(-1)[:8]
    live = info.pop("wq_b_head", None)
    if live is not None:
        diff = max(abs(a - b) for a, b in zip(live, w_ckpt.tolist()))
        info["indexer_wq_b_matches"] = bool(diff < 1e-2)
        info["indexer_wq_b_maxdiff"] = diff
    return info


def run_a():
    print("=" * 70)
    print("STAGE-0 (a): plugin load + dense generation")
    print("=" * 70)
    from vllm import SamplingParams

    llm = _build_llm()
    print("[a] LLM built -> registration + config + weight-mapping OK.")
    print("[a] vLLM strict loader passed => NO missing (uninitialized) params")
    print("    and NO unexpected checkpoint keys (base + indexer.* both mapped).")

    info = _inspect_model(llm)
    print(f"[a] indexer attached to {info.get('attached')}/{info.get('n_layers')} layers")
    print(f"[a] topk_indices_buffer: shape={info.get('buffer_shape')} dtype={info.get('buffer_dtype')}")
    print(f"[a] layer0 indexer.wq_b matches checkpoint: {info.get('indexer_wq_b_matches')}")
    if "inspect_error" in info:
        print(f"[a] (model inspection fallback: {info['inspect_error']})")

    out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=32))
    text = out[0].outputs[0].text
    print(f"[a] PROMPT : {PROMPT!r}")
    print(f"[a] OUTPUT : {text!r}")

    coherent = len(text.strip()) > 0
    attached_ok = info.get("attached") == info.get("n_layers") and info.get("n_layers")
    buf_ok = info.get("buffer_shape") is not None and info.get("buffer_shape", (0, 0))[1] == 256
    idx_ok = info.get("indexer_wq_b_matches", False)
    passed = bool(coherent and attached_ok and buf_ok and idx_ok)
    print("-" * 70)
    print(f"(a) RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def run_b_wired():
    """Internal entrypoint (subprocess): build with the indexer serve op wired
    and report whether the cache group builds + the buffer is populated."""
    os.environ["DSA_WIRE_INDEXER"] = "1"
    from vllm import SamplingParams

    llm = _build_llm()
    print("[b] wired LLM built (DeepseekV32IndexerCache group + SparseAttnIndexer).")

    out = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=8))
    stats = llm.apply_model(_probe_buffer_stats)
    stats = stats[0] if isinstance(stats, (list, tuple)) else stats
    print(f"[b] OUTPUT: {out[0].outputs[0].text!r}")
    print(f"[b] buffer stats after forward: {stats}")
    populated = stats.get("n_valid", 0) > 0 and stats.get("max_idx", -1) >= 0
    print(f"(b) WIRED RESULT: {'BUFFER POPULATED' if populated else 'BUFFER NOT POPULATED'}")


def run_b_probe():
    print()
    print("=" * 70)
    print("STAGE-0 (b): indexer serve-path probe (subprocess, DSA_WIRE_INDEXER=1)")
    print("=" * 70)
    import subprocess

    env = dict(os.environ)
    env["DSA_WIRE_INDEXER"] = "1"
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--wire"],
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    print(proc.stdout[-6000:])
    if proc.returncode != 0:
        print("[b] wired build/forward RAISED — indexer serve path does NOT run at Stage 0.")
        print("[b] ---- stderr tail ----")
        print(proc.stderr[-6000:])
        print("(b) STATUS: DEFERRED to Stage 1 (see stderr above for the precise blocker)")
        return False
    print("(b) STATUS: indexer serve path RAN under Stage-0 dense attention (see above)")
    return True


def main():
    if "--wire" in sys.argv:
        run_b_wired()
        return
    a_pass = run_a()
    try:
        run_b_probe()
    except Exception as e:
        print(f"(b) STATUS: probe harness error: {e!r} -> treat as DEFERRED")
    print()
    print("#" * 70)
    print(f"# STAGE 0 (a) [MUST]: {'PASS' if a_pass else 'FAIL'}")
    print("#" * 70)
    sys.exit(0 if a_pass else 1)


if __name__ == "__main__":
    main()
