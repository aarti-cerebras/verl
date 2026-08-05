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
"""P0 acceptance (docs/qwen3_4b_msa/serving_plan.md §9 P0): does this venv have
everything the Qwen3-MSA port needs from vLLM's MiniMax-M3 implementation?

serving_plan §2.1 established that ``vllm/models/minimax_m3/`` ships in released tags
(present at v0.24.0/v0.25.0/v0.26.0, absent at v0.21.0) and that 0.26.0 pins the torch
we already run -- so no source build. What was **[UNVERIFIED]** is whether the binary
wheel carries the *compiled* ``fused_minimax_m3_qknorm_rope_kv_insert`` symbol, since
only the .cu source was confirmed present in the tag. That is check 4 below, and it is
the one that can still force Route B (§4.3).

Run:
  cd <repo> && .devlibs/vllm026/bin/python tests/msa/test_vllm026_env.py
"""

import importlib
import sys

CHECKS: list[tuple[str, str]] = [
    # (label, dotted path to import)
    ("sparse attention impl", "vllm.models.minimax_m3.common.sparse_attention"),
    ("indexer + side cache", "vllm.models.minimax_m3.common.indexer"),
    ("index score/top-k kernels", "vllm.models.minimax_m3.common.ops.index_topk"),
    ("block-sparse attn kernels", "vllm.models.minimax_m3.common.ops.sparse_attn"),
    ("M3 model (nvidia)", "vllm.models.minimax_m3.nvidia.model"),
    ("stock Qwen3", "vllm.model_executor.models.qwen3"),
]

SYMBOLS: list[tuple[str, str, str]] = [
    # (label, module, attribute) -- the classes the port subclasses or constructs
    ("MiniMaxM3SparseAttention", "vllm.models.minimax_m3.nvidia.model", "MiniMaxM3SparseAttention"),
    ("MiniMaxM3Indexer", "vllm.models.minimax_m3.common.indexer", "MiniMaxM3Indexer"),
    ("MiniMaxM3SparseBackend", "vllm.models.minimax_m3.common.sparse_attention", "MiniMaxM3SparseBackend"),
    ("Qwen3ForCausalLM", "vllm.model_executor.models.qwen3", "Qwen3ForCausalLM"),
]


def main() -> int:
    rc = 0

    import torch

    print("=== environment ===")
    print(f"python      {sys.version.split()[0]}")
    print(f"executable  {sys.executable}")
    print(f"torch       {torch.__version__} (cuda {torch.version.cuda})")
    try:
        import vllm

        print(f"vllm        {vllm.__version__}")
        if not vllm.__version__.startswith("0.26."):
            print(f"  WARN: expected 0.26.x, got {vllm.__version__}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: cannot import vllm: {e!r}")
        return 1
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"gpu         {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
        if cap[0] != 9:
            print(f"  NOTE: serving_plan assumes SM90 (Triton path); this is sm_{cap[0]}{cap[1]}")
    else:
        print("gpu         NONE VISIBLE (import checks still valid)")

    print("\n=== 1-2. module imports ===")
    for label, path in CHECKS:
        try:
            importlib.import_module(path)
            print(f"  PASS  {label:28s} {path}")
        except Exception as e:  # noqa: BLE001
            rc = 1
            print(f"  FAIL  {label:28s} {path}\n          {type(e).__name__}: {str(e)[:160]}")

    print("\n=== 3. classes the port needs ===")
    for label, mod, attr in SYMBOLS:
        try:
            m = importlib.import_module(mod)
            getattr(m, attr)
            print(f"  PASS  {label}")
        except Exception as e:  # noqa: BLE001
            rc = 1
            print(f"  FAIL  {label}: {type(e).__name__}: {str(e)[:160]}")

    print("\n=== 4. compiled fused op in the wheel  [the P0 unknown] ===")
    try:
        from vllm import _custom_ops as ops

        if hasattr(ops, "fused_minimax_m3_qknorm_rope_kv_insert"):
            print("  PASS  vllm._custom_ops.fused_minimax_m3_qknorm_rope_kv_insert present")
            print("        -> Route A stays viable; numerics still gated by the P1 rotary_dim=128 test")
        else:
            rc = 1
            print("  FAIL  symbol ABSENT from the wheel -> Route B (own forward, no fused kernel)")
    except Exception as e:  # noqa: BLE001
        rc = 1
        print(f"  FAIL  cannot import vllm._custom_ops: {type(e).__name__}: {str(e)[:160]}")

    print("\n=== 5. block-size contract ===")
    try:
        from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerBackend
        from vllm.models.minimax_m3.common.sparse_attention import MiniMaxM3SparseBackend

        for name, cls in (("sparse attn", MiniMaxM3SparseBackend), ("indexer", MiniMaxM3IndexerBackend)):
            sizes = cls.get_supported_kernel_block_sizes()
            ok = sizes == [128]
            rc = rc or (0 if ok else 1)
            print(f"  {'PASS' if ok else 'FAIL'}  {name:12s} get_supported_kernel_block_sizes() = {sizes}")
        print("        -> serve with --block-size 128")
    except Exception as e:  # noqa: BLE001
        print(f"  WARN  could not query block sizes: {type(e).__name__}: {str(e)[:160]}")

    print()
    print("P0 PASS" if rc == 0 else "P0 FAILED -- see above")
    return rc


if __name__ == "__main__":
    sys.exit(main())
