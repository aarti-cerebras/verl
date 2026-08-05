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
"""P1 / S1 sub-checks 2 and 3 (docs/qwen3_4b_msa/serving_plan.md §6.2).

**(2) Does the Triton top-k compile at our k?** ``plan.md`` §4.2 #1 / open item 9 warns
that ``_topk_index_kernel`` carries ``tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)``
with ``BLOCK_SIZE_T = next_power_of_2(topk)`` (``common/ops/index_topk.py:173, 209``)
and autotune configs ``BLOCK_SIZE_K in {2048, 1024, 512, 256, 128, 64}`` (``:174-183``).

Analytically our production settings are safe -- k=8 -> BLOCK_SIZE_T=8 and k=16 ->
BLOCK_SIZE_T=16, so even the smallest config (64) satisfies the assert. Only k=256
(the served dense-equivalence control we dropped in serving_plan §6.4) knocks out
three of six. This test confirms that empirically rather than by reading, and records
what k=256 actually does so the note in the doc is fact rather than inference.

**(3) The G=4 decode head-axis pad.** ``BLOCK_SIZE_H = max(16, next_pow2(G))``
(``common/ops/sparse_attn.py:227``), so Qwen3's 4 heads/group occupy 16 slots -- a 4x
waste on that axis that M3 (G=16) never sees. Prefill is unaffected: it uses plain
``next_power_of_2(gqa_group_size)`` with no floor (``:46``). Performance only.

Run (inside the serving venv):
  cd <repo> && .devlibs/vllm026/bin/python tests/msa/test_qwen3_msa_topk_compile.py
"""

import sys

import torch

NKV = 8  # == num index heads
HEAD_DIM = 128
SPARSE_BLOCK = 128
SEQ_LEN = 4096  # 32 blocks -- comfortably more than any k under test


def try_topk(topk: int, dev) -> tuple[bool, str]:
    """Compile+run the prefill top-k at this k. Returns (ok, detail)."""
    from vllm.models.minimax_m3.common.ops.index_topk import minimax_m3_index_topk

    max_block = (SEQ_LEN + SPARSE_BLOCK - 1) // SPARSE_BLOCK
    total_q = 128
    # score is [num_idx_heads, total_q, max_block]; strides kept 16-divisible upstream.
    score = torch.randn(NKV, total_q, max_block, device=dev, dtype=torch.float32)
    cu_seqlens_q = torch.tensor([0, total_q], device=dev, dtype=torch.int32)
    prefix_lens = torch.tensor([SEQ_LEN - total_q], device=dev, dtype=torch.int32)
    try:
        out = minimax_m3_index_topk(
            score, cu_seqlens_q, prefix_lens,
            max_query_len=total_q, topk=topk, init_blocks=0, local_blocks=1,
        )
        torch.cuda.synchronize()
        return True, f"out {tuple(out.shape)} dtype={out.dtype}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 2
    dev = torch.device("cuda")
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")

    def npow2(x):
        return 1 << (x - 1).bit_length()

    configs_k = [2048, 1024, 512, 256, 128, 64]

    print("\n(2) Triton top-k compile:")
    print(f"{'topk':>6} {'BLOCK_SIZE_T':>13} {'configs surviving':>18}  result")
    rc = 0
    for topk in (8, 16, 256):
        bt = npow2(topk)
        surviving = sum(1 for bk in configs_k if bk > bt)
        ok, detail = try_topk(topk, dev)
        status = "PASS" if ok else "FAIL"
        print(f"{topk:>6} {bt:>13} {surviving:>13}/{len(configs_k)}  {status}  {detail}")
        # Only k=8 and k=16 are production settings; k=256 is informational.
        if topk in (8, 16) and not ok:
            rc = 1

    print("\n(3) decode head-axis pad at G = num_q_heads/num_kv_heads = 32/8 = 4:")
    g = 4
    bsh = max(16, npow2(g))
    print(f"  decode  BLOCK_SIZE_H = max(16, next_pow2({g})) = {bsh}  -> {bsh // g}x waste on the head axis")
    print(f"  prefill BLOCK_SIZE_H = next_pow2({g})          = {npow2(g)}  -> no pad")
    print("  (performance only; measured end-to-end in P5)")

    print()
    print("S1 sub-checks PASS" if rc == 0 else "S1 sub-check 2 FAILED at a production k")
    return rc


if __name__ == "__main__":
    sys.exit(main())
