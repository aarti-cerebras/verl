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
"""Memory-SCALING invariants for the MSA tiled KL. Catches the bug class that OOM'd every 32K run.

The bug: per-tile tensors were passed as arguments to ``torch.utils.checkpoint``. Checkpoint saves its
input tensors so it can replay the function in backward, and holds them until backward runs — so anything
freshly allocated per tile is retained for the whole layer:

    p_blk [1, H_kv, T_q, T] fp32  537 MB  +  bias 67 MB  +  allow 17 MB
    = 621 MB x 64 tiles = 39.7 GB per layer at 32K/T_q=512, x33 layers = 1.31 TB.

The invariant that catches it, and the reason the existing suites missed it:

    ** At fixed sequence length, PEAK MEMORY MUST NOT GROW WITH THE NUMBER OF TILES. **

Retained-per-tile bugs scale as ``n_tiles x per_tile``, which is *independent of tile size* — halving
``kl_block_size`` doubles the tile count and leaves the total unchanged. So a run at T_q=T/2 and one at
T_q=T/8 must peak at the same memory; if the small-tile run peaks higher, per-tile state is being
retained. The other suites run at T=512 with T_q=128 (4 tiles), where 64x retention is invisible.

Run:
  cd <repo> && PYTHONPATH=$(pwd) python3 tests/msa/test_qwen3_msa_memory_scaling.py
"""

import argparse
import sys

import torch

from verl.models.transformers.qwen3_msa import (
    attach_indexers,
    build_msa_config,
    freeze_base_train_indexer,
    install_kl_accumulation,
    qwen3_msa_attn_forward,
)

OK, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"  [{OK if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def peak_for(model, cfg, ids, mode, kl_block, ckpt=True):
    """Peak allocated bytes for one fwd+bwd at a given tile size."""
    cfg.mode, cfg.kl_block_size, cfg.kl_checkpoint = mode, kl_block, ckpt
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    model(input_ids=ids)
    model._msa_indexer_kl.backward()
    peak = torch.cuda.max_memory_allocated() - base
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/cb/ml-eng/aarti/models/qwen3_0p6b")
    ap.add_argument("--seq-len", type=int, default=4096, help="must be >> kl_block_size to get many tiles")
    ap.add_argument("--tol", type=float, default=1.35, help="allowed peak ratio between tile counts")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"

    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3 import modeling_qwen3

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="eager").cuda()
    cfg_hf = model.config
    cfg = build_msa_config(cfg_hf, mode="dense_warmup", top_k=16, dense_prefix=3, diag_interval=0,
                           kl_checkpoint=True)
    attach_indexers(model, cfg)
    modeling_qwen3.Qwen3Attention.forward = qwen3_msa_attn_forward
    install_kl_accumulation(model)
    freeze_base_train_indexer(model)
    model.train()
    ids = torch.randint(0, cfg_hf.vocab_size, (1, a.seq_len), device="cuda")

    T = a.seq_len
    print(f"\nmodel={cfg_hf.num_hidden_layers}L H_kv={cfg.num_kv_heads}  T={T}")

    print("\n-- Phase 1: peak memory vs tile count (fixed T) --")
    rows = []
    for kb in (T // 2, T // 4, T // 8, T // 16):
        pk = peak_for(model, cfg, ids, "dense_warmup", kb)
        rows.append((kb, T // kb, pk))
        print(f"     kl_block_size={kb:6d}  tiles={T // kb:3d}  peak={pk / 1e9:6.3f} GB")
    peaks = [p for _, _, p in rows]
    ratio = max(peaks) / min(peaks)
    check("peak does NOT grow with tile count (retained-per-tile state)", ratio < a.tol,
          f"max/min = {ratio:.2f}x across {rows[0][1]}..{rows[-1][1]} tiles (tol {a.tol})")
    # A retained-per-tile bug shows up as monotone growth as tiles increase.
    growth = peaks[-1] / peaks[0]
    check("16x more tiles does not inflate peak", growth < a.tol,
          f"{rows[-1][1]} tiles / {rows[0][1]} tiles = {growth:.2f}x")

    print("\n-- Phase 2: same invariant on the sparse path --")
    rows2 = []
    for kb in (T // 2, T // 8):
        pk = peak_for(model, cfg, ids, "sparse", kb)
        rows2.append((kb, T // kb, pk))
        print(f"     kl_block_size={kb:6d}  tiles={T // kb:3d}  peak={pk / 1e9:6.3f} GB")
    r2 = rows2[1][2] / rows2[0][2]
    check("sparse path peak does not grow with tile count", r2 < a.tol,
          f"{rows2[1][1]} tiles / {rows2[0][1]} tiles = {r2:.2f}x")

    print("\n-- checkpointing must REDUCE peak, not just move it --")
    cfg.mode = "dense_warmup"
    on = peak_for(model, cfg, ids, "dense_warmup", T // 8, ckpt=True)
    off = peak_for(model, cfg, ids, "dense_warmup", T // 8, ckpt=False)
    check("kl_checkpoint=True peaks below kl_checkpoint=False", on < off,
          f"on={on / 1e9:.3f} GB vs off={off / 1e9:.3f} GB ({off / on:.2f}x saving)")

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
