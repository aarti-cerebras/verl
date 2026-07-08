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
"""#12 profile: per-LAYER cost split of the DSA KL recompute at real 32K/full-model shapes.

Splits time across (A) target `p` per-head recompute [current], (A') head-batched variant [Tier-1 preview],
(B) indexer project+scores, (C) KL math. Single GPU (per-layer cost is per-GPU). Run:
    PYTHONPATH=.devlibs/tf457lib:$PWD python tests/models/dsa_kl_profile.py [T] [block] [head_chunk]
"""

import sys
import time

import torch

from transformers import AutoConfig

from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer
from verl.models.transformers.minicpm_dsa import _causal_doc_bias_block

DEV = "cuda"
DT = torch.bfloat16
T = int(sys.argv[1]) if len(sys.argv) > 1 else 32768
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
HC = int(sys.argv[3]) if len(sys.argv) > 3 else 8  # head-chunk for the batched variant


def sync():
    torch.cuda.synchronize()


def timeit(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.time()
    for _ in range(iters):
        fn()
    sync()
    return (time.time() - t0) / iters * 1000.0  # ms


def peak_gb(fn):
    torch.cuda.reset_peak_memory_stats()
    fn()
    sync()
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    cfg = AutoConfig.from_pretrained("openbmb/MiniCPM3-4B", trust_remote_code=True)
    H = cfg.num_attention_heads
    q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    hidden, q_lora, rope = cfg.hidden_size, cfg.q_lora_rank, cfg.qk_rope_head_dim
    scale = q_head_dim**-0.5
    print(f"config: H={H} q_head_dim={q_head_dim} hidden={hidden} q_lora={q_lora} rope={rope}")
    print(f"shapes: T={T} block={BLOCK} head_chunk={HC} bsz=1 dtype={DT}\n")

    torch.manual_seed(0)
    qs = torch.randn(1, H, T, q_head_dim, device=DEV, dtype=DT)
    ks = torch.randn(1, H, T, q_head_dim, device=DEV, dtype=DT)
    hidden_states = torch.randn(1, T, hidden, device=DEV, dtype=DT)
    qr = torch.randn(1, T, q_lora, device=DEV, dtype=DT)
    cos = torch.randn(1, T, rope, device=DEV, dtype=DT)
    sin = torch.randn(1, T, rope, device=DEV, dtype=DT)
    pos = torch.arange(T, device=DEV).unsqueeze(0)

    dsa = DSAConfig(
        enabled=True, n_heads=16, head_dim=64, rope_head_dim=rope, q_lora_rank=q_lora,
        hidden_size=hidden, fp8=True, kl_block_size=BLOCK,
    )
    idx = LightningIndexer(dsa).to(DEV).to(DT)
    q_idx, k_idx, weights = idx.project(hidden_states, qr, cos, sin)  # projections once (as in the real path)

    # ---- Phase A: current per-head target recompute ----
    def phase_a():
        for q0 in range(0, T, BLOCK):
            q1 = min(q0 + BLOCK, T)
            bias = _causal_doc_bias_block(pos, q0, q1, T, DEV)
            qb = qs[:, :, q0:q1, :]
            p = torch.zeros(1, q1 - q0, T, device=DEV, dtype=torch.float32)
            for h in range(H):
                s = torch.matmul(qb[:, h], ks[:, h].transpose(1, 2)) * scale
                p += torch.softmax(s.float() + bias, dim=-1)
            p /= H

    # ---- Phase A': head-batched (chunked) target recompute (Tier-1 preview) ----
    def phase_a_batched():
        for q0 in range(0, T, BLOCK):
            q1 = min(q0 + BLOCK, T)
            bias = _causal_doc_bias_block(pos, q0, q1, T, DEV)  # [1, B, T]
            qb = qs[:, :, q0:q1, :]
            p = torch.zeros(1, q1 - q0, T, device=DEV, dtype=torch.float32)
            for h0 in range(0, H, HC):
                h1 = min(h0 + HC, H)
                s = torch.einsum("bhqd,bhkd->bhqk", qb[:, h0:h1], ks[:, h0:h1]) * scale  # [1, Hc, B, T]
                p += torch.softmax(s.float() + bias.unsqueeze(1), dim=-1).sum(dim=1)
            p /= H

    # ---- Phase A variants to isolate matmul vs fp32-softmax ----
    def phase_a_matmul_only():  # per-head QK matmul, NO softmax (isolates matmul cost)
        for q0 in range(0, T, BLOCK):
            q1 = min(q0 + BLOCK, T)
            qb = qs[:, :, q0:q1, :]
            acc = torch.zeros(1, q1 - q0, T, device=DEV, dtype=DT)
            for h in range(H):
                acc += torch.matmul(qb[:, h], ks[:, h].transpose(1, 2))

    def phase_a_bf16_softmax():  # per-head matmul + BF16 softmax (no fp32 upcast)
        for q0 in range(0, T, BLOCK):
            q1 = min(q0 + BLOCK, T)
            bias = _causal_doc_bias_block(pos, q0, q1, T, DEV).to(DT)
            qb = qs[:, :, q0:q1, :]
            p = torch.zeros(1, q1 - q0, T, device=DEV, dtype=torch.float32)
            for h in range(H):
                s = torch.matmul(qb[:, h], ks[:, h].transpose(1, 2)) * scale
                p += torch.softmax(s + bias, dim=-1).float()
            p /= H

    # ---- Phase B: indexer project + scores ----
    def phase_b():
        qi, ki, w = idx.project(hidden_states, qr, cos, sin)
        for q0 in range(0, T, BLOCK):
            q1 = min(q0 + BLOCK, T)
            bias = _causal_doc_bias_block(pos, q0, q1, T, DEV)
            idx.scores(qi[:, q0:q1], ki, w[:, q0:q1], attn_bias=bias)

    # ---- Phase C: KL math given p and I (precompute one block's tensors) ----
    bias0 = _causal_doc_bias_block(pos, 0, BLOCK, T, DEV)
    p0 = torch.softmax(torch.randn(1, BLOCK, T, device=DEV) + bias0, dim=-1)
    I0 = idx.scores(q_idx[:, :BLOCK], k_idx, weights[:BLOCK].unsqueeze(0) if weights.dim() == 2 else weights[:, :BLOCK], attn_bias=bias0)

    def phase_c():
        for _ in range(0, T, BLOCK):  # same #blocks as A/B
            allow = bias0 == 0.0
            log_q = torch.log_softmax(I0.float(), dim=-1)
            term = p0 * (torch.log(p0.clamp_min(1e-9)) - log_q)
            torch.where(allow, term, torch.zeros_like(term)).sum(-1)

    tA = timeit(phase_a)
    tA_mm = timeit(phase_a_matmul_only)
    tA_bf16 = timeit(phase_a_bf16_softmax)
    tAb = timeit(phase_a_batched)
    tB = timeit(phase_b)
    tC = timeit(phase_c)
    mA = peak_gb(phase_a)
    mAb = peak_gb(phase_a_batched)

    per_layer = tA + tB + tC
    print("=== per-LAYER cost (ms), diag OFF ===")
    print(f"  A  target p (per-head, current) : {tA:8.1f}  ({100*tA/per_layer:4.1f}%)   peak {mA:.2f} GB")
    print(f"  B  indexer project+scores       : {tB:8.1f}  ({100*tB/per_layer:4.1f}%)")
    print(f"  C  KL math                      : {tC:8.1f}  ({100*tC/per_layer:4.1f}%)")
    print(f"  ---- current per-layer total    : {per_layer:8.1f}")
    print("\n=== Phase-A breakdown (what to attack) ===")
    print(f"  A  matmul-only (no softmax)     : {tA_mm:8.1f}  ({100*tA_mm/tA:4.1f}% of A)")
    print(f"  A  softmax overhead (A - mm)    : {tA - tA_mm:8.1f}  ({100*(tA-tA_mm)/tA:4.1f}% of A)")
    print(f"  A  with bf16 softmax            : {tA_bf16:8.1f}   -> {tA/tA_bf16:.1f}x vs current")
    print(f"  A' head-batched HC={HC}            : {tAb:8.1f}   peak {mAb:.2f} GB   -> {tA/tAb:.1f}x vs current")
    L = cfg.num_hidden_layers
    print(f"\n  x{L} layers: current ~{per_layer*L/1000:.2f}s  |  with A' ~{(tAb+tB+tC)*L/1000:.2f}s  "
          f"(fwd only, per step; checkpointing ~doubles)")


if __name__ == "__main__":
    main()
