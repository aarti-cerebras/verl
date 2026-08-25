#!/usr/bin/env python3
"""P0 spike: can FA3 do token-granular sparse attention on **GQA** via a page-size-1 block table?

This is the gate for the whole Qwen3-DSA serving route (docs/qwen3_4b_dsa/serving_eval_plan.md §4 P0).
vLLM's MLA sparse backend reduces DSA to one ordinary FA3 varlen call by viewing the paged KV cache
as page-size-1 -- `flashattn_mla_sparse.py:238-259`. That file's cache view hardcodes
``num_kv_heads=1`` (the MLA latent) and uses FA3's asymmetric ``q_v`` path. This probe asks whether
the *symmetric GQA* form of the same call works, and how fast it is:

    k_cache = kv[..].view(-1, 1, num_kv_heads, head_dim)     # "pages" are single tokens
    flash_attn_varlen_func(q, k_cache, v_cache, max_seqlen_q=1,
                           cu_seqlens_q=arange(T+1), max_seqlen_k=topk,
                           seqused_k=valid_counts, block_table=topk_slots,
                           causal=True, fa_version=3)

Every query token is its own length-1 sequence over its own gathered key set; causality comes from
the *selection*, not from the kernel mask.

Run (H100, SM90):
    .devlibs/vllm026/bin/python tests/dsa/probe_fa3_sparse_gqa.py

Exit code 0 iff every correctness case passes. Timings are information, not a gate.
"""

import argparse
import time

import torch
import torch.nn.functional as F

from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

H_Q = 32          # Qwen3-4B
H_KV = 8
HEAD_DIM = 128
BLOCK_SIZE = 64   # the compatibility keystone (serving_eval_plan.md §2.2)
DTYPE = torch.bfloat16


def make_paged_cache(num_blocks: int, device: str = "cuda"):
    """Standard GQA paged K/V caches, ``[num_blocks, BLOCK_SIZE, H_KV, HEAD_DIM]``."""
    g = torch.Generator(device=device).manual_seed(0)
    shape = (num_blocks, BLOCK_SIZE, H_KV, HEAD_DIM)
    k = torch.randn(shape, generator=g, device=device, dtype=DTYPE) * 0.5
    v = torch.randn(shape, generator=g, device=device, dtype=DTYPE) * 0.5
    return k, v


def scatter_block_table(num_reqs: int, seq_len: int, num_blocks: int, device: str = "cuda"):
    """A deliberately NON-identity block table, so the probe really exercises paging.

    Returns ``block_table [num_reqs, blocks_per_req] int32``.
    """
    blocks_per_req = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    need = num_reqs * blocks_per_req
    assert need <= num_blocks, f"need {need} blocks, have {num_blocks}"
    perm = torch.randperm(num_blocks, generator=torch.Generator().manual_seed(1))[:need]
    return perm.to(torch.int32).view(num_reqs, blocks_per_req).to(device)


def token_to_slot(block_table: torch.Tensor, req_id: torch.Tensor, tok_idx: torch.Tensor):
    """Per-request token index -> global cache slot, i.e. what
    ``triton_convert_req_index_to_global_index`` computes. ``-1`` propagates as ``-1``."""
    valid = tok_idx >= 0
    safe = tok_idx.clamp(min=0)
    base = block_table[req_id.unsqueeze(-1).expand_as(safe), safe // BLOCK_SIZE]
    return torch.where(valid, base * BLOCK_SIZE + safe % BLOCK_SIZE, -1).to(torch.int32)


def reference_sparse_attn(q, k_cache, v_cache, slots, counts, scale):
    """Gather + SDPA reference. ``q [T, H_Q, D]``, ``slots [T, K] int32`` (-1 padded)."""
    T, K = slots.shape
    kf = k_cache.view(-1, H_KV, HEAD_DIM)
    vf = v_cache.view(-1, H_KV, HEAD_DIM)
    safe = slots.clamp(min=0).long()
    kg = kf[safe]                                        # [T, K, H_KV, D]
    vg = vf[safe]
    # expand kv heads to query heads (GQA)
    rep = H_Q // H_KV
    kg = kg.repeat_interleave(rep, dim=2)                # [T, K, H_Q, D]
    vg = vg.repeat_interleave(rep, dim=2)
    logits = torch.einsum("thd,tkhd->thk", q.float(), kg.float()) * scale
    ar = torch.arange(K, device=slots.device).view(1, 1, K)
    mask = (ar < counts.view(T, 1, 1)) & (slots.view(T, 1, K) >= 0)
    logits = logits.masked_fill(~mask, float("-inf"))
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("thk,tkhd->thd", p, vg.float())


def fa3_sparse_attn(q, k_cache, v_cache, slots, counts, scale):
    """The call under test: FA3 varlen with a page-size-1 view of a GQA cache."""
    T = q.shape[0]
    k_view = k_cache.view(-1, 1, H_KV, HEAD_DIM)
    v_view = v_cache.view(-1, 1, H_KV, HEAD_DIM)
    cu_q = torch.arange(0, T + 1, dtype=torch.int32, device=q.device)
    return flash_attn_varlen_func(
        q=q,
        k=k_view,
        v=v_view,
        max_seqlen_q=1,
        cu_seqlens_q=cu_q,
        max_seqlen_k=slots.shape[1],
        seqused_k=counts,
        block_table=slots,
        softmax_scale=scale,
        causal=True,
        fa_version=3,
    )


def build_case(num_reqs: int, seq_len: int, num_queries_per_req: int, topk: int, seed: int = 0):
    """Random causal top-k selection. Returns (q, k_cache, v_cache, slots, counts)."""
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(seed)
    num_blocks = max(1024, num_reqs * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) + 16)
    k_cache, v_cache = make_paged_cache(num_blocks, device)
    block_table = scatter_block_table(num_reqs, seq_len, num_blocks, device)

    T = num_reqs * num_queries_per_req
    q = (torch.randn((T, H_Q, HEAD_DIM), generator=g, device=device, dtype=DTYPE) * 0.5)
    req_id = torch.arange(num_reqs, device=device).repeat_interleave(num_queries_per_req)

    # Query position within its request: the last `num_queries_per_req` positions.
    pos = (seq_len - num_queries_per_req) + torch.arange(
        num_queries_per_req, device=device
    ).repeat(num_reqs)
    n_causal = pos + 1                                   # keys 0..pos are visible
    counts = torch.minimum(n_causal, torch.full_like(n_causal, topk)).to(torch.int32)

    # For each query row draw `counts[i]` distinct token ids from [0, pos_i]; pad tail with -1.
    tok = torch.full((T, topk), -1, dtype=torch.long, device=device)
    r = torch.rand((T, seq_len), generator=g, device=device)
    r = r.masked_fill(torch.arange(seq_len, device=device).view(1, -1) > pos.view(-1, 1), 2.0)
    order = r.argsort(dim=-1)                            # invisible keys sort last
    ncols = min(topk, seq_len)                           # topk may exceed the sequence (dense-eq case)
    col = torch.arange(ncols, device=device).view(1, -1)
    keep = col < counts.view(-1, 1).long()
    tok[:, :ncols][keep] = order[:, :ncols][keep]
    slots = token_to_slot(block_table, req_id, tok)
    return q, k_cache, v_cache, slots, counts


def check(name, num_reqs, seq_len, nq, topk, tol=6e-3):
    q, kc, vc, slots, counts = build_case(num_reqs, seq_len, nq, topk)
    scale = HEAD_DIM ** -0.5
    ref = reference_sparse_attn(q, kc, vc, slots, counts, scale)
    got = fa3_sparse_attn(q, kc, vc, slots, counts, scale).float()
    err = (got - ref).abs().max().item()
    rel = err / max(ref.abs().max().item(), 1e-6)
    ok = rel < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} T={q.shape[0]:>6} topk={topk:>5} "
          f"max_abs={err:.4e} rel={rel:.3e}")
    return ok


def bench(name, num_reqs, seq_len, nq, topk, iters=20):
    q, kc, vc, slots, counts = build_case(num_reqs, seq_len, nq, topk)
    scale = HEAD_DIM ** -0.5
    for _ in range(3):
        fa3_sparse_attn(q, kc, vc, slots, counts, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fa3_sparse_attn(q, kc, vc, slots, counts, scale)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    # FLOPs: T * topk * H_Q * D * 2 (QK) * 2 (PV)
    tflops = q.shape[0] * topk * H_Q * HEAD_DIM * 4 / (ms * 1e-3) / 1e12
    print(f"  {name:<34} T={q.shape[0]:>6} topk={topk:>5} {ms:8.3f} ms  {tflops:7.2f} TFLOP/s")
    return ms


def bench_dense(name, seq_len, iters=20):
    """Dense FA3 prefill on the same geometry, for a sanity reference point."""
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(3)
    q = torch.randn((seq_len, H_Q, HEAD_DIM), generator=g, device=device, dtype=DTYPE) * 0.5
    k = torch.randn((seq_len, H_KV, HEAD_DIM), generator=g, device=device, dtype=DTYPE) * 0.5
    v = torch.randn((seq_len, H_KV, HEAD_DIM), generator=g, device=device, dtype=DTYPE) * 0.5
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    scale = HEAD_DIM ** -0.5
    fn = lambda: flash_attn_varlen_func(  # noqa: E731
        q=q, k=k, v=v, max_seqlen_q=seq_len, cu_seqlens_q=cu,
        max_seqlen_k=seq_len, cu_seqlens_k=cu, softmax_scale=scale,
        causal=True, fa_version=3)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"  {name:<34} T={seq_len:>6} {'dense':>10} {ms:8.3f} ms")
    return ms


def bench_dense_decode(name, num_reqs, seq_len, iters=50):
    """Dense paged decode on the same geometry — the baseline sparse decode must beat.

    Ordinary FA3 varlen: one query per request, page size ``BLOCK_SIZE``, full causal KV.
    """
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(4)
    blocks_per_req = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_reqs * blocks_per_req + 16
    k_cache, v_cache = make_paged_cache(num_blocks, device)
    block_table = scatter_block_table(num_reqs, seq_len, num_blocks, device)
    q = torch.randn((num_reqs, H_Q, HEAD_DIM), generator=g, device=device, dtype=DTYPE) * 0.5
    cu_q = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device)
    seqused = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)
    scale = HEAD_DIM ** -0.5
    fn = lambda: flash_attn_varlen_func(  # noqa: E731
        q=q, k=k_cache, v=v_cache, max_seqlen_q=1, cu_seqlens_q=cu_q,
        max_seqlen_k=seq_len, seqused_k=seqused, block_table=block_table,
        softmax_scale=scale, causal=True, fa_version=3)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    gb = num_reqs * seq_len * H_KV * HEAD_DIM * 2 * 2 / 1e9
    print(f"  {name:<34} T={num_reqs:>6} {'dense':>10} {ms:8.3f} ms  "
          f"{gb / (ms * 1e-3):7.0f} GB/s")
    return ms


def bench_sorted(name, num_reqs, seq_len, nq, topk, iters=20):
    """Same call, but with each row's selected slots sorted ascending.

    Top-k selection is a SET — the kernel's output does not depend on column order — so if
    sorting improves coalescing on the page-size-1 gather it is a free win, and the indexer's
    score-ordered output can be sorted before the attend.
    """
    q, kc, vc, slots, counts = build_case(num_reqs, seq_len, nq, topk)
    scale = HEAD_DIM ** -0.5
    # sort valid slots ascending, keeping -1 padding at the tail (-1 sorts first, so push to +inf)
    big = torch.iinfo(torch.int32).max
    keyed = torch.where(slots >= 0, slots, torch.full_like(slots, big))
    ssorted = keyed.sort(dim=-1).values
    ssorted = torch.where(ssorted == big, torch.full_like(ssorted, -1), ssorted).to(torch.int32)
    ref = fa3_sparse_attn(q, kc, vc, slots, counts, scale).float()
    got = fa3_sparse_attn(q, kc, vc, ssorted, counts, scale).float()
    rel = (got - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    for _ in range(3):
        fa3_sparse_attn(q, kc, vc, ssorted, counts, scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fa3_sparse_attn(q, kc, vc, ssorted, counts, scale)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    print(f"  {name:<34} T={q.shape[0]:>6} topk={topk:>5} {ms:8.3f} ms  "
          f"(sorted==unsorted rel {rel:.1e})")
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--bench-32k", action="store_true", help="run the 32K prefill shape (slow to build)")
    args = ap.parse_args()

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}  "
          f"cc {torch.cuda.get_device_capability(0)}")
    print(f"geometry: H_q={H_Q} H_kv={H_KV} d={HEAD_DIM} page={BLOCK_SIZE} dtype={DTYPE}")

    print("\n[1] correctness vs gather+SDPA reference")
    ok = True
    ok &= check("decode, 1 req", 1, 4096, 1, 256)
    ok &= check("decode, 32 reqs", 32, 4096, 1, 256)
    ok &= check("decode, 8 reqs, k>seq (dense-eq)", 8, 512, 1, 2048)
    ok &= check("prefill-shaped, 512 queries", 1, 512, 512, 64)
    ok &= check("prefill-shaped, 2048 queries", 1, 2048, 2048, 128)
    ok &= check("mixed batch, 4 reqs x 64 q", 4, 1024, 64, 256)

    if not args.skip_bench:
        print("\n[2] decode timing (the regime sparsity is supposed to win)")
        for reqs in (1, 8, 32):
            for sl, tk in ((8192, 2048), (32768, 2048)):
                bench(f"decode {reqs} req @ {sl // 1024}K", reqs, sl, 1, tk)

        print("\n[2b] dense paged decode baseline (what sparse decode must beat)")
        for reqs in (1, 8, 32):
            for sl in (8192, 32768):
                bench_dense_decode(f"dense decode {reqs} req @ {sl // 1024}K", reqs, sl)

        print("\n[2c] sorted block table (coalescing; selection is a set, so this is free)")
        bench_sorted("sorted decode 32 req @ 32K", 32, 32768, 1, 2048)
        bench_sorted("sorted prefill 4K", 1, 4096, 4096, 2048)
        bench_sorted("sorted prefill 8K", 1, 8192, 8192, 2048)

        print("\n[3] prefill timing (the unmeasured regime)")
        bench_dense("dense FA3 prefill 4K", 4096)
        bench("sparse prefill 4K", 1, 4096, 4096, 2048)
        bench_dense("dense FA3 prefill 8K", 8192)
        bench("sparse prefill 8K", 1, 8192, 8192, 2048)
        if args.bench_32k:
            bench_dense("dense FA3 prefill 32K", 32768)
            bench("sparse prefill 32K", 1, 32768, 32768, 2048, iters=5)
            bench_sorted("sorted prefill 32K", 1, 32768, 32768, 2048, iters=5)

    print(f"\nP0 correctness: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
