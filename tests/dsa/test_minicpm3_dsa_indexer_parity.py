#!/usr/bin/env python3
"""GPU parity test: vLLM-side ``MiniCPM3DSAIndexer.project_and_score`` (DeepGEMM
FP8 kernel) vs the trusted training-side ``LightningIndexer`` (pure torch).

Both are built with IDENTICAL random weights (checkpoint-name-matched state dict
copied from the serve module into each HF reference), fed the SAME random inputs
(fixed seed), and compared by top-256 key-selection overlap per query — the
metric that actually matters for sparse-attention serving.

Reports (probe_deepgemm_indexer M3 methodology), over selection-saturated
queries (valid keys > top_k, where selection is a real sub-selection):
  * MiniCPM3DSAIndexer(kernel)  vs HF(rotate=True,  fp8=True)  -> serve-vs-train
  * MiniCPM3DSAIndexer(kernel)  vs HF(rotate=False, fp8=True)  -> Hadamard-drop
  * MiniCPM3DSAIndexer(kernel)  vs HF(fp8=False) bf16-exact    -> sanity
Plus: top_k>=T selects all valid keys; shapes/finiteness.

Run (serve env):
    cd /net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl
    export PYTHONPATH=$(pwd)/.devlibs/tf457lib:$(pwd)
    /usr/bin/python3 tests/dsa/test_minicpm3_dsa_indexer_parity.py
"""

import sys

import torch

# Importing the package installs the vendored-deep_gemm shim.
from scripts.dsa.vllm_minicpm3_dsa.indexer import MiniCPM3DSAIndexer
from verl.models.transformers.dsa_indexer import DSAConfig, LightningIndexer

DEV = "cuda"
SEED = 1234

# MiniCPM3 DSA serving config.
N_HEADS = 16
HEAD_DIM = 64
ROPE_HEAD_DIM = 32
Q_LORA_RANK = 768
HIDDEN = 2560
TOP_K = 256
T = 2048
SOFTMAX_SCALE = HEAD_DIM**-0.5

# Acceptance thresholds. Legacy training used a continuous absmax scale vs the kernel's UE8M0 (power-of-2)
# scale -> ~2% selection drift, tolerated at 0.95. Training with fp8_ue8m0=True quantizes the scale the same
# way the kernel does, so serve-vs-train overlap must clear a much tighter bar AND beat the legacy path.
OVERLAP_MIN = 0.95
OVERLAP_UE8M0_MIN = 0.99


def build_rope(t: int, dim: int, base: float = 10000.0, device=DEV):
    """Standard (non-interleaved / llama) RoPE cos/sin tables [1, T, dim]."""
    assert dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))  # [dim/2]
    pos = torch.arange(t, device=device).float()  # [T]
    freqs = torch.outer(pos, inv_freq)  # [T, dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim] (rotate_half convention)
    return emb.cos()[None], emb.sin()[None]  # [1, T, dim]


def causal_bias(m: int, n: int, device=DEV):
    b = torch.zeros(m, n, device=device)
    b.masked_fill_(~torch.tril(torch.ones(m, n, device=device, dtype=torch.bool)), float("-inf"))
    return b


def topk_overlap(A, B, top_k, only_saturated=True):
    """Per-query fraction overlap of top-min(top_k, valid) indices; valid = m+1
    (causal). only_saturated restricts to queries where valid > top_k (real
    sub-selection). Returns (mean, min, n_queries)."""
    M = A.shape[0]
    ov = []
    for m in range(M):
        v = m + 1
        kk = min(top_k, v)
        if only_saturated and v <= top_k:
            continue
        ia = A[m].topk(kk).indices
        ib = B[m].topk(kk).indices
        ov.append(len(set(ia.tolist()) & set(ib.tolist())) / kk)
    if not ov:
        return float("nan"), float("nan"), 0
    tt = torch.tensor(ov)
    return tt.mean().item(), tt.min().item(), len(ov)


def hf_scores(hf: LightningIndexer, hidden, qr, cos, sin):
    """HF LightningIndexer raw scores [T, T] + causal bias."""
    with torch.no_grad():
        s = hf(hidden, qr, cos, sin)  # [1, T, T]
    return s[0] + causal_bias(T, T)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(SEED)
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
    print(f"config: n_heads={N_HEADS} head_dim={HEAD_DIM} rope_dim={ROPE_HEAD_DIM} top_k={TOP_K} T={T}\n")

    # --- inputs (fixed seed) ---
    hidden = torch.randn(1, T, HIDDEN, device=DEV)
    qr = torch.randn(1, T, Q_LORA_RANK, device=DEV)
    cos, sin = build_rope(T, ROPE_HEAD_DIM)

    # --- serve module (weight owner) ---
    m3 = MiniCPM3DSAIndexer(
        n_heads=N_HEADS, head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM,
        top_k=TOP_K, q_lora_rank=Q_LORA_RANK, hidden_size=HIDDEN,
        fp8=True, rotate_activation=True,
    ).to(DEV)

    # --- HF references with the SAME weights (checkpoint-name-matched state dict) ---
    def make_hf(rotate, fp8, fp8_ue8m0=False):
        cfg = DSAConfig(
            n_heads=N_HEADS, head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM,
            q_lora_rank=Q_LORA_RANK, hidden_size=HIDDEN, top_k=TOP_K,
            fp8=fp8, rotate_activation=rotate, fp8_ue8m0=fp8_ue8m0,
        )
        hf = LightningIndexer(cfg, softmax_scale=SOFTMAX_SCALE).to(DEV)
        missing, unexpected = hf.load_state_dict(m3.state_dict(), strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        return hf

    hf_rot_fp8 = make_hf(rotate=True, fp8=True)  # legacy training numerics (continuous scale)
    hf_rot_fp8_ue8m0 = make_hf(rotate=True, fp8=True, fp8_ue8m0=True)  # UE8M0 (matches the kernel's scale)
    hf_norot_fp8 = make_hf(rotate=False, fp8=True)
    hf_bf16 = make_hf(rotate=True, fp8=False)

    # --- forward ---
    kern = m3.project_and_score(hidden, qr, cos, sin)  # [T, T], -inf off-causal
    s_rot = hf_scores(hf_rot_fp8, hidden, qr, cos, sin)
    s_rot_ue8m0 = hf_scores(hf_rot_fp8_ue8m0, hidden, qr, cos, sin)
    s_norot = hf_scores(hf_norot_fp8, hidden, qr, cos, sin)
    s_bf16 = hf_scores(hf_bf16, hidden, qr, cos, sin)

    results = {}
    passed = True

    # ---- shapes / finiteness ----
    cmask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    shape_ok = tuple(kern.shape) == (T, T)
    finite_ok = torch.isfinite(kern[cmask]).all().item()
    offcausal_ok = torch.isinf(kern[~cmask]).all().item()
    print("=" * 78)
    print("Shape / finiteness")
    print("=" * 78)
    print(f"  kernel logits shape = {tuple(kern.shape)}  (expect ({T}, {T}))  -> {'OK' if shape_ok else 'BAD'}")
    print(f"  causal entries finite         : {finite_ok}")
    print(f"  off-causal entries are -inf   : {offcausal_ok}")
    print(f"  logit range (causal): [{kern[cmask].min().item():.3f}, {kern[cmask].max().item():.3f}]\n")
    passed &= shape_ok and finite_ok and offcausal_ok

    # ---- top_k >= T selects ALL valid keys (trivially full) ----
    print("=" * 78)
    print("top_k >= T -> selected set == all valid keys")
    print("=" * 78)
    sel_full = m3.select_topk(kern, top_k=T)  # [T, T]
    # spot-check a few queries: finite (valid) keys must all be selected.
    trivial_ok = True
    for m in (0, 1, 100, T // 2, T - 1):
        valid = set(range(m + 1))
        selected = set(sel_full[m].tolist())
        # every valid key is selected (padded/off-causal are -inf, ranked last)
        if not valid.issubset(selected):
            trivial_ok = False
    print(f"  all valid keys selected for sampled queries: {trivial_ok}\n")
    passed &= trivial_ok

    # ---- top-256 selection overlap (saturated queries: valid > top_k) ----
    print("=" * 78)
    print(f"top-{TOP_K} selection overlap  (saturated queries: valid > {TOP_K})")
    print("=" * 78)
    # enforce: (min_overlap or None). The UE8M0 row also must not regress vs the legacy row (checked below).
    comps = [
        ("kernel vs HF(rot=True, fp8, legacy-scale) [serve-vs-train]  ", s_rot, OVERLAP_MIN),
        ("kernel vs HF(rot=True, fp8, UE8M0-scale)  [serve-vs-train]  ", s_rot_ue8m0, OVERLAP_UE8M0_MIN),
        ("kernel vs HF(rot=False, fp8)              [Hadamard-drop]   ", s_norot, OVERLAP_MIN),
        ("kernel vs HF(fp8=False) bf16-exact        [sanity]          ", s_bf16, None),
    ]
    print(f"  {'comparison':<60} {'mean':>8} {'min':>8}  n")
    for label, ref, enforce in comps:
        mean, mn, nq = topk_overlap(kern, ref, TOP_K, only_saturated=True)
        results[label] = (mean, mn, nq)
        flag = ""
        if enforce is not None:
            ok = mean >= enforce
            passed &= ok
            flag = f"  [{'PASS' if ok else 'FAIL'} >= {enforce}]"
        print(f"  {label:<60} {mean:>8.4f} {mn:>8.4f}  {nq}{flag}")

    # UE8M0 must be at least as close to the kernel as the legacy continuous scale (the whole point of the fix).
    legacy_mean = results[comps[0][0]][0]
    ue8m0_mean = results[comps[1][0]][0]
    improved = ue8m0_mean >= legacy_mean - 1e-6
    passed &= improved
    print(f"\n  UE8M0 vs legacy serve-parity: {ue8m0_mean:.4f} vs {legacy_mean:.4f}  "
          f"(Δ={ue8m0_mean - legacy_mean:+.4f})  [{'PASS' if improved else 'FAIL'}: no regression]")
    print()

    print("=" * 78)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    print("=" * 78)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
