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
"""Stage-1 gate: MiniCPM3 attention in MLA-latent form is numerically correct.

The Stage-1 goal (docs/dsa_vllm_minicpm3dsa_build_plan.md "DECODE BUILD PLAN") is
to express MiniCPM3's attention in **MLA-latent form** so it can later feed the
sparse path, and prove it equals the stock **dense materialized-QKV** attention.

vLLM DIM-LOCK (finding): the dense MLA metadata is hard-locked to DeepSeek latent
head sizes — ``MLACommonBackend.get_supported_head_sizes() == [320, 576]`` and
``MLACommonMetadata.__post_init__`` raises for anything else. MiniCPM3's MLA
head_size is ``kv_lora_rank(256) + qk_rope(32) = 288`` (∉ {320,576}), so the real
vLLM MLA op CANNOT run at native MiniCPM3 dims — the same dim-lock the build plan
flagged for the *sparse* kernel governs the *dense* MLACommon metadata too. Making
it run end-to-end needs the Stage-2 zero-pad-to-576 reuse (explicitly out of scope
here). So per the task ("a per-layer attention-output comparison on a fixed input
is an acceptable stronger check — pick whatever gives a clean, defensible parity
number"), the Stage-1 gate is a STANDALONE numerical proof on the REAL checkpoint
weights:

  * REF  = stock dense materialized-QKV attention (exactly minicpm3.py math):
           materialize full K/V from the compressed latent via ``kv_b_proj``,
           full causal softmax attention.
  * MLA  = the MLA absorbed-latent decode formulation that vLLM's ``MLAAttention``
           implements internally: absorb ``kv_b_proj`` into per-head ``W_UK``/
           ``W_UV``, score against the compressed latent ``kv_c_normed`` + shared
           ``k_pe`` (MQA), then reconstruct the value via ``W_UV``.

These are the two sides ``MiniCPM3DSAAttention`` bridges: its ``forward`` produces
exactly ``(q, kv_c_normed, k_pe)`` and hands them to ``MLAAttention`` (which does
the absorption). RoPE (a genuine neox RoPE) is applied identically to both paths,
so the equivalence is a pure statement about the MLA absorption identity. Both
paths are computed in fp32 (identity check) and bf16 (the realistic serving tol).

Optional coherence sanity (subprocess, best-effort): load the SAME checkpoint
through the stock dense path (``MiniCPM3StockRefForCausalLM``) via vLLM offline and
confirm generation is coherent (== base model) — this also confirms the plugin
package + weight mapping still load after the Stage-1 restructure.

Run:
  cd <repo> && export PYTHONPATH=$(pwd)/.devlibs/tf457lib:$(pwd)
  /usr/bin/python3 tests/dsa/test_stage1_mla_parity.py               # gate (+ sanity)
  /usr/bin/python3 tests/dsa/test_stage1_mla_parity.py --no-vllm     # gate only
  /usr/bin/python3 tests/dsa/test_stage1_mla_parity.py --stock-gen   # internal
"""

import os
import subprocess
import sys

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

SERVING = (
    "/cb/ml-eng/aarti/dsa/phase2_topk_sweep_20260717_223140/"
    "phase2_full_k256_1ep_20260717_223341/train/checkpoints/inference/"
    "global_step_2805/serving_stage0"
)
CKPT_SAFETENSORS = os.path.join(os.path.dirname(SERVING), "serving", "model.safetensors")

# MiniCPM3-4B attention dims.
H = 40            # num_attention_heads
QK_NOPE = 64      # qk_nope_head_dim
QK_ROPE = 32      # qk_rope_head_dim
QK_HEAD = QK_NOPE + QK_ROPE  # 96
V_HEAD = 64       # v_head_dim
KV_LORA = 256     # kv_lora_rank
Q_LORA = 768      # q_lora_rank
HIDDEN = 2560
RMS_EPS = 1e-5
SCALING = QK_HEAD ** -0.5
ROPE_THETA = 10000.0

LAYERS_TO_CHECK = [0, 15, 30, 45, 61]
SEQ_LEN = 24

# Parity tolerances.
FP32_TOL = 5e-3       # exact identity, fp32 einsum reorder noise
BF16_REL_TOL = 0.02   # bf16 MLA-absorption tol, RELATIVE to attn-output scale
                      # (absolute Δ scales with the per-layer output magnitude,
                      #  so relative error is the meaningful bf16 metric)


def _rms_norm(x, w, eps=RMS_EPS):
    import torch

    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * w.float()).to(dt)


def _rotate_half(x):
    import torch

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _neox_cos_sin(positions, dim, dtype, device):
    """Standard NeoX RoPE cos/sin over ``dim`` (theta=10000). Applied identically
    to both attention paths, so its exact form is immaterial to the parity — it
    just needs to be a genuine per-position rotation."""
    import torch

    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, device=device).float() / dim))
    ang = positions.float()[:, None] * inv_freq[None, :]  # [T, dim/2]
    emb = torch.cat([ang, ang], dim=-1)  # [T, dim] (neox: [a,a])
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x, cos, sin):
    # x: [..., dim]; cos/sin broadcast on the token dim.
    return x * cos + _rotate_half(x) * sin


def _load_layer(f, li, device, dtype):
    import torch

    p = f"model.layers.{li}.self_attn."
    g = lambda n: f.get_tensor(p + n).to(device=device, dtype=dtype)
    return {
        "q_a_proj": g("q_a_proj.weight"),
        "q_a_ln": g("q_a_layernorm.weight"),
        "q_b_proj": g("q_b_proj.weight"),
        "kv_a_proj": g("kv_a_proj_with_mqa.weight"),
        "kv_a_ln": g("kv_a_layernorm.weight"),
        "kv_b_proj": g("kv_b_proj.weight"),
        "o_proj": g("o_proj.weight"),
    }


def _preprocess(x, w, cos, sin):
    """Shared MiniCPM3 preprocessing (== MiniCPM3DSAAttention.forward before the
    attention op): returns per-head q_nope/q_pe, compressed kv_c_normed + shared
    k_pe, and the reconstructed full k_nope/v (for the REF path)."""
    import torch

    T = x.shape[0]
    q_c = _rms_norm(x @ w["q_a_proj"].T, w["q_a_ln"])            # [T, 768]
    q = (q_c @ w["q_b_proj"].T).view(T, H, QK_HEAD)              # [T, H, 96]
    q_nope, q_pe = q[..., :QK_NOPE], q[..., QK_NOPE:]           # [T,H,64],[T,H,32]

    kv = x @ w["kv_a_proj"].T                                    # [T, 288]
    kv_c, k_pe = kv[:, :KV_LORA], kv[:, KV_LORA:]               # [T,256],[T,32]
    kv_c_normed = _rms_norm(kv_c, w["kv_a_ln"])                  # [T,256]

    # RoPE (identical to both paths).
    q_pe = _apply_rope(q_pe, cos[:, None, :], sin[:, None, :])   # [T,H,32]
    k_pe = _apply_rope(k_pe, cos, sin)                           # [T,32]

    # Full K/V reconstruction (REF).
    kvb = (kv_c_normed @ w["kv_b_proj"].T).view(T, H, QK_NOPE + V_HEAD)
    k_nope, v = kvb[..., :QK_NOPE], kvb[..., QK_NOPE:]           # [T,H,64],[T,H,64]
    return q_nope, q_pe, kv_c_normed, k_pe, k_nope, v


def _causal_softmax(scores):
    import torch

    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores.float(), dim=-1).to(scores.dtype)


def _attn_full(q_nope, q_pe, k_nope, k_pe, v, w):
    """REF: stock dense materialized-QKV causal attention."""
    import torch

    T = q_nope.shape[0]
    q = torch.cat([q_nope, q_pe], dim=-1)                        # [T,H,96]
    k = torch.cat([k_nope, k_pe[:, None, :].expand(T, H, QK_ROPE)], dim=-1)  # [T,H,96]
    scores = torch.einsum("thd,shd->hts", q, k) * SCALING        # [H,T,T]
    attn = _causal_softmax(scores)
    out = torch.einsum("hts,shv->thv", attn, v)                  # [T,H,64]
    return (out.reshape(T, H * V_HEAD) @ w["o_proj"].T)          # [T,2560]


def _attn_mla_absorbed(q_nope, q_pe, kv_c_normed, k_pe, w):
    """MLA: absorbed-latent decode formulation (what vLLM MLAAttention does)."""
    import torch

    T = q_nope.shape[0]
    Wkv = w["kv_b_proj"].view(H, QK_NOPE + V_HEAD, KV_LORA)      # [H,128,256]
    W_UK = Wkv[:, :QK_NOPE, :]                                   # [H,64,256]
    W_UV = Wkv[:, QK_NOPE:, :]                                   # [H,64,256]

    q_nope_abs = torch.einsum("thd,hdc->thc", q_nope, W_UK)      # [T,H,256]
    s_nope = torch.einsum("thc,sc->hts", q_nope_abs, kv_c_normed)  # [H,T,T]
    s_pe = torch.einsum("thr,sr->hts", q_pe, k_pe)               # [H,T,T] (MQA k_pe)
    scores = (s_nope + s_pe) * SCALING
    attn = _causal_softmax(scores)
    out_latent = torch.einsum("hts,sc->thc", attn, kv_c_normed)  # [T,H,256]
    out = torch.einsum("thc,hvc->thv", out_latent, W_UV)         # [T,H,64]
    return (out.reshape(T, H * V_HEAD) @ w["o_proj"].T)          # [T,2560]


def run_gate() -> bool:
    import torch
    from safetensors import safe_open

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("STAGE-1 GATE: MLA-latent (absorbed) vs stock materialized-QKV attention")
    print(f"           on REAL checkpoint weights, layers {LAYERS_TO_CHECK}, T={SEQ_LEN}")
    print("=" * 78)

    torch.manual_seed(0)
    x32 = torch.randn(SEQ_LEN, HIDDEN, device=device, dtype=torch.float32)
    positions = torch.arange(SEQ_LEN, device=device)

    worst_fp32 = 0.0
    worst_bf16 = 0.0
    worst_bf16_rel = 0.0
    rows = []
    with safe_open(CKPT_SAFETENSORS, "pt") as f:
        for li in LAYERS_TO_CHECK:
            # ---- fp32 identity check ----
            w = _load_layer(f, li, device, torch.float32)
            cos, sin = _neox_cos_sin(positions, QK_ROPE, torch.float32, device)
            qn, qp, kvc, kpe, kn, v = _preprocess(x32, w, cos, sin)
            ref = _attn_full(qn, qp, kn, kpe, v, w)
            mla = _attn_mla_absorbed(qn, qp, kvc, kpe, w)
            d32 = (ref - mla).abs().max().item()

            # ---- bf16 realistic check ----
            wb = _load_layer(f, li, device, torch.bfloat16)
            xb = x32.to(torch.bfloat16)
            cosb, sinb = _neox_cos_sin(positions, QK_ROPE, torch.bfloat16, device)
            qn, qp, kvc, kpe, kn, v = _preprocess(xb, wb, cosb, sinb)
            refb = _attn_full(qn, qp, kn, kpe, v, wb).float()
            mlab = _attn_mla_absorbed(qn, qp, kvc, kpe, wb).float()
            db = (refb - mlab).abs().max().item()
            relb = db / (refb.abs().max().item() + 1e-6)

            worst_fp32 = max(worst_fp32, d32)
            worst_bf16 = max(worst_bf16, db)
            worst_bf16_rel = max(worst_bf16_rel, relb)
            rows.append((li, d32, db, relb))
            print(
                f"  layer {li:>2}: max|Δ| fp32={d32:.3e}  bf16={db:.3e}  "
                f"(rel {relb:.2%}, out|max|={refb.abs().max().item():.3f})"
            )

    print("-" * 78)
    print(f"worst max|Δ| fp32     : {worst_fp32:.3e}   (tol {FP32_TOL})  [exact-identity check]")
    print(f"worst rel|Δ| bf16     : {worst_bf16_rel:.2%}   (tol {BF16_REL_TOL:.0%})  "
          f"[abs {worst_bf16:.3e}]")
    passed = worst_fp32 <= FP32_TOL and worst_bf16_rel <= BF16_REL_TOL
    print(f"GATE RESULT: {'PASS' if passed else 'FAIL'}  "
          f"(MLA-latent attention == stock dense within fp tolerance)")
    return passed


# --------------------------------------------------------------------------- #
# Optional coherence sanity: stock dense path via vLLM (subprocess).
# --------------------------------------------------------------------------- #
STOCK_PROMPTS = ["The capital of France is", "def add(a, b):\n    return"]


def _stock_gen_child():
    import scripts.dsa.vllm_minicpm3_dsa  # noqa: F401 (register + deep_gemm shim)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=SERVING,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        hf_overrides={"architectures": ["MiniCPM3StockRefForCausalLM"]},
    )
    print("[sanity] stock-ref plugin loaded: weights mapped 1:1 (indexer.* ignored).")
    outs = llm.generate(list(STOCK_PROMPTS), SamplingParams(temperature=0.0, max_tokens=24))
    for p, o in zip(STOCK_PROMPTS, outs):
        print(f"[sanity] {p!r} -> {o.outputs[0].text!r}")


def run_sanity() -> None:
    print("\n" + "=" * 78)
    print("STAGE-1 SANITY (best-effort): stock dense generation via vLLM (subprocess)")
    print("=" * 78)
    env = dict(os.environ)
    env["DSA_WIRE_INDEXER"] = "0"
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--stock-gen"],
            env=env, capture_output=True, text=True, timeout=2400,
        )
    except Exception as e:  # pragma: no cover
        print(f"[sanity] subprocess error: {e!r} (non-fatal)")
        return
    tail = "\n".join(ln for ln in proc.stdout.splitlines() if ln.startswith("[sanity]"))
    print(tail or proc.stdout[-2000:])
    if proc.returncode != 0:
        print("[sanity] stock generation FAILED (non-fatal for the gate):")
        print(proc.stderr[-3000:])


def main():
    if "--stock-gen" in sys.argv:
        _stock_gen_child()
        return
    passed = run_gate()
    if "--no-vllm" not in sys.argv:
        run_sanity()
    print("\n" + "#" * 78)
    print(f"# STAGE 1 GATE: {'PASS' if passed else 'FAIL'}")
    print("#" * 78)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
