# Serving the DSA (Phase-2) MiniCPM3-4B model on vLLM's sparse-attention path

Notes for adapting our MiniCPM3-4B + lightning-indexer (top_k=512) model into vLLM's DeepSeek-V3.2 DSA
backend. Captures the version check, the adaptation plan, and — most importantly — the **kernel dim finding
and the head_dim padding workaround** so it isn't re-derived.

---

## 1. vLLM DSA availability (checked in-container, 2026-07-17)

- vLLM shipped **Day-0 DSA support on 2025-09-29** (DeepSeek-V3.2-Exp): lightning-indexer FP8 kernels in
  **DeepGEMM** + sparse-MLA in **FlashMLA**.
- **Our container's `vllm 0.20.2` already has it** — verified:
  - `DeepseekV32ForCausalLM` is registered (`registry.py` → `deepseek_v2.py`).
  - `deepseek_v2.py` has `class Indexer(nn.Module)`, `DeepseekV32IndexerCache`, and imports
    `sparse_attn_indexer`.
  - `import deep_gemm` works; it exports `fp8_mqa_logits`, `fp8_paged_mqa_logits`.
  - vLLM also has a **dense** `MiniCPM3ForCausalLM` (`minicpm3.py`) — but no sparse variant.
- No vLLM upgrade needed. (SGLang also supports V3.2 DSA as an alternative.)

---

## 2. Why it's not plug-and-play

vLLM's DSA is bound to the DeepSeek-V3.2 model class and its dims (MoE MLP, `index_n_heads=64`,
`index_head_dim=128`, `top_k=2048`, non-interleaved indexer RoPE, FP8 via DeepGEMM). MiniCPM3 is dense-MLP +
muP-scaled with different dims and **our** indexer (16 heads, head_dim 64, top_k 512). We can't masquerade as
DeepSeek-V3.2 (it's MoE). The path is a dedicated **`MiniCPM3DSAForCausalLM`** vLLM class that reuses vLLM's
`Indexer` / `sparse_attn_indexer` / `DeepseekV32IndexerCache` ops but wires them onto vLLM's MiniCPM3 MLA.

---

## 3. Step-4 finding: which dims are parametric vs constrained

vLLM's `Indexer` (`deepseek_v2.py:609+`) reads most dims from config — these are **non-issues** (just set
them): `index_topk` (**512**), `index_n_heads` (**16**), `qk_rope_head_dim` (**32**), `q_lora_rank` (**768**).
`wq_b`, `k_norm=LayerNorm(head_dim)`, `softmax_scale=head_dim**-0.5` match our `LightningIndexer`.

**The one hard constraint — FP8 block size forces `head_dim ≥ 128`.** The indexer FP8 quant uses a hardcoded
`quant_block_size = 128` (`# TODO: get from config`), and the K-cache layout is
`head_dim + head_dim//quant_block_size * 4` (data + one fp32 scale per 128-block):
- DeepSeek `head_dim=128` → `128 + 1*4 = 132` (1 scale group ✓)
- **Ours `head_dim=64` → `64 + 0*4 = 64`** → `64//128 = 0` scale groups → **FP8 path degenerate.**

So `index_head_dim=64` is incompatible with the 128-wide FP8 blocking. This is *the* dim clash (not
top_k / n_heads).

---

## 4. The head_dim padding workaround (64 → 128)

Pad the indexer's contraction axis `head_dim` `64 → 128` with **zeros** on both tensors that enter the FP8
dot (`fp8_mqa_logits`): the per-head query `q_idx [T,16,64]→[T,16,128]` and the shared MQA key
`k_idx [T,64]→[T,128]`. Because the score is `I[t,s] = Σ_h w[t,h]·(q_idx[t,h]·k_idx[s])`, a dot over 128 =
`Σ_{d<64} q·k + Σ_{64≤d<128} 0·0` = **exactly the 64-dim dot** → logits and top-k selection unchanged.

### Pad the ACTIVATIONS, not the weights (critical)
- **✗ weight-padding** (`wq_b`/`wk` outputs 64→128 with zero rows) **breaks `k_norm`**: `k_norm` is a
  `LayerNorm(head_dim)` on `k_idx` before the dot; over `[64 real + 64 zeros]` the mean/variance are computed
  over 128, so the 64 zeros shift the stats and the real dims get normalized differently than in training →
  wrong logits.
- **✓ activation-padding** in the custom indexer forward (we own it), inserted **after** `k_norm`+rope and
  **before** the FP8 quant/kernel:
  ```
  q_idx = wq_b(qr)                 # [T,16,64]
  k_idx = k_norm(wk(hidden))       # LayerNorm over the REAL 64  (as trained)
  q_idx, k_idx = apply_rope(...)   # rope on rope_dim(32) portion (as trained)
  q_idx = F.pad(q_idx, (0,64)); k_idx = F.pad(k_idx, (0,64))   # 64 -> 128, zeros
  # -> fp8 quant (block 128 = exactly 1 clean block) -> fp8_mqa_logits
  ```
  Weights load at native 64 (no surgery); `k_norm`/rope run on the real 64 exactly as trained.

### Exact under FP8 — including negative values
FP8 block quant uses **symmetric absmax** scaling: `scale = amax / FP8_MAX`, `amax = max(|x|)` per block
(per token's 128-vector).
- **Scale is unchanged by the zeros** — `|0| = 0` is the minimal magnitude, so appending zeros never changes
  `amax`, **regardless of the sign of the 64 real values** (all-negative reals still have positive
  magnitudes; e.g. `[-5,-3,-2]` → `amax=5`, zeros keep it 5). E4M3 is signed, so negatives round natively.
- **Dot is exact:** dequant `Σ real·real + Σ 0·0`. The zeros quantize to 0 and contribute 0.
- Caveat: only an **all-zero real vector** is degenerate (`amax=0 → div-by-zero`) — pre-existing, epsilon-
  clamped, not caused by padding. (This relies on symmetric absmax; an asymmetric min/max+zero-point quant
  would be shifted by zeros — DeepGEMM's is symmetric absmax, so we're fine.)

### Cost & why padding beats patching the block size
Indexer runs at head_dim 128 (~2× its tiny FLOPs/mem — negligible), and lands on the kernel's **native
128 tile**. The alternative — patch `quant_block_size=64` — needs DeepGEMM's compiled `fp8_mqa_logits` to
support a 64-wide block, which is unverifiable from the `.so`; padding to 128 keeps the proven kernel path
with zero kernel changes.

---

## 5. Other weight-remap items for the custom class
- vLLM **fuses** `wk` + `weights_proj` into one `wk_weights_proj = MergedColumnParallelLinear([head_dim,
  n_head])`; ours are **separate** `wk` and `weights_proj` → concat in the loader.
- Convert the indexer RoPE to vLLM's **non-interleaved** layout (the 2025-11-17 vLLM indexer-RoPE fix; MLA
  stays interleaved).

---

## 6. Remaining empirical check & sequencing
- Standalone probe: call `deep_gemm.fp8_mqa_logits` with **head_dim=128 (padded), n_heads=16, rope=32** to
  confirm the kernel path runs at our (padded) dims. Low risk since 128 is DeepSeek's native head_dim.
- Parity: vLLM sparse-served logits ≈ our HF sparse forward (`dsa_mode=sparse, top_k=512`) on sample prompts.
- **Now (capability eval):** serve dense (mild sparsity at benchmark lengths) — no DSA integration needed.
- **DSA serving:** only for the long-context efficiency goal; build `MiniCPM3DSAForCausalLM` then, starting
  with the kernel probe above.
