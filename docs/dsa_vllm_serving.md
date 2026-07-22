# Serving the DSA (Phase-2) MiniCPM3-4B model on vLLM's sparse-attention path

Notes for adapting our MiniCPM3-4B + lightning-indexer (top_k=512) model into vLLM's DeepSeek-V3.2 DSA
backend. Captures the version check, the adaptation plan, and — most importantly — the **kernel dim finding
and the head_dim padding workaround** so it isn't re-derived.

---

## 0. Review status (source-verified 2026-07-20, vllm 0.20.2 in-container)

The kernel-dim finding (§3), the padding workaround (§4), and the weight fusion (§5) were re-checked against
the actual `vllm/model_executor/models/deepseek_v2.py`, `vllm/v1/attention/backends/mla/indexer.py`,
`vllm/model_executor/layers/sparse_attn_indexer.py`, and the vendored DeepGEMM CUDA templates — **all hold**.
Findings, updated with the **Phase-0 GPU probe** (`tests/dsa/probe_deepgemm_indexer.py`, run 2026-07-20 on
H100 against the real DeepGEMM kernel):

- **[VERIFIED] head_dim 64→128 zero-padding is BIT-EXACT on the real kernel** (`max|Δlogit| = 0.0`) — the pad
  can't change the absmax, so the ue8m0 scale and dequant are identical. §4's central claim is now empirical.
- **[NEW — must handle] `index_n_heads=16` is REJECTED by the kernel** — only 32/64/128 are legal
  (`block_qh % num_heads == 0`). Pad heads 16→32 with **zero q AND zero weights** (contributes 0; verified
  exact). §3's "n_heads is a non-issue" was wrong — this is a *second* padding axis alongside head_dim.
- **[NEW — prereq/blocker] this env's deep_gemm lacks `fp8_fp4_mqa_logits`** — vLLM's wrapper returns a
  `_missing()` stub because the external `~/.local` deep_gemm shadows the vendored `vllm.third_party.deep_gemm`
  and predates the unified symbol. Serving must use the vendored deep_gemm (or upgrade the external one). The
  probe called legacy `deep_gemm.fp8_mqa_logits` directly (identical FP8 computation) to get around it.
- **[DOWNGRADED] the missing Hadamard is a ~2% effect, not load-bearing.** On saturated queries (T=2048,
  top_k=512) dropping it changes only ~2% of the top-512 selection (97.8% agreement); and the **FP8 scale
  format** — kernel uses **UE8M0** (scale rounded to a power of two), training used a **plain-absmax float
  scale** — perturbs selection by about the *same* magnitude (kernel-vs-trained ≈ kernel-vs-no-Hadamard, both
  0.978). Note this is purely the *scale format*, not block size: the 64→128 pad is bit-exact (above), so the
  64-vs-128 block is a non-issue. Re-inserting the Hadamard is cheap/harmless but not critical (§4).
- **[CONFIRMED] the kernel applies per-head ReLU** (`max|kernel − Σ_h w·ReLU(q·k)| = 1.6e-4` vs `5.4` for the
  no-ReLU reference) — the custom forward must not double-apply ReLU.
- **[eval scope] "serve dense" is faithful only for `T ≤ top_k` (512)** — unchanged (§6).

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
them): `index_topk` (**512**), `qk_rope_head_dim` (**32**), `q_lora_rank` (**768**).
`wq_b`, `k_norm=LayerNorm(head_dim)`, `softmax_scale=head_dim**-0.5` match our `LightningIndexer`.

**⚠ Correction (Phase-0): `index_n_heads=16` is NOT a free config.** The DeepGEMM logits kernel only accepts
**32/64/128** heads (`block_qh % num_heads == 0` — 12/24/48 also fail). Our 16 heads must be **padded to 32**
in the custom forward, with the 16 extra heads carrying **zero q and zero weights** — since each head
contributes `w_h·ReLU(q_h·k)` and both factors are 0, the padded heads add exactly 0 (verified numerically).
This is a second, independent padding axis on top of the head_dim 64→128 pad.

**The one hard constraint — FP8 block size forces `head_dim ≥ 128`.** The indexer FP8 quant uses a hardcoded
`quant_block_size = 128` (`# TODO: get from config`), and the K-cache layout is
`head_dim + head_dim//quant_block_size * 4` (data + one fp32 scale per 128-block):
- DeepSeek `head_dim=128` → `128 + 1*4 = 132` (1 scale group ✓)
- **Ours `head_dim=64` → `64 + 0*4 = 64`** → `64//128 = 0` scale groups → **FP8 path degenerate.**

So `index_head_dim=64` is incompatible with the 128-wide FP8 blocking. This is *the* dim clash (not
top_k / n_heads).

**Verified against source (2026-07-20):** `deepseek_v2.py:653` `self.quant_block_size = 128  # TODO: get
from config`; k-cache layout `head_dim + head_dim // quant_block_size * 4` at `deepseek_v2.py:660` (→ 132 at
128, degenerate 64 at 64); dims read from config at `deepseek_v2.py:626-628`. Note the *kernel itself*
(`fp8_mqa_logits`) only requires `head_dim % 32 == 0` (WGMMA K-tile = 32; the MLA backend even advertises
`get_supported_head_sizes() -> [32, 64, 128]`) — so 64 is *kernel*-legal. The true blocker is purely the
hardcoded `quant_block_size=128` + k-cache layout, exactly as above. Padding to 128 remains the right fix.

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
  q_idx = rotate_activation(q_idx); k_idx = rotate_activation(k_idx)  # Hadamard over the REAL 64 (as trained; see below)
  q_idx = F.pad(q_idx, (0,64)); k_idx = F.pad(k_idx, (0,64))   # 64 -> 128, zeros
  # -> fp8 quant (block 128 = exactly 1 clean block) -> fp8_mqa_logits
  ```
  Weights load at native 64 (no surgery); `k_norm`/rope run on the real 64 exactly as trained.

### Exact under FP8 — including negative values
FP8 block quant uses **symmetric absmax** scaling: `scale = amax / FP8_MAX`, `amax = max(|x|)` per block
(per token's 128-vector). **Confirmed against the Q kernel** (`fp8_utils.py:346-350`): `_absmax = max(|y|)`,
`scale = _absmax * (1/fp8_max)`, and — since vLLM sets `use_ue8m0=True` (`scale_fmt="ue8m0"`,
`deepseek_v2.py:652,717`) — the scale is then rounded UP to a power of two, `2^ceil(log2(scale))`. The
padding argument survives UE8M0 unchanged: appending zeros can't change `amax`, and the power-of-2 rounding
is monotone in `amax`, so the scale is identical either way. (K-side quant lives in compiled `_C_cache_ops`
and is only inferable by analogy — the §6 parity probe confirms it empirically.)
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

### The missing `rotate_activation` (Hadamard) — must be re-inserted (trained-with, absent in vLLM)
Our indexer trains with an orthonormal Hadamard rotation on q/k **before** FP8 quant (`dsa_indexer.py:279-283`,
`rotate_activation=True` by default; Phase-2 doesn't override it, so the shipped checkpoint uses it). vLLM's
serving path has **no** Hadamard — confirmed at both layers: `Indexer.forward` (`deepseek_v2.py:681-727`)
goes rope → quant directly, and grep over the DeepGEMM CUDA templates (`sm90/sm100_fp8_mqa_logits.cuh`, the
k-cache insert) finds zero hadamard/rotate. So without action the served FP8 q/k differ from training →
noisier ReLU-dots → drifted top-k.

The rotation is orthonormal, so it preserves the *true* dot; its only job is to spread magnitude for cleaner
FP8 blocks. So this is a quant-noise mismatch, not an algebraic one — but it's exactly the noise the trained
model was tuned against, so serve-without-it is strictly worse. **Fix (unambiguous now that the kernel does
NOT rotate):** apply our `rotate_activation` over the **real 64** dims, *before* the pad — same "operate on
the real 64, then pad" rule as `k_norm`. Rotating over the padded 128 instead would mix the zeros into the
reals (H₁₂₈·[v;0] ≠ [H₆₄·v;0]) and change the FP8 scale — the same trap as weight-padding vs `k_norm`.

**Phase-0 update — this is a ~2% effect, so it's cheap-insurance not a blocker.** Measured on saturated
queries (T=2048, top_k=512): dropping the Hadamard changes only ~2% of the top-512 selection (97.8% overlap,
≥95.3% worst-query). Crucially, the residual drift is dominated not by the Hadamard but by the **FP8 scale
format**: the kernel uses **UE8M0** (`scale = 2^ceil(log2(amax/FP8_MAX))`, rounded up to a power of two)
while our training `_fake_quant_fp8` used a **plain-absmax float scale** (`scale = amax/FP8_MAX`,
`dsa_indexer.py:180-182`). Kernel-vs-trained (0.978) ≈ kernel-vs-no-Hadamard (0.978) — adding the Hadamard
difference on top of the scale-format difference doesn't lower the overlap, so the scale format accounts for
essentially all of it. **This is not a block-size issue**: the 64→128 pad is bit-exact (M2 above), so the
64-vs-128 block contributes nothing — it is purely UE8M0 vs plain-absmax.

Why our "faithful DeepSeek port" still differs: we mirrored DeepSeek's *reference* `act_quant` (plain
absmax); the **production DeepGEMM kernel** vLLM calls uses the UE8M0 power-of-2 scaling introduced in the
V3.1/V3.2 FP8 design. So vLLM matches DeepSeek's real kernel; our port matched the simplified reference.
UE8M0 is fixed in the kernel and **not removable at serve time**. The clean fix is training-side and small —
add the power-of-2 rounding to `_fake_quant_fp8` (`scale = 2**ceil(log2(scale))`) and continue/redo the
Phase-2 finetune so the weights adapt to the kernel's numerics. Net for *this* checkpoint: re-insert the
Hadamard (one free op, avoids compounding) but expect ~2% selection drift — inherent unless training is
realigned to UE8M0.

**Update (2026-07-22) — fix landed under a flag.** `_fake_quant_fp8` now takes `use_ue8m0`, wired to the new
`DSAConfig.fp8_ue8m0` (default **False** so prior runs reproduce; set via `+model.override_config={...
dsa_fp8_ue8m0: true}`). When on, training quantizes the indexer's per-row scale to the same power-of-2 the
kernel uses. Verified at two levels:
- **CPU unit** (`tests/dsa/test_indexer_fp8_ue8m0_parity.py`): the training UE8M0 dequant is **bit-identical**
  (`torch.equal`) to the serve reference `_quant_fp8_rows(use_ue8m0=True)`; legacy path unchanged; STE
  gradient intact.
- **GPU kernel** (`tests/dsa/test_minicpm3_dsa_indexer_parity.py`, extended): top-256 selection overlap vs the
  real DeepGEMM kernel rises from **0.9698 mean / 0.9297 min (legacy)** to **1.0000 mean / 0.9961 min (UE8M0)**
  — i.e. the ~3% drift closes to ~0. (UE8M0-fp8 even beats bf16-exact 0.9777, since the kernel is itself FP8.)

To *realize* the gain the indexer must be (re)trained with `dsa_fp8_ue8m0: true`; the existing
`phase2_full_k256_1ep` checkpoint was trained legacy, so it still carries the drift until retrained.

---

## 5. Other weight-remap items for the custom class
- vLLM **fuses** `wk` + `weights_proj` into one `wk_weights_proj = MergedColumnParallelLinear([head_dim,
  n_head])`; ours are **separate** `wk` and `weights_proj` → concat in the loader. **Confirmed:**
  `deepseek_v2.py:641-647` (module) + loader shards `("wk_weights_proj","wk",0)` / `(...,"weights_proj",1)`
  at `deepseek_v2.py:1481-82`.
- **RoPE — our side is already fine; verify vLLM's.** Our indexer uses non-interleaved `rotate_half`
  (`dsa_indexer.py:108-112`), so there is nothing to "convert" on our side — the task is to confirm vLLM
  0.20.2's indexer rope is non-interleaved (that the 2025-11-17 fix is in this build). Also: the "MLA stays
  interleaved" caveat is a *DeepSeek* fact — **MiniCPM3's MLA rope is non-interleaved (llama `rotate_half`)**,
  so the custom class must use MiniCPM3's convention for MLA, not inherit DeepSeek's interleaved one.
- **`softmax_scale`.** vLLM sets `softmax_scale = head_dim**-0.5` (`deepseek_v2.py:650`) = 128^-0.5 with the
  padded config, vs 64^-0.5 in training. It's a positive global scale → **top-k selection is invariant** (so
  selection is unaffected), but it shifts raw indexer *scores* by √0.5 — in §6 parity, compare LM logits /
  selected-index sets, not raw indexer scores.

---

## 6. Evaluation feasibility & sequencing

**The faithfulness bound that governs everything.** The Phase-2 model masks its main attention to the top-k
selected keys (`_sparse_attn`, `minicpm_dsa.py:432`), with exact parity `sparse ≡ dense` **iff `T ≤ top_k`**
(`minicpm_dsa.py:448-450`). So:

- **Tier 1 — dense vLLM, now, zero integration.** Load the Phase-2 *main* weights into stock
  `MiniCPM3ForCausalLM` (drop `*.indexer.*`). For any benchmark with prompt+generation `≤ ~512` tokens this is
  **numerically identical** to the trained sparse model — a faithful eval. Above 512 it diverges and flatters
  the model. Tag each benchmark with its token-length distribution; treat dense eval as a *capability /
  regression check* (did behavior-cloning preserve the base?), **not** as validation of the sparse mechanism.
- **Tier 2 — HF sparse forward, faithful at any length, no new code.** `_sparse_attn` already runs true
  top-k sparse attention. It's slow and has no incremental-decode KV cache, so it's limited to teacher-forced
  scoring — but that covers **logprob / multiple-choice / perplexity** benchmarks faithfully at long context
  *today*. Use this to validate the sparse mechanism before investing in the vLLM kernel path.
- **Tier 3 — `MiniCPM3DSAForCausalLM` on vLLM's sparse path.** Needed only for fast long-context *generation*
  throughput. Build it per §2-§5 (padding, re-inserted Hadamard, wk-weights_proj concat, RoPE checks).

**Empirical checks before/with Tier 3:**
- Kernel probe: call the indexer logits kernel with **head_dim=128 (padded), n_heads=16, rope=32** to confirm
  the path runs at our (padded) dims. Low risk — 128 is DeepSeek's native head_dim, and the kernel only needs
  `head_dim % 32 == 0`.
- Parity: vLLM sparse-served **LM logits / selected-index sets** ≈ HF sparse forward (`dsa_mode=sparse,
  top_k=512`) on sample prompts (this also empirically pins down the compiled K-side quant). Compare logits,
  not raw indexer scores (§5 `softmax_scale`).
