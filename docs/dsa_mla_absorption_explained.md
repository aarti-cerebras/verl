# MLA: decompress vs. absorbed modes (and why sparse serving needs the absorbed one)

Reference note explaining how MiniCPM3's MLA attention can be computed two mathematically-identical ways, why
vLLM uses the "decompress" way today, and why our Tier-3 sparse serving must switch it to the "absorbed" way.
Written to settle the recurring "MiniCPM3 already has MLA — why re-express it?" question.

---

## 1. What MLA stores (MiniCPM3 numbers)

Multi-head Latent Attention does **not** cache a full key/value per head. Per token it produces a small
**latent**, and the full per-head K/V are *reconstructed* from it via a weight when needed.

MiniCPM3-4B dims:
- hidden = 2560, heads = 40, `qk_nope=64`, `qk_rope=32` (⇒ `qk_head_dim=96`), `v_head=64`.
- Query: `q_a_proj` (2560→768) → `q_a_layernorm` → `q_b_proj` (768 → 40×96 = 3840).
- KV: `kv_a_proj_with_mqa` (2560 → **256** kv-latent `+ 32` shared RoPE key = **288**). The 256 latent is what
  gets cached. `kv_b_proj` (256 → 40×(64 nope + 64 value) = 5120) reconstructs full per-head K (nope) and V.

So per token: **latent ≈ 288 numbers** vs **full per-head K/V ≈ 40×(96+64) ≈ 6400 numbers**. Caching the
latent instead of full K/V is the entire point of MLA.

---

## 2. "Decompress back to full per-head Q/K/V"

`kv_b_proj` (and `q_b_proj`) **expand** the small latent into the full 40-head Q, K, V tensors that ordinary
attention uses. "Decompress" = run those `*_b_proj` weights so you hold full per-head Q/K/V. Once you have
them, you run standard multi-head attention `softmax(Q·Kᵀ/√d)·V` per head — nothing MLA-specific.

**MLA doesn't "become" full attention — it already *is* full attention**, where K and V just happen to be
produced by a low-rank factorization (latent → `kv_b_proj`). Materialize K/V and it's ordinary attention.

---

## 3. The absorbed mode, and why it gives the identical number

The absorbed mode never materializes full K/V. It uses one fact: **matrix multiplication is associative.**

A key is `k = W·c` (c = latent, W = `kv_b_proj`'s key slice `W_UK` for a head). The score is `q·k`:

```
q · k  =  q · (W·c)  =  (Wᵀ·q) · c
```

Group `q·W·c` as `(W·c)` first → **decompress** the key, then dot with q.
Group it as `(Wᵀ·q)` first → **fold W into the query** (once, offline), then dot the **latent** directly.

### Worked example (tiny dims: latent c is 3-d, key/query are 2-d, so W is 2×3)
```
q = [1, 2]
W = [[1, 0, 1],
     [0, 1, 1]]
c = [1, 2, 3]
```
**Way 1 — decompress then dot:**
```
k = W·c = [1·1+0·2+1·3, 0·1+1·2+1·3] = [4, 5]
q·k     = 1·4 + 2·5 = 14
```
**Way 2 — fold weight into query (q̃ = Wᵀ·q) then dot the latent:**
```
q̃ = Wᵀ·q = [1·1+0·2, 0·1+1·2, 1·1+1·2] = [1, 2, 3]
q̃·c      = 1·1 + 2·2 + 3·3 = 14
```
**Both = 14.** That is the whole trick: `q·(W·c) = (Wᵀ·q)·c`.

### The value / output side (same idea, via linearity)
Each value is also decompressed, `v_s = W_V·c_s`, so the head output pulls `W_V` out of the sum:
```
out = Σ_s a_s·v_s = Σ_s a_s·(W_V·c_s) = W_V·(Σ_s a_s·c_s)
```
i.e. attend over the **latents** to get a context `Σ_s a_s·c_s`, then apply `W_V` once (fold it into `o_proj`).

---

## 4. Why the cache differs

- `c` is **one small latent per token, shared across heads** (256-d).
- Each head's key `k_h = W_h·c` differs per head. **Way 1 (decompress)** must build & cache all 40 `k_h` → big
  KV cache. **Way 2 (absorbed)** keeps only `c` in the cache and folds each head's `W_h` into that head's
  query → cache stays the tiny latent.

| mode | cache holds | how attention is done | result |
|---|---|---|---|
| **Decompress (dense)** — vLLM's MiniCPM3 today | latent expanded to full per-head K/V | ordinary `softmax(QKᵀ)V` | identical |
| **Absorbed (MLA-latent)** — DeepSeek path | the small shared latent `c` | fold `kv_b` into q/o, attend over `c` | identical |

Numerically identical (associativity for scores, linearity for values) — this is the `max|Δ|=5.7e-5` fp32
parity measured in Stage 1.

---

## 5. Why sparse serving needs the absorbed mode

The DSA sparse machinery (the indexer's top-k selection + the FlashMLA-sparse kernel) **operates on the cached
latents** — it selects which latent tokens `c_s` to attend to and does the absorbed-MLA math over them. vLLM's
decompress/dense path has full per-head K/V and **no sparse variant**. So:

> MiniCPM3 already has MLA *weights*, but vLLM *executes* them in decompress/dense mode (which can't be made
> sparse). Tier-3 Stage 1 switches execution to the absorbed/latent mode — same math (guaranteed by
> `q·(W·c)=(Wᵀ·q)·c`), but now the cache is latents the sparse kernel can operate on.

Analogy: a zip file. Unzip it and work with the full contents (decompress-dense), or operate directly on the
zipped bytes with tools that understand the format (absorbed-latent). Same data — but the sparse tool only
knows how to work on the zipped form.

See `docs/dsa_vllm_minicpm3dsa_build_plan.md` for how this drives the build (Stage 1 = absorbed-mode rewrite,
proven exact; Stage 2 = pad the MLA path to 576 + enable the sparse kernel).
