# Qwen3-4B MSA — vLLM serving plan

**Goal:** serve the Phase-2b sparse Qwen3-4B-Thinking-2507 checkpoint on vLLM, with enough parity
evidence that any number the eval ladder produces is attributable to the model rather than to the
serving path.

**Status:** written 2026-08-04, while both Phase-2b runs are mid-flight.
**P0–P4 DONE (2026-08-04/05)** — see §9. Environment is `.devlibs/vllm026` (vLLM 0.26.0,
torch 2.11.0+cu130). **Route A** — reuse M3's fused kernel, export main `q_norm`/`k_norm` as
`w − 1`. Serving dir `/cb/ml-eng/aarti/msa/serving/k8_step1400`; plugin
`scripts/msa/vllm_qwen3_msa/`. **The served model computes the trained function**: prefill and
decode both agree with the training forward inside the measured floors, and the selection kernels
agree with our torch selection on 98.5% of queries with every disagreement a near-tie (§9 P4).
**P5 (throughput) is next** — the last gate before benchmarks.

Parity work also uncovered a real **training-side bug** (partial-block aliasing), now fixed —
[§11](#11-training-side-bug-found-by-parity-partial-block-aliasing), including the restart note.

**The plan is [§9](#9-order-of-work)** — six phases, P0–P5, ~5–6 days, none of them blocked on
training finishing. §1–§8 are the justification: what changed since [plan.md](plan.md) §7 (§2), how
the kernel reuse works (§3), what we write (§4–5), what has to pass before a benchmark counts (§6),
how to launch it (§7), and what can still go wrong (§8).

**Relationship to existing docs.** This supersedes [plan.md](plan.md) §7 on three points of fact
(§2 below) and adds the execution detail §7 never carried. It does not touch the training plan.
[../qwen3_4b_dsa/eval_plan.md](../qwen3_4b_dsa/eval_plan.md) supplies the benchmarks, but **its §3
de-confounding ladder is not being run** — see §6.4 for the scope decision and its consequence.

**Verification discipline** ([plan.md](plan.md) §11): every claim about vLLM below was read from
source at tag **`v0.26.0`**, cited `file:line`. Claims that are *not* verified are labelled
**[UNVERIFIED]** and each has an assigned check in §9. Nothing here is inferred from a signature.

---

## 1. Where things stand

**Training (as of 2026-08-04 22:00 UTC).** Both Phase-2b runs are live and healthy; neither is the
"final" checkpoint.

| run (`/cb/ml-eng/aarti/msa/sparse/_ckpt/`) | step | s/it | remaining |
|---|--:|--:|--:|
| `p2_…_k8_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t16w2048_st11214` | 1428 / 11214 | 7.8–11.5 | ~21–31 h |
| `p2_…_k16_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t16w2048_st11214` | 858 / 11214 | 21.9–26.2 | ~63–75 h |

`indexer/kl` 0.067 (k8) / 0.089 (k16) and falling, `kl_share_of_loss` 0.4–1.2, peak 34.5 GB, no NaN.
Checkpoints every 100 steps as FSDP2 DTensor shards plus a `huggingface/` dir holding config +
tokenizer only.

**Dense baseline (eval ladder row 0) is done.** `/cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking`:
IFEval 88.72, AIME25 avg@16 83.13 ± 0.83, GPQA-diamond avg@4 64.40 ± 1.18, MMLU-Pro 72.50,
LiveCodeBench v6 53.82, RULER 4K/8K/16K/32K 97.15 / 96.46 / 96.65 / 95.33, MRCR, GSM-Infinite.
NoLiMa and BFCL v3 not started.

**The blocking fact.** There is currently no way to sample from this model at all.

1. The container ships vLLM 0.20.2, which has no `minimax_m3`.
2. There is no HF fallback either: `_sparse_attn_and_kl` recomputes index keys from the *current*
   `hidden_states` (`verl/models/transformers/qwen3_msa.py:756`), so under a KV cache at decode
   `k_idx` has length 1 while `key_states` has length `T`. The training module is prefill-only.

Consequence: the serving stack is the critical path for evaluation, and (2) also means the only
oracle we can build parity against is a **prefill / teacher-forced** one. That is sufficient — see
§6.5.

---

## 2. Corrections to [plan.md](plan.md) §7

§7 was written as a feasibility study and reached the right conclusion — no new kernels are needed.
Three of its premises have since gone stale or were inverted.

### 2.1 No source build is required

§7 states the section "requires vLLM `main` at or past PR #45381" and §10 item 5 schedules the
build as *"must-do, blocks all serving"*.

**`vllm/models/minimax_m3/` ships in released tags.** GitHub contents API: present at `v0.24.0`,
`v0.25.0`, `v0.26.0`; 404 at `v0.21.0` and `v0.20.2`. PyPI latest is **0.26.0** (uploaded
2026-07-25) and its metadata pins **`torch==2.11.0`** — the exact torch already installed
(`torch 2.11.0+cu130`).

Item 5 therefore collapses to `pip install vllm==0.26.0` in a fresh venv. **Confirmed 2026-08-04
(P0):** the binary wheel *does* ship the compiled `fused_minimax_m3_qknorm_rope_kv_insert` symbol —
this was the one thing §2.1 could not establish from the tag alone, and it is what keeps Route A open.

### 2.2 The checkpoint weight contract is violated by our own checkpoint

§4.1 settles the naming as `self_attn.{index_q_proj, index_k_proj, index_q_norm, index_k_norm}` and
notes vLLM folds the projections into its fused GEMM itself, "no manual fusion".

vLLM's mapping (`nvidia/model.py:902-911`):

```python
(".qkv_proj", ".index_q_proj", "index_q"),
(".qkv_proj", ".index_k_proj", "index_k"),
```

Our checkpoint, read from `…/global_step_1400/model_world_size_8_rank_0.pt`:

```
model.layers.3.self_attn.indexer.index_q_proj.weight   (1024, 2560)
model.layers.3.self_attn.indexer.index_k_proj.weight   ( 128, 2560)
model.layers.3.self_attn.indexer.index_q_norm.weight   ( 128,)
model.layers.3.self_attn.indexer.index_k_norm.weight   ( 128,)
```

An extra `.indexer.` level, because the training module nests them in an `MSAIndexer` submodule
(`verl/models/transformers/msa_indexer.py:210-213`). 132 tensors across 33 layers. The export must
strip that level. **The failure mode is silent** — every branch of M3's `load_weights` ends in
`if name not in params_dict: continue` (`nvidia/model.py:940-941`).

### 2.3 The Gemma-norm problem is the other way round

§3.2 and §7.1 frame the Gemma `w − 1` conversion as an *index*-branch concern and offer "write our
own norm/RoPE path" as the escape, calling the fused op "optional for our port".

Both halves are inverted:

- **Our index norms already match.** `MSARMSNorm` is Gemma-parameterised — `x * rsqrt(...) *
  (1 + weight)`, zero-init — and the docstring says so explicitly: *"Matches vLLM's
  `MiniMAXGemmaRMSNorm`"* (`verl/models/transformers/msa_indexer.py:169-186`). No conversion.
- **The main q/k norms do not.** M3 builds `MiniMAXGemmaRMSNorm` for `q_norm`/`k_norm`
  (`nvidia/model.py:464-465`); Qwen3's are standard RMSNorm, and that is what we trained.
- **The fused op is unconditional, and its norm is Gemma for *all four*.** The kernel has a single
  norm helper, `normAndRope`, computing `x * rsqrt(mean(x²)+eps) * (1 + w)`
  (`csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:135-149`, the
  `1.0f + weight[dim]` at `:145`), applied to whichever weight pointer it is handed.
  `MiniMaxM3SparseAttention.forward` calls it with no branch (`nvidia/model.py:604`), passing
  `q_norm`, `k_norm`, `index_q_norm` and `index_k_norm` alike.

**So there are four norms per sparse layer, and we got the two that were a design decision right:**

| norm | our training module | kernel expects | conversion at export |
|---|---|---|---|
| `index_q_norm`, `index_k_norm` | Gemma `x*(1+w)`, zero-init — deliberately matched | Gemma | **none** ✅ |
| `q_norm`, `k_norm` (main attention) | stock Qwen3 `RMSNorm`, `x*w` — inherited from the pretrained model, never touched | Gemma | `w' = w − 1` |

So reusing M3's attention class **forces** a `w − 1` export of the main q/k norms. The escape hatch
exists only if we override `forward` wholesale (Route B, §4.3). Open item 6 ("Gemma-norm decision")
is thereby answered for the index branch and re-scoped to the main branch.

### 2.4 Two §7 claims that verify clean

Recorded so the audit is even-handed.

- **Score scale is omitted at serving.** `common/ops/index_topk.py:291`: *"The score scale is
  omitted"*. `MiniMaxM3Indexer` stores `self.scale` (`common/indexer.py:375`) and never uses it. §3.2's
  `√128` warning is correct: never compare train/serve score magnitudes, never threshold on absolute
  scores. Selection is unaffected (top-k is scale-invariant).
- **The silent-dense mechanism.** `_sparse_attention_layer_ids` returns `set()` when
  `sparse_attention_config` or its `sparse_attention_freq` is missing (`nvidia/model.py:95-103`), so
  every layer builds dense and every `index_*` weight is then dropped by the loader. §7.3 is accurate.

Also checked and found to be a non-issue: `sparse_disable_index_value` appears **only in a docstring**
(`nvidia/model.py:400`) and is never read. M3 checkpoints simply lack `index_{v,o}_proj`; we need no
config key for it.

---

## 3. How the reuse works

vLLM's M3 implementation is three layers with very different coupling.

| layer | files | coupling to M3 | our action |
|---|---|---|---|
| **Kernels** | `common/ops/index_topk.py`, `common/ops/sparse_attn.py`, `csrc/…qknorm_rope_kv_insert_kernel.cu` | none — pure functions | reuse verbatim, **never called directly** |
| **Attention plumbing** | `common/indexer.py` (`MiniMaxM3Indexer`, side cache, backend, metadata builder), `common/sparse_attention.py` (`MiniMaxM3SparseImpl`, backend, metadata builder) | none — constructed from plain ints | reuse verbatim |
| **Model** | `nvidia/model.py` (`MiniMaxM3SparseAttention`, decoder layer, `ForCausalLM`, MTP, vision) | heavy — MoE, Gemma norms throughout, partial RoPE | subclass one class, write the rest |

**We never call a kernel.** We inherit the class that owns them. `MiniMaxM3SparseAttention.__init__`
builds two objects out of plain integers — `self.impl` via `select_main_impl_cls(...)`
(`nvidia/model.py:513`) and `self.indexer = MiniMaxM3Indexer(...)` (`nvidia/model.py:527`) — and the
whole per-token compute is two calls inside one eager break (`nvidia/model.py:637-641`):

```python
self.indexer(index_query)                                      # score + top-k -> topk_indices_buffer
return self.impl.forward(self, query, self.kv_cache, output)   # block-sparse attend
```

Neither constructor sees an M3 config object. That is why this is a subclass and not a fork: we
replace only the class's *config-facing edges* and inherit the backend wiring, both KV-cache specs,
both metadata builders, seven Triton kernels and one CUDA kernel.

### 3.1 Geometry fit

`MiniMaxM3SparseAttention.__init__` reads only generic fields (`nvidia/model.py:414-475`):
`hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `rms_norm_eps`,
`max_position_embeddings`, `rope_theta`, `partial_rotary_factor`, plus the
`sparse_attention_config` dict.

| quantity | Qwen3-4B MSA | vLLM requirement | fit |
|---|--:|---|---|
| index heads | 8 | `== num_key_value_heads` (asserted) | ✅ |
| `index_dim` | 128 | backends report `get_supported_head_sizes() -> [128]` | ✅ |
| block size | 128 | `get_supported_kernel_block_sizes() -> [128]` (`common/sparse_attention.py:107-109`, `common/indexer.py:93-94`) | ✅ |
| dense prefix | 3 | any per-layer `sparse_attention_freq` list | ✅ (M3 ships `[0]*3 + [1]*57`) |
| `top_k` | 8 / 16 | any; SM100 CuTe path additionally gates on `{4,8,16,32}` | ✅ (we are SM90 → Triton either way) |
| head_dim | 128 | fused kernel `kHeadDim=128` | ✅ |

Three mismatches, all in §4:

1. main `q_norm`/`k_norm` parameterisation (§2.3);
2. RoPE — M3 is partial (`partial_rotary_factor`), Qwen3 is full → set `partial_rotary_factor: 1.0`,
   giving `rotary_dim = 128`, admitted by the kernel's own guard (`rotary_dim > 0 && %8 == 0 && <= 128`)
   but **never exercised at 128 by M3** → numerical check, §9 P1;
3. `o_proj(reduce_results=False)` (`nvidia/model.py:455-462`) assumes M3's decoder layer fuses the
   all-reduce with a following GemmaRMSNorm. **We run TP=1**, which moots it — a 4B model on one
   H100 has no reason to shard.

### 3.2 What the kernels replace — the train/serve correspondence

One sparse layer at serving is five stages. Mapped against the training-side sequence in
[phase2_plan.md](phase2_plan.md) §2, this is the correspondence the P4 parity tests check row by row.

| training step (`verl/models/transformers/qwen3_msa.py`) | serving component | kind |
|---|---|---|
| **A1** `q_proj`/`k_proj`/`v_proj` **+ B4/B5** `index_q_proj`/`index_k_proj` | `qkv_proj` — **one** fused GEMM emitting `[q ǀ k ǀ v ǀ index_q ǀ index_k]` (`MinimaxM3QKVParallelLinearWithIndexer`) | Linear layer, not a kernel |
| **A1** `q_norm`/`k_norm`, **A2** RoPE on q/k, **B4/B5** index norms + RoPE | `fused_minimax_m3_qknorm_rope_kv_insert` | **1 CUDA kernel** |
| *(no training analogue)* — write k/v to the paged cache, `index_k` to the side cache | same kernel, same launch | — |
| **C6** `S_idx = q_idx·k_idxᵀ` + masking, **C7** max-pool to blocks | `_index_block_score_kernel` | Triton |
| **C8** force local block, **C9** `topk(k)` | `_topk_index_kernel`; decode: `_decode_index_score_kernel` → `_topk_index_partial_kernel` → `_topk_index_merge_kernel` | Triton |
| **C10** expand block ids → tokens, **D11** gather K/V, **D12–15** scores / fp32 softmax / `·V` | `_gqa_sparse_fwd_kernel`; decode: `_gqa_sparse_decode_kernel` → `_merge_topk_attn_out_kernel` | Triton |
| **D16** `o_proj` | stock `RowParallelLinear` | vLLM |
| **E17–20** teacher + KL, **F** loss | **nothing** | training-only |

1 CUDA + 7 Triton kernels per sparse layer; 5 launches per decode step.

Three consequences worth stating explicitly:

1. **The fused CUDA kernel does no attention.** It is pre-attention plumbing — norms, RoPE and cache
   writes for all five projections in one pass. It is also the *only* thing that touches
   `q_norm`/`k_norm`, which is the entire origin of §2.3's `w − 1`.
2. **The five projections collapse into one GEMM.** Training keeps them as five separate `nn.Linear`s;
   serving fuses them, which is what `stacked_params_mapping` does at load time — and why the export
   must strip `.indexer.` (§2.2).
3. **Serving never materialises the gather.** Training step D11 builds `K_g`/`V_g` at
   `[b, 8, T_q, 2048, 128]`, ~2.1 GB apiece in bf16 and the dominant memory cost of Phase 2.
   `_gqa_sparse_fwd_kernel` does not: because `sparse_block_size == page_size == 128`, selected block
   ids *are* page ids, so it reads the paged cache in place. This is also why the training path had to
   be reimplemented in torch rather than calling these kernels ([plan.md](plan.md) §4.1) — they are all
   `@torch.no_grad()`, the score kernel applies `max` *inside* and never emits the token-level `S_idx`
   that Eq. 9's KL student needs, and their K side is a paged cache that does not exist in training.

**Untouched by any of this:** embeddings, MLP, input/post-attention layernorms, `lm_head`, sampling.
And layers 0–2 use none of the sparse path at all — `msa_dense_prefix = 3`, so they build a plain
`Qwen3Attention` with FlashAttention and no index branch, exactly as they trained. Per sparse layer
the replaced region is precisely: after the residual-stream input, through the attention output,
nothing else.

---

## 4. What we write

Modelled directly on `scripts/dsa/vllm_minicpm3_dsa/`, which is the proven in-repo pattern.

### 4.1 Size comparison against the DSA plugin

| DSA file | lines | MSA equivalent | why |
|---|--:|---|---|
| `__init__.py` | 207 | **~20** | no `deep_gemm` shim (M3 is Triton); no force-MLA monkeypatch — DSA needed one because vLLM gates MLA on a DeepSeek-family allowlist keyed on `model_type`, whereas the M3 sparse backend is bound directly by the attention module (`self.attn_backend = MiniMaxM3SparseBackend`) |
| `attention.py` | 465 | **~60** | DSA had to restructure attention into MLA-latent form and pad heads 40→64 and latent →576, because `MLACommonBackend.get_supported_head_sizes() == [320, 576]` excludes MiniCPM3's native 288. MSA needs no restructuring: GQA→GQA, head_dim 128 native |
| `indexer.py` | 478 | **0** | `MiniMaxM3SparseAttention` builds `MiniMaxM3Indexer` itself, side cache included |
| `model.py` | 307 | **~250** | closest 1:1 port |

### 4.2 `scripts/msa/vllm_qwen3_msa/`

**`__init__.py`** — registration only, lazy string form so importing the package does not drag in
vLLM's model layer (`scripts/dsa/vllm_minicpm3_dsa/__init__.py:181-201`):

```python
ModelRegistry.register_model("Qwen3MSAForCausalLM",
                             "scripts.msa.vllm_qwen3_msa.model:Qwen3MSAForCausalLM")
```

**`attention.py` — not needed** (confirmed in P3). `MiniMaxM3SparseAttention` is used *directly*:
it reads only generic config fields, so it takes Qwen3 geometry unmodified. The one adaptation is a
config shim, not a subclass — `_ensure_m3_rope_fields()` sets `rope_theta` and
`partial_rotary_factor` as **attributes on the live config** before the layers are built, because
transformers 5's `Qwen3Config` keeps rope settings only in the nested `rope_parameters` dict while
M3's attention reads them flat (`nvidia/model.py:468-475`). Writing them flat into `config.json`
(§5.2) is not sufficient — HF drops them on load.

**`model.py`** — three shells:

- `Qwen3MSADecoderLayer` — stash the sparse wiring before building the attention block, exactly as
  `MiniCPM3DSADecoderLayer.__init__` does (`model.py:90-136`); Qwen3's dense MLP and standard
  RMSNorms; layers `< msa_dense_prefix` get stock `Qwen3Attention` (M3's own branch is
  `nvidia/model.py:676-690`).
- `Qwen3MSAModel` — override `_init_layers` to allocate the shared `topk_indices_buffer` **before**
  building layers, pulling the live config with `get_current_vllm_config()`, then thread buffer +
  config through `make_layers` (`MiniCPM3DSAModel._init_layers`, `model.py:139-181`). Shape per
  `nvidia/model.py:795-806`: `[pad4(max_num_batched_tokens), num_index_heads, topk_blocks]` int32,
  token-major, padded to a multiple of 4 for `build_k2q_csr`'s int4 loads. **Keep it a plain tensor
  attribute**, never a Parameter or registered buffer, or it enters `state_dict` and gets pickled
  onto `hf_config` (DSA `model.py:152-154`).
- `Qwen3MSAForCausalLM` — `_init_model` returns the above; strict `AutoWeightsLoader`.

Not written, per §6.4's scope: the `MiniCPM3StockRefForCausalLM` analogue (`model.py:271-282`), which
would serve the same directory through stock dense Qwen3. Eleven lines if it is ever wanted back —
but note it would need main `q_norm`/`k_norm` unshifted by `+1` under Route A (§2.3), since stock
Qwen3 `RMSNorm` computes `x * w` where the Gemma norm computes `x * (1 + w)`.

Import note: `vllm/models/` at `v0.26.0` contains only `deepseek_v32`, `deepseek_v4`, `inkling`,
`minimax_m3`. Qwen3 remains at `vllm/model_executor/models/qwen3.py`, so the port straddles two
package conventions. Cosmetic, but it contradicts §7.1's suggestion that a `vllm/models/qwen3_msa/`
shape is the idiomatic target.

**Staged bring-up flag.** `MSA_SPARSE=0` builds the model with the indexer attached but sparsity off,
mirroring `DSA_SPARSE` (`scripts/dsa/vllm_minicpm3_dsa/model.py:65-69`). This is what makes Stage 0
debuggable independently of the sparse kernels.

### 4.3 Route A vs Route B

| | Route A — reuse M3's `forward` | Route B — override `forward` |
|---|---|---|
| fused CUDA kernel | used | not used |
| main q/k norms | must export `w − 1` | Qwen3's own, no conversion |
| RoPE | via the fused kernel at `rotary_dim=128` | vLLM's `rotary_emb`, standard path |
| KV / index cache writes | inside the fused kernel | we write them (index cache layout `[num_blocks, 128, head_dim]`) |
| risk | `rotary_dim=128` unexercised | more launches/layer; more of our own code in the hot path |

**DECIDED 2026-08-04: Route A.** P1 ran the fused op at `rotary_dim = 128` on Qwen3 shapes against a
torch reference and every surface agreed to ≤ 1.7 bf16 ULP of peak magnitude, with V bit-exact
(`tests/msa/test_qwen3_msa_fused_op_parity.py`; results in §9 P1). So the export applies the `w − 1`
shift to main `q_norm`/`k_norm`, and §4.2's attention subclass overrides RoPE construction only.

---

## 5. Export path

`scripts/msa/build_msa_serving_dir.py`, generalising the DSA pair
(`scripts/dsa/consolidate_indexer_ckpt.py`, which reconstructs DTensor shards with no distributed
context and takes `--key-substr ""` for a full base+indexer consolidation, and
`scripts/dsa/build_vllm_serving_dir.py`).

### 5.1 Weights

1. Consolidate the 8 FSDP2 rank shards → one `{name: bf16 cpu tensor}` dict.
2. **Rename** `self_attn.indexer.index_{q,k}_{proj,norm}` → `self_attn.index_{q,k}_{proj,norm}` (§2.2).
3. **The `w − 1` shift** (Route A, decided in §4.3): `q_norm.weight -= 1` and `k_norm.weight -= 1`,
   so the kernel's `1 + w'` reproduces Qwen3's `w`.

   **Scope is narrow and getting it wrong is silent.** Main `q_norm`/`k_norm` only, and only on the
   33 sparse layers (3–35). **Not** the index norms — already Gemma (§2.3). **Not** layers 0–2 —
   they build stock `Qwen3Attention`, never touch the fused kernel, and would break if shifted.
   Assert applied exactly once; double-applying is also silent.

   **It is lossy, by ≤ 1 bf16 ULP.** Measured on k8 @ step 1400 over all 66 full tensors: main q/k
   gains have mean **+1.718**, range **−1.008 … +44.0**, with **9.3%** of elements below 1.0 — versus
   the index norms at mean **+0.146**, range −0.125 … +0.500. That gap *is* the two conventions,
   visible in the data. For `|w| < 1` the shift moves the value into a *coarser* binade (at
   `w = −0.247`, ULP goes 0.00098 → 0.00781), so the round trip is not exact. Worst case:
   **3.799e-3**, at `model.layers.4.self_attn.k_norm.weight` — under 1 ULP at `|w| ≈ 1` (0.00391),
   i.e. inside the quantisation the bf16 weights already carry. Storing fp32 is not an option: the
   kernel's host check requires norm-weight dtype to match `qkv`. So the export **measures**
   `max |(1 + bf16(w−1)) − w|` per tensor and fails above `--max-shift-err` (default 2^-8) rather
   than assuming exactness.

   > **Trap, hit twice.** These norms are `Shard(0)`-placed across the 8 ranks, so rank 0 holds 16 of
   > 128 elements. Calling `to_local()` on one rank silently yields a 1/8 slice, and stats computed
   > that way are wrong but plausible-looking (it reported mean 0.9998 and range −0.247…+1.99).
   > Always reconstruct via `_full_from_shards` before measuring anything.
4. Write `model.safetensors`.

Assertions that must fail loudly: 531 tensors in, 531 accounted for; 132 `index*` tensors present;
no key still containing `.indexer.`.

### 5.2 `config.json`

```jsonc
{
  "architectures": ["Qwen3MSAForCausalLM"],
  "rope_theta": 5000000,          // documentary only — tf5's Qwen3Config DROPS this on load, so the
  "partial_rotary_factor": 1.0,   // plugin re-sets both as live attributes (§4.2). Qwen3 is
                                  // full-rotary => rotary_dim = 128
  "sparse_attention_config": {
    "sparse_attention_freq":   [0,0,0, 1,1, /* … 33 ones … */ 1],
    "sparse_topk_blocks":      8,      // from msa_top_k
    "sparse_block_size":       128,    // from msa_block_size
    "sparse_num_index_heads":  8,      // == num_key_value_heads
    "sparse_index_dim":        128,
    "sparse_init_block":       0,      // msa_init_blocks
    "sparse_local_block":      1,      // msa_local_blocks
    "sparse_score_type":       "max"
  }
}
```

Source for every key: `nvidia/model.py:434, 513-530` and `_sparse_attention_layer_ids` at `:95-103`.
The builder must refuse to write a directory whose `sparse_attention_freq` is missing, is not
`num_hidden_layers` long, or does not match `msa_dense_prefix` / `msa_sparse_layers` — that check is
the difference between a sparse model and a fluent, benchmark-passing, entirely dense one.

Round-trip test: reload the written dir and diff every tensor against the training module's
parameters, undoing the `w − 1` shift.

---

## 6. Bring-up ladder

Adapted from `tests/dsa/test_stage{0,1,2a,2b,3}*.py`. Each DSA stage caught a distinct failure class;
the mapping is not 1:1 because MSA skips the restructuring DSA needed.

| DSA stage | MSA stage | gate |
|---|---|---|
| 0 — plugin loads, strict weight load, coherent text | **S0** same | catches the §2.2 rename and the §5.2 config gate |
| 1 — MLA-latent == dense materialised QKV | **dropped** | no attention restructuring |
| — | **S1** fused op at `rotary_dim = 128` vs a torch reference | open item 3; decides Route A/B |
| 2a — dense MLA end-to-end | **dropped** | out of scope, §6.4 |
| 2b — short-prompt degeneracy + HF first-token parity | **S2b** same | see §6.3 |
| 3 — teacher-forced per-position decode parity | **S3** same | see §6.5 |

### 6.1 S0 — plugin load

`Qwen3MSAForCausalLM` registers, all 531 tensors load under the strict loader with no missing or
unexpected keys, short prompt generates coherent text with `MSA_SPARSE=0`. Run as a subprocess so a
sparse-path failure cannot contaminate the result (DSA's pattern).

### 6.2 S1 — fused-op admissibility

Run `fused_minimax_m3_qknorm_rope_kv_insert` on Qwen3-shaped input at `rotary_dim=128` and diff
against a torch reference implementing Gemma QK-norm → NeoX RoPE → cache insert, in that order.
Also confirm the Triton top-k compiles at `k ∈ {8, 16}` (open item 9 concerns `k = 256`, which under
§6.4 we no longer need). Also measure the `G = 4` **decode** head-axis pad — `BLOCK_SIZE_H = max(16,
next_pow2(G))` (`common/ops/sparse_attn.py:227`), so our 4 real heads occupy 16 slots, a 4× waste on
that axis. Decode only: the prefill kernel uses plain `next_power_of_2(gqa_group_size)` with no floor
(`:46`). Performance, not correctness — M3 at `G = 16` never sees it.

### 6.3 S2b — first-token correctness  ✅ done, see §9 P4

**Degeneracy.** On prompts shorter than `k · B_k` (1024 tokens at k8, 2048 at k16) the indexer can
only select valid blocks, so every key is visible and the sparse output **must** equal dense
attention over the same weights. Cheap, and it exercises the real sparse kernel rather than a
degenerate config. Report greedy token match plus max |Δ| over shared first-step top-K logprobs.

**HF first-token parity.** On a prompt longer than `k · B_k`, compare vLLM's first-step logits
against the training forward on the same tokens. This is the load-bearing one: it proves the served
function equals the trained function on a support where selection is actually doing something.

### 6.4 Scope — no de-confounding ladder

**Decided 2026-08-04: this project evaluates exactly one config** — the Phase-2b weights served
sparse with the trained indexer. No `StockRef` dense reference, no stock-base + Phase-1-indexer row,
no randomised-indexer ablation, no `Δ` decomposition into drift vs. sparsity.

Consequence, recorded so it is not rediscovered later: any comparison against the dense baseline in
§1 is a **combined** effect of weight drift and sparsity, and cannot be attributed to either. The
ladder in [../qwen3_4b_dsa/eval_plan.md](../qwen3_4b_dsa/eval_plan.md) §3 is not being run.

What is **not** dropped is §6.6. Those are not ladder controls — they are the checks that establish
the model is sparse at all, they cost a log grep and a startup assert rather than an eval run, and
without them a benchmark number is uninterpretable in the specific way DSA already hit once
(`docs/dsa_eval_report.md` §2: a serving dir silently served dense and posted good scores).

### 6.5 S3 — decode parity, and why a prefill-only oracle suffices  ✅ done, see §9 P4

Teacher-forced per-position agreement, not free-running sequence equality — one bf16 flip cascades,
so free-run divergence is expected and is a secondary report only (DSA Stage 3's finding).

The training module has no decode path (§1), but **teacher forcing is a single full-sequence prefill**:
greedy-generate from the vLLM sparse server, then push `prompt + generation` through the training
forward in one pass and compare per position. Only free-running HF generation would need an index-K
cache, and we never need that. Use prompts longer than `k · B_k` so selection is real: prose, code,
math, and at least one ≥ 16K to exercise the length regime the evals run at.

### 6.6 The anti-dense gate — wire before any benchmark

From [plan.md](plan.md) §7.3. Checks 1 and 2 are mandatory and free — a log grep and a startup
assert, no eval run, no extra code beyond the assert itself. Check 3 is optional under §6.4's scope
since it needs a `randomize_indexer_ckpt.py` port.

1. **Log grep** for both `info_once` lines — `"MiniMax M3 sparse attention selected %s
   (kv_cache_dtype=%s, topk_blocks=%s)"` (`common/sparse_attention.py:462-467`) and `"MiniMax M3
   indexer: selected Triton (no fmha_sm100) [topk_blocks=…, indexer_kv_dtype=…, sm100=…]"`
   (`common/indexer.py:516-522`). Absent ⇒ dense. On SM90 the first must report `Triton`, not `MSA`
   (the `MSA` variant at `:504-510` is the fmha_sm100 path).
2. **The index side caches must be allocated — count them, do not count groups.**
   ~~Two KV-cache groups must exist.~~ **Corrected 2026-08-05 (P3):** vLLM 0.26 merges everything
   into a *single* `UniformTypeKVCacheSpecs` group, so [plan.md](plan.md) §7.3's two-group
   expectation is simply false here and a "1 group" reading is not evidence of anything. The right
   invariant is the cached-layer roster: the indexer registers its key-only cache under
   `f"{layer}.attn.index_cache"` (`common/indexer.py:390`), so a correct sparse model shows
   **36 main + 33 `.index_cache` = 69** cached layers. Measured exactly that. This is the stronger
   check anyway — it counts index branches directly instead of inferring them from grouping.
3. *(optional)* **Random-indexer positive control** — same weights, randomised indexer, outputs must
   change. The strongest of the three, and the only one that proves selection is *load-bearing*
   rather than merely *running*. Costs an MSA port of `scripts/dsa/randomize_indexer_ckpt.py`.

---

## 7. Launch configuration

Modelled on `serving/serve_dsa.sh`.

| flag / env | value | why |
|---|---|---|
| `--block-size` | **128** | mandatory — both backends report `[128]` only |
| `--tensor-parallel-size` | **1** | moots `o_proj(reduce_results=False)`; 4B needs no sharding |
| `--max-model-len` | 131072 | matches the dense baseline's server, which the eval harness pins against |
| `--served-model-name` | `Qwen3-4B-Thinking-2507` | so the existing OpenCompass configs run unchanged |
| `--gpu-memory-utilization` | re-tune | two cache groups at 128K ≠ DSA's 8192-token single group |
| `--dtype` | bfloat16 | |
| indexer cache dtype | **pin bf16 explicitly** | the side cache accepts `bf16` or `fp8_e4m3` (`common/indexer.py:132-149`), and fp8 would quantise exactly the keys the selector scores — the DSA UE8M0 drift class. On SM90 this is **loud, not silent**: the Triton indexer impl raises `NotImplementedError` for any non-bf16 dtype (`common/indexer.py:511-515`), so we cannot fall into it by accident. Pin it anyway, and assert it from the `info_once` line, so the guarantee survives a move to SM100 |
| `--enforce-eager` | **open** | DSA needed it whole-model. MSA declares `AttentionCGSupport.UNIFORM_BATCH` (`common/sparse_attention.py:205`, `common/indexer.py:225`) with a targeted eager break — see risk R2 |
| `PYTHONPATH` | `_pluginboot` **first** | see below |

**Plugin boot.** vLLM spawns EngineCore as a subprocess that does **not** inherit the parent's
imports on the async server path. Importing the plugin in the entry script alone is not sufficient.
The proven fix is a `sitecustomize.py` in a `_pluginboot/` dir placed first on `PYTHONPATH`, which
every interpreter auto-imports
(`/cb/ml-eng/aarti/dsa/evals/minicpm3-4B-dsa-k128/serving/_pluginboot/sitecustomize.py`), plus an
entry script that imports the plugin and then `runpy.run_module("vllm.entrypoints.openai.api_server")`.

**Manifest.** Copy `serve_dsa.sh`'s block verbatim: timestamp, hostname, verl git sha + dirty count,
model dir, `PYTHONPATH`, sparse env flags, port / mem-util / max-len, and the full command. Same
discipline as the training runs.

---

## 8. Risks

**R1 — The eval-side vLLM version confound. [new, not in any prior doc]**
The dense baseline was measured on vLLM 0.20.2; the sparse model will serve on 0.26.0. Six minor
versions of kernel and sampler churn would sit inside every reported Δ. **Mitigation:** re-measure a
subset of row 0 (IFEval + RULER 8K, a few hours) on 0.26.0 and confirm it reproduces, *before* the
ladder. Doubles as the P0 end-to-end smoke test.

**R2 — Cudagraph capture is an unresolved, silent gate. [UNVERIFIED]**
`eager_break_during_capture`'s docstring says the decorated function "executes normally" outside a
capture context (`vllm/compilation/breakable_cudagraph.py:59-89`), and the mechanism is gated on
`VLLM_USE_BREAKABLE_CUDAGRAPH` (`:52-53`). What happens when that is off while the backend advertises
`UNIFORM_BATCH` is not established. Because the failure would be wrong numbers rather than a crash,
resolve it empirically: run S2b/S3 with capture on and off and diff.

**R3 — Throughput is unmeasured, and the eval schedule depends on it. [UNVERIFIED]**
Five kernel launches per sparse layer per decode step plus an eager break could plausibly run
*slower* than dense FlashAttention at eval batch sizes. The baseline's own costs (AIME25 ≈ 4–6 h,
MMLU-Pro ≈ 4 h per GPU) were measured dense. **Measure tokens/s immediately after the first
successful serve, before committing to any benchmark schedule.** This is [plan.md](plan.md) item 11
promoted from a schedule note to a gate.

**R4 — Silent failure is the dominant mode. CONFIRMED, and it generalised past the code.** All four
predicted silent failures were real (§9 P2/P3), and three *tests* also failed silently — passing
while measuring nothing, or gating below the achievable floor (§9 P4). Assume any new green result
is wrong until the test has been shown capable of failing.

Original note: §2.2 (rename), §2.3 (norm shift), §5.2 (config gate),
§4.2 (buffer shape) all fail without an exception, and all produce a fluent model. This argues for
building the strict loader and the "531 in, 531 consumed" assertion *first*, not last.

**R5 — Memory at 128K with two cache groups.** DSA served at 8192. Unknown whether
`--max-model-len 131072` fits at a usable `gpu-memory-utilization` alongside the index side cache.
Cheap to find out; may force a lower served length than the baseline, which would itself be a
confound.

---

## 9. Order of work

**~5–6 days.** P1 runs first because it picks the implementation route; P2 and P3 are independent of
each other and can overlap. **Nothing here depends on training finishing** — build and debug against
**k8 @ step 1400**, then re-export when the runs land.

| # | phase | gate that ends it | est. |
|--:|---|---|--:|
| [P0](#p0--environment--done-2026-08-04) | environment | **✅ DONE** — venv up, compiled fused op present | 0.5 d |
| [P1](#p1--kernel-spike--done-2026-08-04--route-a) | kernel spike | **✅ DONE** — **Route A** | 0.5 d |
| [P2](#p2--export--done-2026-08-05) | export | **✅ DONE** — `k8_step1400` built + verified | 1 d |
| [P3](#p3--plugin--done-2026-08-05) | plugin | **✅ DONE** — S0 passes, dense + sparse | 1–1.5 d |
| [P4](#p4--parity--done-2026-08-05) | parity | **✅ DONE** — S2b, S3, kernel parity all pass | 1.5–2 d |
| [P5](#p5--throughput) | throughput | tokens/s known; R2, R5 resolved | 0.5 d |

### P0 — environment ✅ DONE 2026-08-04

Env lives at **`.devlibs/vllm026`** (gitignored, alongside `tf457lib` and the existing `vllm-src`
checkout). Note `python3 -m venv` fails on this host — `ensurepip` is unavailable and there is no
sudo — so it was created with **`uv`** (`uv venv --python 3.12`; CPython 3.12.6). 7.5 GB installed.

Acceptance run: `.devlibs/vllm026/bin/python tests/msa/test_vllm026_env.py` → **P0 PASS**

| check | result |
|---|---|
| vLLM / torch | 0.26.0 / 2.11.0+cu130, H100 sm_90 |
| all 5 `minimax_m3` modules + stock `qwen3` import | PASS |
| `MiniMaxM3SparseAttention` / `MiniMaxM3Indexer` / `MiniMaxM3SparseBackend` / `Qwen3ForCausalLM` | PASS |
| **compiled `fused_minimax_m3_qknorm_rope_kv_insert` in the wheel** | **PASS** — the P0 unknown, resolved |
| `get_supported_kernel_block_sizes()` on both backends | `[128]` — serve with `--block-size 128` |

**Not done, deliberately:** re-serving stock Qwen3 to close **R1** (the 0.20.2 → 0.26.0 baseline
confound). Under §6.4's narrowed scope we are not running paired comparisons, so this is only worth
doing if a dense reference re-enters scope. R1 stays open and unmitigated — see §8.

### P1 — kernel spike ✅ DONE 2026-08-04 → **Route A**

`.devlibs/vllm026/bin/python tests/msa/test_qwen3_msa_fused_op_parity.py` → **S1 PASS**

Fused op at `rotary_dim = 128` on Qwen3 shapes vs a torch reference. Gate is max|Δ| relative to each
tensor's **peak** magnitude, in bf16 ULP:

| surface | max\|Δ\|/peak | | surface | max\|Δ\|/peak |
|---|--:|---|---|--:|
| `q_out` | 1.68 ULP | | `index_k` | 1.04 ULP |
| `index_q_out` | 0.76 ULP | | `kv_cache` K / V | 1.70 ULP / **bit-exact** |
| `k` (in-place) | 1.70 ULP | | `index_cache` | 1.04 ULP |
| `v` (untouched) | **bit-exact** | | | |

`.devlibs/vllm026/bin/python tests/msa/test_qwen3_msa_topk_compile.py` → **PASS**

| topk | `BLOCK_SIZE_T` | autotune configs surviving | compiles |
|--:|--:|--:|---|
| 8 | 8 | 6/6 | ✅ |
| 16 | 16 | 6/6 | ✅ |
| 256 | 256 | 3/6 | ✅ |

**Open item 9 is closed, and more broadly than expected.** k=256 compiles too: Triton's autotuner
*prunes* configs failing `tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)` rather than erroring, so
three surviving configs are enough. A served dense-equivalence control at k=256 is therefore
available if scope ever re-expands — `plan.md` §4.2 #1's warning does not bite.

`G = 4` head-axis pad: decode `BLOCK_SIZE_H = max(16, next_pow2(4)) = 16`, a 4× waste on that axis;
prefill uses `next_pow2(4) = 4` with no pad. Performance only, quantified end-to-end in P5.

**Two methodology notes**, recorded because both produced a false FAIL before being caught:
1. The first reference had an **aliasing bug** — `x1`/`x2` sliced from the *output* buffer, so the
   first RoPE assignment clobbered `x1` before the second line read it. Fixed by reading from the
   input and writing to a clone.
2. Two tolerance schemes were wrong before the third was right. A flat absolute tolerance is
   meaningless without the data scale (1 ULP near 4.0 is already 3.1e-2). Per-element ULP is worse:
   RoPE's `x1·c − x2·s` **cancels**, so a near-zero output carries absolute error inherited from its
   O(1) inputs and reads as thousands of ULP. Peak-relative is the meaningful gate. Reuse it in P4.

### P2 — export ✅ DONE 2026-08-05

`scripts/msa/build_msa_serving_dir.py`. Reads the per-rank DTensor shards directly, reusing the
verified `_full_from_shards` from `scripts/dsa/consolidate_indexer_ckpt.py`, so there is no ~9 GB
intermediate `.pt`.

First artifact: **`/cb/ml-eng/aarti/msa/serving/k8_step1400`** (9.02 GB), built from
`…_k8_…/global_step_1400`. ~2.5 min end to end.

| | |
|---|---|
| tensors | 531 in → 531 out, 132 `index_*`, **0** keys still containing `.indexer.` |
| rename | 132 indexer keys lifted to `self_attn.index_{q,k}_{proj,norm}` |
| `w − 1` shift | 66 tensors (main q/k norms, layers 3–35); worst round-trip error **3.799e-3** ≤ 2^-8 |
| layer 3 `q_norm` | exported `0.1562, 0.2891, 0.1094, −1.25` (was `1.1562, 1.2891, 1.1094, −0.2471`) |
| layer 0 `q_norm` | **unshifted**, mean 1.787 — dense prefix, never sees the kernel |
| config | `freq` len 36 sum 33, `topk=8`, `block=128`, `idx_heads=8`, `local=1`, `init=0`, flat `rope_theta=5e6`, `partial_rotary_factor=1.0` |

`config.json` also records `msa_norm_shift_applied: true` so a re-export cannot double-shift silently.

**Exit criteria met**, verified independently of the builder's own assertions: safetensors keys ==
source, spot-checked tensors round-trip bit-identically, all finite, un-shift recovers all 66
originals, index keys sit directly under `self_attn`, and the 3 dense layers carry no index weights.

Re-run for k16 or a later step with `--ckpt-dir <other global_step_N> --out <dir>`; `--no-norm-shift`
switches to Route B.

### P3 — plugin ✅ DONE 2026-08-05

`scripts/msa/vllm_qwen3_msa/` — **two files, ~290 lines, no `attention.py`, no monkeypatching.**
S0 (`tests/msa/test_stage0_plugin_load.py`) passes both sub-goals:

| | dense (`MSA_SPARSE=0`) | sparse (`MSA_SPARSE=1`) |
|---|---|---|
| attention classes | 36 × `Qwen3Attention` | 3 × `Qwen3Attention` + **33 × `MiniMaxM3SparseAttention`** |
| cached layers | 36 | **69** = 36 main + **33 `.index_cache`** |
| backend log lines | — | both present (`Triton`, as expected on SM90) |
| generation | coherent | coherent, identical text |

**Four bugs found by running it, none visible to inspection:**

1. **`config.rope_theta` does not exist.** transformers 5's `Qwen3Config` folds rope settings into
   `rope_parameters` and drops the flat key, so §5.2's flat `rope_theta` in `config.json` is *not*
   what makes this work — HF strips it on load. `MiniMaxM3SparseAttention` reads it flat
   (`nvidia/model.py:468-475`), which suits `MiniMaxM3Config` and not ours. Fixed by
   `_ensure_m3_rope_fields()`, which sets `rope_theta` and `partial_rotary_factor` on the live
   config before the layers are built. **This corrects §4.2's claim that the class needs no shim.**
2. **§6.6's "two KV-cache groups" is the wrong invariant** — see the correction in §6.6.
3. **`MSA_SPARSE=0` could not load at all**: dense layers have no `index_*` parameters, so a strict
   load of a checkpoint holding 132 of them must raise. `load_weights` now drops them in that mode
   (DSA instead attached an indexer post-hoc, `vllm_minicpm3_dsa/model.py:229-265`).
4. **The dense mode emitted fluent garbage** — the §2.3 `w − 1` trap, exactly as predicted for any
   dense path on a Route-A export: `Qwen3Attention` computes `x·w` on norms stored as `w − 1`. That
   mode now un-shifts on load, which is what makes it a usable diagnostic instead of noise.

**Methodology note.** S0's first run reported PASS while emitting `" the the big red of the big
1000000…"` — the gate only checked that the child process exited. It now asserts a coherent
continuation *and* the index-side-cache count. A gate that cannot fail is not a gate.

**Serve scaffolding done and tested.** `scripts/msa/serving/{serve_msa.sh, serve_msa_entry.py,
_pluginboot/sitecustomize.py}`, driven by `tests/msa/test_serve_smoke.sh`. This covers what S0
cannot: S0 uses an in-process `LLM()`, whereas the real server spawns EngineCore as a subprocess
that does not inherit the parent's imports. Result:

| check | |
|---|---|
| server reaches `Application startup complete` | PASS |
| sitecustomize registered the plugin in **3** interpreters (parent + EngineCore + worker) | PASS |
| both sparse backend log lines | PASS |
| `/v1/completions` over HTTP returns `" Paris. The capital of Germany is Berlin…"` | PASS |

Served as `Qwen3-4B-Thinking-2507` so the existing OpenCompass configs at
`/cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking` run against it unchanged. Launch flags per §7,
including `--block-size 128` and `--enforce-eager` (R2 still unresolved). The manifest block
records git sha + dirty count, venv/vLLM version, model dir, `PYTHONPATH`, `MSA_SPARSE` and the
full command, as the DSA runs did.

### P4 — parity ✅ DONE 2026-08-05

**Everything is gated against a floor measured in the same run, never against zero.** Two
independent floors exist on this setup and both were measured, not assumed:

| floor | value | source |
|---|--:|---|
| `w − 1` round trip, on first-step logprobs | **6.8e-2** | two DENSE runs differing only by the shift (`test_stage2b_degeneracy.py`) |
| kernel-vs-torch block-score noise | **0.141** | `test_qwen3_msa_index_parity.py`, recomputed per run |
| decoding determinism (control) | **0.000** | same config twice — proves the first floor is weights, not variance |

| gate | result |
|---|---|
| **S2b degeneracy** (`test_stage2b_degeneracy.py`) | **PASS.** Short prompt (5 tok, all blocks selected): sparse-vs-dense = 0.94× floor, greedy 8/8. Long prompt (1627 tok): **5.31× floor** — selection is load-bearing, so the test can detect a difference |
| **Index-score parity** vs `minimax_m3_index_score` | **PASS.** max\|Δ\| 0.141 on a scale of 64.2 (0.22%), 540k live blocks |
| **Selected-block SET parity** vs `minimax_m3_index_topk` — [plan.md](plan.md) open item 1, *"must-do before training"* | **PASS.** mean overlap **0.9966**, worst-query 0.7778 over 32,768 (head,query) pairs; 1.53% disagree, and **100% of those are near-ties** (max gap 0.082 vs 0.141 noise) |
| **HF parity** (vLLM vs the training forward) | **PASS.** top-1 matches; teacher-forced mean \|Δ\| **2.8e-2** vs 7e-2 floor; 1.3% of 1626 positions above 3× floor |
| **S3 decode parity** (`test_stage3_decode_parity.py`) | **PASS.** Decode kernels agree *better* than prefill on identical tokens — prose 4.7e-3, code 1.7e-2, math 4.0e-2 |
| **Anti-dense gate** (§6.6 checks 1–2) | **PASS** in S0 — both backend log lines, 33 `.index_cache` layers |

**Bit-exact selection is unreachable, and that is settled by measurement, not opinion.** The two
implementations reduce over 128 dims in different orders and float addition is not associative.
Matching serving's bf16 dtype in the reference made agreement *worse* (0.9923 vs 0.9965) — it is
accumulation order, not precision. Training's fp32 scores are deliberate (there is an existing
assertion, and they feed the KL's `log_softmax`), so downgrading them would harm training *and*
agreement. The gate is therefore **"every flip is a near-tie inside the measured noise"**, which is
a real check: DSA's UE8M0 drift produced flips with large gaps and would fail it.

**Cross-environment oracle.** The training forward must run in the TRAINING env (transformers
5.3.0, `ray`), not the venv (5.14.1, no `ray`) — it mirrors `Qwen3Attention.forward` for a specific
transformers version. `scripts/msa/hf_msa_oracle.py` runs there and exchanges tensors on disk. It
needs the `--no-norm-shift` export (exact weights, standard RMSNorm); feeding it the shifted export
would compute `x*(w−1)`.

**Four methodology failures worth not repeating.** Every one produced a confident wrong answer:

1. **A test that could not fail.** The set-parity fixture had 8 blocks and `SEL_K=8` — top-8-of-8
   selects everything, so 1.000000 overlap was guaranteed. Reported as a headline result before the
   error was caught. The file's own constant block warns about exactly this. Fixture now asserts
   `n_blocks > SEL_K` and that <50% of queries select all blocks.
2. **Two thresholds set below the achievable floor.** An absolute `|Δlogprob| ≤ 0.05` "failed" a
   correct configuration at 6.4e-2 (floor is 6.8e-2); `mean overlap > 0.999` "failed" at 0.9966.
   Both would have condemned working code.
3. **A statistic with no resolution.** S3 gated an outlier *rate* at 5% on 48 samples, where one
   outlier is 2.1 points. Fixed by raising to 256 and gating against prefill on the same sequence.
4. **A test that measured the wrong thing.** S3 first captured "decode" logprobs by re-scoring the
   finished sequence as a prompt — a PREFILL pass. It reported math at mean 0.52 / 21% outliers and
   nearly triggered a hunt for a decode bug that does not exist (real values: 0.040 / 5.9%).

**Unexplained, recorded for completeness:** vLLM prefill re-scoring the math continuation disagrees
with the oracle far more than the decode kernels do on the same tokens (5.2e-1 vs 4.0e-2). Inside
S2b's already-passing envelope and not a decode issue, so not chased.

### P5 — throughput

Tokens/s at eval-shaped batches and lengths. Resolve **R2** (capture on vs off, diff the outputs) and
**R5** (does 131072 fit at a usable `gpu-memory-utilization` with two cache groups).

**Exit:** a measured cost model for the eval ladder — or the finding that sparse is slower than dense
at our batch sizes, which reshapes the schedule rather than the code.

### Running in parallel, independent of all of the above

The offline indexer probes — `topk_recall` / `topk_overlap` swept over length (4K/8K/16K/32K), query
position, key distance, per layer (mean **and** min), on held-out long docs and held-out thinking
traces. These need only the training forward, can start today against k8 @ step 1400, and are the
only measurement that explains *why* a benchmark moved. Cheapest information per GPU-hour available.

---

## 10. Follow-ups outside this doc

- [plan.md](plan.md) §7 needs the three §2 corrections inline, and §10's items 5 and 6 re-scoped.
- [plan.md](plan.md) §7.3's "two kv-cache groups must exist" is **wrong for vLLM 0.26** — it merges
  into one group. Replace with the cached-layer roster check (§6.6 check 2).

- The **decode split-K index path** is still unwired (`test_parity_decode_path_separately`,
  explicitly skipped). S3 covers decode end-to-end and passes, but the kernel-level comparison for
  that specific path does not exist.
- The **prefill-vs-decode gap on the math continuation** (§9 P4, 5.2e-1 vs 4.0e-2) is unexplained.

Deliberately **not** follow-ups, per §6.4: the [../qwen3_4b_dsa/eval_plan.md](../qwen3_4b_dsa/eval_plan.md)
§3 ladder, a `StockRef` dense reference, and `scripts/msa/randomize_indexer_ckpt.py`. Each is cheap to
add later; none is on this path.

---

## 11. Training-side bug found by parity: partial-block aliasing

Found because the serving path disagreed with training, and traced back to **training** being wrong.

**Mechanism.** `_selected_token_index` expanded block ids to token positions as
`block * B_k + offset`, then `clamp_max(seq_len - 1)`. When the final block is only partly filled
(`seq_len % 128 != 0`) its surplus slots were pinned onto `seq_len - 1`, aliasing onto the last real
token. `slot_ok` tested only "did this come from a selected block", which is true for them.

Every query except the one at `seq_len - 1` is saved by the causal mask — the alias is in its
future. That last query is not, so it attended to the final token `128 - (seq_len % 128)` times
over. At `seq_len = 5` that is 124 of 128 slots (~97% of the attention mass) and the position
degenerates into echoing its own last token: it answered `' is'` to *"The capital of France is"*,
and gave the same answer for a different prompt.

**Impact on training: negligible, which is why it survived.** The corrupted position is the final
one, whose prediction has no next-token target and is dropped by the shifted loss mask
(`workers/utils/losses.py:148`). It reaches only the KL, at ~1 row in 7,000 (BC rows are p50 ~7K
tokens, and `micro_batch_size_per_gpu=1` means the tensor width *is* the sample length, so there is
no padding to absorb it — that setting is the worst case; with `mbs>1` only the longest row of each
micro-batch is affected). Phase 1 is unaffected (dense path). The serving path was never affected —
vLLM pages by sequence length and never clamps.

**Fix.** Range-test before the clamp, because clamping is what destroys the evidence:

```python
raw = sel.clamp_min(0).unsqueeze(-1) * bk + off          # BEFORE clamping
tok = raw.reshape(b, h, tq, k * bk).clamp_max(seq_len - 1)
slot_ok = ((sel >= 0).unsqueeze(-1) & (raw < seq_len)).reshape(b, h, tq, k * bk)
```

**Why the existing suite missed it.** `test_qwen3_msa_phase2.py` D1 asserts dense equivalence and
would have caught this — at `--seq-len 512`, and `512 % 128 == 0`. Block-aligned is the one case
with zero surplus slots. Coverage added:

- `tests/msa/test_selected_token_index.py` (new, CPU, <1s) — sweeps every `seq_len` in 1..3·B_k for
  B_k ∈ {8,128}; **verified non-vacuous** by re-running against the pre-fix code, where 4/11 fail;
- **D1** extended to `[512, 500, 385, 300]` with `n_blocks` corrected to **ceil** (`T // bk`
  under-counts a partial block and would set `k` too low) and an explicit last-position assertion;
- **D7** (new) — `mbs > 1`: padded batch `[300, 411, 512]`, per-row dense equivalence.

Full suite after the fix: phase2 28/28 · unit 11/11 · phase1 24/24 · monkey-patch 17/17 ·
index-parity 16 passed / 2 skipped.

### 11.1 Training restart

**The Phase-2b runs were stopped and will be restarted on the fixed code** (decision 2026-08-05).
The bug did not materially damage the stopped runs — no LM gradient ever touched the corrupted
position — so this is a cleanliness call, not a recovery. Consequences:

- The existing `k8 @ step 1400` and `k16 @ step 800` checkpoints remain **valid for serving-stack
  work**; every artifact in P0–P4 was built from k8 @ 1400 and none of it needs redoing.
- Restarted runs produce **bitwise different** trajectories from step 0. Do not compare loss curves
  across the boundary.
- Re-export (`build_msa_serving_dir.py`) against the new checkpoints when they land; P2 takes ~2.5
  min and P4 re-runs unchanged.
