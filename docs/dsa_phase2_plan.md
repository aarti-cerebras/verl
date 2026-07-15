# DSA Phase 2 — sparse finetuning plan (MiniCPM3-4B → DSA)

Companion to `docs/dsa_minicpm3_plan.md` (original master plan; **Part E is superseded by this doc**),
`docs/dsa_indexer_worklog_2026-07-09.md` (Phase-1 state), and `docs/dsa_kl_loss_math.md`.

## Goal

Convert the **released instruct model `openbmb/MiniCPM3-4B`** (EN+ZH chat / reasoning / code / tool-use,
32K native context, MLA backbone) to **DeepSeek Sparse Attention (DSA)** — top-k sparse token selection per
query (**`top_k=512`**, see the top_k decision below) — **while preserving its capabilities**. We do **not**
have MiniCPM3-4B's pretraining corpus.

## Ground truth: what DeepSeek-V3.2 actually does (arxiv 2512.02556 §2.1.1)

DSA is added by **continued pre-training in two stages** — *no separate-teacher distillation in the
conversion* (distillation is a later, separate post-training step):

| Stage | Trainable | Indexer loss | Main-model loss | Budget |
|---|---|---|---|---|
| **Dense warm-up** | indexer only (base frozen) | `KL(p_{t,:} ‖ softmax(I_{t,:}))`, `p`=attn summed over heads, L1-norm | — | LR 1e-3, 2.1B tok |
| **Sparse** | **all params** | same KL but **restricted to selected set `S_t`** (eq. 4); **indexer input detached** | **next-token LM loss only** | LR 7.3e-6, 943.7B tok |

Key sentence (§2.1.1): *"we detach the indexer input from the computational graph … the training signal of
the indexer is from only `L^I`, while the optimization of the main model is according to only the language
modeling loss."* → the indexer and main model are **cleanly decoupled** in the sparse stage.

**Implication for the old plan:** `docs/dsa_minicpm3_plan.md` Part E (online co-located teacher +
`dsa_distill_loss` + `compute_forward_kl_topk` + vocab top-k logit plumbing) is **not what the paper does
and is not needed.** We drop it.

## Why we deviate from the paper's *data* (but not its objective structure)

The paper's main-model loss is plain LM on their **own 128K pretraining corpus**. We can't copy that, and
running plain LM on a generic web corpus (UltraFineWeb) against an **instruct** model would pull it back
toward a base-LM distribution → **catastrophic forgetting of the instruct capabilities we want to keep.**

So the main-model objective becomes **self-distillation (behavior cloning)**: teacher = frozen dense
MiniCPM3-4B, student = same weights + sparse attention. Objective = *"under top-k sparse attention,
reproduce what you produced under dense attention."* Well-posed because the student **starts identical** to
the teacher and only compensates for the (small) sparse-attention approximation at LR ~7.3e-6 → low drift,
low degeneration risk. Sequence-level form = **SFT on trajectories the dense model generated** (reuses the
existing `sft_loss`; no teacher resident at train time; no vocab-logit dump).

## The Phase-2 recipe (two decoupled objectives)

1. **Indexer** ← selected-set KL only: `KL(p_{t,S_t} ‖ softmax(I_{t,S_t}))`, `p` renormalized over the
   selected keys `S_t`. Indexer input **detached**. This is a light variant of the existing
   `_dense_warmup_kl` (`verl/models/transformers/minicpm_dsa.py:225`). Top-k selection is non-differentiable
   → no LM-loss gradient leaks into the indexer; detach → no indexer gradient leaks into the base.
2. **Main model** ← `sft_loss` (`verl/workers/utils/losses.py:76`, already built) = next-token CE on
   **self-generated dense trajectories**, in MiniCPM3's chat template.

`loss = sft_loss(response tokens) + λ · indexer_kl_on_selected_set`.

## top_k / sparsity-ratio decision (why 512)

`top_k` is DSA's per-query key budget. DeepSeek's **absolute** 2048 was tuned for **their 128K context**;
transplanting it to MiniCPM3-4B's **32K** gives 4× *gentler* sparsity than DeepSeek intended:

| Model | context | top_k | sparsity ratio |
|---|---|---|---|
| DeepSeek-V3.2 | 128K | 2048 | 1.6% |
| MiniCPM3-4B @ 2048 | 32K | 2048 | 6.25% |
| **MiniCPM3-4B @ 512** | 32K | 512 | **1.6%** ← matches DeepSeek's ratio |

So **`top_k=512` reproduces DSA's intended sparsity ratio at 32K** (512/32768 = 1.5625%); 2048 was the
un-rescaled DeepSeek number. Consequences that reshape this plan:

- **Sparsity activates at `seq_len > 512`** (real ≥50%-drop signal at `≥1024`), not 2048.
- **Ordinary self-generated trajectories clear the bar.** MiniCPM3-4B instruct responses (~200–540 tok) +
  prompt land most trajectories in 512–1024 → **no long-document sourcing needed** (the big unlock; see
  Datasets). This is why `top_k=2048` made every candidate dataset "too short" and `512` makes them usable.
- **Memory eases**: gather 512 keys (~4× cheaper attention), and self-gen trajectories are **short (~1–2K)**,
  so the 32K OOM analysis below does **not** bind the near-term run (it governs only the later genuine-32K tier).
- **Consistency rule:** train at the `top_k` you deploy; if they differ, go train-hard/deploy-easy
  (train ≤ deploy). Train-at-512 is the safe floor.
- **`top_k=512` is LOCKED** (train + deploy). Consequence (M2 probe): only ~34% of self-gen trajectories
  clear 512 and ~6.5% clear 1024, so the sparse signal leans on **Code (+the ≥1024 tail)**; short domains are
  preserved dense-for-free. Filter keeps total ≥512 (prefer ≥1024) → generate ~3× the usable target. Revisit
  only if the parity gate fails.

### Constraints (still true)

- **Trajectories must span the capability surface** (math, code, IF, chat, **EN + ZH**), or preservation is
  fake — we only preserve what the prompt set exercised.
- **Generate with DSA OFF (dense);** targets are the dense model's own outputs.
- **Filter trajectories to total ≥ `top_k` (≥512; ideally ≥1024)** so the sparse path is exercised. Sub-512
  aren't harmful (sparse==dense==target → ~0 gradient) but waste compute.
- **Token budget ≪ 943B** — a 4B model starting at the target → ~0.5–20B tokens; early-stop on parity.
- **Per-document `position_ids` reset** in packing (varlen `cu_seqlens`; same as Phase 1).

## Datasets — prompt sources for self-gen (measured)

We self-generate, so we need **prompts** (targets come from MiniCPM3-4B); gold responses are discarded. At
`top_k=512` the length bar is 512, which ordinary instruct data clears. Length distributions we measured
(MiniCPM3 tokenizer, over HF shards via pyarrow):

| Source | role | total ≥512 | total ≥1024 | EN/ZH | verdict |
|---|---|---|---|---|---|
| **openbmb/UltraData-SFT-2605** (`no_think`: Math/Code/IF/Knowledge/Chinese-general) | prompt backbone | ~half (median ~500) | ~15% | **EN+ZH** | **winner — on-distribution (MiniCPM family), domain+lang labeled, gated (access granted)** |
| openbmb/UltraData-Math-L3 (Conversation/Textbook/QA/Multi-Style) | (not used) | 65–94% | 33–45% | EN+ZH | dropped — `no_think/Math` already covers math; single-source keeps it simple |
| nebius/…-Infinity-Instruct-0625 | EN prompts | ~50–70% | 18% | EN | usable, EN-only |
| allenai/Dolci-Instruct-SFT (Olmo3) | prompts (domain-labeled) | high | 25% | multiling | usable; multilingual dilution; long tail is non-math |

Findings from profiling:
- **Responses are short across all instruct models** (Llama-3.1-8B resp median 542, ≥4096 = 0.1%; Olmo3 258;
  UltraData no_think/Code 216). A non-reasoning 4B won't generate long → **length must come from the
  trajectory clearing 512**, which at `top_k=512` it does. (At `top_k=2048` every candidate failed the ≥2048
  bar at ~0.1–2%; the profiling is what motivated dropping to 512.)
- **Foreign gold responses are NOT used** (Nebius=Llama, Dolci=Olmo/OpenThoughts, UltraData-SFT=MiniCPM5):
  using them would distill *that* model into MiniCPM3-4B — capability change, not preservation.
- **Pick (single source): `UltraData-SFT-2605/no_think` prompts** — capability-spanning (Math/Code/IF/
  Knowledge/Chinese-general), EN+ZH, on-distribution. Its `no_think/Math` config already supplies the long,
  sparsity-rich trajectories (measured p50≈971, 51% ≥1024), so `UltraData-Math-L3` is **not needed** as a
  second source. Discard gold responses; self-generate with MiniCPM3-4B. Per-domain `no_think` lengths
  (MiniCPM3 tok): Math p50 971 / p90 8366 / p99 15332 (51% total ≥1024); Code 214 / 676 / 8862 (15%); IF
  140 / 405 / 958 (1%); Knowledge 280 / 531 / 739 (1%); Chinese-general 312 / 549 / 1060 (3%) → **Math (+Code
  tail) is the sparsity workhorse; short domains preserve breadth (run ~dense at <1024).**
- **Unmeasured:** MiniCPM3-4B's *own* response-length distribution (proxied above). Confirm with a GPU
  generation probe on ~1–2k prompts before scaling.

## Engineering surface

Most scaffolding already exists (Phase 1). New work is small and localized.

### T1 — Sparse attention forward (`minicpm_dsa.py`, replaces the stub at `:476-477`)

Recommended mechanism: **tiled gathered-KV** (autograd-native, no custom kernel; swap a fused DSA kernel in
later for throughput). In `mode == "sparse"`, tiled over query blocks (mirror `_dense_warmup_kl`'s tiling):

```
I_blk   = indexer.scores(..., attn_bias=causal_doc_bias)   # [b, B, T]   (reuse existing path)
idx     = I_blk.topk(min(top_k, T), dim=-1).indices        # [b, B, k]   (within-doc, causal via bias; detached)
Kg, Vg  = gather(key_states, idx), gather(value_states, idx)  # [b, H, B, k, d]
s       = einsum(qb, Kg) * softmax_scale + causal_doc_bias_gathered  # [b, H, B, k]
a       = softmax(s)
o_blk   = einsum(a, Vg)                                     # [b, H, B, d_v]   → assemble attn_output
```

- Uses the **same** `query_states`/`key_states`/`value_states`/`cos`/`sin`/`position_ids`/`attention_mask`
  already captured in `minicpm3_dsa_attn_forward`.
- When `k >= T` the gathered set is the full sequence → **must equal the dense flash path** (parity test).
- Grad flows into the base through `q/k/v` (the LM loss trains it); the `idx` gather is index-only (no grad),
  so selection is a stop-gradient, exactly as intended.
- Activation-checkpoint per tile at 32K (reuse the `kl_checkpoint` pattern).

### T2 — Selected-set indexer KL (`minicpm_dsa.py`)

Add a `_sparse_indexer_kl` (or branch inside `_dense_warmup_kl`): compute `p_blk` and `I_blk` as today, but
restrict the KL to `S_t` — build a mask that keeps only the top-k selected keys per query, renormalize
`p_blk` over `S_t` (`p / p[S_t].sum()`), and take `KL(p_S ‖ softmax(I_S))`. **Detach the indexer input**
(`hidden_states`/`qr`) for this branch so its gradient doesn't reach the base. Feeds the same
`self._dsa_kl` accumulator → `install_kl_accumulation` (`:125`) and metrics are unchanged. `topk_recall`
diagnostic already measures exactly the selected-set attention mass — keep it as the parity gate.

### T3 — Loss + trainer wiring

- New `dsa_sparse_loss(config, model_output, data, model=...)` in `verl/workers/utils/losses.py`:
  `sft_loss(...) + λ · model._dsa_indexer_kl` (read the KL off the model like `indexer_kl_loss` does at
  `:54`; `λ` from config). Returns merged metrics (`indexer/*` + LM loss).
- Add `loss_mode == "dsa_sparse"` branch in `sft_trainer.py:178` (next to `sft` / `indexer_kl`), binding the
  loss to `self.engine.module`.
- Main model now needs real `log_probs` (unlike Phase 1's KL-only short-circuit), so the sparse forward must
  produce LM logits — it does (the gathered-KV attn output flows through the unchanged LM head).

### T4 — Unfreeze base + two param groups

- `monkey_patch.py:513-516` already gates the freeze on `mode == "dense_warmup"` → sparse mode leaves the
  base trainable. Verify no path re-freezes.
- Two param groups: base LR ~7.3e-6, indexer LR ~1e-3, grad-clip 1.0. Check `build_optimizer`
  (`verl/workers/config/optimizer.py`) supports per-group LR (filter by `.indexer.` name).
- FSDP: base now trains → full-shard + activation checkpointing; `use_orig_params=True` (or FSDP2) still
  required for mixed grad. Indexer stays its own `fully_shard` unit (Phase-1 sharding notes hold).

### T5 — Trajectory generation + sample selection (offline)

**Generate.** `scripts/dsa/gen_trajectories.py`: serve the **stock released MiniCPM3-4B** (DSA off = vanilla
dense; no fork code) via **vLLM** (HF `generate` fallback if vLLM lacks MiniCPM3-MLA on the pinned version),
MiniCPM3 chat template. **Decoding: temperature 0.7** (top_p 0.9, `n=1` per prompt, fixed `seed`) — clones
MiniCPM3-4B's deployment-temperature behavior (sampled, not greedy). **Per-domain `max_new_tokens`**:
**Math / Code / Multi-lang-Math = 16K** (kept high — "cap high, filter after": don't bake truncation into the
data; future-proof for long-input prompts), **all other domains = 4K**.
**Stop = EOS + repetition guard** so degenerate samples die early. Drop `finish_reason==length` (the ~1.6%
runaways that hit 16K) + repetition in post-processing. Use **data-parallel generation (8× TP=1)** — TP=8 on
a 4B model is comms-bound; DP removes the runaway-straggler penalty.

**`MANIFEST.json` (reproducibility) logs:** model snapshot hash; **`temperature=0.7`, `top_p`, `seed`, `n`**;
per-domain `max_new_tokens` + stop config; source dataset revision; prompt-selection seed; filter thresholds;
per-domain × bucket counts/yield. (Under sampling the exact tokens depend on seed + vLLM version + hardware —
log all three; the `seed` makes a given (prompt, seed, engine) reproducible.)

**Prompts — single source: `UltraData-SFT-2605/no_think`.** Extract the **user turns** from its configs
(Math/Code/IF/Knowledge/Chinese-general; optional Multi-lang-{Math,Knowledge}); **discard the gold responses**
(MiniCPM5's). No second corpus — the `no_think/Math` config already supplies the long, sparsity-rich math
trajectories (measured p50≈971, 51% ≥1024). Pre-filter: dedup, decontaminate vs eval sets
(GSM8K/MATH/HumanEval/MBPP/C-Eval/CMMLU/MMLU), EN+ZH, drop oversized/degenerate prompts (the IF `max=323K`
artifacts). Hold out a val slice for the student↔dense parity eval.

**Choose prompts (pre-generation selection).** Subset the huge `no_think` corpus (millions of prompts) to the
M3a budget (~100–300k) — **clean → domain-stratified select → calibrate from probe**:
- *Clean:* dedup (exact + near-dup MinHash — synthetic rewrites give ~identical greedy outputs, wasted
  compute), decontaminate vs eval sets, EN+ZH filter, drop oversized/degenerate prompts (IF `max=323K` class).
- *Stratify by config × language* (M2-probe yield + natural-proportion audit): target mix —
  **Code(EN) 30 / Math(EN) 30 / Multi-lang-Math(ZH) 10 / Multi-lang-Knowledge(ZH) 10 / Knowledge(EN) 10 /
  Chinese-general(ZH) 5 / IF(EN) 5** → ≈ **75% EN / 25% ZH** (spec: `examples/dsa/m3a_split.json`). **Code is the sparsity workhorse** (probe: 74%
  ≥512, 18.6% ≥1024) — NOT Math (median short; MiniCPM3-4B ≠ long MiniCPM5 math). **ZH is sourced from
  `Multi-lang-*` (ZH-filtered), not `Chinese-general`** — natural proportions: the corpus is 61% Math / 18%
  Code by tokens, `Chinese-general` is only 1.7%, and ZH/multilingual lives in `Multi-lang-{Math,Knowledge}`
  (15%). `Multi-lang-*` is multilingual → keep only ZH (MiniCPM3-4B is EN+ZH; other langs = noise). Gaps: no
  ZH-code / ZH-IF configs exist. The kept-≥512 mix skews further toward Code (fine — short domains preserved
  dense-for-free). Within a config: random for M3a; **cluster-balanced** for scale.
  *(Needs `select_prompts.py`: per-config target counts + a per-config language filter — currently uniform
  `--per-domain`, lang only tagged. ZH share is a target; verify against Multi-lang ZH-yield.)*
- *Why math-weighting is safe for preservation:* short-domain behavior (mostly <1024) is preserved **for
  free** (sparse ≡ dense there), so few short trajectories are needed; the long-context behavior we must
  actively preserve is concentrated in math/code — so one weighting serves both breadth and sparse signal.
- *Calibrate weights from the M2 probe:* the weights that hit a target length-bucket mix depend on
  **MiniCPM3-4B's own** per-domain lengths (likely shorter than the MiniCPM5 reference) → set them after the
  probe, then iterate on realized ≥1024 yield.
- *Deliberately NOT:* per-prompt length prediction (signal too weak to pay for) or correctness/difficulty
  filtering (would bias toward an *upgrade*, not preservation). `source`/`domain` labels are fair for
  stratification only.

**Select (post-gen).** Keep trajectories with **total ≥ `top_k` (512), prefer ≥1024**; drop degeneration
(repetition loops, truncated-at-cap, non-EOS) — but **NOT by correctness** (cloning the model's *actual*
behavior, mistakes included, is preservation; correctness-filtering would be an *upgrade*). Mix to
domain/lang/length/budget targets; report histograms + yield.

**Write + load — reuse `MultiTurnSFTDataset`, NOT `PackedPretrainDataset`.** `PackedPretrainDataset` is
Phase-1 pretraining-only (one-doc-per-row, docs ≥ seq_len). For SFT:
- Write **raw `messages`** parquet `[{role:user,content:prompt},{role:assistant,content:response}]` — *not*
  tokenized/padded (compact + retokenizable; `max_length` stays a train-time knob).
- **Each row carries a back-reference to the origin sample:** `source_uid` (UltraData-SFT-2605's native
  `uid`, e.g. `Code_no_think_0000001`) + `source_dataset`/`source_config`/`domain`, and **`prompt_sha256`**
  (sha256 of the prompt text — stable, source-agnostic; doubles as the dedup key). Plus `lang`,
  `realized_len`, `length_bucket`, `finish_reason`, and the `seed` used. So any trajectory maps back to its
  exact original row (via `source_uid`) and is content-verifiable (via `prompt_sha256`).
- Apply the **≥512 filter before writing** (the dataset only pads, it won't re-filter).
- Train via the default **`MultiTurnSFTDataset`** (`sft_trainer.py:553`): chat-templates, sets a
  **response-only `loss_mask`** (`:235-239`), `position_ids=arange` (single doc, `:362`), and **pads to
  `max_length` at load** in `__getitem__` under `pad_mode="right"` (`:367-378`) — padding is a *runtime* cost,
  not in the parquet.
- Config: **`pad_mode="right"`, `use_remove_padding=False`** — the `[bsz, T]` layout the DSA forward expects.
- The DSA forward + KL consume this directly: single-doc `position_ids` → correct causal mask; `attention_mask`
  → `_dense_warmup_kl` excludes pad tokens (already implemented). No new dataset code for M3a.

### T6 — Config + launch script

- `verl/trainer/config/sft_trainer_minicpm_dsa_phase2.yaml` (from Phase-1 yaml + `dsa_mode=sparse`,
  `loss_mode=dsa_sparse`, `indexer_kl_lambda`, two LRs, `RESUME_PATH`=Phase-1 indexer ckpt).
- `examples/dsa/run_minicpm3_dsa_phase2.sh` (mirror `run_minicpm3_dsa_phase1.sh`).

### T7 — Tests

- `test_minicpm_dsa_sparse.py`: (a) sparse forward with `k>=T` **≡** dense flash output (parity);
  (b) grad flows to **both** base and indexer, and NOT from LM loss into indexer / not from indexer KL into
  base (detach check); (c) selected-set KL numerics vs a reference; (d) `topk` respects causal + doc mask.
- Extend `test_minicpm_dsa_overfit.py` with a tiny sparse overfit (loss ↓, recall ↑).

### T8 — Ulysses SP for the DSA path (unlocks full-32K training)

Prereq for M3b (sustained full-base training at 32K). verl already does Ulysses SP for **dense MLA**
attention, but the DSA sparse forward + indexer + selected-set KL are new surface SP doesn't cover.

- **Layout.** Ulysses hands the attention module a **head-parallel, full-sequence** layout
  (`[full seq, H/sp heads]`) via an all-to-all; the current DSA code runs *before* the flash transpose
  assuming **all heads + full sequence local** (SP=1). Make the sparse forward operate in the post-all-to-all
  layout.
- **MQA key + per-head weights.** The indexer key is single-head (MQA) and `weights` are per query-head,
  shared across the sequence — they must be available to every rank's head subset under the head split
  (all-gather the key/weights across the SP group, or replicate).
- **Selection.** Top-k + gather-KV run over the **full** sequence — fine post-all-to-all (full seq present).
- **Grad/loss.** SP peers contribute partial grads for the same params (different token slices); FSDP
  reduce-scatters over the whole world and the loss normalization (`× dp_size`, `÷ num_valid_queries`, already
  in `indexer_kl_loss`/`sft_loss`) reconciles sum-over-SP vs mean-over-DP. Verify the selected-set KL
  normalization holds when queries are sequence-sharded.
- Tests: SP=2 sparse forward ≡ SP=1 (numerical parity); grad equivalence.

### T9 — Variable-length efficiency (deferred, not an M3a blocker)

`pad_mode="right"` wastes compute on padding (~50–65% for short trajectories). The efficient paths —
`no_padding`/`use_remove_padding` (rmpad) **and** genuine multi-doc packing — both require the DSA forward to
handle the varlen `[1, total_nnz]` + `cu_seqlens` + per-doc `position_ids` layout (the KL side already
supports per-doc masking via `position_ids == 0`; the LM flash path and the score/gather tiling need the
varlen wiring). One deferred efficiency task; at M3a's ~1–2K sequences the padding waste is tolerable.

## Memory (only bites at genuine 32K — not the near-term top_k=512 run)

**Scope:** with `top_k=512` + self-gen trajectories (~1–2K tokens), training sequences are short → full-base
training **fits easily on 4×H100 and the OOM analysis below does not apply to M3a.** This section governs the
**later genuine-32K long-context tier** (M3b), where sequences are long.

MiniCPM3-4B: 62 layers, 40 heads, hidden 2560, vocab 73448, ~4B params, 32K ctx (torch_dtype bf16).

**Why Phase 1 fit but naive Phase 2 won't.** Phase 1 froze the base → model state was ~2 GB/rank (bf16 params
only, no grads/optim) and log_probs were short-circuited; the measured 66–69 GB/rank peak on 4×H100 was almost
entirely **per-sequence activations + the tiled KL score graphs** at 32K. Phase 2 unfreezes the full base and
adds, per rank: full AdamW state + grads (`~16·P/N` bytes), and the LM head + CE (Phase 1 skipped it). Naive
full-32K ≈ 66 + ~14 (optim/grad) + ~8 (unfused logits) + backward overhead ≈ **90+ GB → OOM on 80 GB**.
(Phase-2 tailwind: the indexer KL is now over the **selected set (k=512)**, not full T=32768 → those score
buffers shrink ~64× vs Phase 1.)

**FSDP shards model state; it does NOT shard activations.** So *more data-parallel GPUs do not help the
bottleneck.* 4→8 GPUs (pure DP, no SP) only halves the ~16 GB model-state term (÷world_size); each rank still
holds a full 32K sequence, so the dominant activation term is unchanged → still OOM.

**SP × FSDP are orthogonal axes** (two device meshes, `transformer_impl.py:217,223`): FSDP shards
params/grads/optim across the **whole world** (÷world_size); Ulysses SP factors the world into `dp_size × sp`
and splits **one sample's sequence** across the `sp` ranks (activations ÷ sp), with an all-to-all swapping
`[seq/sp, H] ↔ [full seq, H/sp]` for attention. Enabling SP gives the activation reduction *on top of* full
model-state sharding (SP does **not** reduce FSDP's shard factor).

| Config (8×H100, ~4B, micro-bsz 1) | FSDP shard | Activation factor | Model state /rank | Activations /rank |
|---|---|---|---|---|
| 8 GPU, **SP=1** (pure DP) | ÷8 | ÷1 | ~8 GB | full 32K ← **bottleneck** |
| 8 GPU, **SP=2** (dp=4) | ÷8 | ÷2 | ~8 GB | half |
| 8 GPU, **SP=4** (dp=2) | ÷8 | ÷4 | ~8 GB | quarter ← **target for full-32K** |

**Mitigation ladder (cheap → structural):**
1. **Fused CE + `use_remove_padding=True`** — Phase-1 script has `use_remove_padding=False`, no fused CE;
   fused linear-CE never materializes `[32768, 73448]` logits → LM-head cost ~8 GB → ~1 GB. Config-only, always.
2. **Optimizer-state CPU offload** (`offload_policy`) — moves ~12 GB/rank of fp32 AdamW state to host; speed cost.
3. **Curriculum / shorter train context** — sparsity activates above `top_k` (512), so even short sequences
   exercise DSA; activation ~linear in seq (16K ≈ ½, 8K ≈ ¼ of 32K). Train the bulk short/at 8–16K, reserve
   32K for eval / a short final slice. Cheapest big win. (With top_k=512 self-gen trajectories this is automatic.)
4. **Activation CPU offload** (`ACT_OFFLOAD=True`, existing knob) — offloads saved activations to host.
5. **Ulysses SP (T8)** — the structural fix; **8 GPU + SP=4** makes full-32K comfortable (model state ÷8,
   activations ÷4). Requires the DSA-specific SP work.

**Recommended stacks:** near-term (no SP) → fused CE (1) + optimizer offload (2) + train at 8–16K (3), 32K
eval-only; sustained full-32K → **8 GPU + SP=4** (5) + fused CE (1). Estimates are unvalidated — measure a
single fwd+bwd peak before committing (see M3a).

## Milestones

- [ ] **M0 — sparse forward correctness (short ctx).** Implement T1+T2 behind `mode=sparse`; parity test
  (`k>=T` ≡ dense) + grad/detach tests (T7 a,b,d) on CPU/1-GPU at 4K. *No training yet.*
- [ ] **M1 — end-to-end shakeout (4K, Phase-A data).** T3+T4 + a tiny self-gen trajectory set; run a few
  steps: assert loss decreases, `indexer/topk_recall` rises, base+indexer both update, no NaNs.
- [ ] **M2 — trajectory pipeline (T5).** Generate the self-gen corpus from `UltraData-SFT-2605/no_think`
  prompts (all configs) with dense MiniCPM3-4B; filter total ≥512; decontaminate. (Optional: GPU probe of
  MiniCPM3-4B's own per-domain response-length distribution first, to confirm the caps + ≥512 yield.)
- [ ] **M3a — full-base sparse run, `top_k=512`, self-gen medium trajectories.** SFT (T3) with base+indexer
  (T4) from the Phase-1 indexer ckpt, on the ≥512 trajectories. Sequences are ~1–2K → **fits on 4×H100
  without SP or memory gymnastics.** Monitor parity (student↔dense agreement, `topk_recall`, KL); early-stop
  on the ~1% gate. Validates the full recipe on abundant data.
- [ ] **M3b — genuine-32K long-context tier (optional).** Only if we need to serve/preserve 32K contexts:
  source long docs, apply the memory ladder + Ulysses SP for DSA (T8) on **8 GPU + SP=4**. Separate from M3a;
  pursue only if the parity gate at long context demands it.
- [ ] **M4 (if parity gate fails) — token-level self-distillation upgrade.** Replace/augment the SFT term
  with KL to the dense model's per-token distribution (reintroduces a teacher-logit source; stronger signal).

## Decisions locked / open

- **Locked:** `top_k=512` — train + deploy (matches DeepSeek's 1.6% ratio at 32K); self-gen
  (behavior-cloning) main-model objective; decoupled selected-set indexer KL; no online teacher /
  no `dsa_distill_loss`; tiled gathered-KV sparse forward first. **Single prompt source:**
  `UltraData-SFT-2605/no_think` (all configs; `no_think/Math` supplies the long math — `UltraData-Math-L3`
  dropped). Self-gen at **temperature 0.7**, per-domain `max_new_tokens` (math/code/ML-Math 16K, rest 4K),
  data-parallel gen (DP=8). With top_k=512 + medium trajectories the M3a run fits on 4×H100 **without SP** —
  the 32K memory/SP work moves to the optional M3b tier.
- **Open:** trajectory loss mask (response-only vs full-seq); `λ` (indexer-KL weight); token budget;
  whether the genuine-32K tier (M3b) is pursued at all; whether M4 (token-level KD) is needed.
