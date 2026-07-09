# Plan: Training the indexer — Phase 1 dense warm-up (wiring + pipeline)

Makes Phase-1 indexer training **runnable end-to-end** on top of what's implemented. This is Part C (loss +
engine + freeze) + Part D (data + config + launch) of `docs/dsa_minicpm3_plan.md`. Do **not** implement
until reviewed.

## What's already done (Parts A/B)
- `LightningIndexer`/`DSAConfig` (`dsa_indexer.py`) — FP8-numeric scores, tested.
- `minicpm_dsa.py` — monkey-patch attaches the indexer per layer, the patched `dense_warmup` forward
  computes the per-layer KL, and hooks sum it into **`model._dsa_indexer_kl`** (+ diagnostics dict). Tested.
- `monkey_patch.py` `minicpm3` branch (gated `config.dsa_enabled`) + `flops_counter` entry.
- Env: transformers **4.57.1 + `get_usable_length` shim** (applied by the patch), `fast_hadamard_transform`
  built. See `docs/dsa_minicpm3_plan.md` Step-0 + memory `minicpm3-transformers5-incompat`.

So the **loss already exists as a tensor on the model** (`model._dsa_indexer_kl`); training = surface it as
the loss, freeze the base, feed packed long data, and run verl's SFT loop.

## Data flow (loss → wandb)
`model(**inputs)` (hooks set `model._dsa_indexer_kl` + `model._dsa_metrics` on the module) → the
**loss closure** `indexer_kl_loss(model_output, data, dp_group, model=…)` reads those off the module and
returns `(kl, metrics)` (called in `forward_step`, `transformer_impl.py:1275-1282`) → SFT trainer logs
`metrics` to wandb as `train/*` (`sft_trainer.py:391-406`). Stock `language_model` engine — no custom
engine.

## Work items

### 1. Surface the KL into the loss  (Part C1/C2) — **no custom engine required**
The forward hooks already set `model._dsa_indexer_kl`/`_dsa_metrics`. Two ways to get them into the loss;
**start with (1a)**:
- **(1a) Loss closure (default, simplest).** Bind the model into the loss fn — it reads the KL off the
  module directly, using the stock `language_model` engine unchanged:
  `self.loss_fn = lambda **kw: indexer_kl_loss(**kw, model=self.engine.module)` (in
  `sft_trainer._build_engine`, replacing the `sft_loss` partial at `:163`). By the time the loss runs in
  `forward_step` (after `self.module(**inputs)`, `transformer_impl.py:1275-1282`) the hook has populated the
  attribute. (Slightly breaks verl's "engine produces `model_output`, loss consumes it" separation, but is
  functionally fine.) Note: access through the FSDP wrapper (`engine.module._dsa_indexer_kl` forwards via
  FSDP `__getattr__`).
- **(1b) log-probs skip — a shared-engine FLAG, not a phase-specific engine (OPTIONAL, Phase-1 memory).**
  A *"DSA engine that removes log_probs"* is **Phase-1-only by nature and does NOT generalize**: Phase 2's
  distillation loss (`forward_kl_topk(student_logits, teacher_topk)`) *needs* the LM-head logits, so Phase 2
  must **keep** them. So skipping is **conditional**, best expressed as a config flag
  (`calculate_log_probs=false`) that `prepare_model_outputs` honors — flipped **on in Phase 1** (skip the
  `[T, 73448]` ≈ 5 GB logits→log_probs block the loss ignores) and **off in Phase 2** — rather than a
  separate engine subclass. Deferrable memory optimization, not correctness; for short-seq bring-up just let
  the frozen LM head compute-and-be-ignored.

**Both phases run on the stock `language_model` engine.** Phase 2 needs **no** engine change at all: the
standard engine already computes student logits + runs the existing `distillation_use_topk` logits-processor
(`transformer_impl.py:1221-1228`); `dsa_distill_loss` (closure) combines that distillation term
(`model_output`) with `indexer_kl` (module). So there is **no custom DSA engine** in the design — only the
optional Phase-1 `calculate_log_probs=false` flag above.

### 2. Loss fn + selection  (Part C3/C4)
- Add `indexer_kl_loss(config, model_output, data, dp_group=None)` to `verl/workers/utils/losses.py`:
  returns `(model_output["indexer_kl"], {"indexer/kl": kl.detach(), **model_output["indexer_metrics"]})`.
  **No LM CE.**
- Add `loss_mode` to the SFT config; branch at `sft_trainer.py:163` (currently hard-codes `sft_loss`) to
  pick `indexer_kl_loss` when `loss_mode==indexer_kl`.

### 3. Freeze base, train only the indexer  (reuse the LoRA pattern)
- In the model-build path (after `apply_monkey_patch` at `transformer_impl.py:292`, **before** the FSDP wrap
  at `:340`): set `requires_grad=False` on all params, `True` on `*.indexer.*`.
- **Memory falls out for free:** with the base frozen, autograd retains no base activation graph (only the
  small indexer path needs grad) → **no gradient checkpointing needed** in Phase 1 (and grad-ckpt would
  anyway conflict with the per-layer `_dsa_kl` side-effect state — so keep it **off**).
- FSDP: set `engine.use_orig_params=true` (FSDP1) or FSDP2 so mixed `requires_grad` in a FlatParameter is
  legal (`transformer_impl.py:402`) — same requirement the `_is_lora` path (`:141`, `:307`) already relies on.
- Optimizer (`_build_optimizer`, `transformer_impl.py:451`): filter `module.parameters()` to
  `requires_grad` (indexer only), LR `1e-3`, constant + short warmup.

### 4. Monitoring  (Part C7 — see main plan)
Extend `_dense_warmup_kl` to also record per-layer `topk_recall`/`topk_overlap`/score-stats into
`attn._dsa_diag` (gated by a `diag_interval`); aggregate in the KL post-hook into `model._dsa_metrics`
(`Metric(AggregationType.MIN/MAX/…)`); the loss returns them → wandb `train/indexer/*`. Early-stop when
`indexer/kl` plateaus and `indexer/topk_recall` saturates.

### 5. Packed long-context dataset  (Part D)
New `verl/utils/dataset/packed_pretrain_dataset.py` (registered via `data.custom_cls.path/name`, honored by
`create_sft_dataset`, `sft_trainer.py:464-471`): raw text → **MiniCPM3 tokenizer** → pack to a fixed length
with **`position_ids` reset to 0 per document**. Emits `input_ids` + `position_ids` (no chat template; no
`loss_mask` needed — indexer KL uses all positions via its own causal+doc mask). Source: `openbmb/
InfLLM-V2-data-5B` (use a small shard for bring-up).

### 6. Doc-boundary base attention (varlen) — **the key correctness item**
The built `dense_warmup` forward mirrors the *stock* MiniCPM flash forward, which applies a **full causal
mask** — correct for **one document per sequence**, but for **packed multi-doc** it would attend **across
doc boundaries**, producing contaminated hidden states (the indexer would then learn to mimic garbage).
The indexer *target* `p` is already doc-masked (`_build_causal_doc_bias`), but the **base attention output**
is not. Two options:
- **(6a) Bring-up:** one doc per sequence (or short fixed length), no packing — validates the whole loop
  and indexer learning without varlen.
- **(6b) Real packing:** extend the forward's base attention to **varlen flash** using `cu_seqlens` derived
  from `position_ids==0` (pattern: `qwen2_vl.py:164-179 prepare_fa2_from_position_ids` + `flash_attn_varlen_func`;
  also verl's generic `_ulysses_flash_attention_forward`). Then base + target masks are consistent. Required
  for genuine 32K packed warm-up.

### 7. Config + launch + env
- `verl/trainer/config/sft_trainer_minicpm_dsa_phase1.yaml`: model=MiniCPM3-4B (`trust_remote_code`, flash,
  bf16), `model.dsa_enabled=true`, `model.dsa_overrides={n_heads:16, head_dim:64, rope_head_dim:32,
  top_k:2048, mode:dense_warmup, kl_block_size:1024, fp8:true}`, `engine.strategy=fsdp,
  use_orig_params=true`, `optim.lr=1e-3` (constant+warmup), `data.custom_cls=…packed…`,
  `data.max_length` (start small; 32K target), `loss_mode=indexer_kl`, `trainer.total_training_steps` (~2B
  tokens), `trainer.logger=[console,wandb]`, `project/experiment`, `save_freq`.
- `examples/dsa/run_minicpm3_dsa_phase1.sh`: launcher (`PYTHONPATH=/tmp/fht_clean:/tmp/tf457lib` in dev;
  bake transformers 4.57.1 + fast-hadamard into the training image for real runs).

### 8. Checkpoint the indexer
Phase-1 output = the warmed-up indexer. verl saves the FSDP model (indexer params are submodules → included).
For Phase 2, load this checkpoint's indexer. Consider saving an **indexer-only** shard (small, ~50–60 MB) in
addition, to make Phase-2 init clean.

## Critical decisions / risks
- **Packed vs one-doc (item 6)** — the top correctness item; start 6a, then 6b for 32K.
- **Skip LM head (item 1)** — mandatory at 32K (logits tensor ~5 GB) — call the backbone, not the CausalLM.
- **Grad checkpointing OFF** in Phase 1 (unneeded with frozen base; conflicts with `_dsa_kl` side-effect
  state under recompute).
- **Freezing in FSDP** — needs `use_orig_params=true`/FSDP2 (reuse LoRA path).
- **no_padding vs padded** — the built forward is the dense `[bsz,T]` path; confirm it composes with the SFT
  trainer's `pad_mode` (start with padded/fixed-length; the rmpad/nested path is a later optimization).
- **SP>1** deferred (indexer needs global keys post all-to-all — mirror kimi_vl).

## Staged bring-up
1. **Overfit one batch** (tiny model, short single-doc seq): KL must drop → ~0 and `topk_recall` → ~1.
   Proves the loss trains the indexer and grads reach `wq_b`/`wk` only.
2. **Freeze check:** only `*.indexer.*` have `requires_grad`; base bit-identical after a step; FSDP wraps
   under `use_orig_params`.
3. **Real MiniCPM3-4B, short single-doc seqs** (item 6a), small InfLLM shard: KL decreasing, memory stable,
   throughput/MFU sane, wandb curves live.
4. **Packed 32K varlen** (item 6b): confirm zero cross-doc leakage (base + target), then scale to ~2B tokens.

## Files
**Create:** `verl/utils/dataset/packed_pretrain_dataset.py`;
`verl/trainer/config/sft_trainer_minicpm_dsa_phase1.yaml`; `examples/dsa/run_minicpm3_dsa_phase1.sh`.
**Modify:** `verl/workers/utils/losses.py` (+`indexer_kl_loss`, +`dsa_distill_loss`);
`verl/trainer/sft_trainer.py` (`loss_mode` select + bind the loss closure to `self.engine.module`); the FSDP
model-build + `_build_optimizer` (freeze base / indexer-only param filter); `minicpm_dsa.py` (item 4
diagnostics; item 6b varlen). **No custom engine** — only an *optional* `calculate_log_probs=false` flag in
the shared `prepare_model_outputs` for the Phase-1 memory skip.

## Verification
- **Overfit-a-batch** (the key correctness test): a few hundred steps on one fixed batch drives
  `indexer/kl`→~0 and `indexer/topk_recall`→~1 (indexer can represent the dense attention).
- **Freeze/optimizer test:** optimizer param count == indexer param count; base unchanged after a step.
- **Skip-lm-head:** peak memory at 32K well below the +5 GB the logits tensor would add.
- **Packed masking (6b):** attention/target have zero cross-document mass (extend the Part-B mask test).
- **End-to-end smoke:** short run on a real InfLLM shard → decreasing `train/indexer/kl`, rising
  `train/indexer/topk_recall`, stable memory/throughput, wandb logging.

---

# Training & Validation Runs

The sections above are the *wiring* plan (mostly done). This section is the plan for the actual
indexer **training runs and held-out validation**. Decisions locked in: real run at **32K on 4×H100**;
validate on **held-out same-corpus (in-distribution) + a second OOD corpus**.

## Objective & success criteria

Train **only the lightning indexer** (base frozen) so its per-position key ranking matches the base
model's dense attention, and **prove it generalizes to unseen documents** (in-dist and OOD).

- **Primary:** `val/indexer/topk_recall` at the *deployment* sparsity ratio — fraction of true attention
  mass captured by the indexer's top-k keys on held-out docs. Predicts Phase-2 sparse quality. Target **≥ 0.9**.
- **Secondary:** `val/indexer/kl` (+ the train↔val gap), `val/indexer/topk_overlap`,
  `val/indexer/entropy_frac` (health — must not collapse), `val/indexer/nan_frac` ≈ 0.
- **Convergence:** `train/indexer/kl` ↓ and plateaus; `val/indexer/kl` tracks it with a small gap;
  OOD gap quantifies cross-domain generalization.

## Data — disjoint train / val / OOD-val

All windows are one-doc-per-row, retokenized with the MiniCPM3-4B tokenizer, filtered to ≥ `seq_len`,
truncated to exactly `seq_len` (item 6a shape).

1. **Train + in-dist val (disjoint, one pass):** extend `prepare_real_data.py` with `--val_out` /
   `--val_windows` — after collecting `--num_windows` train windows it keeps scanning and collects the
   next `--val_windows` into `--val_out`. Disjoint by construction (different source docs), same
   tokenizer/seq_len.
2. **OOD val (second corpus):** a different long-context distribution than InfLLM-V2 (which is web/mixed).
   Recommended: **PG19** (books) — clearly OOD, long docs, easy HF access. Needs generalizing the loader
   (the current shard-path pattern + `text` column is InfLLM-specific); add a `--repo`/`--path-scheme`
   path or a small second loader that yields the OOD `text` column, then the same clean/tokenize/truncate.

## Recipe — two phases

**Phase A — pipeline shakeout (cheap, 1 GPU, 4K).** Validate the full train+val loop before spending
long-context compute.
- `seq_len 4096`, 512 train / 128 in-dist val / 128 OOD val windows, ~200 steps, `test_freq` every 25 steps.
- **Pass criteria:** `train/indexer/kl` drops, the `val/…` panel logs (both val sets), recall rises,
  `nan_frac ≈ 0`, `entropy_frac` doesn't collapse. (Complements the existing overfit-a-batch test.)

**Phase B — real run (32K, 4×H100).** Memory notes confirm 32K Phase-1 fits on 4×H100 (~66 GB peak).
- `seq_len 32768`, `NPROC=4`, ~2048 train / 256 in-dist val / 256 OOD val windows.
- Longer schedule (≈3–4 epochs) with **LR warmup** (current script is constant `8e-3`; add warmup for the
  real run), `test_freq=after_each_epoch`, periodic `save_freq`, keep best by `val/indexer/topk_recall`.

## Validation methodology

- **Cadence:** `trainer.test_freq` (per-N-steps or `after_each_epoch`) drives the val loop, which now emits
  `val/loss` + the full `val/indexer/*` & `val/attn/*` panel (diagnostics are force-enabled in eval mode).
- **Two named val sets:** log in-dist and OOD separately (e.g. `val_indist/*`, `val_ood/*`). The current
  trainer supports a **single** `val_dataloader`; supporting two prefixed val sets is a work item
  (iterate a dict of named val loaders, prefix the logged keys). Interim fallback: run eval on one set at
  a time, or concatenate (loses the in-dist vs OOD split — not recommended for the real run).
- **Sparsity-honest recall:** `TOPK=2048` at 4K keeps 50% (not a real sparsity test); at 32K it is ~6%
  (meaningful). Log recall at **2–3 k values** to see the recall-vs-k curve (small diag addition).
- **Checkpoint selection:** track best `val_indist/indexer/topk_recall`; the indexer is tiny so keeping
  several checkpoints is cheap. Report both in-dist and OOD recall/KL for the chosen checkpoint.

## Remaining work items

1. ~~`prepare_real_data.py`: disjoint seeded in-dist val split.~~ **Done** — seeded shuffle + `--val_out`/
   `--val_windows`, reproducible (pinned revision + seed + `MANIFEST.json`).
2. ~~OOD data prep.~~ **Done** — `prepare_ood_data.py` builds the OOD val from **`openbmb/Ultra-FineWeb`**
   (`content`) at **multiple lengths** (one parquet per length). One-command build:
   `examples/dsa/build_phase_a_datasets.sh` (see `examples/dsa/README_datasets.md`). Shared helper:
   `examples/dsa/_dsa_data_utils.py`.
3. ~~Run-script val knobs.~~ **Done** — `run_minicpm3_dsa_phase1.sh` has `VAL_FILES` / `TEST_FREQ` /
   `VAL_MAX_SAMPLES`; sets `data.val_files`, `data.val_max_samples`, `trainer.test_freq` only when
   `VAL_FILES` is set (else forces `test_freq=-1` so the empty val loop can't crash). A single-val-set
   validated run is now launchable.
4. Trainer: support **multiple named val sets** in ONE run with logged prefixes (`val_indist/*`,
   `val_ood_L{len}/*`) — the OOD set is per-length, so this also drives the recall-vs-length curve.
   **Interim available:** `trainer.val_only` + `trainer.val_prefix` (see below) let you eval each set in a
   **separate** invocation now; item 4 is only needed to log them all in a single run.

**Val-only / eval-from-checkpoint (done).** `sft_trainer.py` has a reusable `validate()` plus a
`trainer.val_only` mode: load a checkpoint, run one val pass, log `<val_prefix>/loss` + `<val_prefix>/indexer/*`,
exit. Run OOD eval later on a trained run via the run script:
`VAL_ONLY=1 RESUME_PATH=<ckpt> SEQ_LEN=<L> VAL_FILES=<ood_L parquet> VAL_PREFIX=val_ood_L<L> run_minicpm3_dsa_phase1.sh`.
5. (Optional) recall@multiple-k in the diag block for the sparsity-vs-k curve.
6. (Optional) LR warmup schedule for Phase B.

Data (items 1–2) + run-script wiring (item 3) are done. Item 4 is the only thing left to log in-dist and
all OOD-length val sets in one run.

## Already done (val wiring)

- `indexer_kl_loss` normalizes by **valid (non-pad) query count** (`num_valid_queries`), not `loss_mask`
  (`losses.py`, `tensordict_utils.py`; `batch_num_valid_queries` all-reduced in `transformer_impl.py`,
  gated on `dsa_enabled`).
- Val loop emits `val/loss` **and** the `val/indexer/*` + `val/attn/*` panel (`sft_trainer.py`,
  `_scalarize_metric`); diagnostics are forced on in eval mode (`minicpm_dsa.py` KL pre-hook via
  `not model.training`).
