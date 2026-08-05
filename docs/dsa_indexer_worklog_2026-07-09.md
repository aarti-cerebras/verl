# DSA indexer — train + validate work session (2026-07-09)

Durable summary of the session: what changed, why, the decisions, the datasets, and the exact commands to
reproduce. Companion to `docs/dsa_train_indexer_plan.md` (plan) and `docs/dsa_kl_loss_math.md` (loss math).

## Goal

Get DSA Phase-1 (dense warm-up) indexer training + held-out validation launch-ready on MiniCPM3-4B, with
reproducible datasets (in-distribution + OOD) and a proper LR schedule.

## Code changes

**Loss normalization — normalize by valid (non-pad) query count, not `loss_mask`.**
- `verl/utils/tensordict_utils.py::num_valid_queries` — real-token count (nested offsets / `attention_mask`).
- `verl/workers/engine/fsdp/transformer_impl.py` — all-reduces it into `batch_num_valid_queries`, **gated on
  `dsa_enabled`** (no extra collective for non-DSA runs).
- `verl/workers/utils/losses.py::indexer_kl_loss` — weights each micro-batch by `mb_valid / batch_num_valid_queries * dp_size`.
- Why: the KL is meaned over ALL non-pad query rows (`total_cnt`), which ≠ `loss_mask` when prompts are
  masked. Coincided before only because Phase-1 `loss_mask` is all-ones. Test updated:
  `tests/workers/test_indexer_kl_loss_on_cpu.py` (4 pass).

**Validation — emit the full indexer panel on held-out data.**
- `verl/trainer/sft_trainer.py`: hoisted `_scalarize_metric`; extracted a reusable `validate()`; the val loop
  logs `val/loss` + `val/indexer/*` + `val/attn/*` (recall/overlap/entropy).
- `verl/models/transformers/minicpm_dsa.py` KL pre-hook: force diagnostics ON in eval mode
  (`not model.training`) so every held-out batch gets recall/overlap.

**Val-only / eval-from-checkpoint.**
- `trainer.val_only` short-circuit in `fit()` (load ckpt → one val pass → exit) + `trainer.val_prefix`
  (label metrics, e.g. `val_ood_L2048`). Lets OOD eval run later, per invocation.

**Run script (`examples/dsa/run_minicpm3_dsa_phase1.sh`).**
- Val knobs: `VAL_FILES` / `TEST_FREQ` / `VAL_MAX_SAMPLES` / `VAL_PREFIX`.
- Val-only knobs: `VAL_ONLY` / `RESUME_PATH` (forces `TRAIN_FILES=VAL_FILES`, `SAVE_FREQ=-1`).
- LR schedule: `LR_SCHED=cosine` (warmup→peak→cosine decay), `LR=1e-3` (peak), `WARMUP_RATIO=0.03`,
  `MIN_LR_RATIO=0.1`. (Cosine already supported by the FSDP engine; config-only.)

## Decisions

- **Peak LR 1e-3, cosine warmup+decay** (was 8e-3 constant). 1e-3 = DeepSeek-V3.2 indexer warm-up reference;
  loss is mean-normalized so LR is batch-independent.
- **Context:** Phase A = 4K (shakeout), Phase B = 32K on 4×H100 (real).
- **Val sets:** in-distribution (held-out InfLLM) + OOD (**Ultra-FineWeb**, quality-filtered, multiple lengths).
- **Eval:** one val set per invocation (val-only mode); multi-named-val-in-one-run (plan item 4) skipped.
- **Token budget:** indexer warm-up does NOT scale with base params; ~0.3–1B is plenty for a 4B indexer,
  2.1B (DeepSeek) is an upper bound. Stop on val-recall plateau.

## Datasets — reproducible build

Scripts: `examples/dsa/{prepare_real_data.py, prepare_ood_data.py, _dsa_data_utils.py,
build_phase_a_datasets.sh}`; see `examples/dsa/README_datasets.md`. Pinned HF revision + seed + MANIFEST.json
=> byte-identical.

**Built (2026-07-09), in `data/dsa/phase_a/`:**
- `infllm_minicpm3_4096_train.parquet` — 2048 win × 4096 = 8.39M tok (InfLLM `deeb03b5…`, seed 1234)
- `infllm_minicpm3_4096_val.parquet` — 256 win (in-dist, disjoint)
- `ood_ultrafineweb_minicpm3_L{1024,2048,4096}.parquet` — 128 win each (Ultra-FineWeb `7ddd4170…`,
  min_score 0.9; quality filter dropped 17561/25262 = 69.5%; all buckets filled)

**In progress (task at write time):** 100M-token train set — `infllm_minicpm3_4096_100M_train.parquet`
(24,414 win ≈ 100M tok) + `_100M_val.parquet` (512 win). Log: `data/dsa/phase_a/build_infllm_100M.log`.
Note: ~0.64% yield (most InfLLM docs <4096 tok) → scans ~3.8M docs; may fall short if corpus lacks that many.

## Reproduce / run

```bash
# 1. Build Phase-A datasets (needs .devlibs/tf457lib env + HF network)
examples/dsa/build_phase_a_datasets.sh              # small set; pin REV_INFLLM/REV_UFW from manifests to reproduce

# 2. Train (cosine warmup+decay, peak 1e-3). 1 epoch = train_windows / BATCH steps (2048/8 = 256).
NPROC=4 STEPS=256 SEQ_LEN=4096 TEST_FREQ=50 \
  TRAIN_FILES=$(pwd)/data/dsa/phase_a/infllm_minicpm3_4096_train.parquet \
  VAL_FILES=$(pwd)/data/dsa/phase_a/infllm_minicpm3_4096_val.parquet \
  examples/dsa/run_minicpm3_dsa_phase1.sh

# 3. OOD eval from a checkpoint, per length (later)
VAL_ONLY=1 RESUME_PATH=dsa_runs/<run>/checkpoints/global_step_<N> \
  SEQ_LEN=2048 VAL_FILES=$(pwd)/data/dsa/phase_a/ood_ultrafineweb_minicpm3_L2048.parquet \
  VAL_PREFIX=val_ood_L2048 examples/dsa/run_minicpm3_dsa_phase1.sh
```

`STEP` = one optimizer update = one global batch (`BATCH` windows = `BATCH×SEQ_LEN` tokens). GPU count doesn't
change steps/epoch (global batch is fixed). `total_epochs` (config, default 4) caps max steps at
`steps_per_epoch × total_epochs`.

## State at end of session

- Datasets: small Phase-A set built ✓; 100M train set building (background).
- Code: all changes compile; loss + data logic unit-tested. Nothing run on GPUs (a 256-step 4-GPU test run
  was launched then **stopped**; GPUs verified clean afterward — no orphans).
- `/scratch/aarti` is root-owned & unwritable (no sudo) → datasets written to repo NFS instead.

## Open / next

- Finish 100M build; set `STEPS = 100M/(BATCH×4096)` for 1 epoch (≈3050 at BATCH=8).
- Optional: `EPOCHS` knob in the run script; recall@multiple-k diag; LR warmup already done.
- Phase B: 32K on 4×H100 (memory: fits, ~66 GB peak).
