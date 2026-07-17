# DSA checkpoint loading — scenarios & exact commands

How to source **base** and **indexer** weights for every start/resume situation across Phase-1 (dense
warm-up) and Phase-2 (sparse). Companion to `docs/dsa_phase2_implementation.md`.

Phases:
- **Phase 1 (`dsa_mode=dense_warmup`, `loss_mode=indexer_kl`)** — base **frozen**, only the lightning indexer
  trains. So a Phase-1 checkpoint's base == stock `openbmb/MiniCPM3-4B`.
- **Phase 2 (`dsa_mode=sparse`, `loss_mode=dsa_sparse`)** — base **and** indexer train (two optimizer groups).

---

## The three loading mechanisms

| Mechanism | Loads | Optimizer / step | GPU count | Use for |
|---|---|---|---|---|
| **`model.path`** (`openbmb/MiniCPM3-4B`) | base only (no indexer; grafted fresh by `attach_indexers`) | fresh / step 0 | any | every from-scratch start |
| **`WARMSTART_PATH`** (consolidated dict from `consolidate_indexer_ckpt.py`, loaded `strict=False` pre-FSDP) | indexer-only **or** full base+indexer, whatever the file carries | **fresh / step 0** | **any** (world-size-agnostic) | carry trained weights into a NEW run |
| **`RESUME_PATH`** (verl native resume) | model **+ optimizer + step + RNG + dataloader** | **continues** | **locked to the saved count** (`model_world_size_{N}`) | continue an interrupted run |

Rule of thumb: **continue exactly where I left off → `RESUME_PATH`** (same GPU count). **Take trained weights
into a new run (new phase/data/schedule/GPU count) → consolidate → `WARMSTART_PATH`** (fresh optimizer, any
GPU count).

### ⚠ Changing GPU count mid-run is NOT a seamless resume
Native resume is **world-size-locked**: the checkpoint is per-rank shards keyed to the saved count —
`model_world_size_{N}_rank_{r}.pt`, `optim_world_size_{N}_*`, `extra_state_world_size_{N}_*`, and
`data_{0..N-1}.pt` (StatefulDataLoader state per dp-rank). If you start on 4 GPUs, interrupt, and relaunch
`RESUME_PATH` on **8** GPUs, the loader looks for `model_world_size_8_*` / `data_{4..7}.pt` which don't exist
→ it **hard-fails** (`fsdp_checkpoint_manager.py` load_checkpoint). None of those artifacts reshards
automatically.

What each option preserves when you change the GPU count (e.g. 4 → 8):

| State | `RESUME_PATH`, **same** count | `WARMSTART_PATH`, **any** count (full consolidate) |
|---|---|---|
| Model weights (base+indexer) | ✅ restored | ✅ restored |
| Optimizer (Adam m/v moments) | ✅ restored | ❌ reset to zero (transient until re-warmed) |
| Step counter / LR-schedule position | ✅ continues | ❌ resets to step 0 (warmup + cosine restart) |
| Dataloader position | ✅ continues | ❌ restarts from the beginning (re-sees data) |
| RNG | ✅ | ❌ |

So **`WARMSTART_PATH` across a different GPU count is a fresh run initialized from your weights — not a
continuation.** To continue seamlessly (optimizer momentum, step, LR position, data position all intact),
**resume on the same GPU count the checkpoint was saved with.** If you know you'll scale up later, start the
run on the larger count from the beginning so every resume stays on that count. (Making 4→8 a true resume
would require resharding the model AND the optimizer moments + re-keying the dataloader/step — not supported
by verl's per-rank `torch.save` checkpoint format here.)

---

## Scenario table

| # | Scenario | Base ← | Indexer ← | Optimizer / step | GPUs |
|---|---|---|---|---|---|
| 1 | **Start Phase 1** (dense warmup) | `model.path` (stock; frozen) | fresh random | indexer-only group; step 0 | any |
| 2 | **Resume Phase 1** | P1 ckpt (==stock) | P1 ckpt | indexer-only; continues | same as saved |
| 3 | **Start Phase 2 from a Phase-1 ckpt** (the recipe) | `model.path` (stock == P1 frozen base) | **consolidated P1 indexer** → `WARMSTART_PATH` | fresh 2-group (base 7.3e-6 / idx 1e-3); step 0 | any |
| 4 | **Resume Phase 2** | P2 ckpt | P2 ckpt | 2-group; continues | same as saved |
| 5 | **Start Phase 2, base-only / fresh indexer** | `model.path` (stock) | fresh random | fresh 2-group; step 0 | any |
| 6 | **Branch/restart Phase 2 from a Phase-2 ckpt** (new schedule/data/GPU count) | **consolidated P2 full** | consolidated P2 full | fresh 2-group; step 0 | any |
| 7 | **Start either phase from a custom (non-hub) base** | `model.path=<local dir>` | fresh (or warmstart) | fresh; step 0 | any |

A consolidated file with **no `*.indexer.*` keys fails loudly** (`n_idx > 0` assert in
`_warmstart_from_consolidated`) — a base-only/mis-pathed file can't silently degrade to a fresh indexer. For a
genuine fresh-indexer start you deliberately leave `WARMSTART_PATH` empty (row 5).

**Row 4 requires the same GPU count** the checkpoint was saved with (native resume is world-size-locked).
To continue a run on a **different** GPU count you must fall back to **row 6** (`WARMSTART_PATH`), which keeps
only the weights — optimizer/step/data reset. See "Changing GPU count mid-run is NOT a seamless resume" above.

---

## Exact commands per scenario

Common env (in-container): `export PYTHONPATH=$REPO/.devlibs/tf457lib`. `NPROC` = GPU count; `BATCH` must be
divisible by `NPROC`. Native-resume rows require `NPROC` == the count the checkpoint was saved with.

### 1. Start Phase 1
```bash
bash examples/dsa/run_minicpm3_dsa_phase1.sh          # dsa_mode=dense_warmup, loss_mode=indexer_kl
```

### 2. Resume Phase 1 (same GPU count as saved)
```bash
RESUME_PATH=/…/phase1/checkpoints/global_step_900 \
bash examples/dsa/run_minicpm3_dsa_phase1.sh          # native resume: model+optim(indexer)+step
```

### 3. Start Phase 2 from a Phase-1 checkpoint  ← the normal Phase-2 start
```bash
# (a) consolidate the warmed indexer (indexer-only; base stays stock) — world-size-agnostic, run once
python scripts/dsa/consolidate_indexer_ckpt.py \
  --ckpt-dir /…/phase1/checkpoints \
  --out      /…/indexer_full_from_phase1.pt \
  --key-substr ".indexer."
# (b) launch sparse training warm-started from it (any NPROC; fresh 2-group optimizer, step 0)
WARMSTART_PATH=/…/indexer_full_from_phase1.pt TRAIN_FILES=/…/m3a_sft_code.parquet \
NPROC=4 BATCH=32 \
bash examples/dsa/run_minicpm3_dsa_phase2.sh
# …or just run the orchestrator, which does (a)+(b)+data-prep+step-calc, all logged:
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash examples/dsa/run_phase2_long_pipeline.sh
```

### 4. Resume Phase 2 (same GPU count as saved)
```bash
RESUME_PATH=/…/phase2/…/train/checkpoints/global_step_N \
bash examples/dsa/run_minicpm3_dsa_phase2.sh          # native resume: model+optim(2-group)+step
```

### 5. Start Phase 2, base-only / fresh indexer (no warm-start)
```bash
TRAIN_FILES=/…/m3a_sft_code.parquet NPROC=4 BATCH=32 \
bash examples/dsa/run_minicpm3_dsa_phase2.sh          # WARMSTART_PATH & RESUME_PATH empty -> KL warms a fresh indexer
```

### 6. Branch/restart Phase 2 from a Phase-2 checkpoint (keep base+indexer, new run, any GPU count)
```bash
# consolidate the FULL model (base+indexer) — note --key-substr ''
python scripts/dsa/consolidate_indexer_ckpt.py \
  --ckpt-dir /…/phase2/…/train/checkpoints/global_step_N \
  --out      /…/phase2_full.pt \
  --key-substr ''
WARMSTART_PATH=/…/phase2_full.pt TRAIN_FILES=/…/other.parquet NPROC=6 BATCH=48 \
bash examples/dsa/run_minicpm3_dsa_phase2.sh          # loads base+indexer strict=False, fresh optimizer, step 0
```

### 7. Custom (non-hub) base
```bash
bash examples/dsa/run_minicpm3_dsa_phase2.sh model.path=/local/minicpm3-base   # + optional WARMSTART_PATH
```

---

## Why row 3 needs only the indexer (and its one assumption)
Phase-1 runs `freeze_base_train_indexer` (`requires_grad=False` on all but `*.indexer.*`), so the base in a
Phase-1 checkpoint is **bit-identical to stock** `openbmb/MiniCPM3-4B`. Pulling the base from `model.path` and
only the indexer from the consolidated file therefore reproduces Phase-1's model exactly, while keeping the
warm-start file tiny (310 tensors vs the full 16 GB). **Assumption:** this holds only while Phase-1 freezes
the base. If a future Phase-1 variant trains the base, treat it like row 6 (consolidate the full model).
*(Optional guard, not yet added: verify a few base tensors in the P1 ckpt match stock and error otherwise.)*

## Mechanics reference
- Consolidation reconstructs full tensors from per-rank DTensor shards generically (Replicate → take one;
  Shard(d) → concat local shards along d, trim FSDP padding), no GPU/distributed context needed:
  `scripts/dsa/consolidate_indexer_ckpt.py`.
- Warm-load happens in `attach_indexers` → `_warmstart_from_consolidated` (`verl/models/transformers/minicpm_dsa.py`),
  **before** FSDP wrap, so it is GPU-count-agnostic. Config field: `dsa_warmstart_path`.
- Native resume: `trainer.resume_mode=resume_path trainer.resume_from_path=<global_step_N dir>`
  (`verl/utils/checkpoint/*`); loads `model_world_size_{N}_rank_{r}.pt` → world-size locked.
