#!/usr/bin/env bash
# Qwen3-4B MSA Phase-2b on the LONG-CONTEXT band (16K-45K), initialised from the 32K run's step 10700.
#
#   ./examples/msa/run_qwen3_msa_phase2_longctx.sh
#
# This is a THIN WRAPPER over examples/msa/run_qwen3_msa_phase2.sh -- it only sets env overrides and
# execs it. Deliberately not a copy: the canonical script owns the run identity (CONFIG_TAG keyed
# checkpoint dir so `resume_mode=auto` works), logs/run-<TS>.log, hydra/<TS>/, WANDB_DIR and the tee.
# A forked launcher would drift from all of that. The `outer-<TS>.log` companion is written by the
# caller's redirect, matching the existing runs.
#
# WHAT DIFFERS FROM THE SOURCE RUN
#   p2_qwen3-4b-thinking-2507_ph2b_split_v1_L32k_bs8_k16_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t16w2048_st11214_v2
#
# 1. MODEL_PATH -> $CKPT (merged step-10700 HF weights, inside the source checkpoint dir). Fresh run (step 0, new optimizer/schedule) that
#    INITIALISES from those weights; NOT a resume.
#
# 2. WARMSTART -> the step-10700 CONSOLIDATED INDEXER. Load-bearing; never clear it.
#    An earlier revision set msa_warmstart_path='' reasoning that the merged safetensors already
#    contain the trained indexer. THAT SILENTLY TRAINS A RANDOM INDEXER. Ordering is why:
#
#        from_pretrained(MODEL_PATH)   <- indexer modules DO NOT EXIST yet, so the 132 indexer tensors
#                                         are reported UNEXPECTED and DROPPED by the HF loader
#        attach_indexers(...)          <- NOW creates MSAIndexer modules, randomly initialised
#        _warmstart_from_consolidated  <- the ONLY hook that fills them (skipped when the path is '')
#
#    Verified in the aborted first attempt: all 132 keys logged as UNEXPECTED. This is exactly the
#    Phase-1 -> Phase-2 mechanism; only the warmstart FILE changes (step-10700 indexer, not Phase-1's).
#    Base weights DO load from MODEL_PATH -- only indexer keys were unexpected -- so MODEL_PATH plus
#    this warmstart together give a complete step-10700 initialisation.
#    Built by: scripts/dsa/consolidate_indexer_ckpt.py --arch msa  (132 params, 97.33M elements).
#
# 3. SEQ_LEN 32768 -> 46080. The band's measured max. MSASFTDataset TRUNCATES longer rows rather than
#    dropping them, so 32768 here would silently discard ~40% of the band's tokens.
#
# 4. LENGTH_TIERS 16 -> 23 (TIER_WIDTH unchanged at 2048). Tier index is
#    `min(length // width, n_tiers - 1)`. The 32K run's 16 x 2048 = 32,768 covered its whole window,
#    i.e. one tier per 2,048 tokens. Preserving that DENSITY at 46,080 needs ceil(46080/2048) = 23.
#    At 16, everything >= 30,720 (about a third of this band; p90 = 40,227) collapses into one clamped
#    tier. DATALOADER-ONLY: does not touch the model, sparsity or FLOPs. NOT msa_top_k.
#
# 5. STEPS 11214 -> 4083 (32,666 train rows / BATCH 8 = one epoch).
#
# UNCHANGED, deliberately: TOPK=16, BLOCK_SIZE=128, DENSE_PREFIX=3, KL_LAMBDA=1.0, LR=5e-6,
# INDEXER_LR=1e-4, cosine + 3% warmup, MIN_LR_RATIO=0.1, CLIP_GRAD=1.0, BATCH=8, fsdp2, bf16,
# ACT_OFFLOAD, TILED_MLP x4.
set -euo pipefail

BASE=/home/aarti_cerebras
export REPO_ROOT=${REPO_ROOT:-${BASE}/dsa/verl_msa}
export RUNS_BASE=${RUNS_BASE:-${BASE}/msa/sparse}

SPLIT=${SPLIT:-${BASE}/msa/data/qwen3-4b-thinking-2507__longctx_tierA__L131072_20260811_022925/split_16k_45k_v1}

# EVERY derived artifact for this checkpoint lives in ONE named folder inside the source
# checkpoint dir. They were previously split across ~/models/ (merged HF) and
# ~/msa/indexer_warmup/ (indexer .pt) -- each following its own type convention, which scattered
# one checkpoint across three directories. See $CKPT/DERIVED.json.
CKPT=${CKPT:-${BASE}/msa/sparse/p2_qwen3-4b-thinking-2507_ph2b_split_v1_L32k_bs8_k16_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t16w2048_st11214_v2/global_step_10700/qwen3_4b_msa_p2b_k16_step10700}
export MODEL_PATH=${MODEL_PATH:-$CKPT}
export WARMSTART=${WARMSTART:-$CKPT/msa_p2b_k16_step10700_indexer_full.pt}
export TRAIN_FILES=${TRAIN_FILES:-${SPLIT}/train-00000.parquet}
export VAL_FILES=${VAL_FILES:-${SPLIT}/val-00000.parquet}

export SEQ_LEN=${SEQ_LEN:-46080}
export STEPS=${STEPS:-4083}
export BATCH=${BATCH:-8}
export LENGTH_TIERS=${LENGTH_TIERS:-23}
export TIER_WIDTH=${TIER_WIDTH:-2048}
export TOPK=${TOPK:-16}
# 4 -> 8. The MLP forward/backward is chunked along the sequence dim (torch.chunk(x, shards, dim=-2)),
# so only 1/N of the FFN intermediates are live at once. This is the SANCTIONED substitute for gradient
# checkpointing, which is forbidden here: HF's version runs the first pass under no_grad, so the
# `_msa_kl` side effect would be stashed WITHOUT a graph and contribute ZERO indexer gradient, silently.
# Raised because the previous attempt died at step 6 with host RAM at 1,416 GB of 1,771 (80%) from
# activation offload, while each GPU sat at only 29.7 GB of 143. Halves peak MLP activation memory.
# (ACT_GPU_LIMIT would be the better dial, but run_qwen3_msa_phase2.sh never passes it to the trainer,
#  and verl only reads it on the veomni engine -- not the FSDP path we use. Unplumbed = no effect.)
export TILED_MLP_SHARDS=${TILED_MLP_SHARDS:-8}
# -1 = use the ENTIRE val split (1,023 rows). The source run capped this at 256, but MSASFTDataset
# applies the cap by taking the FIRST N entries of the already-tiered order -- not a stratified draw --
# so small sources could be absent entirely (docqarl has only 8 rows in val). Full val costs ~128 eval
# steps instead of 32, every TEST_FREQ steps, and makes per-source val loss meaningful.
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export TEST_FREQ=${TEST_FREQ:-150}
export SAVE_FREQ=${SAVE_FREQ:-100}
export MAX_CKPT=${MAX_CKPT:-5}

# Pinned rather than auto-derived: DATA_TAG would otherwise come out as "train-00000", which says
# nothing about which dataset this is. CONFIG_TAG keys the checkpoint dir, so it must be stable and
# descriptive -- and must change if any experiment-defining knob changes.
export CONFIG_TAG=${CONFIG_TAG:-p2b_qwen3-4b-thinking-2507_longctx16k45k_L45k_bs${BATCH}_k${TOPK}_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t${LENGTH_TIERS}w${TIER_WIDTH}_st${STEPS}_from10700}

# --- preflight: the two failures that look like a healthy run --------------------------------------
[[ -f "$MODEL_PATH/config.json" ]] || { echo "ERROR: merged model missing at $MODEL_PATH"; exit 1; }
[[ -f "$TRAIN_FILES" ]] || { echo "ERROR: train parquet missing at $TRAIN_FILES"; exit 1; }
[[ -f "$VAL_FILES"   ]] || { echo "ERROR: val parquet missing at $VAL_FILES"; exit 1; }
[[ -f "$WARMSTART"   ]] || { echo "ERROR: indexer warmstart missing at $WARMSTART -- run scripts/dsa/consolidate_indexer_ckpt.py --arch msa"; exit 1; }
python3 - "$WARMSTART" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
n = sum(1 for k in sd if ".indexer." in k)
print(f"[longctx-preflight] warmstart carries {n} indexer tensors")
assert n >= 100, f"only {n} indexer tensors in {sys.argv[1]} -- refusing to launch onto a random indexer"
PY

echo "[longctx] CONFIG_TAG=$CONFIG_TAG"
echo "[longctx] init-from (weights only, step 0): $MODEL_PATH"
echo "[longctx] indexer warmstart:               $WARMSTART"
echo "[longctx] seq_len=$SEQ_LEN tiers=${LENGTH_TIERS}x${TIER_WIDTH} top_k=$TOPK steps=$STEPS"

exec bash "${REPO_ROOT}/examples/msa/run_qwen3_msa_phase2.sh" "$@"
