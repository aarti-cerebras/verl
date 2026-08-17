#!/usr/bin/env bash
# Qwen3-4B MSA Phase-2b on a LONG-CONTEXT band, initialised from the 32K run's step 10700.
#
#   BAND=16k32k ./examples/msa/run_qwen3_msa_phase2_longctx.sh   # 8xH100 80GB  (the only band that fits)
#   BAND=16k45k ./examples/msa/run_qwen3_msa_phase2_longctx.sh   # needs >=143 GB/GPU -- see WHICH BAND
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
# 3. SEQ_LEN -> the band's measured MAX (46080 or 32768). MSASFTDataset TRUNCATES longer rows rather
#    than dropping them, so a SEQ_LEN below the band max silently discards tokens instead of failing.
#
# 4. LENGTH_TIERS -> ceil(SEQ_LEN / TIER_WIDTH), TIER_WIDTH unchanged at 2048. Tier index is
#    `min(length // width, n_tiers - 1)`. The 32K run's 16 x 2048 = 32,768 covered its whole window,
#    i.e. one tier per 2,048 tokens; each band preserves that DENSITY. Leaving tiers at 16 for the 45K
#    band would collapse everything >= 30,720 (about a third of it) into one clamped tier.
#    DATALOADER-ONLY: does not touch the model, sparsity or FLOPs. NOT msa_top_k.
#
# 5. STEPS -> train_rows / BATCH = one epoch.
#
# UNCHANGED, deliberately: TOPK=16, BLOCK_SIZE=128, DENSE_PREFIX=3, KL_LAMBDA=1.0, LR=5e-6,
# INDEXER_LR=1e-4, cosine + 3% warmup, MIN_LR_RATIO=0.1, CLIP_GRAD=1.0, BATCH=8, fsdp2, bf16,
# ACT_OFFLOAD.
#
# ---------------------------------------------------------------------------------------------------
# WHICH BAND: 16k45k DOES NOT FIT ON 8xH100 80GB. Measured 2026-08-12 on ml-eng-gpu-21.
#
# GPU memory -- not host RAM -- is the binding constraint, and it scales linearly off the 32K run:
#
#            L=32768 (proven)          L=46080 (extrapolated, x1.406)     80 GB card
#   alloc    34.6 GB                   ~49 GB                             ok
#   reserved 48.4 GB                   ~68 GB  (nvidia-smi showed 72)     ~85-90% -> DEAD
#
# At that occupancy the caching allocator stops finding blocks and churns cudaMalloc under the NVIDIA
# driver lock. The signature is NOT an OOM traceback -- it is 0% SM utilisation, ranks in D state on
# `os_acquire_rwlock_read`, and forward progress that never reaches step 1. Two max-tier smokes (299
# rows of 45.5K-46.08K) confirmed it: 33 min and 14 min, ZERO steps posted, at 66-73 GB.
#
# Host RAM was never the problem: it plateaued at ~1760 GB of 1999, essentially the 32K run's own
# 1736 GB, so TILED_MLP_SHARDS=8 does absorb the 1.4x sequence growth as intended.
#
# The two standard escapes are closed BY DESIGN, so do not go looking for them:
#   * Ulysses SP     -- monkey_patch.py asserts ulysses_sp_size == 1 for MSA: the index branch scores
#                       every query against the FULL key sequence and the Eq.-9 teacher is a softmax
#                       over the full causal support, so a sharded sequence normalises both over a
#                       fragment.
#   * fused linear+CE -- would drop the 14 GB logits tensor (46080 x 151936 x bf16), but the MSA branch
#                       `return`s before patch_forward_with_backends, so use_fused_kernels is silently
#                       ignored. This recipe also runs use_remove_padding=False + pad_mode=no_padding,
#                       which the engine's fused branches are not wired for. Real work + parity test.
# FSDP2 does not help: it shards params/grads/optimizer (~8-10 GB/rank), not the ~60 GB of activations.
# Neither did the free levers (KL_BLOCK 512->128, TILED_MLP_SHARDS 8->16, garbage_collection_threshold
# 0.8): they held 100% SM for ~8 min, then hit the same ceiling. Do not re-run that experiment.
#
# Worth knowing but NOT sufficient: all 8 ranks open a CUDA primary context on all 8 GPUs (the launch
# leaves CUDA_VISIBLE_DEVICES unset), costing 7 x ~524 MiB = ~3.7 GB per card. Reclaiming it needs
# ~15 GB more to matter.
#
# So: run 16k32k here. The 32768-46080 tail (9,467 rows, ~365M tokens) stays in split_16k_45k_v1 and
# needs either a >=143 GB/GPU host -- where the 16k45k numbers above were originally measured -- or
# fused CE on the MSA path.
# ---------------------------------------------------------------------------------------------------
set -euo pipefail

# Two independent roots, deliberately. An earlier revision had a single BASE=/home/aarti_cerebras holding
# BOTH the repo and the artifacts; on the ml-eng-gpu-* hosts those are different filesystems (repo on the
# home NFS, artifacts on the /cb/ml-eng FSx share), so one variable cannot name both.
export REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}   # same default as run_qwen3_msa_phase2.sh
MSA_BASE=${MSA_BASE:-/cb/ml-eng/aarti}                                    # data + checkpoints
export RUNS_BASE=${RUNS_BASE:-${MSA_BASE}/msa/sparse}

# --- BAND: the four values below (split, seq_len, tiers, steps) are ONE decision, not four ----------
# They must move together -- SEQ_LEN below the split's max silently truncates, and LENGTH_TIERS is
# ceil(SEQ_LEN/TIER_WIDTH) by construction -- so they are set here as a unit rather than left as four
# independent env vars a caller can desynchronise. STEPS is train_rows / BATCH(8) = one epoch.
# Default is 16k32k: 16k45k does not run on 80 GB cards (see WHICH BAND above).
BAND=${BAND:-16k32k}
LONGCTX_ARTIFACT=${LONGCTX_ARTIFACT:-${MSA_BASE}/msa/data/qwen3-4b-thinking-2507__longctx_tierA__L131072_20260811_022925}
case "${BAND}" in
  16k32k)  # 23,199 rows / 0.567B tokens; val 739. Runs at the PROVEN 32K profile (34.6 GB alloc).
    _SPLIT_DIR=split_16k_32k_v1; _SEQ_LEN=32768; _TIERS=16; _STEPS=2899 ;;
  16k45k)  # 32,666 rows / 0.932B tokens; val 1,023. REQUIRES >=143 GB/GPU.
    _SPLIT_DIR=split_16k_45k_v1; _SEQ_LEN=46080; _TIERS=23; _STEPS=4083 ;;
  *) echo "ERROR: unknown BAND='${BAND}' (expected 16k32k or 16k45k)"; exit 1 ;;
esac
SPLIT=${SPLIT:-${LONGCTX_ARTIFACT}/${_SPLIT_DIR}}

# EVERY derived artifact for this checkpoint lives in ONE named folder inside the source
# checkpoint dir. They were previously split across ~/models/ (merged HF) and
# ~/msa/indexer_warmup/ (indexer .pt) -- each following its own type convention, which scattered
# one checkpoint across three directories. See $CKPT/DERIVED.json.
# NOTE the `_ckpt/` segment: run_qwen3_msa_phase2.sh puts every run under ${RUNS_BASE}/_ckpt/${CONFIG_TAG}
# (line 206), so the source run's directory -- and therefore this derived folder -- is one level deeper
# than the sparse root. Omitting it makes the preflight fail on a missing config.json.
# Anchored on MSA_BASE, NOT RUNS_BASE: RUNS_BASE is an OUTPUT knob (a smoke run redirects it), and the
# input checkpoint must not move when it does.
CKPT=${CKPT:-${MSA_BASE}/msa/sparse/_ckpt/p2_qwen3-4b-thinking-2507_ph2b_split_v1_L32k_bs8_k16_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t16w2048_st11214_v2/global_step_10700/qwen3_4b_msa_p2b_k16_step10700}
export MODEL_PATH=${MODEL_PATH:-$CKPT}
export WARMSTART=${WARMSTART:-$CKPT/msa_p2b_k16_step10700_indexer_full.pt}
export TRAIN_FILES=${TRAIN_FILES:-${SPLIT}/train-00000.parquet}
export VAL_FILES=${VAL_FILES:-${SPLIT}/val-00000.parquet}

export SEQ_LEN=${SEQ_LEN:-${_SEQ_LEN}}
export STEPS=${STEPS:-${_STEPS}}
export BATCH=${BATCH:-8}
export LENGTH_TIERS=${LENGTH_TIERS:-${_TIERS}}
export TIER_WIDTH=${TIER_WIDTH:-2048}
export TOPK=${TOPK:-16}
# 4 -> 8. The MLP forward/backward is chunked along the sequence dim (torch.chunk(x, shards, dim=-2)),
# so only 1/N of the FFN intermediates are live at once. This is the SANCTIONED substitute for gradient
# checkpointing, which is forbidden here: HF's version runs the first pass under no_grad, so the
# `_msa_kl` side effect would be stashed WITHOUT a graph and contribute ZERO indexer gradient, silently.
# Raised because an early 46K attempt died at step 6 with host RAM at 1,416 GB of 1,771 (80%) from
# activation offload, while each GPU sat at only 29.7 GB of 143. Halves peak MLP activation memory.
# Kept at 8 for 16k32k too, even though the source 32K run used 4: pure memory/compute tradeoff with no
# numerics change, and it keeps host RAM under the 1,736 GB of 1,999 (87%) that run sat at.
# (ACT_GPU_LIMIT would be the better dial, but run_qwen3_msa_phase2.sh never passes it to the trainer,
#  and verl only reads it on the veomni engine -- not the FSDP path we use. Unplumbed = no effect.)
export TILED_MLP_SHARDS=${TILED_MLP_SHARDS:-8}
# -1 = use the ENTIRE val split. The source run capped this at 256, but MSASFTDataset applies the cap by
# taking the FIRST N entries of the already-tiered order -- not a stratified draw -- so small sources
# could be absent entirely (docqarl has only 4-8 rows in val). Full val costs more eval steps every
# TEST_FREQ, and makes per-source val loss meaningful.
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export TEST_FREQ=${TEST_FREQ:-150}
export SAVE_FREQ=${SAVE_FREQ:-100}
export MAX_CKPT=${MAX_CKPT:-5}

# Pinned rather than auto-derived: DATA_TAG would otherwise come out as "train-00000", which says
# nothing about which dataset this is. CONFIG_TAG keys the checkpoint dir, so it must be stable and
# descriptive -- and must change if any experiment-defining knob changes.
_LEN_TAG=$(awk -v l="${SEQ_LEN}" 'BEGIN{printf "L%dk", int(l/1024)}')
export CONFIG_TAG=${CONFIG_TAG:-p2b_qwen3-4b-thinking-2507_longctx${BAND}_${_LEN_TAG}_bs${BATCH}_k${TOPK}_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t${LENGTH_TIERS}w${TIER_WIDTH}_st${STEPS}_from10700}

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
