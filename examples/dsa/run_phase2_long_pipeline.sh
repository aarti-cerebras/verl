#!/usr/bin/env bash
# DSA Phase-2 LONG RUN — end-to-end, fully scripted & logged so it reproduces from this one file.
# Pipeline (each step's exact command is printed via `set -x` and tee'd to the master log):
#   1. consolidate the Phase-1 warmed-indexer weights -> world-size-agnostic indexer_full.pt
#      (consolidate_indexer_ckpt.py; run ONCE, cached/reused if present)
#   2. build the Code-weighted SFT parquet from the self-gen corpus (trajectories_to_sft_parquet.py)
#   3. compute STEPS for ~1 epoch = floor(rows / BATCH)
#   4. launch sparse training (run_minicpm3_dsa_phase2.sh) warm-started from indexer_full.pt, top_k=512,
#      base_lr 7.3e-6 / indexer_lr 1e-3, kl_checkpoint on, pad_mode=no_padding, on however many GPUs you give.
#
# Reproduce yourself:  edit the env knobs below (or export them) and run:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 setsid nohup bash examples/dsa/run_phase2_long_pipeline.sh \
#     </dev/null >/cb/ml-eng/aarti/dsa/phase2_long_pipeline.out 2>&1 &
# (warm-start is world-size-agnostic, so NPROC can be anything; BATCH must be divisible by NPROC.)
set -Eeuo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH=${REPO_ROOT}/.devlibs/tf457lib:${PYTHONPATH:-}

# ---------------- knobs (all env-overridable) ----------------
DATA_ROOT=${DATA_ROOT:-/cb/ml-eng/aarti/dsa}
CORPUS=${CORPUS:-${DATA_ROOT}/m3a_gen_20260714_163317/trajectories.jsonl}
PHASE1_CKPT=${PHASE1_CKPT:-${DATA_ROOT}/indexer_warmup/phase1-klckpt_infllm_minicpm3_32768_250M_st934_bs8_20260711_205333/checkpoints}
DOMAINS=${DOMAINS:-full}                 # 'full'/'all' = all 7 domains; or a single domain, e.g. Code
MIN_TOTAL=${MIN_TOTAL:-0}                # use ALL samples (no length floor); <512-token ones just run dense
NPROC=${NPROC:-8}                        # use all 8 GPUs on the node (warm-start is world-size-agnostic)
BATCH=${BATCH:-64}                       # global batch (must be divisible by NPROC); 8/GPU at micro_bsz=1
MICRO_BSZ=${MICRO_BSZ:-1}                # keep 1 for the long code docs (micro_bsz>1 OOMs)
SEQ_LEN=${SEQ_LEN:-4096}
TOPK=${TOPK:-512}                        # sparse key budget (train == deploy)
FP8_UE8M0=${FP8_UE8M0:-false}            # true => UE8M0 indexer fake-quant (match serve kernel; closes ~2% drift)
VAL_FILES=${VAL_FILES:-}                  # optional held-out val parquet (self-gen, disjoint prompts) for val loss
TEST_FREQ=${TEST_FREQ:-190}              # eval val every N steps (aligned with SAVE_FREQ); ignored if no VAL_FILES
EPOCHS=${EPOCHS:-1}                       # ~1 epoch over the data
BASE_LR=${BASE_LR:-1e-5}                  # MiniCPM3-4B official full-finetune LR (OpenBMB LLaMA-Factory recipe)
WARMUP_RATIO=${WARMUP_RATIO:-0.1}         # 10% warmup, matching that recipe
SAVE_FREQ=${SAVE_FREQ:-190}               # ~every 2h (≈38 s/step × 190 ≈ 7220 s); ~14 saves over a 2805-step epoch
MAX_CKPT=${MAX_CKPT:-20}                  # keep only the last N checkpoints (~12GB each) -> bounds disk; empty = keep all
# leave INDEXER_LR / KL_BLOCK / KL_CKPT / LAMBDA / LR_SCHED at run_minicpm3_dsa_phase2.sh defaults
#   (1e-3 / 512 / 1024 / true / 1.0 / cosine) unless overridden here.

# resolve domain filter + data tag: 'full'/'all'/'' => all domains (no --domains filter)
if [[ -z "${DOMAINS}" || "${DOMAINS,,}" == "full" || "${DOMAINS,,}" == "all" ]]; then
    DATA_TAG=full; DOMAIN_ARG=()
else
    DATA_TAG=${DOMAINS,,}; DOMAIN_ARG=(--domains "${DOMAINS}")
fi

RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_BASE=${RUN_BASE:-${DATA_ROOT}/phase2_long}
RUN_DIR=${RUN_DIR:-${RUN_BASE}/phase2_${DATA_TAG}_k${TOPK}_1ep_${RUN_TS}}
mkdir -p "${RUN_DIR}/logs"
MASTER_LOG=${MASTER_LOG:-${RUN_DIR}/pipeline-${RUN_TS}.log}

# resolve the Phase-1 shard folder (global_step_N) and co-locate the consolidated indexer there (step-tagged)
if [[ "$(basename "${PHASE1_CKPT}")" == global_step_* ]]; then
    P1_STEP_DIR="${PHASE1_CKPT}"
else
    P1_STEP_DIR="${PHASE1_CKPT}/global_step_$(cat "${PHASE1_CKPT}/latest_checkpointed_iteration.txt" 2>/dev/null)"
fi
P1_STEP_NUM=$(basename "${P1_STEP_DIR}" | sed 's/global_step_//')
INDEXER_FULL=${INDEXER_FULL:-${P1_STEP_DIR}/consolidated_indexer_step${P1_STEP_NUM}.pt}
TRAIN_PARQUET=${TRAIN_PARQUET:-${DATA_ROOT}/m3a_sft_${DATA_TAG}.parquet}

# tee EVERYTHING (this script's set -x trace + all child stdout/stderr) to the master log
exec > >(tee -a "${MASTER_LOG}") 2>&1
echo "[pipeline] ===== DSA Phase-2 long run ====="
echo "[pipeline] host=$(hostname) date=$(date -Is) git=$(git rev-parse --short HEAD 2>/dev/null)"
echo "[pipeline] RUN_DIR=${RUN_DIR}  MASTER_LOG=${MASTER_LOG}"
echo "[pipeline] CORPUS=${CORPUS}"
echo "[pipeline] PHASE1_CKPT=${PHASE1_CKPT}"
echo "[pipeline] knobs: DOMAINS=${DOMAINS} MIN_TOTAL=${MIN_TOTAL} NPROC=${NPROC} BATCH=${BATCH} MICRO_BSZ=${MICRO_BSZ} SEQ_LEN=${SEQ_LEN} TOPK=${TOPK} FP8_UE8M0=${FP8_UE8M0} EPOCHS=${EPOCHS} BASE_LR=${BASE_LR} WARMUP_RATIO=${WARMUP_RATIO}"
echo "[pipeline] INDEXER_FULL=${INDEXER_FULL}  TRAIN_PARQUET=${TRAIN_PARQUET}"
[ $((BATCH % NPROC)) -eq 0 ] || { echo "[pipeline] ERROR: BATCH ${BATCH} not divisible by NPROC ${NPROC}"; exit 1; }

set -x

# --- 1. consolidate Phase-1 indexer weights (idempotent: skip if already built). --out/--log both default
#        INTO the shard folder (step-tagged); we pass --out explicitly so INDEXER_FULL is deterministic here. ---
if [[ ! -f "${INDEXER_FULL}" ]]; then
    python3 scripts/dsa/consolidate_indexer_ckpt.py \
        --ckpt-dir "${P1_STEP_DIR}" --out "${INDEXER_FULL}" --key-substr ".indexer."
else
    echo "[pipeline] reusing existing ${INDEXER_FULL}"
fi

# --- 2. Code-weighted SFT parquet (idempotent) ---
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
    python3 scripts/dsa/trajectories_to_sft_parquet.py \
        --input "${CORPUS}" --out "${TRAIN_PARQUET}" --log-dir "${RUN_DIR}/logs" \
        ${DOMAIN_ARG[@]+"${DOMAIN_ARG[@]}"} --min-total "${MIN_TOTAL}"
else
    echo "[pipeline] reusing existing ${TRAIN_PARQUET}"
fi

# --- 3. steps for ~1 epoch = floor(rows / BATCH) * EPOCHS ---
ROWS=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('${TRAIN_PARQUET}')))")
STEPS=$(python3 -c "print(max(1, (${ROWS}//${BATCH}) * ${EPOCHS}))")
{ set +x; } 2>/dev/null
echo "[pipeline] parquet rows=${ROWS}  BATCH=${BATCH}  EPOCHS=${EPOCHS}  =>  STEPS=${STEPS}"
set -x

# --- 4. launch sparse training (warm-started, ~1 epoch). run_minicpm3_dsa_phase2.sh logs its own exact argv. ---
WARMSTART_PATH="${INDEXER_FULL}" \
TRAIN_FILES="${TRAIN_PARQUET}" \
NPROC="${NPROC}" BATCH="${BATCH}" MICRO_BSZ="${MICRO_BSZ}" SEQ_LEN="${SEQ_LEN}" TOPK="${TOPK}" STEPS="${STEPS}" \
FP8_UE8M0="${FP8_UE8M0}" VAL_FILES="${VAL_FILES}" TEST_FREQ="${TEST_FREQ}" \
BASE_LR="${BASE_LR}" WARMUP_RATIO="${WARMUP_RATIO}" SAVE_FREQ="${SAVE_FREQ}" MAX_CKPT="${MAX_CKPT}" \
RUN_DIR="${RUN_DIR}/train" EXP_NAME="phase2_${DATA_TAG}_k${TOPK}_1ep_${RUN_TS}" \
bash examples/dsa/run_minicpm3_dsa_phase2.sh "$@"

{ set +x; } 2>/dev/null
echo "[pipeline] ===== done. master log: ${MASTER_LOG} ; train logs: ${RUN_DIR}/train ====="
