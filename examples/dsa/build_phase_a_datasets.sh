#!/usr/bin/env bash
# Reproducibly build ALL DSA Phase-A datasets from pinned sources + a fixed seed:
#   1. InfLLM-V2-data-5B  -> TRAIN + in-distribution VAL (seeded shuffle, document-disjoint)  @ SEQ_LEN
#   2. Ultra-FineWeb (en) -> OUT-OF-DISTRIBUTION VAL at MULTIPLE lengths (one parquet per length)
#
# Every knob below is pinned so a later run regenerates byte-identical splits (given the same dataset
# commit SHAs, which the prep scripts resolve and record in each output's MANIFEST.json). To reproduce an
# EXACT past build, copy REV_INFLLM / REV_UFW from its manifest into the env before running.
#
# Usage:
#   examples/dsa/build_phase_a_datasets.sh                      # defaults below
#   SEED=7 SEQ_LEN=4096 TRAIN_WINDOWS=2048 examples/dsa/build_phase_a_datasets.sh
set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}
# transformers 4.57.1 staged in-repo so it shadows system transformers 5.x (needed for MiniCPM3 tokenizer).
DEVLIBS=${DEVLIBS:-${REPO_ROOT}/.devlibs/tf457lib}
export PYTHONPATH=${DEVLIBS}:${REPO_ROOT}/examples/dsa:${PYTHONPATH:-}

# --- pinned build parameters (change deliberately; they define the dataset identity) ---
SEED=${SEED:-1234}
SEQ_LEN=${SEQ_LEN:-4096}                 # Phase A context length (InfLLM train + in-dist val)
TRAIN_WINDOWS=${TRAIN_WINDOWS:-2048}     # InfLLM train one-doc windows
VAL_WINDOWS=${VAL_WINDOWS:-256}          # InfLLM in-distribution val windows (disjoint)
OVERSAMPLE=${OVERSAMPLE:-1.0}            # collect this x (train+val) before shuffle/split
OOD_LENGTHS=${OOD_LENGTHS:-1024,2048,4096}  # OOD val lengths (Ultra-FineWeb; short corpus -> keep modest)
OOD_PER_LEN=${OOD_PER_LEN:-128}          # OOD val windows per length
OOD_MAX_FILES=${OOD_MAX_FILES:-64}       # Ultra-FineWeb shards to sample (of 2048)
OOD_MIN_SCORE=${OOD_MIN_SCORE:-0.9}      # quality filter: keep docs with classifier score >= this (0 disables)
MODEL=${MODEL:-openbmb/MiniCPM3-4B}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/data/dsa/phase_a}
# Optional: pin exact source commits for perfect reproduction (else current main is used AND recorded).
REV_INFLLM=${REV_INFLLM:-}
REV_UFW=${REV_UFW:-}

mkdir -p "${OUT_DIR}"

# --- tee ALL stdout+stderr (incl. the `set -x` trace = every command actually run) to a timestamped log,
#     so a build is self-documenting and manually reproducible from the log. ---
BUILD_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-${OUT_DIR}/build-${BUILD_TS}.log}
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[phase-a-data] log=${LOG_FILE}"
echo "[phase-a-data] out_dir=${OUT_DIR} seed=${SEED} seq_len=${SEQ_LEN} train=${TRAIN_WINDOWS} val=${VAL_WINDOWS}"
echo "[phase-a-data] ood_lengths=${OOD_LENGTHS} ood_per_len=${OOD_PER_LEN} ood_max_files=${OOD_MAX_FILES} ood_min_score=${OOD_MIN_SCORE}"
echo "[phase-a-data] rev_infllm=${REV_INFLLM:-<main>} rev_ufw=${REV_UFW:-<main>} model=${MODEL}"
# --- record EXACTLY how this build was launched (self-reproducing log): argv + every consumed env var ---
echo "[phase-a-data] cwd=$(pwd) host=$(hostname)"
echo "[phase-a-data] argv: $0 $*"
echo "[phase-a-data] env: SEED=${SEED} SEQ_LEN=${SEQ_LEN} TRAIN_WINDOWS=${TRAIN_WINDOWS} VAL_WINDOWS=${VAL_WINDOWS}" \
     "OVERSAMPLE=${OVERSAMPLE} OOD_LENGTHS=${OOD_LENGTHS} OOD_PER_LEN=${OOD_PER_LEN} OOD_MAX_FILES=${OOD_MAX_FILES}" \
     "OOD_MIN_SCORE=${OOD_MIN_SCORE} MODEL=${MODEL} OUT_DIR=${OUT_DIR} REV_INFLLM=${REV_INFLLM:-<main>}" \
     "REV_UFW=${REV_UFW:-<main>} DEVLIBS=${DEVLIBS} PYTHONPATH=${PYTHONPATH:-}"

# 1) InfLLM train + in-distribution val (seeded, disjoint)
python "${REPO_ROOT}/examples/dsa/prepare_real_data.py" \
    --out "${OUT_DIR}/infllm_minicpm3_${SEQ_LEN}_train.parquet" \
    --val_out "${OUT_DIR}/infllm_minicpm3_${SEQ_LEN}_val.parquet" \
    --num_windows "${TRAIN_WINDOWS}" --val_windows "${VAL_WINDOWS}" \
    --seq_len "${SEQ_LEN}" --seed "${SEED}" --oversample "${OVERSAMPLE}" --model "${MODEL}" \
    ${REV_INFLLM:+--revision "${REV_INFLLM}"}

# 2) Ultra-FineWeb OOD val at multiple lengths (quality-filtered)
python "${REPO_ROOT}/examples/dsa/prepare_ood_data.py" \
    --out_prefix "${OUT_DIR}/ood_ultrafineweb_minicpm3" \
    --lengths "${OOD_LENGTHS}" --per_len "${OOD_PER_LEN}" --max_files "${OOD_MAX_FILES}" \
    --min_score "${OOD_MIN_SCORE}" --seed "${SEED}" --model "${MODEL}" \
    ${REV_UFW:+--revision "${REV_UFW}"}

echo "[phase-a-data] DONE. Files + MANIFEST.json under ${OUT_DIR}"
ls -la "${OUT_DIR}"
