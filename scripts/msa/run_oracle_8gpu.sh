#!/usr/bin/env bash
# MSA block-oracle probe, fanned out over all local GPUs and merged.
#
# The probe is embarrassingly parallel over documents: each rank takes a strided slice
# (docs[rank::NGPU]) and writes raw sums/counts; merge_oracle.py combines them EXACTLY (weighted by
# sample count -- averaging the shards' means would be wrong).
#
# Reproduce:  bash scripts/msa/run_oracle_8gpu.sh
# Override any knob via env, e.g.  NGPU=4 KS="16" BLOCK_SIZES="128" NUM_DOCS=32 bash ...
set -Eeuo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

MODEL=${MODEL:-/cb/ml-eng/aarti/models/qwen3_4b_thinking_2507}
INPUT=${INPUT:-/cb/ml-eng/aarti/msa/data/long_docs_qwen3_32768.jsonl}
OUT_DIR=${OUT_DIR:-/cb/ml-eng/aarti/msa/oracle}
TAG=${TAG:-qwen3_4b_thinking_32k}
SEQ_LEN=${SEQ_LEN:-32768}
NUM_DOCS=${NUM_DOCS:-64}
NUM_QUERIES=${NUM_QUERIES:-512}
KS=${KS:-"8 16 32 64"}
BLOCK_SIZES=${BLOCK_SIZES:-"64 128"}
NGPU=${NGPU:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}
NGPU=${NGPU:-1}

RUN_DIR="${OUT_DIR}/${TAG}"
mkdir -p "${RUN_DIR}/shards"
MASTER_LOG="${RUN_DIR}/run.log"

{
  echo "=== $(date -Is) MSA oracle probe ==="
  echo "CMD:   bash $0 $*"
  echo "ENV:   MODEL=${MODEL} INPUT=${INPUT} SEQ_LEN=${SEQ_LEN} NUM_DOCS=${NUM_DOCS}"
  echo "       NUM_QUERIES=${NUM_QUERIES} KS='${KS}' BLOCK_SIZES='${BLOCK_SIZES}' NGPU=${NGPU}"
  echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "GIT:   $(git rev-parse HEAD 2>/dev/null || echo n/a)"
} | tee -a "${MASTER_LOG}"

[[ -d "${MODEL}" || "${MODEL}" == */* ]] || { echo "MODEL not found: ${MODEL}" >&2; exit 1; }
[[ -f "${INPUT}" ]] || { echo "INPUT not found: ${INPUT}" >&2; exit 1; }

pids=()
for ((r = 0; r < NGPU; r++)); do
    CUDA_VISIBLE_DEVICES=${r} python3 scripts/msa/probe_block_oracle.py \
        --model "${MODEL}" \
        --input "${INPUT}" \
        --seq-len "${SEQ_LEN}" \
        --num-docs "${NUM_DOCS}" \
        --num-queries "${NUM_QUERIES}" \
        --ks ${KS} \
        --block-sizes ${BLOCK_SIZES} \
        --num-shards "${NGPU}" --shard-id "${r}" \
        --out "${RUN_DIR}/shards/shard_${r}.json" \
        >"${RUN_DIR}/shards/shard_${r}.out" 2>&1 &
    pids+=($!)
    echo "launched rank ${r} on GPU ${r} (pid ${pids[-1]})" | tee -a "${MASTER_LOG}"
done

fail=0
for ((r = 0; r < NGPU; r++)); do
    if wait "${pids[r]}"; then
        echo "rank ${r} OK" | tee -a "${MASTER_LOG}"
    else
        echo "rank ${r} FAILED -- see ${RUN_DIR}/shards/shard_${r}.out" | tee -a "${MASTER_LOG}"
        fail=1
    fi
done
[[ ${fail} -eq 0 ]] || { echo "one or more ranks failed; NOT merging (partial merge would silently under-sample)" | tee -a "${MASTER_LOG}"; exit 1; }

python3 scripts/msa/merge_oracle.py --out "${RUN_DIR}/${TAG}.json" \
        "${RUN_DIR}"/shards/shard_*.json 2>&1 | tee -a "${MASTER_LOG}"

echo "=== $(date -Is) done -> ${RUN_DIR}/${TAG}.json ===" | tee -a "${MASTER_LOG}"
