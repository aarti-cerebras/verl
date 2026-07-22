#!/usr/bin/env bash
# DSA Phase-2 top_k sweep on the FULL dataset. Runs, IN ORDER: TOPK=256, 128, 512.
# Each is a full ~1-epoch run on all 8 GPUs via run_phase2_long_pipeline.sh, so they run ONE AT A TIME.
# Before each run it waits until all 8 GPUs are free, so it cleanly follows any currently-running job.
#
# Launch (detached, survives terminal loss):
#   setsid nohup bash examples/dsa/run_phase2_topk_sweep.sh </dev/null \
#     >/cb/ml-eng/aarti/dsa/phase2_topk_sweep.out 2>&1 &
# Knobs: TOPKS (order/list), NGPU, FREE_MEM_MIB (a GPU is "free" below this used-MiB), STABLE, POLL.
set -Eeuo pipefail
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

TOPKS=${TOPKS:-"256 128 512"}            # sweep order
NGPU=${NGPU:-8}
CVD=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPU - 1)))}
FREE_MEM_MIB=${FREE_MEM_MIB:-2000}       # a GPU counts "free" when used-mem < this
STABLE=${STABLE:-3}                      # require this many consecutive all-free polls (avoid teardown races)
POLL=${POLL:-60}                         # seconds between polls

SWEEP_TS=$(date +%Y%m%d_%H%M%S)
SWEEP_BASE=${SWEEP_BASE:-/cb/ml-eng/aarti/dsa/phase2_topk_sweep_${SWEEP_TS}}
mkdir -p "${SWEEP_BASE}"
MASTER=${MASTER:-${SWEEP_BASE}/sweep-${SWEEP_TS}.log}
exec > >(tee -a "${MASTER}") 2>&1
echo "[topk-sweep] order=[${TOPKS}] gpus=${CVD} NGPU=${NGPU} SWEEP_BASE=${SWEEP_BASE} host=$(hostname) date=$(date -Is)"

wait_for_gpus() {
    local ok=0 free total
    while :; do
        free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk -v t="${FREE_MEM_MIB}" 'BEGIN{c=0} $1<t{c++} END{print c}')
        total=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        if [ "${free}" -ge "${NGPU}" ]; then ok=$((ok + 1)); else ok=0; fi
        echo "[topk-sweep] gpu-wait $(date +%H:%M:%S): ${free}/${total} free (need ${NGPU}), stable ${ok}/${STABLE}"
        [ "${ok}" -ge "${STABLE}" ] && break
        sleep "${POLL}"
    done
}

for TOPK in ${TOPKS}; do
    echo "[topk-sweep] ===== waiting for ${NGPU} free GPUs before TOPK=${TOPK} ====="
    wait_for_gpus
    echo "[topk-sweep] ===== launching FULL run TOPK=${TOPK} ($(date -Is)) ====="
    CUDA_VISIBLE_DEVICES="${CVD}" NPROC="${NGPU}" TOPK="${TOPK}" DOMAINS=full RUN_BASE="${SWEEP_BASE}" \
        bash "${REPO_ROOT}/examples/dsa/run_phase2_long_pipeline.sh" \
        && echo "[topk-sweep] TOPK=${TOPK} DONE ($(date -Is))" \
        || echo "[topk-sweep] TOPK=${TOPK} FAILED (continuing to next) ($(date -Is))"
done
echo "[topk-sweep] ===== all runs finished. per-run logs + checkpoints under ${SWEEP_BASE} ====="
