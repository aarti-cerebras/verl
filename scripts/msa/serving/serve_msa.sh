#!/usr/bin/env bash
# Durable vLLM OpenAI server for the Qwen3-4B MSA sparse model.
# docs/qwen3_4b_msa/serving_plan.md §7. Modelled on the DSA harness's serve_dsa.sh.
#
# Served under the BASELINE's model name by default, so the existing OpenCompass configs at
# /cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking run against it unchanged.
#
# Env: GPU (0) · PORT (8000+GPU) · SERVING_DIR · MAX_LEN (131072) · GPU_MEM_UTIL (0.85)
#      MSA_SPARSE (1) · SERVED_NAME (Qwen3-4B-Thinking-2507)
set -euo pipefail

REPO=${REPO:-/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl}
VENV=${VENV:-$REPO/.devlibs/vllm026}
PY=$VENV/bin/python
SERVING_DIR=${SERVING_DIR:-/cb/ml-eng/aarti/msa/serving/k8_step1400}

GPU="${GPU:-0}"
PORT="${PORT:-$((8000 + GPU))}"
MAX_LEN="${MAX_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SERVED_NAME="${SERVED_NAME:-Qwen3-4B-Thinking-2507}"
LOGDIR=${LOGDIR:-/cb/ml-eng/aarti/msa/serving/logs}; mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOGDIR/${TS}_serve_msa_gpu${GPU}_p${PORT}.log
MANIFEST=$LOGDIR/${TS}_serve_msa_gpu${GPU}_p${PORT}.manifest

export CUDA_VISIBLE_DEVICES="$GPU"
# _pluginboot FIRST: its sitecustomize registers the plugin in the EngineCore subprocess too,
# which does NOT inherit the parent's imports (serving_plan §7).
export PYTHONPATH="$REPO/scripts/msa/serving/_pluginboot:$REPO"
export MSA_SPARSE="${MSA_SPARSE:-1}"
export VLLM_NO_USAGE_STATS=1

CMD=("$PY" "$REPO/scripts/msa/serving/serve_msa_entry.py"
  --model "$SERVING_DIR"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --tensor-parallel-size 1
  --block-size 128          # mandatory: both MSA backends report [128] only
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --dtype bfloat16
  --enforce-eager           # R2 (cudagraph capture) unresolved; revisit in P5
  ${EXTRA_ARGS:-})          # extra flags, e.g. --no-enable-prefix-caching

{
  echo "=== SERVE-MSA MANIFEST ==="
  echo "timestamp:   $TS"
  echo "date:        $(date -Is)"
  echo "hostname:    $(hostname)"
  echo "verl_git:    $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NA) (+$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) dirty)"
  echo "venv:        $VENV  (vllm $("$PY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo NA))"
  echo "model_dir:   $SERVING_DIR"
  echo "served_name: $SERVED_NAME"
  echo "PYTHONPATH:  $PYTHONPATH"
  echo "MSA_SPARSE:  $MSA_SPARSE"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "port:$PORT gpu_mem_util:$GPU_MEM_UTIL max_len:$MAX_LEN block_size:128"
  echo "command:     ${CMD[*]}"
  echo "log:         $LOG"
  echo "=========================="
} | tee "$MANIFEST"

setsid nohup "${CMD[@]}" </dev/null >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOGDIR/server_gpu${GPU}.pid"
echo "[serve_msa] pid=$PID gpu=$GPU port=$PORT"
echo "[serve_msa] tail -f $LOG   (wait for 'Application startup complete')"
