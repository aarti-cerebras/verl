#!/usr/bin/env bash
# Durable vLLM OpenAI server for the Qwen3-4B DSA sparse model.
# docs/qwen3_4b_dsa/serving_eval_plan.md §2. Modelled on scripts/msa/serving/serve_msa.sh.
#
# Served under the BASELINE's model name by default, so the existing OpenCompass configs at
# /cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking run against it unchanged.
#
# Env: GPU (0) · PORT (8000+GPU) · SERVING_DIR · MAX_LEN (32768) · GPU_MEM_UTIL (0.85)
#      DSA_SPARSE (1) · SERVED_NAME (Qwen3-4B-Thinking-2507) · EAGER (0)
set -euo pipefail

REPO=${REPO:-/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl}
VENV=${VENV:-$REPO/.devlibs/vllm026}
PY=$VENV/bin/python

# flashinfer JIT-compiles on first use and shells out to `ninja` BY NAME, so $VENV/bin must be on
# PATH even though we invoke $PY directly. Without it, on a node with no warm flashinfer cache every
# EngineCore dies with FileNotFoundError: 'ninja' ~100 lines before the visible error, reading as a
# memory/model problem. (Cost the MSA bring-up all 8 servers on one node.)
export PATH="$VENV/bin:$PATH"

SERVING_DIR=${SERVING_DIR:-/cb/ml-eng/aarti/dsa_qwen3/serving/p2_mix5050_k2048_step1200}
GPU="${GPU:-0}"
PORT="${PORT:-$((8000 + GPU))}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SERVED_NAME="${SERVED_NAME:-Qwen3-4B-Thinking-2507}"
EAGER="${EAGER:-0}"
LOGDIR=${LOGDIR:-/cb/ml-eng/aarti/dsa_qwen3/serving/logs}; mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOGDIR/${TS}_serve_dsa_gpu${GPU}_p${PORT}.log
MANIFEST=$LOGDIR/${TS}_serve_dsa_gpu${GPU}_p${PORT}.manifest

export CUDA_VISIBLE_DEVICES="$GPU"
# _pluginboot FIRST: its sitecustomize registers the architecture AND the CUSTOM attention backend
# in the EngineCore SUBPROCESS, which does not inherit the parent's imports.
export PYTHONPATH="$REPO/scripts/dsa/serving/_pluginboot:$REPO"
export DSA_SPARSE="${DSA_SPARSE:-1}"
# Controls/diagnostics: exported explicitly so they reach the EngineCore subprocess even when the
# caller sets them as one-shot assignments (`DSA_RANDOM_INDEXER=1 bash serve_qwen3_dsa.sh`).
export DSA_RANDOM_INDEXER="${DSA_RANDOM_INDEXER:-0}"
export DSA_DEBUG_SELECTION="${DSA_DEBUG_SELECTION:-0}"
export VLLM_NO_USAGE_STATS=1

CMD=("$PY" "$REPO/scripts/dsa/serving/serve_qwen3_dsa_entry.py"
  --model "$SERVING_DIR"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --tensor-parallel-size 1
  --block-size 64            # mandatory: the sparse backend reports [64] only, and the indexer's
                             # paged-logits kernel plus the top-k -> global-slot conversion assume it
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --dtype bfloat16
  --seed "${SEED:-1234}"     # baseline parity: every baseline manifest recorded --seed 1234
  --no-enable-prefix-caching # baseline parity: all baseline manifests show prefix_cache: 0. Also
                             # semantically important here -- a cache hit skips the prefill that
                             # would have populated the indexer's side cache for those tokens.
  ${EXTRA_ARGS:-})
if [[ "$EAGER" != "0" ]]; then
  # Escape hatch, no longer the default. Cudagraphs are ON because capture was VERIFIED correct,
  # not assumed: with FULL_AND_PIECEWISE and with PIECEWISE, the top_k >= T dense-equivalence
  # control reproduces the eager dense run TOKEN-EXACTLY (docs/qwen3_4b_dsa/serving_bringup_results.md
  # P3/P4), and the 21K needle still resolves. Eager costs 2.4x output throughput, so it is worth
  # having the evidence rather than the caution. Set EAGER=1 to bisect a suspected capture problem.
  CMD+=(--enforce-eager)
fi

{
  echo "=== SERVE-QWEN3-DSA MANIFEST ==="
  echo "timestamp:   $TS"
  echo "date:        $(date -Is)"
  echo "hostname:    $(hostname)"
  echo "verl_git:    $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NA) (+$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) dirty)"
  echo "venv:        $VENV  (vllm $("$PY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo NA))"
  echo "model_dir:   $SERVING_DIR"
  echo "index_topk:  $(python3 -c "import json;print(json.load(open('$SERVING_DIR/config.json')).get('index_topk'))" 2>/dev/null || echo NA)"
  echo "served_name: $SERVED_NAME"
  echo "PYTHONPATH:  $PYTHONPATH"
  echo "DSA_SPARSE:  $DSA_SPARSE   DSA_RANDOM_INDEXER: ${DSA_RANDOM_INDEXER:-0}"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "port:$PORT gpu_mem_util:$GPU_MEM_UTIL max_len:$MAX_LEN block_size:64 eager:$EAGER"
  echo "command:     ${CMD[*]}"
  echo "log:         $LOG"
  echo "================================"
} | tee "$MANIFEST"

setsid nohup "${CMD[@]}" </dev/null >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOGDIR/server_gpu${GPU}.pid"
echo "[serve_dsa] pid=$PID gpu=$GPU port=$PORT"
echo "[serve_dsa] tail -f $LOG   (wait for 'Application startup complete')"
echo "[serve_dsa] verify the sparse path is live:  grep -E 'DSA indexer decode path|CUSTOM' $LOG"
