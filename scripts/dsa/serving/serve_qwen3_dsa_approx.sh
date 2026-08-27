#!/usr/bin/env bash
# Isolated vLLM OpenAI server for Qwen3-4B DSA approximate selection.
# docs/qwen3_4b_dsa/serving_eval_plan.md §2. Modelled on scripts/msa/serving/serve_msa.sh.
#
# Served under the BASELINE's model name by default, so the existing OpenCompass configs at
# /cb/ml-eng/aarti/dsa/evals/qwen3-4b-thinking run against it unchanged.
#
# SERVING_DIR must have been created by build_qwen3_dsa_approx_serving_dir.py.
set -euo pipefail

REPO=${REPO:-/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl}
VENV=${VENV:-$REPO/.devlibs/vllm026}
PY=$VENV/bin/python

# flashinfer JIT-compiles on first use and shells out to `ninja` BY NAME, so $VENV/bin must be on
# PATH even though we invoke $PY directly. Without it, on a node with no warm flashinfer cache every
# EngineCore dies with FileNotFoundError: 'ninja' ~100 lines before the visible error, reading as a
# memory/model problem. (Cost the MSA bring-up all 8 servers on one node.)
export PATH="$VENV/bin:$PATH"

SERVING_DIR=${SERVING_DIR:?set SERVING_DIR to an isolated approximate serving directory}
GPU="${GPU:-0}"
PORT="${PORT:-$((8000 + GPU))}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SERVED_NAME="${SERVED_NAME:-Qwen3-4B-Thinking-2507}"
EAGER="${EAGER:-1}"
LOGDIR=${LOGDIR:-/cb/ml-eng/aarti/dsa_qwen3/serving/logs}; mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOGDIR/${TS}_serve_dsa_approx_gpu${GPU}_p${PORT}.log
MANIFEST=$LOGDIR/${TS}_serve_dsa_approx_gpu${GPU}_p${PORT}.manifest
# Written by GET /selector/telemetry, not on shutdown -- see the --worker-extension-cls note below.
export DSA_SELECTOR_ARTIFACT="${DSA_SELECTOR_ARTIFACT:-$LOGDIR/${TS}_selector_gpu${GPU}_p${PORT}.json}"

export CUDA_VISIBLE_DEVICES="$GPU"
# The isolated bootstrap registers only the approximate architecture and its selector hooks.
# in the EngineCore SUBPROCESS, which does not inherit the parent's imports.
export PYTHONPATH="$REPO/scripts/dsa/serving/_pluginboot_approx:$REPO"
export DSA_SPARSE="${DSA_SPARSE:-1}"
# Controls/diagnostics: exported explicitly so they reach the EngineCore subprocess even when the
# caller sets them as one-shot assignments (`DSA_RANDOM_INDEXER=1 bash serve_qwen3_dsa.sh`).
export DSA_RANDOM_INDEXER="${DSA_RANDOM_INDEXER:-0}"
export DSA_DEBUG_SELECTION="${DSA_DEBUG_SELECTION:-0}"
export VLLM_NO_USAGE_STATS=1

CMD=("$PY" "$REPO/scripts/dsa/serving/serve_qwen3_dsa_approx_entry.py"
  --model "$SERVING_DIR"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --tensor-parallel-size 1
  --block-size 64            # mandatory: the sparse backend reports [64] only, and the indexer's
                             # paged-logits kernel plus the top-k -> global-slot conversion assume it
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --dtype bfloat16
  --worker-extension-cls scripts.dsa.selector_telemetry_rpc.SelectorTelemetryExtension
                             # Telemetry accumulates in the EngineCore SUBPROCESS, which vLLM
                             # terminates rather than exiting cleanly, so the atexit dump inside
                             # SelectorRuntime.configure never runs. This extension gives the API
                             # server a method to pull it with; the entry script exposes it at
                             # GET /selector/telemetry (and POST /selector/telemetry/reset).
  --seed "${SEED:-1234}"     # baseline parity: every baseline manifest recorded --seed 1234
  --no-enable-prefix-caching # baseline parity: all baseline manifests show prefix_cache: 0. Also
                             # semantically important here -- a cache hit skips the prefill that
                             # would have populated the indexer's side cache for those tokens.
  ${EXTRA_ARGS:-})
if [[ "$EAGER" != "0" ]]; then
  # Eager IS the default here (EAGER=1 above), unlike serve_qwen3_dsa.sh which defaults to 0. The
  # comment that used to sit here was copied from that script and claimed cudagraphs were on; it
  # was wrong for this one.
  #
  # Active approximate selectors support CUDA graphs with dsa_telemetry=off, graph_safety, or
  # graph_verify_exact. graph_safety records lightweight fixed-address safety counters;
  # graph_verify_exact retains host-folded eager prefill telemetry and records persistent decode
  # quality moments/histograms. summary/verify_exact remain entirely host-folded and are rejected
  # in graph mode rather than writing an incomplete artifact.
  #
  # Set EAGER=0 for a `topk` serving dir, or a radix serving dir built with --telemetry off or
  # --telemetry graph_safety, or a radix serving dir built with --telemetry graph_verify_exact.
  CMD+=(--enforce-eager)
fi

{
  echo "=== SERVE-QWEN3-DSA-APPROX MANIFEST ==="
  echo "timestamp:   $TS"
  echo "date:        $(date -Is)"
  echo "hostname:    $(hostname)"
  echo "verl_git:    $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo NA) (+$(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) dirty)"
  echo "venv:        $VENV  (vllm $("$PY" -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo NA))"
  echo "model_dir:   $SERVING_DIR"
  echo "index_topk:  $(python3 -c "import json;print(json.load(open('$SERVING_DIR/config.json')).get('index_topk'))" 2>/dev/null || echo NA)"
  echo "selector:    $(python3 -c "import json; c=json.load(open('$SERVING_DIR/config.json')); print(c.get('dsa_selector'), c.get('dsa_selector_backend'), c.get('dsa_telemetry'))" 2>/dev/null || echo NA)"
  echo "speed_claim: false (dsa_csx_reference invokes exact top-k internally)"
  echo "telemetry:   $DSA_SELECTOR_ARTIFACT"
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
echo "$PID" > "$LOGDIR/server_approx_gpu${GPU}.pid"
echo "[serve_dsa_approx] pid=$PID gpu=$GPU port=$PORT"
echo "[serve_dsa_approx] tail -f $LOG   (wait for 'Application startup complete')"
echo "[serve_dsa_approx] verify selector config in the Qwen3DSA-approx startup line"
