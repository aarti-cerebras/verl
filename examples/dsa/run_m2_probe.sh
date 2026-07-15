#!/usr/bin/env bash
# DSA Phase-2 — M2 GPU probe (see docs/dsa_phase2_plan.md, milestone M2).
#
# Measures MiniCPM3-4B's OWN per-domain response-length distribution by self-generating trajectories from a
# small per-domain prompt sample (UltraData-SFT-2605/no_think), at temperature 0.7 with per-domain
# max_new_tokens. The output calibrates the per-domain caps + selection weights before the full M3a run.
#
# The SAME scripts drive the full M3a corpus later — just raise PER_DOMAIN / drop --limit.
#
# Everything is logged: this wrapper writes a MASTER log (the exact command it was invoked with, the
# resolved env, nvidia-smi, git commit, library versions) and tees each step; each python step ALSO writes
# its own timestamped log via scripts/dsa/_dsa_log.py (which records its exact argv).
#
# Usage:
#   NPROC=8 PER_DOMAIN=2000 examples/dsa/run_m2_probe.sh
#   BACKEND=hf PER_DOMAIN=500 examples/dsa/run_m2_probe.sh          # transformers fallback
#   MODEL=/path/to/MiniCPM3-4B examples/dsa/run_m2_probe.sh          # local snapshot
#
# Prereq: HF token with access to the GATED openbmb/UltraData-SFT-2605
#   (~/.cache/huggingface/token or HF_TOKEN). Run in an env with vllm+torch (verl container) for BACKEND=vllm.
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

# --- knobs (env-overridable) ---
MODEL=${MODEL:-openbmb/MiniCPM3-4B}
BACKEND=${BACKEND:-vllm}                 # vllm | hf
# --- prompt selection: split spec (M3a) OR uniform per-domain (probe) ---
SPLIT_JSON=${SPLIT_JSON:-}               # path/JSON of per-config {frac|count, lang}; e.g. examples/dsa/m3a_split.json
TOTAL=${TOTAL:-}                         # total prompts (required when SPLIT_JSON uses 'frac')
DOMAINS=${DOMAINS:-"Math Code IF Knowledge Chinese-general"}   # uniform-mode configs
PER_DOMAIN=${PER_DOMAIN:-2000}           # uniform-mode prompts/config (probe: 2000)
MAX_SHARDS=${MAX_SHARDS:-20}             # adaptive shard-scan cap per config
SHARDS_PER_DOMAIN=${SHARDS_PER_DOMAIN:-1}
LIMIT=${LIMIT:-0}                        # cap total prompts sent to gen (0 = all selected)
# --- generation parallelism: DP replicas x TP GPUs (DP=8, TP=1 optimal for a 4B model) ---
DP=${DP:-8}                              # data_parallel_size (independent replicas, one per GPU slice)
TP=${TP:-1}                              # tensor_parallel_size per replica
TEMP=${TEMP:-0.7}
TOP_P=${TOP_P:-0.9}
SEED=${SEED:-1234}
N=${N:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}         # empty => derived from caps
MAX_NEW_TOKENS_JSON=${MAX_NEW_TOKENS_JSON:-}   # e.g. '{"Math":8192}' to override caps
HF_BATCH_SIZE=${HF_BATCH_SIZE:-16}
# transformers 4.57.1 staged in-repo shadows the container's (MiniCPM3 trust-remote-code needs it).
# NOTE: if vLLM conflicts with 4.57.1, unset DEVLIBS to fall back to the container's transformers.
DEVLIBS=${DEVLIBS:-${REPO_ROOT}/.devlibs/tf457lib}
export PYTHONPATH="${DEVLIBS}:${PYTHONPATH:-}"

RUN_TS=$(date -u +%Y%m%d_%H%M%S)
# All data artifacts (run outputs, prompts, trajectories, logs, gated-dataset shard downloads) live here.
DATA_ROOT=${DATA_ROOT:-/cb/ml-eng/aarti/dsa}
HF_CACHE=${HF_CACHE:-${DATA_ROOT}/hf_cache}          # UltraData-SFT-2605 shard downloads
RUN_DIR=${RUN_DIR:-${DATA_ROOT}/m2_probe_${RUN_TS}}
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "$DATA_ROOT" "$HF_CACHE"
mkdir -p "$LOG_DIR"
MASTER_LOG="${RUN_DIR}/master.log"
PROMPTS_JSONL="${RUN_DIR}/prompts.jsonl"
TRAJ_JSONL="${RUN_DIR}/trajectories.jsonl"
REPORT_JSON="${RUN_DIR}/length_report.json"

PY=${PY:-python3}

# --- master log: exact invocation + resolved env + system snapshot ---
{
  echo "==================== M2 PROBE $(date -u) ===================="
  echo "WRAPPER CMD: $0 $*"
  echo "RUN_DIR=$RUN_DIR"
  echo "--- resolved config ---"
  for v in DATA_ROOT HF_CACHE RUN_DIR MODEL BACKEND SPLIT_JSON TOTAL DOMAINS PER_DOMAIN MAX_SHARDS \
           SHARDS_PER_DOMAIN LIMIT DP TP TEMP TOP_P SEED N GPU_MEM_UTIL MAX_MODEL_LEN MAX_NEW_TOKENS_JSON \
           HF_BATCH_SIZE DEVLIBS PYTHONPATH PY; do
    echo "  $v=${!v}"
  done
  echo "--- host / git ---"
  hostname; git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || true
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "--- nvidia-smi ---"; nvidia-smi 2>&1 | head -20 || echo "no nvidia-smi"
  echo "--- library versions ---"
  $PY - <<'PYV' 2>&1 || true
for m in ("torch","vllm","transformers","huggingface_hub"):
    try:
        mod=__import__(m); print(m, getattr(mod,"__version__",""))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
PYV
  echo "============================================================"
} | tee "$MASTER_LOG"

run() {  # log the exact command, run it, tee to master log, preserve exit code
  echo "+ $*" | tee -a "$MASTER_LOG"
  "$@" 2>&1 | tee -a "$MASTER_LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "STEP FAILED (rc=$rc): $*" | tee -a "$MASTER_LOG"; exit "$rc"; }
}

# --- Step 1: select prompts (CPU; needs gated-repo HF token) ---
SELECT_ARGS=(scripts/dsa/select_prompts.py --out "$PROMPTS_JSONL" --log-dir "$LOG_DIR"
  --local-dir "$HF_CACHE" --seed "$SEED" --max-shards-per-config "$MAX_SHARDS")
if [ -n "$SPLIT_JSON" ]; then
  SELECT_ARGS+=(--split-json "$SPLIT_JSON")
  [ -n "$TOTAL" ] && SELECT_ARGS+=(--total "$TOTAL")
else
  SELECT_ARGS+=(--domains $DOMAINS --per-domain "$PER_DOMAIN" --shards-per-domain "$SHARDS_PER_DOMAIN")
fi
run "$PY" "${SELECT_ARGS[@]}"

# --- Step 2: generate trajectories (GPU) ---
GEN_ARGS=(scripts/dsa/gen_trajectories.py
  --prompts "$PROMPTS_JSONL" --out "$TRAJ_JSONL" --log-dir "$LOG_DIR"
  --model "$MODEL" --backend "$BACKEND"
  --temperature "$TEMP" --top-p "$TOP_P" --n "$N" --seed "$SEED"
  --data-parallel-size "$DP" --tensor-parallel-size "$TP" --gpu-memory-utilization "$GPU_MEM_UTIL"
  --hf-batch-size "$HF_BATCH_SIZE")
[ -n "$MAX_MODEL_LEN" ] && GEN_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "$MAX_NEW_TOKENS_JSON" ] && GEN_ARGS+=(--max-new-tokens-json "$MAX_NEW_TOKENS_JSON")
[ "$LIMIT" -gt 0 ] && GEN_ARGS+=(--limit "$LIMIT")
run "$PY" "${GEN_ARGS[@]}"

# --- Step 3: analyze lengths ---
run "$PY" scripts/dsa/analyze_lengths.py \
  --trajectories "$TRAJ_JSONL" --out-report "$REPORT_JSON" --log-dir "$LOG_DIR" --tokenizer "$MODEL"

echo "M2 PROBE COMPLETE. Artifacts in $RUN_DIR" | tee -a "$MASTER_LOG"
echo "  prompts:      $PROMPTS_JSONL" | tee -a "$MASTER_LOG"
echo "  trajectories: $TRAJ_JSONL" | tee -a "$MASTER_LOG"
echo "  length report:$REPORT_JSON" | tee -a "$MASTER_LOG"
echo "  logs:         $LOG_DIR/  (+ master.log)" | tee -a "$MASTER_LOG"
