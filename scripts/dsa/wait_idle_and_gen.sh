#!/usr/bin/env bash
# DSA — wait until enough GPUs are TRULY FREE (idle utilization AND near-zero memory used) continuously for
# IDLE_MIN minutes, then launch trajectory generation on the free subset + length analysis.
#
# A GPU counts as free only if:  utilization.gpu <= UTIL_THRESH  AND  memory.used <= MEM_FREE_MiB.
# (util alone is not enough on this SHARED host — GPUs sit at 0% while holding 20+ GB reserved by other
# tenants, which caused the earlier vLLM OOM. Requiring near-zero memory guarantees the launch fits.)
#
# When >= MIN_FREE_GPUS are free for the window, it launches DP generation on EXACTLY those GPU indices
# (CUDA_VISIBLE_DEVICES = the free set), with gpu_memory_utilization sized from their free memory.
#
# Run DETACHED (setsid nohup). Knobs (env):
#   IDLE_MIN=30 POLL_SEC=60 UTIL_THRESH=0 MEM_FREE_MiB=2048 MIN_FREE_GPUS=8 TP=1 MEM_MARGIN_GB=6 MAX_WAIT_HOURS=72
set -u

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

RUN_DIR=${RUN_DIR:?set RUN_DIR to the run dir containing prompts.jsonl}
PROMPTS=${PROMPTS:-$RUN_DIR/prompts.jsonl}
OUT=${OUT:-$RUN_DIR/trajectories.jsonl}
REPORT=${REPORT:-$RUN_DIR/length_report.json}
LOG_DIR=${LOG_DIR:-$RUN_DIR/logs}
MODEL=${MODEL:-openbmb/MiniCPM3-4B}
TP=${TP:-1}
TEMP=${TEMP:-0.7}; TOP_P=${TOP_P:-0.9}; SEED=${SEED:-1234}; N=${N:-1}
IDLE_MIN=${IDLE_MIN:-30}            # required continuous free window
POLL_SEC=${POLL_SEC:-60}
UTIL_THRESH=${UTIL_THRESH:-0}       # GPU idle if utilization.gpu <= this (%)
MEM_FREE_MiB=${MEM_FREE_MiB:-2048}  # GPU counts as free if memory.used <= this (MiB)
MIN_FREE_GPUS=${MIN_FREE_GPUS:-8}   # trigger when at least this many GPUs are free
MEM_MARGIN_GB=${MEM_MARGIN_GB:-6}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-72}
DEVLIBS=${DEVLIBS:-$REPO_ROOT/.devlibs/tf457lib}
export PYTHONPATH="${DEVLIBS}:${PYTHONPATH:-}"
PY=${PY:-python3}

mkdir -p "$LOG_DIR"
WLOG="$LOG_DIR/wait_idle_and_gen_$(date -u +%Y%m%d_%H%M%S).log"
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$WLOG"; }

log "==================== wait_idle_and_gen ===================="
log "CMD: $0 $*  host=$(hostname) git=$(git rev-parse --short HEAD 2>/dev/null || echo NA)"
log "cfg: RUN_DIR=$RUN_DIR TP=$TP IDLE_MIN=$IDLE_MIN POLL_SEC=$POLL_SEC UTIL_THRESH=${UTIL_THRESH}% MEM_FREE_MiB=$MEM_FREE_MiB MIN_FREE_GPUS=$MIN_FREE_GPUS"
[ -s "$PROMPTS" ] || { log "ERROR: prompts file missing/empty: $PROMPTS"; exit 1; }

# echo one integer per free GPU index (util<=thresh AND mem_used<=MEM_FREE_MiB)
free_gpus(){
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v u="$UTIL_THRESH" -v m="$MEM_FREE_MiB" '{gsub(/ /,"");
        if ($2+0<=u && $3+0<=m) print $1}'
}

need=$(( IDLE_MIN * 60 / POLL_SEC ))
max_polls=$(( MAX_WAIT_HOURS * 3600 / POLL_SEC ))
streak=0; polls=0
log "waiting for >= ${MIN_FREE_GPUS} GPUs free (util<=${UTIL_THRESH}% AND mem<=${MEM_FREE_MiB}MiB) for ${IDLE_MIN}min (${need} polls @ ${POLL_SEC}s)"
while true; do
  mapfile -t FREE < <(free_gpus)
  nfree=${#FREE[@]}
  if [ "$nfree" -ge "$MIN_FREE_GPUS" ]; then streak=$((streak+1)); else streak=0; fi
  polls=$((polls+1))
  log "poll ${polls}: free=${nfree} [${FREE[*]:-}]  streak=${streak}/${need}"
  if [ "$streak" -ge "$need" ]; then log ">>> FREE CONDITION MET"; break; fi
  if [ "$polls" -ge "$max_polls" ]; then log "!!! MAX_WAIT_HOURS=${MAX_WAIT_HOURS} exceeded; giving up"; exit 2; fi
  sleep "$POLL_SEC"
done

# Re-read the free set right before launch (it may have shifted); require it still meets the bar.
mapfile -t FREE < <(free_gpus)
nfree=${#FREE[@]}
if [ "$nfree" -lt "$MIN_FREE_GPUS" ]; then
  log "free set shrank to ${nfree} at launch time (<${MIN_FREE_GPUS}); NOT launching. Re-run to keep waiting."
  exit 3
fi
CVD=$(IFS=,; echo "${FREE[*]}")
DP=$nfree
export CUDA_VISIBLE_DEVICES="$CVD"
minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CVD" 2>/dev/null | sort -n | head -1)
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
util=$("$PY" -c "mf=float('${minfree:-0}'); tot=float('${total:-1}'); m=${MEM_MARGIN_GB}*1024.0; print(round(max(0.30, min(0.90, (mf-m)/tot)),2))")
log "launch: free GPUs [$CVD] DP=$DP TP=$TP min_free=${minfree}MiB -> gpu_memory_utilization=${util}"

"$PY" scripts/dsa/gen_trajectories.py --prompts "$PROMPTS" --out "$OUT" --log-dir "$LOG_DIR" \
  --model "$MODEL" --backend vllm --temperature "$TEMP" --top-p "$TOP_P" --n "$N" --seed "$SEED" \
  --data-parallel-size "$DP" --tensor-parallel-size "$TP" --gpu-memory-utilization "$util" 2>&1 | tee -a "$WLOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then log "!!! GEN FAILED rc=$rc"; exit "$rc"; fi

log "generation done; analyzing lengths"
"$PY" scripts/dsa/analyze_lengths.py --trajectories "$OUT" --out-report "$REPORT" \
  --log-dir "$LOG_DIR" --tokenizer "$MODEL" 2>&1 | tee -a "$WLOG"
log ">>> ALL DONE. trajectories=$OUT report=$REPORT"
