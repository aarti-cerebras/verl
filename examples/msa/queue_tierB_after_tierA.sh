#!/usr/bin/env bash
# Queue the Tier B (ChatQA2, CC-BY-NC) generation to start once Tier A finishes.
#
# Runbook: docs/qwen3_4b_msa/phase2_long_context_gen.md §2.2, §5.4.
#
# This is a GATED queue, not a blind chain. It refuses to start Tier B unless Tier A actually
# SUCCEEDED, because the failure mode we care about is Tier A dying part-way (which is resumable) and
# Tier B then trampling the node before anyone notices. Three preconditions:
#
#   1. the Tier A launcher process is gone;
#   2. Tier A wrote >= MIN_ROWS rows AND every DP replica logged its "done:" line;
#   3. the GPUs are actually free -- an orphaned VLLM::EngineCore survives a kill of the launcher and
#      holds ~130 GB per GPU, which would make Tier B die at engine init (verified the hard way,
#      §9 B / B9). We poll for release and ABORT rather than launch into a busy node.
#
# Tier B differs from Tier A in two load-bearing ways, both already baked into the prompt pool:
#   * window 163,840 (not 131,072) -- ChatQA2 docs were truncated to 131,072 *Llama-3* tokens upstream
#   * max_num_seqs 5 (not 12) -- KV is 22.5 GiB/seq at that window, ~5 per H200
set -uo pipefail

REPO=${REPO:-/home/aarti_cerebras/dsa/verl_msa}
MODEL=${MODEL:-/home/aarti_cerebras/models/qwen3_4b_thinking_2507}
FULL_A=$(cat /tmp/full_run)
POOL_B=$(cat /tmp/tierB_run)
QLOG=${QLOG:-${POOL_B}/logs/queue.log}
MIN_ROWS=${MIN_ROWS:-43000}          # of 43,524; allows for the 49 skipped-nofit rows and a small margin
GPU_FREE_MB=${GPU_FREE_MB:-2000}
GPU_WAIT_S=${GPU_WAIT_S:-900}

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$QLOG"; }
mkdir -p "$(dirname "$QLOG")"
log "queue armed: waiting for Tier A launcher to exit (pool_b=$POOL_B)"

# --- 1. wait for the Tier A launcher --------------------------------------------------------------
while pgrep -f "gen_trajectories.py --prompts ${FULL_A%/full_gen}/prompts.jsonl" > /dev/null; do sleep 60; done
log "Tier A launcher exited"

# --- 2. did it SUCCEED? ---------------------------------------------------------------------------
ROWS=$(cat "$FULL_A"/trajectories.jsonl.part* 2>/dev/null | wc -l)
# NOTE: `grep -c` prints the count AND exits 1 when it is zero, so `|| echo 0` would append a SECOND
# zero and produce "0\n0" -- which breaks the numeric test below. There is no `set -e` here, so a
# non-zero grep exit is harmless; just take the count.
DONE=$(grep -c "done: [0-9]* new trajectories" "$FULL_A/logs/stdout.log" 2>/dev/null); DONE=${DONE:-0}
PREEMPT=$(grep -c "PREEMPTION DETECTED" "$FULL_A/logs/stdout.log" 2>/dev/null); PREEMPT=${PREEMPT:-0}
log "Tier A: rows=$ROWS replicas_done=$DONE preemption_hits=$PREEMPT"
if [ "$ROWS" -lt "$MIN_ROWS" ] || [ "$DONE" -lt 8 ]; then
    log "ABORT: Tier A incomplete (rows=$ROWS < $MIN_ROWS or replicas_done=$DONE < 8). Tier B NOT started."
    log "       Tier A is resumable: relaunch with the same --out to continue by prompt_sha256."
    exit 1
fi
[ "$PREEMPT" -gt 0 ] && log "WARNING: Tier A reported preemption; consider lowering --max-num-seqs for Tier B"

# --- 3. are the GPUs actually free? ---------------------------------------------------------------
waited=0
while :; do
    BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk -v t="$GPU_FREE_MB" '$1>t' | wc -l)
    [ "$BUSY" -eq 0 ] && break
    if [ "$waited" -ge "$GPU_WAIT_S" ]; then
        log "ABORT: $BUSY GPU(s) still hold >${GPU_FREE_MB}MB after ${GPU_WAIT_S}s."
        log "       Likely an orphaned VLLM::EngineCore. Kill the process GROUP, not the launcher."
        exit 1
    fi
    sleep 30; waited=$((waited+30))
done
log "GPUs free after ${waited}s"

# --- 4. launch Tier B -----------------------------------------------------------------------------
RUNB="$POOL_B/full_gen"
mkdir -p "$RUNB/logs"
cd "$REPO"
git rev-parse HEAD > "$RUNB/git_sha.txt"; git diff > "$RUNB/git_diff.patch"
cp "$0" "$RUNB/queued_by.sh"
python3 - "$RUNB" "$POOL_B" <<'PY'
import json, os, socket, subprocess, sys, time
run, pool = sys.argv[1], sys.argv[2]
json.dump({"kind": "msa_phase2b_longctx_generation", "tier": "B", "mode": "full",
 "licence": "CC-BY-NC-2.0 (NONCOMMERCIAL) — keep segregated from Tier A",
 "prompt_pool": os.path.join(pool, "prompts.jsonl"), "n_prompts": 30147,
 "generator_model": "/home/aarti_cerebras/models/qwen3_4b_thinking_2507",
 "window": 163840, "min_gen_budget": 8192,
 "why_window": "ChatQA2 docs truncated to 131,072 LLAMA-3 tokens upstream; Qwen3 retokenization "
               "reaches 136,833. Rows can exceed 131,072 TOTAL — NOT drop-in at a 128K training window.",
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0, "n": 1, "seed": 1234},
 "parallelism": {"dp": 8, "tp": 1, "max_num_seqs": 5, "gpu_memory_utilization": 0.90},
 "queued_after": "Tier A full generation",
 "runbook": "docs/qwen3_4b_msa/phase2_long_context_gen.md",
 "host": socket.gethostname(),
 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
 "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(os.path.join(run, "MANIFEST.json"), "w"), indent=1)
PY

log "launching Tier B -> $RUNB"
setsid python3 scripts/dsa/gen_trajectories.py \
    --prompts "$POOL_B/prompts.jsonl" --out "$RUNB/trajectories.jsonl" --log-dir "$RUNB/logs" \
    --model "$MODEL" \
    --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
    --max-model-len 163840 --fit-window 163840 --min-gen-budget 8192 \
    --data-parallel-size 8 --tensor-parallel-size 1 --max-num-seqs 5 \
    --gpu-memory-utilization 0.90 --chunk-size 128 --seed 1234 --no-merge \
    > "$RUNB/logs/stdout.log" 2>&1 &
log "Tier B launched (pid $!)"
