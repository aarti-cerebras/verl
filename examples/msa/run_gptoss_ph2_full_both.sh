#!/usr/bin/env bash
# Both gpt-oss-20b Phase-2b full runs, back to back: reasoning_effort=medium, then =high.
#
#   nohup ./examples/msa/run_gptoss_ph2_full_both.sh > ~/msa/data/gptoss_full_both.log 2>&1 &
#
# SEQUENTIAL, not parallel: each leg wants all 8 H200s at --data-parallel-size 8. Running them
# concurrently at DP=4 each would halve concurrency per replica and change the throughput regime, so the
# two legs' timings would no longer be comparable to each other or to the Qwen3 run.
#
# RESUMABLE. The RUN directories are FIXED NAMES, not $(date)-stamped. Re-running this script after an
# interruption re-enters the same directories, and gen_trajectories appends to the .partN files and skips
# prompts already present -- so a killed leg continues where it stopped instead of forking a new dataset.
# This is the whole reason the date is pinned (doc §4): a leg resumed tomorrow still serves the
# 2026-08-09 system prompt, so the parquet cannot end up with two different prefixes in it.
#
# The medium leg is NOT skipped if it already finished -- gen_trajectories will find all 93,889 prompts
# present and fall through to the splice/verify/split stages, which are idempotent. To force a rerun,
# delete or rename the RUN directory.
set -euo pipefail

export REPO_ROOT=${REPO_ROOT:-/home/aarti_cerebras/dsa/verl_msa}
BASE=/home/aarti_cerebras
STAMP=20260809

cd "$REPO_ROOT"

for EFF in medium high; do
    RUN="${BASE}/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_full93889_eff${EFF}_L32768_${STAMP}"
    echo "############################################################"
    echo "# LEG: reasoning_effort=${EFF}"
    echo "# RUN: ${RUN}"
    echo "# started: $(date -Is)"
    echo "############################################################"
    _t0=$SECONDS
    EFFORT="$EFF" RUN="$RUN" ./examples/msa/gen_gptoss_ph2.sh full
    echo "# LEG ${EFF} DONE in $(( (SECONDS - _t0) / 60 )) min -> ${RUN}"
done

echo "ALL LEGS DONE $(date -Is)"
