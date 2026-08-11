#!/usr/bin/env bash
# gpt-oss-20b Phase-2b full run, leg 2: reasoning_effort=high at a 64K window.
#
#   WAIT_PID=<leg1 pid> nohup ./examples/msa/run_gptoss_ph2_full_high64k.sh >> ~/msa/data/gptoss_full_both.log 2>&1 &
#
# WHY 64K HERE AND 32K FOR MEDIUM. The 100-prompt pilots (2026-08-09) measured, on the same prompts with
# the same pin and seed, only reasoning_effort differing:
#
#                  truncated @32K      response p50
#     Math         medium 0/25         medium  7,248        high 13/25 (52%)   high 32,534
#     ALL          medium 1/100 (1%)   medium  2,530        high 21/100 (21%)  high  7,987
#
# At `high` the MEDIAN Math trace is 32,534 tokens -- the window, not the model, is what ends it. Run at
# 32K the high leg would drop half of Math and keep a non-random short tail, so its decode-long share
# (0.50 vs medium's 0.21) would be inflated by deleting exactly the rows that make the bucket. 64K is
# what makes the high leg measure the model instead of the cap.
#
# 64K is native: config.json max_position_embeddings=131072 (YaRN, factor 32 over 4096), so no rope
# override and no re-scaling. 12 of the 24 layers are `full_attention`; the other 12 are a 128-token
# sliding window and are unaffected by the window change either way.
#
# CONSEQUENCE FOR TRAINING, NOT HANDLED HERE: this leg's rows can be up to 65,536 tokens, so it is NOT
# drop-in for a 32K Phase-2 training config. Rows over the training window have to be dropped or the
# window raised -- a decision for the training run, deliberately not made in the data.
#
# SEQUENCING. Leg 1 (medium @32K) holds all 8 H200s. WAIT_PID blocks until that process exits, then this
# starts. Leg 1 was reparented to init when its driver was cancelled, so it is polled by liveness, not
# `wait`. Set WAIT_PID= (empty) to start immediately.
set -euo pipefail

export REPO_ROOT=${REPO_ROOT:-/home/aarti_cerebras/dsa/verl_msa}
BASE=/home/aarti_cerebras
WAIT_PID=${WAIT_PID:-}

cd "$REPO_ROOT"

if [[ -n "$WAIT_PID" ]]; then
    echo "# [high64k] waiting for leg 1 (pid $WAIT_PID) to finish -- $(date -Is)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "# [high64k] leg 1 exited -- $(date -Is)"
fi

# Belt and braces: leg 1's 8 vLLM replicas must actually be gone before 8 more claim the same cards.
for _i in $(seq 1 60); do
    _n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [[ "$_n" == "0" ]] && break
    echo "# [high64k] $_n compute process(es) still on the GPUs, waiting ($_i/60)"
    sleep 60
done

RUN="${BASE}/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_full93889_effhigh_L65536_20260809"
echo "############################################################"
echo "# LEG: reasoning_effort=high  window=65536"
echo "# RUN: ${RUN}"
echo "# started: $(date -Is)"
echo "############################################################"
_t0=$SECONDS
EFFORT=high WINDOW=65536 RUN="$RUN" ./examples/msa/gen_gptoss_ph2.sh full
echo "# LEG high@64K DONE in $(( (SECONDS - _t0) / 60 )) min -> ${RUN}"
echo "ALL LEGS DONE $(date -Is)"
