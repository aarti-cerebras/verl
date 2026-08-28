#!/usr/bin/env bash
# Stage A/B/C GPU validation runner for the isolated Qwen3-4B DSA approximate selector.
# Run from a shell with /dev/nvidia* access. All artifacts stay below .agents/.
set -euo pipefail

REPO=${REPO:-/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl}
PY=${PY:-$REPO/.devlibs/vllm026/bin/python}
SOURCE=${SOURCE:-/cb/ml-eng/aarti/dsa_qwen3/serving/p2_mix5050_k2048_step1200}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$REPO/.agents/dsa_approx_gpu_validation/$STAMP}
PROMPT_TOKENS=${PROMPT_TOKENS:-6000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_TOKENS=${MAX_TOKENS:-32}
DECODE_BATCH_SIZE=${DECODE_BATCH_SIZE:-7}
GRAPH_TELEMETRY=${GRAPH_TELEMETRY:-graph_verify_exact}
TOP_LOGPROBS=${TOP_LOGPROBS:-20}
LOGPROB_MEAN_TOLERANCE=${LOGPROB_MEAN_TOLERANCE:-0.20}
LOGPROB_MAX_TOLERANCE=${LOGPROB_MAX_TOLERANCE:-1.0}
QUALITY_RELATIVE_TOLERANCE=${QUALITY_RELATIVE_TOLERANCE:-0.01}
QUALITY_ABSOLUTE_TOLERANCE=${QUALITY_ABSOLUTE_TOLERANCE:-0.001}

mkdir -p "$OUT/models" "$OUT/logs" "$OUT/results" "$OUT/telemetry"
exec > >(tee "$OUT/runner.log") 2>&1

echo "[runner] output=$OUT"
echo "[runner] source=$SOURCE"
echo "[runner] revision=$(git -C "$REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
test -e /dev/nvidia0
test -x "$PY"
test -f "$SOURCE/config.json"

build() {
  local name=$1 selector=$2 margin=$3 telemetry=${4:-verify_exact}
  local telemetry_args=(--telemetry "$telemetry")
  if [[ "$telemetry" != off ]]; then
    telemetry_args+=(--telemetry-artifact "$OUT/telemetry/$name.json")
  fi
  "$PY" "$REPO/scripts/dsa/build_qwen3_dsa_approx_serving_dir.py" \
    --source "$SOURCE" \
    --out "$OUT/models/$name" \
    --selector "$selector" \
    --selector-margin "$margin" \
    "${telemetry_args[@]}" \
    >"$OUT/logs/build_$name.log" 2>&1
}

build topk topk 0
build ceil radix_ceil 0
build ceil_graph radix_ceil 0 "$GRAPH_TELEMETRY"
build midpoint radix_midpoint 512
build floor radix_floor 2048
build floor_graph radix_floor 2048 "$GRAPH_TELEMETRY"

run_smoke() {
  local label=$1 gpu=$2 model=$3 mode=$4 bootstrap=$5 cg_mode=${6:-FULL_AND_PIECEWISE}
  local graph_args=()
  if [[ "$mode" == graph ]]; then
    graph_args=(--cudagraph --cudagraph-mode "$cg_mode")
  else
    graph_args=(--eager)
  fi
  echo "[runner] launch $label gpu=$gpu mode=$mode"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH="$REPO/scripts/dsa/serving/$bootstrap:$REPO" \
    DSA_SELECTOR_ARTIFACT="$OUT/telemetry/$label.json" \
    timeout 2400 "$PY" "$REPO/tests/dsa/qwen3_dsa_offline_smoke.py" \
      --model "$model" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-tokens "$MAX_TOKENS" \
      --decode-batch-size "$DECODE_BATCH_SIZE" \
      --prompt-tokens "$PROMPT_TOKENS" \
      --logprobs "$TOP_LOGPROBS" \
      --gpu-mem-util 0.80 \
      --out-json "$OUT/results/$label.json" \
      --label "$label" \
      "${graph_args[@]}" \
      >"$OUT/logs/$label.log" 2>&1
}

# One process per GPU. Cross-process greedy equality is reported but is not a hard gate: three
# fresh exact-plugin replicas produced both sides of a zero-margin token decision with the same
# seed. Centered logprobs on identical prefixes, selector semantics, and safety are hard gates.
run_smoke exact_eager 0 "$SOURCE" eager _pluginboot & p0=$!
run_smoke approx_topk_eager 1 "$OUT/models/topk" eager _pluginboot_approx & p1=$!
run_smoke approx_topk_graph 2 "$OUT/models/topk" graph _pluginboot_approx & p2=$!
run_smoke ceil_eager 3 "$OUT/models/ceil" eager _pluginboot_approx & p3=$!
# FULL_DECODE_ONLY, not FULL_AND_PIECEWISE. Measured 2026-08-26: with layer attribution moved off
# the traced path, PIECEWISE capture of an ACTIVE selector succeeds (51/51) and FULL then dies with
# "operation not permitted when stream is capturing" -- the host sync at selector_hooks.py:103
# (`.tolist()` on cu_seqlen_ks/ke to derive request boundaries), which CUDA graph capture cannot
# contain. dsa-csx scopes that same sync out of its capture tier and calls it the prerequisite for
# FULL_AND_PIECEWISE (dsa-csx/glm_52/vllm_study/GRAPH.md, "Deliberately not in Tier A"). Prefill
# runs eager under FULL_DECODE_ONLY, so the sync is legal and decode is still captured. The `topk`
# control stays on FULL_AND_PIECEWISE because its selector is the stock graph-compatible kernel.
run_smoke ceil_graph 4 "$OUT/models/ceil_graph" graph _pluginboot_approx FULL_DECODE_ONLY & p4=$!
run_smoke midpoint_eager 5 "$OUT/models/midpoint" eager _pluginboot_approx & p5=$!
run_smoke floor_eager 6 "$OUT/models/floor" eager _pluginboot_approx & p6=$!
run_smoke floor_graph 7 "$OUT/models/floor_graph" graph _pluginboot_approx FULL_DECODE_ONLY & p7=$!

fail=0
for pair in \
  "exact_eager:$p0" \
  "approx_topk_eager:$p1" \
  "approx_topk_graph:$p2" \
  "ceil_eager:$p3" \
  "ceil_graph:$p4" \
  "midpoint_eager:$p5" \
  "floor_eager:$p6" \
  "floor_graph:$p7"
do
  label=${pair%%:*}
  pid=${pair#*:}
  if wait "$pid"; then
    echo "[runner] PASS process $label"
  else
    rc=$?
    echo "[runner] FAIL process $label rc=$rc"
    tail -80 "$OUT/logs/$label.log" || true
    fail=1
  fi
done

set +e
"$PY" "$REPO/tests/dsa/qwen3_dsa_approx_validation_summary.py" "$OUT" \
  --batch-size "$DECODE_BATCH_SIZE" \
  --tokens "$MAX_TOKENS" \
  --logprob-mean-tolerance "$LOGPROB_MEAN_TOLERANCE" \
  --logprob-max-tolerance "$LOGPROB_MAX_TOLERANCE" \
  --quality-relative-tolerance "$QUALITY_RELATIVE_TOLERANCE" \
  --quality-absolute-tolerance "$QUALITY_ABSOLUTE_TOLERANCE"
compare_rc=$?
set -e

echo "[runner] summary=$OUT/summary.json"
echo "[runner] logs=$OUT/logs"
if [[ $fail -ne 0 || $compare_rc -ne 0 ]]; then
  echo "[runner] GPU VALIDATION FAILED"
  exit 1
fi
echo "[runner] GPU VALIDATION PASSED"
