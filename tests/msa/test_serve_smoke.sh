#!/usr/bin/env bash
# Serve-path smoke test for the Qwen3-MSA plugin (docs/qwen3_4b_msa/serving_plan.md §7, P3).
#
# S0 drives an IN-PROCESS `LLM()`, so it never exercises the thing that actually breaks in
# production: vLLM spawns EngineCore as a subprocess that does NOT inherit the parent's imports,
# so `Qwen3MSAForCausalLM` must be registered there too via _pluginboot/sitecustomize.py. This
# test starts the real server through serve_msa.sh and asserts:
#
#   1. sitecustomize registered the plugin (in the parent AND the EngineCore child)
#   2. the server reaches "Application startup complete"
#   3. an OpenAI /v1/completions request returns coherent text
#   4. the sparse backends were selected (the §6.6 anti-dense log grep)
#
# Usage:  GPU=0 bash tests/msa/test_serve_smoke.sh
set -uo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GPU=${GPU:-0}
PORT=${PORT:-$((8100 + GPU))}
# Small footprint: this is a smoke test, not a benchmark.
export MAX_LEN=${MAX_LEN:-8192}
export GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.30}
export GPU PORT REPO

cleanup() {
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "[smoke] stopping server pid=$PID"
    kill -TERM -- -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
    sleep 3
    kill -KILL -- -"$PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[smoke] launching serve_msa.sh (gpu=$GPU port=$PORT max_len=$MAX_LEN)"
OUT=$(bash "$REPO/scripts/msa/serving/serve_msa.sh")
echo "$OUT" | sed -n '/MANIFEST/,/====$/p' | head -20
PID=$(echo "$OUT" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
LOG=$(echo "$OUT" | sed -n 's/^log: *//p' | head -1)
[[ -z "$LOG" ]] && LOG=$(ls -t /cb/ml-eng/aarti/msa/serving/logs/*serve_msa_gpu${GPU}_p${PORT}.log | head -1)
echo "[smoke] pid=$PID log=$LOG"

echo "[smoke] waiting for startup (up to 300s)..."
for i in $(seq 1 300); do
  grep -q "Application startup complete" "$LOG" 2>/dev/null && break
  if ! kill -0 "$PID" 2>/dev/null; then echo "[smoke] FAIL: server died"; tail -25 "$LOG"; exit 1; fi
  sleep 1
done

rc=0
check() { if [[ "$1" == "0" ]]; then echo "  PASS  $2"; else echo "  FAIL  $2"; rc=1; fi; }

grep -q "Application startup complete" "$LOG"; check $? "server started"
grep -q "\[sitecustomize\] Qwen3-MSA plugin registered" "$LOG"; check $? "sitecustomize registered the plugin"
# EngineCore runs as a separate process; its stderr is tagged. Two registrations => parent + child.
n_reg=$(grep -c "\[sitecustomize\] Qwen3-MSA plugin registered" "$LOG" || true)
[[ "$n_reg" -ge 2 ]]; check $? "registered in >=2 interpreters (parent + EngineCore), saw $n_reg"
grep -q "MiniMax M3 sparse attention selected" "$LOG"; check $? "sparse-attention backend selected"
grep -q "MiniMax M3 indexer: selected" "$LOG"; check $? "indexer backend selected"

echo "[smoke] querying /v1/completions ..."
RESP=$(curl -s -m 120 "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B-Thinking-2507","prompt":"The capital of France is","max_tokens":24,"temperature":0}')
echo "  response: $(echo "$RESP" | head -c 300)"
echo "$RESP" | grep -q "Paris"; check $? "coherent completion (expects 'Paris')"

echo
if [[ "$rc" == "0" ]]; then echo "SERVE SMOKE PASS"; else echo "SERVE SMOKE FAILED"; fi
exit $rc
