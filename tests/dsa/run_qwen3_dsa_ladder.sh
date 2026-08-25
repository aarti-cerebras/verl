#!/usr/bin/env bash
# The de-confounding ladder for the Qwen3-DSA serving path (eval_plan.md §3, serving_eval_plan.md P3).
#
# Each row changes exactly one variable, and the ROWS THAT MUST FAIL matter as much as the ones that
# must pass -- a sparse model that silently serves dense passes every benchmark it is given.
#
#   A_dense        DSA_SPARSE=0                      reference (stock dense Qwen3 on these weights)
#   B_topk_ge_T    top_k >= T                        MUST equal A token-exactly  (kernel faithfulness)
#   C_topk2048     top_k=2048, T~9K  (22% density)   the real config
#   D_topk256      top_k=256,  T~9K  (2.8% density)  MUST degrade (selection is live and matters)
#   E_rand256      random indexer, top_k=256         MUST collapse (not silently dense)
#   F_rand_ge_T    random indexer, top_k >= T        MUST recover  (collapse came from SELECTION,
#                                                    not from broken machinery)
#
# top_k is capped at 4096 by vLLM 0.26.0's compiled top-k kernel (see indexer.py::MAX_KERNEL_TOP_K),
# so the "top_k >= T" rows run at a shorter prompt. That is a limit on the CONTROL, not on serving.
#
# Usage:  bash tests/dsa/run_qwen3_dsa_ladder.sh [serving_dir]
#         python3 tests/dsa/compare_ladder.py /tmp/dsa_ladder
set -u
REPO=${REPO:-/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl}
PY=${PY:-$REPO/.devlibs/vllm026/bin/python}
MODEL=${1:-/cb/ml-eng/aarti/dsa_qwen3/serving/p2_mix5050_k2048_step1200}
OUT=${OUT:-/tmp/dsa_ladder}; mkdir -p "$OUT"
GPU=${GPU:-0}

run() { label=$1; shift
  echo "=== ROW $label ==="
  env "$@" CUDA_VISIBLE_DEVICES=$GPU timeout 1800 "$PY" "$REPO/tests/dsa/qwen3_dsa_offline_smoke.py" \
    --model "$MODEL" --max-tokens 32 --out-json "$OUT/$label.json" --label "$label" \
    ${EXTRA:-} 2>&1 | grep -E "^\[smoke\]|RuntimeError:|AcceleratorError" | tail -6
}

# Long-prompt rows: real sparsity.
LONG="--prompt-tokens 6000 --max-model-len 16384"
EXTRA="$LONG"                 run A_dense     DSA_SPARSE=0
EXTRA="$LONG"                 run C_topk2048  DSA_SPARSE=1
EXTRA="$LONG --top-k 256"     run D_topk256   DSA_SPARSE=1
EXTRA="$LONG --top-k 256"     run E_rand256   DSA_SPARSE=1 DSA_RANDOM_INDEXER=1
# Short-prompt rows: top_k >= T, i.e. selection is a no-op and the output must be the dense one.
SHORT="--prompt-tokens 1200 --max-model-len 8192 --top-k 4096"
EXTRA="--prompt-tokens 1200 --max-model-len 8192" run A2_dense DSA_SPARSE=0
EXTRA="$SHORT"                run B2_topk_ge_T DSA_SPARSE=1
EXTRA="$SHORT"                run F2_rand_ge_T DSA_SPARSE=1 DSA_RANDOM_INDEXER=1

echo; echo "compare with: $PY $REPO/tests/dsa/compare_ladder.py $OUT"
