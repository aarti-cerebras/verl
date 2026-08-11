#!/usr/bin/env bash
# Qwen3-4B-Thinking-2507 Phase-2b LONG-CONTEXT data generation, end to end, on this host (slinky-0).
#
#   ./examples/msa/gen_qwen3_longctx_ph2.sh tierA pilot     # the gate -- runbook §10
#   ./examples/msa/gen_qwen3_longctx_ph2.sh tierA full
#   ./examples/msa/gen_qwen3_longctx_ph2.sh tierB pilot
#
# Runbook: docs/qwen3_4b_msa/phase2_long_context_gen.md. Sibling of gen_gptoss_ph2.sh, which does the
# same job for the short/decode-long Dolci bank.
#
# THE TWO TIERS ARE NOT INTERCHANGEABLE -- they differ in three load-bearing ways:
#
#            licence            window     max_num_seqs   drop-in at a 128K training window?
#   tierA    permissive/undecl  131,072    12             yes
#   tierB    CC-BY-NC-2.0       163,840    5              NO -- rows can exceed 131,072 total
#
#   * The WINDOW is part of the dataset's identity, like the generator model. Tier B is raised to
#     163,840 because ChatQA2's documents were truncated to 131,072 *Llama-3* tokens upstream, and
#     Qwen3 retokenization pushes them to 125,885-131,195 -- leaving no room to answer inside a
#     131,072 total window (runbook §2.1, §2.2). Every row records the window it was built under.
#   * max_num_seqs follows from KV: 144 KiB/token means 18.0 GiB/seq at 131,072 and 22.5 GiB at
#     163,840, i.e. ~6 and ~5 per H200. Oversubscribing makes vLLM preempt and RECOMPUTE a 100K
#     prefill, which reads as bad throughput rather than an error (runbook §5.4).
#   * Tier B is licence-segregated so the NC rows can be dropped at training-mix time without
#     regenerating.
set -euo pipefail

TIER=${1:-tierA}
MODE=${2:-pilot}
[[ "$TIER" == "tierA" || "$TIER" == "tierB" ]] || { echo "usage: $0 {tierA|tierB} {pilot|full}"; exit 1; }
[[ "$MODE" == "pilot" || "$MODE" == "full" ]]  || { echo "usage: $0 {tierA|tierB} {pilot|full}"; exit 1; }

export REPO_ROOT=${REPO_ROOT:-/home/aarti_cerebras/dsa/verl_msa}
BASE=/home/aarti_cerebras
MODEL=${MODEL:-${BASE}/models/qwen3_4b_thinking_2507}
HF_CACHE=${HF_CACHE:-${BASE}/msa/hf_cache/longctx}

# --- dataset identity: changing any of these forks the dataset ------------------------------------
if [[ "$TIER" == "tierA" ]]; then
    SOURCES="longcite loongrl longreward longalpaca longalign docqarl"
    LICENCE_TIER=A
    WINDOW=${WINDOW:-131072}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-12}
    MAX_PER_DOC=${MAX_PER_DOC:-0}
else
    SOURCES="chatqa2"
    LICENCE_TIER=B
    WINDOW=${WINDOW:-163840}
    MAX_NUM_SEQS=${MAX_NUM_SEQS:-5}
    MAX_PER_DOC=${MAX_PER_DOC:-4}   # NarrativeQA asks many questions per book; prefill dominates cost
fi
MIN_GEN_BUDGET=${MIN_GEN_BUDGET:-8192}
MIN_PREFILL=${MIN_PREFILL:-16384}
# Qwen3-4B-Thinking-2507's own card. NOT the gpt-oss run's 1.0/1.0.
TEMP=${TEMP:-0.6}; TOP_P=${TOP_P:-0.95}; TOP_K=${TOP_K:-20}; MIN_P=${MIN_P:-0}
SEED=${SEED:-1234}

# --- throughput knobs: safe to change between legs -------------------------------------------------
DP=${DP:-8}; TP=${TP:-1}; GPU_UTIL=${GPU_UTIL:-0.90}; CHUNK=${CHUNK:-128}

RUN=${RUN:-${BASE}/msa/data/qwen3-4b-thinking-2507__longctx_${TIER}__L${WINDOW}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN/logs"

# --- preflight: fail in a second, not ten minutes into engine construction --------------------------
[[ -f "$MODEL/config.json"    ]] || { echo "ERROR: model missing at $MODEL"; exit 1; }
[[ -f "$MODEL/tokenizer.json" ]] || { echo "ERROR: tokenizer missing at $MODEL"; exit 1; }
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
# The node is shared: check memory.used, NOT the process list -- another container's allocation is
# invisible to --query-compute-apps from inside ours (phase2_data_gen.md §11.1).
_busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>2000' | wc -l)
[[ "$_busy" -eq 0 ]] || echo "WARNING: $_busy GPU(s) already hold >2 GB — sizing may fail at engine init"

cd "$REPO_ROOT"
git rev-parse HEAD > "$RUN/git_sha.txt"
git diff > "$RUN/git_diff.patch"
cp "$0" "$RUN/launch_gen.sh"

echo "[gen] tier=$TIER mode=$MODE window=$WINDOW max_num_seqs=$MAX_NUM_SEQS run=$RUN"

# --- 1. prompts (runbook §4) ------------------------------------------------------------------------
# NOTE: NEVER add --limit-per-source here. It reads sequentially from row 0, which is a biased sample
# (LongAlpaca is ordered long-first, NarrativeQA groups questions per book). The pilot subset is drawn
# from the FULL extraction in step 2.
PROMPTS="$RUN/prompts.jsonl"
python3 scripts/dsa/select_prompts.py --source longctx --licence-tier "$LICENCE_TIER" \
    --sources $SOURCES --tokenizer "$MODEL" \
    --window "$WINDOW" --min-gen-budget "$MIN_GEN_BUDGET" --min-prefill-tokens "$MIN_PREFILL" \
    --max-per-document "$MAX_PER_DOC" --local-dir "$HF_CACHE" \
    --log-dir "$RUN/logs" --out "$PROMPTS" \
    ${EXCLUDE_SHA:+--exclude-sha $EXCLUDE_SHA}

# --- 2. HARD GATE on the prompt set (runbook §9 test 5) ---------------------------------------------
# Before any GPU time: a prompt that violates the window budget yields a truncated trace at FULL
# prefill cost, and nothing downstream notices until the length histogram comes back wrong.
python3 scripts/msa/verify_longctx_prompts.py --prompts "$PROMPTS" --log-dir "$RUN/logs"

# --- 3. pool report + survey regression (runbook §9 test 2) -----------------------------------------
python3 scripts/msa/longctx_pool_report.py --prompts "$PROMPTS" \
    --out-report "$RUN/pool.json" --log-dir "$RUN/logs"

if [[ "$MODE" == "pilot" ]]; then
    PROMPTS="$RUN/prompts_pilot.jsonl"
    # Stratified by source AND prefill band (runbook §10): decode length and truncation rate both
    # vary with prefill, so a flat sample would leave the 64K+ band -- where the window bites --
    # unmeasured.
    python3 scripts/msa/sample_prompts_stratified.py --src "$RUN/prompts.jsonl" \
        --out "$PROMPTS" --seed "$SEED" \
        --bands "${PILOT_BANDS:-32768,65536,131072,1e9}" --per-band "${PILOT_PER_BAND:-8}" \
        ${PILOT_ARGS:-}
fi

# --- 3. generate (runbook §5) -----------------------------------------------------------------------
python3 scripts/dsa/gen_trajectories.py \
    --prompts "$PROMPTS" --out "$RUN/trajectories.jsonl" --log-dir "$RUN/logs" \
    --model "$MODEL" \
    --temperature "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P" \
    --max-model-len "$WINDOW" --fit-window "$WINDOW" --min-gen-budget "$MIN_GEN_BUDGET" \
    --data-parallel-size "$DP" --tensor-parallel-size "$TP" --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" --chunk-size "$CHUNK" --seed "$SEED" --no-merge

# --- 4. splice -> input_ids + loss_mask (runbook §3, inherits phase2_data_gen.md §6) ----------------
python3 scripts/dsa/trajectories_to_sft_parquet.py --input "$RUN/trajectories.jsonl.part*" \
    --tokenizer "$MODEL" --emit-input-ids --max-length "$WINDOW" \
    --out "$RUN/bc_2b_longctx.parquet" --log-dir "$RUN/logs"

# --- 5. verify: hard gate, exits non-zero -----------------------------------------------------------
python3 scripts/dsa/verify_sft_parquet.py --parquet "$RUN/bc_2b_longctx.parquet" \
    --max-length "$WINDOW" --tokenizer "$MODEL" --log-dir "$RUN/logs"

# --- 6. reports (runbook §6.1: by prefill band, not just by source) ---------------------------------
python3 scripts/dsa/analyze_lengths.py --trajectories "$RUN/trajectories.jsonl.part*" \
    --window "$WINDOW" --group-by domain --out-report "$RUN/lengths.json" --log-dir "$RUN/logs"

echo "[gen] DONE -> $RUN"
