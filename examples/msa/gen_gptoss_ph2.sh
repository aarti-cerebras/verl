#!/usr/bin/env bash
# gpt-oss-20b Phase-2b BC data generation, end to end, on THIS host (slinky-0).
#
#   ./examples/msa/gen_gptoss_ph2.sh pilot     # 100 prompts, 25/domain -- the gate (doc §8)
#   ./examples/msa/gen_gptoss_ph2.sh full      # 93,889 prompts
#
# Runbook: docs/gpt_oss_20b_msa/phase2_data_gen.md. This script IS §10, with the paths repointed from
# /cb/ml-eng/aarti (ml-eng-gpu-09, unreachable from here) to ${BASE}. Every artifact §10 depends on was
# rebuilt locally and verified against the numbers recorded in the doc -- see PROVENANCE below.
#
# PROVENANCE -- what was rebuilt here and how it was checked (2026-08-09)
#   val_prompt_shas.txt   Regenerated from the Qwen3 val parquet that WAS copied to this host: sorted
#                         unique prompt_sha256, one per line, trailing newline. sha256 of the file is
#                         d101c6586ead1726e5739896cd27a1f55fce13e71450c24cffdfeaa5e019c2aa -- byte-identical
#                         to the frozen file recorded in doc §2. The shared split (§2) is therefore intact
#                         even though the original file is on the other host.
#   prompts.jsonl         Re-extracted with select_prompts.py --source dolci-rl from the same HF dataset,
#                         then tokenized with the gpt-oss tokenizer. NOT a copy -- but provably the same
#                         bank: 102,026 raw -> -1,323 multi-turn -> -6,814 dup -> -0 char-bounds ->
#                         93,889, every counter matching doc §1.1; and all 90,230 prompt_sha256 that
#                         survived the Qwen3 run (train 89,719 + val 511) are present, missing 0.
#                         prompt_tokens percentiles reproduce doc §3 exactly (ALL p50 113 / p99 899 /
#                         max 2,015; Math 88/460/2,013; Code 232/1,043/1,935; IF 155/1,015/2,015;
#                         General 38/713/1,989).
#                         Row ORDER differs from the doc's file (that one came from retokenizing the
#                         Qwen3 file; this one is domain-ordered out of select_prompts), so its file-level
#                         sha256 is NOT 817fdbc5... Order is not load-bearing: rows are independent, and
#                         the pilot subset is drawn stratified (below) precisely because order is not.
#
# PIN -- frozen at 2026-08-09, the day generation starts on this host, and NOT to be changed between
# legs. The harmony template bakes strftime_now() into every served prefix (doc §4), so a second leg run
# on a later date silently produces a parquet with two different system prompts. THE SAME VALUE MUST BE
# PASSED AT PHASE-2 SERVING TIME -- it is part of the dataset's identity, like the generator model.
# Nothing has been generated with any other pin, so this is a free choice made once, here.
#
# STACK -- verified on this host before writing this script:
#   vLLM 0.20.2 registers GptOssForCausalLM; MXFP4 loads on H200 (13.64 GiB, MoEPrepareAndFinalizeNoDPEP)
#   and a real generation came back with harmony channels and a <|return|> (200002) terminal.
#   transformers 5.3.0 (doc §4 verified the date/effort pinning on 4.57): tests/dsa/
#   test_harmony_sft_roundtrip.py is 21/21 green here against the real tokenizer, so the pinning, the
#   marker ids, the §6 filter table and the verifier all hold on this stack.
set -euo pipefail

MODE=${1:-pilot}
[[ "$MODE" == "pilot" || "$MODE" == "full" ]] || { echo "usage: $0 {pilot|full}"; exit 1; }

export REPO_ROOT=${REPO_ROOT:-/home/aarti_cerebras/dsa/verl_msa}
BASE=/home/aarti_cerebras

MODEL=${MODEL:-${BASE}/models/gpt-oss-20b}
PROMPT_DIR=${PROMPT_DIR:-${BASE}/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_prompts_20260809_035359}
VAL_SHAS=${VAL_SHAS:-${BASE}/msa/qwen3-4b-thinking-2507__ph2b_split_v1/val_prompt_shas.txt}

# --- dataset identity: changing any of these forks the dataset -----------------------------------
PIN=${PIN:-2026-08-09}
EFFORT=${EFFORT:-medium}          # doc §7, user decision 2026-08-07
WINDOW=${WINDOW:-32768}           # generation window == training window
TEMP=${TEMP:-1.0}                 # OpenAI's card for gpt-oss; NOT Qwen3's 0.6/0.95/20
TOP_P=${TOP_P:-1.0}
SEED=${SEED:-1234}

# --- throughput knobs: safe to change between legs -----------------------------------------------
DP=${DP:-8}
TP=${TP:-1}
GPU_UTIL=${GPU_UTIL:-0.90}
CHUNK=${CHUNK:-512}

if [[ "$MODE" == "pilot" ]]; then
    PROMPTS=${PROMPT_DIR}/prompts_pilot100.jsonl
    TAG=pilot100
else
    PROMPTS=${PROMPT_DIR}/prompts.jsonl
    TAG=full93889
fi

# RUN is resolved ONCE and can be pinned to resume an interrupted leg into the same directory:
#     RUN=<existing dir> ./examples/msa/gen_gptoss_ph2.sh full
# gen_trajectories appends to the .partN files and skips prompts already present, so a resume with the
# same RUN and the same PIN continues; a resume with a *different* PIN silently forks the prefix (§4).
RUN=${RUN:-${BASE}/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_${TAG}_L${WINDOW}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN/logs"

# --- preflight: fail in a second, not ten minutes into engine construction ------------------------
[[ -f "$MODEL/config.json"     ]] || { echo "ERROR: model missing at $MODEL"; exit 1; }
[[ -f "$MODEL/tokenizer.json"  ]] || { echo "ERROR: tokenizer missing at $MODEL"; exit 1; }
[[ -f "$PROMPTS"               ]] || { echo "ERROR: prompts missing at $PROMPTS"; exit 1; }
[[ -f "$VAL_SHAS"              ]] || { echo "ERROR: val sha list missing at $VAL_SHAS"; exit 1; }
# The frozen split is the one artifact that CANNOT be silently rebuilt wrong -- check it, every run.
_got=$(sha256sum "$VAL_SHAS" | cut -d' ' -f1)
_want=d101c6586ead1726e5739896cd27a1f55fce13e71450c24cffdfeaa5e019c2aa
[[ "$_got" == "$_want" ]] || { echo "ERROR: val_prompt_shas.txt sha256 $_got != frozen $_want (doc §2)"; exit 1; }

cd "$REPO_ROOT"

# --- provenance (memory:log-full-invocation, memory:msa-data-artifact-layout) ---------------------
git rev-parse HEAD > "$RUN/git_sha.txt"
git diff > "$RUN/git_diff.patch"
cp "$0" "$RUN/launch_gen.sh"
python3 - "$RUN" "$MODE" "$MODEL" "$PROMPTS" "$PIN" "$EFFORT" "$WINDOW" "$TEMP" "$TOP_P" "$SEED" <<'PY'
import json, socket, subprocess, sys, time, os
run, mode, model, prompts, pin, effort, window, temp, top_p, seed = sys.argv[1:]
json.dump({
    "kind": "msa_phase2b_bc_generation",
    "mode": mode,
    "generator_model": model,
    "prompt_file": prompts,
    "prompt_dataset": "allenai/Dolci-Think-RL-32B",
    "prompt_dataset_license": "ODC-BY (Open Data Commons Attribution License)",
    "reasoning_effort": effort,          # part of the dataset's identity -- see doc §4
    "pinned_date": pin,                  # part of the dataset's identity -- MUST match at serving time
    "window": int(window),
    "sampling": {"temperature": float(temp), "top_p": float(top_p), "n": 1, "seed": int(seed)},
    "chat_format": "harmony",
    "terminator": "<|return|> (200002)",
    "host": socket.gethostname(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}, open(os.path.join(run, "MANIFEST.json"), "w"), indent=1)
PY

echo "[gen] mode=$MODE  run=$RUN"
echo "[gen] model=$MODEL  pin=$PIN  effort=$EFFORT  window=$WINDOW  temp=$TEMP/$TOP_P  dp=$DP tp=$TP"

# --- 1. generate (doc §5, §7) ---------------------------------------------------------------------
# --fit-window makes max_tokens per-row (window - len(served prefix) - 1); the prefix is recomputed
# live from THIS tokenizer, so the Qwen3-era prompt_tokens column is inert.
python3 scripts/dsa/gen_trajectories.py \
    --prompts "$PROMPTS" --out "$RUN/trajectories.jsonl" --log-dir "$RUN/logs" \
    --model "$MODEL" \
    --reasoning-effort "$EFFORT" --pin-date "$PIN" \
    --temperature "$TEMP" --top-p "$TOP_P" \
    --max-model-len "$WINDOW" --fit-window "$WINDOW" \
    --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" --chunk-size "$CHUNK" --seed "$SEED" --no-merge

# --- 2. splice -> input_ids + loss_mask (doc §5, §6). --pin-date MUST match step 1 -----------------
python3 scripts/dsa/trajectories_to_sft_parquet.py --input "$RUN/trajectories.jsonl.part*" \
    --tokenizer "$MODEL" --chat-format harmony --reasoning-effort "$EFFORT" --pin-date "$PIN" \
    --emit-input-ids --max-length "$WINDOW" --out "$RUN/bc_2b.parquet" --log-dir "$RUN/logs"

# --- 3. verify: hard gate, exits non-zero (doc §9) -------------------------------------------------
python3 scripts/dsa/verify_sft_parquet.py --parquet "$RUN/bc_2b.parquet" --max-length "$WINDOW" \
    --chat-format harmony --tokenizer "$MODEL" --log-dir "$RUN/logs"

# --- 4. reports (doc §8). 67 = the harmony wrapper, not Qwen3's 10 --------------------------------
python3 scripts/dsa/analyze_lengths.py --trajectories "$RUN/trajectories.jsonl.part*" --window "$WINDOW" \
    --chat-wrapper-tokens 67 --group-by domain --out-report "$RUN/lengths.json" --log-dir "$RUN/logs"

# --- 5. the SAME train/val split as Qwen3 (doc §2) -- full run only --------------------------------
# Skipped for the pilot: the pilot deliberately EXCLUDES the val prompts, so a split of it is empty by
# construction and would only manufacture a misleading val_shas_missing=511.
if [[ "$MODE" == "full" ]]; then
    python3 scripts/msa/split_bc_val.py --src "$RUN/bc_2b.parquet" --out-dir "${RUN}__split_v1" \
        --val-sha-file "$VAL_SHAS"
fi

echo "[gen] DONE -> $RUN"
