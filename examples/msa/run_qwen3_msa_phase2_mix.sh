#!/usr/bin/env bash
# Qwen3-4B MSA Phase-2b on the COMBINED dataset: the 16K-32K long-context band AND the full short/broad
# ph2b behaviour-cloning bank, trained as ONE run, initialised from the 32K run's step 10700.
#
#   ./examples/msa/run_qwen3_msa_phase2_mix.sh            # the full 14,114-step run
#   SMOKE=1 ./examples/msa/run_qwen3_msa_phase2_mix.sh     # 20 steps, no checkpoints, throwaway RUNS_BASE
#
# WHY THIS EXISTS, as opposed to BAND=16k32k
#   Long-only training at step 1600 bought RULER 32K (mean 88.97 -> 92.48 over k16@10700) and lost short
#   context: MRCR 2needle_4-8K 0.7069 -> 0.4722, bin 4-8K -0.078, bin 8-16K -0.068, GSM-Infinite 32K -3.3
#   (docs/qwen3_4b_msa/longctx_step1600_comparison.md). Interleaving the short bank in the SAME run is the
#   direct fix, rather than annealing the two stages against each other sequentially.
#
# This is a THIN WRAPPER over examples/msa/run_qwen3_msa_phase2.sh -- it only sets env overrides and execs
# it, so the canonical script keeps ownership of the run identity (CONFIG_TAG-keyed checkpoint dir so
# `resume_mode=auto` works), logs/run-<TS>.log, hydra/<TS>/, WANDB_DIR and the tee. It is a SIBLING of
# run_qwen3_msa_phase2_longctx.sh, not a caller of it: that script's BAND case is a 4-way unit
# (split, seq_len, tiers, steps) anchored under one long-context artifact, and its preflight tests
# TRAIN_FILES with `-f`, which a multi-shard DIRECTORY spec fails.
#
# ---------------------------------------------------------------------------------------------------
# WHAT THE DATASET IS
#
# /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__mix_ph2bfull_longctx16k32k_v1/
#     train-00000.parquet -> longctx_tierA/split_16k_32k_v1/train-00000.parquet   23,199 rows  0.567B tok
#     train-00001.parquet -> ph2b_split_v1/train-00000.parquet                    89,719 rows  0.849B tok
#     val-00000.parquet   -> longctx_tierA/split_16k_32k_v1/val-00000.parquet        739 rows
#     val-00001.parquet   -> ph2b_split_v1/val-00000.parquet                         511 rows
#                                                                          TRAIN 112,918 rows  1.416B tok
#
# SYMLINKS, not a rewritten parquet. MSASFTDataset reads each file with its own pq.read_table and touches
# only `input_ids`/`loss_mask` (msa_sft_dataset.py:88-125), so the concatenation happens in the loader.
# That is also why the one schema difference between the two sources needs no cast: `original_dataset` is
# large_string in ph2b and all-null `null` type in longctx, which a real pa.concat_tables would have to
# reconcile and this does not.
#
# Val holdouts stay valid: prompt_sha256 intersection is 0 for all four cross-pairs (ph2b_train n
# longctx_val, longctx_train n ph2b_val, train n train, val n val) -- the two sources draw on disjoint
# prompt pools (Dolci-Think-RL vs longcite/loongrl/longreward/longalpaca/longalign/docqarl).
#
# ---------------------------------------------------------------------------------------------------
# WHAT DIFFERS FROM BAND=16k32k -- exactly two things, plus their consequences
#
# 1. TRAIN_FILES/VAL_FILES are a DIRECTORY, so both shards are picked up. run_qwen3_msa_phase2.sh's
#    _expand_files globs `train-*`/`val-*` SEPARATELY and hands hydra an explicit bracketed list.
#    Do NOT hand this directory to MSASFTDataset directly: its own directory branch
#    (msa_sft_dataset.py:82) takes EVERY *.parquet in the dir, which would swallow val into train.
#
# 2. STEPS 2,899 -> 14,114 (112,918 rows / BATCH 8 = one epoch). This also restretches the cosine
#    schedule and the 3% warmup over the new length.
#
# SEQ_LEN and LENGTH_TIERS are UNCHANGED at 32768 and 16x2048. Both sources are 32768-capped, so the mix
# adds rows in tiers 0-7 and more rows in 8-15; it does not widen the window. Peak GPU memory is therefore
# still the proven 32K profile (34.6 GB alloc / 48.4 GB reserved). Every tier 0-15 holds >=3,400 rows, so
# all tiers form full batches, and tiers 8-15 are MIXED-SOURCE -- those steps see both distributions,
# which is intended.
#
# UNCHANGED, deliberately: TOPK=16, BLOCK_SIZE=128, DENSE_PREFIX=3, KL_LAMBDA=1.0, LR=5e-6,
# INDEXER_LR=1e-4, cosine + 3% warmup, MIN_LR_RATIO=0.1, CLIP_GRAD=1.0, BATCH=8, fsdp2, bf16,
# ACT_OFFLOAD, TILED_MLP_SHARDS=8, DATA_SEED=1234, NUM_WORKERS=8.
#
# ---------------------------------------------------------------------------------------------------
# THE MIX RATIO IS NOT ONE NUMBER -- the two loss terms normalise on different denominators
#
#   LM cross-entropy   / loss_mask.sum()      = RESPONSE tokens only          (losses.py:153)
#   indexer KL         / num_valid_queries    = ALL non-pad tokens, loss_mask-independent
#                                               (tensordict_utils.py:162, losses.py:113)
#
# longctx rows are 88.4% masked-out prefix (resp_frac 0.116, vs ph2b's 0.979): 66.0M response tokens
# against ph2b's 830.7M. So this concatenation gives the long band 40.1% of the INDEXER gradient and only
# 7.4% of the LM CE. That asymmetry is arguably the right shape -- the long data exists to teach the
# indexer to pick 16 blocks out of a 32K key sequence, while the CE's job is to hold short-context
# behaviour -- but it is a consequence of the row counts, not a setting. The only lever on it is the
# ph2b:longctx ROW ratio, i.e. subsampling ph2b; at 25% of ph2b rows the shares move to 72.8% / 24.1%.
#
# ---------------------------------------------------------------------------------------------------
# HOST RAM: MEASURED, not estimated. The mix is affordable; the cost is a STARTUP TRANSIENT, not steady state.
#
# GPU memory is settled (same 32K profile). Host RAM was the open question, so it was measured directly by
# constructing MSASFTDataset on each shard set in a fresh process (2026-08-15, this node, 1999 GB):
#
#                     tokens    RETAINED RSS   TRANSIENT PEAK      x8 ranks: retained / peak
#     longctx only    0.567B      16.71 GB        25.18 GB              134 GB / 201 GB
#     ph2b only       0.849B       6.46 GB        34.92 GB
#     THIS MIX        1.416B      19.17 GB        57.68 GB              153 GB / 461 GB
#     val (blended)   0.023B       2.19 GB         2.14 GB               18 GB
#
# So against the long-only run the mix costs +19 GB of steady state across all 8 ranks -- negligible against
# 1999 GB -- and +260 GB of startup transient. The transient is survivable because it lands during
# _build_dataset, BEFORE the ~1736 GB of activation-offload buffers exist; the two peaks do not overlap.
#
# Do NOT reason about this from tokens x 5 B/token (int32 ids + int8 masks). That model predicts 2.8 GB for
# the long band and it retains 16.71 GB -- MORE than ph2b's 6.46 GB despite having 1.5x FEWER tokens. The
# reason is row-group count, not data volume: the longctx parquet has 511 row groups against ph2b's 1, so
# combine_chunks() plus thousands of small buffers fragment the glibc arena and the freed arrow buffers are
# never returned to the OS. The mix's 19.17 GB is therefore mostly the long shard's fragmentation, which it
# would have paid anyway, and only ~2.5 GB of genuinely extra rows.
#
# Still run SMOKE=1 before committing ~8.6 days: the figures above are single-process, and a host OOM here
# surfaces as a rendezvous-heartbeat error with NO traceback rather than anything that says "out of memory".
#
# NUM_WORKERS is the mitigation if it does bite, and it MUST be chosen before launch: StatefulDataLoader's
# multiprocess snapshot is keyed by worker id, so lowering it later invalidates the saved iterator state.
# Same for DATA_SEED, which regenerates the tiered row order.
#
# ---------------------------------------------------------------------------------------------------
# COST, at the ~1900 ctx-tok/s both parent runs measured (37 s/step on ph2b, 105 s/step on the long band)
#
#   1.416B tokens / 14,114 steps = ~100.3K tok/step = ~52.8 s/step  ->  ~8.6 days for one epoch
#
# TEST_FREQ is 400 here, not the long run's 150: val is now a blended 1,250 rows / 22.9M tokens, which
# forward-only costs ~1.1 h. At 150 steps (2.2 h) that is 34% of wall-clock; at 400 (5.9 h) it is ~19%,
# matching the overhead the long-only run already ran at. Raising it further trades eval resolution for
# throughput. Do NOT cap it with VAL_MAX_SAMPLES instead: MSASFTDataset applies that cap by taking the
# FIRST N of the already-tiered order, not a stratified draw, so small sources vanish entirely.
# ---------------------------------------------------------------------------------------------------
set -euo pipefail

# Two independent roots, deliberately: on the ml-eng-gpu-* hosts the repo (home NFS) and the artifacts
# (/cb/ml-eng FSx share) are different filesystems, so one variable cannot name both.
export REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}   # same default as run_qwen3_msa_phase2.sh
MSA_BASE=${MSA_BASE:-/cb/ml-eng/aarti}                                   # data + checkpoints
export RUNS_BASE=${RUNS_BASE:-${MSA_BASE}/msa/sparse}

MIX=${MIX:-${MSA_BASE}/msa/data/qwen3-4b-thinking-2507__mix_ph2bfull_longctx16k32k_v1}

# EVERY derived artifact for this checkpoint lives in ONE named folder inside the source checkpoint dir;
# see $CKPT/DERIVED.json. NOTE the `_ckpt/` segment: run_qwen3_msa_phase2.sh puts every run under
# ${RUNS_BASE}/_ckpt/${CONFIG_TAG}, so the source run's directory is one level deeper than the sparse root.
# Anchored on MSA_BASE, NOT RUNS_BASE: RUNS_BASE is an OUTPUT knob (SMOKE redirects it below), and the
# input checkpoint must not move when it does.
CKPT=${CKPT:-${MSA_BASE}/msa/sparse/_ckpt/p2_qwen3-4b-thinking-2507_ph2b_split_v1_L32k_bs8_k16_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t16w2048_st11214_v2/global_step_10700/qwen3_4b_msa_p2b_k16_step10700}
export MODEL_PATH=${MODEL_PATH:-$CKPT}

# WARMSTART is LOAD-BEARING; never clear it. Fresh run (step 0, new optimizer/schedule) that INITIALISES
# from step-10700 weights; NOT a resume. The ordering is why the indexer needs its own file:
#
#     from_pretrained(MODEL_PATH)   <- indexer modules DO NOT EXIST yet, so the 132 indexer tensors are
#                                      reported UNEXPECTED and DROPPED by the HF loader
#     attach_indexers(...)          <- NOW creates MSAIndexer modules, randomly initialised
#     _warmstart_from_consolidated  <- the ONLY hook that fills them (skipped when the path is '')
#
# So setting msa_warmstart_path='' on the reasoning that the merged safetensors already contain the
# trained indexer SILENTLY TRAINS A RANDOM INDEXER. Base weights DO load from MODEL_PATH -- only indexer
# keys were unexpected -- so MODEL_PATH plus this warmstart together give a complete step-10700 init.
# Built by: scripts/dsa/consolidate_indexer_ckpt.py --arch msa  (132 params, 97.33M elements).
export WARMSTART=${WARMSTART:-$CKPT/msa_p2b_k16_step10700_indexer_full.pt}

# DIRECTORY specs, not files -- see "WHAT DIFFERS" note 1.
export TRAIN_FILES=${TRAIN_FILES:-${MIX}}
export VAL_FILES=${VAL_FILES:-${MIX}}

export SEQ_LEN=${SEQ_LEN:-32768}
export BATCH=${BATCH:-8}
export STEPS=${STEPS:-14114}           # 112,918 rows / BATCH 8 = one epoch; asserted against the shards below
export LENGTH_TIERS=${LENGTH_TIERS:-16}
export TIER_WIDTH=${TIER_WIDTH:-2048}
export TOPK=${TOPK:-16}
# 4 -> 8, inherited from the long-context run and kept here. The MLP forward/backward is chunked along the
# sequence dim (torch.chunk(x, shards, dim=-2)), so only 1/N of the FFN intermediates are live at once.
# This is the SANCTIONED substitute for gradient checkpointing, which is forbidden: HF's version runs the
# first pass under no_grad, so the `_msa_kl` side effect would be stashed WITHOUT a graph and contribute
# ZERO indexer gradient, silently. Pure memory/compute tradeoff, no numerics change.
export TILED_MLP_SHARDS=${TILED_MLP_SHARDS:-8}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}   # -1 = the entire blended val split; see the TEST_FREQ note
export TEST_FREQ=${TEST_FREQ:-400}
export SAVE_FREQ=${SAVE_FREQ:-100}
export MAX_CKPT=${MAX_CKPT:-5}

# Pinned rather than auto-derived: DATA_TAG would otherwise come out as the mix directory's basename via
# `basename ... .parquet`, and CONFIG_TAG keys the CHECKPOINT dir, so it must be stable, descriptive, and
# must change if any experiment-defining knob changes.
_LEN_TAG=$(awk -v l="${SEQ_LEN}" 'BEGIN{printf "L%dk", int(l/1024)}')
export CONFIG_TAG=${CONFIG_TAG:-p2b_qwen3-4b-thinking-2507_mixph2bfull-longctx16k32k_${_LEN_TAG}_bs${BATCH}_k${TOPK}_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t${LENGTH_TIERS}w${TIER_WIDTH}_st${STEPS}_from10700}

# SMOKE: measure host RAM and confirm step 1 posts, without writing into the real run's namespace. A
# throwaway RUNS_BASE keeps CONFIG_TAG's checkpoint dir clean, so the real launch still starts at step 0.
if [[ "${SMOKE:-0}" == "1" ]]; then
    export RUNS_BASE=${MSA_BASE}/msa/sparse/_smoke_mix
    export STEPS=${SMOKE_STEPS:-20}
    export SAVE_FREQ=-1
    export TEST_FREQ=-1
    # Tag reads "smoke20_..._st14114_...": N smoke steps OF the full config, whose st component is
    # deliberately left intact so the smoke is traceable to the run it is clearing.
    export CONFIG_TAG="smoke${STEPS}_${CONFIG_TAG}"
    echo "[mix] SMOKE: ${STEPS} steps, no checkpoints, RUNS_BASE=${RUNS_BASE}"
    echo "[mix] SMOKE: watch host RAM (free -g) and GPU (nvidia-smi); the failure to look for is a"
    echo "[mix] SMOKE: rendezvous-heartbeat error with no traceback = host OOM, not a GPU OOM."
fi

# --- preflight: the failures that otherwise look like a healthy run --------------------------------
[[ -f "$MODEL_PATH/config.json" ]] || { echo "ERROR: merged model missing at $MODEL_PATH"; exit 1; }
[[ -d "$MIX" ]] || { echo "ERROR: mix dir missing at $MIX"; exit 1; }
[[ -f "$WARMSTART" ]] || { echo "ERROR: indexer warmstart missing at $WARMSTART -- run scripts/dsa/consolidate_indexer_ckpt.py --arch msa"; exit 1; }

# Resolve every symlink. A dangling one, or a source artifact that moved, would otherwise surface as a
# confusing pyarrow error several minutes into startup.
for f in "$MIX"/train-*.parquet "$MIX"/val-*.parquet; do
    [[ -f "$f" ]] || { echo "ERROR: $f is missing or dangling (-> $(readlink -f "$f" 2>/dev/null))"; exit 1; }
done

# The shard SET and ORDER are load-bearing, not cosmetic: _tiered_order indexes rows by position in the
# concatenated list, and verl's resume saves a bare batch counter with no dataset identity
# (checkpoint_handler.py:116-124). So a shard added, removed or renumbered after launch silently resumes
# onto different rows. Assert the row count the STEPS above was computed from.
python3 - "$MIX" "$STEPS" "$BATCH" "${SMOKE:-0}" <<'PY'
import glob, os, sys
import pyarrow.parquet as pq
mix, steps, bsz, smoke = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
for pre in ("train", "val"):
    shards = sorted(glob.glob(os.path.join(mix, f"{pre}-*.parquet")))
    rows = [pq.ParquetFile(p).metadata.num_rows for p in shards]
    print(f"[mix-preflight] {pre}: {len(shards)} shard(s), {sum(rows)} rows "
          f"({', '.join(f'{os.path.basename(p)}={n}' for p, n in zip(shards, rows))})")
    if pre == "train":
        total = sum(rows)
assert total == 112918, (
    f"train rows = {total}, expected 112918 -- a shard changed. STEPS was computed from that count; "
    f"recompute it as rows//{bsz} and update CONFIG_TAG, or the schedule and the epoch boundary are wrong."
)
if not smoke:
    assert steps == total // bsz, f"STEPS={steps} != {total}//{bsz}={total // bsz} (one epoch)"
PY

python3 - "$WARMSTART" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
n = sum(1 for k in sd if ".indexer." in k)
print(f"[mix-preflight] warmstart carries {n} indexer tensors")
assert n >= 100, f"only {n} indexer tensors in {sys.argv[1]} -- refusing to launch onto a random indexer"
PY

echo "[mix] CONFIG_TAG=$CONFIG_TAG"
echo "[mix] init-from (weights only, step 0): $MODEL_PATH"
echo "[mix] indexer warmstart:               $WARMSTART"
echo "[mix] data (both shards):              $MIX"
echo "[mix] seq_len=$SEQ_LEN tiers=${LENGTH_TIERS}x${TIER_WIDTH} top_k=$TOPK steps=$STEPS test_freq=$TEST_FREQ"

exec bash "${REPO_ROOT}/examples/msa/run_qwen3_msa_phase2.sh" "$@"
