#!/usr/bin/env bash
# Resume the Phase-2 k=8 MSA run, driven from the verl_msa checkout.
#
# Ported from ~/dsa/verl/examples/msa/resume_p2_k8_newhost.sh. The ONLY functional change is REPO_ROOT:
# the code now comes from ~/dsa/verl_msa (this checkout, clean at e1272697) instead of ~/dsa/verl (which
# carries 41 uncommitted files). Verified before porting -- the two checkouts are byte-identical in every
# file that governs resume safety:
#     msa_sft_dataset.py        IDENTICAL   <- defines the row ORDER; a difference here would silently
#                                              resume the batch counter onto unrelated rows
#     sft_trainer.py            IDENTICAL   <- the sampler_shuffle=False patch the row order depends on
#     transformer_impl.py       IDENTICAL   <- msa_enabled gate on the KL's valid-query normalization
#     run_qwen3_msa_phase2.sh   IDENTICAL   <- the launcher this execs
#     checkpoint_handler.py     IDENTICAL   <- resume_mode=auto resolution
# qwen3_msa.py DIFFERS, and in the right direction: verl_msa carries a fix for the final-partial-block
# aliasing in the sparse gather (`clamp_max` pinned surplus slots onto seq_len-1, so the query AT
# seq_len-1 attended its own last token up to 128-(seq_len%128) times). No parameter names or shapes
# change, so the checkpoint loads unaltered; training was already blind to it because the final position
# is dropped by the shifted loss mask, but it fixed a train/serve parity failure against vLLM.
#
# RESUME: `resume_mode=auto` reads latest_checkpointed_iteration.txt from CKPT_DIR, so this continues
# from whatever the last completed save was -- no argument needed. CONFIG_TAG is reproduced byte-for-byte
# so the checkpoint dir cannot fork.
#
# Paths on this host (the original run used /cb/ml-eng/aarti and a repo under /net/aarti-vm/...):
#   checkpoints  ~/msa/<CONFIG_TAG>                        (no `sparse/_ckpt` layer -> CKPT_DIR explicit)
#   data         ~/msa/qwen3-4b-thinking-2507__ph2b_split_v1
#   base model   ~/models/qwen3_4b_thinking_2507           (dir name MUST keep the underscore spelling:
#                                                           MODEL_TAG=basename|tr _ - feeds CONFIG_TAG)
#   warmstart    ~/msa/indexer_warmup/p1_.../indexer_full.pt
#
# WARMSTART is INERT ON A RESUME and set only for provenance: `_init_engine()` applies it at model
# construction, then `load_checkpoint()` six lines later (sft_trainer.py:82 -> :88) overwrites every one
# of those tensors from the checkpoint, which already holds further-trained indexer weights. It is NOT in
# CONFIG_TAG so it cannot fork the checkpoint dir. It DOES matter for a COLD start: a random indexer
# routes sparse attention to noise (paper B.4).
#
# KNOWN ISSUE -- host OOM. With ACT_OFFLOAD=True each rank parks activations in PINNED HOST memory:
# measured at ~180 GB RSS per rank, 174 GB of it private, ~1.4 TB total. The box has 1771 GB, sat at
# 90%, and the kernel OOM killer took the whole process group twice (silently -- no traceback, no CUDA
# OOM; torchrun and the wandb service died in the same instant):
#     2026-08-05 07:10  step 1570 @ 1587 GB   (died before the first save; 70 steps lost)
#     2026-08-05 15:40  step 1670 @ 1593 GB   (global_step_1600 banked first)
# ACT_OFFLOAD was calibrated for 80 GB cards ("the defaults OOM at 76/79 GB"); this host has 143 GB H200s
# and ran at only 53 GB/card WITH offload on. Setting ACT_OFFLOAD=False moves that memory back to where
# the headroom is, and is expected to help the 22-49 s/it throughput too (no PCIe round-trip per
# activation). It is NOT a resume invariant and NOT in CONFIG_TAG, so it is safe to flip:
#     ACT_OFFLOAD=False ./examples/msa/resume_p2_k8.sh
# Left at True here so this script reproduces the run as configured; override it on the command line.
set -euo pipefail

export REPO_ROOT=${REPO_ROOT:-/home/aarti_cerebras/dsa/verl_msa}
BASE=/home/aarti_cerebras

# Overridable so this one script serves every variant of this experiment. CONFIG_TAG keys the checkpoint
# dir, so it is what decides WHICH run you resume:
#     ./resume_p2_k8.sh                        -> the original run
#     CONFIG_TAG=..._v2 ./resume_p2_k8.sh      -> the _v2 run
# The v2 run's own overrides.yaml is identical to v1's in every knob below (tiers/width/seed/workers/
# batch/world size/MSA geometry/optimizer) -- it differs ONLY in default_local_dir and experiment_name --
# so the same invariants apply verbatim to both. Verified against
#   <v2>/hydra/20260805_221955/.hydra/overrides.yaml
CONFIG_TAG=${CONFIG_TAG:-p2_qwen3-4b-thinking-2507_ph2b_split_v1_L32k_bs8_k8_B128_dp3_lam1.0_lr5e-6_ilr1e-4_t16w2048_st11214}

export CKPT_DIR="${CKPT_DIR:-${BASE}/msa/${CONFIG_TAG}}"
export CONFIG_TAG                                  # pin it; belt-and-braces vs. the derived tag
export MODEL_PATH="${BASE}/models/qwen3_4b_thinking_2507"
export TRAIN_FILES="${BASE}/msa/qwen3-4b-thinking-2507__ph2b_split_v1"
export VAL_FILES="${BASE}/msa/qwen3-4b-thinking-2507__ph2b_split_v1"
export WARMSTART="${BASE}/msa/indexer_warmup/p1_qwen3-4b-thinking-2507_longmino_1.00Bt_L32k_bs8_k16_B128_dp3_lr1e-3/indexer_full.pt"

# --- resume invariants: identical to the 20260804_163513 launch (hydra/.../overrides.yaml) -----------
# All four fail SILENTLY if changed. The saved dataloader state is a bare batch counter with no dataset
# identity, so a different row order resumes onto unrelated rows.
export NPROC=8            # FSDP shards are world-size-locked (model_world_size_8_rank_*.pt)
export NUM_WORKERS=8      # StatefulDataLoader snapshot is keyed by worker id
export BATCH=8
export LENGTH_TIERS=16
export TIER_WIDTH=2048
export DATA_SEED=1234
export SEQ_LEN=32768
export STEPS=11214

# --- everything else, also from overrides.yaml -------------------------------------------------------
export TOPK=8
export BLOCK_SIZE=128
export INIT_BLOCKS=0
export LOCAL_BLOCKS=1
export DENSE_PREFIX=3
export KL_BLOCK=512
export KL_CKPT=True
export KL_REDUCTION=mean
export KL_LAMBDA=1.0
export DIAG_INTERVAL=10
export LOG_PER_LAYER=true
export GRAD_CKPT=False
export TILED_MLP=True
export TILED_MLP_SHARDS=4
export MODEL_DTYPE=bf16
export ACT_OFFLOAD=${ACT_OFFLOAD:-True}   # see KNOWN ISSUE above; override to False on the command line
export LR=5e-6
export INDEXER_LR=1e-4
export LR_SCHED=cosine
export WARMUP_RATIO=0.03
export MIN_LR_RATIO=0.1
export CLIP_GRAD=1.0
export SAVE_FREQ=100
export MAX_CKPT=5
export TEST_FREQ=150
export VAL_MAX_SAMPLES=256
export VAL_PREFIX=val
export PROJECT=MSA

# --- preflight: fail in a second rather than minutes into model construction -------------------------
[[ -f "${REPO_ROOT}/examples/msa/run_qwen3_msa_phase2.sh" ]] || {
    echo "ERROR: launcher not found under REPO_ROOT=${REPO_ROOT}"; exit 1; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: base model missing at ${MODEL_PATH}"; exit 1; }
# Inert on a resume, but `_warmstart_from_consolidated` does an unguarded torch.load.
[[ -z "${WARMSTART}" || -f "${WARMSTART}" ]] || { echo "ERROR: WARMSTART missing: ${WARMSTART}"; exit 1; }
[[ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]] || { echo "ERROR: no tracker in ${CKPT_DIR}"; exit 1; }
_step=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt")
[[ -d "${CKPT_DIR}/global_step_${_step}" ]] || { echo "ERROR: tracker says ${_step} but that dir is missing"; exit 1; }
for _kind in model optim; do
    _n=$(ls -1 "${CKPT_DIR}/global_step_${_step}"/${_kind}_world_size_8_rank_*.pt 2>/dev/null | wc -l)
    [[ "${_n}" == "8" ]] || { echo "ERROR: expected 8 ${_kind} shards in global_step_${_step}, found ${_n}"; exit 1; }
done
echo "[resume] repo=${REPO_ROOT} ($(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo 'not a git repo'))"
echo "[resume] continuing ${CONFIG_TAG} from global_step_${_step} -> ${STEPS}  (ACT_OFFLOAD=${ACT_OFFLOAD})"

# MUST run from REPO_ROOT -- two separate things silently resolve against the CWD, not REPO_ROOT, and
# both would pull code from whichever checkout you happened to `cd` into:
#   1. `python -m verl.trainer.sft_trainer` puts the CWD FIRST on sys.path, ahead of the PYTHONPATH the
#      launcher exports. Launch this from ~/dsa/verl and you would import the verl package from THERE
#      while believing you were running verl_msa -- defeating the entire point of this script.
#   2. the launcher passes `data.custom_cls.path=verl/utils/dataset/msa_sft_dataset.py` as a RELATIVE
#      path, which load_extern_object resolves against the CWD. That file defines the ROW ORDER, so the
#      wrong copy is a resume-correctness bug, not just a provenance one.
# Harmless today (both checkouts are byte-identical in those files) but it will not stay that way.
cd "${REPO_ROOT}"

exec "${REPO_ROOT}/examples/msa/run_qwen3_msa_phase2.sh" "$@"
