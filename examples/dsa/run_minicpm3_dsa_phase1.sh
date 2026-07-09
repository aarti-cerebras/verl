#!/usr/bin/env bash
# DSA Phase-1 (dense warm-up) REAL-DATA run: train ONLY the lightning indexer on MiniCPM3-4B via verl's
# SFT trainer, on REAL long-context data (InfLLM-V2), ONE document per fixed-length row (the 6a MVP — no
# cross-doc packing / varlen). This is the real-data sibling of run_minicpm3_dsa_phase1_smoke.sh.
#
# Prereq: build the dataset first (re-tokenizes InfLLM text with the MiniCPM3 tokenizer, one doc per row):
#   DEVLIBS=.../.devlibs/tf457lib PYTHONPATH=$DEVLIBS \
#     python examples/dsa/prepare_real_data.py --out data/dsa/infllm_minicpm3_4k.parquet \
#     --seq_len 4096 --num_windows 512
# Also requires: a mostly-free H100, transformers 4.57.1 + fast_hadamard_transform (dev paths below).
# See docs/dsa_phase1_smoke_plan.md and docs/dsa_train_indexer_plan.md (item 6a).
set -xeuo pipefail

# --- env: transformers 4.57.1 staged in-repo (.devlibs, survives container restarts) so it shadows the
#     system transformers 5.x, which can't load MiniCPM3's trust-remote-code modeling. Recreate with:
#     pip install --no-deps --target=.devlibs/tf457lib transformers==4.57.1 'huggingface_hub<1.0'.
#     fast_hadamard is optional (indexer falls back to torch FWHT).
REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}
DEVLIBS=${DEVLIBS:-${REPO_ROOT}/.devlibs/tf457lib}
export PYTHONPATH=${DEVLIBS}:${PYTHONPATH:-}

# --- wandb -> Cerebras self-hosted instance (API key is in ~/.netrc; never hardcode it) ---
export WANDB_BASE_URL=https://cerebras.wandb.io
export WANDB_ENTITY=aartighatkesar

# seq_len MUST match the parquet's window length (prepare_real_data.py --seq_len); rows are pre-tokenized
# to exactly this many tokens, so a larger value skips every row and a smaller one truncates.
TRAIN_FILES=${TRAIN_FILES:-${REPO_ROOT}/data/dsa/infllm_minicpm3_4k.parquet}
NPROC=${NPROC:-1}
SEQ_LEN=${SEQ_LEN:-4096}
STEPS=${STEPS:-200}
BATCH=${BATCH:-8}                      # global batch (windows/step); 512 windows -> ~64 steps/epoch
TOPK=${TOPK:-2048}                     # diagnostic only (recall@k in dense_warmup); track Phase-2 deploy k
GRAD_CKPT=${GRAD_CKPT:-False}          # base is frozen -> not needed; also conflicts with the _dsa_kl side-effect
SAVE_FREQ=${SAVE_FREQ:-${STEPS}}       # set -1 to skip the end-of-run indexer checkpoint
LR=${LR:-1e-3}                         # PEAK lr (cosine warmup+decay); loss is mean-normalized so lr is batch-independent
LR_SCHED=${LR_SCHED:-cosine}           # "cosine" (warmup -> peak -> cosine decay to MIN_LR_RATIO*peak) or "constant"
WARMUP_RATIO=${WARMUP_RATIO:-0.03}     # fraction of total steps spent warming up 0 -> peak (~30 steps at 1000)
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}      # cosine decays to this fraction of peak by the final step
EXP_NAME=${EXP_NAME:-phase1-real}      # base name; the run dir + wandb experiment_name append a timestamp

# --- held-out validation (item 3). Empty VAL_FILES => no eval (TEST_FREQ forced -1). To validate, point
#     VAL_FILES at a same-SEQ_LEN val parquet from build_phase_a_datasets.sh, e.g.:
#       VAL_FILES=${REPO_ROOT}/data/dsa/phase_a/infllm_minicpm3_${SEQ_LEN}_val.parquet TEST_FREQ=25 ...
#     The val loop logs val/loss + the val/indexer/* & val/attn/* panel (recall/overlap/entropy).
#     NOTE: the val parquet MUST be built at the same SEQ_LEN (it is truncated to data.max_length).
VAL_FILES=${VAL_FILES:-}               # val parquet; empty => validation disabled
TEST_FREQ=${TEST_FREQ:-25}             # steps between eval (or "after_each_epoch"); ignored if VAL_FILES empty
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1} # cap val windows (-1 = all)
VAL_PREFIX=${VAL_PREFIX:-val}          # wandb section for val metrics (e.g. val_ood, val_ood_L2048)

# --- val-only mode: skip training, evaluate a trained indexer checkpoint on VAL_FILES, then exit. Use to
#     run the OOD eval LATER against a saved run. Requires RESUME_PATH (a checkpoint dir) + VAL_FILES built
#     at the same SEQ_LEN. Example (OOD at 2048):
#       VAL_ONLY=1 RESUME_PATH=dsa_runs/<run>/checkpoints/global_step_200 SEQ_LEN=2048 \
#         VAL_FILES=data/dsa/phase_a/ood_ultrafineweb_minicpm3_L2048.parquet VAL_PREFIX=val_ood_L2048 \
#         examples/dsa/run_minicpm3_dsa_phase1.sh
VAL_ONLY=${VAL_ONLY:-0}                 # 1 => eval-only from RESUME_PATH, no training
RESUME_PATH=${RESUME_PATH:-}            # checkpoint dir to evaluate (required when VAL_ONLY=1)

if [[ "${VAL_ONLY}" == "1" ]]; then
    [[ -n "${VAL_FILES}"  ]] || { echo "[dsa-phase1] ERROR: VAL_ONLY=1 requires VAL_FILES"; exit 1; }
    [[ -n "${RESUME_PATH}" ]] || { echo "[dsa-phase1] ERROR: VAL_ONLY=1 requires RESUME_PATH"; exit 1; }
    TRAIN_FILES="${VAL_FILES}"          # train set is built (engine needs it) but never iterated in val-only
    SAVE_FREQ=-1                         # never checkpoint an eval-only run
fi

# --- all logs + artifacts under the repo, in one TIMESTAMPED run subfolder (never clobbers a prior run) ---
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=${RUN_NAME:-${EXP_NAME}-${RUN_TS}}
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/dsa_runs/${RUN_NAME}}
mkdir -p "${RUN_DIR}"
LOG_FILE=${LOG_FILE:-${RUN_DIR}/run-${RUN_TS}.log}
export WANDB_DIR="${RUN_DIR}"                 # wandb local files -> ${RUN_DIR}/wandb
# tee ALL stdout+stderr (incl. the `set -x` command trace + per-step KL) to the run log
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[dsa-phase1] run_dir=${RUN_DIR}"
echo "[dsa-phase1] log=${LOG_FILE}"
echo "[dsa-phase1] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} seq_len=${SEQ_LEN} steps=${STEPS} batch=${BATCH} lr=${LR} sched=${LR_SCHED} warmup=${WARMUP_RATIO} min_lr_ratio=${MIN_LR_RATIO}"
echo "[dsa-phase1] train_files=${TRAIN_FILES} wandb=${WANDB_BASE_URL}/${WANDB_ENTITY}/DSA"
# --- ALWAYS record EXACTLY how this run was launched: the outer command line + every env var the script
#     consumes (incl. debug flags that never reach the torchrun trace). Makes any log self-reproducing. ---
echo "[dsa-phase1] cwd=$(pwd)"
echo "[dsa-phase1] cmdline: $(tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null)"
echo "[dsa-phase1] argv: $0 $*"
echo "[dsa-phase1] env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NPROC=${NPROC} SEQ_LEN=${SEQ_LEN}" \
     "STEPS=${STEPS} BATCH=${BATCH} TOPK=${TOPK} LR=${LR} LR_SCHED=${LR_SCHED} WARMUP_RATIO=${WARMUP_RATIO}" \
     "MIN_LR_RATIO=${MIN_LR_RATIO} GRAD_CKPT=${GRAD_CKPT} SAVE_FREQ=${SAVE_FREQ}" \
     "EXP_NAME=${EXP_NAME} TRAIN_FILES=${TRAIN_FILES} VAL_FILES=${VAL_FILES:-none}" \
     "TEST_FREQ=${TEST_FREQ} VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES} VAL_PREFIX=${VAL_PREFIX}" \
     "VAL_ONLY=${VAL_ONLY} RESUME_PATH=${RESUME_PATH:-none} DSA_DEBUG_MASTER=${DSA_DEBUG_MASTER:-unset}" \
     "DSA_DEBUG_WEIGHTS=${DSA_DEBUG_WEIGHTS:-unset} PYTHONPATH=${PYTHONPATH:-}"

# Validation args: only enable eval when VAL_FILES is set (an empty val_dataloader + test_freq>0 would
# crash the val loop). With no VAL_FILES, force test_freq=-1 regardless of TEST_FREQ.
if [[ -n "${VAL_FILES}" ]]; then
    VAL_ARGS=(
        data.val_files="${VAL_FILES}"
        data.val_max_samples="${VAL_MAX_SAMPLES}"
        trainer.test_freq="${TEST_FREQ}"
        +trainer.val_prefix="${VAL_PREFIX}"
    )
    echo "[dsa-phase1] validation ON: val_files=${VAL_FILES} test_freq=${TEST_FREQ} val_max_samples=${VAL_MAX_SAMPLES} prefix=${VAL_PREFIX}"
else
    VAL_ARGS=( trainer.test_freq=-1 )
    echo "[dsa-phase1] validation OFF (set VAL_FILES to enable)"
fi

# val-only: skip training, evaluate RESUME_PATH on VAL_FILES, then exit (validate() short-circuits fit()).
EVAL_ARGS=()
if [[ "${VAL_ONLY}" == "1" ]]; then
    EVAL_ARGS=(
        +trainer.val_only=true
        trainer.resume_mode=resume_path
        trainer.resume_from_path="${RESUME_PATH}"
    )
    echo "[dsa-phase1] VAL_ONLY: eval ${RESUME_PATH} on ${VAL_FILES} -> wandb section '${VAL_PREFIX}'"
fi

# NOTE: engine.reshard_after_forward=True (below) is correct now that the indexer is wrapped as its own
# FSDP2 unit with reshard=False (Option B2; see docs/dsa_fsdp_sharding_notes.md §4). Do NOT re-add the old
# reshard_after_forward=False workaround: it was a Phase-1 dodge for the grad bug that B2 now fixes at the
# root, and forcing it False only keeps the whole frozen base resident per rank (wasted memory).
LAUNCH=(
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}"
    -m verl.trainer.sft_trainer
    hydra.run.dir="${RUN_DIR}/hydra/${RUN_TS}"
    trainer.default_local_dir="${RUN_DIR}/checkpoints"
    +loss_mode=indexer_kl
    data.train_files="${TRAIN_FILES}"
    data.custom_cls.path=verl/utils/dataset/packed_pretrain_dataset.py
    data.custom_cls.name=PackedPretrainDataset
    data.pad_mode=no_padding
    data.max_length="${SEQ_LEN}"
    data.micro_batch_size_per_gpu=1
    data.train_batch_size="${BATCH}"
    data.use_dynamic_bsz=False
    model.path=openbmb/MiniCPM3-4B
    model.trust_remote_code=True
    model.use_remove_padding=False
    model.enable_gradient_checkpointing="${GRAD_CKPT}"
    "+model.override_config={dsa_enabled: true, dsa_n_heads: 16, dsa_head_dim: 64, dsa_rope_head_dim: 32, dsa_top_k: ${TOPK}, dsa_mode: dense_warmup, dsa_kl_block_size: 1024, dsa_fp8: true, dsa_diag_interval: 5, dsa_log_per_layer: true}"
    engine=fsdp
    engine.strategy=fsdp2
    engine.reshard_after_forward=True
    engine.use_orig_params=True
    optim.lr="${LR}"
    optim.lr_scheduler_type="${LR_SCHED}"
    optim.lr_warmup_steps_ratio="${WARMUP_RATIO}"
    optim.min_lr_ratio="${MIN_LR_RATIO}"
    trainer.total_training_steps="${STEPS}"
    trainer.project_name=DSA
    trainer.experiment_name="${RUN_NAME}"
    trainer.logger='["console","wandb"]'
    trainer.save_freq="${SAVE_FREQ}"
    "${VAL_ARGS[@]}"
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
    trainer.n_gpus_per_node="${NPROC}"
    "$@"
)

# --- log the EXACT, fully-resolved torchrun argv (printf %q) so the log self-reproduces (memory
#     log-full-invocation). The env block above already captured every consumed knob. ---
{ set +x; } 2>/dev/null
{
    echo "[dsa-phase1] ===== EXACT LAUNCH ARGV ====="
    printf '  %q' "${LAUNCH[@]}"; echo
    echo "[dsa-phase1] ============================="
}
{ set -x; } 2>/dev/null

"${LAUNCH[@]}"
