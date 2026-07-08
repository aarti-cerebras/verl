#!/usr/bin/env bash
# DSA Phase-1 (dense warm-up) OVERFIT run: train ONLY the lightning indexer on MiniCPM3-4B over a tiny
# fixed set of REAL InfLLM docs (10 docs), repeated for many epochs. This is the meaningful learning test
# (synthetic random data cannot be distilled — see docs/dsa_grad_norm_debugging.md): the indexer KL should
# fall steadily and, with Option B2, the OPTIMIZER MASTERS must actually move.
#
# It also serves as the end-to-end validation of Option B2 (indexer wrapped as its own FSDP2 unit; see
# docs/dsa_fsdp_sharding_notes.md §3b/§4). Run with DSA_DEBUG_MASTER=1 (default) to print the per-step
# [dsa-master] probe: the indexer masters must show grad_norm != None, live Adam state, and max|Δ init| > 0.
#
# Requires: a mostly-free H100, transformers 4.57.1 (dev paths below), and the overfit parquet at
# data/dsa/infllm_minicpm3_4k_10.parquet.
set -xeuo pipefail

# --- env: transformers 4.57.1 staged in-repo (.devlibs) shadows system transformers 5.x (can't load
#     MiniCPM3 trust-remote-code). See memory minicpm3-transformers5-incompat. ---
DEVLIBS=${DEVLIBS:-/cb/home/aarti/ws/code/ws_repos/dsa/verl/.devlibs/tf457lib}
export PYTHONPATH=${DEVLIBS}:${PYTHONPATH:-}

# --- wandb -> Cerebras self-hosted instance (API key in ~/.netrc; never hardcode it) ---
export WANDB_BASE_URL=https://cerebras.wandb.io
export WANDB_ENTITY=aartighatkesar

# --- knobs (env-overridable) ---
REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}
TRAIN_FILES=${TRAIN_FILES:-${REPO_ROOT}/data/dsa/infllm_minicpm3_4k_10.parquet}
NPROC=${NPROC:-1}
SEQ_LEN=${SEQ_LEN:-4096}
EPOCHS=${EPOCHS:-30}                    # repeat the 10-doc set this many times (overfit)
STEPS=${STEPS:-1000}                    # hard cap on optimizer steps (EPOCHS usually binds first)
BATCH=${BATCH:-8}
LR=${LR:-8e-3}
GRAD_CKPT=${GRAD_CKPT:-False}
EXP_NAME=${EXP_NAME:-phase1-overfit-b2}
# The [dsa-master] probe is the whole point of this run (does the OPTIMIZER master move under B2?) -> on by default.
export DSA_DEBUG_MASTER=${DSA_DEBUG_MASTER:-1}

# --- all logs + artifacts under the repo, in one run subfolder ---
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/dsa_runs/${EXP_NAME}-${RUN_TS}}
mkdir -p "${RUN_DIR}"
LOG_FILE=${LOG_FILE:-${RUN_DIR}/run-${RUN_TS}.log}
export WANDB_DIR="${RUN_DIR}"
# tee ALL stdout+stderr (incl. the set -x trace + per-step KL + [dsa-master]) to the run log
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[dsa-overfit] run_dir=${RUN_DIR}"
echo "[dsa-overfit] log=${LOG_FILE}"
echo "[dsa-overfit] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} seq_len=${SEQ_LEN} epochs=${EPOCHS} batch=${BATCH} lr=${LR}"

# NOTE: engine.reshard_after_forward=True is correct because the indexer is wrapped as its OWN FSDP2 unit
# with reshard=False (Option B2; docs/dsa_fsdp_sharding_notes.md §4). Do NOT revert to =False: that was the
# Phase-1 grad-bug workaround B2 replaces, and forcing it False only wastes memory (frozen base resident).
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
    '+model.override_config={dsa_enabled: true, dsa_n_heads: 16, dsa_head_dim: 64, dsa_rope_head_dim: 32, dsa_top_k: 2048, dsa_mode: dense_warmup, dsa_kl_block_size: 1024, dsa_fp8: true, dsa_diag_interval: 5, dsa_log_per_layer: true}'
    engine=fsdp
    engine.strategy=fsdp2
    engine.reshard_after_forward=True
    engine.use_orig_params=True
    optim.lr="${LR}"
    optim.lr_scheduler_type=constant
    trainer.total_training_steps="${STEPS}"
    trainer.total_epochs="${EPOCHS}"
    trainer.project_name=DSA
    trainer.experiment_name="${EXP_NAME}-${RUN_TS}"
    trainer.logger='["console","wandb"]'
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.n_gpus_per_node="${NPROC}"
    "$@"
)

# --- log the EXACT, fully-resolved invocation + consumed env so this log self-reproduces (memory
#     log-full-invocation). printf %q emits a copy-pasteable, shell-safe rendering of the ACTUAL argv. ---
{ set +x; } 2>/dev/null
{
    echo "[dsa-overfit] ===== EXACT LAUNCH COMMAND ====="
    echo "# host=$(hostname)  pwd=$(pwd)  date=$(date -Is)"
    echo "# env (consumed):"
    echo "  export DEVLIBS=$(printf '%q' "${DEVLIBS}")"
    echo "  export PYTHONPATH=$(printf '%q' "${PYTHONPATH}")"
    echo "  export WANDB_BASE_URL=$(printf '%q' "${WANDB_BASE_URL}") WANDB_ENTITY=$(printf '%q' "${WANDB_ENTITY}") WANDB_DIR=$(printf '%q' "${WANDB_DIR}")"
    echo "  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} DSA_DEBUG_MASTER=${DSA_DEBUG_MASTER}"
    echo "# argv:"
    printf '  %q' "${LAUNCH[@]}"; echo
    echo "[dsa-overfit] ==============================="
}
{ set -x; } 2>/dev/null

"${LAUNCH[@]}"
