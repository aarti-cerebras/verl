#!/usr/bin/env bash
# DSA Phase-2 (SPARSE) OVERFIT sanity run: train MiniCPM3-4B + lightning indexer with top-k sparse attention
# on a TINY fixed set of self-gen trajectories (16 Code docs, total 1-4k tokens > top_k=512 so the sparse
# path is genuinely exercised), repeated for many epochs. This is the end-to-end smoke test for T1-T5 on GPU:
#   * the sparse forward + selected-set KL run without NaN/OOM (T1/T2);
#   * dsa_sparse_loss = LM CE + lambda*KL drives training and the combined loss FALLS toward ~0 (overfit);
#   * BOTH param groups move — base at BASE_LR, *.indexer.* at INDEXER_LR (T4 two-group optimizer). The
#     [dsa-master] probe (DSA_DEBUG_MASTER=1) prints per-step grad_norm/Adam-state/|Δinit| for indexer masters.
# It is NOT a capability run (high-ish base LR on 16 docs will overfit/forget) — it only proves the plumbing.
#
# Prereq: data/dsa/phase2_overfit16.parquet (built from the M3a corpus; see docs/dsa_phase2_implementation.md).
set -xeuo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEVLIBS=${DEVLIBS:-${REPO_ROOT}/.devlibs/tf457lib}
export PYTHONPATH=${DEVLIBS}:${PYTHONPATH:-}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://cerebras.wandb.io}
export WANDB_ENTITY=${WANDB_ENTITY:-aartighatkesar}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/cb/ml-eng/aarti/dsa/.vllm/config}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/cb/ml-eng/aarti/dsa/.vllm/cache}
# The two-param-group probe is the point of this run (does the base AND indexer master move?) -> on by default.
export DSA_DEBUG_MASTER=${DSA_DEBUG_MASTER:-1}

# --- data: the tiny overfit parquet (default MultiTurnSFTDataset, response-only loss mask) ---
TRAIN_FILES=${TRAIN_FILES:-${REPO_ROOT}/data/dsa/phase2_overfit16.parquet}
NPROC=${NPROC:-1}
SEQ_LEN=${SEQ_LEN:-4096}                # all 16 docs are <= 4096 -> no truncation
MICRO_BSZ=${MICRO_BSZ:-1}
BATCH=${BATCH:-8}                       # 16 docs / batch 8 = 2 steps/epoch
EPOCHS=${EPOCHS:-50}                    # repeat the 16-doc set (overfit)
STEPS=${STEPS:-500}                     # hard cap (EPOCHS usually binds first)

# --- DSA sparse config ---
TOPK=${TOPK:-512}                       # sparse key budget (train == deploy); docs are 1-4k > 512 -> sparse active
KL_BLOCK=${KL_BLOCK:-1024}              # query-tile for the sparse forward + selected-set KL
KL_CKPT=${KL_CKPT:-true}                # activation-checkpoint the DSA sparse graph (base is trained -> big graph)
LAMBDA=${LAMBDA:-1.0}                   # indexer-KL weight in the combined loss

# --- optim: two param groups. Overfit LRs (higher than the real recipe) so the loss visibly falls fast. ---
BASE_LR=${BASE_LR:-5e-5}                # main-model LR (overfit; real sparse-stage recipe is ~7.3e-6)
INDEXER_LR=${INDEXER_LR:-1e-3}          # lightning-indexer LR
LR_SCHED=${LR_SCHED:-constant}

# --- memory: base is TRAINED now -> gradient checkpointing on, bf16 base ---
GRAD_CKPT=${GRAD_CKPT:-True}
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
ACT_OFFLOAD=${ACT_OFFLOAD:-False}

# --- init: WARMSTART_PATH = consolidated weights (world-size-agnostic, fresh optim/step); RESUME_PATH = native
#     resume (same GPU count). Empty both => fresh indexer. See docs/dsa_ckpt_loading.md. ---
WARMSTART_PATH=${WARMSTART_PATH:-}
RESUME_PATH=${RESUME_PATH:-}

EXP_NAME=${EXP_NAME:-phase2-overfit16}
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR=${RUN_DIR:-${RUNS_BASE:-/cb/ml-eng/aarti/dsa/dsa_runs}/${EXP_NAME}-${RUN_TS}}  # writable NFS w/ space (NOT repo ws)
mkdir -p "${RUN_DIR}"
LOG_FILE=${LOG_FILE:-${RUN_DIR}/run-${RUN_TS}.log}
export WANDB_DIR="${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[dsa-p2-overfit] run_dir=${RUN_DIR} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[dsa-p2-overfit] seq_len=${SEQ_LEN} batch=${BATCH} epochs=${EPOCHS} steps=${STEPS} topk=${TOPK}" \
     "base_lr=${BASE_LR} indexer_lr=${INDEXER_LR} lambda=${LAMBDA} grad_ckpt=${GRAD_CKPT}"

DSA_WARMSTART_KV=""
if [[ -n "${WARMSTART_PATH}" ]]; then
    DSA_WARMSTART_KV=", dsa_warmstart_path: ${WARMSTART_PATH}"
    echo "[dsa-p2-overfit] WARM-START weights from ${WARMSTART_PATH} (fresh optimizer, step 0)"
fi
RESUME_ARGS=()
if [[ -n "${RESUME_PATH}" ]]; then
    RESUME_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="${RESUME_PATH}")
    echo "[dsa-p2-overfit] resuming (warmed indexer) from ${RESUME_PATH}"
fi

LAUNCH=(
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}"
    -m verl.trainer.sft_trainer
    hydra.run.dir="${RUN_DIR}/hydra/${RUN_TS}"
    trainer.default_local_dir="${RUN_DIR}/checkpoints"
    +loss_mode=dsa_sparse
    +indexer_kl_lambda="${LAMBDA}"
    data.train_files="${TRAIN_FILES}"
    data.pad_mode=no_padding
    data.max_length="${SEQ_LEN}"
    data.micro_batch_size_per_gpu="${MICRO_BSZ}"
    data.train_batch_size="${BATCH}"
    data.use_dynamic_bsz=False
    model.path=openbmb/MiniCPM3-4B
    model.trust_remote_code=True
    model.use_remove_padding=False
    model.enable_gradient_checkpointing="${GRAD_CKPT}"
    model.enable_activation_offload="${ACT_OFFLOAD}"
    "+model.override_config={dsa_enabled: true, dsa_n_heads: 16, dsa_head_dim: 64, dsa_rope_head_dim: 32, dsa_top_k: ${TOPK}, dsa_mode: sparse, dsa_kl_block_size: ${KL_BLOCK}, dsa_kl_checkpoint: ${KL_CKPT}, dsa_fp8: true, dsa_diag_interval: 5, dsa_log_per_layer: true${DSA_WARMSTART_KV}}"
    engine=fsdp
    engine.strategy=fsdp2
    engine.reshard_after_forward=True
    engine.use_orig_params=True
    engine.model_dtype="${MODEL_DTYPE}"
    optim.lr="${BASE_LR}"
    +optim.indexer_lr="${INDEXER_LR}"
    optim.lr_scheduler_type="${LR_SCHED}"
    trainer.total_training_steps="${STEPS}"
    trainer.total_epochs="${EPOCHS}"
    trainer.project_name=DSA
    trainer.experiment_name="${EXP_NAME}-${RUN_TS}"
    trainer.logger='["console","wandb"]'
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.n_gpus_per_node="${NPROC}"
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    "$@"
)

{ set +x; } 2>/dev/null
{
    echo "[dsa-p2-overfit] ===== EXACT LAUNCH COMMAND ====="
    echo "# host=$(hostname)  pwd=$(pwd)  date=$(date -Is)"
    echo "# env (consumed):"
    echo "  export DEVLIBS=$(printf '%q' "${DEVLIBS}") PYTHONPATH=$(printf '%q' "${PYTHONPATH}")"
    echo "  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} DSA_DEBUG_MASTER=${DSA_DEBUG_MASTER}"
    echo "# argv:"
    printf '  %q' "${LAUNCH[@]}"; echo
    echo "[dsa-p2-overfit] ==============================="
}
{ set -x; } 2>/dev/null

"${LAUNCH[@]}"
