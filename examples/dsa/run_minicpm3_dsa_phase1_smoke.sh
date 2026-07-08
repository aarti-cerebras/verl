#!/usr/bin/env bash
# DSA Phase-1 (dense warm-up) SMOKE run: train ONLY the lightning indexer on MiniCPM3-4B via verl's SFT
# trainer, single-doc fixed-length data, ~20 steps. Validates the end-to-end verl integration.
#
# Requires: a mostly-free H100, transformers 4.57.1 + fast_hadamard_transform (dev paths below), and a
# tiny parquet (run prepare_smoke_data.py first). See docs/dsa_phase1_smoke_plan.md.
set -xeuo pipefail

# --- env: transformers 4.57.1 staged in-repo (.devlibs, survives container restarts) so it shadows the
#     system transformers 5.x, which can't load MiniCPM3's trust-remote-code modeling (missing
#     is_torch_fx_available). Recreate with: pip install --no-deps --target=.devlibs/tf457lib \
#     transformers==4.57.1 'huggingface_hub<1.0'. fast_hadamard is optional (indexer falls back to torch FWHT).
DEVLIBS=${DEVLIBS:-/cb/home/aarti/ws/code/ws_repos/dsa/verl/.devlibs/tf457lib}
export PYTHONPATH=${DEVLIBS}:${PYTHONPATH:-}

# --- wandb -> Cerebras self-hosted instance (API key is in ~/.netrc; never hardcode it) ---
export WANDB_BASE_URL=https://cerebras.wandb.io
export WANDB_ENTITY=aartighatkesar

TRAIN_FILES=${TRAIN_FILES:-~/data/dsa_smoke/train.parquet}
NPROC=${NPROC:-1}
SEQ_LEN=${SEQ_LEN:-2048}
STEPS=${STEPS:-20}
GRAD_CKPT=${GRAD_CKPT:-False}          # gradient checkpointing (recommended True at long seq_len)
SAVE_FREQ=${SAVE_FREQ:-${STEPS}}       # set -1 to skip the end-of-run checkpoint (e.g. memory probes)
EXP_NAME=${EXP_NAME:-phase1-smoke}     # names the run: dsa_runs/${EXP_NAME}/ + wandb experiment_name

# --- all logs + artifacts under the repo, in one run subfolder ---
REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}
RUN_DIR=${RUN_DIR:-${REPO_ROOT}/dsa_runs/${EXP_NAME}}
RUN_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "${RUN_DIR}"
LOG_FILE=${LOG_FILE:-${RUN_DIR}/run-${RUN_TS}.log}
export WANDB_DIR="${RUN_DIR}"                 # wandb local files -> ${RUN_DIR}/wandb
# tee ALL stdout+stderr (incl. the `set -x` command trace + per-step KL) to the run log
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[dsa-smoke] run_dir=${RUN_DIR}"
echo "[dsa-smoke] log=${LOG_FILE}"
echo "[dsa-smoke] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} seq_len=${SEQ_LEN} steps=${STEPS}"
echo "[dsa-smoke] train_files=${TRAIN_FILES} wandb=${WANDB_BASE_URL}/${WANDB_ENTITY}/DSA"

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
    data.train_batch_size=4
    data.use_dynamic_bsz=False
    model.path=openbmb/MiniCPM3-4B
    model.trust_remote_code=True
    model.use_remove_padding=False
    model.enable_gradient_checkpointing="${GRAD_CKPT}"
    '+model.override_config={dsa_enabled: true, dsa_n_heads: 16, dsa_head_dim: 64, dsa_rope_head_dim: 32, dsa_top_k: 256, dsa_mode: dense_warmup, dsa_kl_block_size: 1024, dsa_fp8: true, dsa_diag_interval: 5}'
    engine=fsdp
    engine.strategy=fsdp2
    engine.reshard_after_forward=True
    engine.use_orig_params=True
    optim.lr=1e-3
    optim.lr_scheduler_type=constant
    trainer.total_training_steps="${STEPS}"
    trainer.project_name=DSA
    trainer.experiment_name="${EXP_NAME}"
    trainer.logger='["console","wandb"]'
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq=-1
    trainer.n_gpus_per_node="${NPROC}"
    "$@"
)

# --- log the EXACT, fully-resolved invocation + consumed env so this log self-reproduces (memory
#     log-full-invocation). printf %q emits a copy-pasteable, shell-safe rendering of the ACTUAL argv. ---
{ set +x; } 2>/dev/null
{
    echo "[dsa-smoke] ===== EXACT LAUNCH COMMAND ====="
    echo "# host=$(hostname)  pwd=$(pwd)  date=$(date -Is)"
    echo "# env (consumed):"
    echo "  export DEVLIBS=$(printf '%q' "${DEVLIBS}")"
    echo "  export PYTHONPATH=$(printf '%q' "${PYTHONPATH}")"
    echo "  export WANDB_BASE_URL=$(printf '%q' "${WANDB_BASE_URL}") WANDB_ENTITY=$(printf '%q' "${WANDB_ENTITY}") WANDB_DIR=$(printf '%q' "${WANDB_DIR}")"
    echo "  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} DSA_DEBUG_MASTER=${DSA_DEBUG_MASTER:-}"
    echo "# argv:"
    printf '  %q' "${LAUNCH[@]}"; echo
    echo "[dsa-smoke] ================================"
}
{ set -x; } 2>/dev/null

"${LAUNCH[@]}"
