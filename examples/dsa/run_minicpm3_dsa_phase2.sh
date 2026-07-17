#!/usr/bin/env bash
# DSA Phase-2 (SPARSE training): fine-tune MiniCPM3-4B + lightning indexer with top-k sparse attention on
# self-generated behavior-cloning trajectories. Loss = LM cross-entropy (trains the base through the sparse
# attention) + lambda * selected-set indexer KL (trains the indexer; detached from the base). See
# docs/dsa_phase2_impl.md (T1-T5) and docs/dsa_phase2_plan.md.
#
# Prereqs:
#   1. Trajectories generated (examples/dsa/run_m2_probe.sh with SPLIT_JSON) -> trajectories.jsonl.
#   2. Convert -> SFT messages parquet (filters total>=top_k, drops runaways):
#        PYTHONPATH=.devlibs/tf457lib:/tmp/pylibs python scripts/dsa/trajectories_to_sft_parquet.py \
#          --input /cb/ml-eng/aarti/dsa/m3a_gen_*/trajectories.jsonl* \
#          --out   /cb/ml-eng/aarti/dsa/m3a_sft.parquet  [--domains Math]
#   3. A (mostly) free 8xH100; transformers 4.57.1 staged in .devlibs; vLLM/torch in the container.
set -xeuo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DEVLIBS=${DEVLIBS:-${REPO_ROOT}/.devlibs/tf457lib}
export PYTHONPATH=${DEVLIBS}:${PYTHONPATH:-}
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://cerebras.wandb.io}
export WANDB_ENTITY=${WANDB_ENTITY:-aartighatkesar}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# vLLM/HF config+cache dirs (avoid the ~/.config/vllm FileNotFound on some nodes); harmless for training too.
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-/cb/ml-eng/aarti/dsa/.vllm/config}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/cb/ml-eng/aarti/dsa/.vllm/cache}

# --- data: the SFT messages parquet from trajectories_to_sft_parquet.py (default MultiTurnSFTDataset) ---
TRAIN_FILES=${TRAIN_FILES:-/cb/ml-eng/aarti/dsa/m3a_sft.parquet}
VAL_FILES=${VAL_FILES:-}
NPROC=${NPROC:-8}
SEQ_LEN=${SEQ_LEN:-4096}               # max sequence length
TRUNCATION=${TRUNCATION:-right}        # samples > SEQ_LEN: 'right' truncate (keep all samples) | 'error' crash
MICRO_BSZ=${MICRO_BSZ:-1}
BATCH=${BATCH:-64}                     # global batch (sequences/step)
STEPS=${STEPS:-500}

# --- DSA sparse config ---
TOPK=${TOPK:-512}                      # sparse key budget (train == deploy)
KL_BLOCK=${KL_BLOCK:-1024}             # query-tile for the sparse forward + selected-set KL
KL_CKPT=${KL_CKPT:-true}               # activation-checkpoint the DSA sparse graph (base is trained -> big graph)
LAMBDA=${LAMBDA:-1.0}                  # indexer-KL weight in the combined loss

# --- optim: two param groups (base + indexer) via the LR schedule; both follow the same shape ---
BASE_LR=${BASE_LR:-7.3e-6}             # main-model LR (paper sparse stage)
INDEXER_LR=${INDEXER_LR:-1e-3}         # lightning-indexer LR
LR_SCHED=${LR_SCHED:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}

# --- memory: base is TRAINED now -> gradient checkpointing on, bf16 base ---
GRAD_CKPT=${GRAD_CKPT:-True}
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
ACT_OFFLOAD=${ACT_OFFLOAD:-False}

# --- init (two independent options; pick one) ---
#  WARMSTART_PATH: a CONSOLIDATED (world-size-agnostic) state dict from consolidate_indexer_ckpt.py —
#    indexer-only (Phase-1 ckpt; base stays stock) OR full base+indexer (Phase-2 ckpt). Loaded via
#    model.load_state_dict(strict=False) BEFORE FSDP wrap => works on ANY GPU count, FRESH optimizer + step 0.
#    This is the normal way to start Phase-2 from a Phase-1 (or branch from a Phase-2) checkpoint.
#  RESUME_PATH: verl NATIVE resume (loads model+optimizer+step); CONTINUES a run but is locked to the SAME
#    GPU count the ckpt was saved with. Use only to resume an interrupted Phase-2 run.
#  Empty both => fresh indexer (the sparse-stage KL warms it).
WARMSTART_PATH=${WARMSTART_PATH:-}
RESUME_PATH=${RESUME_PATH:-}
SAVE_FREQ=${SAVE_FREQ:-${STEPS}}
MAX_CKPT=${MAX_CKPT:-}
TEST_FREQ=${TEST_FREQ:-50}

RUNS_BASE=${RUNS_BASE:-/cb/ml-eng/aarti/dsa/phase2}
RUN_TS=$(date +%Y%m%d_%H%M%S)
DATA_TAG=$(basename "${TRAIN_FILES}" .parquet)
RUN_NAME=${RUN_NAME:-phase2_${DATA_TAG}_k${TOPK}_st${STEPS}_${RUN_TS}}
RUN_DIR=${RUN_DIR:-${RUNS_BASE}/${RUN_NAME}}
mkdir -p "${RUN_DIR}"
LOG_FILE=${LOG_FILE:-${RUN_DIR}/run-${RUN_TS}.log}
export WANDB_DIR="${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[dsa-phase2] run_dir=${RUN_DIR} host=$(hostname)"
echo "[dsa-phase2] argv: $0 $*"
echo "[dsa-phase2] env: NPROC=${NPROC} SEQ_LEN=${SEQ_LEN} MICRO_BSZ=${MICRO_BSZ} BATCH=${BATCH} STEPS=${STEPS}" \
     "TOPK=${TOPK} KL_BLOCK=${KL_BLOCK} LAMBDA=${LAMBDA} BASE_LR=${BASE_LR} INDEXER_LR=${INDEXER_LR}" \
     "LR_SCHED=${LR_SCHED} WARMUP_RATIO=${WARMUP_RATIO} MIN_LR_RATIO=${MIN_LR_RATIO} GRAD_CKPT=${GRAD_CKPT}" \
     "MODEL_DTYPE=${MODEL_DTYPE} ACT_OFFLOAD=${ACT_OFFLOAD} TRAIN_FILES=${TRAIN_FILES} VAL_FILES=${VAL_FILES:-none}" \
     "RESUME_PATH=${RESUME_PATH:-none} SAVE_FREQ=${SAVE_FREQ} PYTHONPATH=${PYTHONPATH}"

if [[ -n "${VAL_FILES}" ]]; then
    VAL_ARGS=(data.val_files="${VAL_FILES}" trainer.test_freq="${TEST_FREQ}")
else
    VAL_ARGS=(trainer.test_freq=-1)
fi
RESUME_ARGS=()
if [[ -n "${RESUME_PATH}" ]]; then
    RESUME_ARGS=(trainer.resume_mode=resume_path trainer.resume_from_path="${RESUME_PATH}")
    echo "[dsa-phase2] NATIVE resume (model+optim+step, same GPU count) from ${RESUME_PATH}"
fi
# warm-start weights (consolidated; fresh optimizer/step; any GPU count) go into the DSA override_config
DSA_WARMSTART_KV=""
if [[ -n "${WARMSTART_PATH}" ]]; then
    DSA_WARMSTART_KV=", dsa_warmstart_path: ${WARMSTART_PATH}"
    echo "[dsa-phase2] WARM-START weights from ${WARMSTART_PATH} (fresh optimizer, step 0)"
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
    data.truncation="${TRUNCATION:-right}"   # right-truncate the rare >SEQ_LEN samples (default 'error' would crash)
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
    optim.lr_warmup_steps_ratio="${WARMUP_RATIO}"
    optim.min_lr_ratio="${MIN_LR_RATIO}"
    trainer.total_training_steps="${STEPS}"
    trainer.project_name=DSA
    trainer.experiment_name="${RUN_NAME}"
    trainer.logger='["console","wandb"]'
    trainer.save_freq="${SAVE_FREQ}"
    ${MAX_CKPT:+trainer.max_ckpt_to_keep="${MAX_CKPT}"}
    "${VAL_ARGS[@]}"
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    trainer.n_gpus_per_node="${NPROC}"
    "$@"
)

{ set +x; } 2>/dev/null
echo "[dsa-phase2] ===== EXACT LAUNCH ARGV ====="; printf '  %q' "${LAUNCH[@]}"; echo
echo "[dsa-phase2] ============================="
{ set -x; } 2>/dev/null

"${LAUNCH[@]}"
