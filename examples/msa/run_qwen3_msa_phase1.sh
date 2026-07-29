#!/usr/bin/env bash
# MSA Phase-1 (indexer warm-up) on Qwen3: train ONLY the MSA index branch via verl's SFT trainer, on real
# long-context data, ONE document per fixed-length row (no cross-doc packing / varlen).
#
# Phase 1 runs the main branch DENSE and the index branch as a pure side-channel KL, so the LM forward is
# bit-identical to the stock model (asserted by tests/msa/test_qwen3_msa_phase1.py) and this phase carries
# zero capability risk. Paper: arXiv 2606.13392 §3.2 "Indexer Warmup" — "the model runs full attention in
# both branches and trains the newly added index projections with L_KL ... The same schedule is used when
# sparsifying a pretrained full-attention checkpoint."
#
# Prereq: build the dataset first. Default is the OLMo-3 longmino long-doc mix, re-tokenized with the
# QWEN3 tokenizer, one 32768-token window per document (no cross-doc packing):
#   python3 scripts/msa/build_longmino_dataset.py --model <qwen3> --seq-len 32768 \
#     --target-tokens 3.0e9 --val-windows 2048 --out-dir /cb/ml-eng/aarti/msa/data/longmino_qwen3_32768
# Built 2026-07-28: 92,180 train windows = 3.021B tokens (46 shards) + 2,062 val windows.
# TRAIN_FILES/VAL_FILES accept a directory, a glob, or a single parquet.
#   (0.5B tokens = 15259 windows at 32768; 1B = 30518; 3.02B = 92180.)
#
# See docs/qwen3_4b_msa/plan.md §5 and docs/qwen3_4b_msa/kl_loss.md §4-§5.
set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}
# NOTE: no .devlibs staging here (unlike the MiniCPM3-DSA scripts). Qwen3 is natively supported by the
# system transformers, so nothing has to shadow it. Verified: transformers 5.3.0 + torch 2.11.
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}

# --- wandb -> Cerebras self-hosted instance (API key is in ~/.netrc; never hardcode it) ---
export WANDB_BASE_URL=https://cerebras.wandb.io
export WANDB_ENTITY=aartighatkesar

# Reclaim fragmented HBM; at 32K the KL tiles allocate/free large blocks every layer.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-/cb/ml-eng/aarti/models/qwen3_4b_thinking_2507}
# seq_len MUST match the parquet's window length (prepare_real_data.py --seq_len): rows are pre-tokenized
# to exactly this many tokens, so a larger value skips every row and a smaller one truncates.
DATA_DIR=${DATA_DIR:-/cb/ml-eng/aarti/msa/data/longmino_qwen3_32768}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}}
NPROC=${NPROC:-8}
SEQ_LEN=${SEQ_LEN:-32768}
STEPS=${STEPS:-1000}
BATCH=${BATCH:-8}                      # global batch (windows/step); 8 = 1 per GPU at NPROC=8

# --- MSA architecture knobs (see plan §3.1). Geometry (d_idx, H_kv) is forced from the model config.
TOPK=${TOPK:-16}                       # k selected BLOCKS per (query, group). k*B_k = 2048 tokens at B_k=128.
                                       # Phase 1 is dense, so this only drives the selection DIAGNOSTICS —
                                       # but keep it at the Phase-2 deploy value so recall is comparable.
BLOCK_SIZE=${BLOCK_SIZE:-128}          # B_k; 128 is hardcoded in vLLM's kernels (SPARSE_BLOCK_SIZE)
INIT_BLOCKS=${INIT_BLOCKS:-0}          # forced sink blocks; paper C.2 + M3 ship 0 (the sink is learned)
LOCAL_BLOCKS=${LOCAL_BLOCKS:-1}        # forced local block; consumes ONE of the k slots, not a (k+1)th
DENSE_PREFIX=${DENSE_PREFIX:-3}        # layers [0, DENSE_PREFIX) stay dense (matches M3's [0]*3 + [1]*57)
SPARSE_LAYERS=${SPARSE_LAYERS:-}       # optional explicit layer list, e.g. "0,5,6,...,35" for the §12
                                       # data-driven alternative (dense = {1,2,3,4}). Overrides DENSE_PREFIX.

# --- loss / memory knobs
KL_BLOCK=${KL_BLOCK:-512}              # query tile for the teacher/KL recompute. 512 -> 512 MiB per retained
                                       # fp32 [1, H_kv, 512, 32768] tensor. Drop to 256 if tight. Pure tiling
                                       # granularity: no effect on numerics.
KL_CKPT=${KL_CKPT:-True}               # recompute the per-tile index scores in backward. REQUIRED at 32K.
KL_REDUCTION=${KL_REDUCTION:-mean}     # layer reduction: "mean" (default) | "sum" (paper Algorithm 1).
                                       # Per-layer indexer params are disjoint, so sum == mean * n_layers —
                                       # a pure gradient scale (= LR * n_layers), same optimum. See plan §4.1.
COMPILE_TEACHER=${COMPILE_TEACHER:-True}  # torch.compile the Eq. 9 teacher. Measured on 8xH100/32K: it is
                                       # the single largest step-time win, and validated MORE accurate than
                                       # eager against an fp64 reference (8.4e-05 vs 1.6e-04 max err).
                                       # Set False to bisect an Inductor problem.
DIAG_INTERVAL=${DIAG_INTERVAL:-10}     # block-level selection diagnostics every N forwards (always on in eval)
LOG_PER_LAYER=${LOG_PER_LAYER:-true}   # per-layer captured_over_oracle / block_recall keys — the Phase-1 GATE
                                       # is per layer (>= 0.90), so the layer breakdown is what you read it off

GRAD_CKPT=${GRAD_CKPT:-False}          # MUST stay False: the base is frozen (so it buys nothing) AND
                                       # recomputing a decoder layer re-fires the `_msa_kl` side effect.
MODEL_DTYPE=${MODEL_DTYPE:-bf16}       # dtype the (frozen) base is initialized in. bf16 at 32K; fp32 gives a
                                       # more accurate KL teacher at ~2x memory (fine at short context).
ACT_OFFLOAD=${ACT_OFFLOAD:-False}      # CPU-offload activations saved for backward; transparent, so it does
                                       # NOT re-trigger the `_msa_kl` side effect (unlike GRAD_CKPT).
ACT_GPU_LIMIT=${ACT_GPU_LIMIT:-0}
SAVE_FREQ=${SAVE_FREQ:-${STEPS}}       # -1 to skip the end-of-run indexer checkpoint
MAX_CKPT=${MAX_CKPT:-}
LR=${LR:-1e-3}                         # PEAK lr (cosine); loss is mean-normalized so lr is batch-independent
LR_SCHED=${LR_SCHED:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}            # verl's default. Exposed deliberately: with a side-channel KL the
                                       # grad norm can sit far above the clip (the DSA runs sat at ~330,
                                       # i.e. ~330x clipping every step, making the step effectively
                                       # LR * unit-direction). Watch train/grad_norm and raise if it is
                                       # pinned. See plan §4.1.
WARMSTART=${WARMSTART:-}               # consolidated indexer state dict to resume the BRANCH from (fresh
                                       # optimizer, step 0); see scripts/dsa/consolidate_indexer_ckpt.py
EXP_NAME=${EXP_NAME:-phase1}
PROJECT=${PROJECT:-MSA}

# --- held-out validation. Empty VAL_FILES => no eval (test_freq forced -1). The val loop always computes
#     diagnostics (diag_interval is bypassed in eval mode), so every val batch reports the full
#     val/indexer/* panel incl. captured_over_oracle per layer.
VAL_FILES=${VAL_FILES:-${DATA_DIR}}
TEST_FREQ=${TEST_FREQ:-25}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
VAL_PREFIX=${VAL_PREFIX:-val}

# --- val-only mode: skip training, evaluate a trained indexer checkpoint on VAL_FILES, then exit.
VAL_ONLY=${VAL_ONLY:-0}
RESUME_PATH=${RESUME_PATH:-}

if [[ "${VAL_ONLY}" == "1" ]]; then
    [[ -n "${VAL_FILES}"   ]] || { echo "[msa-phase1] ERROR: VAL_ONLY=1 requires VAL_FILES"; exit 1; }
    [[ -n "${RESUME_PATH}" ]] || { echo "[msa-phase1] ERROR: VAL_ONLY=1 requires RESUME_PATH"; exit 1; }
    TRAIN_FILES="${VAL_FILES}"
    SAVE_FREQ=-1
fi

# --- preflight: fail with an actionable message rather than a cryptic dataloader/model error ---
[[ -d "${MODEL_PATH}" || "${MODEL_PATH}" != /* ]] || {
    echo "[msa-phase1] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"; exit 1; }
# TRAIN_FILES/VAL_FILES may be a single parquet, a glob, or a DIRECTORY (the builder emits 46 shards).
# Expand to the bracketed list hydra wants; PackedPretrainDataset accepts a list of parquet paths.
_expand_files() {  # $1 = path|glob|dir, $2 = shard prefix (train|val)
    local spec="$1" pre="$2" files=()
    if [[ -d "${spec}" ]]; then
        mapfile -t files < <(ls -1 "${spec}"/${pre}-*.parquet 2>/dev/null)
    else
        mapfile -t files < <(ls -1 ${spec} 2>/dev/null)
    fi
    (( ${#files[@]} )) || return 1
    local IFS=,; echo "[${files[*]}]"
}
TRAIN_LIST=$(_expand_files "${TRAIN_FILES}" train) || {
    echo "[msa-phase1] ERROR: no train parquet matched: ${TRAIN_FILES}"
    echo "[msa-phase1] build it with:"
    echo "  python3 scripts/msa/build_longmino_dataset.py --model ${MODEL_PATH} \\"
    echo "    --seq-len ${SEQ_LEN} --target-tokens 3.0e9 --val-windows 2048 \\"
    echo "    --out-dir ${TRAIN_FILES}"
    exit 1; }

# --- run artifacts under RUNS_BASE, one subfolder per run (name encodes stage/data/steps/bsz + timestamp)
RUNS_BASE=${RUNS_BASE:-/cb/ml-eng/aarti/msa/indexer_warmup}
STAGE=${STAGE:-phase1}
RUN_TS=$(date +%Y%m%d_%H%M%S)
DATA_TAG=$(basename "${TRAIN_FILES%/}" .parquet); DATA_TAG=${DATA_TAG%_train}
RUN_NAME=${RUN_NAME:-${STAGE}_${DATA_TAG}_L${SEQ_LEN}_k${TOPK}_st${STEPS}_bs${BATCH}_${RUN_TS}}
RUN_DIR=${RUN_DIR:-${RUNS_BASE}/${RUN_NAME}}
mkdir -p "${RUN_DIR}"
LOG_FILE=${LOG_FILE:-${RUN_DIR}/run-${RUN_TS}.log}
export WANDB_DIR="${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[msa-phase1] run_dir=${RUN_DIR}"
echo "[msa-phase1] log=${LOG_FILE}"
echo "[msa-phase1] model=${MODEL_PATH}"
echo "[msa-phase1] train_shards=$(tr -cd , <<<"${TRAIN_LIST}" | wc -c | awk '{print $1+1}') spec=${TRAIN_FILES} wandb=${WANDB_BASE_URL}/${WANDB_ENTITY}/${PROJECT}"
# --- ALWAYS record EXACTLY how this run was launched: the outer command line + every env var the script
#     consumes (incl. debug flags that never reach the torchrun trace). Makes any log self-reproducing. ---
echo "[msa-phase1] cwd=$(pwd)"
echo "[msa-phase1] cmdline: $(tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null)"
echo "[msa-phase1] argv: $0 $*"
echo "[msa-phase1] env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NPROC=${NPROC} SEQ_LEN=${SEQ_LEN}" \
     "STEPS=${STEPS} BATCH=${BATCH} TOPK=${TOPK} BLOCK_SIZE=${BLOCK_SIZE} INIT_BLOCKS=${INIT_BLOCKS}" \
     "LOCAL_BLOCKS=${LOCAL_BLOCKS} DENSE_PREFIX=${DENSE_PREFIX} SPARSE_LAYERS=${SPARSE_LAYERS:-none}" \
     "KL_BLOCK=${KL_BLOCK} KL_CKPT=${KL_CKPT} KL_REDUCTION=${KL_REDUCTION} DIAG_INTERVAL=${DIAG_INTERVAL}" \
     "LOG_PER_LAYER=${LOG_PER_LAYER} LR=${LR} LR_SCHED=${LR_SCHED} WARMUP_RATIO=${WARMUP_RATIO}" \
     "MIN_LR_RATIO=${MIN_LR_RATIO} CLIP_GRAD=${CLIP_GRAD} GRAD_CKPT=${GRAD_CKPT} MODEL_DTYPE=${MODEL_DTYPE}" \
     "ACT_OFFLOAD=${ACT_OFFLOAD} ACT_GPU_LIMIT=${ACT_GPU_LIMIT} SAVE_FREQ=${SAVE_FREQ} MAX_CKPT=${MAX_CKPT:-all}" \
     "WARMSTART=${WARMSTART:-none} EXP_NAME=${EXP_NAME} PROJECT=${PROJECT} MODEL_PATH=${MODEL_PATH}" \
     "TRAIN_FILES=${TRAIN_FILES} VAL_FILES=${VAL_FILES:-none} TEST_FREQ=${TEST_FREQ}" \
     "VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES} VAL_PREFIX=${VAL_PREFIX} VAL_ONLY=${VAL_ONLY}" \
     "RESUME_PATH=${RESUME_PATH:-none} PYTHONPATH=${PYTHONPATH:-}"

# --- MSA override block. verl's `update_model_config` recurses into nested dict values, so only FLAT
#     scalars survive injection — hence msa_<field> keys (msa_overrides_from_config reads them back).
MSA_OV="msa_enabled: true, msa_mode: dense_warmup"
MSA_OV+=", msa_top_k: ${TOPK}, msa_block_size: ${BLOCK_SIZE}"
MSA_OV+=", msa_init_blocks: ${INIT_BLOCKS}, msa_local_blocks: ${LOCAL_BLOCKS}"
MSA_OV+=", msa_kl_block_size: ${KL_BLOCK}, msa_kl_checkpoint: ${KL_CKPT}, msa_kl_reduction: ${KL_REDUCTION}"
MSA_OV+=", msa_compile_teacher: ${COMPILE_TEACHER}"
MSA_OV+=", msa_diag_interval: ${DIAG_INTERVAL}, msa_log_per_layer: ${LOG_PER_LAYER}"
if [[ -n "${SPARSE_LAYERS}" ]]; then
    MSA_OV+=", msa_sparse_layers: '${SPARSE_LAYERS}'"     # explicit list wins over dense_prefix
else
    MSA_OV+=", msa_dense_prefix: ${DENSE_PREFIX}"
fi
[[ -n "${WARMSTART}" ]] && MSA_OV+=", msa_warmstart_path: '${WARMSTART}'"

if [[ -n "${VAL_FILES}" ]]; then
    VAL_LIST=$(_expand_files "${VAL_FILES}" val) || {
        echo "[msa-phase1] ERROR: no val parquet matched: ${VAL_FILES}"; exit 1; }
    VAL_ARGS=(
        data.val_files="${VAL_LIST}"
        data.val_max_samples="${VAL_MAX_SAMPLES}"
        trainer.test_freq="${TEST_FREQ}"
        +trainer.val_prefix="${VAL_PREFIX}"
    )
    echo "[msa-phase1] validation ON: val_files=${VAL_FILES} test_freq=${TEST_FREQ}"
else
    VAL_ARGS=( trainer.test_freq=-1 )
    echo "[msa-phase1] validation OFF (set VAL_FILES to enable)"
fi

EVAL_ARGS=()
if [[ "${VAL_ONLY}" == "1" ]]; then
    EVAL_ARGS=(
        +trainer.val_only=true
        trainer.resume_mode=resume_path
        trainer.resume_from_path="${RESUME_PATH}"
    )
    echo "[msa-phase1] VAL_ONLY: eval ${RESUME_PATH} on ${VAL_FILES} -> wandb section '${VAL_PREFIX}'"
fi

# NOTE: engine.reshard_after_forward=True is correct BECAUSE each MSAIndexer is wrapped as its own FSDP2
# unit with reshard=False (Option B2; verl/utils/fsdp_utils.py, docs/dsa_fsdp_sharding_notes.md §4). Do NOT
# "fix" a flat loss by setting this False — that was the old Phase-1 dodge and it only keeps the whole
# frozen base resident per rank. If the loss is flat, check that the OPTIMIZER'S masters move:
# grad_norm > 0 is NOT evidence of training under a side-channel loss (§3b).
LAUNCH=(
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}"
    -m verl.trainer.sft_trainer
    hydra.run.dir="${RUN_DIR}/hydra/${RUN_TS}"
    trainer.default_local_dir="${RUN_DIR}/checkpoints"
    +loss_mode=indexer_kl
    data.train_files="${TRAIN_LIST}"
    data.custom_cls.path=verl/utils/dataset/packed_pretrain_dataset.py
    data.custom_cls.name=PackedPretrainDataset
    data.pad_mode=no_padding
    data.max_length="${SEQ_LEN}"
    data.micro_batch_size_per_gpu=1
    data.train_batch_size="${BATCH}"
    data.use_dynamic_bsz=False
    model.path="${MODEL_PATH}"
    model.trust_remote_code=False
    model.use_remove_padding=False
    model.enable_gradient_checkpointing="${GRAD_CKPT}"
    model.enable_activation_offload="${ACT_OFFLOAD}"
    "+model.override_config={${MSA_OV}}"
    engine=fsdp
    engine.strategy=fsdp2
    engine.reshard_after_forward=True
    engine.use_orig_params=True
    engine.model_dtype="${MODEL_DTYPE}"
    optim.lr="${LR}"
    optim.lr_scheduler_type="${LR_SCHED}"
    optim.lr_warmup_steps_ratio="${WARMUP_RATIO}"
    optim.min_lr_ratio="${MIN_LR_RATIO}"
    optim.clip_grad="${CLIP_GRAD}"
    trainer.total_training_steps="${STEPS}"
    trainer.project_name="${PROJECT}"
    trainer.experiment_name="${RUN_NAME}"
    trainer.logger='["console","wandb"]'
    trainer.save_freq="${SAVE_FREQ}"
    ${MAX_CKPT:+trainer.max_ckpt_to_keep="${MAX_CKPT}"}
    "${VAL_ARGS[@]}"
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
    trainer.n_gpus_per_node="${NPROC}"
    "$@"
)

# --- log the EXACT, fully-resolved torchrun argv (printf %q) so the log self-reproduces ---
{ set +x; } 2>/dev/null
{
    echo "[msa-phase1] ===== EXACT LAUNCH ARGV ====="
    printf '  %q' "${LAUNCH[@]}"; echo
    echo "[msa-phase1] ============================="
}
{ set -x; } 2>/dev/null

"${LAUNCH[@]}"
