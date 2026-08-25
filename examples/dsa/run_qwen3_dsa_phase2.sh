#!/usr/bin/env bash
# Qwen3 DSA Phase-2 (sparse adaptation): sparsity ON, base UNFROZEN. Loss = LM cross-entropy through the
# token-sparse attention + lambda * the selected-set indexer KL.
#
# This is the published recipe. DeepSeek-V3.2 §2.1.1: "We train both the main model and the indexer",
# "the training signal of the indexer is from only L^I, while the optimization of the main model is
# according to only the language modeling loss", and "we detach the indexer input from the computational
# graph". Keye-VL-2.0 Eq. 7 is the same shape. Neither paper has a frozen-base sparse stage.
#
# The Phase-2 KL teacher is FREE: in the sparse stage the main attention *is* sparse, so DeepSeek's `p`
# construction applied to it is exactly the sparse softmax's own head-average. Do NOT port
# minicpm_dsa._sparse_indexer_kl, which recomputes a dense pass (8.8 TFLOP/layer at 32K -- 8x the attention
# it supervises) and lands on a different target. See plan_v2.md §4.1.
#
# Prereqs:
#   1. a Phase-1 checkpoint, consolidated to a world-size-agnostic indexer state dict:
#        python3 scripts/dsa/consolidate_indexer_ckpt.py --ckpt-dir <phase1>/global_step_N \
#          --out <phase1>/indexer_full.pt
#      passed as WARMSTART. REQUIRED: starting sparse from a random indexer routes attention to noise, which
#      is the entire purpose of the warm-up.
#   2. the behaviour-cloning parquet: PRE-TOKENIZED `input_ids` + `loss_mask` (prompt masked, the model's own
#      trace trained), already built for this model:
#        /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__dolci-think-rl-32b__ph2b_full93889_L32768_*
#      Read by MSASFTDataset, NOT MultiTurnSFTDataset and NOT PackedPretrainDataset:
#        * MultiTurnSFTDataset re-templates per message and DELETES Qwen3 `<think>` traces -- training a
#          Thinking model on trace-stripped data teaches it not to think, and it only shows up much later as
#          a reasoning regression (memory: qwen3-phase2-dolci-rl-data);
#        * PackedPretrainDataset forces loss_mask to all-ones and drops rows shorter than max_length, and BC
#          rows are variable-length (p50 ~7K), so it would silently discard the whole dataset.
#
# See docs/qwen3_4b_dsa/plan_v2.md §4.
set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
SCRIPT_ARGV="$*"
source "$(dirname "${BASH_SOURCE[0]}")/_qwen3_dsa_common.sh"

export WANDB_BASE_URL=https://cerebras.wandb.io
export WANDB_ENTITY=aartighatkesar
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-/cb/ml-eng/aarti/models/qwen3_4b_thinking_2507}
DATA_DIR=${DATA_DIR:-/cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__dolci-think-rl-32b__ph2b_full93889_L32768_20260730_231930}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}}
NPROC=${NPROC:-8}
# Rows are VARIABLE length here (one conversation each), unlike Phase 1 where every row was exactly
# SEQ_LEN. So SEQ_LEN is a CAP, not a required length.
SEQ_LEN=${SEQ_LEN:-32768}
BATCH=${BATCH:-8}
STEPS=${STEPS:-10000}

# --- indexer architecture: MUST match the Phase-1 checkpoint being warm-started, or the state dict will
#     load into differently-shaped modules. The warm-start asserts on unexpected keys but cannot catch a
#     silent shape-compatible mismatch (e.g. a different rope width), so keep these in sync by hand.
N_HEADS=${N_HEADS:-16}
HEAD_DIM=${HEAD_DIM:-64}
ROPE_HEAD_DIM=${ROPE_HEAD_DIM:-64}
TOPK=${TOPK:-2048}                     # selected TOKENS per query. NOW LOAD-BEARING (Phase 1 only used it
                                       # for diagnostics). 2048 matches MSA k16 for an apples-to-apples
                                       # comparison -- but see the review: MSA's own conclusion was that the
                                       # binding constraint was BUDGET, so run 4096 as a primary arm too.
DENSE_PREFIX=${DENSE_PREFIX:-4}        # layers [0, DENSE_PREFIX) stay DENSE (no indexer, no KL, and in
                                       # THIS phase they keep running full attention while the rest go
                                       # sparse). MUST equal the Phase-1 value: the warm-start loads
                                       # `*.indexer.*` by name with strict=False, so a mismatch is not a
                                       # shape error -- a larger prefix here silently drops trained
                                       # indexers, a smaller one leaves fresh ones untrained. See the
                                       # DENSE_PREFIX comment in run_qwen3_dsa_phase1.sh for why 4.
SPARSE_LAYERS=${SPARSE_LAYERS:-}       # explicit layer ids instead of a prefix; overrides DENSE_PREFIX
FP8=${FP8:-True}
FP8_UE8M0=${FP8_UE8M0:-True}

# --- loss / memory knobs
KL_BLOCK=${KL_BLOCK:-256}              # query tile. LOWER than Phase 1's 512 on purpose: the gathered K/V
                                       # are [1, H_kv, T_q, k, d] bf16 = 2.15 GB EACH at 512, and Phase 2
                                       # also trains the base.
KL_CKPT=${KL_CKPT:-True}               # recompute the whole sparse tile (attention + KL) in backward.
                                       # REQUIRED at 32K -- the base is trained here, so the gathered-KV
                                       # graph is the memory peak.
KL_REDUCTION=${KL_REDUCTION:-mean}
KL_LAMBDA=${KL_LAMBDA:-1.0}            # lambda on the KL. No published value in either paper. Note it is
                                       # NEARLY INERT: the indexer and base parameter sets are disjoint and
                                       # AdamW is scale-invariant, so lambda only rescales the indexer's
                                       # effective LR (memory: msa-phase2-kl-lambda-nearly-inert).
                                       # DO NOT spend compute sweeping it.
FULL_SUPPORT_PROB=${FULL_SUPPORT_PROB:-0.0}  # fraction of forwards that take the KL over the FULL causal
                                       # support instead of the selected set. The restricted KL gives the
                                       # indexer no gradient about tokens it failed to select -- ~6% of
                                       # columns get gradient at k=2048/T=32K -- so the ranking outside the
                                       # top-k can decalibrate, and Phase 2 shifts the data distribution at
                                       # the same time. The review argues for 0.05 rather than 0.0; costs one
                                       # dense teacher pass when it fires.
COMPILE_TEACHER=${COMPILE_TEACHER:-True}
DIAG_INTERVAL=${DIAG_INTERVAL:-10}
LOG_PER_LAYER=${LOG_PER_LAYER:-true}

GRAD_CKPT=${GRAD_CKPT:-False}          # MUST stay False even though the base now trains, for a CORRECTNESS
                                       # reason, not a memory one: HF gradient checkpointing runs the first
                                       # pass under no_grad, so `self._dsa_kl` would be stashed WITHOUT a
                                       # graph and the KL would contribute EXACTLY ZERO gradient to the
                                       # indexer -- silently, with a healthy-looking loss curve. The module
                                       # checkpoints its own attention+KL region instead (KL_CKPT). This is
                                       # the same conclusion the MSA Phase-2 launcher reached; plan §4 lists
                                       # it under "details to carry over" and the first version of this
                                       # script shipped True anyway.
                                       # Separately, it also OOM'd: the layer-level checkpoint wrapping our
                                       # already-checkpointed tiles meant every tile's gathered K/V stayed
                                       # live during the layer's recompute (n_tiles x per_tile is invariant
                                       # in KL_BLOCK, which is why shrinking the tile never moved the
                                       # ceiling -- 3 OOMs at 78.7/79.2 GiB before this was found).
TILED_MLP=${TILED_MLP:-True}           # shard the MLP forward/backward: the replacement for the activation
                                       # savings GRAD_CKPT would have provided, now that the base trains.
                                       # Applied in monkey_patch.py:318, well before the DSA branch at :519,
                                       # so our early return preserves it.
TILED_MLP_SHARDS=${TILED_MLP_SHARDS:-4}
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
ACT_OFFLOAD=${ACT_OFFLOAD:-True}       # REQUIRED at 32K (memory: msa-phase2-32k-needs-act-offload -- the
                                       # defaults OOM at 76/79 GB). Needs ~1720 GB HOST RAM, which picks the
                                       # node; a host-OOM kill presents as a rendezvous-heartbeat error with
                                       # no traceback (memory: msa-phase2-host-ram-1720gb).
SAVE_FREQ=${SAVE_FREQ:-500}
MAX_CKPT=${MAX_CKPT:-}

LR=${LR:-5e-6}                         # PEAK lr for the BASE. DeepSeek-V3.2's sparse stage uses 7.3e-6.
INDEXER_LR=${INDEXER_LR:-1e-4}         # separate lr for *.indexer.* (Phase 1 trained it at 1e-3). Handled by
                                       # the engine's existing two-group path (optim.indexer_lr).
LR_SCHED=${LR_SCHED:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}
# See the Phase-1 script for why 0.0. In Phase 2 the base trains too, so if base decay is wanted this is the
# knob to revisit -- and it needs the per-group split in qwen3_dsa.indexer_param_groups to be wired into the
# engine first, because a global non-zero value would decay the indexer norms toward 0 and suppress the branch.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMSTART=${WARMSTART:-}               # REQUIRED: the consolidated Phase-1 indexer
STAGE=${STAGE:-phase2}
EXP_NAME=${EXP_NAME:-phase2}
PROJECT=${PROJECT:-DSA-QWEN3}
RUNS_BASE=${RUNS_BASE:-/cb/ml-eng/aarti/dsa_qwen3/sparse}

VAL_FILES=${VAL_FILES:-}
TEST_FREQ=${TEST_FREQ:-50}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
VAL_PREFIX=${VAL_PREFIX:-val}
RESUME_PATH=${RESUME_PATH:-}

# --- preflight ----------------------------------------------------------------------------------------
[[ -d "${MODEL_PATH}" ]] || { echo "[dsa-phase2] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"; exit 1; }
if [[ -z "${WARMSTART}" ]]; then
    echo "[dsa-phase2] ERROR: WARMSTART is required — sparse training from a random indexer routes attention"
    echo "                    to noise. Consolidate a Phase-1 checkpoint first:"
    echo "  python3 scripts/dsa/consolidate_indexer_ckpt.py --ckpt-dir <phase1>/global_step_N \\"
    echo "    --out <phase1>/indexer_full.pt"
    echo "                    Set WARMSTART=0 explicitly ONLY for a deliberate from-scratch ablation."
    exit 1
fi
[[ "${WARMSTART}" == "0" ]] || [[ -f "${WARMSTART}" ]] || {
    echo "[dsa-phase2] ERROR: WARMSTART file not found: ${WARMSTART}"; exit 1; }
(( 128 % N_HEADS == 0 )) || { echo "[dsa-phase2] ERROR: N_HEADS=${N_HEADS} must divide 128 (serving)"; exit 1; }
if [[ -n "${SPARSE_LAYERS}" && "${DENSE_PREFIX}" != "0" ]]; then
    echo "[dsa-phase2] ERROR: set either DENSE_PREFIX (${DENSE_PREFIX}) or SPARSE_LAYERS (${SPARSE_LAYERS}), not both"
    exit 1
fi
TRAIN_LIST=$(dsa_expand_files "${TRAIN_FILES}" train) || {
    echo "[dsa-phase2] ERROR: no train parquet matched: ${TRAIN_FILES}"; exit 1; }

TAG_EXTRA="_k${TOPK}_${N_HEADS}x${HEAD_DIM}_lam${KL_LAMBDA}_ilr${INDEXER_LR}"
# Appended only when nonzero, so DENSE_PREFIX=0 reproduces the pre-dense_prefix CONFIG_TAG exactly.
if [[ -n "${SPARSE_LAYERS}" ]]; then
    TAG_EXTRA+="_ls$(printf '%s' "${SPARSE_LAYERS}" | md5sum | cut -c1-6)"
elif (( DENSE_PREFIX > 0 )); then
    TAG_EXTRA+="_dp${DENSE_PREFIX}"
fi
dsa_setup_run_identity
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[dsa-phase2] run_dir=${RUN_DIR}"
echo "[dsa-phase2] ckpt_dir=${CKPT_DIR} (stable, keyed on config -> resume_mode=auto continues here)"
echo "[dsa-phase2] existing checkpoints: $(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | wc -l)"
echo "[dsa-phase2] log=${LOG_FILE}  model=${MODEL_PATH}  warmstart=${WARMSTART}"
echo "[dsa-phase2] train_shards=$(tr -cd , <<<"${TRAIN_LIST}" | wc -c | awk '{print $1+1}') spec=${TRAIN_FILES}"
echo "[dsa-phase2] cwd=$(pwd)"
echo "[dsa-phase2] cmdline: $(tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null)"
echo "[dsa-phase2] argv: $0 $*"

MANIFEST_KNOBS=(MODEL_PATH DATA_DIR TRAIN_FILES VAL_FILES NPROC SEQ_LEN STEPS BATCH
                N_HEADS HEAD_DIM ROPE_HEAD_DIM TOPK DENSE_PREFIX SPARSE_LAYERS FP8 FP8_UE8M0
                KL_BLOCK KL_CKPT KL_REDUCTION KL_LAMBDA FULL_SUPPORT_PROB COMPILE_TEACHER
                TILED_MLP TILED_MLP_SHARDS
                DIAG_INTERVAL LOG_PER_LAYER GRAD_CKPT MODEL_DTYPE ACT_OFFLOAD
                LR INDEXER_LR LR_SCHED WARMUP_RATIO MIN_LR_RATIO CLIP_GRAD WEIGHT_DECAY
                SAVE_FREQ MAX_CKPT TEST_FREQ VAL_MAX_SAMPLES VAL_PREFIX
                WARMSTART RESUME_PATH STAGE PROJECT EXP_NAME RUNS_BASE
                CUDA_VISIBLE_DEVICES PYTHONPATH PYTORCH_CUDA_ALLOC_CONF WANDB_BASE_URL WANDB_ENTITY)
dsa_write_manifest

DSA_OV="dsa_enabled: true, dsa_mode: sparse"
DSA_OV+=", dsa_n_heads: ${N_HEADS}, dsa_head_dim: ${HEAD_DIM}, dsa_rope_head_dim: ${ROPE_HEAD_DIM}"
DSA_OV+=", dsa_top_k: ${TOPK}, dsa_fp8: ${FP8}, dsa_fp8_ue8m0: ${FP8_UE8M0}"
DSA_OV+=", dsa_dense_prefix: ${DENSE_PREFIX}"
[[ -n "${SPARSE_LAYERS}" ]] && DSA_OV+=", dsa_sparse_layers: '${SPARSE_LAYERS}'"
DSA_OV+=", dsa_kl_block_size: ${KL_BLOCK}, dsa_kl_checkpoint: ${KL_CKPT}, dsa_kl_reduction: ${KL_REDUCTION}"
DSA_OV+=", dsa_full_support_kl_prob: ${FULL_SUPPORT_PROB}, dsa_compile_teacher: ${COMPILE_TEACHER}"
DSA_OV+=", dsa_diag_interval: ${DIAG_INTERVAL}, dsa_log_per_layer: ${LOG_PER_LAYER}"
[[ "${WARMSTART}" != "0" ]] && DSA_OV+=", dsa_warmstart_path: '${WARMSTART}'"

if [[ -n "${VAL_FILES}" ]]; then
    VAL_LIST=$(dsa_expand_files "${VAL_FILES}" val) || {
        echo "[dsa-phase2] ERROR: no val parquet matched: ${VAL_FILES}"; exit 1; }
    VAL_ARGS=(data.val_files="${VAL_LIST}" data.val_max_samples="${VAL_MAX_SAMPLES}"
              trainer.test_freq="${TEST_FREQ}" +trainer.val_prefix="${VAL_PREFIX}")
else
    VAL_ARGS=(trainer.test_freq=-1)
fi

RESUME_ARGS=()
[[ -n "${RESUME_PATH}" ]] && RESUME_ARGS=(trainer.resume_mode=resume_path
                                          trainer.resume_from_path="${RESUME_PATH}")

LAUNCH=(
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}"
    -m verl.trainer.sft_trainer
    hydra.run.dir="${RUN_DIR}/hydra/${RUN_TS}"
    trainer.default_local_dir="${CKPT_DIR}"
    +loss_mode=dsa_sparse
    +indexer_kl_lambda="${KL_LAMBDA}"
    data.train_files="${TRAIN_LIST}"
    # MSASFTDataset: pre-tokenized input_ids + loss_mask, variable-length rows. See the header for why
    # neither MultiTurnSFTDataset (strips `<think>`) nor PackedPretrainDataset (drops short rows) works.
    data.custom_cls.path=verl/utils/dataset/msa_sft_dataset.py
    data.custom_cls.name=MSASFTDataset
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
    model.tiled_mlp.enabled="${TILED_MLP}"
    model.tiled_mlp.num_shards="${TILED_MLP_SHARDS}"
    "+model.override_config={${DSA_OV}}"
    engine=fsdp
    engine.strategy=fsdp2
    engine.reshard_after_forward=True
    engine.use_orig_params=True
    engine.model_dtype="${MODEL_DTYPE}"
    optim.lr="${LR}"
    +optim.indexer_lr="${INDEXER_LR}"
    optim.lr_scheduler_type="${LR_SCHED}"
    optim.lr_warmup_steps_ratio="${WARMUP_RATIO}"
    optim.min_lr_ratio="${MIN_LR_RATIO}"
    optim.clip_grad="${CLIP_GRAD}"
    optim.weight_decay="${WEIGHT_DECAY}"
    trainer.total_training_steps="${STEPS}"
    trainer.project_name="${PROJECT}"
    trainer.experiment_name="${RUN_NAME}"
    trainer.logger='["console","wandb"]'
    trainer.save_freq="${SAVE_FREQ}"
    ${MAX_CKPT:+trainer.max_ckpt_to_keep="${MAX_CKPT}"}
    "${VAL_ARGS[@]}"
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    trainer.n_gpus_per_node="${NPROC}"
    "$@"
)
dsa_run
