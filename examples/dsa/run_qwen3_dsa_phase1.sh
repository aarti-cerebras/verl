#!/usr/bin/env bash
# Qwen3 DSA Phase-1 (indexer warm-up): train ONLY the lightning indexer, on real long-context data, one
# document per fixed-length row (no cross-doc packing / varlen).
#
# Phase 1 runs the main attention DENSE and the indexer as a pure side-channel KL, so the LM forward is
# BIT-IDENTICAL to the stock model (asserted by tests/dsa/test_qwen3_dsa.py) and this phase carries zero
# capability risk. DeepSeek-V3.2 §2.1.1: "we keep dense attention and freeze all model parameters except for
# the lightning indexer", target = the main attention "summed across all attention heads[, then]
# L1-normalized along the sequence dimension".
#
# Prereq — the dataset, already built for this exact tokenizer and window length:
#   /cb/ml-eng/aarti/msa/data/longmino_qwen3_32768   (92,180 windows x 32768 tok = 3.02B, 46 shards)
#   1B tokens = 30,518 windows;  2B = 61,036.
# TRAIN_FILES/VAL_FILES accept a directory, a glob, or a single parquet.
#
# See docs/qwen3_4b_dsa/plan_v2.md §2 (architecture), §2.4 (initialization) and §3 (this phase).
set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/cb/home/aarti/ws/code/ws_repos/dsa/verl}
# No .devlibs staging (unlike the MiniCPM3-DSA scripts): Qwen3 is natively supported by the system
# transformers. Verified on transformers 5.3.0 + torch 2.11.
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}
SCRIPT_ARGV="$*"
source "$(dirname "${BASH_SOURCE[0]}")/_qwen3_dsa_common.sh"

# --- wandb -> Cerebras self-hosted instance (API key is in ~/.netrc; never hardcode it) ---
export WANDB_BASE_URL=https://cerebras.wandb.io
export WANDB_ENTITY=aartighatkesar
# Reclaim fragmented HBM: at 32K the KL tiles allocate/free large blocks every layer.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODEL_PATH=${MODEL_PATH:-/cb/ml-eng/aarti/models/qwen3_4b_thinking_2507}
DATA_DIR=${DATA_DIR:-/cb/ml-eng/aarti/msa/data/longmino_qwen3_32768}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}}
NPROC=${NPROC:-8}
# seq_len MUST match the parquet's window length: rows are pre-tokenized to exactly this many tokens, so a
# larger value skips every row and a smaller one truncates.
SEQ_LEN=${SEQ_LEN:-32768}
BATCH=${BATCH:-8}                      # global batch (windows/step); 8 = 1 per GPU at NPROC=8
# 2B tokens = 7630 steps at 8x32768/step. 2B is the converged warm-up budget across BOTH precedents:
# DeepSeek-V3.2 uses 2.1B, Keye-VL-2.0 "approximately 2B multimodal tokens". Gate at ~1B (step 3815) and
# stop early if per-layer topk_recall already clears 0.95.
STEPS=${STEPS:-7630}

# --- indexer architecture (plan §2.1). Geometry is FORCED from the model; these are the free choices.
N_HEADS=${N_HEADS:-16}                 # indexer query heads. Keye's `indexer_num_heads`, and the MiniCPM3
                                       # default -- two independent DSA ports landed on 16. Must divide 128
                                       # (the serving kernel's block_q = 128 // n_heads).
HEAD_DIM=${HEAD_DIM:-64}               # d_idx. Keye's `indexer_head_dim`; must be in {32,64,128} to serve.
                                       # Halves indexer FLOPs vs 128 (12.5% of dense attention, not 25%).
ROPE_HEAD_DIM=${ROPE_HEAD_DIM:-64}     # rope width. 64 == HEAD_DIM => ALL dims roped, mirroring Qwen3's
                                       # fully-rotary attention. 32 (DeepSeek's fraction, which mirrors
                                       # MLA's partly-roped attention) is the documented ablation.
TOPK=${TOPK:-2048}                     # selected TOKENS per query. Phase 1 is dense, so this drives only
                                       # the selection DIAGNOSTICS -- keep it at the Phase-2 deploy value so
                                       # recall is comparable. NOTE the review's open question: MSA's own
                                       # result suggested the binding constraint was BUDGET, not
                                       # granularity, so 4096 belongs in the sweep too.
DENSE_PREFIX=${DENSE_PREFIX:-4}        # layers [0, DENSE_PREFIX) stay DENSE: no indexer, no KL, stock
                                       # attention. 4 comes from this project's own per-layer topk_recall:
                                       # at step ~4150 of the lr1e-4 run L00-L03 were 0.79/0.79/0.82/0.88,
                                       # while every layer from L07 up was >= 0.92 and the top third hit
                                       # 0.95-0.97. Early layers attend broadly and are the hardest to
                                       # sparsify, which is why MSA/M3 ship a dense prefix too
                                       # (msa_indexer.dense_prefix=3; M3's shipped sparse_attention_freq =
                                       # [0]*3 + [1]*57). 32/36 layers stay sparse, so this costs ~11% of
                                       # the attention savings and buys back the four worst layers.
                                       #
                                       # THIS IS ARCHITECTURE, not a training preference: it must match at
                                       # serve time or the model runs dense where it was trained sparse. So
                                       # it is in the CONFIG_TAG (resume_mode=auto must not cross a
                                       # 36-indexer checkpoint with a 32-indexer model) and it is written
                                       # into the serving dir's config.json.
                                       #
                                       # DENSE_PREFIX=0 => every layer sparse: the DeepSeek-V3.2 recipe and
                                       # what the existing p1_*_lr3e-4 / _lr1e-4 checkpoints hold. It also
                                       # reproduces the old CONFIG_TAG exactly (no _dp suffix), which is
                                       # what you need to resume either of those runs.
SPARSE_LAYERS=${SPARSE_LAYERS:-}       # explicit comma-separated layer ids instead of a prefix, e.g.
                                       # "4,8,12". Overrides DENSE_PREFIX; setting both is an error.
SIGMA_TARGET=${SIGMA_TARGET:-0.3}      # init score scale (plan §2.4). weights_proj is scaled PER LAYER by
                                       # 1/rms(input_layernorm.weight) to hit this, which absorbs the
                                       # measured 186x gain spread across the 36 layers. Watch the per-layer
                                       # indexer/entropy_frac on step 1: every layer must be in [0.99, 1.0].
FP8=${FP8:-True}                       # fake-quantized E4M3 score matmul, as the reference does
FP8_UE8M0=${FP8_UE8M0:-True}           # power-of-2 scale, matching the serving kernels. Leaving this off
                                       # cost the MiniCPM3 run ~2% train/serve selection drift.

# --- loss / memory knobs
KL_BLOCK=${KL_BLOCK:-512}              # query tile for the teacher/KL. 512 -> a retained fp32
                                       # [1,512,32768] teacher is 67 MB. Pure tiling granularity: asserted
                                       # not to change the loss.
KL_CKPT=${KL_CKPT:-True}               # recompute the per-tile score graph in backward. REQUIRED at 32K.
KL_REDUCTION=${KL_REDUCTION:-mean}     # layer reduction: "mean" | "sum". Per-layer indexer params are
                                       # disjoint, so this is a pure gradient scale (sum == mean*n_layers),
                                       # same optimum. "mean" keeps grad_norm readable.
COMPILE_TEACHER=${COMPILE_TEACHER:-True}  # torch.compile the head-averaged teacher: the single largest cost
                                       # and bandwidth-bound. MSA measured 2.70x on the equivalent function,
                                       # and MORE accurate than eager vs an fp64 reference. False to bisect.
DIAG_INTERVAL=${DIAG_INTERVAL:-10}     # diagnostics every N forwards (always on in eval, and on step 1)
LOG_PER_LAYER=${LOG_PER_LAYER:-true}   # per-layer topk_recall / entropy_frac -- the Phase-1 gate is PER
                                       # LAYER, so this is what you read it off

GRAD_CKPT=${GRAD_CKPT:-False}          # MUST stay False: the base is frozen (so it buys nothing) AND
                                       # recomputing a decoder layer re-fires the `_dsa_kl` side effect.
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
ACT_OFFLOAD=${ACT_OFFLOAD:-False}      # transparent, so it does NOT re-trigger the side effect (unlike
                                       # GRAD_CKPT). Measure peak first: DSA's teacher is 1/G of MSA's, so
                                       # 32K should fit without it (MSA needed 46.9 GB WITH a per-group
                                       # teacher).
SAVE_FREQ=${SAVE_FREQ:-500}
MAX_CKPT=${MAX_CKPT:-}

LR=${LR:-1e-3}                         # PEAK lr (cosine); the loss is mean-normalized so lr is
                                       # batch-independent. DeepSeek-V3.2 warm-up uses 1e-3.
LR_SCHED=${LR_SCHED:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}            # exposed deliberately: with a side-channel KL the grad norm can sit
                                       # far above the clip (the MiniCPM3 runs sat at ~330, i.e. ~330x
                                       # clipping every step, making the step LR * unit-direction). Watch
                                       # train/grad_norm; the §2.4 per-layer init is partly what keeps this
                                       # in a usable range.
# 0.0 is CORRECT here, not a workaround. Every trainable parameter in Phase 1 is either a norm gain, the
# gate, or a projection whose scale is forward-invisible because a norm follows it -- so decay has nothing
# useful to act on, and on the norms and the gate it is actively harmful (it pulls the gains toward 0,
# suppressing the branch, and pulling weights_proj toward 0 severs the only gradient path into wq/wk). See
# plan §2.4 and qwen3_dsa.indexer_param_groups for the per-group version, which the engine does not yet call.
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMSTART=${WARMSTART:-}               # consolidated indexer state dict to restart the BRANCH from (fresh
                                       # optimizer, step 0); see scripts/dsa/consolidate_indexer_ckpt.py
STAGE=${STAGE:-phase1}
EXP_NAME=${EXP_NAME:-phase1}
PROJECT=${PROJECT:-DSA-QWEN3}
RUNS_BASE=${RUNS_BASE:-/cb/ml-eng/aarti/dsa_qwen3/indexer_warmup}

# --- held-out validation. Empty VAL_FILES => no eval. The val loop always computes diagnostics
#     (diag_interval is bypassed in eval mode), so every val batch reports the full indexer/* panel.
VAL_FILES=${VAL_FILES:-${DATA_DIR}}
# TEST_FREQ/VAL_MAX_SAMPLES are a WALL-CLOCK trap, not just a logging preference. The val set is 2062
# windows, so VAL_MAX_SAMPLES=-1 (all of it) is 258 forward batches, ~73 min -- and at TEST_FREQ=25 that
# fires 305 times over a 7630-step run, i.e. ~370 h of validation against ~70 h of training (84% of
# wall-clock, measured 2026-08-15 before this default was fixed). Eval also bypasses diag_interval, so
# every val batch computes the FULL indexer panel. MSA Phase-1 ran 250/64 across all 7 attempts; 250/256
# keeps that cadence with a tighter per-layer topk_recall estimate at the 1B gate, for ~4.5 h total.
TEST_FREQ=${TEST_FREQ:-250}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
VAL_PREFIX=${VAL_PREFIX:-val}
VAL_ONLY=${VAL_ONLY:-0}
RESUME_PATH=${RESUME_PATH:-}

if [[ "${VAL_ONLY}" == "1" ]]; then
    [[ -n "${VAL_FILES}"   ]] || { echo "[dsa-phase1] ERROR: VAL_ONLY=1 requires VAL_FILES"; exit 1; }
    [[ -n "${RESUME_PATH}" ]] || { echo "[dsa-phase1] ERROR: VAL_ONLY=1 requires RESUME_PATH"; exit 1; }
    TRAIN_FILES="${VAL_FILES}"; SAVE_FREQ=-1
fi

# --- preflight: fail with an actionable message rather than a cryptic dataloader/model error -----------
[[ -d "${MODEL_PATH}" ]] || { echo "[dsa-phase1] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"; exit 1; }
(( 128 % N_HEADS == 0 )) || { echo "[dsa-phase1] ERROR: N_HEADS=${N_HEADS} must divide 128 (serving)"; exit 1; }
case "${HEAD_DIM}" in 32|64|128) ;; *) echo "[dsa-phase1] ERROR: HEAD_DIM must be 32|64|128"; exit 1;; esac
if [[ -n "${SPARSE_LAYERS}" && "${DENSE_PREFIX}" != "0" ]]; then
    echo "[dsa-phase1] ERROR: set either DENSE_PREFIX (${DENSE_PREFIX}) or SPARSE_LAYERS (${SPARSE_LAYERS}), not both"
    exit 1
fi
TRAIN_LIST=$(dsa_expand_files "${TRAIN_FILES}" train) || {
    echo "[dsa-phase1] ERROR: no train parquet matched: ${TRAIN_FILES}"
    echo "[dsa-phase1] build it with:"
    echo "  python3 scripts/msa/build_longmino_dataset.py --model ${MODEL_PATH} \\"
    echo "    --seq-len ${SEQ_LEN} --target-tokens 3.0e9 --val-windows 2048 --out-dir ${TRAIN_FILES}"
    exit 1; }

TAG_EXTRA="_k${TOPK}_${N_HEADS}x${HEAD_DIM}_r${ROPE_HEAD_DIM}"
# Only appended when nonzero: DENSE_PREFIX=0 must reproduce the pre-dense_prefix CONFIG_TAG byte-for-byte,
# or `resume_mode=auto` stops finding the p1_*_lr3e-4 / _lr1e-4 checkpoints. Written with `if` rather than
# `(( )) &&` because a false (( )) returns 1 and `set -e` would kill the script here.
if [[ -n "${SPARSE_LAYERS}" ]]; then
    # Hash, not a count: two different layer SETS of the same size must not share a checkpoint dir.
    TAG_EXTRA+="_ls$(printf '%s' "${SPARSE_LAYERS}" | md5sum | cut -c1-6)"
elif (( DENSE_PREFIX > 0 )); then
    TAG_EXTRA+="_dp${DENSE_PREFIX}"
fi
dsa_setup_run_identity
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[dsa-phase1] run_dir=${RUN_DIR}"
echo "[dsa-phase1] ckpt_dir=${CKPT_DIR} (stable, keyed on config -> resume_mode=auto continues here)"
echo "[dsa-phase1] existing checkpoints: $(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | wc -l)"
echo "[dsa-phase1] log=${LOG_FILE}  model=${MODEL_PATH}"
echo "[dsa-phase1] train_shards=$(tr -cd , <<<"${TRAIN_LIST}" | wc -c | awk '{print $1+1}') spec=${TRAIN_FILES}"
echo "[dsa-phase1] cwd=$(pwd)"
echo "[dsa-phase1] cmdline: $(tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null)"
echo "[dsa-phase1] argv: $0 $*"

MANIFEST_KNOBS=(MODEL_PATH DATA_DIR TRAIN_FILES VAL_FILES NPROC SEQ_LEN STEPS BATCH
                N_HEADS HEAD_DIM ROPE_HEAD_DIM TOPK DENSE_PREFIX SPARSE_LAYERS SIGMA_TARGET FP8 FP8_UE8M0
                KL_BLOCK KL_CKPT KL_REDUCTION COMPILE_TEACHER DIAG_INTERVAL LOG_PER_LAYER
                GRAD_CKPT MODEL_DTYPE ACT_OFFLOAD
                LR LR_SCHED WARMUP_RATIO MIN_LR_RATIO CLIP_GRAD WEIGHT_DECAY
                SAVE_FREQ MAX_CKPT TEST_FREQ VAL_MAX_SAMPLES VAL_PREFIX VAL_ONLY
                WARMSTART RESUME_PATH STAGE PROJECT EXP_NAME RUNS_BASE
                CUDA_VISIBLE_DEVICES PYTHONPATH PYTORCH_CUDA_ALLOC_CONF WANDB_BASE_URL WANDB_ENTITY)
dsa_write_manifest

# --- DSA override block. verl's `update_model_config` recurses into nested dict values, so only FLAT
#     scalars survive injection -- hence dsa_<field> keys (dsa_overrides_from_config reads them back).
DSA_OV="dsa_enabled: true, dsa_mode: dense_warmup"
DSA_OV+=", dsa_n_heads: ${N_HEADS}, dsa_head_dim: ${HEAD_DIM}, dsa_rope_head_dim: ${ROPE_HEAD_DIM}"
DSA_OV+=", dsa_top_k: ${TOPK}, dsa_sigma_target: ${SIGMA_TARGET}"
DSA_OV+=", dsa_dense_prefix: ${DENSE_PREFIX}"
[[ -n "${SPARSE_LAYERS}" ]] && DSA_OV+=", dsa_sparse_layers: '${SPARSE_LAYERS}'"
DSA_OV+=", dsa_fp8: ${FP8}, dsa_fp8_ue8m0: ${FP8_UE8M0}"
DSA_OV+=", dsa_kl_block_size: ${KL_BLOCK}, dsa_kl_checkpoint: ${KL_CKPT}, dsa_kl_reduction: ${KL_REDUCTION}"
DSA_OV+=", dsa_compile_teacher: ${COMPILE_TEACHER}"
DSA_OV+=", dsa_diag_interval: ${DIAG_INTERVAL}, dsa_log_per_layer: ${LOG_PER_LAYER}"
[[ -n "${WARMSTART}" ]] && DSA_OV+=", dsa_warmstart_path: '${WARMSTART}'"

if [[ -n "${VAL_FILES}" ]]; then
    VAL_LIST=$(dsa_expand_files "${VAL_FILES}" val) || {
        echo "[dsa-phase1] ERROR: no val parquet matched: ${VAL_FILES}"; exit 1; }
    VAL_ARGS=(data.val_files="${VAL_LIST}" data.val_max_samples="${VAL_MAX_SAMPLES}"
              trainer.test_freq="${TEST_FREQ}" +trainer.val_prefix="${VAL_PREFIX}")
    echo "[dsa-phase1] validation ON: val_files=${VAL_FILES} test_freq=${TEST_FREQ}"
else
    VAL_ARGS=(trainer.test_freq=-1)
    echo "[dsa-phase1] validation OFF (set VAL_FILES to enable)"
fi

EVAL_ARGS=()
if [[ "${VAL_ONLY}" == "1" ]]; then
    EVAL_ARGS=(+trainer.val_only=true trainer.resume_mode=resume_path
               trainer.resume_from_path="${RESUME_PATH}")
    echo "[dsa-phase1] VAL_ONLY: eval ${RESUME_PATH} on ${VAL_FILES} -> wandb section '${VAL_PREFIX}'"
fi

# NOTE: engine.reshard_after_forward=True is correct BECAUSE each Qwen3DSAIndexer is wrapped as its own
# FSDP2 unit with reshard=False (Option B2; verl/utils/fsdp_utils.py, docs/dsa_fsdp_sharding_notes.md §4).
# Do NOT "fix" a flat loss by setting this False -- that only keeps the frozen base resident per rank. If
# the loss is flat, check that the OPTIMIZER'S masters move: under a side-channel loss, grad_norm > 0 is NOT
# evidence of training (§3b).
LAUNCH=(
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}"
    -m verl.trainer.sft_trainer
    hydra.run.dir="${RUN_DIR}/hydra/${RUN_TS}"
    trainer.default_local_dir="${CKPT_DIR}"
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
    "+model.override_config={${DSA_OV}}"
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
    optim.weight_decay="${WEIGHT_DECAY}"
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
dsa_run
