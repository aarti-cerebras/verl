#!/usr/bin/env bash
# MSA Phase-2 (sparse adaptation) on Qwen3: sparsity ON, base UNFROZEN. Loss = LM cross-entropy through
# the block-sparse attention + lambda * the selected-set indexer KL.
#
# This is the published recipe: both MSA (arXiv 2606.13392 §5.1, "MSA-CPT") and DeepSeek-V3.2
# (arXiv 2512.02556 §2.1.1) go dense-warm-up -> sparse training with ALL parameters optimized. There is no
# frozen-base sparse stage in either paper.
#
# Prereqs:
#   1. a Phase-1 checkpoint, consolidated to a world-size-agnostic indexer state dict:
#        python3 scripts/dsa/consolidate_indexer_ckpt.py --ckpt-dir <phase1>/global_step_N \
#          --out <phase1>/indexer_full.pt
#      then pass it as WARMSTART=<...>/indexer_full.pt. REQUIRED: starting sparse from a random indexer
#      routes attention to noise, which is the whole point of the warm-up (paper B.4).
#   2. the Phase-2 behaviour-cloning parquet, already built by
#        scripts/dsa/select_prompts.py -> gen_trajectories.py -> trajectories_to_sft_parquet.py
#      i.e. PRE-TOKENIZED `input_ids` + `loss_mask` (prompt masked, the model's own trace trained).
#      It is read by MSASFTDataset, NOT PackedPretrainDataset: that loader forces loss_mask to all-ones
#      and drops every row shorter than max_length, and BC rows are variable-length (p50 ~7K), so it
#      would silently discard the entire dataset.
#
# See docs/qwen3_4b_msa/phase2_plan.md (esp. §2 for the exact computation sequence) and
# docs/qwen3_4b_msa/phase2_data_gen.md §6 for why the data is pre-tokenized rather than `messages`.
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
# The BC parquet. Rows are VARIABLE length (one conversation each, p50 ~7K, max 32,767) -- unlike Phase 1,
# where every row was exactly SEQ_LEN. SEQ_LEN is therefore a CAP here, not a required length: rows longer
# than it are truncated, shorter rows are kept as-is (no padding is materialised -- see the tier note).
TRAIN_FILES=${TRAIN_FILES:-/cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__dolci-think-rl-32b__ph2b_full93889_L32768_20260730_231930/bc_2b.parquet}
NPROC=${NPROC:-8}                      # world size; set CUDA_VISIBLE_DEVICES to match. CHANGING THIS
                                       # BREAKS RESUME: verl FSDP checkpoints are world-size-locked
                                       # (model_world_size_<N>_rank_*.pt).
SEQ_LEN=${SEQ_LEN:-32768}
STEPS=${STEPS:-11278}                  # 90,230 BC rows / BATCH 8 = 11,278 = exactly one epoch
BATCH=${BATCH:-8}                      # global batch (rows/step); 8 = 1 per GPU at NPROC=8

# --- length tiering: the difference between a 2.4-day run and a 7.4-day one -------------------------
# A step is BATCH rows, one per rank, and the collective ends when the SLOWEST rank finishes -- so
# wall-clock is paid on the LONGEST row in the step while the other ranks idle. (There is no padding to
# blame: with micro_batch_size_per_gpu=1 the engine pads each micro-batch to its own single row.) On this
# data, 8 rows drawn at random span ~1.3K..32.7K, which measures 32.1% efficient. Grouping each step into
# one TIER_WIDTH band lifts that to ~100%:
#     random order  32.1% -> 7.4 days      16 tiers  ~100% -> 2.4 days
# LENGTH_TIERS=0 disables it (parquet row order, sampler shuffles as usual).
LENGTH_TIERS=${LENGTH_TIERS:-16}
TIER_WIDTH=${TIER_WIDTH:-2048}
DATA_SEED=${DATA_SEED:-1234}           # seeds the within-tier shuffle AND the tier order. The row order is
                                       # recomputed from it on every launch, so it MUST NOT change across a
                                       # resume: the dataloader state is a bare batch counter with no
                                       # dataset identity, so a different order silently resumes onto
                                       # different rows (checkpoint_handler.py:116-124).
NUM_WORKERS=${NUM_WORKERS:-8}          # dataloader workers. ALSO resume-relevant: StatefulDataLoader's
                                       # multiprocess snapshot is keyed by worker id, so changing this
                                       # invalidates a saved iterator state.

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
DIAG_INTERVAL=${DIAG_INTERVAL:-10}     # block-level selection diagnostics every N forwards (always on in eval)
LOG_PER_LAYER=${LOG_PER_LAYER:-true}   # per-layer captured_over_oracle / block_recall keys — the Phase-1 GATE
                                       # is per layer (>= 0.90), so the layer breakdown is what you read it off

GRAD_CKPT=${GRAD_CKPT:-False}          # MUST stay False even though the base now trains: HF gradient
                                       # checkpointing runs the first pass under no_grad, so the `_msa_kl`
                                       # side effect would be stashed WITHOUT a graph and contribute zero
                                       # gradient, silently (phase2_plan §1.1). The module checkpoints its
                                       # own attention+KL region instead (KL_CKPT below). Use TILED_MLP for
                                       # the FFN.
TILED_MLP=${TILED_MLP:-True}           # shard the MLP forward/backward to cut activation memory -- the
                                       # replacement for GRAD_CKPT now that the base trains. Applied in
                                       # monkey_patch BEFORE the MSA branch, so our early return keeps it.
TILED_MLP_SHARDS=${TILED_MLP_SHARDS:-4}
MODEL_DTYPE=${MODEL_DTYPE:-bf16}       # dtype the (frozen) base is initialized in. bf16 at 32K; fp32 gives a
                                       # more accurate KL teacher at ~2x memory (fine at short context).
ACT_OFFLOAD=${ACT_OFFLOAD:-True}       # CPU-offload activations saved for backward; transparent, so it does
                                       # NOT re-trigger the `_msa_kl` side effect (unlike GRAD_CKPT).
                                       # REQUIRED at 32K: the defaults OOM at 76/79 GB, while this runs at
                                       # 34.6 GB peak (measured, 3-step smoke 2026-07-30).
ACT_GPU_LIMIT=${ACT_GPU_LIMIT:-0}
# Checkpoint OFTEN: this is a multi-day run and SAVE_FREQ is what bounds how much a crash costs. The old
# default (== STEPS) wrote exactly one checkpoint, at the very end, so any interruption lost everything.
SAVE_FREQ=${SAVE_FREQ:-100}            # -1 to skip checkpointing entirely. verl's save_freq is in STEPS,
                                       # not wall-clock; step time here varies ~5x with the length tier, so
                                       # the interval between saves is not constant in time.
MAX_CKPT=${MAX_CKPT:-5}                # retained checkpoints. Phase 2 trains the BASE too, so each one is
                                       # model+optimizer for all 4.02B params -- MEASURED at 24 GB, not
                                       # Phase 1's indexer-only 8 GB. 5 is ~120 GB.
LR=${LR:-5e-6}                         # PEAK lr for the BASE (plan §6.2: 5e-6 - 1e-5). The base is now
                                       # trained, so this is the drift knob.
INDEXER_LR=${INDEXER_LR:-1e-4}         # separate LR for *.indexer.* (plan §6.2: 1e-3 -> 1e-4 in Phase 2).
                                       # verl builds two param groups keyed on ".indexer." in the name.
KL_LAMBDA=${KL_LAMBDA:-1.0}            # lambda on the KL. NO published value in either paper -- pick it so
                                       # indexer/kl_share_of_loss lands ~0.05-0.2 on the first batch. With
                                       # KL_REDUCTION=mean, a paper lambda needs x n_sparse_layers (33).
LR_SCHED=${LR_SCHED:-cosine}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}
CLIP_GRAD=${CLIP_GRAD:-1.0}            # verl's default. Exposed deliberately: with a side-channel KL the
                                       # grad norm can sit far above the clip (the DSA runs sat at ~330,
                                       # i.e. ~330x clipping every step, making the step effectively
                                       # LR * unit-direction). Watch train/grad_norm and raise if it is
                                       # pinned. See plan §4.1.
                                       # REQUIRED in Phase 2: the consolidated Phase-1 indexer state dict.
                                       # Starting sparse from a random indexer routes attention to noise --
                                       # the whole point of the warm-up (paper B.4). Consolidated 2026-08-03
                                       # from global_step_3815 (132 tensors = 33 sparse layers x 4, 97.33M
                                       # params, layers 3..35; final Phase-1 full-support KL 0.166).
WARMSTART=${WARMSTART:-/cb/ml-eng/aarti/msa/indexer_warmup/_ckpt/p1_qwen3-4b-thinking-2507_longmino_1.00Bt_L32k_bs8_k16_B128_dp3_lr1e-3/indexer_full.pt}
EXP_NAME=${EXP_NAME:-phase2}
PROJECT=${PROJECT:-MSA}

# --- held-out validation. Empty VAL_FILES => no eval (test_freq forced -1). The val loop always computes
#     diagnostics (diag_interval is bypassed in eval mode), so every val batch reports the full
#     val/indexer/* panel incl. captured_over_oracle per layer.
VAL_FILES=${VAL_FILES:-}
TEST_FREQ=${TEST_FREQ:-25}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
VAL_PREFIX=${VAL_PREFIX:-val}

# --- val-only mode: skip training, evaluate a trained indexer checkpoint on VAL_FILES, then exit.
VAL_ONLY=${VAL_ONLY:-0}
RESUME_PATH=${RESUME_PATH:-}

if [[ "${VAL_ONLY}" == "1" ]]; then
    [[ -n "${VAL_FILES}"   ]] || { echo "[msa-phase2] ERROR: VAL_ONLY=1 requires VAL_FILES"; exit 1; }
    [[ -n "${RESUME_PATH}" ]] || { echo "[msa-phase2] ERROR: VAL_ONLY=1 requires RESUME_PATH"; exit 1; }
    TRAIN_FILES="${VAL_FILES}"
    SAVE_FREQ=-1
fi

# --- preflight: fail with an actionable message rather than a cryptic dataloader/model error ---
[[ -d "${MODEL_PATH}" || "${MODEL_PATH}" != /* ]] || {
    echo "[msa-phase2] ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"; exit 1; }
# TRAIN_FILES/VAL_FILES may be a single parquet, a glob, or a DIRECTORY (the builder emits 46 shards).
# Expand to the bracketed list hydra wants; MSASFTDataset accepts a list of parquet paths (or a directory).
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
    echo "[msa-phase2] ERROR: no train parquet matched: ${TRAIN_FILES}"
    echo "[msa-phase2] this stage consumes the PRE-TOKENIZED behaviour-cloning parquet (input_ids +"
    echo "[msa-phase2] loss_mask) produced by scripts/dsa/trajectories_to_sft_parquet.py --emit-input-ids"
    exit 1; }

# --- run identity: TWO names, deliberately different (same scheme as run_qwen3_msa_phase1.sh) --------
#   CONFIG_TAG  everything that DEFINES the experiment, with NO timestamp. This keys the CHECKPOINT dir so
#               `resume_mode: auto` (sft_trainer_engine.yaml:79) finds the newest global_step_* when the
#               same command is relaunched. A TIMESTAMPED checkpoint dir -- which is what this script used
#               to build -- makes an interrupted multi-day run silently restart from step 0. That is the
#               single failure mode this split exists to prevent.
#   RUN_NAME    CONFIG_TAG + timestamp = this ATTEMPT. Drives the wandb experiment name, log file and hydra
#               dir, so a restart shows up as a second wandb run continuing the same step count.
RUNS_BASE=${RUNS_BASE:-/cb/ml-eng/aarti/msa/sparse}
STAGE=${STAGE:-phase2}
RUN_TS=$(date +%Y%m%d_%H%M%S)
MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}" | tr '_' '-')}
DATA_TAG=$(basename "${TRAIN_FILES%/}" .parquet); DATA_TAG=${DATA_TAG%_train}
DATA_TAG=${DATA_TAG#*__}   # artifact dirs are "<model>__<dataset>"; the model is already in MODEL_TAG
LEN_TAG=$(awk -v l="${SEQ_LEN}" 'BEGIN{ if (l%1024==0) printf "L%dk", l/1024; else printf "L%d", l }')
if [[ -n "${SPARSE_LAYERS}" ]]; then
    LAYER_TAG="sl$(tr -cd , <<<"${SPARSE_LAYERS}" | wc -c | awk '{print $1+1}')"
else
    LAYER_TAG="dp${DENSE_PREFIX}"
fi
# EVERY knob that changes the experiment goes in the tag, so two different configs can never share a
# checkpoint dir and only a true continuation collides. lam/lr/ilr/tiers/steps are all load-bearing:
# changing the tier config changes the ROW ORDER, which a resumed dataloader state cannot survive.
CONFIG_TAG=${CONFIG_TAG:-${STAGE/phase/p}_${MODEL_TAG}_${DATA_TAG}_${LEN_TAG}_bs${BATCH}_k${TOPK}_B${BLOCK_SIZE}_${LAYER_TAG}_lam${KL_LAMBDA}_lr${LR}_ilr${INDEXER_LR}_t${LENGTH_TIERS}w${TIER_WIDTH}_st${STEPS}}
RUN_NAME=${RUN_NAME:-${CONFIG_TAG}_${RUN_TS}}
CKPT_DIR=${CKPT_DIR:-${RUNS_BASE}/_ckpt/${CONFIG_TAG}}
RUN_DIR=${RUN_DIR:-${CKPT_DIR}}
LOG_DIR="${CKPT_DIR}/logs"
mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${LOG_DIR}"
LOG_FILE=${LOG_FILE:-${LOG_DIR}/run-${RUN_TS}.log}
export WANDB_DIR="${CKPT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "[msa-phase2] run_dir=${RUN_DIR}"
# The resume target: the one path whose staleness silently costs days.
echo "[msa-phase2] ckpt_dir=${CKPT_DIR} (stable, keyed on config -> resume_mode=auto continues here)"
echo "[msa-phase2] existing checkpoints: $(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | wc -l)" \
     "latest=$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo none)"
# Three things a RESTART must not change, all of which fail SILENTLY if they do. The dataloader state saved
# per step is a bare batch counter with no dataset identity (checkpoint_handler.py:116-124), so:
#   - a different dataset/row order resumes onto unrelated rows (and if it is SHORTER than the counter, the
#     epoch body never executes and training looks hung);
#   - NPROC changes make the world-size-locked FSDP shards unloadable;
#   - NUM_WORKERS changes invalidate the multiprocess dataloader snapshot (keyed by worker id).
echo "[msa-phase2] RESUME INVARIANTS -- must be identical on every relaunch:"
echo "[msa-phase2]   data=${TRAIN_FILES}"
echo "[msa-phase2]   LENGTH_TIERS=${LENGTH_TIERS} TIER_WIDTH=${TIER_WIDTH} DATA_SEED=${DATA_SEED} BATCH=${BATCH}"
echo "[msa-phase2]   NPROC=${NPROC} NUM_WORKERS=${NUM_WORKERS}"
echo "[msa-phase2] log=${LOG_FILE}"
echo "[msa-phase2] model=${MODEL_PATH}"
echo "[msa-phase2] train_shards=$(tr -cd , <<<"${TRAIN_LIST}" | wc -c | awk '{print $1+1}') spec=${TRAIN_FILES} wandb=${WANDB_BASE_URL}/${WANDB_ENTITY}/${PROJECT}"
# --- ALWAYS record EXACTLY how this run was launched: the outer command line + every env var the script
#     consumes (incl. debug flags that never reach the torchrun trace). Makes any log self-reproducing. ---
echo "[msa-phase2] cwd=$(pwd)"
echo "[msa-phase2] cmdline: $(tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null)"
echo "[msa-phase2] argv: $0 $*"
echo "[msa-phase2] env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} NPROC=${NPROC} SEQ_LEN=${SEQ_LEN}" \
     "STEPS=${STEPS} BATCH=${BATCH} TOPK=${TOPK} BLOCK_SIZE=${BLOCK_SIZE} INIT_BLOCKS=${INIT_BLOCKS}" \
     "LOCAL_BLOCKS=${LOCAL_BLOCKS} DENSE_PREFIX=${DENSE_PREFIX} SPARSE_LAYERS=${SPARSE_LAYERS:-none}" \
     "KL_BLOCK=${KL_BLOCK} KL_CKPT=${KL_CKPT} KL_REDUCTION=${KL_REDUCTION} DIAG_INTERVAL=${DIAG_INTERVAL}" \
     "LOG_PER_LAYER=${LOG_PER_LAYER} LR=${LR} LR_SCHED=${LR_SCHED} WARMUP_RATIO=${WARMUP_RATIO}" \
     "MIN_LR_RATIO=${MIN_LR_RATIO} CLIP_GRAD=${CLIP_GRAD} GRAD_CKPT=${GRAD_CKPT} MODEL_DTYPE=${MODEL_DTYPE}" \
     "INDEXER_LR=${INDEXER_LR} KL_LAMBDA=${KL_LAMBDA} TILED_MLP=${TILED_MLP} TILED_MLP_SHARDS=${TILED_MLP_SHARDS}" \
     "ACT_OFFLOAD=${ACT_OFFLOAD} ACT_GPU_LIMIT=${ACT_GPU_LIMIT} SAVE_FREQ=${SAVE_FREQ} MAX_CKPT=${MAX_CKPT:-all}" \
     "WARMSTART=${WARMSTART:-none} EXP_NAME=${EXP_NAME} PROJECT=${PROJECT} MODEL_PATH=${MODEL_PATH}" \
     "TRAIN_FILES=${TRAIN_FILES} VAL_FILES=${VAL_FILES:-none} TEST_FREQ=${TEST_FREQ}" \
     "VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES} VAL_PREFIX=${VAL_PREFIX} VAL_ONLY=${VAL_ONLY}" \
     "LENGTH_TIERS=${LENGTH_TIERS} TIER_WIDTH=${TIER_WIDTH} DATA_SEED=${DATA_SEED}" \
     "NUM_WORKERS=${NUM_WORKERS} CKPT_DIR=${CKPT_DIR} CONFIG_TAG=${CONFIG_TAG}" \
     "RESUME_PATH=${RESUME_PATH:-none} PYTHONPATH=${PYTHONPATH:-}"

# --- MSA override block. verl's `update_model_config` recurses into nested dict values, so only FLAT
#     scalars survive injection — hence msa_<field> keys (msa_overrides_from_config reads them back).
MSA_OV="msa_enabled: true, msa_mode: sparse"
MSA_OV+=", msa_top_k: ${TOPK}, msa_block_size: ${BLOCK_SIZE}"
MSA_OV+=", msa_init_blocks: ${INIT_BLOCKS}, msa_local_blocks: ${LOCAL_BLOCKS}"
MSA_OV+=", msa_kl_block_size: ${KL_BLOCK}, msa_kl_checkpoint: ${KL_CKPT}, msa_kl_reduction: ${KL_REDUCTION}"
MSA_OV+=", msa_diag_interval: ${DIAG_INTERVAL}, msa_log_per_layer: ${LOG_PER_LAYER}"
if [[ -n "${SPARSE_LAYERS}" ]]; then
    MSA_OV+=", msa_sparse_layers: '${SPARSE_LAYERS}'"     # explicit list wins over dense_prefix
else
    MSA_OV+=", msa_dense_prefix: ${DENSE_PREFIX}"
fi
[[ -n "${WARMSTART}" ]] && MSA_OV+=", msa_warmstart_path: '${WARMSTART}'"

if [[ -n "${VAL_FILES}" ]]; then
    VAL_LIST=$(_expand_files "${VAL_FILES}" val) || {
        echo "[msa-phase2]] ERROR: no val parquet matched: ${VAL_FILES}"; exit 1; }
    VAL_ARGS=(
        data.val_files="${VAL_LIST}"
        data.val_max_samples="${VAL_MAX_SAMPLES}"
        trainer.test_freq="${TEST_FREQ}"
        +trainer.val_prefix="${VAL_PREFIX}"
    )
    echo "[msa-phase2] validation ON: val_files=${VAL_FILES} test_freq=${TEST_FREQ}"
else
    VAL_ARGS=( trainer.test_freq=-1 )
    echo "[msa-phase2] validation OFF (set VAL_FILES to enable)"
fi

EVAL_ARGS=()
if [[ "${VAL_ONLY}" == "1" ]]; then
    EVAL_ARGS=(
        +trainer.val_only=true
        trainer.resume_mode=resume_path
        trainer.resume_from_path="${RESUME_PATH}"
    )
    echo "[msa-phase2] VAL_ONLY: eval ${RESUME_PATH} on ${VAL_FILES} -> wandb section '${VAL_PREFIX}'"
fi

# NOTE: engine.reshard_after_forward=True is correct in Phase 2, but NOT for the reason the Phase-1 comment
# gave. fsdp_utils.py:590-598 gates the per-indexer FSDP2 unit ("Option B2") on `mode == "dense_warmup"`, so
# in sparse mode `indexer_units` is EMPTY and each MSAIndexer is swept into its parent Qwen3DecoderLayer's
# unit by the generic loop. That is correct here because the CE flows through each layer's output, so the
# standard layer gates fire and the indexer needs no special unit. Do NOT set this False to "fix" a flat
# loss: check that the OPTIMIZER'S masters move -- grad_norm > 0 is NOT evidence of training under a
# side-channel loss.
# --local-addr: the address workers are told to use as MASTER_ADDR. Without it, torch derives it from
# `local_addr or socket.getfqdn()` (elastic/rendezvous/api.py:90) -- and on hosts whose own FQDN does NOT
# resolve (ml-eng-gpu-22: /etc/hosts has only localhost, DNS has no record), every rank then spends 300 s
# failing to TCP-connect to `<fqdn>:<port>` and the run dies before step 1 with a c10d timeout, not a
# Python traceback. `--standalone` is single-node by construction (it hardcodes rdzv_endpoint=localhost:0,
# run.py:964), so loopback is always the right answer here and needs no per-host lookup.
LOCAL_ADDR=${LOCAL_ADDR:-127.0.0.1}
LAUNCH=(
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" --local-addr "${LOCAL_ADDR}"
    -m verl.trainer.sft_trainer
    hydra.run.dir="${RUN_DIR}/hydra/${RUN_TS}"
    # Stable, config-keyed: this is what makes `resume_mode=auto` continue instead of restarting at 0.
    trainer.default_local_dir="${CKPT_DIR}"
    +loss_mode=msa_sparse
    +indexer_kl_lambda="${KL_LAMBDA}"
    data.train_files="${TRAIN_LIST}"
    # Pre-tokenized variable-length BC rows: honours `loss_mask` (so the prompt is NOT trained on) and keeps
    # rows shorter than max_length. PackedPretrainDataset would drop all 90,230 of them.
    data.custom_cls.path=verl/utils/dataset/msa_sft_dataset.py
    data.custom_cls.name=MSASFTDataset
    +data.length_tiers="${LENGTH_TIERS}"
    +data.tier_width="${TIER_WIDTH}"
    +data.seed="${DATA_SEED}"
    # The dataset's row order IS the batching (length tiers); the sampler must not re-permute it.
    +data.sampler_shuffle=$([[ "${LENGTH_TIERS}" -gt 0 ]] && echo False || echo True)
    data.num_workers="${NUM_WORKERS}"
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
    "+model.override_config={${MSA_OV}}"
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
    echo "[msa-phase2] ===== EXACT LAUNCH ARGV ====="
    printf '  %q' "${LAUNCH[@]}"; echo
    echo "[msa-phase2] ============================="
}
{ set -x; } 2>/dev/null

"${LAUNCH[@]}"
