#!/usr/bin/env bash
# Shared launch machinery for the Qwen3-DSA phase scripts. SOURCED, not executed.
#
# Factored out rather than copy-pasted between phase1/phase2 because the parts that matter here are the
# parts that have silently cost runs before, and they should exist once:
#
#   * the CONFIG_TAG / RUN_NAME split, so `resume_mode=auto` finds the right checkpoint dir after a crash
#     instead of restarting a multi-day run at step 0 (memory: msa-run-naming-and-resume);
#   * a manifest written NEXT TO THE CHECKPOINTS carrying the git sha *and the working-tree diff* — these
#     runs launch from a dirty tree, so the sha alone does not identify the code — plus every resolved knob,
#     because env-var assignments prefixed to the command never appear in /proc/<pid>/cmdline
#     (memory: log-full-invocation);
#   * the fully-resolved torchrun argv, printf %q so the log is paste-runnable.
#
# Caller must set, before sourcing:  STAGE MODEL_PATH TRAIN_FILES SEQ_LEN STEPS BATCH LR RUNS_BASE
# Caller must set, before calling dsa_write_manifest:  MANIFEST_KNOBS (array of variable names)
# Caller must set, before calling dsa_run:             LAUNCH (array)

# Expand a path | glob | directory into the bracketed list hydra wants.
#   $1 = spec, $2 = shard prefix (train|val)
dsa_expand_files() {
    local spec="$1" pre="$2" files=()
    if [[ -d "${spec}" ]]; then
        mapfile -t files < <(ls -1 "${spec}"/${pre}-*.parquet 2>/dev/null)
        (( ${#files[@]} )) || mapfile -t files < <(ls -1 "${spec}"/*.parquet 2>/dev/null)
    else
        mapfile -t files < <(ls -1 ${spec} 2>/dev/null)
    fi
    (( ${#files[@]} )) || return 1
    local IFS=,; echo "[${files[*]}]"
}

# Run identity: TWO names, deliberately different.
#   CONFIG_TAG  everything that DEFINES the experiment, NO timestamp. Keys the CHECKPOINT dir so that
#               `resume_mode: auto` finds the newest global_step_* when the same command is relaunched. It
#               carries the full config (incl. lr and the token budget) precisely so two different configs
#               can never share a checkpoint dir; only a true continuation collides.
#   RUN_NAME    CONFIG_TAG + timestamp = this ATTEMPT. Drives the wandb experiment name, the log file and
#               the hydra dir, so a restart appears as a second wandb run continuing the same step count.
# Optional input: TAG_EXTRA (phase-specific suffix, e.g. "_k2048_16x64").
dsa_setup_run_identity() {
    RUN_TS=$(date +%Y%m%d_%H%M%S)
    MODEL_TAG=${MODEL_TAG:-$(basename "${MODEL_PATH%/}" | tr '_' '-')}
    # DATA_TAG must IDENTIFY the dataset, because CONFIG_TAG keys the checkpoint dir and `resume_mode=auto`
    # will happily continue a run across two different datasets that share a tag. The generated Phase-2
    # artifact layout is `<model>__<dataset>__<stage>_<ts>` (memory: msa-data-artifact-layout), so a naive
    # "first underscore-separated token" yields the MODEL name and silently drops the dataset entirely.
    # Strip the leading model field, flatten `__`, then drop the timestamp and the L<len> that LEN_TAG
    # already records. Directories without `__` (e.g. longmino_qwen3_32768) keep the short first token.
    local _base
    _base=$(basename "${TRAIN_FILES%/}" .parquet)
    _base=${_base%_train}
    if [[ "${_base}" == *__* ]]; then
        DATA_TAG=$(sed -E 's/_[0-9]{8}_[0-9]{6}$//; s/_L[0-9]+//' <<< "${_base#*__}")
        DATA_TAG=${DATA_TAG//__/_}
    else
        DATA_TAG=${_base%%_*}
    fi
    # tokens actually SCHEDULED = steps * global_batch * seq_len -- the real budget, not a nominal label
    TOK_TAG=$(awk -v s="${STEPS}" -v b="${BATCH}" -v l="${SEQ_LEN}" 'BEGIN{
        n=s*b*l; if (n>=1e9) printf "%.2fBt", n/1e9; else printf "%.0fMt", n/1e6 }')
    LEN_TAG=$(awk -v l="${SEQ_LEN}" 'BEGIN{ if (l%1024==0) printf "L%dk", l/1024; else printf "L%d", l }')
    CONFIG_TAG=${CONFIG_TAG:-${STAGE/phase/p}_${MODEL_TAG}_${DATA_TAG}_${TOK_TAG}_${LEN_TAG}_bs${BATCH}${TAG_EXTRA:-}_lr${LR}}
    RUN_NAME=${RUN_NAME:-${CONFIG_TAG}_${RUN_TS}}
    CKPT_DIR=${CKPT_DIR:-${RUNS_BASE}/_ckpt/${CONFIG_TAG}}
    # EVERY artifact lives with the checkpoints it produced: logs, manifest, hydra config, wandb dir. One
    # directory answers "what produced these weights", and it survives restarts since CKPT_DIR has no
    # timestamp. Per-attempt files are keyed by RUN_TS so attempts accumulate instead of overwriting.
    RUN_DIR=${RUN_DIR:-${CKPT_DIR}}
    LOG_DIR="${CKPT_DIR}/logs"
    mkdir -p "${RUN_DIR}" "${CKPT_DIR}" "${LOG_DIR}"
    LOG_FILE=${LOG_FILE:-${LOG_DIR}/run-${RUN_TS}.log}
    MANIFEST="${LOG_DIR}/launch-${RUN_TS}.txt"
    export WANDB_DIR="${CKPT_DIR}"
}

# The reproduction recipe, in one file, next to the checkpoints. Captures what the transcript cannot.
dsa_write_manifest() {
    { set +x; } 2>/dev/null
    {
        echo "# Qwen3 DSA ${STAGE} launch manifest"
        echo "run_ts:        ${RUN_TS}"
        echo "date:          $(date -Is)"
        echo "host:          $(hostname -f)   user: $(whoami)"
        echo "cwd:           $(pwd)"
        echo "run_name:      ${RUN_NAME}"
        echo "config_tag:    ${CONFIG_TAG}"
        echo "ckpt_dir:      ${CKPT_DIR}"
        echo "log_file:      ${LOG_FILE}"
        echo "wandb:         ${WANDB_BASE_URL:-unset}/${WANDB_ENTITY:-unset}/${PROJECT}  experiment=${RUN_NAME}"
        echo
        echo "## exact invocation"
        echo "outer_cmdline: $(tr '\0' ' ' < /proc/$$/cmdline 2>/dev/null)"
        echo "argv:          $0 ${SCRIPT_ARGV:-}"
        echo "# NOTE: env-var assignments prefixed to the command are consumed by the shell and do NOT"
        echo "#       appear in outer_cmdline. Reproduce from the resolved knobs below, not the cmdline."
        echo
        echo "## resolved knobs (every variable this script reads)"
        for v in "${MANIFEST_KNOBS[@]}"; do printf '%-26s %s\n' "${v}=" "${!v-<unset>}"; done
        echo
        echo "## tokens"
        awk -v s="${STEPS}" -v b="${BATCH}" -v l="${SEQ_LEN}" 'BEGIN{
            printf "scheduled_tokens           %d  (= %d steps x %d batch x %d seq_len)\n", s*b*l, s, b, l
            printf "tokens_per_step            %d\n", b*l }'
        echo "data_manifest              ${TRAIN_FILES%/}/MANIFEST.json"
        echo
        echo "## code provenance (the tree is usually DIRTY -- the sha alone is not enough)"
        echo "repo:          ${REPO_ROOT}"
        echo "git_branch:    $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        echo "git_sha:       $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null)"
        echo "git_dirty:"
        git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | sed 's/^/  /'
        echo
        echo "## environment"
        echo "python:        $(python3 -c 'import sys;print(sys.version.split()[0])' 2>/dev/null) ($(command -v python3))"
        python3 - <<'PYVER' 2>/dev/null
import torch, transformers
print(f"torch:         {torch.__version__}  cuda {torch.version.cuda}")
print(f"transformers:  {transformers.__version__}")
print(f"gpu_count:     {torch.cuda.device_count()}")
PYVER
        echo "nvidia_driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
        echo "gpus:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
    } > "${MANIFEST}"
    # the full working-tree diff, so the run reproduces from the sha + this patch alone
    git -C "${REPO_ROOT}" diff HEAD > "${LOG_DIR}/gitdiff-${RUN_TS}.patch" 2>/dev/null || true
    echo "[dsa-${STAGE}] manifest=${MANIFEST}"
    echo "[dsa-${STAGE}] gitdiff=${LOG_DIR}/gitdiff-${RUN_TS}.patch"
    { set -x; } 2>/dev/null
}

# Log the exact resolved argv (into both the transcript and the manifest), then exec it.
dsa_run() {
    { set +x; } 2>/dev/null
    {
        echo "[dsa-${STAGE}] ===== EXACT LAUNCH ARGV ====="
        printf '  %q' "${LAUNCH[@]}"; echo
        echo "[dsa-${STAGE}] ============================="
    }
    {
        echo
        echo "## exact torchrun argv (paste-runnable)"
        printf '%q ' "${LAUNCH[@]}"; echo
    } >> "${MANIFEST}"
    { set -x; } 2>/dev/null
    "${LAUNCH[@]}"
}
