#!/usr/bin/env bash
# scripts/msa/setup_eval_root.sh — clone an isolated eval root for a new MSA checkpoint.
#
# The harness is path-absolute: every run_*.sh and _common/run_opencompass.sh hardcodes
# EVAL_ROOT. Cloning is therefore "copy the tree, sed the path", plus a set of symlinks that
# keep the ~12 GB of read-only inputs (model, OpenCompass data, MRCR/GSM parquets, the LCB HF
# cache) shared instead of duplicated per checkpoint.
#
# What must NOT be shared, and why: oc_workdir/, results/, logs/, output/ and opencompass/tmp
# are WRITE targets. An earlier run wrote scratch into the baseline tree; names never collided
# so nothing was damaged, but the isolation is structural now (SCORECARD_msa_k8v2 §4).
#
# Usage:  bash scripts/msa/setup_eval_root.sh <src_eval_root> <dst_eval_root>
set -euo pipefail

SRC=${1:?usage: setup_eval_root.sh <src_eval_root> <dst_eval_root>}
DST=${2:?usage: setup_eval_root.sh <src_eval_root> <dst_eval_root>}
SRC=${SRC%/}; DST=${DST%/}
[ -d "$SRC" ] || { echo "[setup] ERROR: src $SRC does not exist" >&2; exit 1; }
[ -e "$DST" ] && { echo "[setup] ERROR: dst $DST already exists" >&2; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$DST"
LOG=$DST/setup_eval_root_${TS}.log
exec > >(tee -a "$LOG") 2>&1
# `set -e` inside a `tee` pipeline loses the last lines on exit, so say WHY we are dying while
# the fd is still alive.
trap 'rc=$?; [ $rc -ne 0 ] && echo "[setup] ABORT rc=$rc at line $LINENO"; sync' EXIT

echo "[setup] $(date -Is)  host=$(hostname)"
echo "[setup] src=$SRC"
echo "[setup] dst=$DST"

# 1. scripts, config templates and the two LCB repo copies. Everything write-shaped is excluded
#    and recreated empty below; rendered configs (eval_*__p*.py) are regenerated per pass by
#    render_config.sh, so copying stale ones would only risk running the wrong slice.
#    (tar, not rsync — rsync is not installed in this container)
tar -C "$SRC" \
  --exclude='./*/oc_workdir' --exclude='./*/results' --exclude='./*/logs' \
  --exclude='./*/output' --exclude='./*/hf_cache' --exclude='./*/data' \
  --exclude='__pycache__' \
  --exclude='./opencompass/tmp' --exclude='./opencompass/icl_inference_output' \
  --exclude='./opencompass/.cache' --exclude='./opencompass/data' --exclude='./opencompass/zips' \
  --exclude='eval_*__p*.py' \
  --exclude='./env' --exclude='./model' --exclude='./SCORECARD.md' \
  --exclude='setup_eval_root_*.log' \
  -cf - . | tar -C "$DST" -xf -

# 2. repoint every hardcoded EVAL_ROOT. This is the whole of the "port the harness" work.
#    NOTE: no `--` before the pattern. `--` ends option processing, so the --include filters
#    would become path operands, the filter would silently not apply, and sed would rewrite
#    THIS SCRIPT'S OWN LOG -- which replaces the inode `tee` is holding and makes every
#    subsequent line, including error messages, vanish. Cost an hour once.
mapfile -t FILES < <(grep -rl --include='*.sh' --include='*.py' --include='*.tmpl' \
                       -e "$SRC" "$DST" 2>/dev/null || true)
for f in "${FILES[@]}"; do sed -i "s|$SRC|$DST|g" "$f"; done
echo "[setup] repointed ${#FILES[@]} file(s)"

# 3. read-only shared inputs. Resolve through the source's own symlinks so a chain of clones
#    does not build a chain of indirections.
link() {  # link <target> <linkname>
  local tgt=$1 name=$2
  tgt=$(readlink -f "$tgt")
  [ -e "$tgt" ] || { echo "[setup] ERROR: link target missing: $tgt" >&2; exit 1; }
  mkdir -p "$(dirname "$name")"; rm -rf "$name"; ln -s "$tgt" "$name"
}
link "$SRC/env"                      "$DST/env"
link "$SRC/model"                    "$DST/model"
link "$SRC/opencompass/data"         "$DST/opencompass/data"
link "$SRC/opencompass/zips"         "$DST/opencompass/zips"
link "$SRC/gsm_infinite/data"        "$DST/gsm_infinite/data"
link "$SRC/mrcr/data"                "$DST/mrcr/data"
link "$SRC/livecodebench_v6/hf_cache" "$DST/livecodebench_v6/hf_cache"

# 4. fresh write targets
for d in _common ifeval aime25 aime25_cap32k gpqa mmlu_pro ruler mrcr gsm_infinite \
         livecodebench_v6 livecodebench_v6_cap32k; do
  mkdir -p "$DST/$d/logs" "$DST/$d/results" "$DST/$d/oc_workdir"
done
mkdir -p "$DST/opencompass/tmp" "$DST/opencompass/icl_inference_output" "$DST/opencompass/.cache"
mkdir -p "$DST/livecodebench_v6/output" "$DST/livecodebench_v6_cap32k/output"

# 5. LCB loads few-shot examples via repo-relative paths, so it runs from inside its repo copy
#    and writes to <repo>/output/<model>/. Both variants serve the SAME model name, so the two
#    copies must point at different output trees or they overwrite each other.
rm -f "$DST/tools/LiveCodeBench/output" "$DST/tools/LiveCodeBench_cap32k/output"
ln -s "$DST/livecodebench_v6/output"        "$DST/tools/LiveCodeBench/output"
ln -s "$DST/livecodebench_v6_cap32k/output" "$DST/tools/LiveCodeBench_cap32k/output"

# 6. gates — a silently mis-cloned root produces plausible numbers against the wrong tree
# `|| true` is load-bearing: grep exits 1 when it finds nothing, which is the PASS case here,
# and under `set -o pipefail` that would abort the script on success.
STALE=$( { grep -rl --include='*.sh' --include='*.py' --include='*.tmpl' -e "$SRC" "$DST" \
             2>/dev/null || true; } | wc -l)
[ "$STALE" -eq 0 ] || { echo "[setup] FAIL: $STALE file(s) still reference $SRC" >&2; exit 1; }
for p in env model opencompass/data opencompass/zips gsm_infinite/data mrcr/data \
         livecodebench_v6/hf_cache; do
  [ -e "$DST/$p" ] || { echo "[setup] FAIL: $p unresolved" >&2; exit 1; }
done
for s in ifeval/run_ifeval.sh ruler/run_ruler.sh mrcr/run_mrcr.py gsm_infinite/run_gsm_infinite.py \
         livecodebench_v6/run_livecodebench_v6.sh livecodebench_v6_cap32k/run_livecodebench_v6_cap32k.sh \
         _common/run_opencompass.sh; do
  [ -f "$DST/$s" ] || { echo "[setup] FAIL: missing $s" >&2; exit 1; }
  grep -q -- "$DST" "$DST/$s" || { echo "[setup] FAIL: $s does not reference $DST" >&2; exit 1; }
done
echo "[setup] gates OK — 0 stale paths, all shared inputs resolve, all run scripts repointed"
echo "[setup] DONE -> $DST"
