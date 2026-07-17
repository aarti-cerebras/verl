#!/usr/bin/env bash
# DSA Phase-2 correctness sweep (verification "B"): the computed objective must be INVARIANT to the
# micro-batch split and the dp degree, because each per-micro-batch loss is globally normalized
# (÷ global tokens/queries × dp_size) and summed. We run the SAME 12-doc global batch under 3 configs and:
#   * HARD-ASSERT step-1 train/loss, loss/lm, loss/kl_weighted match across configs (to fp tolerance).
#     A systematic offset (factor/additive) at step 1 = a real bug (normalization / padding / sharding).
#   * report the multi-step drift — expected to be small & non-systematic (fp non-associativity in the
#     gradient reduction compounds through the optimizer; NOT asserted, only shown).
# See docs/dsa_phase2_loss_metrics.md and the fsdp/no_padding discussion.
#
# Configs (all: BATCH=12 over the 12-doc parquet => step 1 sees all 12 docs, order-independent). All use
# MICRO_BSZ=1 (long ~4K code docs OOM at micro_bsz>1 with few-way sharding); we vary the DP DEGREE, which
# also varies the per-rank accumulation count -> tests dp-invariance AND accumulation-invariance cheaply.
# (Intra-micro-batch padding of unequal-length seqs is covered separately by the unit test
# test_padding_invariance.)
#   base4_mb1 : NPROC=4 MICRO_BSZ=1  (4 ranks x 3 micro-batches)   <- baseline
#   dp3_mb1   : NPROC=3 MICRO_BSZ=1  (3 ranks x 4 micro-batches)   <- different dp / accumulation
#   dp2_mb1   : NPROC=2 MICRO_BSZ=1  (2 ranks x 6 micro-batches)   <- different dp / accumulation
set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-${REPO_ROOT}/data/dsa/phase2_overfit12.parquet}
STEPS=${STEPS:-10}
TOL=${TOL:-2e-3}                         # relative tolerance for the step-1 equality assert (bf16-ish)
SWEEP_DIR=${SWEEP_DIR:-${RUNS_BASE:-/cb/ml-eng/aarti/dsa/dsa_runs}/phase2-invariance-$(date +%Y%m%d_%H%M%S)}  # writable NFS w/ space (NOT repo ws)
mkdir -p "${SWEEP_DIR}"
MASTER_LOG="${SWEEP_DIR}/sweep.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "[sweep] SWEEP_DIR=${SWEEP_DIR} DATA=${DATA} STEPS=${STEPS} TOL=${TOL} host=$(hostname) date=$(date -Is)"

CONFIGS=("base4_mb1:4:1" "dp3_mb1:3:1" "dp2_mb1:2:1")
for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r name nproc mbsz <<< "${cfg}"
    RUN_DIR="${SWEEP_DIR}/${name}"
    CVD=$(seq -s, 0 $((nproc - 1)))
    echo "[sweep] ===== ${name}: NPROC=${nproc} MICRO_BSZ=${mbsz} CUDA_VISIBLE_DEVICES=${CVD} ====="
    CUDA_VISIBLE_DEVICES="${CVD}" NPROC="${nproc}" MICRO_BSZ="${mbsz}" BATCH=12 \
        EPOCHS=$((STEPS + 3)) STEPS="${STEPS}" DSA_DEBUG_MASTER=0 \
        TRAIN_FILES="${DATA}" RUN_DIR="${RUN_DIR}" EXP_NAME="inv-${name}" \
        bash "${REPO_ROOT}/examples/dsa/run_minicpm3_dsa_phase2_overfit.sh" \
        'trainer.logger=["console"]' && echo "[sweep] ${name} done -> ${RUN_DIR}" \
        || echo "[sweep] ${name} FAILED (continuing to next config)"
done

echo "[sweep] ===== COMPARISON ====="
SWEEP_DIR="${SWEEP_DIR}" TOL="${TOL}" python3 - <<'PY'
import glob, os, re

sweep = os.environ["SWEEP_DIR"]; tol = float(os.environ["TOL"])
names = ["base4_mb1", "dp3_mb1", "dp2_mb1"]
keys = ["train/loss", "loss/lm", "loss/kl_weighted"]

def parse(run_dir):
    logs = sorted(glob.glob(os.path.join(run_dir, "run-*.log")))
    if not logs:
        return {}
    steps = {}
    with open(logs[-1]) as fh:
        for line in fh:
            if not line.startswith("step:"):
                continue
            s = int(re.match(r"step:(\d+)", line).group(1))
            d = {}
            for k in keys:
                m = re.search(re.escape(k) + r":([-\d.eE+]+)", line)
                if m:
                    d[k] = float(m.group(1))
            steps[s] = d
    return steps

data = {n: parse(os.path.join(sweep, n)) for n in names}

print("\n--- step 1 (must match across configs) ---")
ok = True
for k in keys:
    vals = {n: data[n].get(1, {}).get(k) for n in names}
    base = vals[names[0]]
    line = f"{k:18s} " + "  ".join(f"{n}={vals[n]:.6g}" for n in names if vals[n] is not None)
    maxrel = 0.0
    for n in names[1:]:
        if vals[n] is not None and base:
            maxrel = max(maxrel, abs(vals[n] - base) / (abs(base) + 1e-12))
    status = "OK" if maxrel <= tol else "FAIL"
    ok = ok and (maxrel <= tol)
    print(f"  {line}   max_rel_diff={maxrel:.2e} [{status}]")

print("\n--- train/loss trajectory (drift expected from fp, not asserted) ---")
allsteps = sorted(set().union(*[set(data[n]) for n in names]))
print("  step  " + "  ".join(f"{n:>12s}" for n in names))
for s in allsteps:
    row = "  ".join(f"{data[n].get(s, {}).get('train/loss', float('nan')):12.6g}" for n in names)
    print(f"  {s:>4d}  {row}")

print(f"\n[sweep] STEP-1 INVARIANCE: {'PASS' if ok else 'FAIL'} (tol={tol:g})")
PY
# this is a correctness test — the per-config checkpoints (the trainer always saves one at the last step,
# ~12 GB each) are not needed; drop them so the sweep doesn't accumulate model dirs.
echo "[sweep] removing per-config checkpoints (test run; not needed)"
rm -rf "${SWEEP_DIR}"/*/checkpoints 2>/dev/null || true
echo "[sweep] full logs under ${SWEEP_DIR}"
