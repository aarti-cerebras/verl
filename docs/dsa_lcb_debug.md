# LiveCodeBench (DSA sparse) — issue & debug log

Investigation of why LiveCodeBench v3 returns anomalously low pass@1 for the MiniCPM3-DSA **sparse** model
served on vLLM, while the same model scores fine on other code benches. Companion to `docs/dsa_eval_report.md`
(§4) and the scorecard at `/cb/ml-eng/aarti/dsa/evals/minicpm3-4B-dsa-k256/`.

**Status: RESOLVED (root cause found) — the LCB number is a HARNESS ARTIFACT, not a model regression.**
A one-line bug in the LCB runner (`base_runner.py::run_batch`) misaligns generations to problems whenever a
parallel task fails. On the slow sparse server ~25–31 tasks fail per run, and each failure both (a) creates 9
empty problems and (b) shifts every subsequent generation back by 9 — stapling correct code onto the wrong
problem. The model is producing coherent, correct solutions; the 2.67% score and the ~230 "empties" are the
**same bug**. An **offline realign + re-grade of the existing trajectories** (§5b, no GPU) confirms it: on the
382 problems DSA answered, corrected pass@1 = **19.14% vs baseline 19.84%** (parity). LCB should still be
**re-run with the fixed harness** for a clean full number; the other 8/9 card benchmarks are unaffected.

---

## 1. Symptom
- LCB v3 (release_v3, n=10, temp 0.2): pass@1 **~0.8%** first, **2.67%** after the mp=6 change.
- Reference: **baseline stock dense = 20.7**, HF reported = 22.6.
- Mechanism of the low score (as first understood): a large fraction of problems (**~43% → 38%**, 260 → 230 of
  612) return **empty generations** → no extractable code → graded 0.
- **Anomalous, not capability:** the same model scores HumanEval+ 65.2 / MBPP+ 61.9 — it clearly can code.

## 2. Environment
- Model: Phase-2 `global_step_2805`, `MiniCPM3DSAForCausalLM` vLLM plugin, `index_topk=256`, bf16, TP=1,
  **`enforce_eager`** (no CUDA graphs validated for the sparse path → slow decode).
- Harness: `tools/LiveCodeBench` `lcb_runner`, OpenAIChat style, `oai_runner`, `--n 10 --temperature 0.2
  --release_version release_v3 --multiprocess <mp> --evaluate`. Driver: `<eval_root>/run_lcb_dsa.sh`.
  Repo copy of the harness: `/cb/ml-eng/aarti/dsa/evals/minicpm3-4B/tools/LiveCodeBench`.
- Serving for the runs: single replica @8192 (run 1); DP=6 @ 32K (runs 2–3).

## 3. Root cause (the alignment bug)
`lcb_runner/runner/base_runner.py::run_batch` assembles parallel results **positionally**:

```python
for output in parallel_outputs:        # pebble pool.map → yields in TASK ORDER
    if output.is_success():
        outputs.append(output.result)  # success: appends ONE list of n strings
    else:
        outputs.extend([""] * self.args.n)   # failure: extends with n SCALAR ""  ← BUG
```

`run_main` then zips `outputs` positionally onto the ordered `benchmark`. Alignment therefore requires **exactly
one entry per task**. On a failed task the code does `extend([""] * n)` — adding **n=10 scalar** entries instead
of `append`-ing a single `["", …]` list. Consequently every failed task injects **9 extra entries** (n−1), and:

1. **creates 9 empty problems** — the benchmark slots that land on the junk `""` scalars, and
2. **shifts every later generation back by 9** — so a correct solution gets filed under the problem 9·k slots
   later (k = number of failures before it).

**One bug explains both symptoms** — the ~230 empties *and* the 2.67% collapse.

**Why DSA-only:** baseline (fast dense replica) had **0 failed pool tasks**, so alignment held (its 20.7 is
real). DSA's slow `enforce_eager` sparse decode produces ~25–31 pool failures per run (timeouts /
`ProcessExpired` / RLIMIT_AS mem-limit / API errors — the `else` branch catches *all* non-success, not just
timeouts). `mp=6` merely reduced failures (260→230 empties); it did not remove the corruption. The
`20260722_133148` run logged **31** "Failed to run the model" prints → 31×9 ≈ 279 shifted entries, of which 230
surface as empties (remainder runs off the end of the 612-problem list).

### The one-line fix
```python
-        outputs.extend([""] * self.args.n)
+        outputs.append([""] * self.args.n)   # one list per task → alignment preserved
```
(Ideally also map results back by `question_id` rather than by position, as a belt-and-suspenders guard.)

### Where the failed tasks come from (the residual empties, after the alignment fix)
With `append`, empties = exactly the genuinely-failed pool tasks (the `20260722_133148` run: **31**,
`succ=581 exc=31 timeouts=0 p_exp=0`). All 31 are `TaskRunStatus.EXCEPTION` and the traceback is
**`assert len(result) == args.n` at `base_runner.py:63`** — the model call returned ≠ n=10 completions. Chain:
`oai_runner._run_single` always requests `n=10` in one call (`client_kwargs["n"]` is fixed; the `n` recursion
arg is only a *retry counter*, it never lowers the requested count). On any OpenAI error — overwhelmingly
`APITimeoutError`, because `--openai_timeout` defaults to **90 s** but even a *solo* n=10 request takes ~79 s on
the slow `enforce_eager` server — it sleeps 30 s and retries up to 10× then returns `[]`. `[]` (len 0 ≠ 10)
trips the assert → the whole problem is dropped to empty.

### Empty-generation fixes (both applied)
1. **`run_lcb_dsa.sh`** — pass `--openai_timeout 1200` (overridable via `OPENAI_TIMEOUT` env). 90 s was below even
   the unloaded request time; 1200 s lets a slow-but-valid n=10 request finish instead of erroring. *Cause fix.*
2. **`base_runner.py:63`** — replace the hard `assert len(result) == args.n` with pad-to-n:
   ```python
   if len(result) < args.n:
       result = result + [""] * (args.n - len(result))
   result = result[: args.n]
   ```
   A short/empty return no longer raises → the problem is graded on whatever completions arrived instead of being
   dropped (and can't cascade). *Safety net.* #1 prevents the short returns; #2 stops any residual one from
   zeroing a problem.

Deeper/optional: split the single n=10 call into per-sample requests in `oai_runner` (one slow sample then costs
1/10, not 10/10); and the real root cause of the slowness is `enforce_eager` — validating CUDA graphs for the
sparse decode path would remove the timeouts at the source.

## 4. Evidence (how the alignment bug was proven)
Compared DSA vs baseline `_eval_all.json` on the **382 problems DSA answered** (identical set):

| set | baseline pass@1 | DSA pass@1 |
|---|---|---|
| full 612 (empties=0) | 20.74% | 1.67% |
| the 382 DSA answered | 18.59% | **2.67%** |
| the 230 DSA left empty | 24.30% (baseline) | — (empty) |

Empties were **not** a hard subset (baseline scored *higher* on them), so empties alone didn't explain the gap.
Then the failure-mode split (same 382 problems, 3820 samples) exposed the real signature:

| | pass@1 | runtime error (−4) | wrong answer (−2) | TLE (−3) |
|---|---|---|---|---|
| baseline | 18.6% | 11.5% | 66.6% | 3.3% |
| DSA k256 | 2.7% | **80.5%** | 16.0% | 0.8% |

DSA fails by **crashing**, not by wrong answers. Inspecting the code:
- **283 leetcode-style problems**: DSA emits the *expected* method name in only **56**; in **227** it never does.
- Of DSA's "wrong-method" generations, **90% (137/152) are literally another problem's expected method** from
  the same run.
- The mismatch offsets are **clean multiples of 9** (−9, −18, −27, … −81) in contiguous blocks — the (n−1)
  fingerprint of the `extend` bug.
- **Content proof:** the slot for qid **2916** ("check-if-it-is-possible-to-split-array") contains a coherent,
  correct solution to qid **2884** ("length-of-the-longest-valid-substring") — the problem 9 slots earlier.
  The model wrote good code; the harness stapled it to the wrong problem.

## 5. Superseded hypotheses (from earlier in the investigation)
| Hypothesis | Verdict |
|---|---|
| Throughput / timeouts | Real in run 1 (552 timeouts) and a *trigger* of pool failures, but not the mechanism; mp=6 gave timeouts=0 yet the bug persisted via non-timeout failures |
| DP deployment / request crossing | ✗ — not a serving cross-talk; it's the local harness assembling results positionally |
| Prompt length / long-context sparsity | ✗ — empty vs non-empty prompt lengths ~equal |
| Code extraction | ✗ — 0 extraction misses; clean ```python``` on all non-empty outputs |
| General model code ability | ✗ — HumanEval+ 65.2 / MBPP+ 61.9; and DSA's LCB code is coherent, just misfiled |
| **Real competitive-programming weakness / BC drift** | ✗ — the 2.67% is a harness artifact, not model degradation |

## 5b. Offline re-grade (corrected estimate WITHOUT a GPU re-run)
The `extend()` corruption is a *deterministic* permutation, so the saved trajectories can be realigned offline
and re-graded against the correct test cases (script: `<eval_root>/regrade_realigned.py`, uses the stock LCB
`codegen_metrics`, CPU-only). Inversion: walk benchmark/file order — each non-empty slot = the next successful
prompt; each run of exactly 10 empties = one failed prompt. Empty slots form **23 clean runs of 10** (=230), and
after realigning, generated method names match the true problem's expected method in **258/268** checkable
(leetcode) cases — inversion confirmed.

**Corrected pass@1 (realigned trajectories, graded vs the correct problems):**

| set | DSA k256 (misaligned→**realigned**) | baseline dense (same set) |
|---|---|---|
| the 382 problems DSA generated | 2.67% → **19.14%** | 19.84% |
| all 612 | 1.67% → **11.94%** | 20.74% |

→ **On the problems it actually answered, DSA sparse ≈ dense (−0.7 pt, within the ~1–2 pt sparsity noise).**
The full-612 gap is entirely the 230 non-generated problems (failed pool tasks + truncated tail — a
serving/throughput artifact of slow `enforce_eager` decode, not code quality). The 2.67% was 100% the alignment
bug.

## 6. Status of fixes & re-run
**All three fixes applied:**
| Fix | Location | Purpose |
|---|---|---|
| alignment: `extend`→`append` | `base_runner.py:96` | one entry/task → generations stay aligned; only true failures empty |
| empty-gen #1: `--openai_timeout 1200` | `run_lcb_dsa.sh` | slow n=10 request finishes instead of `APITimeoutError`→retry→`[]` |
| empty-gen #2: pad-to-n vs hard assert | `base_runner.py:63` | short return graded on partial samples, never drops the problem |

(`base_runner.py` is the shared vendored harness under `minicpm3-4B/tools/LiveCodeBench`; safe for the dense
baseline — it had 0 failed tasks, and padding only changes behavior on a short return.)

**Fixed re-run — DONE (2026-07-23 01:26, ml-eng-gpu-16).** Fresh **DP=8** k256 sparse server on :8010 (all 8 GPUs;
`serving/serve_dsa_dp.sh`, `serving_stage0`, `global_step_2805`, `enforce_eager`, `DSA_SPARSE=1`,
`index_topk=256`), LCB at **MP=8**, `--openai_timeout 1200`. Completed 612/612 in **1h34m** with
**`succ=612 exc=0 timeouts=0`** and **0 empties**. Result:

| | pass@1 (612) | fail split: WA(−2) / RTE(−4) / TLE(−3) |
|---|---|---|
| **DSA k256 (fixed re-run)** | **20.52%** | 62% / 12.8% / 4.5% |
| baseline dense | 20.74% | 66.6% / 11.5% / 3.3% |

→ **Parity: −0.22 pt.** The old pathological 80.5%-runtime-error signature is gone — DSA now fails the *normal*
way (wrong answer), same shape as dense. End-to-end confirmation that the 2.67% was 100% the two harness bugs;
DSA sparse k256 ≈ dense on LiveCodeBench. Matches the §5b offline re-grade (19.14% on the answered set; now
higher because all 612 are answered).
Run log `/tmp/dsa_lcb_dp8_fixed_20260722_233554.log`; harness log
`<eval_root>/livecodebench/logs/20260722_233554_livecodebench.log`.

**Remaining (optional):** sanity re-run **stock dense** through the padded harness to confirm it still
reproduces ~20.7 (isolates any harness regression from the padding change — expected none, since dense had 0
failures so padding never triggers).

## 7. Recommendation
The prior "LCB unresolved / possible model weakness" conclusion is **withdrawn**. LCB was measuring a harness
alignment bug, not the sparse model. The DSA sparse model writes coherent, correct competitive-programming code;
its true LCB score is unknown until the fixed-harness re-run but is expected to be near dense. The other 8/9
benchmarks already gave a clean, positive picture (sparse ≈ dense; HumanEval+ the one code cost, mostly BC-drift).

## 8. Artifacts / repro
- Bug: `<harness>/lcb_runner/runner/base_runner.py::run_batch`, the `else: outputs.extend([""] * self.args.n)`
  line. Parallel driver: `lcb_runner/utils/multiprocess.py` (pebble `pool.map`, order-preserving).
- Analysis (offset/method-name/content proof): ad-hoc python over the two `_eval_all.json` files under
  `minicpm3-4B` (baseline) and `minicpm3-4B-dsa-k256` (DSA), `output/MiniCPM3-4B/Scenario.codegeneration_10_0.2*`.
- Run logs: `<eval_root>/livecodebench/logs/20260722_133148_livecodebench.log` (31 "Failed to run the model"
  prints); earlier runs `/tmp/dsa_lcb_rerun*.log`.
- Driver: `<eval_root>/run_lcb_dsa.sh` (`MP` env). Server: DP via `serve_dsa_entry.py --data-parallel-size N`.
- eval_root = `/cb/ml-eng/aarti/dsa/evals/minicpm3-4B-dsa-k256`.
