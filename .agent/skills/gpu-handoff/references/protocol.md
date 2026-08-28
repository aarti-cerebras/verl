# GPU Job File Protocol

Use this protocol for both sides of a GPU handoff. Paths are relative to the repository root unless
the request explicitly says otherwise.

## Directory lifecycle

```text
.agents/gpu_jobs/<job-id>/
    request.md       requester writes once
    claim.md         executor creates before running anything
    result.md        executor writes at terminal state
    logs/            complete stdout/stderr per command
    artifacts/       requested machine-readable outputs, profiles, traces, or summaries
```

The job ID format is:

```text
YYYYMMDDTHHMMSSZ-short-lowercase-slug
```

Do not use one file as both request and result. This preserves the original experiment contract.

Terminal statuses have these meanings:

- `PASS`: every mandatory success criterion passed.
- `FAIL`: execution completed and at least one mandatory criterion failed.
- `BLOCKED`: execution could not start or continue because a prerequisite, permission, GPU
  allocation, dataset, checkpoint, or required choice was unavailable.
- `ERROR`: the harness, environment, or command failed before the requested behavior could be
  evaluated reliably.

## Requester workflow

### Decide what belongs in one job

One job should answer one coherent validation or performance question. Commands that require the
same checkout, environment, GPU allocation, and interpretation may share a job. Split unrelated
experiments or incompatible environments into separate jobs.

Prefer checked-in or already-created test runners over large inline shell programs. If a new probe
is required, add it to the workspace first and ask the executor to run that file. The executor
should not need to reconstruct code from prose.

### Request template

Create `request.md` with every section below. Replace all instructional placeholders.

```markdown
---
schema_version: 1
job_id: 20260828T190000Z-short-slug
created_at_utc: 2026-08-28T19:00:00Z
status: pending
repository_root: /absolute/path/to/repository
requested_git_revision: <git rev-parse HEAD>
requested_gpus: 1
gpu_memory_min_gib: 40
timeout_minutes: 30
mutation_policy: workspace-artifacts-only
network_required: false
---

# Objective

One precise question this GPU run must answer.

# Why GPU execution is required

Name the CUDA behavior, kernel, memory property, performance measurement, graph capture, or
distributed behavior that a CPU run cannot validate.

# Code and context

- Relevant files and symbols.
- Current implementation or hypothesis being tested.
- Required repository dirty changes, if any.
- Prior result that must be reproduced or compared, if any.

# Preconditions

- Required GPU model or capability, count, and minimum free memory.
- Required Python environment, packages, checkpoint, dataset, and environment variables.
- Required services or ports.
- State that must not already be running.

# Mutation and safety boundaries

- State exactly what may be written or changed.
- State what must remain read-only.
- State whether dependency installation, downloads, process termination, or service startup is
  authorized. Absence means not authorized.
- State cleanup requirements for processes started by this job.

# Execution

Run commands in order from `repository_root`. Each command gets a numbered log such as
`logs/01-environment.log` or `logs/03-tests.log` containing complete stdout and stderr.

## Step 1: Environment capture

```bash
<exact command>
```

Expected exit code: `0`
Timeout: `2 minutes`

## Step 2: Main validation

```bash
<exact command>
```

Expected exit code: `0`
Timeout: `20 minutes`

# Required artifacts

| Path | Required | Purpose |
|---|---:|---|
| `logs/01-environment.log` | yes | environment evidence |
| `logs/02-main-validation.log` | yes | complete validation output |
| `artifacts/metrics.json` | no | machine-readable measurements |

# Success criteria

| ID | Mandatory | Criterion | Evidence to report |
|---|---:|---|---|
| C1 | yes | All requested commands return their expected exit codes | command table and logs |
| C2 | yes | State the actual correctness invariant | exact observed value/assertion |
| C3 | no | State an informative performance target | measured latency/throughput |

# Required result fields

In addition to the standard result template, report these job-specific values:

| Field | Type/unit | Source |
|---|---|---|
| `example_metric` | milliseconds | profiler or metrics artifact |

# Retry policy

State the maximum retries and the exact retryable conditions. Default: no retries. Never change
test parameters merely to obtain a pass.

# Result path

Write the terminal report to `.agents/gpu_jobs/<job-id>/result.md` using the GPU handoff result
template. Do not report success only in chat.
```

### Request quality gate

Before handing off, confirm:

- no placeholder remains;
- every command is copy-paste executable from the stated working directory;
- required inputs exist or are clearly named as executor prerequisites;
- success criteria distinguish correctness from optional performance observations;
- timeouts and GPU requirements are plausible;
- source mutation is not accidentally authorized;
- the result fields are sufficient to decide the next engineering step.

## Executor workflow

### Claim

Read the complete request before running anything. Resolve the repository root and verify the
requested revision and relevant dirty files. Create `claim.md`:

```markdown
---
schema_version: 1
job_id: <job-id>
claimed_at_utc: <UTC timestamp>
executor: <agent or session identifier>
hostname: <hostname>
cuda_visible_devices: <literal value or unset>
---

The request has been read. Execution will follow its mutation, timeout, retry, and cleanup bounds.
```

If another valid claim exists, do not execute concurrently. Write or request coordination instead.

### Execute

1. Capture the requested environment evidence before the main command.
2. Run only the listed commands, in order, with the stated timeouts.
3. Record exact commands, start/end timestamps, duration, exit code, and complete combined output.
4. Do not improvise a materially different experiment. Mark `BLOCKED` if a missing choice would
   change what is being tested.
5. Apply only the stated retry policy. Record every attempt.
6. Clean up only processes and temporary resources started by the job, according to the request.
7. Inspect required artifacts before assigning the terminal status.

### Result template

Write `result.md` with every section below. Use `not available` with an explanation when a field
cannot be obtained; do not silently omit it.

```markdown
---
schema_version: 1
job_id: <job-id>
completed_at_utc: <UTC timestamp>
status: PASS
requested_git_revision: <revision from request>
observed_git_revision: <git rev-parse HEAD>
executor: <agent or session identifier>
hostname: <hostname>
---

# Summary

One to three sentences answering the objective. State the terminal status and decisive evidence.

# Environment

| Field | Observed value |
|---|---|
| Repository root | `/absolute/path` |
| Git revision | `<sha>` |
| Relevant dirty files | `<paths or none>` |
| CUDA_VISIBLE_DEVICES | `<literal value or unset>` |
| GPU model and count | `<model, count>` |
| Driver | `<version or unavailable>` |
| CUDA runtime/toolkit | `<version or unavailable>` |
| Framework | `<PyTorch/vLLM/Triton versions>` |
| Peak GPU memory | `<GiB or not measured>` |

# Commands

| Step | Exact command | Exit code | Duration | Log |
|---:|---|---:|---:|---|
| 1 | `<command>` | 0 | 4.2 s | `logs/01-environment.log` |

# Criteria

| ID | Mandatory | Verdict | Observed evidence |
|---|---:|---|---|
| C1 | yes | PASS | All expected exit codes observed |

# Requested measurements

| Field | Observed value | Evidence |
|---|---:|---|
| `example_metric` | `1.23 ms` | `artifacts/metrics.json` |

# Artifacts

| Path | Present | Description |
|---|---:|---|
| `logs/01-environment.log` | yes | complete environment output |

# Failures or deviations

Write `None` only if there were no failures, retries, skipped steps, parameter changes, warnings
that affect interpretation, or differences from the requested checkout.

# Conclusion

State what the evidence establishes, what it does not establish, and the next engineering action
supported by the result.
```

### Status rules

- Do not use `PASS` if a mandatory criterion lacks evidence, even when commands returned zero.
- Use `FAIL` for an observed assertion, parity, safety, or required performance failure.
- Use `BLOCKED` when required hardware, files, authorization, or an input choice is unavailable.
- Use `ERROR` for harness crashes, environment corruption, unexplained timeouts, or failures that
  prevent interpretation of the target behavior.
- A cleanup failure is a deviation and may make the job `ERROR` when it leaves the environment in
  an unsafe or ambiguous state.

## Consumer workflow

When reading a result:

1. Match `job_id` and requested revision to `request.md`.
2. Check the observed revision and relevant dirty files.
3. Verify that every mandatory criterion has evidence and every referenced required artifact
   exists.
4. Inspect the relevant log or artifact when the summary alone is insufficient.
5. Separate correctness, graph-capture, memory, and performance conclusions; one does not imply
   another.
6. If the result is stale, malformed, or incomplete, state that explicitly and create a corrected
   follow-up job rather than retroactively editing the claimed request.
