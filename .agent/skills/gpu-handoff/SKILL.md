---
name: gpu-handoff
description: Hand GPU/CUDA execution to another agent through a self-contained file request and consume its structured result. Use when meaningful validation, benchmarking, profiling, compilation, or reproduction must run on a GPU, or when explicitly asked to execute a queued GPU job. Do not use for CPU-only work or merely editing GPU-related code.
---

# GPU Handoff

Keep GPU execution reproducible and separate from the agent doing ordinary implementation or
analysis. Communicate through a job directory under `.agents/gpu_jobs/` at the repository root.

Read [references/protocol.md](references/protocol.md) completely before creating, executing, or
consuming a GPU job.

## Choose the role

- **Requester is the default.** If the current task reaches a step that requires a GPU, do not run
  it directly. Create a self-contained `request.md` for the GPU agent and tell the user its path.
- **Executor is explicit-only.** Enter this role only when the user or calling agent explicitly
  asks to claim or execute a particular GPU request. Execute only the authorized request, retain
  raw logs, and write the required `result.md`.
- **Consumer.** When a completed `result.md` is available, verify that it answers the request and
  use the reported evidence. Do not silently treat missing, stale, or malformed results as a pass.

## Required behavior

1. Use a unique directory: `.agents/gpu_jobs/<UTC timestamp>-<short-slug>/`.
2. Make `request.md` sufficient for an agent with repository access but no conversation history.
3. Include exact commands, working directory, GPU needs, timeout, mutation permission, expected
   artifacts, success criteria, and the tailored result fields.
4. Keep the request immutable after an executor creates `claim.md`. Create a new job for revised
   commands or criteria.
5. The executor writes raw command output under `logs/` and concise evidence to `result.md` using
   exactly one terminal status: `PASS`, `FAIL`, `BLOCKED`, or `ERROR`.
6. Never fabricate GPU results. A queued request is pending evidence, not validation.
7. Preserve existing authorization boundaries. A handoff does not authorize source edits,
   dependency installation, destructive cleanup, downloads, or external mutations unless the
   request explicitly does so.
8. Keep GPU identity and allocation explicit in the result, including `CUDA_VISIBLE_DEVICES`, GPU
   model, count, driver/runtime information available, and peak memory when relevant.
9. Report every requested command with exit code and duration. Keep full logs in files rather than
   pasting unbounded output into `result.md`.
10. On consumption, check repository revision and dirty-state information before attributing a
    result to the current code. State any mismatch or uncertainty.

## Completion

- A requester completes the handoff when a valid `request.md` exists and its path is reported.
- An executor completes only when `result.md` and referenced logs/artifacts exist.
- A consumer accepts a pass only when all mandatory success criteria have explicit supporting
  evidence. Otherwise report the result as incomplete, failed, blocked, or stale.
