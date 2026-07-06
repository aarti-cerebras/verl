# DSA Indexer Checkpoint Round-Trip (#10)

Whether the trained lightning indexer survives save→reload so Phase 2 can consume it. Verified end-to-end
on 4×H100. Related: [`dsa_fsdp_sharding_notes.md`](dsa_fsdp_sharding_notes.md).

## TL;DR

- **Sharded / resume path WORKS** — the path Phase 2 will use. The indexer round-trips completely and
  training resumes seamlessly. ✅
- **HF-export merge path** — merges all keys (incl. indexer) but currently fails earlier for MiniCPM3
  (trust-remote-code staging), and is **not needed for Phase 2**. Secondary; documented gap.

## What was verified

### L1 — module-level round-trip (committed unit tests, CPU)
`tests/models/test_dsa_indexer.py`:
- `test_indexer_state_dict_roundtrip_is_functionally_identical` — a trained indexer's `state_dict`
  round-trips and reproduces **bit-identical scores** (fp8 production path).
- `test_strict_load_rejects_missing_indexer_keys` — a checkpoint containing `indexer.*` keys **fails a
  strict load** into a model without indexers attached (the Phase-2 ordering gotcha can't pass silently).

### L2 — real path on 4×H100 (run `dsa_runs/ckpt-verify`, 2026-07-06)
Actual `FSDPCheckpointManager` under FSDP2, DP=4, sharded state dict:
- **Completeness:** all **310 indexer params** (`wq_b`, `wk`, `k_norm.weight`, `k_norm.bias`,
  `weights_proj` × 62 layers) present in the sharded checkpoint; **key sets identical across all 4 ranks**;
  DTensors with `weights_proj` correctly fp32. (`self.model.state_dict()` at save has no `requires_grad`
  filter, so frozen base + trainable indexer both saved.)
- **Reload:** resuming from `global_step_2` — all 4 ranks `Loaded model` with **no missing/unexpected-key
  errors** (strict `load_state_dict`), steps 3–4 continued with **no loss discontinuity**
  (2.92 → 2.914 → 2.938), optimizer state restored.

### #6 — cross-rank value-diff (4-GPU verification script)
`tests/models/dsa_ckpt_multirank_valuediff.py` (torchrun, 4 ranks — manual, not CI): set rank-0's indexer to
a known deterministic pattern, **corrupt every other rank's indexer with `-999`**, run the exact engine path
(`apply_fsdp2` + `fsdp2_load_full_state_dict`, which broadcasts `module.state_dict()` from rank 0), then
gather each indexer param's full tensor on every rank and bit-compare to the pattern. **Result:
PASS — 40 checks across 4 ranks, 0 mismatches**; the non-rank-0 corruption was overwritten, so the freshly
created `indexer.*` params propagate correctly. Root cause understood from code: the engine captures
`full_state = module.state_dict()` *after* `attach_indexers`, so indexer params are in the broadcast set.
Run: `PYTHONPATH=.devlibs/tf457lib:$PWD torchrun --standalone --nproc_per_node=4 tests/models/dsa_ckpt_multirank_valuediff.py`.

## Phase-2 load recipe (the decided path)

Phase 2 continues training in verl, so it consumes the **sharded checkpoint via resume**, NOT the HF merge:

1. Build the base model, `attach_indexers(...)` (creates the `indexer.*` param slots), freeze/config as
   Phase 2 needs, FSDP2-wrap — **exactly as Phase 1** (indexers must exist before load).
2. Point at the Phase-1 `default_local_dir` and let `resume_mode=auto` (or `resume_from_path`) load
   `global_step_N` — strict load then fills the trained indexer weights.

**Ordering gotcha:** `attach_indexers` MUST run before the checkpoint load. If indexers aren't attached, the
`indexer.*` keys are unexpected → strict load errors (safe, caught) — but if any code path uses
`strict=False`, they'd be silently dropped → a re-initialized (untrained) indexer. Keep the load strict.

## HF-export merge — known gap (secondary)

`python -m verl.model_merger merge --backend fsdp` on the checkpoint:
- The merge logic iterates **all** keys, so indexer weights would be carried into the merged state dict.
- But it currently **fails earlier for MiniCPM3**: the merger expects `<ckpt>/huggingface/modeling_minicpm.py`
  (trust-remote-code modeling file) which `save_checkpoint` did not stage. This is a trust-remote-code
  export friction, independent of the indexer.
- Whether `save_pretrained` then preserves the non-base-arch `indexer.*` keys in the safetensors is
  **untested** (blocked by the above).

This path only matters for a portable HF checkpoint for **external inference**, not for Phase 2. Fixing it
(stage the modeling file + confirm indexer keys survive `save_pretrained`) is a separate follow-up task if
an external export is ever needed.
