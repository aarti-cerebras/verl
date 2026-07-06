# Plan: Phase-1 indexer warm-up — first real SFT-trainer run (smoke)

Goal: run the dense-warmup indexer training **through verl's actual SFT trainer** end-to-end and validate
the integration the standalone overfit test couldn't (loss closure fires, FSDP freeze holds under
`use_orig_params`, `train/indexer/*` reaches wandb, memory/throughput sane). Small scale, single-doc,
short seqlen. Builds on Parts A/B/C (all tested; `model._dsa_indexer_kl` + `_dsa_metrics` already produced).

## Scope / non-goals
**In:** minimal fixed-length dataset, Phase-1 config, launch script, a ~50-step run on **real MiniCPM3-4B**,
checkpoint check. **Out (later):** varlen multi-doc packing (item 6b), `calculate_log_probs=false`, full
32K / ~2B tokens, long-context eval, Phase 2. Single-doc chunks (no per-doc position reset / cross-doc
masking) — fine for a mechanics smoke; the base attention is plain causal over each chunk.

## Prerequisites
- A **mostly-free H100** — MiniCPM3-4B is ~8 GB params (bf16) + activations + the ~0.6 GB LM-head logits at
  seqlen 4096; won't fit on the contended GPU we used for unit tests. Check `nvidia-smi` first.
- Env: **transformers 4.57.1** + `fast_hadamard_transform` (dev: `PYTHONPATH=/tmp/fht_clean:/tmp/tf457lib`;
  the monkey-patch applies the `get_usable_length` shim).
- A tiny text parquet for the smoke (a few MB; a slice of `openbmb/InfLLM-V2-data-5B` or any long-ish text
  with a `text` column).

## Item 1 — minimal dataset  `verl/utils/dataset/packed_pretrain_dataset.py`
Contract from `create_sft_dataset` (`sft_trainer.py:471-484`) and the no-padding collator:
```python
class PackedPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        self.seq_len = config.get("max_length", 4096)
        self.text_key = config.get("text_key", "text")
        # read parquet -> tokenize text_key -> ONE concatenated token stream ->
        # chunk into fixed windows of exactly seq_len (drop remainder). Store windows.
    def __getitem__(self, i):
        ids = self.windows[i]                        # LongTensor [seq_len]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),  # all real (fixed length -> no padding)
            "position_ids": torch.arange(len(ids)),  # single contiguous chunk == one "doc"
            "loss_mask":   torch.ones_like(ids),      # unused by indexer_kl; kept for the contract
        }
```
- `pad_mode=no_padding`; every window is exactly `seq_len`, so the collator's nested tensors are uniform and
  the engine's `use_remove_padding=False` padding is a no-op (no pad tokens → clean `[bsz, seq_len]`).
- Single-doc chunking (no `position_ids` reset) is deliberate for the smoke; multi-doc packing + resets is
  item 6b.
- **Doc-mask coupling for 6b (multi-doc packing).** When per-doc `position_ids` resets are introduced, the
  KL target `p` and the base attention must be masked *the same way* or `p != true base attention`:
  `_dense_warmup_kl` already builds a per-doc block-diagonal causal mask from `position_ids == 0`, so the
  **base flash attention must also be made per-doc** (flash varlen `cu_seqlens`, or a block-diagonal mask) —
  otherwise the base attends cross-doc while the target masks per-doc. Today's `arange` position_ids
  degenerate both to plain causal, so they agree. The target-side math is validated by
  `tests/models/test_minicpm_dsa_integration.py::test_dense_warmup_target_is_blockdiagonal_multidoc` (the
  acceptance oracle 6b must satisfy on the base-attention side).

## Item 2 — config  `verl/trainer/config/sft_trainer_minicpm_dsa_phase1.yaml`
Extends `sft_trainer_engine`. Key overrides (grounded in the config schema):
```yaml
defaults: [ {model@model: hf_model}, {engine@engine: fsdp}, {optim@optim: fsdp}, {profiler@profiler: profiler}, _self_ ]

loss_mode: indexer_kl                     # NEW top-level key read by sft_trainer._build_engine

model:
  path: openbmb/MiniCPM3-4B
  trust_remote_code: true
  use_remove_padding: false               # -> dense [bsz,T] path our forward supports (hf_model.yaml:40)
  enable_gradient_checkpointing: false     # base frozen -> not needed; also avoids _dsa_kl recompute issues
  override_config:                         # merged onto the HF config via setattr (utils/model.py:81-85)
    _attn_implementation: flash_attention_2   # MiniCPMFlashAttention2 (the class our patch swaps)
    dsa_enabled: true
    dsa_overrides: {n_heads: 16, head_dim: 64, rope_head_dim: 32, top_k: 2048,
                    mode: dense_warmup, kl_block_size: 1024, fp8: true, diag_interval: 10}

engine:
  strategy: fsdp
  use_orig_params: true                    # required: frozen base + trainable indexer in one FlatParameter

optim:
  lr: 1.0e-3
  lr_scheduler_type: constant              # + short warmup
  weight_decay: 0.0

data:
  custom_cls: {path: verl/utils/dataset/packed_pretrain_dataset.py, name: PackedPretrainDataset}
  train_files: <tiny.parquet>
  pad_mode: no_padding
  max_length: 4096                         # start small (keeps LM-head logits ~0.6GB); scale later
  micro_batch_size_per_gpu: 1
  train_batch_size: 8
  use_dynamic_bsz: false

trainer:
  total_training_steps: 50
  logger: [console, wandb]
  project_name: DSA                        # -> https://cerebras.wandb.io/aartighatkesar/DSA
  experiment_name: phase1-smoke
  save_freq: 50
  test_freq: -1
  n_gpus_per_node: 1
```
**wandb → Cerebras instance:** verl reads the entity from `WANDB_ENTITY` and the project from
`trainer.project_name`; the base URL is **not** in the config, so it must be set via env (default is
`api.wandb.ai`). Set in the launch script: `WANDB_BASE_URL=https://cerebras.wandb.io`,
`WANDB_ENTITY=aartighatkesar`. The API key is already stored in `~/.netrc` (via
`wandb login --host=https://cerebras.wandb.io`) — **do not** hardcode it in configs/scripts.
(Confirm at impl: how `_attn_implementation` reaches `from_pretrained` — if `override_config` setattr isn't
honored by the loader, pass it via the model loader's automodel kwargs instead.)

## Item 3 — launch  `examples/dsa/run_minicpm3_dsa_phase1_smoke.sh`
```bash
export PYTHONPATH=/tmp/fht_clean:/tmp/tf457lib:$PYTHONPATH   # dev; bake into the image for real runs
export WANDB_BASE_URL=https://cerebras.wandb.io             # Cerebras self-hosted instance
export WANDB_ENTITY=aartighatkesar                          # verl reads entity from this env
# API key lives in ~/.netrc (wandb login --host=https://cerebras.wandb.io) — never hardcode it here
torchrun --nproc_per_node=1 -m verl.trainer.sft_trainer \
  --config-name sft_trainer_minicpm_dsa_phase1
```
(Mirror an existing `examples/sft/*.sh` for the exact verl launcher invocation.)

## Item 4 — run + checkpoint check
- Run ~50 steps; **watch:** `train/indexer/kl` ↓, `train/indexer/topk_recall` ↑, `indexer/nan_frac == 0`,
  stable memory, sane step time. (A `Monitor` on the log grepping `indexer/kl|Traceback|OOM|nan` is handy.)
- **Checkpoint:** confirm a checkpoint is written (`save_contents` includes `"model"`) and that
  `*.indexer.*` params are in it. Optionally add a small script to extract an **indexer-only shard**
  (~50–60 MB) for Phase-2 init.

## Risks / things to watch (from prior analysis)
- **attn_implementation must be flash** — our patch only swaps `MiniCPMFlashAttention2`; if it loads eager,
  the indexer isn't attached. Verify the class after load (Step-0 already confirmed flash works on 4.57.1).
- **Data path = dense** — `use_remove_padding=false` + fixed-length → clean `[bsz, seq_len]` with a 2D
  `attention_mask`; our `_dense_warmup_kl` consumes the 2D mask (and it's all-ones here anyway).
- **Freeze under FSDP** — `use_orig_params=true` is mandatory; verify only `*.indexer.*` have grads and the
  optimizer's param count matches the indexer's.
- **LM head not skipped yet** — at seqlen 4096 the logits tensor is ~0.6 GB (tolerable); keep seqlen small
  until the `calculate_log_probs=false` flag lands (Phase-1 scale-up).
- **GPU contention** — needs a mostly-free H100 (unlike the tiny unit tests).
- **loss closure timing** — `self.engine.module` is resolved lazily at call time; confirm the first step's
  loss reads a non-None `_dsa_indexer_kl`.

## Success criteria
The trainer completes ~50 steps with `train/indexer/kl` clearly decreasing and `topk_recall` rising, no
NaNs, base params unchanged, a checkpoint containing the indexer written. That proves the full verl
integration and unblocks packing / 32K / Phase 2.
