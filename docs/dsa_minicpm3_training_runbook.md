# DSA on MiniCPM3-4B — End-to-End Training Runbook

Consolidated, reproducible record of how we convert the released instruct **`openbmb/MiniCPM3-4B`**
to **DSA (DeepSeek Sparse Attention)** — data → checkpoints → training → serving → evals — with the
exact scripts, commands, and every gotcha/debug we hit along the way.

Repo root: `/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl`
Data root (writable NFS): `/cb/ml-eng/aarti/dsa`

---

## 0. Overview & environment

### The recipe (2-stage, DeepSeek-V3.2 §2.1.1 / arxiv 2512.02556)

DSA is added by **continued training in two stages** — NOT by separate-teacher distillation:

- **Phase 1 — dense warm-up:** freeze the whole base, train ONLY the lightning indexer. Indexer loss =
  `KL(p_{t,:} ‖ softmax(I_{t,:}))` where `p` = main attention summed over heads, L1-normalized. Peak LR 1e-3.
- **Phase 2 — sparse stage:** top-k sparse attention on, **train all params**. Two decoupled objectives:
  - indexer ← selected-set KL only (KL restricted to the selected set `S_t`, indexer input **detached**);
  - base ← plain LM cross-entropy.

Because we do **not** have MiniCPM3's pretraining corpus, Phase 2's LM loss is **behavior-cloning
self-distillation**: next-token CE on trajectories generated offline from the *dense* MiniCPM3-4B in its own
chat template. The student starts == teacher, so it only has to absorb the sparse-attention approximation
(low LR ~1e-5) → minimal instruct-capability drift. (This replaces the old repo plan's online-teacher /
`dsa_distill_loss` / top-k vocab-logit KL machinery — none of it is needed.)

**`top_k=512` is the locked default** (train == deploy). DeepSeek used 2048 for 128K context (1.6% sparsity);
MiniCPM3-4B is 32K, so 512/32K = 1.6% matches their ratio. Sweeps later tightened to 256/128 (see §5).

### Compute & container

- Runs **inside the verl `vllm020.dev1` container** on `ml-eng-gpu-11` (SLURM), **8× H100 80 GB**.
- Compatible stack: `torch 2.11.0+cu130`, `vllm 0.20.2`, `transformers 4.57.1`, `huggingface_hub 0.36.2`.
- ⚠ **GPUs are SHARED** — other tenants (different PID namespace) hold 20–35 GB/GPU but are invisible to
  `nvidia-smi --query-compute-apps`. vLLM at util 0.9 OOMs. Size util from *actual free* memory, or gate on
  idle (see `wait_idle_and_gen.sh`).
- **Durable long runs:** `setsid nohup <cmd> </dev/null >LOG 2>&1 &` (SIGHUP-immune, survives SSH loss; not
  harness-tracked → poll the logfile). Kill by process-group, never blanket `pkill`.

### The transformers-4.57.1 requirement (critical, affects every step)

MiniCPM3-4B's trust-remote-code modeling targets transformers 4.x and **does not load under the container's
transformers 5.x**. Pinning system-wide breaks vLLM. Solution: stage 4.57.1 **in-repo** (NFS, persists across
container restarts) and shadow it via `PYTHONPATH`:

```bash
# one-time (recreate .devlibs/tf457lib):
pip install --no-deps --target=.devlibs/tf457lib transformers==4.57.1 'huggingface_hub<1.0'
```

Two libs must be shadowed: **transformers** (system 5.x) AND **huggingface_hub** (4.57.1 pins <1.0; 0.36.2
works). `tokenizers`/`safetensors`/`accelerate` from the system are already in range — do NOT stage them.
A single build-time compat shim is applied automatically at model build:
`DynamicCache.get_usable_length = lambda self, n, layer_idx=0: self.get_seq_length(layer_idx)`.

Common env exported for every training / tokenizing / serving command:

```bash
export REPO=/net/aarti-vm/srv/nfs/aarti-data/ws/code/ws_repos/dsa/verl
export DEVLIBS=$REPO/.devlibs/tf457lib
export PYTHONPATH=$DEVLIBS:$PYTHONPATH          # MiniCPM3 tokenizer/model need transformers 4.57.1
export WANDB_BASE_URL=https://cerebras.wandb.io # Cerebras self-hosted wandb (key in ~/.netrc)
export WANDB_ENTITY=aartighatkesar             # project = DSA; dashboard cerebras.wandb.io/aartighatkesar/DSA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Logging convention (always):** every run must record its full launch — outer command line + every env var
consumed (incl. debug flags like `DSA_DEBUG_MASTER` that never reach the traced `torchrun` args). The launch
scripts echo `cwd`, `cmdline`, `argv`, and an `env:` line so logs are self-reproducing.

---

## 0.5. DSA mechanics — how the sparse attention kernel uses top-k

The whole method is: a lightweight **lightning indexer** scores every key, `top-k` picks the k best-scoring
keys per query, and the **main attention is computed over only those k keys** (not all T). Selection is
non-differentiable (stop-grad), which is what lets the indexer and the base model train on decoupled losses.

**Step 1 — the indexer produces a score matrix `I`** (`verl/models/transformers/dsa_indexer.py::scores`):
```
I[t, s] = Σ_h  (w[t,h] · softmax_scale) · ReLU( ⟨q_idx[t,h], k_idx[s]⟩ )
```
`q_idx` = per-head indexer query (16 heads × 64 dim); `k_idx` = a single MQA key shared across heads;
`w` = per-head weights — all projected from the hidden state. The dot runs in the **dequantized FP8 (E4M3)**
domain (optional Hadamard `rotate_activation` first), so training sees exactly the serving numerics. This is a
separate cheap attention whose only job is to *rank* keys.

**Step 2 — top-k selection** (`minicpm_dsa.py:475`, training path):
```python
top_k = min(attn.dsa.top_k, T)
idx_b = (indexer_scores[:, q0:q1] + bias_b).topk(top_k, dim=-1).indices.detach()   # [bsz, B, k]
```
- The **causal + document bias** (`-inf` on future / cross-doc keys) is added *before* the top-k, so masked
  keys can never be selected.
- `.detach()` (stop-grad) is essential: **no gradient flows through the selection.** The indexer is trained
  *only* by the selected-set KL (`_sparse_indexer_kl`); the base is trained *only* by LM loss over the
  gathered attention. This is the §0 decoupling made concrete.

**Step 3 — attention over only the k selected keys** (`minicpm_dsa.py:480-486`):
```python
Kg = key_states[b].index_select(1, idx_b[b].reshape(-1)).reshape(H, B, top_k, dqk)   # gather selected keys
Vg = value_states[b].index_select(1, idx_b[b].reshape(-1)).reshape(H, B, top_k, dv)  # gather selected values
s  = einsum("bhBd,bhBkd->bhBk", query, Kg) * scale + bias_sel     # scores over k keys only
a  = softmax(s)                                                    # softmax denominator over k, not T
o  = einsum("bhBk,bhBkd->bhBd", a, Vg)                            # weighted sum of the k values
```
The softmax denominator spans **k keys, not T** — that's the sparsity. Tiled over query blocks to bound the
`[b,H,B,k,d]` peak. **Parity oracle:** with `top_k ≥ T` the selected set is the full causal set, so this
equals dense attention exactly (softmax is permutation-invariant) — the free correctness test used everywhere.

**At serve time — vLLM's FlashMLA-sparse kernel** (`scripts/dsa/vllm_minicpm3_dsa/attention.py:422`): same
selection, fused gather+attend instead of `index_select`+`einsum`:
```python
if self.use_sparse:
    self.indexer(hidden_states, q_c, positions, self.rotary_emb)   # side effect: writes topk_indices_buffer
...
attn_out = self.mla_attn(q_pad, kv_c_pad, k_pe_pad, ...)           # FLASHMLA_SPARSE reads that buffer
```
- The indexer runs **first** and writes a shared `topk_indices_buffer [max_batched_tokens, index_topk]`
  (int32) = the selected key positions per query.
- `MLAAttention` is built with `is_sparse=True` + that buffer → vLLM selects the `FLASHMLA_SPARSE` backend,
  which gathers the selected latent-KV and does MQA over just those keys (on the absorbed 256-d latent, padded
  to 576).
- **Unused slots use the `-1` sentinel** (`topk_indices_buffer[:, k:] = -1`) = "no key"; this is also how the
  `DSA_FORCE_TOPK` debug knob shrinks the effective k. `top_k` must be a multiple of 128; `index_topk` in
  `config.json` is what flips `use_sparse` on.

**One-line summary:** indexer scores all keys → `topk` picks the k best per query (detached, mask applied
first) → attention runs over only those k gathered keys. Training uses `index_select`+`einsum` (autograd flows
into the *values*, never the *selection*); serving hands the same index buffer to the FlashMLA-sparse kernel.

---

## 1. Data setup & generation

Two disjoint pipelines: Phase-1 long-context pretrain windows, and Phase-2 self-distillation SFT trajectories.

### 1a. Phase-1 data — long-context windows (indexer warm-up)

| Split | HF repo | Column | Role |
|---|---|---|---|
| train + in-dist val | `openbmb/InfLLM-V2-data-5B` | `text` | long-context corpus; seeded disjoint split |
| OOD val | `openbmb/Ultra-FineWeb` | `content` | distinct distribution, quality-scored |

Each doc's **raw text is re-tokenized with the MiniCPM3-4B tokenizer**, filtered to ≥ `seq_len` tokens,
truncated to exactly `seq_len` — one document per row. Row schema: single `input_ids` column (list of
`seq_len` ints), consumed by `PackedPretrainDataset`. Each parquet gets a sibling `.MANIFEST.json`
(seed, resolved source SHA, tokenizer, counts, git commit) for byte-identical reproduction.

**One-command build:**

```bash
examples/dsa/build_phase_a_datasets.sh
# defaults: SEED=1234 SEQ_LEN=4096 TRAIN_WINDOWS=2048 VAL_WINDOWS=256
#           OOD_LENGTHS=1024,2048,4096 OOD_PER_LEN=128 OOD_MAX_FILES=64 OOD_MIN_SCORE=0.9
#           OUT_DIR=$REPO/data/dsa/phase_a
# reproduce a past build: REV_INFLLM=<sha> REV_UFW=<sha> SEED=1234 examples/dsa/build_phase_a_datasets.sh
```

Underlying scripts (what the wrapper runs):

```bash
# InfLLM train + document-disjoint in-dist val
python examples/dsa/prepare_real_data.py \
  --out data/dsa/phase_a/infllm_minicpm3_4096_train.parquet \
  --val_out data/dsa/phase_a/infllm_minicpm3_4096_val.parquet \
  --num_windows 2048 --val_windows 256 --seq_len 4096 --seed 1234 \
  --oversample 1.0 --model openbmb/MiniCPM3-4B

# Ultra-FineWeb OOD val at multiple lengths (one parquet per length)
python examples/dsa/prepare_ood_data.py \
  --out_prefix data/dsa/phase_a/ood_ultrafineweb_minicpm3 \
  --lengths 1024,2048,4096 --per_len 128 --max_files 64 \
  --min_score 0.9 --seed 1234 --model openbmb/MiniCPM3-4B
```

Outputs (`data/dsa/phase_a/`): `infllm_minicpm3_4096_{train,val}.parquet`,
`ood_ultrafineweb_minicpm3_L{1024,2048,4096}.parquet`. Phase-B 32K: rerun with `SEQ_LEN=32768`.
Smoke (synthetic, no download): `python examples/dsa/prepare_smoke_data.py --out ~/data/dsa_smoke/train.parquet --rows 600 --words 500`.

**Gotchas (Phase-1 data):**
- **Never reuse InfLLM's `token_ids` column** — it's CPM-5 vocab, incompatible with MiniCPM3-4B (vocab 73440).
  Always retokenize `text`. Also strip literal `<s>`/`</s>` first (else `<s>`→BOS id 1 gives a double-BOS).
- `prepare_real_data.py` dedups train (token-tuple key) and drops any val row whose tokens appear in train
  (leak-free). `--windows_per_doc>1` splits at the *document* level.
- **InfLLM 4K yield ≈ 0.64–14%** (most docs < 4096 tok); a 100M-token build scans ~3.8M docs. Ultra-FineWeb is
  short → high lengths starve; raise `OOD_MAX_FILES`. Quality filter at `min_score 0.9` ≈ top quartile
  (dropped ~69.5% of docs in the recorded build).
- `/scratch/aarti` is root-owned/unwritable → write datasets to repo NFS (`data/dsa/phase_a/`).

### 1b. Phase-2 data — self-distillation SFT trajectories

Source: gated `openbmb/UltraData-SFT-2605` `no_think` split (needs an HF token with access). Pipeline:
**select prompts → generate trajectories (dense MiniCPM3-4B) → filter to SFT parquet.**

**Step 1 — select prompts** (`select_prompts.py`, CPU-only; gold responses discarded, we self-generate).
Uses `examples/dsa/m3a_split.json` for per-config fractions + language filter
(Code 30% / Math 30% en; Multi-lang-Math, Multi-lang-Knowledge 10% zh; Knowledge 10% en; Chinese-general 5% zh;
IF 5% en → ~75% EN / 25% ZH):

```bash
python scripts/dsa/select_prompts.py \
  --out <run_dir>/prompts.jsonl \
  --split-json examples/dsa/m3a_split.json --total 60000 --seed 1234
```

Each row carries `source_uid` + `prompt_sha256` (stable dedup key). **Net-new / additional data** is made with
`--exclude-sha` (accepts prior `prompts.jsonl` or a bare-sha file; unions and skips those shas so a re-run is
disjoint) — this is how the +60k code boost was built on top of the original corpus:

```bash
python scripts/dsa/select_prompts.py \
  --out <run2_dir>/prompts.jsonl --split-json examples/dsa/m3a_split.json \
  --total 60000 --seed 1234 --exclude-sha <run1_dir>/prompts.jsonl
```

**Step 2 — generate trajectories** (`gen_trajectories.py`): serve stock dense MiniCPM3-4B, sample one response
per prompt at **temperature 0.7, top_p 0.9, seed 1234**. **DP=8, TP=1 is optimal** for this 4B model
(TP=8 is comms-bound). Per-domain `max_new_tokens` caps: Math/Code/Multi-lang-Math = **16384**
("cap high, filter after"), everything else = 4096.

```bash
python scripts/dsa/gen_trajectories.py \
  --prompts <run_dir>/prompts.jsonl --out <run_dir>/trajectories.jsonl \
  --log-dir <run_dir>/logs --model openbmb/MiniCPM3-4B --backend vllm \
  --temperature 0.7 --top-p 0.9 --n 1 --seed 1234 \
  --data-parallel-size 8 --tensor-parallel-size 1 --gpu-memory-utilization 0.90
```

Resumable: appends + fsyncs per `--chunk-size` and skips already-done `prompt_sha256` → relaunch with the same
`--out` to resume a killed run. Idle-GPU auto-launcher for the shared box:

```bash
RUN_DIR=<run_dir> setsid nohup scripts/dsa/wait_idle_and_gen.sh &   # gates on BOTH util AND memory.used
```

**Step 3 — trajectories → SFT parquet** (`trajectories_to_sft_parquet.py`): emits the `messages` parquet
`MultiTurnSFTDataset` consumes, dropping rows with `total_tokens < top_k` (they'd run dense → no DSA signal)
and `finish_reason == "length"` runaways:

```bash
python scripts/dsa/trajectories_to_sft_parquet.py \
  --input <run_dir>/trajectories.jsonl --out <run_dir>/sft.parquet --min-total 512
# code-boost merge: --domains Code --min-total 0 on the extra run, then pandas concat + dedup on prompt_sha256
```

Helpers: `make_overfit_subset.py` (tiny fixed subset for overfit runs, `--min-total 1024 --max-total 4096`);
`analyze_lengths.py` (per-domain length distributions + recommended caps + sparse-active fraction).

**Recorded artifacts:**
- `m3a_sft_full.parquet` — 179,565 rows (original corpus, run `m3a_gen_20260714_163317`, 181,686 trajectories).
- `m3a_sft_full_code60k.parquet` — **239,524 rows** (Code doubled to ~50% because code degraded at top_k
  256/128); sparse-active ≥512 rose 46.8%→53.5%. Standalone extra: `m3a_sft_code_extra60k.parquet` (59,959).
- `m3a_sft_val.parquet` — 2,443 held-out rows (self-gen, disjoint).

**Gotchas (Phase-2 data):**
- `transformers 4.57.1` required for the MiniCPM3 tokenizer in *every* tokenizing script (`.devlibs` on PYTHONPATH).
- `top_k=512` threshold: rows shorter than top_k run dense. Only ~34% of self-gen clears ≥512, ~6.5% ≥1024 →
  **Code is the sparsity workhorse** (74% ≥512). MiniCPM3-4B generates 3–10× shorter than the MiniCPM5 gold
  references, so gold responses are discarded (using them would distill MiniCPM5, not preserve MiniCPM3).
- vLLM emits a harmless `FileNotFoundError: ~/.config/vllm` telemetry warning — ignore (or set
  `VLLM_NO_USAGE_STATS=1` / `VLLM_CONFIG_ROOT`).

---

## 2. Checkpoint setup

### Base model
Stock `openbmb/MiniCPM3-4B` from the HF hub, passed as `model.path=openbmb/MiniCPM3-4B`
(`trust_remote_code`). Phase-1 freezes the base, so a Phase-1 checkpoint's base == stock — which is what lets
Phase-2 pull the base from `model.path` and only the indexer from a warm-start file.

### Indexer initialization
The lightning indexer params (`wq_b`, `wk`, `k_norm`, `weights_proj` × 62 layers) are created fresh by
`attach_indexers` (`verl/models/transformers/minicpm_dsa.py`). `LightningIndexer.reset_parameters`
(`dsa_indexer.py`) does a seeded per-fan-in width-scaled normal init (`std = 0.5/√fan_in`), `k_norm` at
identity, `weights_proj` small-but-nonzero (zeroing it would sever gradients). Goal: near-uniform `softmax(I)`
at init (`entropy_frac ≈ 0.995`) for the KL objective. `indexer/entropy_frac ≈ 1.0` at step 0 = healthy.

### Consolidation (sharded FSDP ckpt → world-size-agnostic .pt)
`consolidate_indexer_ckpt.py` reconstructs full tensors from per-rank DTensor shards so a checkpoint can be
warm-loaded on **any** GPU count. Verifies round-trip + finiteness.

```bash
# indexer-only (Phase-1 → Phase-2 warm-start)
python scripts/dsa/consolidate_indexer_ckpt.py \
  --ckpt-dir /…/phase1/checkpoints --out /…/indexer_full_from_phase1.pt --key-substr ".indexer."

# full base+indexer (for serving-dir build / Phase-2 restart)
python scripts/dsa/consolidate_indexer_ckpt.py \
  --ckpt-dir /…/phase2/…/checkpoints/global_step_N --out /…/phase2_full.pt --key-substr ''
```

Indexer dims default to MiniCPM3-4B: `--n-heads 16 --head-dim 64 --rope-head-dim 32 --q-lora-rank 768 --hidden-size 2560`.

### Randomized-indexer negative control (eval sanity)
`randomize_indexer_ckpt.py` overwrites ONLY `*.indexer.*` in a full consolidated file with a fresh untrained
init, base bit-identical — confirms serving actually routes selection through the indexer (trained k128 → 78.5
GSM8K; random → 0.08):

```bash
python scripts/dsa/randomize_indexer_ckpt.py \
  --consolidated /…/consolidated_model_stepN.pt --out /…/model_random_indexer.pt --seed 0
```

### Three ways a checkpoint enters training

| Mechanism | Flag/env | Loads | Optimizer/step | GPU count |
|---|---|---|---|---|
| base only | `model.path=openbmb/MiniCPM3-4B` | base; indexer grafted fresh | fresh | any |
| warm-start | `WARMSTART_PATH` → `dsa_warmstart_path`, `strict=False` pre-FSDP | indexer-only OR full | **fresh / step 0** | **any** |
| native resume | `RESUME_PATH` → `trainer.resume_mode=resume_path` | model+optim+step+RNG+dataloader | **continues** | **locked to saved count** |

**Gotchas (checkpoints):**
- **`attach_indexers` MUST run before the checkpoint load** (param slots must exist); keep native-resume/model
  load **strict** (`strict=False` would silently drop `indexer.*` → an untrained indexer). A warm-start file
  with no `*.indexer.*` keys **fails loudly** (`n_idx > 0` assert).
- **Native resume is world-size-locked** (per-rank shards `*_world_size_{N}_rank_{r}.pt`); to change GPU count
  you must fall back to `WARMSTART_PATH` (weights only, fresh optimizer/step). If you'll scale up, start on the
  larger count.
- `BATCH % NPROC == 0`; native-resume requires `NPROC` == the saved count.
- HF-export merge (`verl.model_merger`) is a known gap for MiniCPM3 (expects a staged `modeling_minicpm.py`);
  only matters for a portable external HF checkpoint.

---

## 3. Training

Entrypoint: `verl.trainer.sft_trainer` via `torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC`.
`engine=fsdp engine.strategy=fsdp2 engine.reshard_after_forward=True engine.use_orig_params=True`.
**`data.pad_mode=no_padding` is mandatory** (see gotchas).

### Phase 1 — indexer dense warm-up (`examples/dsa/run_minicpm3_dsa_phase1.sh`)

Trains only the lightning indexer (base frozen); loss = per-layer dense-warmup KL, no LM CE. Logged as
`train/indexer/*` (kl, topk_recall, topk_overlap, entropy_frac, nan_frac).

```bash
# defaults: NPROC=1 SEQ_LEN=4096 BATCH=8 STEPS=200 TOPK=2048(diag-only) LR=1e-3 cosine
#           WARMUP_RATIO=0.03 MIN_LR_RATIO=0.1 KL_BLOCK=1024 MODEL_DTYPE=fp32
CUDA_VISIBLE_DEVICES=0 NPROC=1 examples/dsa/run_minicpm3_dsa_phase1.sh

# real 32K on 4×H100:
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 SEQ_LEN=32768 MODEL_DTYPE=bf16 \
  KL_CKPT=True KL_BLOCK=256 DIAG_OVERLAP_SAMPLE=256 STEPS=934 BATCH=8 \
  examples/dsa/run_minicpm3_dsa_phase1.sh
```

Key `+model.override_config` (flat `dsa_*` keys, NOT nested):
```
{dsa_enabled: true, dsa_n_heads: 16, dsa_head_dim: 64, dsa_rope_head_dim: 32,
 dsa_top_k: $TOPK, dsa_mode: dense_warmup, dsa_kl_block_size: $KL_BLOCK,
 dsa_fp8: true, dsa_diag_interval: 5, dsa_log_per_layer: true,
 dsa_diag_overlap_sample: $DIAG_OVERLAP_SAMPLE, dsa_kl_checkpoint: $KL_CKPT}
```
`+loss_mode=indexer_kl`; loss normalized by valid (non-pad) query count. `dsa_top_k` is diagnostic-only
(recall@k) in `dense_warmup` — it does not affect gradients. Success target: `val/indexer/topk_recall ≥ 0.9`.

Validation / eval-only: `VAL_FILES=<same-SEQ_LEN val parquet> TEST_FREQ=25 VAL_PREFIX=val`; val-only from a
checkpoint: `VAL_ONLY=1 RESUME_PATH=<ckpt_dir> VAL_FILES=<parquet> VAL_PREFIX=val_ood_L<L>`.

### Phase 2 — sparse stage (`examples/dsa/run_minicpm3_dsa_phase2.sh`)

Fine-tunes base + indexer with top-k sparse attention on self-gen trajectories.
**Loss = LM CE (response-only) + λ · selected-set indexer KL**, decoupled: top-k is non-differentiable (no LM
grad into indexer) and indexer input is detached (no indexer grad into base).

```bash
# defaults: NPROC=8 SEQ_LEN=4096 BATCH=64 STEPS=500 TOPK=512(train==deploy)
#           BASE_LR=7.3e-6 INDEXER_LR=1e-3 (two param groups) cosine WARMUP_RATIO=0.03
#           KL_BLOCK=1024 KL_CKPT=true LAMBDA=1.0 GRAD_CKPT=True MODEL_DTYPE=bf16
#           TRUNCATION=right FP8_UE8M0=false
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
  WARMSTART_PATH=/…/consolidated_indexer_step934.pt \
  TRAIN_FILES=/cb/ml-eng/aarti/dsa/m3a_sft.parquet \
  TOPK=512 FP8_UE8M0=true BATCH=64 STEPS=500 \
  examples/dsa/run_minicpm3_dsa_phase2.sh
```

`+loss_mode=dsa_sparse +indexer_kl_lambda=$LAMBDA`; `optim.lr=$BASE_LR +optim.indexer_lr=$INDEXER_LR`.
override_config adds `dsa_mode: sparse, dsa_kl_checkpoint: $KL_CKPT, dsa_fp8_ue8m0: $FP8_UE8M0,
dsa_warmstart_path: $WARMSTART_PATH`. Metrics: `train/loss`, `loss/lm`, `loss/kl_weighted`.

**End-to-end long pipeline** (`examples/dsa/run_phase2_long_pipeline.sh`) — recommended for full runs; does
consolidate-Phase1-indexer → build SFT parquet → compute `STEPS = floor(rows/BATCH)*EPOCHS` → launch Phase-2
warm-started. Overrides some defaults: `BASE_LR=1e-5` (MiniCPM3 official full-finetune LR), `WARMUP_RATIO=0.1`,
`SAVE_FREQ=190`, `TEST_FREQ=190`, `EPOCHS=1`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 setsid nohup bash examples/dsa/run_phase2_long_pipeline.sh \
  </dev/null >/cb/ml-eng/aarti/dsa/phase2_long_pipeline.out 2>&1 &
# knobs: DOMAINS=full MIN_TOTAL=0 PHASE1_CKPT=… TOPK=512 FP8_UE8M0=true
```

### Overfit & sweep variants
- `run_minicpm3_dsa_phase1_overfit.sh` — 10-doc InfLLM set, EPOCHS=30, LR=8e-3 constant, `DSA_DEBUG_MASTER=1`
  (prints per-step `[dsa-master]` grad_norm / Adam state / max|Δinit|). Learning sanity for the indexer.
- `run_minicpm3_dsa_phase2_overfit.sh` — 16 Code docs, BASE_LR=5e-5, proves sparse forward + selected-set KL
  run without NaN/OOM and both param groups move.
- `run_phase2_invariance_sweep.sh` — asserts step-1 loss is **invariant to micro-batch split / DP degree**
  (base4/dp3/dp2) within `TOL=2e-3` (globally-normalized loss correctness test).
- `run_phase2_topk_sweep.sh` — full ~1-epoch runs at `TOPK=256,128,512` in sequence (waits for GPUs to free).
- `run_m2_probe.sh` — NOT training; the data-gen probe (select → self-gen → analyze lengths).

**Gotchas (training):**
- **`pad_mode=no_padding` REQUIRED** — the FSDP engine's `prepare_model_inputs` asserts it; `pad_mode=right`
  crashes at the *first* train step (`AssertionError: pad_mode right not supported`) — model loads fine so it
  looks healthy until step 1. `no_padding` still gives the model a rectangular `[bsz,T]` + attention_mask.
- **`engine.reshard_after_forward=True` is correct** — do NOT revert to False (old Phase-1 grad-bug workaround
  is obsolete under the Option-B2 indexer-as-own-FSDP2-unit; False just wastes memory).
- **Synthetic random data cannot be distilled** — use the overfit scripts (real docs) + `DSA_DEBUG_MASTER=1`
  to verify learning (grad_norm≠None, Adam state live, max|Δinit|>0). See `docs/dsa_grad_norm_debugging.md`,
  `docs/dsa_fsdp_sharding_notes.md`.
- **`GRAD_CKPT` conflicts with the `_dsa_kl` side-effect** under recompute → keep OFF in Phase-1 (base frozen);
  in Phase-2 use `ACT_OFFLOAD` as the transparent memory lever. Turn `KL_CKPT` on + lower `KL_BLOCK`/
  `DIAG_OVERLAP_SAMPLE` at 32K.
- **FP8 UE8M0 train/serve drift** — default fake-quant uses plain absmax; serving FLASHMLA/DeepGEMM uses a
  UE8M0 power-of-2 scale → ~2% selection drift. `FP8_UE8M0=true` closes it (top-256 overlap 0.9698→1.0000).
  Legacy checkpoints carry the drift; only realized by retraining. Parity: `tests/dsa/test_indexer_fp8_ue8m0_parity.py`.
- 4×H100 32K fits with ~12 GB headroom (peak 66/69 GB/GPU) — no sequence-parallel needed.

---

## 4. Serving (vLLM)

Both paths run vLLM with **system python `/usr/bin/python`** + **`PYTHONPATH=$DEVLIBS`** (transformers 4.57.1).
The eval venv's `python` symlink is broken outside its build container (servers die instantly, 0 GPU mem).

### 4a. Baseline dense MiniCPM3-4B
```bash
# multi-replica launcher (one server per GPU, port = 8000+gpu):
serve_multi_devlibs.sh "0 1 2 3 4 5 6 7"
# per-GPU it runs:
CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$DEVLIBS" setsid nohup /usr/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model .../model/MiniCPM3-4B --served-model-name MiniCPM3-4B --trust-remote-code \
  --port $((8000+g)) --tensor-parallel-size 1 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --dtype bfloat16 </dev/null >"$log" 2>&1 &
# stop all: pkill -f 'vllm.entrypoints.openai.api_server'
```
**ChatML / no-`<s>` finding:** MiniCPM3-4B's chat path is ChatML (`<|im_start|>…<|im_end|>`, eos `<|im_end|>`
id 73440) and does NOT prepend `<s>`. `add_bos_token=True` only affects the plain base-model path — the v1
`<s>` bug is irrelevant on the instruct chat path. vLLM served tokenization == local `apply_chat_template`.

### 4b. Sparse MiniCPM3-DSA
The vLLM plugin lives in-repo at `scripts/dsa/vllm_minicpm3_dsa/` (`model.py`, `attention.py`, `indexer.py`,
`__init__.py`). Importing it: (1) installs a `deep_gemm` meta_path shim forcing the vendored
`vllm.third_party.deep_gemm` (the only copy with `fp8_fp4_mqa_logits`); (2) monkeypatches vLLM's MLA
convertor so `MiniCPM3DSAForCausalLM` routes through the MLA path with padded head size 576; (3) registers
`MiniCPM3DSAForCausalLM` (+ a dense parity ref).

**Build a serving dir** from a full consolidated (base+indexer) checkpoint:
```bash
python scripts/dsa/build_vllm_serving_dir.py \
  --consolidated /…/consolidated_model_stepN.pt \
  --hf-src /…/huggingface/ --out <serving_dir>
# ⚠ THEN MANUALLY add index_topk to <serving_dir>/config.json  (see gotcha)
```

**Serve** (via the entry wrapper that imports the plugin first):
```bash
# single replica: GPU=0 PORT=8000 MAX_LEN=8192 SERVING_DIR=<dir> serve_dsa.sh
# data-parallel:  DP=8 GPUS=0-7 PORT=8010 MAX_LEN=32768 serve_dsa_dp.sh
# key env: PYTHONPATH="_pluginboot:$REPO:$DEVLIBS"  DSA_SPARSE=1  VLLM_NO_USAGE_STATS=1
# launch flags: --tensor-parallel-size 1 --dtype bfloat16 --enforce-eager
```
`serve_dsa_entry.py` does `import scripts.dsa.vllm_minicpm3_dsa` then execs the OpenAI api_server;
`_pluginboot/sitecustomize.py` (first on PYTHONPATH) re-imports the plugin in vLLM's spawned EngineCore
subprocess.

MLA absorption: sparse serving **requires absorbed/latent mode** (fold `kv_b_proj` into q and `o_proj`,
attend over the cached 256-d latent) because the indexer top-k + FlashMLA-sparse kernel operate directly on
the cached latents. Absorbed ≡ dense at fp32 `max|Δ|=5.7e-5`. See `docs/dsa_mla_absorption_explained.md`.

### 4c. How the vLLM sparse serving model was built

vLLM 0.20.2 already ships Day-0 DeepSeek-V3.2 DSA (lightning-indexer FP8 kernels in **DeepGEMM** + sparse-MLA
in **FlashMLA**), but that machinery is bound to `DeepseekV2MLAAttention` and DeepSeek's dims (MoE MLP,
`index_n_heads=64`, `index_head_dim=128`, `top_k=2048`, head_size 576). vLLM's own `MiniCPM3Attention` is
**dense** (materializes full Q/K/V from the MLA latents) and has no sparse variant, and `model_type="minicpm3"`
isn't in vLLM's `is_deepseek_mla` allowlist. So we can't masquerade as DeepSeek-V3.2 (it's MoE) — the model is
a dedicated out-of-tree plugin, `scripts/dsa/vllm_minicpm3_dsa/`, that **reuses** vLLM's `Indexer` /
`sparse_attn_indexer` / `DeepseekV32IndexerCache` / FlashMLA-sparse ops but wires them onto MiniCPM3's MLA at
MiniCPM3 dims (16 indexer heads, head_dim 64, top_k 512; dense MLP + muP scaling). It was built and validated
in staged milestones, each gated against the HF sparse forward (`minicpm_dsa.py::_sparse_attn`) and the free
`top_k ≥ T ⇒ output == dense` oracle:

- **Two make-or-break probes settled the approach before any model code:**
  - **DeepGEMM indexer probe** (`tests/dsa/probe_deepgemm_indexer.py`, `check_vendored_deepgemm.py`): the
    indexer FP8 logits kernel runs at our (padded) dims. Confirmed (i) head_dim **64→128 zero-pad is
    bit-exact** (`max|Δlogit|=0` — padding can't change the absmax so the UE8M0 scale is identical);
    (ii) `index_n_heads=16` is **kernel-rejected** (only 32/64/128 legal via `block_qh % num_heads == 0`) →
    pad heads **16→32 with zero q AND zero weights** (each head contributes `w·ReLU(q·k)`; both 0 → adds 0);
    (iii) the kernel applies **per-head ReLU internally** (do NOT double-apply); (iv) the Hadamard rotation is
    only a ~2% selection effect (not load-bearing).
  - **FlashMLA-padding probe** (`tests/dsa/probe_flashmla_padding.py`): we can **reuse** vLLM's FlashMLA-sparse
    decode kernel by zero-padding MiniCPM3's MLA dims up to DeepSeek's (nope 256→512, rope 32→64 ⇒ head **576**;
    value 256→**512**; heads 40→**64**), BF16 cache, `max_abs_err=5.5e-4`. **This dropped the planned custom
    Triton kernel entirely** — the blocker was solved by padding, not a rewrite.

- **The `index_head_dim=64` clash and its fix.** vLLM's indexer hardcodes `quant_block_size=128` and lays out
  the K-cache as `head_dim + head_dim//quant_block_size * 4` (data + one fp32 scale per 128-block). At
  head_dim 64 that gives `64//128 = 0` scale groups → the FP8 path degenerates. Fix = pad the **activations**
  (not the weights) to 128 in the custom indexer forward, *after* `k_norm`+rope and *before* FP8 quant.
  Padding weights would break `k_norm` (a `LayerNorm(head_dim)` — the 64 zeros shift mean/variance); padding
  activations keeps `k_norm`/rope on the real 64 exactly as trained, and the 128-dot equals the 64-dot since
  the zeros contribute 0 under symmetric-absmax FP8. The trained-with Hadamard is likewise re-applied over the
  **real 64** before the pad (rotating over 128 would mix zeros into reals).

- **Plugin structure** (`__init__.py` does three things at import time, before `import vllm`):
  1. **deep_gemm meta_path shim** — makes top-level `import deep_gemm` raise `ImportError` so vLLM falls back
     to the vendored `vllm.third_party.deep_gemm` (the only copy with `fp8_fp4_mqa_logits`; the external
     `~/.local` deep_gemm shadows it and lacks the symbol). Touches neither disk nor `sys.path`, so
     flashinfer / fast_hadamard_transform / cupy stay importable. Do **not** use `PYTHONNOUSERSITE`.
  2. **force-MLA monkeypatch** — patches `ModelArchConfigConvertorBase.{is_deepseek_mla,get_head_size}` so
     `MiniCPM3DSAForCausalLM` routes through vLLM's MLA path and reports the padded head size 576. Scoped to
     the DSA arch only (stock dense reference untouched).
  3. **register()** — `ModelRegistry.register_model("MiniCPM3DSAForCausalLM", ...)` + a
     `MiniCPM3StockRefForCausalLM` dense parity reference.

- **Class hierarchy** (keeps the MiniCPM muP shells so `scale_emb`/`scale_depth`/logit scaling survive):
  `MiniCPM3DSAForCausalLM(MiniCPMForCausalLM)` → `MiniCPM3DSAModel` (holds the model-level
  `topk_indices_buffer [max_batched_tokens, index_topk] int32`, threaded to every layer) →
  `MiniCPM3DSADecoderLayer` → `MiniCPM3DSAAttention(DeepseekV2MLAAttention)` (reuses is_v32/Indexer/MLA wiring
  but MiniCPM3 dims). Weight-remap items: our **separate** `q_a_proj`+`kv_a_proj_with_mqa` vs DeepSeek's fused;
  our **separate** `wk`+`weights_proj` vs vLLM's fused `wk_weights_proj` (concat in loader, or keep separate
  since we own the module); MiniCPM3 scaling (no yarn mscale); **longrope** (not yarn); **non-interleaved**
  indexer + MLA rope (`indexer_rope_interleave=false` — do NOT inherit DeepSeek's interleaved MLA rope).

- **Staged validation gates:** Stage 0 plugin skeleton (loads 1:1, both KV-cache groups build, indexer
  populates `topk_indices_buffer`, short prompt == base); Stage 1 MLA absorbed-latent form ≡ stock
  materialized-QKV (fp32 `max|Δ|=5.7e-5`); Stage 2 sparse impl `top_k≥T` == dense + prefill logits == HF
  sparse; Stage 3 greedy decode matches HF sparse token-for-token; Stage 5 serve the card suite. Tests:
  `tests/dsa/test_stage{0,1,2a,2b,3}_*.py`.

### 4d. Inference tweaks (what actually matters at serve time)

- **`--enforce-eager` is mandatory** for the sparse path (no validated CUDA graph) — this is the single
  biggest inference knob. It makes decode slow, which is exactly what triggered the LiveCodeBench timeout
  cascade (§5); compensate with **data-parallel replicas** (`serve_dsa_dp.sh --data-parallel-size N`), not
  tensor parallelism.
- **`--tensor-parallel-size 1` only.** The 40→64 head-pad is bit-exact only single-GPU; scale throughput with
  DP replicas (one per GPU, port = base+gpu).
- **`index_topk` in `config.json` is the on/off switch for sparsity** — the plugin does
  `use_sparse = _sparse_enabled() and hasattr(config, "index_topk")`. Absent → silent dense
  `FLASH_ATTN_MLA`. Must be added manually after `build_vllm_serving_dir.py` (which emits only `dsa_*`).
  vLLM reads `index_*` names, so the plugin also aliases `index_n_heads=16`, `index_head_dim=128` (padded),
  `indexer_rope_interleave=false`.
- **`DSA_SPARSE=1`** env gates the sparse route at runtime (belt-and-suspenders with `index_topk`).
- **`top_k` must be a multiple of 128** for the fast FlashMLA-sparse decode kernel (`2·B_TOPK`): 256 = 2×128
  works, 64 is rejected. Unused index slots are set to the **-1 sentinel** (`topk_indices_buffer[:, k:] = -1`).
  `index_topk=256 ∉ {512,1024,2048}` → decode falls to the correct-but-slower `top_k_per_row_decode` (not
  `persistent_topk`) — acceptable.
- **`--max-model-len`:** 32K for knowledge/IF/GSM8K; 8192 for MATH + code benchmarks (score-neutral — a 32K
  re-serve matched 8192 within ≤0.45). Lower it to shrink the KV cache on the shared box.
- **`--gpu-memory-utilization 0.85`, `--dtype bfloat16`.** On the shared host, 0.9 (→71 GB) OOMs against other
  tenants' hidden reservations; 0.85 (~70 GB) is the safe ceiling — size from *actual free* memory if busy.
- **`VLLM_NO_USAGE_STATS=1`** silences the harmless `~/.config/vllm` telemetry `FileNotFoundError`.
- **`_pluginboot/sitecustomize.py` first on `PYTHONPATH`** guarantees the plugin re-registers inside vLLM's
  spawned EngineCore subprocess (which doesn't inherit the parent's imports on the async server path).
- **Sampling for eval generation:** greedy (temp 0) for code (EvalPlus); LiveCodeBench `n=10, temperature 0.2`;
  `--openai_timeout 1200` (the slow eager decode blows the default 90 s on n=10-in-one-call).
- **Confirm sparse is live in the serve log:** `Using FLASHMLA_SPARSE attention backend` +
  `Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend`. If you see
  `FLASH_ATTN_MLA out of potential backends: [...]`, `index_topk` is missing → dense fallback.
- **Faithfulness bound:** sparse ≡ dense exactly iff `T ≤ top_k`; a `top_k ≥ T` (e.g. 2048) serve run is a
  validated dense-equivalent control on short benchmarks. FP8 UE8M0 drift is fixed serve-side (not removable
  at serve time); the shipped `phase2_full_k256_1ep` ckpt still carries the training-side drift.

**Gotchas (serving — build-level; operational knobs are in §4d):**
- **`index_topk` gate — the #1 gotcha** (also §4d): `build_vllm_serving_dir.py` emits only `dsa_*`, NOT
  `index_topk`, so a fresh serving dir **silently serves dense `FLASH_ATTN_MLA`** (indexer runs but its
  selection is unused). Add `index_topk` to `config.json` by hand and verify the log (§4d).
- **deep_gemm `fp8_fp4_mqa_logits` missing** — the external `~/.local` deep_gemm shadows the vendored copy and
  lacks the symbol → vLLM resolves a `_missing()` stub → sparse kernel unavailable. The plugin's meta_path
  shim forces the vendored `vllm.third_party.deep_gemm`. Tier-3-only; dense serving unaffected.
- **Pad activations, not weights.** head_dim 64→128 and heads 16→32 padding is applied to the *activations*
  after `k_norm`/rope; padding weights breaks `k_norm` (`LayerNorm(head_dim)` stats shift). All-zero real
  vectors are the only degenerate case (`amax=0`), pre-existing and epsilon-clamped.
- **Don't double-apply ReLU** — the DeepGEMM kernel applies per-head ReLU internally; the custom forward must
  not. `index_n_heads=16` is kernel-rejected (32/64/128 only) → the 16→32 zero-q/zero-weight pad is required.
- **RoPE convention:** MiniCPM3's MLA rope is **non-interleaved** (llama `rotate_half`) — do NOT inherit
  DeepSeek's interleaved MLA rope; indexer rope is neox-style / non-interleaved (`indexer_rope_interleave=false`).
- **`softmax_scale` differs** at serve (128^-0.5 padded vs 64^-0.5 trained) — a positive global scale, so
  top-k *selection* is invariant, but raw indexer scores shift by √0.5 → in parity checks compare LM logits /
  selected-index sets, never raw indexer scores.

---

## 5. Evals

**Harness = one server, many clients.** Stand up ONE persistent vLLM OpenAI chat endpoint (4B fits on one
H100) and point every tool at it, so chat-template/BOS handling is identical. Per-benchmark best-of-breed
tools (NOT lm-eval loglikelihood — avoids the MiniCPM `<s>`/BOS MC-scoring pitfall):
OpenCompass (`_gen` configs: MMLU, CMMLU, CEval, GSM8K, MATH, IFEval), EvalPlus (HumanEval+, MBPP+),
LiveCodeBench (`lcb_runner`, v3). BFCL deferred (no judge API).

Eval roots (outside repo): baseline dense `/cb/ml-eng/aarti/dsa/evals/minicpm3-4B/`; DSA sparse
`/cb/ml-eng/aarti/dsa/evals/minicpm3-4B-dsa-k256/` (`DSA_SCORECARD.md`). Env: `virtualenv --system-site-packages
-p /usr/bin/python` at `env/venv-mc3eval`, then `pip install --no-deps transformers==4.57.1 huggingface_hub==0.36.2`
into it (materializes the same pins without the PYTHONPATH shim). Per-benchmark drivers in the eval root:
`run_oc_dsa.sh <bench>`, `run_evalplus_dsa.sh <folder> <dataset>`, `run_lcb_dsa.sh`.

```bash
# EvalPlus (chat mode, greedy):
evalplus.evaluate --model MiniCPM3-4B --dataset humaneval \
  --backend openai --base-url http://localhost:8000/v1 --greedy --root outputs/evalplus
# OpenCompass: opencompass <cfg>_gen.py -w outputs/...   (mmlu 5-shot, gsm8k 8-shot CoT, ...)
# LiveCodeBench: --n 10 --temperature 0.2 --release_version release_v3 --multiprocess 8 --evaluate --openai_timeout 1200
```

### Scorecard — Reported / Baseline-dense / DSA tk256 / Δ dense / DSA tk128
| Benchmark | Reported | Baseline | DSA tk256 | Δ | DSA tk128 |
|---|--:|--:|--:|--:|--:|
| MMLU (5-shot) | 67.2 | 66.80 | 67.19 | +0.39 | 67.36 |
| CMMLU (5-shot) | 73.3 | 72.72 | 72.84 | +0.12 | 72.56 |
| CEval (5-shot) | 73.6 | 72.18 | 72.03 | −0.15 | 72.01 |
| GSM8K (8-shot CoT) | 81.1 | 79.98 | 78.77 | −1.21 | 79.38 |
| MATH (0-shot CoT) | 46.6 | 46.58 | 47.34 | +0.76 | 47.44 |
| IFEval (prompt-strict) | 68.4 | 70.79 | 71.72 | +0.93 | 72.27 |
| HumanEval+ (0-shot) | 68.3 | 69.5 | 65.2 | −4.3 | — |
| MBPP+ (0-shot) | 63.2 | 56.3 | 61.9 | ≈reported | — |
| LiveCodeBench v3 | 22.6 | 20.7 | 20.52 (fixed) | −0.2 | — |

**Headline:** sparse matches dense within noise (mean |Δ| ≈ 1.1); tk128 ≈ tk256 (±0.6). Only real cost is
HumanEval+ (−4.3), decomposed as −3.0 Phase-2 BC drift (weights) + −1.3 sparsity — mostly BC drift on code
(this motivated the +60k code-boost dataset in §1b). Model: `phase2_full_k256_1ep` @ `global_step_2805`,
`index_topk=256`.

**Sparsity verified genuine** — GSM8K top_k causal sweep: tk2→0.08, tk128→79.08, tk256→78.77, tk2048→79.76,
dense→79.98. Random-indexer ablation (k128 @ step 2805): trained 78.54 vs random 0.08 → accuracy is caused by
*which* tokens the indexer selects.

### LiveCodeBench debug (RESOLVED — harness artifact, not model regression)
Symptom: LCB v3 pass@1 ~0.8%, ~43% of problems returned empty generations — but the same model scored
HumanEval+ 65.2 / MBPP+ 61.9, so it can code. Two bugs in `lcb_runner`:
1. **Alignment:** on a failed parallel task `run_batch` did `outputs.extend([""]*n)` (n scalar entries)
   instead of appending one list → 9 extra entries per failure shift every later generation → code stapled
   to the wrong problem (positional zip). DSA-only because baseline (fast) had 0 pool failures; DSA's slow
   `enforce_eager` decode produced ~25–31 failures/run.
2. **Empty gens:** `oai_runner` requests all n=10 in one ~79 s call; default `--openai_timeout 90` → timeout →
   retry → `[]` → trips `assert len(result)==args.n` → problem dropped.

Fixes: `extend`→`append`; pass `--openai_timeout 1200`; replace the hard assert with pad-to-n. Fixed re-run
(DP=8 server + MP=8) completed 612/612 with 0 empties → **DSA k256 20.52% vs baseline 20.74% (parity, −0.22)**.
See `docs/dsa_lcb_debug.md`.

**Other eval gotchas:** MATH is prompt-sensitive — use **0-shot CoT** (`math_0shot_gen_393424`), not 4-shot
Minerva (its "I hope it is correct" exemplars suppress CoT → 36.3 vs 46.6). MBPP+ baseline (56.3) was itself
anomalously low (a set-version harness gap); read DSA MBPP+ as ≈ reported. Long-context evals (where sparsity
should separate from dense) not yet run.

---

## 5.5. Verification — did eval & inference do the right thing?

Two independent worries: (a) is the served model *genuinely sparse* (not silently falling back to dense), and
(b) does the vLLM serving stack reproduce the training/HF numerics? We answered both with a ladder of checks,
from unit parity up to end-to-end ablations.

### A. Is it genuinely sparse? (not silently dense)
- **Serve-log check (first line of defense).** Confirm `Using FLASHMLA_SPARSE attention backend` +
  `Setting kv cache block size to 64 for DEEPSEEK_V32_INDEXER backend` appear at startup. If instead
  `FLASH_ATTN_MLA out of potential backends: [...]`, `index_topk` is missing → dense fallback (§4 gotcha #1).
- **top_k causal sweep (GSM8K)** — vary the effective key budget on the *same* checkpoint via `DSA_FORCE_TOPK`
  and watch accuracy move; if the model ignored selection, accuracy would be flat:

  | top_k | 2 | 128 | 256 | 2048 | dense |
  |---|--:|--:|--:|--:|--:|
  | GSM8K | 0.08 | 79.08 | 78.77 | 79.76 | 79.98 |

  tk2 collapses to 0.08 (attention starved) while tk128+ recovers → selection is load-bearing, and tk2048 ≈
  dense confirms the `top_k ≥ T ⇒ dense` oracle end-to-end.
- **Random-indexer ablation** (`scripts/dsa/randomize_indexer_ckpt.py`) — the decisive test. Replace ONLY the
  trained indexer weights with a fresh random init (base bit-identical), rebuild the serving dir, re-eval:

  | k128 checkpoint | GSM8K |
  |---|--:|
  | trained indexer | 78.54 |
  | random indexer @ tk128 | 0.08 |
  | random indexer @ tk2048 (≈dense) | 79.61 |

  Random selection at tk128 destroys accuracy but recovers at tk2048 (where selection is a no-op) → the score
  is **caused by which tokens the indexer selects**, not by anything else in the stack.

### B. Does vLLM inference match training/HF numerics?
Golden reference throughout = the HF sparse forward (`minicpm_dsa.py::_sparse_attn`) + the `top_k ≥ T ⇒ dense`
invariant. Each build stage had its own parity gate (`tests/dsa/test_stage{0,1,2a,2b,3}_*.py`):
- **Indexer kernel parity** (`test_minicpm3_dsa_indexer_parity.py`, GPU) — top-256 selection **overlap** vs
  the real DeepGEMM kernel: 0.9698 mean / 0.9297 min (legacy FP8) → **1.0000 / 0.9961** with UE8M0. Plus the
  CPU unit `test_indexer_fp8_ue8m0_parity.py` (training UE8M0 dequant **bit-identical** to the serve
  `_quant_fp8_rows`; STE gradient intact).
- **Padding probes** — `probe_deepgemm_indexer.py` (head_dim 64→128 pad **bit-exact**, `max|Δlogit|=0`; kernel
  applies per-head ReLU) and `probe_flashmla_padding.py` (MLA dims padded to 576/512/64 exact to
  `max_abs_err=5.5e-4`).
- **Stage 1 — MLA form** — absorbed-latent attention ≡ stock materialized-QKV MiniCPM3 at fp32
  `max|Δ|=5.7e-5` (isolates "MLA expressed correctly" from the sparse kernel).
- **Stage 2 — sparse impl** — (a) `top_k ≥ T` output == Stage-1 dense exactly; (b) prefill LM logits == HF
  sparse forward within tol; (c) selected-index sets match the indexer. Compared LM logits / index sets, NOT
  raw indexer scores (serving `softmax_scale` = 128^-0.5 vs training 64^-0.5 shifts raw scores by √0.5 but is
  selection-invariant).
- **Stage 3 — decode** — greedy generation from the vLLM server matches the HF sparse model **token-for-token**
  on sample prompts; `top_k ≥ T` generation == dense generation.
- **Tokenization parity** — vLLM's served chat tokenization == local `apply_chat_template` (checked via the
  server `/tokenize` endpoint) → confirms the ChatML template + no-`<s>` handling is identical everywhere, so
  the one-server/many-clients eval design is apples-to-apples.

### C. Cross-checking suspicious eval numbers (don't trust a single low score)
- **HumanEval+ de-confound ladder** — the −4.3 drop was decomposed by serving the *same* Phase-2 weights at
  `top_k ≥ ctx` (dense-equivalent) vs sparse, isolating **weight/BC drift (−3.0)** from **sparsity cost
  (−1.3)** — so we knew the loss was mostly Phase-2 behavior-cloning on code, not the sparse mechanism.

  | Config | weights | attention | HumanEval+ |
  |---|---|---|--:|
  | stock dense | stock | full | 69.5 |
  | Phase-2 @ tk2048 | Phase-2 | full (2048 ≥ ctx) | 66.5 |
  | Phase-2 @ tk256 | Phase-2 | sparse | 65.2 |
- **LiveCodeBench** — an anomalously low ~0.8% was traced to a *harness* bug (parallel-task index
  misalignment + n=10-in-one-call timeouts on the slow eager decode), NOT the model: proven by an offline
  re-grade (`regrade_realigned.py`, 19.14% vs baseline 19.84% on answered problems) and a fixed DP=8 re-run
  (612/612, 0 empties → **20.52% vs baseline 20.74%, parity**). See §5 and `docs/dsa_lcb_debug.md`.

**Takeaway:** genuine sparsity is proven by the top_k sweep + random-indexer ablation; numerical faithfulness
by the stage parity ladder against HF; and every surprising eval delta is de-confounded (weights-vs-attention
serve, harness re-grade) before being attributed to DSA.

---

## 6. Cross-cutting gotchas — quick index

| # | Gotcha | Where |
|---|---|---|
| 1 | transformers 4.57.1 in `.devlibs/tf457lib` on PYTHONPATH (system 5.x can't load MiniCPM3) | §0, all |
| 2 | Never reuse InfLLM `token_ids` (CPM-5 vocab); retokenize `text`, strip `<s>`/`</s>` | §1a |
| 3 | `top_k=512` threshold — rows < top_k run dense (drop them); Code is the sparsity workhorse | §1b, §3 |
| 4 | `--exclude-sha` = net-new/additional data mechanism | §1b |
| 5 | `attach_indexers` before load; keep load strict; warm-start = any GPU count, resume = locked | §2 |
| 6 | `data.pad_mode=no_padding` REQUIRED (right crashes at step 1) | §3 |
| 7 | `FP8_UE8M0=true` closes ~2% train/serve selection drift (retrain to realize) | §3, §4 |
| 8 | `index_topk` must be manually added to serving `config.json` or it silently serves dense | §4 |
| 9 | `enforce_eager` + TP=1 for the sparse path; DP replicas for throughput | §4 |
| 10 | Shared-host GPUs: size util from free mem / gate on idle (`wait_idle_and_gen.sh`) | §0, §1b |
| 11 | Durable runs: `setsid nohup … </dev/null >LOG 2>&1 &`; log full invocation + env | §0 |
| 12 | Evals: MATH 0-shot CoT (not 4-shot); LCB harness alignment+timeout fixes | §5 |

---

## Appendix — file map

| Area | Files |
|---|---|
| Phase-1 data | `examples/dsa/build_phase_a_datasets.sh`, `prepare_real_data.py`, `prepare_ood_data.py`, `prepare_smoke_data.py`, `_dsa_data_utils.py`, `README_datasets.md` |
| Phase-2 data | `scripts/dsa/select_prompts.py`, `gen_trajectories.py`, `trajectories_to_sft_parquet.py`, `make_overfit_subset.py`, `analyze_lengths.py`, `wait_idle_and_gen.sh`, `examples/dsa/m3a_split.json` |
| Checkpoints | `scripts/dsa/consolidate_indexer_ckpt.py`, `randomize_indexer_ckpt.py`; docs `dsa_checkpoint_notes.md`, `dsa_ckpt_loading.md`, `dsa_indexer_init_proposal.md` |
| Training | `examples/dsa/run_minicpm3_dsa_phase1.sh`, `run_minicpm3_dsa_phase2.sh`, `run_phase2_long_pipeline.sh`, `run_*_overfit.sh`, `run_phase2_{invariance,topk}_sweep.sh`, `run_m2_probe.sh`; `verl/models/transformers/{minicpm_dsa.py,dsa_indexer.py}`, `verl/workers/utils/losses.py`, `verl/trainer/sft_trainer.py`; docs `dsa_phase2_plan.md`, `dsa_phase2_implementation.md`, `dsa_train_indexer_plan.md`, `dsa_fsdp_sharding_notes.md`, `dsa_grad_norm_debugging.md` |
| Serving | `scripts/dsa/build_vllm_serving_dir.py`, `scripts/dsa/vllm_minicpm3_dsa/{model,attention,indexer,__init__}.py`; docs `dsa_vllm_serving.md`, `dsa_vllm_minicpm3dsa_build_plan.md`, `dsa_mla_absorption_explained.md` |
| Evals | docs `dsa_eval_report.md`, `minicpm3_eval_plan.md`, `dsa_lcb_debug.md`; eval roots under `/cb/ml-eng/aarti/dsa/evals/` |
| Tests/probes | `tests/dsa/test_indexer_fp8_ue8m0_parity.py`, `test_minicpm3_dsa_indexer_parity.py`, `test_stage{0,1,2a,2b,3}_*.py`, `probe_flashmla_padding.py`, `probe_deepgemm_indexer.py`, `check_vendored_deepgemm.py` |
