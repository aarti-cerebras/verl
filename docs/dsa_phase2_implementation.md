# DSA Phase-2 — Implementation, Runbook & Status

**Self-contained handoff doc.** Companion to `docs/dsa_phase2_plan.md` (full design rationale). This one
captures *operational* state: what's built, how to run it, the environment/gotchas, the current data, the
decisions locked in the 2026-07-13/14 sessions, and what remains. Read this first to continue the work from
a fresh session.

---

## 0. TL;DR — where things stand (2026-07-14)

- **Goal:** convert released instruct **MiniCPM3-4B** → DeepSeek Sparse Attention (DSA, `top_k=512`) while
  **preserving capabilities**, via **behavior-cloning self-distillation** (SFT on the model's own dense
  trajectories) + a decoupled selected-set indexer KL. No teacher, no `dsa_distill_loss`.
- **BUILT & DONE:** the whole **data-generation pipeline** (prompt selection → self-gen → analysis) and a
  **finished M3a trajectory corpus** (181,686 trajectories at `/cb/ml-eng/aarti/dsa/m3a_gen_20260714_163317/`).
- **NOT built (the blocker):** **sparse *training*** — `mode=="sparse"` in `minicpm_dsa.py` is still a
  `NotImplementedError` stub. Tasks **T1–T4** below are the remaining core engineering.
- **Next experiment:** implement T1–T4, then **train sparse (top_k=512) on math, eval on benchmarks vs dense**
  as the go/no-go method check.

---

## 1. Environment (CRITICAL — read before running anything)

- **We run INSIDE the verl `vllm020.dev1` container** on `ml-eng-gpu-11` / `-gpu-12` (SLURM), **8× H100 80GB**.
  No `docker` CLI inside (no docker-in-docker). `nvidia-smi` works.
- **Versions (compatible):** `torch 2.11.0+cu130`, **`vllm 0.20.2`**, **`transformers 4.57.1`** (staged in-repo
  at `.devlibs/tf457lib`, must be on `PYTHONPATH`), `huggingface_hub 0.36.2`. vLLM 0.20.2 + transformers
  4.57.1 + MiniCPM3 coexist — keep `PYTHONPATH=$REPO/.devlibs/tf457lib`.
- **GPUs are SHARED** across tenants (other jobs in other PID namespaces → `nvidia-smi` shows memory used but
  "No running processes found"). Consequence: `gpu_memory_utilization=0.9` **OOMs** if others hold memory
  ("Free memory 37 GiB < desired 71 GiB"). Mitigation: run when GPUs are actually free, or size util from
  free memory.
- **Data root: `/cb/ml-eng/aarti/dsa`** (writable NFS). Gated-dataset shards cache in `.../hf_cache`.
- **HF token** (for gated `openbmb/UltraData-SFT-2605`) is at `~/.cache/huggingface/token`. Works in-container.
- **vLLM telemetry-dir gotcha:** vLLM tries `~/.config/vllm` and errors `FileNotFoundError` if missing (it's
  **non-fatal** — telemetry only — but noisy). Fix: `VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
  VLLM_CONFIG_ROOT=/cb/ml-eng/aarti/dsa/.vllm/config VLLM_CACHE_ROOT=/cb/ml-eng/aarti/dsa/.vllm/cache`.
- **Durable long jobs:** harness/`run_in_background` dies on terminal loss. Use
  `setsid nohup <cmd> </dev/null >LOG 2>&1 &` → detached, survives terminal loss (not container/allocation
  death). Poll the logfile (no completion ping). Kill by **PID / process-group** (`kill -- -<pgid>`), never
  `pkill -f <pattern>` (your own shell's argv matches the pattern → self-kill).

---

## 2. Decisions locked (this session)

| Decision | Value | Why |
|---|---|---|
| **`top_k`** | **512** (train + deploy) | DeepSeek's 2048 was for 128K (1.6%); 512/32K = 1.6% matches their ratio |
| Objective | **self-gen behavior cloning** | preserve *this* model (no corpus; raw web LM would forget instruct) |
| Loss | `sft_loss(response) + λ·indexer_KL(selected set)`, **decoupled** (indexer input detached) | DeepSeek §2.1.1; no teacher |
| Prompt source | **`openbmb/UltraData-SFT-2605/no_think`** (single, gated) | on-distribution (MiniCPM family), EN+ZH |
| Split (config×lang) | Code(EN) 30 / Math(EN) 30 / ML-Math(ZH) 10 / ML-Knowledge(ZH) 10 / Knowledge(EN) 10 / Chinese-general(ZH) 5 / IF(EN) 5 → ~75% EN / 25% ZH | Code = sparsity workhorse; ZH from Multi-lang-* not tiny Chinese-general |
| `max_new_tokens` | **Math/Code/Multi-lang-Math = 16K**, others 4K | "cap high, filter after"; drop runaways post-hoc |
| Decoding | **temperature 0.7**, top_p 0.9, n=1, seed 1234 | clones deployment-temp behavior |
| Dataset class | **`MultiTurnSFTDataset`** (NOT `PackedPretrainDataset`), `pad_mode=right`, `use_remove_padding=False` | SFT path; DSA forward expects `[bsz,T]` |
| Generation parallelism | **DP=8, TP=1** (data-parallel replicas) | TP=8 on a 4B model is comms-bound |
| Writing | **chunked append + resume** (`--chunk-size 512`) | crash-safe on the flaky shared host |

---

## 3. Data-generation pipeline (BUILT)

### Scripts (`scripts/dsa/`, `examples/dsa/`)
- **`_dsa_log.py`** — shared logger → stdout + timestamped file; records exact argv/cwd/host/git/env.
- **`select_prompts.py`** — subset `UltraData-SFT-2605/no_think` → per-config counts + per-config language
  filter (`--split-json` + `--total`, or uniform `--domains`/`--per-domain`). Adaptive shard scan
  (`--max-shards-per-config`). Emits `messages` JSONL + `source_uid` + `prompt_sha256` (dedup key + back-ref).
- **`gen_trajectories.py`** — self-gen from dense MiniCPM3-4B. Backends `vllm`(default)/`hf`. **Data-parallel**
  (`--data-parallel-size`, spawns 1-GPU replicas over `prompts[rank::world]` shards, merges parts). **Node
  sharding** (`--num-nodes`/`--node-rank`) for multi-node disjoint splits. **Chunked/resumable** append writes
  (`--chunk-size`, fsync per chunk; relaunch skips done `prompt_sha256`). Per-domain `max_new_tokens`
  (`DEFAULT_CAPS`). Each row: messages + `prompt_tokens/resp_tokens/total_tokens/finish_reason` + back-refs.
- **`analyze_lengths.py`** — per-domain input/response/total distributions + DSA bucket yields + recommended
  caps → JSON report.
- **`examples/dsa/run_m2_probe.sh`** — orchestrates select→gen→analyze; master log (exact cmd/env/nvidia-smi/
  versions). Env knobs: `SPLIT_JSON TOTAL DP TP GPU_MEM_UTIL MAX_SHARDS DATA_ROOT RUN_DIR ...`.
- **`examples/dsa/m3a_split.json`** — the locked split spec (frac + lang per config).
- **`scripts/dsa/wait_idle_and_gen.sh`** — detached watcher: waits until ≥N GPUs are *truly free*
  (util≤0 AND mem≤threshold) for a window, then launches gen sized to free memory. (Built for the shared-host
  contention; use if you must wait for a busy box to clear.)

### Runbook — full M3a select+gen+analyze (single node, all 8 GPUs free)
```bash
cd <repo>
RUN_DIR=/cb/ml-eng/aarti/dsa/m3a_gen_$(date -u +%Y%m%d_%H%M%S) \
SPLIT_JSON=$(pwd)/examples/dsa/m3a_split.json TOTAL=200000 DP=8 TP=1 GPU_MEM_UTIL=0.9 MAX_SHARDS=40 \
setsid nohup bash examples/dsa/run_m2_probe.sh </dev/null >/cb/ml-eng/aarti/dsa/m3a_fresh_nohup.out 2>&1 &
```
### Runbook — gen only (select already done), chunked/resumable, telemetry-safe
```bash
D=/cb/ml-eng/aarti/dsa/m3a_gen_<ts>; REPO=$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$REPO/.devlibs/tf457lib \
VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CONFIG_ROOT=/cb/ml-eng/aarti/dsa/.vllm/config VLLM_CACHE_ROOT=/cb/ml-eng/aarti/dsa/.vllm/cache \
setsid nohup bash -c "cd $REPO && \
python3 scripts/dsa/gen_trajectories.py --prompts $D/prompts.jsonl --out $D/trajectories.jsonl --log-dir $D/logs \
  --backend vllm --temperature 0.7 --top-p 0.9 --seed 1234 --data-parallel-size 8 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 --chunk-size 512 && \
python3 scripts/dsa/analyze_lengths.py --trajectories $D/trajectories.jsonl --out-report $D/length_report.json --log-dir $D/logs" \
  </dev/null >/cb/ml-eng/aarti/dsa/m3a_gen_run.out 2>&1 &
```

### Behavior notes / gotchas
- **Part files vs merged:** each replica writes `trajectories.jsonl.part{N}`; launcher merges → `trajectories.jsonl`
  at the end. With chunked writing, parts grow every `chunk_size` prompts (fsync'd, resumable).
- **Slow tail:** the **16K cap on Math/Multi-lang-Math** lets degenerate ZH-math **runaways** decode to 16,384
  tokens → chunks in that region can take ~40 min. Those are `finish_reason==length` and get **dropped** in
  post-processing anyway. For future runs, a lower math cap (~2K; probe shows genuine math p95≈1K) or larger
  `--chunk-size` cuts wall-clock a lot.
- **Multi-node disjoint split** (two 8-GPU nodes, no duplicate data): run on each node with
  `--num-nodes 2 --node-rank {0,1}` (each node's DP replicas take `global_rank::global_world`); merge parts after.

---

## 4. Current data: the finished M3a corpus

**`/cb/ml-eng/aarti/dsa/m3a_gen_20260714_163317/`** — `prompts.jsonl` (181,686) + `trajectories.jsonl`
(**181,686 rows**, 471 MB) + `length_report.json` + `logs/`.

- **finish_reason:** `stop` 179,565 (98.8%) · **`length` 2,121 (1.2%) → DROP as runaways** in post-filter.
- **Length (MiniCPM3-4B @ T=0.7), total p50=491 / p90=1049 / p99=16445; resp p50=332 / p90=720.**
- **DSA yield: total ≥512 = 47% (~85k), ≥1024 = 11% (~20k), ≥2048 = 1.8%.**
- Per-domain (total p50 / ≥512 / ≥1024): **Code 672 / 73% / 18%** (workhorse), Multi-lang-Math 527 / 52% / 16%,
  Math 453 / 42% / 7%, Chinese-general 391 / 22% / 2%, Knowledge 286 / 19% / 3%, IF 202 / 10% / 1%,
  **Multi-lang-Knowledge 75 / 0% / 0% (only 1,686 rows — ⚠ ZH-knowledge config is nearly empty of Chinese)**.
- **Realized languages: ~150k EN / ~31.7k ZH (~17% ZH, below the 25% target)** — because `Multi-lang-Knowledge`
  had almost no Chinese (kept 1,686/20,000). Backfill ZH later (more `Chinese-general`, other ZH sources).

**M2-probe finding that shaped this:** MiniCPM3-4B generates **3–10× shorter** than the MiniCPM5 reference in
`UltraData-SFT-2605`; **Code is the sparsity workhorse, not Math** (Math median is short; its long tail is
runaways). So caps were kept high but runaways are filtered, and Code is weighted up.

---

## 5. Remaining engineering — sparse TRAINING (NOT built; the blocker)

`minicpm_dsa.py:~476` still raises `NotImplementedError` for `mode=="sparse"`. Needed (see plan doc T1–T4):
- **T1 — sparse attention forward** (recommended **tiled gathered-KV**: top-`k` select per query → gather K/V →
  attention over the selected set, tiled over query blocks; autograd-native, no custom kernel). Parity test:
  `k≥T` ≡ dense flash.
- **T2 — selected-set indexer KL** (restrict `_dense_warmup_kl` to `S_t`; renormalize `p` over selected keys;
  **detach indexer input** so its grad doesn't reach the base).
- **T3 — `dsa_sparse_loss`** in `verl/workers/utils/losses.py` = `sft_loss + λ·model._dsa_indexer_kl`; add
  `loss_mode=="dsa_sparse"` in `sft_trainer.py`.
- **T4 — unfreeze base + two param groups** (base LR ~7e-6, indexer ~1e-3); sparse mode already skips the
  Phase-1 freeze in `monkey_patch.py`.
- **Post-gen data prep** (before training): filter `total_tokens ≥ 512` (prefer ≥1024) + drop
  `finish_reason==length` (the 2,121 runaways) + dedup; the kept `messages` feed `MultiTurnSFTDataset`
  (response-only loss mask, `pad_mode=right`).
- **Config + launch** (`sft_trainer_minicpm_dsa_phase2.yaml`, run script), from the Phase-1 indexer checkpoint.
- **Tests** (`test_minicpm_dsa_sparse.py`): parity, grad-to-both-base+indexer, detach checks, selected-set KL.

Memory: at `top_k=512` with these short (~1–2K) trajectories, sparse full-base training **fits on 4×/8×H100
without SP**; the 32K memory/Ulysses-SP work is an optional later "genuine-32K" tier.

---

## 6. Validation experiment (math-only go/no-go)

Once T1–T4 exist: **train sparse (`top_k=512`) on the math trajectories → eval vs dense baseline.**
- Benchmarks **exceed 512 tokens on average → sparse selection IS active during eval** (mild sparsity at
  ~512–1.5K; not dense-equivalent). So GSM8K/MATH genuinely test the sparse path + capability preservation.
- **Eval broad** (math-only *training* can forget others): GSM8K/MATH **+** HumanEval/MBPP + MMLU/C-Eval +
  a chat/IF probe, **all vs the dense baseline** (`/cb/ml-eng/aarti/dsa/evals/minicpm3-4B`,
  `docs/minicpm3_eval_plan.md`). Optional long-context (RULER/LongBench) for the extreme-sparsity regime.
- **Pass:** sparse within ~1% of dense on math AND non-math not collapsed → method works → expand to balanced.

---

## 7. Inference / serving (reuse DeepSeek-V3.2 DSA)

- **DSA is MLA-native and MiniCPM3 is MLA**, so vLLM's / SGLang's **DeepSeek-V3.2 DSA serving kernels**
  (lightning-indexer top-k select + **fused sparse-MLA attention** + FP8 indexer + paged-KV) are the right
  thing to reuse — they're the hard parts and they exist.
- **Not plug-and-play:** stock vLLM `MiniCPM3ForCausalLM` is **dense**. Need a MiniCPM3-DSA model class that
  grafts our indexer into MiniCPM3 MLA, routes attention through the V3.2 sparse backend, and loads our
  trained indexer weights.
- **Risks:** dim mismatch (V3.2 kernels assume head_dim 128 / top_k 2048 / specific FP8 blocks; MiniCPM3 is
  head_dim 64 / rope 32 / our indexer 16 heads,head_dim 64,top_k 512 → may need kernel params/padding); verify
  which vLLM version first ships the V3.2 DSA backend (0.20.2 may predate it — SGLang is an alternative).
- **For now:** at moderate context, **serve dense** (sparsity is mild there) — the DSA kernel is only needed
  for **long-context efficiency**. So the validation experiment can be served/evaluated on dense vLLM.

---

## 8. Suggested next-step ordering

1. **Post-filter** the finished corpus → training parquet (drop 2,121 runaways + `<512`, dedup). *(quick)*
2. **T1 + T2** (sparse forward + selected-set KL) with parity/detach tests — **the blocker**.
3. **T3 + T4** (loss + unfreeze) → `loss_mode=dsa_sparse`.
4. **Math-only sparse train** from Phase-1 indexer ckpt → **eval vs dense** (§6).
5. If pass → balanced run (+ backfill ZH); if long-context needed → genuine-32K tier (Ulysses-SP-for-DSA) +
   vLLM DSA inference integration (§7).

Full design rationale, milestones (M0–M4), and the memory/SP analysis: **`docs/dsa_phase2_plan.md`**.
