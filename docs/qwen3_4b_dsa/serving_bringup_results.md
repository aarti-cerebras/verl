# Qwen3-4B DSA on vLLM — bring-up results (2026-08-20)

Execution log for [serving_eval_plan.md](serving_eval_plan.md) phases **P0–P4**, all on one H100
(SM90), vLLM 0.26.0 (`.devlibs/vllm026`), checkpoint
`p2_qwen3-4b-thinking-2507_mix5050_L32k_bs8_k2048_16x64_lam1.0_lr5e-6_ilr1e-4_st5000/global_step_1200`.

**Bottom line: the token-granular DSA Qwen3-4B serves on vLLM, computes the trained function, and is
1.60x faster than dense in the regime evaluation actually runs in.** No kernel was written. The new
code is one attention backend, one indexer module, one model shell, one exporter — ~1,300 lines of
plumbing, all listed in §6.

---

## 1. What was built

| file | lines | role |
|---|--:|---|
| `scripts/dsa/vllm_qwen3_dsa/sparse_attention.py` | 358 | non-MLA token-sparse backend: FA3 varlen over a page-size-1 view |
| `scripts/dsa/vllm_qwen3_dsa/indexer.py` | ~390 | serving indexer; drives vLLM's fp8 paged logits + paged top-k |
| `scripts/dsa/vllm_qwen3_dsa/model.py` | ~400 | `Qwen3DSAForCausalLM`, buffer allocation, config gates, controls |
| `scripts/dsa/build_qwen3_dsa_serving_dir.py` | ~200 | FSDP2 DTensor shards -> serving dir (+ `index_topk`) |
| `scripts/dsa/serving/serve_qwen3_dsa.sh` + `_pluginboot` | ~120 | durable OpenAI server |
| `tests/dsa/probe_fa3_sparse_gqa.py` | ~300 | P0 kernel spike |
| `tests/dsa/test_qwen3_dsa_serving_indexer_parity.py` | ~190 | P1 train/serve indexer parity (8 tests) |
| `tests/dsa/qwen3_dsa_offline_smoke.py`, `run_qwen3_dsa_ladder.sh`, `compare_ladder.py` | ~330 | P2/P3 ladder |
| `tests/dsa/bench_qwen3_dsa_throughput.py` | ~110 | P4 |

Reused verbatim from vLLM: `SparseAttnIndexer`, `DeepseekV32IndexerCache`, the CuTeDSL radix top-k,
`triton_convert_req_index_to_global_index`, `reshape_and_cache_flash`, FA3 itself, and the entire
engine (KV cache allocation, scheduling, cudagraphs, OpenAI server).

---

## 2. P0 — the FA3 page-size-1 GQA spike

See [serving_eval_plan.md](serving_eval_plan.md) §4 "P0 RESULT" for the full tables. Summary:
correctness PASS on six shapes (<=3.2e-3 rel vs gather+SDPA); decode **flat in sequence length** and
up to **13x** faster than dense paged decode at 32 reqs x 32K; prefill **3.5x slower** than dense at
32K and sorting the block table does not help.

One extra measurement made while writing the backend: FA3 accepts a **strided** page-1 view
(`torch.as_strided`) of vLLM's *packed* `(num_blocks, block_size, H_kv, 2*head_size)` cache, matching
a contiguous-copy control to 3.18e-3. That is what let the backend keep vLLM's standard KV layout
instead of defining a bespoke one.

## 3. P1 — indexer parity (engine-free, 8/8 pass)

`tests/dsa/test_qwen3_dsa_serving_indexer_parity.py`: identical parameter names (so the export needs
no mapper), projections agree to <5e-3, **selected sets agree >99%** at three (T, top_k) settings
including `top_k > T`, the `16x64 -> 32x128` pad leaves both the dot and the UE8M0 row scale
unchanged, padded heads carry zero gate, and top-k is bit-deterministic across repeated runs.

## 4. P2/P3 — the de-confounding ladder (token-exact)

`bash tests/dsa/run_qwen3_dsa_ladder.sh` then `compare_ladder.py`. Greedy, 32 output tokens, two
prompts (one arithmetic, one needle over 1000 near-identical "Fact i" lines so that no selector can
succeed by luck).

| row | config | vs dense (arith / needle) | needle found |
|---|---|---|---|
| A_dense | `DSA_SPARSE=0`, T=9164 | reference | yes |
| **B2_topk_ge_T** | top_k=4096 >= T=1690 | **EXACT / EXACT** | yes |
| **F2_rand_ge_T** | **random** indexer, top_k >= T | **EXACT / EXACT** | yes |
| C_topk2048 | top_k=2048, T=9164 (22%) | 31/32 (div@12) / **EXACT** | yes |
| D_topk256 | top_k=256, T=9164 (2.8%) | 31/32 / 1/32 | **no** |
| E_rand256 | random indexer, top_k=256 | 1/32 / 1/32 | **no** (gibberish) |

Read the ladder as a whole:
- **B2 exact** ⇒ the sparse code path is correct: when selection is a no-op it reproduces dense
  bit-for-bit. This is the faithfulness control, and it is the single most valuable test here.
- **F2 exact** ⇒ the machinery is sound independently of *what* is selected.
- **E collapses while F2 recovers** ⇒ the collapse is caused by **selection**, not by broken
  plumbing. Together these are rows 4 and 5 of [eval_plan.md](eval_plan.md) §3, and they are what
  rule out the "silently dense" failure that has bitten this project before.
- **D degrades where C does not** ⇒ selection is live, and the *trained* indexer is what preserves
  the model: at 2.8% density the trained selector still holds the non-retrieval task (31/32) while
  the random one destroys everything (1/32).

Through the OpenAI endpoint: a **21,697-token** needle prompt at `top_k=2048` (9.4% density) is
answered correctly, with a full thinking trace.

## 5. P4 — throughput (sparse vs dense, one GPU each, cudagraphs on)

Both servers are the same weights and the same script; only `DSA_SPARSE` differs.

| prompt | output | concurrency | sparse / dense |
|--:|--:|--:|--:|
| 8.5K | 512 | 16 | 0.64x |
| 8.5K | 2048 | 16 | 1.02x |
| 21K | 1024 | 8 | 0.87x |
| **21K** | **4096** | **16** | **1.60x** |

Exactly the shape P0 predicted: the ratio tracks the decode:prefill ratio and the context length,
because sparse decode is flat in `L` while sparse prefill pays a fixed penalty. Thinking traces are
long, so the bottom row is the regime the eval ladder runs in.

**Cudagraphs are worth 2.4x** (0.26x -> 0.64x at the 8.5K/512 point) and were verified, not assumed:
the `top_k >= T` control reproduces the eager dense output **token-exactly** under both
`FULL_AND_PIECEWISE` and `PIECEWISE`. `serve_qwen3_dsa.sh` therefore defaults to graphs on, with
`EAGER=1` as a bisection escape hatch.

One behavioural observation, not yet a result: on the 21K needle the sparse model spent 2,017 output
tokens to dense's 171. If that trace inflation is systematic it partly offsets the throughput win and
matters more than either number — it is [eval_plan.md](eval_plan.md) §7 gate 9, and it needs a real
sample rather than one synthetic prompt.

---

## 6. Five failure modes found and fixed during bring-up

Recorded because every one of them fails *silently or misleadingly*, which is the whole reason this
project runs a ladder instead of a benchmark.

1. **`fast_hadamard_transform` was missing from the serving venv.** The indexer's Hadamard has a
   pure-torch fallback, so nothing errored. But the rotation feeds fp8 quantization, and the two
   implementations differ by ~6.7e-3 in bf16 — enough that **the fp8 bytes and even the UE8M0 row
   scales differ**, perturbing every near-tie in the top-k. The training runs used `/usr/bin/python3`,
   whose user site-packages carries the CUDA kernel (confirmed from the launch manifests), so the
   kernel is the contract. Fixed by symlinking it into the venv **and** by making a missing package a
   hard error (`DSA_ALLOW_TORCH_HADAMARD=1` to override deliberately). Fourth instance of this
   train/serve-drift class after `fp8_ue8m0`, the missing `index_topk` gate, and top-k tie
   determinism.
2. **`import` inside a Dynamo-traced region** (`from fast_hadamard_transform import ...` inside the
   function) is an unconditional graph break: engine startup died with
   `torch._dynamo.exc.Unsupported: Import failure` whenever cudagraphs were on. `@torch._dynamo.disable`
   is *not* the fix either — vLLM's piecewise region rejects it with `Skip calling
   torch.compiler.disable()'d function`. The fix is a real `torch.library.custom_op`, which stays in
   the graph and is still captured. The same latent bug existed in the fp8 quant helper.
3. **vLLM 0.26.0's compiled top-k caps `top_k` at ~4096.** At 8192 it dies with an async
   `CUDA error: invalid argument` whose traceback points at the *preceding* DeepGEMM
   `fp8_fp4_mqa_logits` launch, so it reads as a logits-kernel bug. Irrelevant to the real config
   (2048) but it caps the dense-equivalence control to T <= 4096. Now asserted at construction.
4. **A failed EngineCore survives its parent and keeps ~73 GB of GPU memory**, so the next launch
   fails with "Free memory ... less than desired GPU memory utilization" — which reads as a config
   error, not a leak. `pgrep -af serve_qwen3_dsa_entry` and kill the orphan.
5. **A truncation confound faked a correctness regression.** The 21K needle "failed" under cudagraphs
   purely because that comparison used `max_tokens=1024` while the passing eager run used 2048; the
   sparse model's longer trace hit the cap. At matched budget both retrieve. This is precisely
   [eval_plan.md](eval_plan.md) §5.4 — *any comparison where rows differ in truncation rate is
   invalid* — and it caught the author of this document, so the ladder now records
   `finish_reason` and completion length everywhere.

Also worth noting: `pkill -f serve_qwen3_dsa_entry.py` matches the shell running it and kills your
own session. Use a bracketed pattern.

---

## 7. What is NOT yet done

- **Train/serve selection overlap on real hidden states** (plan §4 P3.3). P1 proves the two modules
  agree given the same input; it does not yet measure agreement inside a running engine, which needs
  hidden states pulled out of EngineCore. The ladder's token-exact rows are strong indirect evidence.
- **Decode parity over a long generation** vs a teacher-forced reference (plan §4 P3.4).
- **The benchmark ladder itself** (P5) and the DSA-vs-MSA equal-KV-budget comparison (P6). The
  harness is ready: `scripts/msa/setup_eval_root.sh` clones an eval root, and the server is served
  under the baseline's model name so the existing OpenCompass configs run unchanged.
- **Trace-length inflation** (§5) needs a proper sample.

## 8. Reproduce

```bash
# 1. export the checkpoint (once)
.devlibs/vllm026/bin/python scripts/dsa/build_qwen3_dsa_serving_dir.py \
  --ckpt-dir /cb/ml-eng/aarti/dsa_qwen3/sparse/_ckpt/<run>/global_step_1200 \
  --base-model /cb/ml-eng/aarti/models/qwen3_4b_thinking_2507 \
  --out /cb/ml-eng/aarti/dsa_qwen3/serving/<tag> \
  --dsa-from-cli --top-k 2048 --n-heads 16 --head-dim 64

# 2. gates
CUDA_VISIBLE_DEVICES=0 .devlibs/vllm026/bin/python tests/dsa/probe_fa3_sparse_gqa.py
CUDA_VISIBLE_DEVICES=0 .devlibs/vllm026/bin/python -m pytest tests/dsa/test_qwen3_dsa_serving_indexer_parity.py -q
bash tests/dsa/run_qwen3_dsa_ladder.sh && .devlibs/vllm026/bin/python tests/dsa/compare_ladder.py /tmp/dsa_ladder

# 3. serve (sparse on GPU 1, dense reference on GPU 2) and benchmark
GPU=1 PORT=8001 bash scripts/dsa/serving/serve_qwen3_dsa.sh
GPU=2 PORT=8002 DSA_SPARSE=0 bash scripts/dsa/serving/serve_qwen3_dsa.sh
.devlibs/vllm026/bin/python tests/dsa/bench_qwen3_dsa_throughput.py --concurrency 16 --prompt-facts 1000 --max-tokens 4096
```
