# MiniCPM3-4B Baseline Eval Plan

**Goal:** reproduce the baseline benchmark scorecard for `openbmb/MiniCPM3-4B`
(https://huggingface.co/openbmb/MiniCPM3-4B) so we have a trustworthy reference point
for the DSA-modified model. This is the *baseline* (unmodified HF) model.

**Decisions (2026-07-14):**
- **Scope:** automatable core first — the benchmarks that need *no* LLM judge.
  Deferred (need a GPT-4-class judge, no API budgeted yet): MT-Bench, AlignBench v1.1,
  FollowBench-zh.
- **Parity strategy:** modern, best-maintained tool per benchmark (OpenCompass /
  EvalPlus / LiveCodeBench / Gorilla BFCL). Accept small deltas vs. reported; document
  methodology and the delta for each. We are *not* resurrecting the (older, less
  maintained) UltraEval configs unless a delta is large enough to warrant a parity dig.
- **Judge API:** none assumed.

## Reported scorecard (targets)

| Category | Benchmark | Reported | Few-shot / mode | Harness (this plan) |
|---|---|--:|---|---|
| EN knowledge | MMLU | 67.2 | 5-shot, gen | OpenCompass |
| EN reasoning | BBH | 70.2 | 3-shot CoT, gen | OpenCompass |
| EN instr-follow | IFEval (prompt strict-acc) | 68.4 | 0-shot, gen | OpenCompass (or lm-eval) |
| ZH knowledge | CMMLU | 73.3 | 5-shot, gen | OpenCompass |
| ZH knowledge | CEval | 73.6 | 5-shot, gen | OpenCompass |
| Math | GSM8K | 81.1 | 8-shot CoT, gen | OpenCompass |
| Math | MATH | 46.6 | 4-shot CoT, gen | OpenCompass |
| Math | MathBench | 65.6 | mixed, gen | OpenCompass |
| Code | HumanEval+ | 68.3 | 0-shot, greedy | EvalPlus |
| Code | MBPP+ | 63.2 | 0-shot, greedy | EvalPlus |
| Code | LiveCodeBench v3 | 22.6 | 0-shot, greedy | LiveCodeBench harness |
| Function calling | BFCL v2 | 76.0 | 0-shot | Gorilla BFCL |

Overall reported average: 66.3. **Deferred (judge-based):** MT-Bench 8.41,
AlignBench v1.1 6.74, FollowBench-zh SSR 66.8.

> Few-shot counts follow OpenBMB conventions (MMLU 5, BBH 3, GSM8K 8, MATH 4, AGIEval 0).
> Exact per-dataset configs are taken from each harness's shipped MiniCPM/instruct config
> and pinned in the run log.

## Critical reproducibility gotcha

OpenBMB explicitly warns that anomalous scores from other frameworks (e.g. lm-eval) are
usually caused by **the missing `<s>` (bos) token**. Verbatim from the MiniCPM-S-1B-sft
model card:

> "Some abnormal results obtained with other popular frameworks such as LM-Eval are
> probably attributed to the absence of the cls token `<s>`."

Their quick fix — prepend token id `1` (which is `<s>`/BOS in MiniCPM's SentencePiece
tokenizer; OpenBMB loosely calls it the "cls token") when it isn't already the first token:

```python
if context_enc[0] != 1:
    context_enc = [1] + context_enc
```

**Why it moves scores.** For a decoder-only LM the first token's hidden state is computed
with nothing before it; models trained to always see `<s>` first learn representations
anchored on it. Drop it and every downstream position's activations shift. This bites
hardest in **loglikelihood/perplexity-style multiple-choice scoring** — how lm-eval scores
MMLU/CMMLU/CEval by default: it sums the option tokens' log-probs and takes the argmax over
options. A missing BOS shifts all option log-probs, flipping the predicted answer on many
close calls; across thousands of items that's a real accuracy swing. UltraEval prepends
`<s>`; lm-eval historically does not add BOS by default (its `add_bos_token` flag) — so the
two disagree on the same weights. OpenBMB lists the other cross-framework delta sources on
the same card: **few-shot settings, data pre-processing, and extra prompts** (hence we pin
few-shot counts and dataset versions per benchmark).

**Provenance caveat — RESOLVED in P0 (2026-07-14).** The quote + fix above are from the
**MiniCPM v1** cards (1B/2B `-sft`), not the MiniCPM3-4B card. Verified on the actual
MiniCPM3-4B model: its **chat/instruct path is ChatML** (`<|im_start|>…<|im_end|>`, eos
`<|im_end|>`) and does **NOT** prepend `<s>`. bos id *is* 1 and `add_bos_token=True`, but
that only affects the **plain base-model path** (`tok("…")` → `[1,…]`) — which is exactly
the path the v1 `<s>` bug concerned. Since we evaluate the *instruct* model through the
chat template, the classic `<s>` bug does not apply, and vLLM's **served chat tokenization
matches local `apply_chat_template` exactly** (checked via `/tokenize`). Net: bos handling
is consistent by construction across all harnesses; no `<s>` shim needed on the chat path.
(Evidence: `/cb/ml-eng/aarti/dsa/evals/minicpm3-4B/serving/logs/*_check_template.*`.)

**Mitigation.** Therefore **every harness must go through the MiniCPM3 chat template with
correct bos handling.** We enforce this by serving the model *once* and having all
harnesses hit the same OpenAI-compatible chat endpoint, so the chat template + bos are
applied identically everywhere. We also use OpenCompass **generation** (`_gen`) configs
rather than lm-eval's loglikelihood MMLU, which sidesteps the classic bug's main habitat.
Where a harness insists on local generation, we verify its prompt string starts with the
same `<s>...` prefix as the served path.

References: [MiniCPM-S-1B-sft card](https://huggingface.co/openbmb/MiniCPM-S-1B-sft) ·
[MiniCPM3-4B card](https://huggingface.co/openbmb/MiniCPM3-4B) ·
[lm-evaluation-harness `add_bos_token`](https://github.com/EleutherAI/lm-evaluation-harness).

## Architecture: one server, many clients

MiniCPM3-4B is 4B params → fits on a single H100 with room to spare, and the box has 8.
Stand up **one persistent vLLM OpenAI server** and point every harness at it. Benefits:
single chat-template/bos config (the parity variable above), high throughput (many eval
clients concurrently), trivial teardown.

```bash
# inside the verl vllm020 container (vllm 0.20.2 + transformers 4.57.1 — MiniCPM3 compatible)
# durable launch (setsid nohup) so it survives disconnect; log full invocation.
setsid nohup vllm serve openbmb/MiniCPM3-4B \
    --trust-remote-code \
    --served-model-name MiniCPM3-4B \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    > logs/vllm_minicpm3_serve.log 2>&1 &
```

Reserve GPU 0 for the server; harnesses that must generate locally (rare) use GPUs 1–7.

## Phase 0 — Serving + template sanity (blocker for everything)

1. Launch the server above; wait for `Application startup complete`.
2. Confirm the model card's chat template is the one vLLM loads (it ships in the repo's
   `tokenizer_config.json`). Dump the exact rendered prompt for one message and eyeball
   the `<s>` prefix and role markers.
3. Smoke test with 3 prompts (an EN QA, a ZH QA, a GSM8K-style math word problem) via
   `curl`/openai client. Confirm coherent chat output.
4. Record: model revision (git sha of the HF snapshot), vllm version, transformers
   version, full serve command, sampling defaults. → run log.

**Exit criteria:** endpoint answers chat completions; rendered prompt begins with `<s>`.

## Phase 1 — OpenCompass minimal set (MMLU + GSM8K)

Trimmed to two benchmarks that prove the whole OpenCompass path before we invest in the
rest. They're chosen to exercise the **two distinct scoring/prompt styles** the remaining
benches reuse:
- **MMLU** (5-shot) — few-shot multiple-choice knowledge; validates the meta-template,
  few-shot exemplar wrapping, and MC answer extraction.
- **GSM8K** (8-shot CoT) — few-shot chain-of-thought *generation*; validates CoT
  prompting and numeric-answer parsing.

The other six (CMMLU, CEval, BBH, MATH, MathBench, IFEval) are **deferred to Phase 6**
(after P5) — see below.

1. Install: `pip install opencompass` (or clone + `pip install -e .` for latest dataset
   configs). Download only the MMLU + GSM8K datasets via OpenCompass's data prep.
2. Model config: **OpenAI-API model** pointing at the vLLM endpoint
   (`openai_api_base=http://localhost:8000/v1`, `path=MiniCPM3-4B`), with the MiniCPM3
   **meta template** (so few-shot exemplars are wrapped in chat turns, matching how the
   instruct model was scored). This keeps bos/template identical to serving.
3. Dataset configs (use the `_gen` variants — generation, not perplexity, for a chat model):
   - `mmlu_gen` (5-shot)
   - `gsm8k_gen` (8-shot CoT)
4. Run: `opencompass <cfg>.py -w outputs/minicpm3_baseline` (or `--models`/`--datasets`
   flags). Batch via concurrent API requests.
5. Collect the summary CSV; compare to targets (MMLU 67.2, GSM8K 81.1); log deltas.

**Exit criteria:** MMLU and GSM8K land within ~2 pts of target. If so, the OpenCompass
path (template, few-shot, extraction) is trusted and Phase 6 is mechanical. If not,
root-cause here (post-processing, few-shot count, template/bos) before scaling out —
that's the whole point of trimming.

**Parity notes:** OpenCompass answer-extraction regexes for GSM8K can differ from
UltraEval. If the GSM8K delta is large (>2 pts), inspect post-processing before blaming
the model.

## Phase 2 — EvalPlus (HumanEval+, MBPP+)

1. Install: `pip install evalplus` (pin a recent release; note version — the `+` test
   sets get expanded over time, which shifts scores).
2. Generate against the vLLM endpoint in **chat mode**, greedy (temperature 0):
   ```bash
   evalplus.evaluate --model MiniCPM3-4B --dataset humaneval \
       --backend openai --base-url http://localhost:8000/v1 \
       --greedy --root outputs/evalplus
   evalplus.evaluate --model MiniCPM3-4B --dataset mbpp \
       --backend openai --base-url http://localhost:8000/v1 \
       --greedy --root outputs/evalplus
   ```
3. Report the `+` (extra-tests) pass@1. Sanitize generations
   (`evalplus.sanitize`) if raw pass@1 looks depressed by formatting.

**Parity notes:** score depends on EvalPlus test-set version and on the chat prompt
wrapper. Pin the EvalPlus version in the log. Confirm code is extracted from markdown
fences (MiniCPM3 tends to fence its code).

## Phase 3 — LiveCodeBench v3

1. Clone LiveCodeBench; select **`release_v3`** (the reported number is v3 — later
   releases add problems and change the number).
2. Run `code_generation` scenario against the vLLM endpoint (LCB supports an
   OpenAI-compatible / vLLM backend), greedy, then its evaluator for pass@1.
3. Report pass@1 over the v3 problem set.

**Parity notes:** must fix `--release_version release_v3` and the date window; otherwise
the denominator changes and 22.6 is not comparable.

## Phase 4 — Gorilla BFCL v2 (function calling)

Highest-effort item — BFCL needs a model *handler*.

1. Clone `gorilla/berkeley-function-call-leaderboard`; install.
2. MiniCPM3 has native tool-calling with a specific template. Options, in order of
   preference:
   - Use BFCL's **OpenAI-compatible / prompting handler** pointed at the vLLM endpoint,
     supplying MiniCPM3's function-calling system prompt format; OR
   - Write a small custom handler that renders tools in MiniCPM3's expected format and
     parses its tool-call output.
3. Run BFCL **v2** categories (AST: simple / multiple / parallel / parallel-multiple +
   relevance; executable subsets), compute the v2 overall accuracy.
4. Report the v2 aggregate to compare with 76.0.

**Parity notes:** BFCL versioning matters (v2 vs v3 scoring differ). Confirm the handler's
tool-call parsing matches MiniCPM3's output grammar — a parsing bug reads as a huge score
drop, not a model regression. This phase may need a dedicated debugging pass.

## Phase 5 — Consolidate

1. Assemble a scorecard table: benchmark | reported | ours | delta | harness+version.
2. Flag any delta > ~2 pts and root-cause (post-processing, few-shot count, template/bos,
   dataset version) before accepting.
3. Save the scorecard + all run commands/env to the run log so results self-reproduce.

## Effort / ordering

- **P0** (serve+sanity): ~½ day — unblocks all.
- **P1** (OpenCompass, 8 benches): ~1–1.5 days incl. dataset download + delta triage.
- **P2** (EvalPlus): ~½ day.
- **P3** (LiveCodeBench): ~½ day.
- **P4** (BFCL): ~1–2 days (handler is the wildcard).
- **P5** (consolidate): ~½ day.

Recommended order: P0 → P1 (biggest payoff, one tool) → P2 → P3 → P4.

## Open risks

- **BFCL handler** is the main unknown; may need custom code for MiniCPM3's tool format.
- **Dataset versioning** (EvalPlus `+` sets, LiveCodeBench release) directly moves scores —
  pin everything.
- **Template/bos** parity is the single highest-leverage correctness factor; the
  one-server design mitigates it but verify in P0.
- Deferred judge-based benches (MT-Bench/AlignBench/FollowBench) remain open until a
  judge API is decided.
```
