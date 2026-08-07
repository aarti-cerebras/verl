# gpt-oss-20b MSA — Phase-2 behaviour-cloning data generation

Build plan for the gpt-oss-20b Phase-2b BC dataset, as a **delta against the Qwen3 run**
(`../qwen3_4b_msa/phase2_data_gen.md`). Read that document first: everything about the prompt bank —
what `allenai/Dolci-Think-RL-32B` is (§1), why it is the right source (§2), the slice policy (§3), and
the extraction contract (§4) — is inherited unchanged and is **not** restated here.

Status as of **2026-08-07**: prompt bank and splits resolved and verified; pipeline code landed and
unit-tested; **nothing generated yet**. The pilot (§8) is the gate.

Decisions taken by the user, 2026-08-07:

| | |
|---|---|
| **reasoning effort** | **`medium`** (the template's own default, but passed explicitly — §4) |
| **window** | **32,768**, generation window == training window, matching Qwen3 |

---

## 1. What transfers, and what does not

| | Qwen3-4B-Thinking-2507 | gpt-oss-20b | transfers? |
|---|---|---|---|
| prompt bank | 93,889 unique Dolci-Think-RL prompts | same file, byte-for-byte | **yes** |
| `prompt_sha256` | sha256(cleaned prompt text) | identical | **yes** — generator-independent |
| train/val split | 511 val prompts, sampled seed 1234 | reuse by frozen sha list | **yes**, §2 |
| response grammar | `<think>`…`</think>` | harmony channels | **no** — §5, §6 |
| chat prefix | 10-token wrapper | **67-token** wrapper + system msg | **no** — §3 |
| prefix determinism | deterministic | **wall-clock date baked in** | **no** — §4 |
| sampling params | 0.6 / 0.95 / top-k 20 | model card: 1.0 / 1.0 | **no** — §7 |

**The prompt bank is reused verbatim — there is no second extraction.** Selection ran once, for Qwen3, on
2026-07-30 (`select_prompts.py --source dolci-rl`: 102,026 raw → −1,323 flattened multi-turn → −6,814
exact-duplicate prompts → **93,889 unique**). gpt-oss inherits that exact file. Nothing is re-derived from
the HF dataset, so the two runs cannot drift on prompt selection — which is what makes §2's shared split
sound in the first place.

The one tokenizer-dependent column, `prompt_tokens`, was Qwen3-tokenized. It is *inert* at generation time
(`--fit-window` recomputes the served prefix live from whichever tokenizer is loaded), but a wrong number
in an artifact eventually becomes a wrong number in a report, so it was rewritten with the gpt-oss
tokenizer. **DONE, 2026-08-07:**

```
/cb/ml-eng/aarti/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_prompts_20260807_155539/prompts.jsonl
  93,889 rows · sha256 817fdbc5c91c24fb… · MANIFEST.json + logs/ alongside
```

`scripts/msa/retokenize_prompt_tokens.py` copies every other field and the row order through untouched,
asserts the `prompt_sha256` **sequence** is identical to the source, and adds `prefix_tokens` (chat wrapper
+ prompt — the number the generation budget is actually computed from). Verified after the fact by
comparing all 93,889 rows field-by-field against the Qwen3 file: **only `prompt_tokens` and `prefix_tokens`
differ; the prompt text is identical on every row.**

---

## 2. The train/val split — same prompts, by construction

**Requirement (user, 2026-08-07): gpt-oss-20b must train and evaluate on the same splits as Qwen3.**

`prompt_sha256` is `sha256(cleaned_prompt_text)` (`select_prompts.py`), so the prompt-level boundary is
generator-independent and *can* be shared. But it does not share itself:

> **Re-running `split_bc_val.py` with the same seed does NOT reproduce the split.** It samples per domain
> in proportion to surviving row counts. Each generator loses a different ~4 % of prompts to the health
> filters, so both `want` and the `np.unique(sha)` universe differ, and a *different* prompt set is drawn.

So the Qwen3 val set is frozen to a file and selected by membership thereafter:

```
/cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__ph2b_split_v1/val_prompt_shas.txt
  511 prompt_sha256, one per line     sha256(file) = d101c6586ead1726e5739896cd27a1f55fce13e71450c24cffdfeaa5e019c2aa
  Code 116 · General 108 · IF 167 · Math 120
```

Consumed by the new `--val-sha-file` flag (§9). **Verified**: replaying it against the Qwen3 parquet
reproduces the existing split exactly — 511 val / 89,719 train, same per-domain counts, 0 missing.

**What "same split" can and cannot mean.** The val *prompts* match; the val *rows* cannot, because each
model's BC val set has to consist of that model's own traces. What this buys is (a) prompt-comparable val
sets across the two runs and (b) a guaranteed absence of cross-split leakage in each. Where a held-out
prompt's gpt-oss trace is dropped by a health filter, that prompt contributes no val row — the val set
comes out slightly under 511. It is **never backfilled from train**; the count is logged and recorded in
the manifest as `val_shas_missing`.

---

## 3. The window, measured

Fixed harmony wrapper is **67 tokens** (Qwen3: 10) — gpt-oss prepends a full system message (identity,
knowledge cutoff, date, reasoning effort, channel list). Measured over all 93,889 prompts with the
gpt-oss-20b tokenizer at `reasoning_effort=medium`:

| prompt tokens | ALL | Math | Code | IF | General |
|---|--:|--:|--:|--:|--:|
| p50 | **113** | 88 | 232 | 155 | 38 |
| p99 | 899 | 460 | 1,043 | 1,015 | 713 |
| max | **2,015** | 2,013 | 1,935 | 2,015 | 1,989 |

o200k is slightly more efficient than Qwen3's tokenizer here (max 2,015 vs 3,202), which more than offsets
the 57-token-larger wrapper.

| | tokens |
|---|--:|
| training window | 32,768 |
| − harmony wrapper | 67 |
| − prompt (p50 / max) | 113 / 2,015 |
| − closing `<|return|>` | 1 |
| **= `max_tokens` (p50 / worst case)** | **32,587 / 30,685** |

**0 prompts leave under 8,192 tokens of budget; 0 exceed the window.** Same conclusion as Qwen3: the 32K
cap is a generation budget, not a filter, and the only length-related yield question is the
`finish_reason == "length"` rate (§8).

---

## 4. The prefix is not deterministic unless you pin it

The harmony template builds its system message with `strftime_now("%Y-%m-%d")`:

```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-08-07          <-- the wall-clock day, baked into EVERY served prefix
Reasoning: medium
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>user<|message|>{prompt}<|end|><|start|>assistant
```

Two ways this bites, both silent:

1. **A resumed run disagrees with its first leg.** The Qwen3 full run took ~9 h and was resumed twice
   across two days (`RESUME_LOG.txt`). The same pattern here yields a parquet whose rows carry two
   different system prompts, differing on one line.
2. **Training skews from serving** the moment the date rolls over — the Phase-2 model would be trained on
   prefixes it never sees at inference.

`reasoning_effort` is the same class of problem: a real template variable defaulting to `"medium"`, so a
run that never passes it records nothing and silently inherits whatever the default is that week.

**Fix (verified against gpt-oss-20b, transformers 4.57):** Jinja context variables shadow globals, so both
are pinnable as ordinary `apply_chat_template` kwargs — no forked model directory needed. This is
`_dsa_tok.pin_template_kwargs`, exposed as `--pin-date` and `--reasoning-effort` on both the generator and
the converter. Effort does not change the prefix *length* (low/medium/high are all one token — 71 tokens
for the same prompt), only its content.

> **The same pinned date and effort must be used at Phase-2 serving time.** Record both in the run
> MANIFEST. They are part of the dataset's identity, exactly like the generator model.

---

## 5. The splice contract (the Qwen3 §6 analogue)

Verified by rendering the real gpt-oss-20b template. Served prefix ends at `<|start|>assistant`; a
well-formed completion is:

```
<|channel|>analysis<|message|>{CoT}<|end|><|start|>assistant<|channel|>final<|message|>{answer}<|return|>
```

Marker ids (resolved from the tokenizer at runtime and asserted, never hardcoded):
`<|return|>` 200002 · `<|constrain|>` 200003 · `<|channel|>` 200005 · `<|start|>` 200006 ·
`<|end|>` 200007 · `<|message|>` 200008 · `<|call|>` 200012. Identical in gpt-oss-120b (templates are
byte-identical), but confirmed against the 20B's own checkpoint.

**The splice is mandatory here for the same reason as Qwen3, plus a stronger one.** The harmony template
renders the analysis channel *only* when the final turn is an assistant turn and `add_generation_prompt`
is false, and it **raises an exception outright** if you pass `<|channel|>` tags in `content`. The
re-render path is not merely lossy, it is hostile. So:

```python
input_ids = prefix_ids + list(completion.token_ids) + [200002]   # <|return|> if not already terminal
loss_mask = [0] * len(prefix_ids) + [1] * (len(input_ids) - len(prefix_ids))
```

**`<|return|>`, not `<|end|>`, is the terminator.** The template's own comment settles it — *"`<|return|>`
indicates the end of generation, but `<|end|>` does not"* — and it renders a final assistant turn with
`<|return|>`. It is also `config.eos_token_id` (200002) and it is what the model actually sampled, so the
spec-correct choice and the on-policy choice agree.

---

## 6. Health filters — why the Qwen3 rules cannot be reused

> **The Qwen3 filter rejects 100 % of gpt-oss rows.** `_malformed()` returns `role_marker_leak` when the
> response contains `<|im_start|>`; a legal harmony completion contains an internal `<|start|>assistant`
> between the analysis and final channels. Run unchanged, the converter would drop every row and then
> die on `assert spliced, "no rows survived"` — loudly, but only after a full generation run.

`--chat-format harmony` selects this table instead. Channel *names* are ordinary text, so only the 1–3
header tokens are decoded for inspection; the payload is never decoded or re-encoded.

| class | detection | policy |
|---|---|---|
| `truncated` | `finish_reason == "length"` | drop (2b) |
| `tool_call_leak` | `<\|call\|>` or `<\|constrain\|>` present | drop — we serve no tools |
| `no_channel_header` | no `<\|channel\|>`…`<\|message\|>` pair | drop |
| `commentary_channel` | a `commentary` channel present | drop — tool/preamble scaffolding |
| `no_final_channel` | zero `final` channels | drop — reasoned, then stopped with no answer |
| `multiple_final` | >1 `final` channel | drop — channel split would mis-parse |
| `empty_answer` | nothing non-whitespace after the final `<\|message\|>` | drop |
| `repetition_loop` | 32-token n-gram repeated ≥8×, >30 % coverage | drop (unchanged) |
| `stub` | `len(resp) < 8` | drop |
| `no_analysis_first` | first channel is not `analysis` | **keep, count** — legal, but a rising rate means the sampling config drifted |
| over-window | `len(input_ids) > 32768` | assert, never filter |

Every class gets a per-run counter in `<out>.COUNTERS.json`, as on the Qwen3 run.

---

## 7. Generation configuration

| knob | value | note |
|---|---|---|
| model | `/cb/ml-eng/aarti/models/gpt-oss-20b` | **configs + tokenizer downloaded; weights NOT yet** (§10) |
| `reasoning_effort` | **medium** | user decision; passed explicitly, recorded per row |
| `--pin-date` | fixed `YYYY-MM-DD`, recorded | §4 |
| temperature / top_p | **1.0 / 1.0** | OpenAI's card for gpt-oss; do not carry over Qwen3's 0.6/0.95/20 |
| `max_tokens` | per row: `32768 − len(prefix) − 1` | `--fit-window 32768`, §3 |
| `max_model_len` | 32768 | |
| `n` | 1 | matches the Qwen3 full run |
| seed | 1234, recorded | |
| parallelism | `--data-parallel-size 8 --tensor-parallel-size 1` | 20B MXFP4 is ~13 GB — one replica per H100 fits |
| chunking | `--chunk-size 512` | append + resume; a multi-hour run *will* be interrupted |

**Throughput is unknown and will not resemble Qwen3's.** 3.6B active params (MoE) cuts against 5× the
total parameters and MXFP4 dequant overhead. The Qwen3 run measured ~1.8K tok/s/GPU → ~9 h for ~510M
tokens; treat that as an unmeasured prior and let the pilot set the schedule.

---

## 8. Pilot before the full run — non-negotiable

Same gate as Qwen3, whose pilot caught two silent bugs that would each have wasted a day of H100 time.
**100 prompts, 25 per slice**, full pipeline end to end.

Report:

1. **`finish_reason == "length"` rate per slice** — the headline number, and the one that decides whether
   32K stands. Qwen3 came back 8 % for Math/Code.
2. Realized response/total length histograms per slice → the decode-long yield, and hence the mixture.
   This is where `medium` effort gets validated: if traces are much shorter than Qwen3's (Math p50 20,669),
   the decode-long bucket undershoots and `high` becomes worth revisiting.
3. All §6 counters, especially `no_analysis_first` and `commentary_channel` — either being non-trivial
   means the served prompt or the effort setting is doing something unintended.
4. Splice verification on real rows: mask boundary, prefix-monotonicity, `<|return|>` terminal,
   exactly one `final` channel, `len ≤ 32768`.
5. Throughput → the wall-clock estimate for the full run.
6. One hand-inspected sample per slice.

**Gate: do not launch the full run until (1) is known and (4) passes on pilot rows.**

---

## 9. Code changes

| file | change | status |
|---|---|---|
| `scripts/msa/split_bc_val.py` | `--val-sha-file`: select val by frozen `prompt_sha256` membership instead of sampling, so a second generator reproduces the same prompt-level split. Records the file's sha256 + `val_shas_missing` in the manifest | **DONE**, verified against the Qwen3 parquet (exact reproduction) |
| `scripts/dsa/_dsa_tok.py` | `pin_template_kwargs()` (pins `strftime_now` / `reasoning_effort`); `chat_prefix_ids(**tpl_kwargs)` | **DONE** |
| `scripts/dsa/gen_trajectories.py` | `--reasoning-effort`, `--pin-date`, `--chat-template-kwargs`; threaded to the vLLM + HF paths **and to the DP children** (a replica falling back to a default would write prefixes unlike its 7 siblings'); both recorded per row | **DONE** |
| `scripts/dsa/trajectories_to_sft_parquet.py` | `--chat-format {qwen3-think,harmony}`; runtime-asserted marker resolution; harmony channel parser; the §6 filter table; `<\|return\|>` terminator; template kwargs for the prefix cross-check | **DONE** |
| `tests/dsa/test_harmony_sft_roundtrip.py` | prefix pinning/determinism, effort plumbing, mask boundary, verbatim trace, terminator, every §6 class, verifier accept/reject, and explicit guards that the qwen3-think converter *and* verifier would each have rejected a legal harmony row | **DONE — 21 passed** |
| `scripts/dsa/verify_sft_parquet.py` | `--chat-format harmony` + `--tokenizer`: prefix ends at `<\|start\|>assistant`, exactly one `final` channel, `<\|return\|>` terminal, no tool-call markers, non-empty answer. An **independent re-derivation** of §6, not a call into it. Without this the verifier — a hard, exit-non-zero gate in the post-process chain — **fails every harmony row** | **DONE** |
| `scripts/msa/sample_prompts_stratified.py` | domain-stratified pilot subset + `--exclude-sha`. `prompts.jsonl` is domain-ordered, so `--limit 100` gives a **100 % Math** pilot — see §8 | **DONE**, pilot set built |
| `scripts/dsa/analyze_lengths.py` | `--chat-wrapper-tokens` (10 → **67** for harmony); the constant was hardcoded and skewed the generation-budget report by 57 tokens | **DONE** |
| `verl/utils/dataset/packed_pretrain_dataset.py` | honour an optional `loss_mask`; do not drop rows shorter than `seq_len` | **inherited from the Qwen3 plan §10 — still required, but this is TRAINING, not data generation** |
| `scripts/dsa/decontaminate.py` | §8 sha + 13-gram pass over the gate sets | **still not written** (was already outstanding for Qwen3; applicable post-hoc via `prompt_sha256`) |

Regression-checked: the Qwen3 path is byte-identical under the defaults (`--chat-format qwen3-think`,
no pinning) — same counters, same `<|im_end|>` terminator, same mask.

---

## 10. Commands

```bash
MODEL=/cb/ml-eng/aarti/models/gpt-oss-20b
PROMPTS=/cb/ml-eng/aarti/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_prompts_20260807_155539/prompts.jsonl
RUN=/cb/ml-eng/aarti/msa/data/gpt-oss-20b__dolci-think-rl-32b__ph2b_L32768_$(date +%Y%m%d_%H%M%S)
PIN=2026-08-07        # freeze once, reuse for every leg AND at serving time
mkdir -p $RUN/logs

# 0. weights (configs + tokenizer already present; this is the ~13 GB MXFP4 part)
python3 -c "from huggingface_hub import snapshot_download; \
  snapshot_download('openai/gpt-oss-20b', local_dir='$MODEL')"

# 1. prompts -- ALREADY BUILT (§1). Same 93,889 Qwen3 prompts, prompt_tokens re-tokenized for gpt-oss.
#    Rebuild only if $PIN or the effort changes:
#    python3 scripts/msa/retokenize_prompt_tokens.py --src <qwen3 run>/prompts.jsonl --out $RUN/prompts.jsonl \
#      --tokenizer $MODEL --reasoning-effort medium --pin-date $PIN --window 32768

# 2. pilot prompt set -- ALREADY BUILT, next to $PROMPTS as prompts_pilot100.jsonl (25/domain, val excluded).
#    Do NOT use `gen_trajectories --limit 100`: prompts.jsonl is domain-ordered, so that is 100% Math (§8).
PILOT=$(dirname $PROMPTS)/prompts_pilot100.jsonl

# 3. generate.  Pilot: --prompts $PILOT.  Full run: --prompts $PROMPTS.
python3 scripts/dsa/gen_trajectories.py --prompts $PILOT --out $RUN/trajectories.jsonl \
  --log-dir $RUN/logs --model $MODEL \
  --reasoning-effort medium --pin-date $PIN \
  --temperature 1.0 --top-p 1.0 \
  --max-model-len 32768 --fit-window 32768 \
  --data-parallel-size 8 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --chunk-size 512 --seed 1234 --no-merge

# 4. splice -> input_ids + loss_mask  (§5, §6).  --pin-date MUST match step 3.
python3 scripts/dsa/trajectories_to_sft_parquet.py --input "$RUN/trajectories.jsonl.part*" \
  --tokenizer $MODEL --chat-format harmony --reasoning-effort medium --pin-date $PIN \
  --emit-input-ids --max-length 32768 --out $RUN/bc_2b.parquet --log-dir $RUN/logs

python3 scripts/dsa/verify_sft_parquet.py --parquet $RUN/bc_2b.parquet --max-length 32768 \
  --chat-format harmony --tokenizer $MODEL --log-dir $RUN/logs

# 5. THE SAME train/val split as Qwen3  (§2)
python3 scripts/msa/split_bc_val.py --src $RUN/bc_2b.parquet --out-dir ${RUN}__split_v1 \
  --val-sha-file /cb/ml-eng/aarti/msa/data/qwen3-4b-thinking-2507__ph2b_split_v1/val_prompt_shas.txt

# 6. reports  (--chat-wrapper-tokens 67: harmony, not Qwen3's 10)
python3 scripts/dsa/analyze_lengths.py --trajectories "$RUN/trajectories.jsonl.part*" --window 32768 \
  --chat-wrapper-tokens 67 --group-by domain --out-report $RUN/lengths.json --log-dir $RUN/logs
```

Write `MANIFEST.json` (generator model, prompt dataset + ODC-BY attribution, sampling params,
**reasoning effort, pinned date**, window, host, created) plus `launch_gen.sh`, `git_sha.txt`,
`git_diff.patch` into `$RUN` — `memory:log-full-invocation`, `memory:msa-data-artifact-layout`.

---

## 11. Open questions

1. **Does Phase 2 even mean the same thing for this architecture?** gpt-oss-20b is 24 layers
   **alternating `sliding_attention` (window 128) and `full_attention`** — so only **12 layers** carry
   global attention at all, and the other 12 are already a 128-token sliding window. Both the sparse-
   attention target and the KL supervision term have half as many layers to act on as they did for Qwen3,
   and the "sparsify long-context attention" premise applies only to those 12. This is a Phase-1
   architecture question, not a data-gen one, but it should be settled before spending ~9 h of H100 time
   on BC data — it could change the window, the mixture, or whether this model is the right vehicle.
2. **vLLM support for `GptOssForCausalLM` + MXFP4 in our container.** Needs `nvidia-smi`-level
   verification before the pilot; gpt-oss also uses attention sinks, which need kernel support. The DSA
   container is vLLM 0.20.2 (gpt-oss landed in 0.10.1) so it should be there, but "should" has cost this
   project two days before.
3. **`medium` vs `high` effort.** Decided as `medium`; revisit only if pilot trace lengths come in far
   under Qwen3's and the decode-long bucket undershoots the 30 % target (§8.2).
4. **Decontamination** (`../qwen3_4b_msa/phase2_data_gen.md` §8) is still not implemented, and is
   load-bearing for 2b in both runs. Every row carries `prompt_sha256`, so it can be applied after
   generation without regenerating.
5. **Short-real bucket.** The Qwen3 mixture (§9.1 there) fills 25 % from re-windowed real docs at 4K.
   Those windows are tokenizer-specific and would need rebuilding with the gpt-oss tokenizer
   (`examples/dsa/prepare_real_data.py --model $MODEL --seq_len 4096`). Not generation work, but not free
   either.
