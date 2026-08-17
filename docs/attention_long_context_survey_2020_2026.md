# Attention Mechanisms & Long-Context Strategies in Large LLMs (2020 → 2026)

*Compiled 2026-08-06. Reverse chronological (latest first). Production/frontier models are
separated from the research literature that feeds them. Closed-weight models are marked
**unconfirmed** — those architecture claims circulate on blogs with no primary source.*

---

> **Start here if you want the short version:** [Part 10 — Cross-lab master summary](#part-10--cross-lab-master-summary)
> has the full roster, a mechanism-adoption matrix across 18 labs, the five disputed questions, and
> a paper index grouped by lab.

## Part 1 — The taxonomy (what the strategies actually are)

Six distinct families. Every frontier model in 2026 is a specific combination of them.

| # | Family | Core idea | Representative production use |
|---|---|---|---|
| 1 | **Dynamic / trainable sparse attention** (learned block gating — MoE routing applied to attention) | A cheap learned scorer picks top-k KV blocks per query; exact attention over the selection only | SeerAttention, NSA, **MoBA**, InfLLM-v2, DeepSeek DSA, MiniMax MSA, GLM-5.x |
| 2 | **Static / structural sparsity** | Fixed pattern: sliding window, local+global interleave, dilation, attention sinks | **Inkling**, Gemma 2/3, gpt-oss, Mistral, MiMo-V2-Flash, Longformer/BigBird |
| 3 | **Linear / recurrent-state attention** | Fixed-size state, O(1) per token decode; delta-rule and gating for memory control | Kimi KDA, Qwen Gated DeltaNet, MiniMax Lightning Attention, Mamba |
| 4 | **KV-cache compression at the architecture level** | Shrink what's stored per token, not how much you attend to | MLA (DeepSeek/Kimi/GLM), GQA/MQA, CSA/HCA, cross-layer sharing, YOCO; retrofit via TransMLA/MHA2MLA (§5) |
| 5 | **Positional encoding & context extension** | Make a short-trained model generalize long | RoPE → NTK → YaRN → LongRoPE; NoPE; iRoPE; HyPE |
| 6 | **Systems / parallelism / memory** | Distribute or offload rather than approximate | FlashAttention, Ring/context parallelism, ByteScale, Titans, Infini-attention, KV offload |
| 7 | **Context as external environment** (§9.2) | Don't put N tokens through attention at all — store the context outside the model and let it query, decompose, or recurse over the context programmatically | RLM, INTELLECT-3 self-managed context, QwenLong-CPRS, RAG |

**The 2026 consensus shape:** hybrid layer stacks (3:1 or 5:1 efficient-to-full ratio) +
MLA-class KV compression + learned top-k sparsity in the remaining full layers + a staged
long-context curriculum. Notably, *the field has not converged* — Qwen bet on linear attention,
DeepSeek and Z.ai on sparse attention, Moonshot on both, MiniMax reversed course twice.

---

## Part 2 — Frontier models, reverse chronological

### 2026

**Kimi K3** (Moonshot, Jul 2026) — [arXiv 2607.24653](https://arxiv.org/pdf/2607.24653)
Layerwise hybrid: 3× **KDA** (Kimi Delta Attention, linear) : 1× **Gated MLA**. Applies **NoPE to
all MLA layers** — a departure from K2/K2.5, which used RoPE'd MLA. 1M context; reported ~2.5×
K2's scaling efficiency. The Kimi Linear recipe carried to frontier scale.

**Inkling / Inkling-Small** (Thinking Machines Lab, Jul 2026) —
[HF release notes](https://huggingface.co/blog/thinkingmachines-inkling) ·
[Raschka architecture notes](https://sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html) ·
[vLLM day-0](https://vllm.ai/blog/2026-07-15-inkling) · [AINews](https://www.latent.space/p/ainews-thinkys-inkling-975b-a41b)
975B total / 41B active (4.2%), Apache 2.0, multimodal, **1,048,576-token context**, 45T training
tokens. Inkling-Small is 276B/12B on the same architecture. Architecturally the most interesting
counterexample in this whole list, because it reaches 1M with **no learned sparsity and no linear
attention**:
- **66 layers: 55 sliding-window (512-token window) + 11 global, repeating 5:1**, final layer global.
- **Asymmetric GQA by layer type** — local layers 64 Q heads / 16 KV heads (4:1); global layers
  64 Q / 8 KV (8:1). The expensive global layers carry *half* the KV heads. Rare and deliberate.
- **No RoPE.** A **learned, input-dependent relative-position bias** computed from the query and
  key states is added to pre-softmax logits. The bias spans the full 512-token window in local
  layers and 1,024 tokens in global layers; **beyond that span there is no explicit positional term
  at all** — effectively relative-bias locally, NoPE-like at range.
- **Short convolutions (SConv)** — four causal convs per decoder layer, kernel size 4: two after
  the key/value projections, two after the attention and MoE outputs before the residual stream.
  Offloads local mixing off attention. (The HF post describes the SConv window as matching the
  sliding-window size; Raschka's teardown specifies kernel size 4 — I'd trust the latter.)
- MoE: 256 routed + 2 shared experts, 6 routed + 2 shared active per token. RMSNorm applied
  directly after the token embedding. **MTP drafter layers** for speculative decoding in both sizes.
  Inkling-Small ships MXFP8 and NVFP4 quantization.

**MiniMax Sparse Attention (MSA)** (Jun 2026) — [arXiv 2606.13392](https://arxiv.org/abs/2606.13392)
Blockwise sparse attention built on **GQA**, not MLA. A lightweight *Index Branch* scores KV
blocks and picks a top-k subset **independently per GQA group**; the Main Branch does exact
block-sparse attention over just those. Deliberately hardware-portable. Note the arc:
MiniMax-01 (linear) → M1/M2 (back to full attention) → MSA (sparse).

**DeepSeek-V4** (Apr–Jun 2026) — [arXiv 2606.19348](https://arxiv.org/pdf/2606.19348) ·
[MarkTechPost](https://www.marktechpost.com/2026/04/24/deepseek-ai-releases-deepseek-v4-compressed-sparse-attention-and-heavily-compressed-attention-enable-one-million-token-contexts/)
Two interleaved mechanisms:
- **CSA (Compressed Sparse Attention)** — a learned token-level compressor folds every *m* tokens
  into one KV entry, then DSA top-k over the compressed entries.
- **HCA (Heavily Compressed Attention)** — much larger compression rate *m′ ≫ m*, but **dense**
  over the result.

Both carry a sliding-window branch over the last *n_win* tokens. Reports 97% NIAH at 1M.
Related: [FlashMemory-DeepSeek-V4: lookahead sparse attention, arXiv 2606.09079](https://arxiv.org/abs/2606.09079).

**GLM-5.2** (Z.ai, Jun 2026) — [Raschka analysis](https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html) ·
[MindStudio deep dive](https://www.mindstudio.ai/blog/glm-5-2-architecture-index-share-sparse-attention)
Adds **IndexShare** on top of MLA+DSA: run the DSA top-k indexer fully only **once every 4
layers** and let following layers reuse the selected indices. Claims 2.9× per-token FLOP
reduction at 1M — but *not* a 2.9× wall-clock speedup, and it does **not** cut KV-cache memory
proportionally. 744B total / ~40B active.

**Qwen3.6** (Alibaba, Apr 2026) — [AI Wiki](https://aiwiki.ai/wiki/qwen3_6)
Keeps the Qwen3.5 generation-wide hybrid: Gated DeltaNet linear attention + sparse MoE + native
vision encoder. Reoriented toward agentic coding rather than architecture change.

**GLM-5.1** (Z.ai, Apr 2026) — MLA + DSA, MIT-licensed weights.

**Mamba-3** (CMU/Princeton/Together/Cartesia, Mar 2026) —
[arXiv 2603.15569](https://arxiv.org/abs/2603.15569) · [ICLR 2026](https://openreview.net/pdf?id=HwCvaJOiCj) ·
[Princeton PLI](https://pli.princeton.edu/blog/2026/mamba-3-improved-sequence-modeling-using-state-space-principles)
Optimized for *inference*, not training throughput (inverting Mamba-2's goal). Three changes:
exponential-trapezoidal discretization (local error O(Δt²)→O(Δt³)), **complex-valued state
updates** for richer state tracking, and a **MIMO** formulation that adds quality without
decode-latency cost. At 1.5B, beats Gated DeltaNet by +0.6pp (+1.8pp with MIMO).

**GLM-5** (Z.ai, Feb 2026) — [arXiv 2602.15763](https://www.emergentmind.com/papers/2602.15763)
744B/40B MoE, 256 experts. **MLA + DeepSeek Sparse Attention**, RoPE, 200K input / 131K
generation. Staged long-context curriculum 32K → 200K over ~28.5T tokens.

**MiniCPM-SALA** (OpenBMB, Feb 2026) — [arXiv 2602.11761](https://arxiv.org/abs/2602.11761) ·
[weights](https://huggingface.co/openbmb/MiniCPM-SALA)
The cleanest example of *sparse × linear* hybridization: 9B with **25% InfLLM-V2 (sparse) layers
+ 75% Lightning Attention (linear) layers**, plus **HyPE (Hybrid Positional Embedding)**. Scales
past 1M with real length generalization.

**Qwen3.5-397B-A17B** (Alibaba, 16 Feb 2026) — [Labonne analysis](https://huggingface.co/blog/mlabonne/qwen35) ·
[gated-deltanet notes](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
3:1 **Gated DeltaNet : full attention** + sparse MoE (397B/17B), 262K context. GDN = delta-rule
error-correcting memory writes + exponential gating + causal Conv1D for local context + L2-normed
Q/K.

**MiMo-V2-Flash** (Xiaomi, Jan 2026) — [arXiv 2601.02780](https://arxiv.org/abs/2601.02780) ·
[code](https://github.com/xiaomimimo/MiMo-V2-Flash)
309B total / 15B active MoE. Structural sparsity: **8 Hybrid Blocks, each interleaving 5 SWA blocks
with 1 Global Attention block** — a 128-token window at 5:1, giving **~6× KV-cache reduction** —
plus a **learnable attention sink bias** (as in gpt-oss) credited with preserving long-context
quality at that window size. MTP, native 32K extended to 256K, 27T tokens, and Multi-Teacher
On-Policy Distillation for post-training. See §9.1.

### 2025

**Kimi Linear** (Moonshot, Oct 2025) — [arXiv 2510.26692](https://arxiv.org/abs/2510.26692)
**KDA** = Gated DeltaNet with *finer-grained (channel-wise) gating*, implemented via a specialized
**Diagonal-Plus-Low-Rank (DPLR)** chunkwise kernel. 3 KDA : 1 MLA. Beats pure MLA at matched
recipe on short-context, long-context, and RL tasks; −75% KV cache, up to 6× decode throughput at
1M. Open KDA kernel + vLLM path.

**DeepSeek-V3.2 / V3.2-Exp** (Sep–Dec 2025) — [arXiv 2512.02556](https://arxiv.org/html/2512.02556v1) ·
[vLLM blog](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)
**DSA** = **lightning indexer** (few heads, FP8-able) computing query↔compressed-key relevance +
**fine-grained token selection** (top-k) → exact attention over the selection. ~3–6× cost
reduction at 128K. Critically, it is introduced by **two-stage continued training** from
V3.1-Terminus: sparsity must be *learned gradually*, not imposed. See also
[Raschka's DSA explainer](https://sebastianraschka.com/llm-architecture-gallery/deepseek-sparse-attention/).

**Qwen3-Next-80B-A3B** (Sep 2025) — [Qwen blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd) ·
[vLLM](https://blog.vllm.ai/2025/09/11/qwen3-next.html)
First Qwen to ship the 3:1 **Gated DeltaNet + Gated Attention** hybrid, ultra-sparse MoE
(80B/3B), MTP. Stated rationale: linear attention is fast but weak at recall; full attention is
expensive; neither monolith wins.

**gpt-oss-120b / gpt-oss-20b** (OpenAI, 5 Aug 2025) —
[model card, arXiv 2508.10925](https://arxiv.org/pdf/2508.10925) ·
[OpenAI PDF](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf) ·
[Raschka: GPT-2 → gpt-oss](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the) ·
[Wolfe](https://cameronrwolfe.substack.com/p/gpt-oss) ·
[NVIDIA Megatron-Bridge config](https://docs.nvidia.com/nemo/megatron-bridge/0.3.1/models/llm/gpt-oss.html)
Two sizes, both MoE with 4 active experts per layer via a plain linear router: **120b** = 128
experts, ~5.1B active; **20b** = 32 experts, ~3.6B active. Long-context strategy is three parts:
- **Alternating dense ↔ locally banded sparse (sliding-window) attention with a 128-token window** —
  a 1:1 interleave, and by far the smallest window of any model on this list (Gemma 3 uses 1024,
  Gemma 2 used 4096, Inkling uses 512).
- **Learned per-head attention-sink bias in the softmax denominator** — an extra logit that lets a
  head attend to *effectively nothing*. This is what makes a 128-token window survivable, and it is
  the direct architectural descendant of the StreamingLLM attention-sink observation (2023).
- **GQA**, 64 attention heads, group size 8, head dim 64. Context extended to **131,072 via YaRN** —
  NTK-by-parts interpolation plus the softmax temperature term. Note gpt-oss does *not* attempt 1M;
  it extends a conventional window rather than redesigning for it.

MoE weights ship in **MXFP4**, which is what lets the 120b fit on a single 80GB accelerator — a
memory strategy operating orthogonally to attention. See also
[Extending Puzzle for MoE reasoning models / GPT-OSS acceleration, 2602.11937](https://arxiv.org/pdf/2602.11937).

**GLM-4.5 / GLM-4.6** (Z.ai, Aug–Oct 2025) — [arXiv 2508.06471](https://arxiv.org/pdf/2508.06471)
**GQA + partial RoPE + QK-Norm**, and a deliberate choice to use **2.5× more attention heads**
(96 heads at 5120 hidden). Notable finding: extra heads *don't improve training loss* but
consistently improve MMLU/BBH reasoning. MoE layer as MTP layer for speculative decoding.

**Kimi K2 / K2.5** (2025) — MLA, 1T/32B MoE; K2.5 with 384 experts and 256K context. Pure MLA, no
linear hybrid yet.

**Gemma 3** (Mar 2025) — [arXiv 2503.19786](https://arxiv.org/pdf/2503.19786)
**5:1 local:global** with the SWA window *reduced* to 1024 (Gemma 2 was 1:1 at 4096). Ablation:
5:1 costs almost nothing in perplexity vs 1:1.

**Llama 4 Scout/Maverick** (Meta, Apr 2025) — [Meta blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) ·
[architecture breakdown](https://medium.com/@mandeep0405/llama-4s-architecture-deconstructed-moe-irope-and-early-fusion-explained-e58eb9403067)
**iRoPE**: 3 RoPE layers : 1 **NoPE** layer, plus **inference-time attention temperature scaling**
for length generalization. RoPE layers decay with distance; NoPE layers attend uniformly,
providing the global channel. Honest caveat: no variant was trained above 256K, so the 10M figure
is extrapolation and quality degrades past ~256K.

**MiniMax-01 / Text-01** (Jan 2025) — [arXiv 2501.08313](https://arxiv.org/abs/2501.08313)
456B/45.9B. **Lightning Attention** (I/O-aware linear attention) at **7 linear : 1 softmax** per
8-layer block. 1M trained, 4M extrapolated.

**Others in 2025:** Falcon-H1 ([2507.22448](https://arxiv.org/html/2507.22448v1)) with attention and
Mamba-2 heads **in parallel** inside one mixer block, ratio independently tunable — distinct from
the sequential interleave everyone else uses; Nemotron-H replacing ~92% of attention layers with
Mamba-2 for ~3× throughput ([NVIDIA ADLR](https://research.nvidia.com/labs/adlr/nemotronh/));
IBM Granite 4.0; Qwen2.5-1M ([2501.15383](https://arxiv.org/pdf/2501.15383)).

### 2024 and earlier (production)

- **DeepSeek-V2** (May 2024) — [arXiv 2405.04434](https://arxiv.org/pdf/2405.04434) — **MLA**,
  low-rank joint KV compression. The ablation that mattered: GQA looked *worse* than MHA on
  quality, MLA held up and could beat MHA. This single design propagated to Kimi, GLM, and most
  Chinese frontier labs. See also [Raschka's MLA page](https://sebastianraschka.com/llm-architecture-gallery/mla/).
- **Jamba** (AI21, Mar 2024) — [arXiv 2403.19887](https://arxiv.org/pdf/2403.19887) — first
  large-scale Transformer–Mamba–MoE hybrid. [AI21's retrospective](https://www.ai21.com/blog/rise-of-hybrid-llms/).
- **Gemma 2** (2024) — 1:1 local:global SWA at 4096.
- **Mistral 7B** (Oct 2023) — SWA + rolling buffer cache.
- **PaLM** (2022) — MQA at scale; **GQA** (2023) as the standard middle ground.

**Closed models — unconfirmed.** Gemini 3, GPT-5/5.5, Grok 4.5, and Claude Opus 5 publish no
architecture. Reported *context* windows (Grok 4.5 ~500K; Claude Opus 5 and Gemini 3.6 Flash ~1M)
are documented; claims like "GPT-5 uses GQA + sliding window, 40% faster" or "GPT-6 uses
hierarchical sparse attention" come from speculation blogs with no primary source. Treat as rumor.

---

## Part 3 — Research literature, reverse chronological

### 2026

**Hybrid architecture design & Transformer→hybrid conversion** — the single hottest subfield:

- [Morphing into Hybrid Attention Models (FlashMorph), 2606.30562](https://arxiv.org/abs/2606.30562)
  — casts hybrid *layer selection* as budget-constrained subset optimization; shows heuristic
  fixed-pattern placement ignores interdependent layer effects
- [Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation, 2606.16429](https://arxiv.org/html/2606.16429v1)
  — up to 88× zero-shot improvement, 4.9–9.2× fewer tokens to recovery
- [Rethinking the Role of Efficient Attention in Hybrid Architectures, 2606.15378](https://arxiv.org/abs/2606.15378)
- [Attention Amnesia in Hybrid LLMs: When CoT Fine-Tuning Breaks Long-Range Recall, 2606.11052](https://arxiv.org/pdf/2606.11052)
- [Mixture of Layers with Hybrid Attention, 2605.09516](https://arxiv.org/abs/2605.09516)
- [DASH: Differentiable Architecture Search for Hybrid Attention, 2605.20936](https://arxiv.org/pdf/2605.20936)
- [Attention Editing: Cross-Architecture Attention Conversion, 2604.05688](https://arxiv.org/pdf/2604.05688)
- [HubRouter: Pluggable Sub-Quadratic Routing Primitive, 2604.22442](https://arxiv.org/pdf/2604.22442)
- [Effective Distillation to Hybrid xLSTM, 2603.15590](https://arxiv.org/pdf/2603.15590)
- [Hybrid Linear Attention Done Right (HALO), 2601.22156](https://arxiv.org/abs/2601.22156)
- [Distill-then-Replace: Task-Specific Hybrid Construction, 2601.11667](https://arxiv.org/pdf/2601.11667)
- [Gated DeltaNet-2: Decoupling Erase and Write, 2605.22791](https://arxiv.org/pdf/2605.22791) —
  separate channel-wise gates for erase vs. write; beats Mamba-2, GDN, KDA, and Mamba-3 at 1.3B
- [Exact Linear Attention, 2605.18848](https://arxiv.org/html/2605.18848v1)

**Sparse attention — training and serving:**

- [SpotAttention: Plug-In Block-Sparse Routing, 2606.22874](https://arxiv.org/pdf/2606.22874)
- [ConSA: Controllable Sparsity in Hybrid Attention via Learnable Allocation, 2606.18056](https://arxiv.org/pdf/2606.18056)
- [How Much Dense Attention is Necessary? Oracle-Guided Sparse Prefill, 2606.07703](https://arxiv.org/pdf/2606.07703)
- [SparDA: Sparse Decoupled Attention, 2606.04511](https://arxiv.org/abs/2606.04511) — adds a
  **fourth projection, "Forecast"**, alongside Q/K/V to predict the *next* layer's needed KV
  blocks, overlapping prefetch with current-layer compute
- [Unifying Sparse Attention with Hierarchical Memory, 2604.26837](https://arxiv.org/pdf/2604.26837)
- [SparseBalance: Load-Balanced Long Context Training, 2604.13847](https://arxiv.org/abs/2604.13847)
  — bidirectional sparsity adjustment to kill stragglers in distributed sparse training
- [Flux Attention: Context-Aware Hybrid Attention, 2604.07394](https://arxiv.org/pdf/2604.07394)
- [AsyncTLS: Asynchronous Two-level Sparse Attention, 2604.07815](https://arxiv.org/pdf/2604.07815)
- [HISA: Hierarchical Indexing for Fine-Grained Sparse Attention, 2603.28458](https://arxiv.org/pdf/2603.28458)
- [IndexCache: Cross-Layer Index Reuse, 2603.12201](https://arxiv.org/pdf/2603.12201) — the
  research analogue of GLM-5.2's IndexShare
- [HySparse: Oracle Token Selection + KV Cache Sharing, 2602.03560](https://arxiv.org/pdf/2602.03560)
- [Punctuation-aware Hybrid Trainable Sparse Attention, 2601.02819](https://arxiv.org/pdf/2601.02819)
- [DashAttention: Differentiable Adaptive Sparse Hierarchical Attention, 2605.18753](https://arxiv.org/pdf/2605.18753)

**KV cache, compression, memory:**

- [End-to-End Context Compression at Scale, 2606.09659](https://arxiv.org/pdf/2606.09659)
- [Fast KV Compaction via Attention Matching, 2602.16284](https://arxiv.org/pdf/2602.16284)
- [HeteroCache: Heterogeneous KV Cache Compression, 2601.13684](https://arxiv.org/html/2601.13684v1)
- [GQLA: Group-Query Latent Attention, 2605.15250](https://arxiv.org/abs/2605.15250) — GQA/MLA
  interpolation for hardware-adaptive decode
- [Top-10 KV Cache Compression Techniques survey (Apr 2026)](https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/)

**Theory / limits / serving:**

- [The Impossibility Triangle of Long-Context Modeling, 2605.05066](https://arxiv.org/pdf/2605.05066)
- [DUET: Disaggregated Hybrid Mamba-Transformer Serving, 2603.15530](https://arxiv.org/html/2603.15530v1)
- [Mamba-3, 2603.15569](https://arxiv.org/abs/2603.15569)

> **Context:** over **150 papers with "sparse attention" in the title** were posted to arXiv
> between Jan 2025 and Jan 2026. The above is the load-bearing subset, not the full set.
> Raschka's [2026 papers list (Jan–May)](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)
> is the best running index.

### 2025

- [Kimi Linear / KDA, 2510.26692](https://arxiv.org/abs/2510.26692)
- [DeepSeek-V3.2, 2512.02556](https://arxiv.org/html/2512.02556v1)
- [NSA: Native Sparse Attention, 2502.11089](https://arxiv.org/abs/2502.11089) — **ACL 2025 Best
  Paper** ([ACL version](https://aclanthology.org/2025.acl-long.1126/)). Three parallel branches:
  compressed (coarse), selected (fine top-k blocks), sliding (local). Arithmetic-intensity-balanced
  kernels; end-to-end **trainable**, unlike the training-free sparsity that preceded it. The direct
  ancestor of DSA, MSA, and InfLLM-v2.
- **MoBA: Mixture of Block Attention for Long-Context LLMs** (Moonshot AI, 18 Feb 2025) —
  [arXiv 2502.13189](https://arxiv.org/abs/2502.13189) · [code](https://github.com/MoonshotAI/MoBA).
  **Applies MoE routing to attention itself**: partition the context into blocks, and let a top-k
  *gate* decide which blocks each query attends to. Explicitly follows a **"less structure"
  principle** — the model decides where to attend, rather than inheriting a predefined sparsity
  pattern (the philosophical opposite of sliding-window designs). Key properties:
  - **Seamless full↔sparse switching** — the same weights run either mode, so sparsity carries no
    commitment risk.
  - **Prefill-only in practice**: MoBA is used for prefill and the model **switches to full
    attention during generation** for better quality. A sharp contrast with NSA/DSA/MSA, which
    target prefill *and* decode.
  - Recipe: from Llama 3.1 8B Base, continual pretraining **128K → 256K → 512K → 1M**, then MoBA
    activated for a further 100B tokens. Llama-8B-1M-MoBA tracks Llama-8B-1M-Full closely.
  - **Deployed in production to serve Kimi's long-context requests.**

  Note the timing: NSA (16 Feb) and MoBA (18 Feb) landed **two days apart** — DeepSeek and Moonshot
  published competing trainable block-sparse attention designs in the same week, which is roughly
  the moment the field pivoted from training-free to trained sparsity.
- [MiniMax-01 / Lightning Attention, 2501.08313](https://arxiv.org/abs/2501.08313)
- [Titans: Learning to Memorize at Test Time, 2501.00663](https://arxiv.org/abs/2501.00663) —
  three-tier memory (attention short-term, surprise-driven neural long-term, static persistent);
  2M+ tokens
- [MiniCPM4 / InfLLM-v2, 2506.07900](https://arxiv.org/pdf/2506.07900) — trainable sparse attention
  covering **both prefill and decode**
- [The Sparse Frontier: Sparse Attention Trade-offs, 2504.17768](https://arxiv.org/pdf/2504.17768)
  — the key empirical reality-check paper
- [Hybrid Architectures for Language Models: Systematic Analysis, 2510.04800](https://arxiv.org/html/2510.04800v3)
- [A Systematic Analysis of Hybrid Linear Attention, 2507.06457](https://arxiv.org/html/2507.06457v1)
  — generations of gating: RetNet (fixed) → GLA (data-dependent) → Mamba-2 (scalar gate) →
  DeltaNet family (delta-rule forgetting)
- [Untangling Component Imbalance in Hybrid Conversion, 2510.05901](https://arxiv.org/html/2510.05901v2)
  — SWA dominates, the linear-attention branch often contributes almost nothing
- [MTraining: Distributed Dynamic Sparse Attention, 2510.18830](https://arxiv.org/abs/2510.18830)
- **[TransMLA, 2502.07864](https://arxiv.org/abs/2502.07864)** (NeurIPS 2025 Spotlight) and
  **[MHA2MLA, 2502.14837](https://arxiv.org/abs/2502.14837)** (ACL 2025) — two independent GQA→MLA
  converters published 8 days apart. **Walked through in detail in Part 5's
  [GQA→MLA conversion cluster](#the-gqamla-conversion-cluster--how-transmla-works)**, along with
  MHA2MLA-VLM, GQLA, YouZhi, and the speculative-decoding audit (2607.27269).
- [FlexPrefill, 2502.20766](https://arxiv.org/pdf/2502.20766) ·
  [ProxyAttn, 2509.24745](https://arxiv.org/pdf/2509.24745) ·
  [Short window attention enables long-term memorization, 2509.24552](https://arxiv.org/pdf/2509.24552)
- [KVzip, 2505.23416](https://arxiv.org/pdf/2505.23416) ·
  [RocketKV, 2502.14051](https://arxiv.org/html/2502.14051v1) ·
  [AttentionPredictor, 2502.04077](https://arxiv.org/pdf/2502.04077)
- [Enhancing Linear Attention with Residual Learning, 2509.25223](https://arxiv.org/pdf/2509.25223)
- [SeerAttention-R: Sparse Attention Adaptation for Long Reasoning/Decoding, 2506.08889](https://arxiv.org/pdf/2506.08889)
- **Surveys:** [Thus Spake Long-Context LLM, 2502.17129](https://arxiv.org/pdf/2502.17129) ·
  [A Comprehensive Survey on Long Context Language Modeling, 2503.17407](https://arxiv.org/pdf/2503.17407) ·
  [Efficient Attention Mechanisms for LLMs: A Survey, 2507.19595](https://arxiv.org/pdf/2507.19595)
- [Gemma 3, 2503.19786](https://arxiv.org/pdf/2503.19786) ·
  [GLM-4.5, 2508.06471](https://arxiv.org/pdf/2508.06471) ·
  [Falcon-H1, 2507.22448](https://arxiv.org/html/2507.22448v1)

### 2024

- **SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs** (Microsoft, Oct 2024) —
  [arXiv 2410.13276](https://arxiv.org/abs/2410.13276) · [code](https://github.com/microsoft/SeerAttention).
  The earliest member of the **learned block-gate** family, predating NSA and MoBA by four months.
  Explicitly MoE-inspired: augments attention with a **learnable gate** that pools Q and K along the
  sequence dimension, passes them through linear layers, and multiplies to produce block-level
  gating scores. Trained by **self-distillation at post-training — only the gate parameters are
  learned**, which is why it converges fast. Paired with a block-sparse FlashAttention kernel.
- **InfLLM** (v1, Feb 2024) — arXiv 2402.04617 — training-free memory-based long context; the
  ancestor of the InfLLM-v2 used in MiniCPM4 and MiniCPM-SALA.
- **Gated Delta Networks: Improving Mamba2 with Delta Rule** — [arXiv 2412.06464](https://arxiv.org/abs/2412.06464).
  The pivotal paper: gating (adaptive memory control) and the delta rule (precise memory
  correction) are **complementary**, not redundant. Everything Qwen3-Next/3.5/3.6 and Kimi KDA
  does descends from this.
- **DeepSeek-V2 / MLA** — [arXiv 2405.04434](https://arxiv.org/pdf/2405.04434)
- **Mamba-2** — arXiv 2405.21060 (SSD duality: SSMs ≡ structured masked attention)
- **Jamba** — [arXiv 2403.19887](https://arxiv.org/pdf/2403.19887)
- **Infini-attention** (Google) — arXiv 2404.07143 — local softmax within a chunk + compressive
  memory updated by linear attention
- **LongRoPE** (Microsoft) — arXiv 2402.13753 — per-dimension *search* over RoPE scaling factors
  instead of a formula
- **DuoAttention** — arXiv 2410.10819 — splits heads into **retrieval heads** (full KV) vs
  **streaming heads** (rolling cache)
- **SnapKV** — arXiv 2404.14469 · **MInference** — arXiv 2407.02490 · **Quest** — arXiv 2406.10774
- **DIFF Transformer** — arXiv 2410.05258 — differential softmax to cancel attention noise
- **YOCO (You Only Cache Once)** — arXiv 2405.05254 · **Star Attention** — arXiv 2411.17116
- **xLSTM** — arXiv 2405.04517
- **Length Generalization of Causal Transformers without Position Encoding** —
  [arXiv 2404.12224](https://arxiv.org/abs/2404.12224) — links NoPE's failure to *attention
  distribution distraction*; fixes it by tuning per-head softmax temperature
- Cross-layer KV sharing / MLKV (Character.AI-style production KV reduction)

### 2023

- **YaRN** — [arXiv 2309.00071](https://arxiv.org/html/2309.00071v3)
  ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/874a4d89f2d04b4bcf9a2c19545cf040-Paper-Conference.pdf))
  — piecewise per-frequency-band NTK scaling + **attention temperature scaling**; 8–32× extension
  on ~0.1% of pretraining data
- **Position Interpolation** (Meta) — arXiv 2306.15595 — LLaMA to 32K in <1000 steps
- **The Impact of Positional Encoding on Length Generalization (NoPE)** —
  [arXiv 2305.19466](https://arxiv.org/abs/2305.19466), NeurIPS 2023 — ALiBi, RoPE, and APE are
  all *worse* than no positional encoding for downstream length generalization
- **GQA** — arXiv 2305.13245
- **Mamba** — arXiv 2312.00752 · **RetNet** — arXiv 2307.08621 · **Hyena** — arXiv 2302.10866
- **StreamingLLM / attention sinks** — arXiv 2309.17453 — the observation that a few initial
  tokens absorb enormous attention mass; keep them + a rolling window
- **H2O (Heavy-Hitter Oracle)** — arXiv 2306.14048
- **Ring Attention with Blockwise Transformers** — arXiv 2310.01889 — sequence parallelism scaling
  linearly with devices; enables >100M-token sequences
- **FlashAttention-2** — arXiv 2307.08691 · **LongNet (dilated attention)** — arXiv 2307.02486
- **LongLoRA** — arXiv 2309.12307 · **Landmark Attention** — arXiv 2305.16300 ·
  **Focused Transformer / LongLLaMA** — arXiv 2307.03170
- **Lost in the Middle** — arXiv 2307.03172 — the paper that made "long context ≠ usable context"
  undeniable
- **Mistral 7B (SWA)** — arXiv 2310.06825 · **Scaling Transformer to 1M tokens (RMT)** — arXiv 2304.11062

### 2022

- **FlashAttention** — arXiv 2205.14135 — the enabling systems result; exact attention, IO-aware
  tiling, linear memory
- **Memorizing Transformers** — arXiv 2203.08913 — kNN lookup into a non-differentiable external
  memory
- **Block-Recurrent Transformers** — arXiv 2203.07852 · **Recurrent Memory Transformer** — arXiv 2207.06881
- **H3 (Hungry Hungry Hippos)** — arXiv 2212.14052 — SSMs become competitive on language

### 2021

- **RoPE / RoFormer** — arXiv 2104.09864 — the positional encoding essentially every model on this
  list uses
- **ALiBi** — arXiv 2108.12409 — train short, test long via a linear attention-score bias
- **S4** — arXiv 2111.00396 — the structured state space foundation
- **Performer (FAVOR+)** — [arXiv 2009.14794](https://arxiv.org/pdf/2009.14794), ICLR 2021 —
  random-feature kernel approximation of softmax
- **Random Feature Attention** — arXiv 2103.02143 · **Nyströmformer** — arXiv 2102.03902
- **Fastformer (additive attention)** — [arXiv 2108.09084](https://arxiv.org/pdf/2108.09084) ·
  **Smart Bird (learnable sparsity)** — [arXiv 2108.09193](https://arxiv.org/pdf/2108.09193)
- **Long Range Arena** — arXiv 2011.04006 — the benchmark that disciplined the whole
  efficient-attention wave

### 2020 — the original efficient-attention explosion

- **Longformer** — [arXiv 2004.05150](https://arxiv.org/abs/2004.05150) — sliding window + dilated
  window + task-motivated global tokens, O(n)
- **Big Bird** — [arXiv 2007.14062](https://arxiv.org/abs/2007.14062) — random + window + global;
  proves the sparse pattern is a universal approximator and Turing complete
- **Reformer** — [arXiv 2001.04451](https://arxiv.org/abs/2001.04451) — LSH attention, O(n log n),
  reversible layers
- **Linformer** — [arXiv 2006.04768](https://arxiv.org/abs/2006.04768) — low-rank projection of K/V
  to fixed length
- **Transformers are RNNs / Linear Transformer** — [arXiv 2006.16236](https://arxiv.org/abs/2006.16236)
  — the kernel-trick reformulation that *all* modern linear attention (Lightning, GDN, KDA) is
  built on
- **Efficient Transformers: A Survey** — [arXiv 2009.06732](https://arxiv.org/abs/2009.06732)
- **ETC** — arXiv 2004.08483 · **Routing Transformer** — arXiv 2003.05997 ·
  **Sinkhorn Attention** — arXiv 2002.11296 ·
  **Clustered Attention** — [arXiv 2007.04825](https://arxiv.org/abs/2007.04825)
- Predecessors carried into 2020: **Sparse Transformer** (1904.10509),
  **Compressive Transformer** (1911.05507, ICLR 2020), **MQA** (1911.02150)
- Useful period retrospective:
  [HuggingFace Reads, Feb 2021 — Long-range Transformers](https://huggingface.co/blog/long-range-transformers)

---

## Part 4 — Three things worth flagging

**1. The 2020–2021 approximations mostly lost; the 2025–2026 ones are winning.**
Linformer/Performer/Reformer-class methods didn't survive contact with scale — approximating
softmax degraded quality. What works now is different in kind: *exact* attention over a *learned*
subset (NSA/DSA/MSA), or *fixed-state recurrence trained from scratch* (KDA/GDN/Mamba). The lesson
repeated in the DeepSeek-V3.2 recipe: sparsity has to be **trained in**, gradually, not bolted on
at inference.

**2. There is no consensus.** As of mid-2026: DeepSeek and Z.ai are all-in on sparse attention over
MLA; Alibaba is all-in on linear (Gated DeltaNet); Moonshot runs both (KDA + Gated MLA); MiniMax
went linear → dense → sparse; Thinking Machines reached 1M with *only* static sliding windows and a
learned relative bias; Xiaomi and OpenAI use plain sliding-window at 128 tokens. Labonne's title —
*"Nobody Agrees on Attention Anymore"* — is the accurate summary of the state of the art.

**3. Advertised context ≠ usable context.** Llama 4's 10M was never trained past 256K. GLM-5.2's
IndexShare gives a 2.9× FLOP reduction that is *not* 2.9× latency and does *not* cut KV memory
proportionally. Read *Lost in the Middle* (2023), *The Sparse Frontier* (2025), and *The
Impossibility Triangle of Long-Context Modeling* (2026) alongside any headline number.

---

## Part 5 — Per-model breakdown: what actually buys the long context

The central observation: **no frontier model gets long context from its attention mechanism alone.**
Every one of them pulls five orthogonal levers together, and the mechanism is only the first.

| Lever | What it controls | Failure if you skip it |
|---|---|---|
| **L1 — Mixer structure** | sparse / linear / hybrid ratio | quadratic cost |
| **L2 — KV footprint** | bytes stored per token per layer | memory-bound decode, tiny batch |
| **L3 — Positional scheme** | behaviour past the trained length | garbage past native window |
| **L4 — Training curriculum** | how long-context + sparse skill is *acquired* | selector never learns; recall collapses |
| **L5 — Kernels & serving** | whether the FLOP saving is realizable | FLOP win ≠ latency win |

### 2026 frontier

**Kimi K3** — 2.8T params, 1M context · [arXiv 2607.24653](https://arxiv.org/pdf/2607.24653) · [tech blog](https://www.kimi.com/blog/kimi-k3)
- **L1** 3× KDA (linear) : 1× Gated MLA per block.
- **L2** KDA layers hold a *fixed-size recurrent state*, so 75% of layers have zero KV growth; the
  remaining MLA layers use latent compression. Net −75% KV vs full-MLA K2.
- **L3** **NoPE on all MLA layers** — no RoPE anywhere. KDA's per-channel decay and short conv
  carry locality, so the global layers are left position-free.
- **L4** Four-stage curriculum **8K → 64K → 256K → 1M**, using synthetic multimodal tasks whose
  evidence is *deliberately scattered across the full sequence*; per-head Muon, cosine with 1%
  warmup. Post-training explicitly targets 1M test-time scaling.
- **L5** DPLR chunkwise KDA kernel; prefix caching reworked for recurrent state (KDA breaks
  conventional prefix caching). ~6.3× faster long-context decode vs K2.
- **Lineage note:** K3's predecessor in Kimi *serving* was **MoBA** (block-sparse routing,
  prefill-only) — see the lineage table below. Moonshot moved from learned block routing to a fixed
  recurrent state; K3 retains no MoBA component.
- **Beyond attention:** **Attention Residuals / Block AttnRes**
  ([arXiv 2603.15031](https://arxiv.org/abs/2603.15031)) — a *depth*-dimension mechanism letting
  layers attend over block-level representations of earlier layers instead of relying on sequential
  residual accumulation. K3 frames scaling along three axes: sequence (hybrid attention), depth
  (AttnRes), width (Stable LatentMoE).

**Inkling** — 975B/41B, 1M context · [HF](https://huggingface.co/blog/thinkingmachines-inkling) · [Raschka teardown](https://sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html)
- **L1** 55 SWA (512-token window) : 11 global, 5:1, final layer global. **Static sparsity only** —
  no top-k selector, no linear attention, no MLA. Plus **SConv**: 4 causal convs per layer, kernel
  size 4, placed after the K/V projections and after the attention/MoE outputs, taking local mixing
  off attention entirely.
- **L2** **Asymmetric GQA by layer role** — local layers 64 Q / 16 KV (4:1), global layers 64 Q /
  8 KV (8:1). The 11 expensive global layers carry half the KV heads of the cheap local ones. This
  is the quiet trick: 55 of 66 layers are window-bounded, and the 11 that aren't are the most
  KV-frugal. No compression scheme needed.
- **L3** **Learned, input-dependent relative-position bias** from query and key states, added to
  pre-softmax logits. Span is 512 tokens on local layers, 1,024 on global layers; **past that span
  there is no positional term at all.** So: relative bias in-window, NoPE-like at range. A third
  distinct answer to the position question, alongside Kimi's NoPE and MiniCPM's HyPE.
- **L4/L5** 45T tokens, multimodal (text/image/video/audio). **MTP drafter layers** for speculative
  decoding. Inkling-Small (276B/12B) shares the architecture and ships MXFP8/NVFP4.
- **Why it matters:** Inkling is the strongest existing counterexample to the assumption that 1M
  context requires a learned selector or a recurrent state. It reaches 1,048,576 tokens with
  sliding windows, a finite-span relative bias, asymmetric GQA, and short convs — techniques all
  available in 2023. Read against DeepSeek-V4 and MSA, it is the live control experiment for
  whether learned sparsity is actually necessary at frontier scale.

**DeepSeek-V4** — 1M context · [arXiv 2606.19348](https://arxiv.org/html/2606.19348v1) · [Raschka: CSA & HCA](https://sebastianraschka.com/llm-architecture-gallery/csa-hca/)
- **L1** Alternating CSA / HCA layers. V4-**Flash**: first 2 layers pure SWA, rest alternate.
  V4-**Pro**: first 2 layers HCA, rest alternate.
  - **CSA** — 4× sequence-dim compression via *softmax-gated pooling with a learned positional
    bias*, then DSA top-k over the compressed entries.
  - **HCA** — 128× compression and **drops top-k entirely**, staying dense. At 1M that is ~7,800
    compressed entries attended densely.
- **L2** The standout number: **~2% of a standard transformer's KV cache at 1M**. V4 is the only
  model here attacking KV bytes *and* attention FLOPs equally hard.
- **L3/L4** Continued-training lineage from V3.2's DSA; each mechanism keeps a **128-token
  uncompressed recent window**.
- **Result** 97% NIAH at 1M — i.e. the retrieval-degradation problem addressed, not just the cost.

**MiniMax MSA** — 109B MoE, 3T-token budget · [arXiv 2606.13392](https://arxiv.org/html/2606.13392) · [paper review](https://andlukyane.com/blog/paper-review-minimax-sparse-attention)
- **L1** Two branches over **GQA** (not MLA): an *Index Branch* scores KV blocks and picks top-k
  **independently per GQA group**; the *Main Branch* does exact block-sparse attention over the
  selection.
- **L4** This is the most instructive part of the paper, because top-k is non-differentiable and
  the LM loss cannot train the index projections. Four devices:
  1. **KL alignment loss** — match Index Branch distribution to the Main Branch attention pattern.
  2. **Gradient detach** — stop-gradient on the Index Branch input, confining the KL loss to the
     index projections only.
  3. **Indexer warmup** — run *full* attention in both branches for the first **40B of 400B**
     tokens, so the indexer learns from KL before it controls routing.
  4. **Forced local block** — one selection slot permanently reserved for nearby context.
- **L5** Exp-free top-k kernel: 5.1× faster than `torch.topk` at 128K with k=16. Built on GQA
  deliberately so it deploys across a broad range of GPUs.

**GLM-5 / 5.1 / 5.2** — 744B/40B, 256 experts, 200K · [arXiv 2602.15763](https://www.emergentmind.com/papers/2602.15763) · [Raschka on IndexShare](https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html)
- **L1** MLA + DSA — content-based dynamic top-k on top of latent-compressed KV.
- **L2** MLA latent KV. Note IndexShare does **not** reduce this.
- **L3** RoPE (no NoPE experiment) — the sparse-attention camp has stayed on RoPE.
- **L4** Staged curriculum **32K → 200K** across ~28.5T pretraining tokens.
- **L5** **IndexShare** (5.2): run the DSA indexer fully only once per **4 layers**, reuse indices
  downstream. 2.9× per-token FLOPs at 1M — explicitly *not* 2.9× end-to-end, and KV memory does
  not fall proportionally. Compare the independent research analogue,
  [IndexCache (2603.12201)](https://arxiv.org/pdf/2603.12201).

**Qwen3.5 / Qwen3.6** — 397B/17B, 60 layers, 512 experts, 262K native · [Labonne](https://huggingface.co/blog/mlabonne/qwen35) · [spec sheet](https://www.morphllm.com/qwen-3-5)
- **L1** 3:1 Gated DeltaNet : full attention + sparse MoE.
- **L2** Only ~25% of layers carry a KV cache at all.
- **L3** Two-part: GDN layers use **causal Conv1D + gating in place of positional encoding**;
  full-attention layers use RoPE. Past the native window, **inference-time YaRN** stretches to
  ~1,010,000 — and the model card itself warns that *static YaRN degrades short-context quality*,
  so it should be enabled only when inputs exceed 262K. Lineage: Qwen2 combined
  **Dual Chunk Attention** with YaRN ([arXiv 2407.10671](https://arxiv.org/pdf/2407.10671)),
  continued in [Qwen2.5-1M](https://arxiv.org/pdf/2501.15383).
- **L4/L5** GDN components: delta-rule error-correcting writes, exponential gating for adaptive
  decay, causal Conv1D, L2-normed Q/K.

**MiniCPM-SALA** — 9B, 1M+ · [arXiv 2602.11761](https://arxiv.org/html/2602.11761v2)
- **L1** Interleaved **25% InfLLM-V2 (sparse) : 75% Lightning Attention (linear)** — the only model
  here hybridizing *sparse with linear* rather than either against full attention.
- **L3** **HyPE** — RoPE is applied to the **linear** attention layers, to keep position-sensitive
  memory and preserve relative token order across the global context. This is the *inverse* of
  Kimi's choice, and the two are directly contradictory (see finding C below).
- **L4** Both components chosen for *conversion-friendliness*: InfLLM-V2 adds **no new parameters**
  and can switch dense↔sparse, so it inherits dense weights for stable initialization; Lightning
  Attention was picked for functional proximity to softmax attention, easing parameter adaptation
  and preserving pretrained knowledge.

**MiMo-V2-Flash** — 309B/15B · [arXiv 2601.02780](https://arxiv.org/abs/2601.02780)
- **L1** Static sparsity: **8 Hybrid Blocks, each = 5 SWA blocks + 1 Global Attention block** — a
  128-token window at 5:1. No learned *selector* and no linear layers, but see L3.
- **L2** **~6× KV-cache reduction** from the window structure alone.
- **L3** A **learnable attention sink bias** — the same device as gpt-oss — which Xiaomi credits for
  holding long-context quality up under a 128-token window. So: static sparsity **plus a learned
  sink**, not purely static.
- **L4** Native 32K → extended to **256K**; 27T tokens; MTP; Multi-Teacher On-Policy Distillation.

### 2025 frontier (for lineage)

| Model | L1 mixer | L2 KV | L3 position | L4 how long ctx acquired |
|---|---|---|---|---|
| **Kimi Linear** 48B/3B | 3 KDA : 1 MLA | −75% vs MLA | NoPE | trained with recipe matched to MLA baseline; 1M |
| **DeepSeek-V3.2** | MLA + DSA (lightning indexer + top-k) | MLA latent | RoPE | **two-stage continued training** from V3.1-Terminus — dense warmup, then sparse |
| **Qwen3-Next** 80B/3B | 3 GDN : 1 Gated Attention | 25% of layers | conv/gating + RoPE | MTP, ultra-sparse MoE |
| **GLM-4.5/4.6** | GQA, 96 heads | GQA groups | **partial RoPE** + QK-Norm | MoE-as-MTP layer for spec decoding |
| **gpt-oss-120b / 20b** | dense ↔ banded sparse (SWA **128**), 1:1 | GQA 64 heads / group 8, head dim 64 | RoPE + **YaRN** to 131,072 | **learned per-head sink bias in the softmax denominator** — what makes a 128-token window survivable; MXFP4 MoE weights for single-GPU fit. Extends a conventional window; does not attempt 1M |
| **Gemma 3** | 5:1 local:global, window 1024 | window-bounded on 5/6 layers | RoPE base raised on *global* layers only, left low on local | 5:1 ablated as near-free vs Gemma 2's 1:1 |
| **Llama 4 Scout** | **iRoPE** 3 RoPE : 1 NoPE | GQA | NoPE layers give the distance-invariant channel | **inference-time attention temperature scaling**; never trained past 256K, so 10M is extrapolation |
| **MiniMax-01** 456B/45.9B | 7 Lightning : 1 softmax per 8-layer block | 1/8 of layers | — | 1M trained → 4M extrapolated |

### The learned block-gate lineage — the through-line of 2024→2026

Nearly every production sparse-attention mechanism in this document is one idea, refined: **treat
attention blocks as experts and route to them with a learned top-k gate.** Tracing it explicitly,
because the family resemblance is much stronger than the papers' varied naming suggests:

| When | Work | What it added |
|---|---|---|
| Oct 2024 | **SeerAttention** ([2410.13276](https://arxiv.org/abs/2410.13276)) | The MoE-style **learnable block gate** (pool Q/K → linear → multiply → block scores). Trained by self-distillation, **gate parameters only**. |
| Feb 2024 | **InfLLM** (2402.04617) | Training-free block/memory retrieval — the pattern before it was learned. |
| 16 Feb 2025 | **NSA** ([2502.11089](https://arxiv.org/abs/2502.11089)) | Made it **natively trainable end-to-end** and hardware-aligned; three branches (compressed / selected / sliding). |
| 18 Feb 2025 | **MoBA** ([2502.13189](https://arxiv.org/abs/2502.13189)) | Made the MoE analogy literal and total — **"less structure"**, model chooses blocks; **full↔sparse switchable**; prefill-only, full attention at decode. Shipped in Kimi serving. |
| Jun 2025 | **InfLLM-v2** ([MiniCPM4, 2506.07900](https://arxiv.org/pdf/2506.07900)) | **Zero added parameters**, top-k block indices **shared across each query group**; dense↔sparse switchable, so it inherits dense weights. Prefill *and* decode. |
| Sep–Dec 2025 | **DSA** ([2512.02556](https://arxiv.org/html/2512.02556v1)) | Split the gate into a separate **lightning indexer** (few heads, FP8) + fine-grained token selection; introduced by **two-stage continued training**. |
| Jun 2026 | **MSA** ([2606.13392](https://arxiv.org/html/2606.13392)) | Gate is a full **Index Branch** with **per-GQA-group independent top-k**; publishes the training recipe (KL alignment, gradient detach, indexer warmup, forced local block). |
| Jun 2026 | **CSA** ([DeepSeek-V4, 2606.19348](https://arxiv.org/html/2606.19348v1)) | Gate operates over **learned-compressed** KV entries (4×) rather than raw blocks. |
| Jun 2026 | **IndexShare** (GLM-5.2) / [IndexCache (2603.12201)](https://arxiv.org/pdf/2603.12201) | Stopped running the gate every layer — **reuse indices across 4 layers**. |

Two things fall out of reading it as one line rather than nine papers:

- **The gate's cost became the bottleneck the gate was meant to solve.** By 2026 the indexer is
  itself expensive enough at 1M tokens that IndexShare/IndexCache exist purely to amortize it. That
  is a predictable end-state for any per-layer learned router.
- **MoBA's prefill-only choice was the honest one and got quietly abandoned.** Moonshot shipped
  sparse prefill + *full attention at decode* because that's where quality held up. Everyone after
  pushed sparsity into decode too — and then had to invent warmups, KL losses, and gradient detach
  to make it train. Whether decode-time sparsity was worth that complexity is still not clearly
  answered by any published ablation I found.

**Moonshot's own production lineage is therefore two distinct bets, not one:** MoBA (2025, block-sparse
routing, prefill-only, served Kimi long-context) → KDA (Kimi Linear/K3, 2025–26, linear recurrent
state, 3:1 with Gated MLA). They abandoned learned block routing for a fixed-state recurrence. That
reversal by the team that wrote MoBA is one of the strongest available signals on the sparse-vs-linear
question.

### The GQA→MLA conversion cluster — how TransMLA works

A second retrofit lineage, parallel to the block-gate one above. Where those papers convert *dense →
sparse*, these convert *GQA → MLA*: same economics (nobody re-pretrains for a mixer change), different
target. The anchor is **TransMLA**.

#### The claim: GQA is already a crippled MLA

Take GQA with `h` query heads, `g` KV groups, head dim `d`, model dim `D`. KV projections are
`W^K, W^V ∈ ℝ^(gd×D)`, and at attention time each group's K/V is **replicated** `h/g` times to serve
its query heads. That replication is just a linear map — a block matrix `R ∈ ℝ^(hd×gd)` of stacked
identity blocks:

```
k_i = R_i · (W^K x)          # down-project to gd (cached), then up-project to hd
```

which is structurally identical to MLA:

```
c^KV = W^DKV x               # ℝ^(r_kv), down — this is what gets cached
k_i  = W^UK_i · c^KV         # ℝ^(hd),   up
```

with `r_kv = gd`. **The only difference: GQA's up-projection is a fixed replication matrix of zeros
and identities; MLA's is free and learnable.** Identical cache size, strictly larger function class.
The paper formalizes this as a hierarchy `GQA ≤ MLA_Factorized ≤ MQA` (their Appendix A). The
practical reading: a GQA checkpoint is leaving expressiveness on the table at *zero* memory cost, and
TransMLA is the procedure for claiming it.

#### The obstacle: RoPE blocks Absorb

MLA's speedup comes from **Absorb** — fold `W^UK` into the query projection and `W^UV` into the
output projection, then attend directly against the cached latent, so you cache `r_kv` dims instead
of `h·d`:

```
q̂_t,i = W^UK_i^T q_t,i      →      score = q̂_t,i^T · c_j^KV
```

RoPE breaks this. It sits between the down-projection and the score, is position-dependent, and does
not commute with an arbitrary cross-head linear map — so `W^UK` cannot be pushed through the
rotation. DeepSeek's native MLA sidesteps this *by design* with a small decoupled-RoPE slice and NoPE
elsewhere. A GQA checkpoint has no such structure: every head carries RoPE on every dim. The
conversion has to manufacture it after the fact.

#### RoRoPE — concentrate the positional signal, then delete it everywhere else

The key observation: RoPE acts on **dimension pairs** at frequency `θ_l`, and within a given frequency
slot the *same* rotation applies in every head. So an orthogonal `U_l ∈ ℝ^(g×g)` mixing across heads
*within* that slot passes straight through the rotation:

```
(U_l q)^R · (U_l k)^R = q^R · k^R        # exact — proven in their Appendix B
```

This is a **lossless reparameterization**: attention scores are unchanged. Choose `U_l` by PCA over
calibration key activations so all variance at frequency `l` piles into head 1's slot. Heads 2…g then
carry almost no positional energy and can drop RoPE at minimal cost. The output is exactly DeepSeek's
structure, reconstructed post hoc: **`K_rope`** (head 1, dim `d`, keeps RoPE, cached separately) and
**`K_nope`** (dims `(g−1)d`, RoPE-free, now compressible).

**FreqFold** extends this: adjacent RoPE frequencies are nearly equal (`θ_l ≈ θ_{l+1}`), so
neighbouring slots can be folded together, letting one head's dims absorb several frequency pairs.
This is what gets them past **90% RoPE removal**.

#### BKV — a rescaling so PCA doesn't eat the values

Now jointly PCA `[K_nope ; V]` into the shared latent. Problem: `‖K_nope‖₂ ≫ ‖V‖₂`, so principal
directions are dominated by keys and value information is crushed. Fix:

```
α = E_t‖W^DK_NoPE x_t‖₂ / E_t‖W^DV x_t‖₂
```

Divide the K down-projection by `α`, multiply the K up-projection by `α`. Algebraically a **no-op on
the function** — it changes only the conditioning of the decomposition, so PCA allocates rank fairly
between K and V. Cheap, exact, and the kind of trick that only shows up when someone actually runs
the decomposition.

#### Pipeline and results

Merge the `g` groups into one latent → RoRoPE + FreqFold on ~50–100 calibration samples → BKV rescale
→ PCA → initialize `W^UK`, `W^UV` to reproduce original outputs → light fine-tune.

| Model | KV reduction | Tokens | Avg score | Speedup |
|---|---|---|---|---|
| LLaMA-2 7B | 68.75% | **0** | 58.20 (orig 59.85) | — |
| LLaMA-2 7B | 92.97% | 6B | 58.68 | **10.6×** @ 8K ctx |
| SmolLM 1.7B | 68.75% | 300M | 55.24 (orig 55.90) | — |

**The zero-token row is the more interesting result**: 68.75% KV reduction with no training at all
and a 1.65-point drop means the conversion is close to structure-preserving — RoRoPE is lossless, so
only RoPE-dropping and PCA truncation cost anything. The headline 92.97% needs 6B tokens. Note the
10.6× is on constrained hardware (24GB, 165 TFLOPS) where KV cache is the binding constraint; expect
less where it isn't.

#### The rest of the cluster

| Date | Work | Approach |
|---|---|---|
| **12 Feb 2025** | **TransMLA** ([2502.07864](https://arxiv.org/abs/2502.07864), NeurIPS 2025 **Spotlight**, [code](https://github.com/MuLabPKU/TransMLA), [OpenReview](https://openreview.net/forum?id=TcVCu2PKb9)) | RoRoPE (lossless orthogonal PCA across heads) + FreqFold + BKV. Explicitly targets **full DeepSeek kernel compatibility** so converted models run the existing MLA serving path. |
| **20 Feb 2025** | **MHA2MLA** ([2502.14837](https://arxiv.org/abs/2502.14837), ACL 2025, [code](https://github.com/JT-Ushio/MHA2MLA)) | Independent, 8 days later. **Partial-RoPE**: drop RoPE from the query/key dims that *contribute least to attention scores* (a scoring criterion, vs TransMLA's lossless rotation). Then **joint SVD** of pretrained K and V. Recovers on **0.3–0.6% of data**; LLaMA2-7B KV **−92.19%** at 0.5% LongBench drop; composes with KV quantization. |
| Jan 2026 | **MHA2MLA-VLM** ([2601.11464](https://arxiv.org/abs/2601.11464)) | Extends it to vision-language models with **modality-adaptive partial-RoPE** and **modality-decoupled** low-rank approximation. |
| May 2026 | **GQLA** ([2605.15250](https://arxiv.org/abs/2605.15250)) | Group-Query Latent Attention — interpolates between GQA and MLA for hardware-adaptive decode rather than converting outright. |
| Jun 2026 | **YouZhi** ([2606.05868](https://arxiv.org/pdf/2606.05868)) | **Adaptive** GQA→MLA transition for high-concurrency serving (financial LLMs) — treats the conversion point as a deployment-time knob. |
| Jul 2026 | **Functional Reconstruction for MLA Draft Models** ([2607.27269](https://arxiv.org/pdf/2607.27269)) | The self-audit. Converted MLA models **sharply lose draft–target agreement in speculative decoding** — attention-function errors that standalone generation tolerates but that gut draft-token acceptance. Fix: optimize each converted module to **reproduce the post-output-projection response** of the original MHA/GQA on calibration hidden states, rather than reconstructing KV. Across 192 configurations (4 Llama/Qwen pairs, 2 converters): improved acceptance in **37 of 64** matched cells, unchanged in 26, worse in 1. |

Three observations on this cluster:

- **February 2025 produced two independent GQA→MLA converters eight days apart** — TransMLA (12th)
  and MHA2MLA (20th) — the same collision pattern as NSA (16th) and MoBA (18th) that same month.
  Both landed on the identical two-step recipe: *decide which RoPE dims matter, strip the rest,
  low-rank what remains.* They differ on how you decide: TransMLA rotates losslessly then measures,
  MHA2MLA scores dims by attention contribution. TransMLA reports **58.20 vs 37.90** against MHA2MLA
  at matched 68.75% reduction, on 4.9% of the tokens — a large enough gap that it deserves
  independent replication rather than acceptance at face value, since it is the winner reporting it.
- **The RoRoPE invariance is the transferable idea.** "Orthogonal mixing *within* a RoPE frequency
  slot leaves attention scores exactly unchanged" is a free reparameterization handle on any RoPE
  model — useful well beyond MLA conversion.
- **Conversion quality is task-dependent, and the field found that out late.** The 2607 paper
  (apparently from the TransMLA group itself) shows converted models can look fine on standalone
  benchmarks while being materially degraded as speculative-decoding drafts. Generalize the lesson:
  every retrofit in this document — dense→sparse, dense→hybrid, GQA→MLA — is validated on
  next-token benchmarks, and none of the papers except this one check whether the *function* was
  preserved well enough for downstream uses that depend on tight agreement with the original.

### Eight cross-cutting findings

**A. The mixer is never the whole answer.** Every model above is mixer + KV scheme + positional
scheme + staged curriculum + custom kernel. Reading only the attention-mechanism section of any of
these reports gives you maybe a third of why the model handles 1M tokens.

**B. Training the non-differentiable selector is the real engineering problem.** Top-k selection
has no gradient, so the LM loss cannot train the scorer. Every member of the learned block-gate
lineage above converged on the same three ingredients: **(i) a dense warmup or dense-weight
inheritance**, **(ii) an auxiliary loss aligning the selector to the dense attention distribution**,
**(iii) a permanently reserved local/recent slot.** MSA states all three explicitly (KL loss +
gradient detach + 40B-token indexer warmup + forced local block); DSA does it as two-stage continued
training from a dense checkpoint; SeerAttention does it by self-distilling **gate parameters only**;
InfLLM-V2 and MoBA do it by being **dense↔sparse switchable**, so sparse mode starts from working
dense weights. This is the single most transferable lesson in the whole set — and note that
switchability (MoBA, InfLLM-v2) is the cheapest of the three routes, since it needs no auxiliary
loss at all if you accept prefill-only sparsity.

**C. Positional handling is the least settled part of the whole stack — four live answers.**
1. **Keep RoPE, change nothing** — DeepSeek V3.2/V4, GLM-5.x. The sparse-attention camp.
2. **Strip it from the full-attention layers (NoPE)** — Kimi K3 on all MLA layers, Llama 4's iRoPE
   3:1. Argument: the linear/local layers' decay already carries position, so global layers should
   be distance-invariant.
3. **Add RoPE to the *linear* layers** — MiniCPM-SALA's **HyPE**, and independently **Ant Group's
   Ling 2.0 Linear**, which applies RoPE to the q/k inputs of its linear attention plus a grouped
   non-shared RMSNorm on the output. The exact inverse of (2): argues the linear layers need RoPE to
   preserve token order in global context. **Two independent labs, converging against Kimi.**
4. **Replace RoPE with a finite-span learned relative bias** — Inkling: input-dependent bias from
   Q/K over 512 tokens (local) / 1,024 (global), and *nothing* beyond that span.

Kimi and MiniCPM are in direct contradiction and both ship working 1M models. Inkling reaches 1M
with no rotary component at all. Meanwhile Qwen3.5 stays on RoPE+YaRN and ships a model card warning
that its own extension method degrades short-context quality. **Nobody has this figured out**, and
it is a more open question than the choice of mixer.

*Also on this axis:* [Randomized YaRN, 2606.23687](https://arxiv.org/pdf/2606.23687) ·
[NoPE gallery](https://sebastianraschka.com/llm-architecture-gallery/nope/)

**D. A local/recent uncompressed window is universal.** NSA's third branch, DeepSeek-V4's
128-token window on both CSA and HCA, MSA's forced local block, GDN's causal Conv1D, and the
SWA layers in Gemma/gpt-oss/MiMo. Not one design trusts its compressed or selected path with local
context. If you build one of these, the recent window is not optional.

**E. Curricula are staged, long, and increasingly synthetic.** 8K→64K→256K→1M (K3),
32K→200K over 28.5T tokens (GLM-5), 32K→256K (MiMo). K3's addition matters: synthetic tasks with
evidence *deliberately scattered across the full sequence*, because natural long documents don't
force long-range retrieval often enough to train it.

**F. Sparsity is retrofitted, not pretrained, almost everywhere.** DSA, MSA, InfLLM-V2, and V4's
CSA/HCA all arrive via continued training or conversion from a dense checkpoint. NSA is the notable
exception — natively trained sparse from scratch. The economics are obvious: nobody wants to risk a
multi-trillion-token pretrain on an unproven mixer.

**G. Static sparsity has not lost — it may be winning on cost-to-build.** The tacit assumption in
the 2025–2026 literature is that 1M context needs a learned selector (DSA/MSA/NSA) or a recurrent
state (KDA/GDN). **Inkling refutes that**: 1,048,576 tokens at 975B, using only sliding windows,
asymmetric GQA, short convs, and a finite-span relative bias — every ingredient available in 2023,
none requiring a KL-aligned indexer or a two-stage sparse curriculum. The **SWA + learned-sink-bias**
recipe specifically now spans three independent instances at three scales: **gpt-oss** (~117B, 131K),
**MiMo-V2-Flash** (309B, 256K, ~6× KV reduction), and Inkling's variant (975B, 1M, relative bias in
place of the sink). Set against DeepSeek-V4 and MSA, this is the live control experiment of the era:
*is learned sparsity actually necessary, or is it buying quality that well-placed static windows plus
a good positional scheme already deliver?* Watch the successors of these models — that's where the
answer lands.

**H. Two separable wins, routinely conflated.** *KV bytes* and *attention FLOPs* are different
resources. Linear layers eliminate KV growth but not full-layer FLOPs. Sparse selection cuts FLOPs
but the entire KV cache stays resident — this is exactly why GLM-5.2's IndexShare cuts FLOPs 2.9×
and memory hardly at all. Only *compression* (MLA, CSA, HCA) cuts both, and DeepSeek-V4 pushing to
2% of baseline KV at 1M is the strongest result on this axis in the set. When you read a headline
multiplier, check which resource it refers to and whether it is FLOPs or wall-clock.

---

## Part 6 — Per-lab sweep: every attention / long-context variant by publisher

Five labs produce most of the primary literature in this space. Read by lab rather than by year,
each one has a coherent (and in two cases self-reversing) thesis.

---

### 6.1 Moonshot AI / Kimi

**Thesis evolution: learned block routing → abandoned → linear recurrent state.**

| Date | Artifact | Attention / long-context content |
|---|---|---|
| Jan 2025 | **Kimi K1.5** ([2501.12599](https://arxiv.org/abs/2501.12599)) | RL scaling with long context; 128K. Long-context *RL* rather than an attention change. |
| Feb 2025 | **Muon is Scalable for LLM Training** ([2502.16982](https://arxiv.org/abs/2502.16982)) / Moonlight | Optimizer, not attention — but it's the optimizer that later enables K3's per-head Muon at 1M. |
| **18 Feb 2025** | **MoBA: Mixture of Block Attention** ([2502.13189](https://arxiv.org/abs/2502.13189), [code](https://github.com/MoonshotAI/MoBA)) | **MoE routing applied to attention blocks.** "Less structure" — model picks blocks, no predefined pattern. Full↔sparse switchable. **Prefill-only; full attention at decode.** Curriculum 128K→256K→512K→1M then 100B tokens with MoBA on. **Served Kimi long-context in production.** |
| Jul 2025 | **Kimi K2** ([2507.20534](https://arxiv.org/pdf/2507.20534)) | **MLA**, hidden 7168, **64 attention heads** (deliberately halved from DeepSeek-V3's 128 to cut inference cost), 384 experts (vs 256), 1T/32B, 128K context. No sparse or linear component. |
| Oct 2025 | **Kimi Linear** ([2510.26692](https://arxiv.org/abs/2510.26692)) | **KDA** — Gated DeltaNet + channel-wise gating via a specialized **DPLR** chunkwise kernel. 3 KDA : 1 MLA. NoPE. −75% KV, 6× decode at 1M. |
| Jan 2026 | **Kimi K2.5** | MLA retained, 384 experts, 256K, native vision. |
| Mar 2026 | **Attention Residuals** ([2603.15031](https://arxiv.org/abs/2603.15031)) | A **depth**-axis mechanism: layers attend over block-level representations of earlier layers instead of pure sequential residual accumulation. Block AttnRes is the scalable variant. |
| Jul 2026 | **Kimi K3** ([2607.24653](https://arxiv.org/pdf/2607.24653), [blog](https://www.kimi.com/blog/kimi-k3)) | 2.8T/41B-class, 1M. 3 KDA : 1 Gated MLA, **NoPE on all MLA layers**, Attention Residuals, LatentMoE, MXFP4 weights. Curriculum 8K→64K→256K→1M on synthetic scattered-evidence tasks. |

**What to take from Moonshot:** they authored the strongest learned-block-routing paper (MoBA), shipped
it in production, and then **built their next two flagships on something else entirely**. If learned
block sparsity were clearly winning, the team with a deployed implementation would not have switched
to a recurrent state. Their attention work also spans three orthogonal axes — sequence (KDA), depth
(AttnRes), width (LatentMoE) — which no other lab here treats as a unified frame.

---

### 6.2 DeepSeek

**Thesis evolution: compress the KV → learn the sparsity → compress *and* select. The most linear, least reversed trajectory of any lab.**

| Date | Artifact | Attention / long-context content |
|---|---|---|
| May 2024 | **DeepSeek-V2** ([2405.04434](https://arxiv.org/pdf/2405.04434)) | **MLA** — low-rank joint KV compression. The ablation showing GQA underperforms MHA while MLA matches or beats it is why half this list uses MLA. |
| Dec 2024 | **DeepSeek-V3** ([2412.19437](https://arxiv.org/abs/2412.19437)) | MLA + DeepSeekMoE + MTP at 671B/37B; FP8 training. Consolidation, not a new mechanism. |
| **16 Feb 2025** | **NSA** ([2502.11089](https://arxiv.org/abs/2502.11089), [ACL](https://aclanthology.org/2025.acl-long.1126/)) | **ACL 2025 Best Paper.** Three parallel branches — compressed / selected / sliding. First **natively trainable**, hardware-aligned sparse attention. Published 2 days before MoBA. |
| May 2025 | **Insights into DeepSeek-V3: Hardware Scaling Reflections** ([2505.09343](https://arxiv.org/abs/2505.09343)) | Explicit hardware-co-design argument for MLA and the KV-cache bottleneck; useful for *why* rather than *what*. |
| Sep 2025 | **DeepSeek-V3.2-Exp** ([vLLM day-0](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)) | **DSA** debut — **lightning indexer** (few heads, FP8-able) + fine-grained top-k, on top of MLA latents. ~3–6× cost at 128K. |
| Oct 2025 | **DeepSeek-OCR: Contexts Optical Compression** ([2510.18234](https://arxiv.org/abs/2510.18234)) | **A completely different long-context strategy: render text as images.** DeepEncoder + 3B-MoE-A570M decoder. **97% decoding precision at <10× compression, ~60% at 20×.** Explicitly framed as an initial investigation into optical 2D mapping as context compression. Belongs in any serious long-context list and is absent from most. |
| Dec 2025 | **DeepSeek-V3.2** ([2512.02556](https://arxiv.org/html/2512.02556v1)) | DSA productionized via **two-stage continued training** from V3.1-Terminus. The "sparsity must be learned gradually" result. |
| Jun 2026 | **DeepSeek-V4** ([2606.19348](https://arxiv.org/html/2606.19348v1), [Raschka](https://sebastianraschka.com/llm-architecture-gallery/csa-hca/)) | **CSA** (4× compress via softmax-gated pooling + learned positional bias, then top-k) alternating with **HCA** (128× compress, dense, no top-k). 128-token uncompressed window on both. ~**2% of baseline KV at 1M**; 97% NIAH. Plus mHC. |

**What to take from DeepSeek:** the only lab that never reversed. Each step strictly subsumes the
last — MLA compresses, NSA/DSA select over the compressed thing, CSA/HCA compress harder and select
over that. Also the only lab pursuing a genuinely orthogonal route (**optical compression**), which
sidesteps attention entirely by changing the tokenizer's modality.
[Raschka's V3→V3.2 tour](https://magazine.sebastianraschka.com/p/technical-deepseek) is the best
single walkthrough.

---

### 6.3 Qwen / Alibaba

**Thesis evolution: extend RoPE by any means → chunked/sparse inference tricks → commit the whole generation to linear attention.**

| Date | Artifact | Attention / long-context content |
|---|---|---|
| 2024 | **Qwen2** ([2407.10671](https://arxiv.org/pdf/2407.10671)) | **Dual Chunk Attention (DCA)** — segment long sequences into chunks, preserve relative position within *and across* chunks — combined with **YaRN** attention rescaling. Training-free extension. |
| Jan 2025 | **Qwen2.5-1M** ([2501.15383](https://arxiv.org/pdf/2501.15383), [blog](https://qwenlm.github.io/blog/qwen2.5-1m/)) | The 1M stack, all inference-side: **DCA + MInference sparse attention + NTK-aware interpolation + chunked prefill**. 3–7× speedup. Notable that 1M was reached with *no architecture change*. |
| May 2025 | **Gated Attention** ([2505.06708](https://arxiv.org/abs/2505.06708), NeurIPS 2025 oral, [code](https://github.com/qiuzh20/gated_attention)) | Head-specific **sigmoid gate after SDPA**. 30 variants ablated on 15B MoE + 1.7B dense over 3.5T tokens. Adds element-wise non-linearity and sparsity, improves training stability, tolerates larger LR — **and eliminates the attention sink**. This is the "Gated Attention" in Qwen3-Next's layer stack; a rare case of a lab publishing the ablation *before* shipping it. |
| May 2025 | **QwenLong-CPRS** ([2505.18092](https://arxiv.org/abs/2505.18092), [code](https://github.com/Tongyi-Zhiwen/QwenLong-CPRS)) | **Context compression, not attention**: query-aware multi-granularity compression with bidirectional reasoning layers, token critics, window-parallel inference. **21.59× compression, +19.15 points**; architecture-agnostic (wraps GPT-4o, Gemini, Claude, DeepSeek-V3). Explicitly beats both RAG and sparse attention on their benchmarks. |
| Sep 2025 | **Qwen3-Next** ([blog](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd), [vLLM](https://blog.vllm.ai/2025/09/11/qwen3-next.html)) | First hybrid: 3 **Gated DeltaNet** : 1 **Gated Attention**, ultra-sparse MoE, MTP. |
| Feb 2026 | **Qwen3.5-397B-A17B** ([Labonne](https://huggingface.co/blog/mlabonne/qwen35)) | 3:1 GDN : full attention generation-wide. 60 layers, 512 experts, 262K native, YaRN to ~1.01M at inference — with a model-card warning that static YaRN degrades short-context quality. |
| Apr 2026 | **Qwen3.6** ([wiki](https://aiwiki.ai/wiki/qwen3_6)) | Same hybrid carried forward + native vision; reoriented to agentic coding. |

**What to take from Qwen:** the only lab whose long-context capability came *primarily from
inference-time technique* for two generations (DCA, YaRN, MInference, chunked prefill) before
committing architecturally. They also maintain the strongest orthogonal line — **context compression
as a separate model** (QwenLong-CPRS) rather than an attention change. And they are the most
aggressive adopter of linear attention at frontier scale, which puts them in direct opposition to
MiniMax below.

---

### 6.4 Z.ai / Zhipu (GLM)

**Thesis evolution: long-context *data and alignment* first, architecture last. Adopted a competitor's mechanism rather than inventing one.**

| Date | Artifact | Attention / long-context content |
|---|---|---|
| 2023 | **LongBench** ([2308.14508](https://arxiv.org/abs/2308.14508)) | The bilingual multitask long-context benchmark that much of the field standardized on. |
| Jan 2024 | **LongAlign** ([2401.18058](https://arxiv.org/pdf/2401.18058)) | Long-context *alignment* recipe — instruction data, training, evaluation. LongBench-Chat at 10–100k. |
| 2024 | **LongWriter** (2408.07055) · **LongCite** (2409.02897) · **LongReward** (2410.21252) | Long-*output* generation, citation grounding, and RL reward for long context. A data/alignment cluster no other lab matches. |
| Jun 2024 | **ChatGLM / GLM-4** ([2406.12793](https://arxiv.org/pdf/2406.12793)) | GLM-130B → GLM-4 family. **GLM-4-9B-Chat-1M** shipped a 1M-token variant. |
| Dec 2024 | **LongBench v2** ([2412.15204](https://arxiv.org/pdf/2412.15204)) | Harder, reasoning-oriented successor — response to v1 saturation. |
| Aug 2025 | **GLM-4.5 / 4.6** ([2508.06471](https://arxiv.org/pdf/2508.06471)) | **GQA + partial RoPE + QK-Norm**, and the deliberate **96 heads at 5120 hidden** choice: *no* training-loss gain, but consistent MMLU/BBH gains. MoE-as-MTP layer. 200K. |
| Feb 2026 | **GLM-5** ([2602.15763](https://www.emergentmind.com/papers/2602.15763)) | 744B/40B, 256 experts. Switches to **MLA + DeepSeek's DSA**. Curriculum 32K→200K over ~28.5T tokens. |
| Apr 2026 | **GLM-5.1** | MLA + DSA, MIT weights. |
| Jun 2026 | **GLM-5.2 + IndexShare** ([Raschka](https://sebastianraschka.com/blog/2026/glm-5-2-indexshare.html), [MindStudio](https://www.mindstudio.ai/blog/glm-5-2-architecture-index-share-sparse-attention)) | Runs the DSA indexer fully **once per 4 layers**, reusing indices. 2.9× per-token FLOPs at 1M — not latency, and not proportional KV savings. |

**What to take from Zhipu:** the outlier in kind. Their long-context contribution is mostly
**benchmarks, alignment recipes, and long-output data** (LongBench/v2, LongAlign, LongWriter,
LongCite, LongReward), and when they needed an efficient mechanism they **adopted DeepSeek's DSA
rather than building one**. Their one original mechanism contribution, IndexShare, is an
*amortization* of someone else's mechanism. Worth remembering when weighing "who solved long
context": the evaluation infrastructure everyone else reports against came largely from here.

---

### 6.5 MiniMax

**Thesis evolution: all-in on linear → publicly reversed to full attention → returned with sparse. The most instructive record in the field, because they documented the reversal.**

| Date | Artifact | Attention / long-context content |
|---|---|---|
| Jul 2023 | **TransNormerLLM** ([2307.14995](https://arxiv.org/abs/2307.14995)) | The architecture built for linear attention: gated linear attention + SRMSNorm. |
| Jan 2024 | **Lightning Attention-2** ([2401.04658](https://arxiv.org/pdf/2401.04658)) | Hardware-aware linear attention, constant speed across sequence length. After 100k iterations / 300B tokens, a 0.001 performance decrement vs Lightning-1. |
| Jan 2025 | **MiniMax-01 / Text-01** ([2501.08313](https://arxiv.org/abs/2501.08313)) | 456B/45.9B, **7 Lightning : 1 softmax per 8-layer block**, 1M trained → 4M extrapolated. |
| Jun 2025 | **MiniMax-M1** ([2506.13585](https://arxiv.org/abs/2506.13585)) | First open-weight large-scale **hybrid-attention reasoning** model; native 1M (8× DeepSeek-R1); CISPO. |
| Oct 2025 | **"Why Did M2 End Up as a Full Attention Model?"** ([post](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model), [docs](https://platform.minimax.io/docs/guides/text-m2-full-attention)) | **Public post-mortem abandoning hybrid linear attention.** See below — the single most valuable primary source on this tradeoff. |
| Jun 2026 | **MiniMax Sparse Attention (MSA)** ([2606.13392](https://arxiv.org/html/2606.13392)) | Return to efficiency via **sparsity, not linearity**: two-branch block-sparse over **GQA**, per-GQA-group top-k, with the full training recipe published (KL alignment, gradient detach, 40B-token indexer warmup, forced local block). 109B MoE, 3T tokens. |

**The M2 post-mortem, in their own terms** — every claim here is from MiniMax, not inference:

- **Quality:** the hybrid Lightning model looked fine on *saturated* benchmarks (MMLU, BBH, MATH) but
  showed **clear deficits on complex multi-hop reasoning** at larger scale.
- **The mechanism they blame — and this is the deepest claim in the document:** global attention
  patterns like **retrieval heads and induction heads form early in pretraining and cannot be
  adequately adjusted afterward.** Their SWA hybrid degraded noticeably as context grew, which is
  disqualifying for agentic use. This is a direct argument that efficient attention must be a
  *pretraining* decision, not a conversion.
- **Systems:** linear attention is **memory-bound and wastes FLOPs** under poor IO; it is **far more
  sensitive to numerical precision** than full attention; and there are **no mature solutions for
  prefix caching, speculative decoding, or low-precision state storage.**
- **Evaluation:** Goodhart's law — "benchmarks are a leaky abstraction"; the real cost of hybrid
  attention only appears at scale, where measuring it is expensive.
- **Conditions to revisit:** better long-context data, mature infrastructure, better evaluation.

**Why this matters more than any benchmark table in this document:** MiniMax's precision-sensitivity
and prefix-caching objections are *exactly* the problems Moonshot reports solving for KDA (FP8-stable
kernels, prefix caching reworked for recurrent state), and their retrieval-head argument is exactly
what Kimi Linear's matched-recipe comparison against pure MLA claims to refute. Two labs, opposite
conclusions, both with shipped frontier models. Anyone evaluating linear attention should read the
M2 post and the Kimi Linear paper side by side and treat the disagreement as unresolved — and note
that MiniMax's eventual answer was **sparse attention**, which is where DeepSeek and Z.ai already
were.

---

### 6.6 Cross-lab summary

| Lab | Original mechanism(s) | Current bet (mid-2026) | Reversed? |
|---|---|---|---|
| **DeepSeek** | MLA, NSA, DSA, CSA/HCA, optical compression | compress + learned select | **No** — strictly cumulative |
| **Moonshot** | MoBA, KDA, Attention Residuals | linear recurrent state (3:1 KDA:MLA) | **Yes** — abandoned own MoBA |
| **Qwen** | DCA, Gated Attention, QwenLong-CPRS | linear (3:1 GDN:full) generation-wide | No, but arrived late |
| **Z.ai** | IndexShare; LongBench/LongAlign ecosystem | **adopted** MLA + DSA | n/a — never had own mechanism |
| **MiniMax** | TransNormerLLM, Lightning Attention, MSA | sparse over GQA | **Yes, twice** — linear → full → sparse |

Three of five converged on **learned sparsity**; two on **linear recurrence**; the two labs that
reversed both reversed *away* from what they themselves invented. That is the honest state of the
question as of August 2026.

---

## Part 7 — Hybrid attention from NVIDIA, Microsoft, Meta, and Google

A separate genealogy from Part 6. Broadly: **the US labs went SSM-hybrid (Mamba / gated linear
recurrence), the Chinese labs went delta-rule-linear or learned-sparse.** Same problem, different
ancestor, and the two lines barely cite each other.

---

### 7.1 NVIDIA — hybrid ratio as a *search* problem

| Date | Paper | Contribution |
|---|---|---|
| Nov 2024 | **Hymba** ([2411.13676](https://arxiv.org/html/2411.13676v1)) | **Hybrid-*head*, not hybrid-layer**: attention heads and SSM heads run **in parallel inside the same layer** — attention for high-resolution recall, SSM for efficient context summarization. Adds **learnable meta tokens** prepended to the prompt to relieve the "forced-to-attend" burden (a structural answer to the same problem gpt-oss solves with sink bias). |
| Nov 2024 | **Star Attention** ([2411.17116](https://arxiv.org/abs/2411.17116)) | Two-phase block-sparse inference for long sequences; blockwise-local prefill then global decode. |
| Apr 2025 | **Nemotron-H** ([2504.03624](https://arxiv.org/pdf/2504.03624), [ADLR](https://research.nvidia.com/labs/adlr/nemotronh/)) | Only ~**8% of layers are self-attention**, evenly dispersed; the rest alternate Mamba-2 and FFN. 8B: 4 attention / 24 Mamba-2 of 52 layers. 56B: 10 / 54 of 118. Up to **3× throughput** vs Llama-3.1/Qwen-2.5 at matched accuracy. |
| Aug 2025 | **Nemotron Nano 2** ([2508.14444](https://arxiv.org/abs/2508.14444)) | Hybrid Mamba-Transformer *reasoning* model. 20T tokens, FP8 recipe, then **Minitron** compression to fit **128k context on a single A10G**. Up to **6× throughput** in reasoning-shaped loads (8k in / 16k out). |
| Aug 2025 | **Jet-Nemotron / PostNAS / JetBlock** ([2508.15884](https://arxiv.org/abs/2508.15884), [code](https://github.com/NVlabs/Jet-Nemotron), [page](https://research.nvidia.com/labs/eai/publication/jetnemotron/)) | Methodologically the most distinct work in this whole document. **PostNAS starts from a pre-trained full-attention model, freezes the MLP weights, and searches only the attention block design** — so hybrid placement is *learned*, not hand-set at 3:1 or 5:1. **JetBlock** = linear attention + dynamic convolution + hardware-aware search. The resulting model **mixes full attention, SWA, and JetBlock, each placed where it measures best.** |
| Dec 2025 | **Nemotron 3** ([2512.20856](https://arxiv.org/pdf/2512.20856)) | Hybrid Mamba-Transformer **MoE**: predominantly interleaved Mamba-2 and MoE layers with a few self-attention layers. |

**Take:** NVIDIA is the only group treating the hybrid ratio as a search/NAS problem rather than a
design choice. Everyone else in this document reports "we used 3:1" or "we used 5:1" with an
ablation table; PostNAS asks where each mechanism *belongs*. Also the only lab (with Falcon-H1)
running mechanisms **in parallel within a layer** rather than interleaving layers.

---

### 7.2 Microsoft — one paper in nearly every family, and the origin of the learned block gate

| Date | Paper | Contribution |
|---|---|---|
| Jul 2023 | **RetNet** ([2307.08621](https://arxiv.org/abs/2307.08621)) | Retention with three equivalent forms — parallel (train), recurrent (decode), chunkwise (long seq). The "fixed-gate" generation of linear attention. |
| Jul 2023 | **LongNet** ([2307.02486](https://arxiv.org/abs/2307.02486)) | **Dilated attention** — exponentially expanding receptive field; the 1B-token-sequence claim. |
| Feb 2024 | **LongRoPE** ([2402.13753](https://arxiv.org/abs/2402.13753)) → **LongRoPE2** ([2502.20082](https://arxiv.org/abs/2502.20082)) | Per-dimension *searched* RoPE rescaling instead of a closed-form formula; v2 addresses the short-context regression that plagues YaRN-style extension. |
| May 2024 | **YOCO** ([2405.05254](https://arxiv.org/abs/2405.05254)) | Decoder-decoder: **cache the KV exactly once** and let the cross-decoder reuse it. Attacks KV bytes structurally rather than by compression or selection. |
| Jun 2024 | **Samba** ([2406.07522](https://arxiv.org/abs/2406.07522), ICLR 2025, [code](https://github.com/microsoft/Samba)) | Layer-wise **Mamba + SwiGLU + Sliding Window Attention**. Mamba carries time-dependent semantics, SWA fills in non-recurrent dependencies. Up to 3.8B / 3.2T tokens. The striking result: **trained at 4K, extrapolates to 256K with perfect passkey recall**, and improved perplexity out to **1M zero-shot**; 3.73× throughput vs GQA at 128K prompts. |
| Jul 2024 | **MInference** ([2407.02490](https://arxiv.org/abs/2407.02490)) | Training-free dynamic sparse prefill — later adopted wholesale by Qwen2.5-1M. |
| **Oct 2024** | **SeerAttention** ([2410.13276](https://arxiv.org/abs/2410.13276)) | **The origin of the learned block gate** (see Part 5's lineage table). MoE-inspired learnable gate over block sparsity, self-distilled, **gate parameters only**. Four months before NSA and MoBA. |
| Oct 2024 | **Differential Transformer** ([2410.05258](https://arxiv.org/abs/2410.05258)) | Subtract two softmax maps to cancel attention noise; improves long-context retrieval and reduces activation outliers. |
| Jun 2025 | **SeerAttention-R** ([2506.08889](https://arxiv.org/pdf/2506.08889)) | Extends the learned gate to long *decoding*/reasoning, where the original targeted prefill. |
| Jul 2025 | **SambaY / Decoder-Hybrid-Decoder + GMU** ([2507.06607](https://arxiv.org/html/2507.06607)) → **Phi-4-mini-flash-reasoning** ([HF](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning), [Azure](https://azure.microsoft.com/en-us/blog/reasoning-reimagined-introducing-phi-4-mini-flash-reasoning/)) | Self-decoder = **Mamba + SWA + a single full-attention layer**; cross-decoder interleaves expensive cross-attention with cheap **Gated Memory Units** for cross-layer memory sharing. Preserves linear prefill; **up to 10× throughput and 2–3× lower latency**. |
| Oct 2025 | **MTraining** ([2510.18830](https://arxiv.org/abs/2510.18830)) | Distributed dynamic sparse attention for ultra-long-context *training* — balanced and hierarchical sparse ring attention. |
| — | **ArchScale** ([code](https://github.com/microsoft/ArchScale)) | Their architecture-research harness: MHA/GQA, SSM variants, GMU, YOCO, DIFF attention, and flexible hybrid stacks in one codebase. |

**Take, and it's the most under-credited fact in this document:** Microsoft **invented the learned
block gate** (SeerAttention, Oct 2024) and the Chinese labs productionized it at frontier scale
(NSA/MoBA Feb 2025 → DSA → MSA). Microsoft also has the widest portfolio here — recurrence, dilation,
SSM-hybrid, cache-once, denoised softmax, learned gating, positional extension, inference sparsity,
distributed sparse training — but shipped none of it in a frontier-scale flagship. Their vehicle is
the Phi line, at 3.8B.

---

### 7.3 Google / DeepMind — memory as a *module*, not sparsity

| Date | Paper | Contribution |
|---|---|---|
| 2019–2020 | **Transformer-XL** (1901.02860, w/ CMU) · **Reformer** ([2001.04451](https://arxiv.org/abs/2001.04451)) · **ETC** (2004.08483) · **Big Bird** ([2007.14062](https://arxiv.org/abs/2007.14062)) · **Performer** ([2009.14794](https://arxiv.org/pdf/2009.14794)) | Segment recurrence + relative position; LSH; global-local; random+window+global; FAVOR+ kernels. The 2020 wave largely came from here. |
| 2022 | **Memorizing Transformers** (2203.08913) · **Block-Recurrent Transformers** (2203.07852) | kNN lookup into non-differentiable external memory; recurrence over blocks. |
| Feb 2024 | **Griffin / Hawk** ([2402.19427](https://arxiv.org/html/2402.19427v1)) | **Hawk** = pure recurrence via **RG-LRU** (Real-Gated Linear Recurrent Unit), beating Mamba on downstream tasks. **Griffin** = RG-LRU blocks **alternated with local MQA attention**. Both **extrapolate well past their training length** — the result that made gated-linear-recurrence hybrids credible. |
| Mar 2024 | **Gemini 1.5** ([2403.05530](https://arxiv.org/abs/2403.05530)) | The 10M-context report; near-perfect retrieval claims that set the industry's long-context expectations (architecture undisclosed). |
| Apr 2024 | **RecurrentGemma** ([2404.07839](https://arxiv.org/pdf/2404.07839), [code](https://github.com/google-deepmind/recurrentgemma)) | Griffin shipped as open weights, 2B/9B. Fixed-size state → long prompts without a growing KV cache. |
| Apr 2024 | **Infini-attention** ([2404.07143](https://arxiv.org/abs/2404.07143)) | Local softmax within a segment + **compressive memory updated by linear attention**; "leave no context behind." |
| Apr 2024 | **Mixture-of-Depths** ([2404.02258](https://arxiv.org/abs/2404.02258)) | Routes *tokens* past whole layers — orthogonal compute-allocation axis. |
| Oct 2024 | **Selective Attention** ([2410.02703](https://arxiv.org/abs/2410.02703)) | Lets the model mask irrelevant past tokens, shrinking the effective context. |
| Jan 2025 | **Titans** ([2501.00663](https://arxiv.org/abs/2501.00663)) | Three-tier memory: attention (short), **surprise-driven neural long-term memory learned at test time**, static persistent. 2M+ tokens. |
| Mar 2025 | **Gemma 3** ([2503.19786](https://arxiv.org/pdf/2503.19786)) | 5:1 local:global, window down to 1024; RoPE base raised on global layers only. |
| Apr 2025 | **Miras — "It's All Connected"** ([2504.13173](https://arxiv.org/abs/2504.13173)) | The unifying theory. Recasts **Transformers, Titans, and modern linear RNNs all as associative-memory modules** differing in four choices: memory architecture, **attentional bias objective**, retention gate, memory learning algorithm. Introduces **Moneta, Yaad, Memora**; shows Titans-LMM is a special case. The only genuine theoretical framework in this entire document. |
| May 2025 | **ATLAS** ([2505.23735](https://arxiv.org/abs/2505.23735)) | Learning to *optimally* memorize context at test time — the Titans line continued. |
| Jul 2025 | **Mixture-of-Recursions** ([2507.10524](https://arxiv.org/abs/2507.10524)) | Adaptive per-token recursion depth with recursion-wise KV sharing. |

**Take:** Google's answer to long context has consistently been **an explicit memory module**, not
sparsity or a cheaper mixer — Memorizing Transformers → Infini-attention → Titans → Miras → ATLAS.
They are also the only lab to publish a *unifying framework* rather than another point architecture,
and the only one whose 2020-era work (BigBird, Performer, Reformer) and 2025-era work (Titans, Miras)
are both foundational. Note the gap, though: none of Titans/Miras/ATLAS is confirmed in a shipped
Gemini, and Gemma 3 ships plain 5:1 local-global.

---

### 7.4 Meta — the lab that explicitly declined sparse attention

| Date | Paper | Contribution |
|---|---|---|
| Sep 2022 | **MEGA** ([2209.10655](https://arxiv.org/abs/2209.10655)) | Moving-average-equipped gated attention — early gated-recurrence-plus-attention hybrid. |
| May 2023 | **MEGABYTE** ([2305.07185](https://arxiv.org/abs/2305.07185)) | Patch-level multiscale decoder for million-byte sequences. |
| Sep 2023 | **Effective Long-Context Scaling of Foundation Models** (Llama 2 Long) ([2309.16039](https://arxiv.org/abs/2309.16039)) | 32K effective context by **continual pretraining with upsampled long texts and a positional-encoding change only.** They state plainly: **sparse attention was not applied**, because at Llama-2-70B's model dimension the attention matrix only becomes the compute bottleneck **past 49,152 tokens**. The most useful negative result in this document — a quantified argument that below ~50K, sparsity buys you nothing at that width. |
| Sep 2023 | **StreamingLLM / attention sinks** ([2309.17453](https://arxiv.org/abs/2309.17453), w/ MIT) | The attention-sink discovery that gpt-oss later turned into a learned bias. |
| May 2024 | **CoPE — Contextual Position Encoding** ([2405.18719](https://arxiv.org/abs/2405.18719)) | Position **conditioned on content**: increment position only on model-selected tokens, enabling "attend to the i-th noun/sentence." Solves selective-copy, counting, and flip-flop tasks where RoPE/ALiBi fail. |
| Dec 2024 | **Memory Layers at Scale** ([2412.09764](https://arxiv.org/abs/2412.09764)) · **Byte Latent Transformer** ([2412.09871](https://arxiv.org/abs/2412.09871)) | Trainable key-value memory as a layer type; dynamic byte patching in place of tokenization. |
| Apr 2025 | **Multi-Token Attention** ([2504.00927](https://arxiv.org/html/2504.00927v1)) | Convolutions over query/key/head dimensions so attention weights condition on **multiple** query and key vectors at once, not single-token similarity. |
| Apr 2025 | **Llama 4** ([blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)) | **iRoPE** — 3 RoPE : 1 NoPE + inference-time attention temperature scaling. Positional, not sparse. Never trained past 256K. |

**Take:** Meta is the clearest dissenter. Their long-context work is almost entirely **positional and
data-side** (Llama 2 Long, CoPE, iRoPE) rather than efficiency-side, and they published a concrete
FLOP-crossover argument for *why* — attention isn't the bottleneck at 70B until ~49K tokens. They
have no frontier hybrid-attention model and no sparse-attention mechanism in the Part 5 lineage.
Their contribution to how everyone else's models work is CoPE-style content-conditioned position and
the original attention-sink observation.

---

### 7.5 What separates the two genealogies

| | US labs (NVIDIA / Microsoft / Google / Meta) | Chinese labs (DeepSeek / Moonshot / Qwen / Z.ai / MiniMax) |
|---|---|---|
| **Efficient mixer of choice** | **SSM / gated linear recurrence** — Mamba-2, RG-LRU, retention | **Delta-rule linear** (KDA, GDN) or **learned sparse** (DSA, MSA) |
| **How the ratio is chosen** | ablation, or **NAS** (NVIDIA PostNAS) | hand-set 3:1 or 5:1 with an ablation table |
| **Memory** | explicit **memory modules** (Titans, Infini-attention, Memory Layers, GMU) | no separate memory module; state lives in the mixer |
| **Theory** | **Miras** unifies attention/Titans/linear-RNNs as associative memory | none published; engineering-first |
| **Frontier deployment** | mostly **not** at frontier scale — Phi 3.8B, Gemma, Nemotron Nano, RecurrentGemma 9B | shipped at 400B–2.8T (K3, V4, GLM-5.2, Qwen3.5) |
| **Positional innovation** | CoPE, LongRoPE/2, iRoPE, NoPE | NoPE (Kimi), HyPE (MiniCPM), DCA+YaRN (Qwen) |

Two asymmetries worth sitting with:

1. **The US labs invented much of what the Chinese labs shipped.** SeerAttention (Microsoft, Oct 2024)
   is the first learned block gate; MInference (Microsoft) went straight into Qwen2.5-1M; Gated
   DeltaNet's delta rule and Mamba-2's SSD duality underpin KDA and GDN. But the frontier-scale
   validations — 1M context at 400B+ parameters — came almost entirely from the Chinese labs.
2. **Nobody has reconciled the two mixer families.** SSM-hybrids (Samba, Griffin, Nemotron-H) and
   delta-rule hybrids (KDA, GDN) are rarely compared head-to-head at scale under a matched recipe.
   [Gated DeltaNet-2 (2605.22791)](https://arxiv.org/pdf/2605.22791) benchmarking against Mamba-3 at
   1.3B is the closest thing to a controlled comparison, and 1.3B is far below where MiniMax says the
   deficits appear.

*Useful cross-lab surveys:* [Speed Always Wins: Efficient Architectures for LLMs, 2508.09834](https://arxiv.org/pdf/2508.09834) ·
[Hybrid Architectures: Systematic Analysis, 2510.04800](https://arxiv.org/html/2510.04800v3) ·
[Forgetting Transformer, 2503.02130](https://arxiv.org/pdf/2503.02130)

---

## Part 8 — Ant Group, ByteDance, and Tencent

The three labs most often left out of attention surveys, despite one of them running the largest
hybrid-linear-attention model in production and another running the first industry-deployed Mamba
model at scale.

---

### 8.1 Ant Group (BaiLing / Ling · inclusionAI) — hybrid linear attention at *trillion* scale

The most aggressive scaler of linear attention anywhere, and the direct counterweight to MiniMax's
argument that linear attention fails at scale.

| Date | Artifact | Attention / long-context content |
|---|---|---|
| Oct 2025 | **Ring-linear 2.0 / Ling 2.0 Linear** ([arXiv 2510.19338](https://arxiv.org/abs/2510.19338), [Ring-flash-linear-2.0 writeup](https://ant-ling.medium.com/ring-flash-linear-2-0-a-highly-efficient-hybrid-architecture-for-test-time-scaling-517b6bd66551)) | **Hybrid linear + standard attention over a sparse MoE**, explicitly to fix pure linear attention's recall weakness. Ring-flash-linear-2.0-128K matches ~40B dense performance at **6.1B activated**. Two specific linear-attention fixes worth noting: **RoPE applied to the q and k inputs of the linear attention**, and a **grouped, non-shared RMSNorm on the linear-attention output**. |
| Feb 2026 | **Ling-2.5-1T / Ring-2.5-1T** ([BusinessWire](https://www.businesswire.com/news/home/20260215551663/en/Ant-Group-Releases-Ling-2.5-1T-and-Ring-2.5-1T-Evolving-Its-Open-Source-AI-Model-Family), [HF](https://huggingface.co/inclusionAI/Ring-2.5-1T)) | **Linear attention + selected softmax attention layers at 1T parameters.** Ant frames the contribution as algorithm–system **co-design**: stability techniques for large-scale linear-attention *training*, and efficient distributed training for ultra-long sequences. |
| Jun 2026 | **Ling and Ring 2.6 Technical Report** ([arXiv 2606.15079](https://arxiv.org/abs/2606.15079)) | Hybrid linear attention integrating **Lightning Attention with MLA** — i.e. MiniMax's linear kernel paired with DeepSeek's KV compression. Base model upgraded by **architectural migration pre-training** (converting an existing dense-attention model), unified co-design of architecture + objectives + serving + agent training. |
| Jan 2026 | **ICLR 2026 Expo:** *Scaling Hybrid Linear Attention Architecture to Trillion-Scale* ([listing](https://iclr.cc/virtual/2026/expo-talk-panel/10020572)) | Their explicit public claim: hybrid linear attention works at 1T. |
| — | Related systems: **ZeCO** ([2507.01004](https://arxiv.org/pdf/2507.01004)) | Zero-communication-overhead sequence parallelism for linear attention. |

**Take:** Ant is doing at 1T what MiniMax said couldn't be made to work, and doing it via *architectural
migration* rather than pretraining from scratch — the very conversion path MiniMax argued against
(their claim: retrieval and induction heads form early and can't be adjusted afterward). Ant's
stated hard problems are **training stability for linear attention at scale** and **distributed
ultra-long-sequence training**, which is a different bottleneck list than either MiniMax's or
Moonshot's. Their RoPE-on-linear-attention choice is also a **second independent vote against Kimi's
NoPE**, alongside MiniCPM-SALA's HyPE.

---

### 8.2 ByteDance (Seed) — systems-first, plus memory layers

Their center of gravity is **training and serving infrastructure for long context** rather than a
signature attention mechanism.

| Date | Artifact | Attention / long-context content |
|---|---|---|
| Nov 2024 | **UltraMem** ([2411.12364](https://arxiv.org/abs/2411.12364)) → **UltraMemV2** ([2508.18756](https://arxiv.org/pdf/2508.18756)) | **Ultra-sparse memory networks** as an alternative to MoE. V1 only matched 2-expert MoE; **V2 closes the gap at 120B with explicitly better long-context learning.** A memory-layer route to long context, parallel to Meta's Memory Layers. |
| Feb 2025 | **ByteScale** ([2502.21231](https://arxiv.org/html/2502.21231v1)) | **Hybrid Data Parallelism (HDP)** — unifies inter- and intra-data partitioning with a dynamic mesh. Evaluated 7B–141B at **256K–2048K context on >12,000 GPUs**, up to **7.89× over prior systems**. The largest-scale long-context training result published by anyone. |
| Apr 2025 | **PHD-Transformer** ([writeup](https://www.aibase.com/news/17454)) | Parallel Hidden Decoding Transformer for pretraining **length scaling to 2048K** while preserving inference efficiency. |
| Jul 2025 | **Scaling Linear Attention with Sparse State Expansion** ([2507.16577](https://arxiv.org/html/2507.16577v1)) | Reframes **state updating as information classification**, giving a row-sparse update formulation for linear attention — a genuinely different angle on the state-capacity problem that KDA and GDN attack with gating. |
| Aug 2025 | **Seed-OSS** ([coverage](https://eu.36kr.com/en/p/3431996374142339)) | Apache-2.0 release with a **native 512K context built during pretraining, not interpolated** — 4× the then-mainstream 128K. Notable as a data/curriculum claim rather than a mechanism claim. |
| Oct 2025 | **Long-Context Attention Benchmark** ([2510.17896](https://arxiv.org/html/2510.17896v1)) | From kernel efficiency through distributed context parallelism — the evaluation harness the field lacked for comparing these mechanisms as *systems*. |
| Oct 2025 | **Alleviating Forgetfulness of Linear Attention** ([2510.20787](https://arxiv.org/pdf/2510.20787)) | Hybrid **sparse** attention + **contextualized learnable token eviction** to patch linear attention's recall failure — a third repair strategy alongside gating (GDN) and hybridization (Kimi). |
| Nov 2025 | **SSA: Sparse Sparse Attention** ([2511.20102](https://arxiv.org/pdf/2511.20102)) | Aligns full and sparse attention **outputs in feature space** — an alternative to MSA's KL-on-attention-distribution alignment. Directly relevant to the selector-training problem in Part 5's finding B. |

**Take:** ByteDance's distinctive contribution is proving the *systems* side — 2048K context across
12,000+ GPUs — and pursuing **memory layers** (UltraMemV2) as a long-context route most labs ignore.
Note also that Seed-OSS's 512K is a **native pretrained** window, which is the cleanest existing
rebuttal to the "advertised ≠ usable" complaint in Part 4.

*Attribution caution:* Seed-OSS, ByteScale, PHD-Transformer, UltraMem/V2, and Sparse State Expansion
are confirmed ByteDance Seed. The remaining 2510/2511 entries surfaced in ByteDance-targeted searches
and I did **not** verify author affiliation — treat those attributions as unconfirmed.

---

### 8.3 Tencent (Hunyuan) — first industry-deployed large-scale Mamba

| Date | Artifact | Attention / long-context content |
|---|---|---|
| Nov 2024 | **Hunyuan-Large** ([2411.02265](https://arxiv.org/abs/2411.02265)) | 389B total / 52B active, **256K context**. Lists a **KV cache compression technique** among its four headline practices (with large-scale synthetic data, mixed expert routing, expert-specific learning rates). At the time, the largest open Transformer MoE. |
| Mar–May 2025 | **Hunyuan-TurboS** ([2505.15431](https://arxiv.org/abs/2505.15431), [code](https://github.com/Tencent-Hunyuan/Hunyuan-TurboS), [announcement](https://x.com/TencentHunyuan/status/1899105803073958010)) | **The first ultra-large hybrid Transformer-Mamba MoE**, and per Tencent the **first industry-deployed large-scale Mamba model**. 560B total / 56B active, **128 layers** in an **AMF/MF block pattern** (Attention-Mamba-FFN and Mamba-FFN): Mamba-2 for linear complexity, **GQA to minimize KV cache**, MoE FFNs. 16T tokens, 256K context. Also ships an **adaptive long/short chain-of-thought** mechanism that switches between fast answers and deep reasoning — a *compute*-allocation strategy for long generation rather than long input. |
| 2025 | **Hunyuan-T1** | Reasoning model on the TurboS Mamba backbone. |
| Jun 2025 | **Hunyuan-A13B** | 80B/13B, 256K context, hybrid reasoning modes. |

**Take:** Tencent is the only lab in Parts 6–8 that went **Mamba** rather than delta-rule linear or
learned sparse — putting them in the *US* genealogy (Nemotron-H, Samba, Jamba) despite being a Chinese
lab, and making TurboS the natural head-to-head against Nemotron-H at frontier scale. Their AMF/MF
block pattern is also more granular than the flat 3:1 / 5:1 ratios everywhere else. Both Hunyuan-Large
and TurboS stop at **256K** — Tencent has not chased 1M, which is itself a position.

---

### 8.4 Where these three sit

| Lab | Mixer bet | Distinctive contribution | Max context |
|---|---|---|---|
| **Ant Group** | **linear + selected softmax** (Lightning + MLA) | hybrid linear attention **at 1T params**, via architectural migration; training-stability techniques | 128K+ |
| **ByteDance** | no signature mixer; **memory layers** | **2048K-context training on 12k+ GPUs**; native-pretrained 512K; feature-space sparse alignment | 512K native / 2048K trained |
| **Tencent** | **Mamba-2 hybrid** (AMF/MF blocks) | first industry-deployed large-scale Mamba; adaptive long/short CoT | 256K |

Adding these three changes one conclusion from Part 6. The linear-attention camp is not two labs
(Qwen, Moonshot) against a sparse-attention consensus — it is **Qwen, Moonshot, Ant, and Tencent**,
with Ant running it at 1T parameters and Tencent running Mamba in production. And Ant's success with
**architectural migration** to hybrid linear attention is direct evidence against the specific
mechanism MiniMax blamed for their reversal. That disagreement is now three labs deep on each side
and still unresolved.

---

## Part 9 — Xiaomi and Prime Intellect

Two labs at opposite ends of the stack: one shipping a conventional attention design very well, the
other bypassing attention as the long-context mechanism entirely.

---

### 9.1 Xiaomi (LLM-Core / MiMo)

| Date | Artifact | Attention / long-context content |
|---|---|---|
| May 2025 | **MiMo-7B** ([2505.07608](https://arxiv.org/abs/2505.07608), [code](https://github.com/XiaomiMiMo/MiMo)) | Reasoning-focused 7B, 25T tokens, **MTP** objective for quality and decode speed. Three-stage data mixing. **No attention innovation** — the contribution is data and RL, and MiMo-7B-RL beating o1-mini at 7B. Listed for completeness. |
| Jan 2026 | **MiMo-V2-Flash** ([2601.02780](https://arxiv.org/abs/2601.02780), [code](https://github.com/xiaomimimo/MiMo-V2-Flash), [page](https://mimo.xiaomi.com/mimo-v2-flash)) | 309B total / 15B active MoE. **8 Hybrid Blocks, each interleaving 5 SWA blocks with 1 Global Attention block** — a 128-token sliding window at 5:1. **~6× KV-cache reduction**, with long-context performance held up by a **learnable attention sink bias**. 27T tokens with MTP; native 32K extended to **256K**. Post-training uses **Multi-Teacher On-Policy Distillation (MOPD)** — domain-specialized teachers giving dense token-level reward. |

**Correction to Parts 2 and 5:** I earlier described MiMo-V2-Flash as "purely static sparsity — no
learned selector, no learned components." That is wrong in one respect: it carries a **learnable
attention sink bias**, the same device gpt-oss uses, and Xiaomi credits it explicitly for preserving
long-context quality under a 128-token window. The correct characterization is *static sparsity plus
a learned sink* — which is exactly gpt-oss's recipe, scaled from 131K to 256K and from ~117B to 309B.
That makes Xiaomi and OpenAI the two clearest instances of the same minimal design, and strengthens
finding G rather than weakening it: **SWA + sink bias is now demonstrated at 309B and 256K.**

**Take:** Xiaomi is the strongest evidence that you do not need a novel mechanism to be competitive —
their two contributions are a well-executed standard hybrid and a distillation paradigm. They are
also the only lab here whose 5:1 block structure is stated as explicit *blocks* (8 × [5 SWA + 1 GA])
rather than an aggregate ratio.

---

### 9.2 Prime Intellect — context as an environment, not as attention input

Prime Intellect is a decentralized-training lab, so they have **no attention-mechanism paper**. But
their long-context position is genuinely distinct and belongs in this document, because it argues the
problem should not be solved in the attention layer at all.

| Date | Artifact | Long-context content |
|---|---|---|
| Nov 2024 | **INTELLECT-1** ([2412.01152](https://arxiv.org/abs/2412.01152)) | First 10B model trained collaboratively across the globe. Decentralized training (OpenDiLoCo / PRIME), not attention. |
| May 2025 | **INTELLECT-2** ([2505.07291](https://arxiv.org/abs/2505.07291)) | Decentralized RL at 32B. |
| Dec 2025 | **INTELLECT-3** ([2512.16144](https://arxiv.org/abs/2512.16144), [blog](https://www.primeintellect.ai/blog/intellect-3)) | 106B MoE **built on the GLM-4.5-Air base** — so it inherits GQA + partial RoPE + QK-Norm from Part 6.4 rather than introducing anything. The long-context content is on the *training and behavior* side: **context parallelism (CP) scaling RL training to 98K context**, and explicitly training **long-horizon behavior in which the model manages its own context — cutting context, branching, and maintaining lightweight external memory.** |
| 2026 | **"Recursive Language Models: the paradigm of 2026"** ([blog](https://www.primeintellect.ai/blog/rlm)) | Their public bet on RLMs as the long-context answer. |

**The idea they're pointing at — RLM** ([arXiv 2512.24601](https://arxiv.org/abs/2512.24601), Zhang,
Kraska & Khattab, MIT CSAIL; [author writeup](https://alexzhang13.github.io/blog/2025/rlm/),
[code](https://github.com/alexzhang13/rlm)) — is not Prime Intellect's paper but is the mechanism
they're advocating, and it is a **seventh family** absent from Part 1's taxonomy:

> Treat the long prompt as **part of an external environment** rather than as model input. The prompt
> is stored in a variable inside a Python REPL; the LLM **programmatically inspects, decomposes, and
> recursively calls itself** over snippets of it.

Reported results: handles inputs **two orders of magnitude beyond the model's context window**, and
even on *short* prompts beats vanilla frontier LLMs and standard scaffolds — on GPT-5, a median
**+26% vs compaction, +130% vs CodeAct with sub-calls, +13% vs Claude Code** across four long-context
tasks, at comparable cost.

**Why this belongs in an attention survey:** every mechanism in Parts 1–8 assumes the answer is to
make attention over N tokens cheaper. RLM and INTELLECT-3's self-managed context assume the answer is
to **not put N tokens through attention at all** — an argument that the 1M-token race is optimizing
the wrong variable. Set against DeepSeek-V4 spending an entire architecture redesign to reach 97%
NIAH at 1M, it is the sharpest available framing of the alternative. It also composes with, rather
than competes against, everything else here: an RLM scaffold over a KDA or DSA model gets both.

*Adjacent:* QwenLong-CPRS (Part 6.3) is the compression-model version of the same instinct;
[LongGenBench, 2410.04199](https://arxiv.org/pdf/2410.04199) evaluates the long-*generation* side.

---

## Part 10 — Cross-lab master summary

Every lab surfaced across Parts 1–9, in four views: the roster, the mechanism matrix, the disputed
questions, and the paper index.

---

### 10.1 Master roster

Sorted by mixer family, then scale. "Max ctx" = largest **shipped/claimed** window; see Part 4 flag 3
on why that number is not the same as usable context.

| Lab | Signature mechanism(s) | Mixer bet | Flagship & scale | Max ctx | Frontier? |
|---|---|---|---|---|---|
| **DeepSeek** | MLA · NSA · DSA · CSA/HCA · optical compression | **compress + learned select** | DeepSeek-V4 | **1M** | ✅ |
| **Z.ai / Zhipu** | IndexShare · LongBench/LongAlign ecosystem | **adopted** MLA + DSA | GLM-5.2, 744B/40B | 200K | ✅ |
| **MiniMax** | Lightning Attention · **MSA** | **sparse over GQA** (post-reversal) | MSA 109B; M1 456B | 1M (M1) | ✅ |
| **Moonshot / Kimi** | **MoBA** · **KDA** · Attention Residuals | **delta-rule linear** 3:1 | Kimi K3, 2.8T | **1M** | ✅ |
| **Qwen / Alibaba** | DCA · **Gated Attention** · QwenLong-CPRS | **Gated DeltaNet** 3:1 | Qwen3.5, 397B/17B | 262K → ~1.01M YaRN | ✅ |
| **Ant Group** | hybrid linear at **1T**; architectural migration | **Lightning + MLA** hybrid linear | Ling/Ring-2.5-1T | 128K+ | ✅ |
| **Tencent** | first industry-deployed large Mamba; AMF/MF blocks | **Mamba-2 hybrid** | Hunyuan-TurboS, 560B/56B | 256K | ✅ |
| **OpenBMB / MiniCPM** | **InfLLM-v2** · **HyPE** | **25% sparse + 75% linear** | MiniCPM-SALA, 9B | **1M+** | ❌ (9B) |
| **Thinking Machines** | asymmetric GQA · finite-span relative bias · SConv | **static SWA 5:1** | Inkling, 975B/41B | **1M** | ✅ |
| **Xiaomi** | SWA blocks + learned sink · MOPD | **static SWA 5:1** | MiMo-V2-Flash, 309B/15B | 256K | ✅ |
| **OpenAI** | **learned attention-sink bias** | **static SWA 1:1**, window 128 | gpt-oss-120b | 131K | ⚠️ (open-weight tier) |
| **NVIDIA** | **Hymba** parallel heads · **PostNAS/JetBlock** | **Mamba-2 hybrid, NAS-searched** | Nemotron 3 / Nano 2 | 128K | ❌ |
| **Microsoft** | **SeerAttention** · Samba · YOCO · DIFF · LongRoPE · MInference | **SSM + SWA** | Phi-4-mini-flash, 3.8B | 128K+ | ❌ |
| **Google / DeepMind** | Infini-attention · **Titans** · **Miras** · Griffin/RG-LRU | **memory module** + local-global | Gemma 3 / Gemini | 1M+ (Gemini) | ✅ (closed) |
| **Meta** | **declined sparsity** · CoPE · iRoPE · attention sinks | **none — positional only** | Llama 4 Scout | 256K trained (10M claimed) | ✅ |
| **ByteDance** | **ByteScale** · UltraMemV2 · Sparse State Expansion | **none signature; memory layers** | Seed-OSS | 512K native / 2048K trained | ⚠️ |
| **Prime Intellect** | self-managed context; **RLM advocacy** | inherits GLM-4.5-Air | INTELLECT-3, 106B | 98K (RL training) | ❌ |
| **MIT CSAIL** | **RLM** — context as environment | **bypasses attention** | n/a (inference paradigm) | ~100× model window | n/a |
| *AI21* | Jamba | Transformer–Mamba–MoE | Jamba | 256K | ❌ |
| *TII* | Falcon-H1 | **parallel** attention + Mamba-2 heads | Falcon-H1 | 256K | ❌ |
| *CMU/Princeton/Together/Cartesia* | **Mamba-3** | SSM (complex state, MIMO) | research | n/a | ❌ |

---

### 10.2 Mechanism adoption matrix

Which of the seven families (Part 1) each lab actually uses. **●** = signature/primary, **○** =
uses it, **–** = no.

| Lab | 1 Learned sparse | 2 Static sparse | 3 Linear/recurrent | 4 KV compression | 5 Positional | 6 Systems | 7 Context-as-env |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **DeepSeek** | ● NSA/DSA | ○ 128-tok window | – | ● MLA, CSA/HCA | ○ RoPE | ○ FP8 indexer | ○ OCR |
| **Z.ai** | ● DSA (adopted) | – | – | ● MLA | ○ partial RoPE | ● IndexShare | – |
| **MiniMax** | ● MSA | ○ (failed SWA expt) | ● Lightning (abandoned) | ○ GQA | – | ○ exp-free topk | – |
| **Moonshot** | ● MoBA (abandoned) | – | ● KDA | ● Gated MLA | ● NoPE | ○ DPLR kernel | – |
| **Qwen** | ○ MInference | – | ● Gated DeltaNet | ○ GQA | ● DCA+YaRN | ○ chunked prefill | ● CPRS |
| **Ant** | – | – | ● Lightning+MLA | ● MLA | ○ RoPE-on-linear | ● ZeCO, stability | – |
| **Tencent** | – | – | ● Mamba-2 | ● GQA + KV compress | – | – | – |
| **OpenBMB** | ● InfLLM-v2 | – | ● Lightning | – | ● HyPE | – | – |
| **Thinking Machines** | – | ● SWA 512 5:1 | ○ SConv | ● asymmetric GQA | ● relative bias | ○ MTP | – |
| **Xiaomi** | – | ● SWA 128 5:1 | – | ○ (6× via window) | ○ learned sink | – | – |
| **OpenAI** | – | ● SWA 128 1:1 | – | ○ GQA | ● sink bias + YaRN | ○ MXFP4 | – |
| **NVIDIA** | – | ○ SWA in Jet | ● Mamba-2, JetBlock | – | ○ meta tokens | ● PostNAS, Star Attn | – |
| **Microsoft** | ● SeerAttention | ○ SWA in Samba | ● RetNet, Mamba+SWA | ● YOCO, GMU | ● LongRoPE/2 | ● MTraining, MInference | – |
| **Google** | – | ● local-global 5:1 | ● RG-LRU/Griffin | ○ MQA local | ○ RoPE base scaling | ○ MoD | ● Titans/Miras memory |
| **Meta** | – | ○ (sinks discovered) | ○ MEGA | ○ GQA | ● CoPE, iRoPE | – | ○ Memory Layers |
| **ByteDance** | ○ SSA | – | ○ Sparse State Exp. | ● UltraMemV2 | – | ● ByteScale 2048K | – |
| **Prime Intellect** | – | – | – | – | – | ● CP to 98K | ● self-managed ctx |

Reading down the columns: **static sparsity (col 2) and KV compression (col 4) are the most widely
adopted**; **learned sparse (col 1) is concentrated in exactly four labs**; **context-as-environment
(col 7) is the least explored and the only column where no frontier pretrainer has committed.**

---

### 10.3 The five disputed questions, by lab

| Question | Position A | Position B | Unresolved because |
|---|---|---|---|
| **Sparse or linear?** | **Sparse:** DeepSeek, Z.ai, MiniMax (after reversal), OpenBMB (partly) | **Linear:** Moonshot, Qwen, Ant, Tencent, OpenBMB (partly) | No matched-recipe comparison above ~1.3B; MiniMax says deficits appear only at scale |
| **Does efficient attention survive conversion?** | **No** — MiniMax: retrieval/induction heads form early and can't be adjusted after | **Yes** — Ant migrated to hybrid linear at **1T**; InfLLM-v2/MoBA are dense↔sparse switchable | Both are production claims; neither published the other's ablation |
| **Where does position go?** | **NoPE on full layers:** Moonshot, Meta (iRoPE) | **RoPE on linear layers:** OpenBMB (HyPE), Ant · **finite-span relative bias:** Thinking Machines · **RoPE+YaRN:** Qwen, OpenAI | Four working 1M models, four different answers |
| **Is a learned selector needed at all?** | **Yes:** DeepSeek, MiniMax, Z.ai, Microsoft | **No:** Thinking Machines (1M), Xiaomi (256K), OpenAI (131K) — SWA + sink/bias only | Inkling reaches 1M with 2023-era ingredients; no head-to-head exists |
| **Should long context be an attention problem?** | **Yes** — everyone in Parts 6–8 | **No** — MIT/Prime Intellect (RLM), Qwen (CPRS), Google (memory modules) | RLM reports beating scaffolds by 13–130% but has no frontier pretrainer behind it |

---

### 10.4 Paper index by lab

Every arXiv ID surfaced in this document, grouped. **Bold** = the one to read first per lab.

| Lab | Papers |
|---|---|
| **DeepSeek** | **2405.04434** (MLA/V2) · 2412.19437 (V3) · **2502.11089** (NSA) · 2505.09343 (HW insights) · 2510.18234 (OCR) · **2512.02556** (V3.2/DSA) · **2606.19348** (V4 CSA/HCA) |
| **Moonshot** | 2501.12599 (K1.5) · **2502.13189** (MoBA) · 2502.16982 (Muon) · 2507.20534 (K2) · **2510.26692** (Kimi Linear/KDA) · 2603.15031 (Attention Residuals) · **2607.24653** (K3) |
| **Qwen** | 2407.10671 (Qwen2/DCA) · 2501.15383 (Qwen2.5-1M) · **2505.06708** (Gated Attention) · 2505.18092 (QwenLong-CPRS) · Qwen3-Next / 3.5 / 3.6 (blogs) |
| **Z.ai** | 2308.14508 (LongBench) · 2401.18058 (LongAlign) · 2406.12793 (GLM-4) · 2412.15204 (LongBench v2) · **2508.06471** (GLM-4.5) · **2602.15763** (GLM-5) · GLM-5.2/IndexShare (blog) · 2408.07055 / 2409.02897 / 2410.21252 (LongWriter/Cite/Reward) |
| **MiniMax** | 2307.14995 (TransNormerLLM) · 2401.04658 (Lightning-2) · **2501.08313** (MiniMax-01) · 2506.13585 (M1) · **M2 post-mortem (blog)** · **2606.13392** (MSA) |
| **Ant Group** | **2510.19338** (Ring-linear 2.0) · 2507.01004 (ZeCO) · **2606.15079** (Ling & Ring 2.6) · Ling/Ring-2.5-1T (release) |
| **ByteDance** | 2411.12364 (UltraMem) · **2502.21231** (ByteScale) · 2507.16577 (Sparse State Expansion) · 2508.18756 (UltraMemV2) · 2510.17896 (LC Attention Benchmark) · Seed-OSS (release) · *unverified:* 2510.20787, 2511.20102 |
| **Tencent** | 2411.02265 (Hunyuan-Large) · **2505.15431** (Hunyuan-TurboS) |
| **Xiaomi** | 2505.07608 (MiMo-7B) · **2601.02780** (MiMo-V2-Flash) |
| **OpenBMB** | 2402.04617 (InfLLM) · **2506.07900** (MiniCPM4/InfLLM-v2) · **2602.11761** (MiniCPM-SALA) |
| **OpenAI** | **2508.10925** (gpt-oss model card) |
| **Thinking Machines** | *no paper* — HF release post + Raschka teardown + vLLM day-0 |
| **NVIDIA** | 2411.13676 (Hymba) · 2411.17116 (Star Attention) · **2504.03624** (Nemotron-H) · 2508.14444 (Nano 2) · **2508.15884** (Jet-Nemotron/PostNAS) · 2512.20856 (Nemotron 3) |
| **Microsoft** | 2307.08621 (RetNet) · 2307.02486 (LongNet) · 2402.13753 (LongRoPE) · 2405.05254 (YOCO) · **2406.07522** (Samba) · 2407.02490 (MInference) · **2410.13276** (SeerAttention) · 2410.05258 (DIFF) · 2502.20082 (LongRoPE2) · 2506.08889 (SeerAttention-R) · **2507.06607** (SambaY/GMU) · 2510.18830 (MTraining) |
| **Google** | 2001.04451 (Reformer) · 2007.14062 (BigBird) · 2009.14794 (Performer) · 2203.08913 (Memorizing) · 2203.07852 (Block-Recurrent) · **2402.19427** (Griffin/Hawk) · 2403.05530 (Gemini 1.5) · 2404.07839 (RecurrentGemma) · **2404.07143** (Infini-attention) · 2404.02258 (MoD) · 2410.02703 (Selective Attn) · **2501.00663** (Titans) · 2503.19786 (Gemma 3) · **2504.13173** (Miras) · 2505.23735 (ATLAS) · 2507.10524 (MoR) |
| **Meta** | 2209.10655 (MEGA) · 2305.07185 (MEGABYTE) · **2309.16039** (Llama 2 Long) · 2309.17453 (StreamingLLM, w/ MIT) · **2405.18719** (CoPE) · 2412.09764 (Memory Layers) · 2412.09871 (BLT) · 2504.00927 (Multi-Token Attention) · Llama 4/iRoPE (blog) |
| **Prime Intellect** | 2412.01152 (INTELLECT-1) · 2505.07291 (INTELLECT-2) · **2512.16144** (INTELLECT-3) · RLM advocacy (blog) |
| **MIT CSAIL** | **2512.24601** (Recursive Language Models) |
| **Others** | AI21 **2403.19887** (Jamba) · TII **2507.22448** (Falcon-H1) · Mistral 2310.06825 · Berkeley 2310.01889 (Ring Attention) · Stanford 2205.14135 (FlashAttention), 2312.00752 (Mamba), 2405.21060 (Mamba-2) · **2603.15569** (Mamba-3) · AI2 2004.05150 (Longformer) · 2412.06464 (Gated DeltaNet) · 2605.22791 (Gated DeltaNet-2) |

---

### 10.5 One-paragraph read

**Eighteen labs, three genealogies.** DeepSeek/Z.ai/MiniMax converged on *learned sparsity over
compressed KV*; Moonshot/Qwen/Ant/Tencent on *linear or SSM recurrence*; OpenAI/Xiaomi/Thinking
Machines on *static windows plus a learned sink or bias* — and that third group, the cheapest to
build, currently holds the largest verified context window (Inkling, 1M). Microsoft and Google
invented much of what the Chinese labs shipped (SeerAttention, MInference, Mamba-2 duality, the delta
rule) without deploying it at frontier scale; NVIDIA is alone in treating hybrid placement as a search
problem; Meta alone declined sparsity outright and published the FLOP-crossover number justifying it;
ByteDance alone proved the 2048K-context training systems; and MIT/Prime Intellect alone argue the
whole framing is wrong. Two labs — Moonshot and MiniMax — publicly abandoned mechanisms they
themselves invented, in opposite directions. **No controlled comparison exists at the scale where the
disagreements actually matter.**

---

## Verification note

Every arXiv ID above traces to a paper surfaced in web search. For 2026 IDs this rests on search
metadata and abstracts rather than a full read of each paper — pull the PDF before citing. The
closed-model claims in Part 2 are explicitly marked unconfirmed because they rest on secondary
blogs only.

**Known coverage limits of this document:**
- The first draft **omitted MoBA entirely**, and with it the whole SeerAttention → NSA/MoBA →
  InfLLM-v2 → DSA → MSA learned-block-gate lineage now in Part 5. The cause was searching by model
  name and by the phrase "sparse attention"; MoBA's title says *Mixture of Block Attention* and
  SeerAttention's says *intrinsic sparse attention*, so neither surfaced. **If something else is
  missing, it is most likely for the same reason — a paper that describes this mechanism class
  without using the word "sparse."**
- Inkling and gpt-oss architectural details come from vendor blogs, HF release posts, and Raschka's
  teardowns. Inkling shipped **without a technical report**; the HF post and Raschka's notes
  disagree on the SConv window, and I went with Raschka's.
- Part 6's per-lab tables are complete for *attention and long-context* artifacts only. Each lab has
  many other papers (RL, multimodal, agents, optimizers) deliberately excluded except where they
  bear on long context (e.g. Muon → K3's 1M curriculum).
- A handful of IDs are from recall rather than confirmed in this session's searches — verify before
  citing: LongWriter 2408.07055, LongCite 2409.02897, LongReward 2410.21252, InfLLM 2402.04617,
  Mamba-2 2405.21060, Transformer-XL 1901.02860, ETC 2004.08483, and most unhyperlinked 2020–2023
  entries. In Part 7 the same applies to ATLAS 2505.23735, Mixture-of-Recursions 2507.10524,
  Selective Attention 2410.02703, Mixture-of-Depths 2404.02258, Memory Layers 2412.09764, BLT
  2412.09871, MEGA 2209.10655, MEGABYTE 2305.07185, and LongRoPE2 2502.20082 — these were located
  by search but I did not open each abstract.
- Part 7's lab attributions are by *primary affiliation* and are imperfect: several papers are
  multi-institution (Transformer-XL = Google Brain + CMU; StreamingLLM = MIT + Meta + CMU;
  NSA = DeepSeek + Peking University; Mamba-3 = CMU/Princeton/Together/Cartesia). Where a paper
  straddles, I placed it with the lab that shipped or drove it.

## Curated indexes worth following

- [Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression)
- [long-context-attention (USP: unified/hybrid sequence parallel attention)](https://github.com/feifeibear/long-context-attention)
- [Raschka's LLM architecture gallery](https://sebastianraschka.com/llm-architecture-gallery/) —
  per-mechanism pages for MLA, SWA, NoPE, gated attention, DSA
- [Raschka: The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
