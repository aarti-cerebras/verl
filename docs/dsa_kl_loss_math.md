# DSA Phase-1 Indexer KL Loss — Math Across All Dimensions

How the indexer KL is computed and normalized, from a single key all the way up to
one optimizer step across many GPUs. Ties the equations to the code:

- Per-layer KL: `verl/models/transformers/minicpm_dsa.py::_dense_warmup_kl`
- Layer aggregation: `verl/models/transformers/minicpm_dsa.py::install_kl_accumulation`
- Global normalization (Layer 2): `verl/workers/utils/losses.py::indexer_kl_loss`
- Global token count all-reduce: `verl/workers/engine/fsdp/transformer_impl.py` (`batch_num_tokens`)

> Rendering: this file uses `$...$` / `$$...$$` math. GitHub, VS Code (with a Markdown+Math
> preview), and most Markdown viewers render it. In VS Code: open Preview (`Ctrl/Cmd+Shift+V`).

---

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| $\ell = 1..L$ | layer | `kl_stack` over layers |
| $r = 1..R$ | GPU / dp rank, $R = \texttt{dp\_size}$ | `dp_size` |
| $m$ | a micro-batch on rank $r$ | grad-accumulation loop |
| $i$ | a real (non-pad) query row = a (sequence, position) pair | rows of `kl_blk` |
| $j \in \mathcal{A}_i$ | keys query $i$ may attend (causal $\wedge$ same-doc $\wedge$ non-pad) | `allow` mask |
| $H$ | attention heads | `H` |
| $c_{r,m}$ | # valid (non-pad) query rows in micro-batch $m$ $=$ `total_cnt` | `mb_valid` = `tu.num_valid_queries(data)` |
| $C$ | $\sum_r \sum_m c_{r,m}$ = global valid-query count | `batch_num_valid_queries` |

**Source of the count:** the normalizer is the **valid (non-pad) query count** the KL actually averages
over (`total_cnt`), computed from `input_ids` (nested offsets) / `attention_mask`, **not** from `loss_mask`.
So the equations below hold for *any* `loss_mask` (all-ones Phase-1, or prompt-masked). See the note at the end.

---

## 1. Atomic unit — per (layer, query row) KL

For layer $\ell$ and query row $i$, the KL between the base attention (teacher) and the
indexer (student), summed over the allowed keys:

$$
\mathrm{KL}_{\ell,i} \;=\; \sum_{j\in\mathcal{A}_i} p^{\ell}_{ij}\,\log\!\frac{p^{\ell}_{ij}}{q^{\ell}_{ij}}
$$

Teacher (detached, head-averaged softmax attention) and student (indexer softmax):

$$
p^{\ell}_{ij}=\frac{1}{H}\sum_{h=1}^{H}\mathrm{softmax}_{j\in\mathcal{A}_i}\!\big(s^{\ell}_{ihj}\big),
\qquad
s^{\ell}_{ihj}=\text{scale}\cdot\langle q^{\ell}_{ih},\,k^{\ell}_{jh}\rangle
$$

$$
q^{\ell}_{ij}=\mathrm{softmax}_{j\in\mathcal{A}_i}\!\big(I^{\ell}_{ij}\big)
$$

This is the sum-over-keys at `minicpm_dsa.py` lines 291–294. Gradient flows **only** through
$q$ (the indexer logits $I$); $p$ is computed under `no_grad`.

## 2. Per micro-batch (what `model._dsa_indexer_kl` holds)

Mean over rows (`_dense_warmup_kl` return, line 368), then mean over layers
(`install_kl_accumulation`, line 156):

$$
\mathrm{kl}_{r,m} \;=\; \frac{1}{L}\sum_{\ell=1}^{L}\;\frac{1}{c_{r,m}}\sum_{i\in m}\mathrm{KL}_{\ell,i}
$$

This scalar is already a mean, so it is ~invariant to micro-batch size, sequence length, and $L$.

## 3. Per-micro-batch loss returned to `backward()`

`indexer_kl_loss` (`losses.py`):

$$
\mathcal{L}_{r,m} \;=\; \mathrm{kl}_{r,m}\cdot\frac{c_{r,m}}{C}\cdot R
$$

with $c_{r,m}=\texttt{tu.num\_valid\_queries(data)}$ (this micro-batch's valid query rows) and
$C=\texttt{batch\_num\_valid\_queries}$ (all-reduced global valid-query count, `transformer_impl.py`).

## 4. Effective optimized objective

The engine **sums** micro-batch losses (gradient accumulation) and FSDP/DDP **averages**
gradients across the $R$ ranks. So the objective whose gradient is actually applied is:

$$
\mathcal{L}_{\text{opt}} \;=\; \frac{1}{R}\sum_{r=1}^{R}\sum_{m}\mathcal{L}_{r,m}
$$

## 5. Closed form (substitute and simplify)

The $\cdot R$ cancels the $1/R$; the $c_{r,m}/C$ cancels the $1/c_{r,m}$ inside $\mathrm{kl}_{r,m}$:

$$
\boxed{\;\mathcal{L}_{\text{opt}}
\;=\;\frac{1}{L\,C}\sum_{\ell=1}^{L}\;\sum_{i\,\in\,\text{all real rows globally}}\;\sum_{j\in\mathcal{A}_i} p^{\ell}_{ij}\,\log\frac{p^{\ell}_{ij}}{q^{\ell}_{ij}}\;}
$$

In words: **mean over layers, mean over every real query row in the entire global batch, of the
(sum-over-allowed-keys) KL.**

- innermost $\sum_j$ — reduce over keys (masked to $\mathcal{A}_i$)
- $\sum_i / C$ — mean over all query rows across all sequences, micro-batches, and GPUs
- $\sum_\ell / L$ — mean over layers

Every batching detail — micro-batch split, $\texttt{dp\_size}$, per-sequence lengths — has
algebraically dropped out. Only $L$ and the global token count $C$ remain as normalizers. That
invariance (effective learning rate independent of GPU count / grad-accum split) is the entire
purpose of the $\cdot\, c_{r,m}/C \cdot R$ factor in step 3.

---

## Worked example — 2 GPUs, uneven micro-batches

Global batch = 8 sequences $\times$ 10 tokens; $R = \texttt{dp\_size} = 2$; 4 sequences per GPU.
Dynamic-bsz splits each GPU's sequences into uneven micro-batches (distinct KLs for realism):

| GPU | micro-batch | seqs | $c_{r,m}$ | $\mathrm{kl}_{r,m}$ |
|---|---|---|---|---|
| 0 | mb0 | A,B,C | 30 | 1.0 |
| 0 | mb1 | D     | 10 | 5.0 |
| 1 | mb2 | E,F   | 20 | 2.0 |
| 1 | mb3 | G,H   | 20 | 3.0 |

All-reduce SUM: $C = 30+10+20+20 = 80$.

**True target** (token-weighted global mean):

$$
\frac{30\cdot1.0 + 10\cdot5.0 + 20\cdot2.0 + 20\cdot3.0}{80}
= \frac{30+50+40+60}{80} = \frac{180}{80} = 2.25
$$

**Layer 2 per micro-batch** $\;\mathcal{L}_{r,m} = \mathrm{kl}_{r,m}\cdot \frac{c_{r,m}}{C}\cdot R$:

```
mb0: 1.0 · 30/80 · 2 = 0.75
mb1: 5.0 · 10/80 · 2 = 1.25    GPU0 backward-accumulates: 0.75 + 1.25 = 2.00
mb2: 2.0 · 20/80 · 2 = 1.00
mb3: 3.0 · 20/80 · 2 = 1.50    GPU1 backward-accumulates: 1.00 + 1.50 = 2.50
```

FSDP averages the two ranks' gradients:

$$
\frac{2.00 + 2.50}{2} = 2.25 \quad\checkmark
$$

Exactly the true global mean.

- $\cdot R$ (=2) cancels the FSDP $\div 2$ dp-average.
- $\cdot\, c_{r,m}/C$ converts the *summed* micro-batches into a *token-weighted mean*
  (weights over all micro-batches sum to $80/80 = 1$).

### Broken variants (same numbers)

**Naive** $\mathcal{L}_{r,m} = \mathrm{kl}_{r,m}$ (no normalization):

```
GPU0: 1.0 + 5.0 = 6.0 ;  GPU1: 2.0 + 3.0 = 5.0 ;  avg = (6.0+5.0)/2 = 5.5   ✗ (scales with #micro-batches)
```

**Unweighted mean over micro-batches** (forgetting $c_{r,m}$):

$$
\frac{1.0 + 5.0 + 2.0 + 3.0}{4} = 2.75 \quad\text{✗}
$$

mb1 has high KL (5.0) but only 10 tokens; without the $c_{r,m}$ weight it is over-counted.

---

## Single-row numeric sanity check (Step 1)

Query position $i=3$ in a doc of length $T=5$; causal $\Rightarrow \mathcal{A}_i=\{0,1,2,3\}$, key 4 masked.

- Teacher $p = [0.1, 0.2, 0.3, 0.4, 0]$
- Student $q = [0.25, 0.25, 0.25, 0.25, 0]$

$$
\mathrm{KL}(p\|q) = 0.1\ln\tfrac{0.1}{0.25} + 0.2\ln\tfrac{0.2}{0.25} + 0.3\ln\tfrac{0.3}{0.25} + 0.4\ln\tfrac{0.4}{0.25}
\approx 0.107 \text{ nats}
$$

That single number is one entry of `kl_blk` `[bsz, B]`; the `sum(dim=-1)` produced it from the 5 keys.

---

## Note — normalization is decoupled from `loss_mask`

The normalizer $c_{r,m}$ / $C$ is the **valid (non-pad) query count** (`tu.num_valid_queries`),
matching the rows the KL sums over (`total_cnt`) — **not** `loss_mask`. So $\mathcal{L}_{\text{opt}}$ is
the true global mean over valid queries for *any* `loss_mask`, including a prompt-masked
`MultiTurnSFTDataset` (which zeroes prompt tokens) or the all-ones Phase-1 `PackedPretrainDataset`.
Averaging the KL over **all** valid positions is the correct target for indexer training — the indexer
must select keys at every position at inference.

Earlier revisions normalized by `loss_mask.sum()` / `batch_num_tokens`, which coincides with the
valid-query count only when `loss_mask` is all-ones. That coupling was removed:

- `verl/utils/tensordict_utils.py::num_valid_queries` — the shared count (nested offsets / `attention_mask`)
- `verl/workers/engine/fsdp/transformer_impl.py` — all-reduces it into `batch_num_valid_queries`
- `verl/workers/utils/losses.py::indexer_kl_loss` — normalizes by it
