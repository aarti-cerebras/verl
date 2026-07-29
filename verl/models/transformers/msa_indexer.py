# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax Sparse Attention (MSA) Index Branch — the per-GQA-group block selector.

Faithful port of the MSA Index Branch (arXiv 2606.13392 §3.1, Eqs. 5-7) as implemented in vLLM's
``vllm/models/minimax_m3/``. Per layer it produces, for query token ``i`` and GQA group ``r``:

    S_idx[r, i, j] = ( q_idx[r, i] . k_idx[j] ) / sqrt(d_idx)          # token-level score  (Eq. 6, left)
    M_idx[r, i, b] = max_{j in block b, j <= i}  S_idx[r, i, j]        # block score        (Eq. 6, right)
    I[r, i]        = TopK_b( M_idx[r, i, :], k )                       # selected blocks    (Eq. 7)

Two projections only (Eq. 5): ``q_proj`` gives ONE index query head per GQA group, ``k_proj`` gives a
SINGLE index key head shared across groups (MQA-shaped, ``(N, 1, d_idx)`` in Algorithm 1 line 2). The
paper's index branch has no norms; vLLM's M3 adds a per-branch RMSNorm on q and k, and we follow vLLM
because we must serve with its kernels.

Three implementation details that MUST match the serving kernels, all verified against vLLM main
@4f56321d (see docs/qwen3_4b_msa/kl_loss.md §2 for the full ledger):

  1. **Gemma-style norm.** ``x * rsqrt(mean(x^2) + eps) * (1 + w)``, ``w`` zero-initialised, with ONE
     shared ``[d_idx]`` gain per branch (not per-head). The serving kernel hardcodes this form
     (``csrc/libtorch_stable/fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:17,135-145``) and asserts
     ``numel() == 128``. Training in the same parameterisation makes the trained object the served
     object — it is an exact reparameterisation of standard RMSNorm (``gain = 1 + w``), with identical
     gradients, so nothing is given up.
  2. **Norm THEN RoPE**, using the base model's own rotary function and tables, so the index branch's
     positional encoding is identical to the attention it distills.
  3. **Selection semantics.** Block grid anchored to ABSOLUTE position (``blk = pos // B_k``, never to
     the query tile); ``-inf`` masking applied BEFORE the max so a masked token can never win a block;
     the block containing the query counts as visible even when partial; forced blocks injected as
     sentinels BEFORE the top-k (local ``1e29``, init ``1e30``) so they consume slots INSIDE the ``k``
     budget; NaN/fully-masked blocks carry ``-1e30``, not ``-inf``.

The score is raw and unnormalized here. Both consumers live elsewhere: the Phase-1/2 KL softmaxes the
TOKEN-level scores over the support (``qwen3_msa.py``), and selection takes an exp-free top-k over the
BLOCK-level scores. Note ``1/sqrt(d_idx)`` affects only the KL temperature — top-k is invariant to
positive scaling, which is why vLLM's kernel omits the constant.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# Sentinel constants copied from vLLM's Triton top-k kernel
# (vllm/models/minimax_m3/common/ops/index_topk.py:226-244). Replicating the MECHANISM, not just the
# resulting counts, is what keeps train-time and serve-time selection identical.
FORCE_INIT_SCORE = 1e30  # init ("sink") blocks win first
FORCE_LOCAL_SCORE = 1e29  # the local/self block wins next
MASKED_SCORE = -1e30  # fully-masked, non-causal, or NaN blocks lose


@dataclass
class MSAConfig:
    """Configuration for the MSA index branch.

    Geometry fields (``hidden_size`` ... ``rms_norm_eps``) are forced from the live base-model config by
    ``qwen3_msa.build_msa_config``; the rest come from training-side overrides (flat ``msa_*`` keys on the
    HF config, mirroring the DSA path's ``dsa_*``).
    """

    enabled: bool = False

    # --- geometry, forced from the base model -------------------------------------------------------
    hidden_size: int = 2560
    num_heads: int = 32  # H_q
    num_kv_heads: int = 8  # H_kv == GQA groups == index query heads (vLLM asserts this equality)
    head_dim: int = 128  # d_h (main attention)
    index_dim: int = 128  # d_idx; MUST be 128 -- both vLLM backends report get_supported_head_sizes()->[128]
    rms_norm_eps: float = 1e-6

    # --- MSA selection ------------------------------------------------------------------------------
    block_size: int = 128  # B_k; vLLM hardcodes SPARSE_BLOCK_SIZE = 128
    top_k: int = 16  # k selected blocks per (query, group); also inside the SM100 path's {4,8,16,32}
    init_blocks: int = 0  # forced sink blocks. Paper C.2 + M3 ship 0: the sink is learned, not forced.
    local_blocks: int = 1  # forced local blocks; the paper forces exactly the "incomplete self block"
    score_type: str = "max"  # block pooling; "max" is the only mode M3 ships
    dense_prefix: int = 3  # layers [0, dense_prefix) stay dense (no indexer, no KL). Matches M3's
    # shipped sparse_attention_freq = [0]*3 + [1]*57. Overridden by `sparse_layers` when that is set.
    sparse_layers: Optional[str] = None  # explicit comma-separated layer ids that get an indexer, e.g.
    # "5,6,...,35" or the §12-motivated alternative (dense = {1,2,3,4}). vLLM accepts any per-layer list
    # ({i for i, f in enumerate(freq) if f != 0}), so this is a config-only choice with no code impact.

    # --- loss / training ----------------------------------------------------------------------------
    mode: str = "dense_warmup"  # "dense_warmup" (Phase 1) | "sparse" (Phase 2)
    index_score_scale: bool = True  # apply the paper's 1/sqrt(d_idx). Selection is invariant to it; it
    # only sets the Eq. 10 softmax temperature. vLLM's kernel omits it (the learned gain absorbs any
    # constant), so raw SERVING scores run ~sqrt(d_idx)=11.3x larger than these -- never compare
    # score magnitudes across train/serve, and use no absolute score thresholds in diagnostics.
    kl_block_size: int = 512  # query tile for the teacher/KL recompute; 512 -> 512 MiB per retained fp32
    # [1, H_kv, 512, 32768] tensor. Drop to 256 if tight.
    kl_reduction: str = "mean"  # per-layer KL -> loss reduction over layers: "mean" (default) | "sum".
    # Each layer's indexer parameters are DISJOINT, so dL/dtheta_i = dKL_i/dtheta_i and the reduction is a
    # pure gradient SCALE: sum == mean * n_sparse_layers, same optimum, equivalent to LR * n_layers. We
    # default to "mean" for both phases because (a) it keeps the reported grad_norm in a range where it is
    # still a usable diagnostic instead of pinned against verl's clip_grad=1.0 default -- the DSA Phase-1
    # runs sat at grad_norm ~330, i.e. ~330x clipping every step, and "sum" over 33 layers would push
    # ~33x deeper into that regime -- and (b) in Phase 1 the reduction carries no meaning at all (with the
    # base frozen there is no L_LM to trade against, so it only rescales the index LR).
    # *** Phase-2 lambda conversion ***: the paper's Algorithm 1 is L = L_LM + lambda * SUM_layers L_KL.
    # With "mean", matching a paper lambda requires lambda_ours = lambda_paper * n_sparse_layers (33 at the
    # planned config). Using a paper lambda verbatim under "mean" trains the indexer 33x weaker relative
    # to L_LM than intended. Set kl_reduction="sum" for a literally paper-faithful Phase 2.
    kl_checkpoint: bool = False  # recompute the per-tile score graph in backward. Required at 32K.
    diag_interval: int = 10  # compute block-level diagnostics every N forwards (they cost a top-k)
    log_per_layer: bool = False  # also emit per-layer scalar keys (debug breakdown)
    warmstart_path: Optional[str] = None  # consolidated state dict to warm-start from (see
    # scripts/dsa/consolidate_indexer_ckpt.py); loaded before FSDP wrap, so GPU-count-agnostic.
    full_support_kl_prob: float = 0.0  # Phase-2 escape hatch (phase2_plan R2). With this probability per
    # forward, compute the KL over the FULL causal support (the Phase-1 teacher) instead of the selected
    # set, while the LM path still runs sparse. The restricted KL gives the index branch no gradient about
    # blocks it failed to select -- a block ranked 17th never enters the loss -- so a small fraction of
    # full-support batches keeps the whole ranking calibrated. Costs one dense attention pass
    # (8.80 TFLOP/layer at 32K) when it fires, i.e. 5% ~= +4% attention FLOPs. 0.0 = off.
    compile_teacher: bool = True  # torch.compile the Eq. 9 teacher (`_group_teacher`). It is the largest
    # single cost in Phase 1 and is bandwidth-bound; compiling fuses the upcast/bias-add/softmax/group-sum
    # chain for a measured 2.70x at 32K, with LOWER error against an fp64 reference than eager
    # (8.42e-05 vs 1.62e-04). Costs ~0.4 GB extra transient peak per tile. Set False to debug.
    serving_compat: bool = True  # enforce the dims the vLLM MSA kernels require (d_idx == B_k == 128).
    # Unit tests that exercise the algebra on toy shapes (e.g. the docs' B_k=2, T=8 worked example) set
    # this False; any real run must leave it True or the checkpoint cannot be served.

    def __post_init__(self):
        if self.kl_reduction not in ("sum", "mean"):
            raise ValueError(f"kl_reduction must be 'sum' or 'mean', got {self.kl_reduction!r}")
        if self.score_type != "max":
            raise ValueError(f"score_type must be 'max' (the only mode vLLM ships), got {self.score_type!r}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")
        if self.serving_compat:
            # Not soft preferences: MiniMaxM3IndexerBackend.get_supported_head_sizes() -> [128],
            # MinimaxM3QKVParallelLinearWithIndexer assumes index_head_size == head_size, and the Triton
            # kernels hardcode SPARSE_BLOCK_SIZE = 128.
            if self.index_dim != 128:
                raise ValueError(f"index_dim must be 128 to serve with the vLLM MSA kernels, got {self.index_dim}")
            if self.block_size != 128:
                raise ValueError(f"block_size must be 128 (vLLM SPARSE_BLOCK_SIZE), got {self.block_size}")
        if self.local_blocks + self.init_blocks > self.top_k:
            raise ValueError(
                f"forced blocks ({self.init_blocks} init + {self.local_blocks} local) exceed the top_k "
                f"budget ({self.top_k}); forced blocks live INSIDE k, they are not additional"
            )

    @property
    def group_size(self) -> int:
        """G = H_q / H_kv — the number of query heads sharing one index query / one selection."""
        return self.num_heads // self.num_kv_heads

    def layer_is_sparse(self, layer_idx: int) -> bool:
        """Whether this layer gets an index branch (and therefore a KL term)."""
        if self.sparse_layers:
            ids = {int(s) for s in str(self.sparse_layers).replace(" ", "").split(",") if s != ""}
            return layer_idx in ids
        return layer_idx >= self.dense_prefix


class MSARMSNorm(nn.Module):
    """Gemma-style RMSNorm: ``x * rsqrt(mean(x^2) + eps) * (1 + weight)``, zero-initialised.

    Matches vLLM's ``MiniMAXGemmaRMSNorm`` (``nn.Parameter(torch.zeros(...))`` + FlashInfer
    ``gemma_rmsnorm``) and the fused serving kernel, which computes ``1.0f + weight[dim]`` in fp32. ONE
    shared gain vector is applied to every head — the serving kernel asserts ``numel() == 128``, so a
    per-head ``[H, 128]`` gain would be an architecture mismatch, not a reparameterisation.

    Zero init makes the layer start as pure RMS normalisation (gain 1.0), which is the right starting
    point for a freshly-added branch. Keep these parameters out of weight decay: in this
    parameterisation L2 shrinks the gain toward 1 (harmless) rather than toward 0 (which would kill the
    branch), but excluding them removes the question entirely.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (x * (1.0 + self.weight.float())).to(in_dtype)


class MSAIndexer(nn.Module):
    """The MSA Index Branch for one attention layer: 2 projections + 2 Gemma norms, shared rotary.

    Parameter count per layer at Qwen3-4B sizing: ``2560*8*128 + 2560*128 = 2.95M`` (plus 256 norm
    gains) — 97M over 33 sparse layers, 2.4% of the 4.02B backbone.
    """

    def __init__(self, cfg: MSAConfig):
        super().__init__()
        self.cfg = cfg
        self.num_index_heads = cfg.num_kv_heads  # one index query head per GQA group (Eq. 5)
        self.index_dim = cfg.index_dim
        # Names mirror the serving-side checkpoint contract: vLLM folds `index_q_proj`/`index_k_proj`
        # into its single fused QKV GEMM via `stacked_params_mapping`, so we keep them separate here and
        # let the loader do the fusion (vllm/models/minimax_m3/nvidia/model.py:903-909).
        self.index_q_proj = nn.Linear(cfg.hidden_size, self.num_index_heads * cfg.index_dim, bias=False)
        self.index_k_proj = nn.Linear(cfg.hidden_size, cfg.index_dim, bias=False)  # SINGLE shared key head
        self.index_q_norm = MSARMSNorm(cfg.index_dim, eps=cfg.rms_norm_eps)
        self.index_k_norm = MSARMSNorm(cfg.index_dim, eps=cfg.rms_norm_eps)
        self.softmax_scale = cfg.index_dim**-0.5 if cfg.index_score_scale else 1.0

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_fn=None):
        """Project + norm + RoPE. Returns ``(q_idx [b, H_kv, T, d_idx], k_idx [b, 1, T, d_idx])``.

        ``hidden_states`` is detached here, implementing Eq. 11's ``stopgrad(X)``: the KL must not reach
        the backbone through the residual stream. The paper's B.3 ablation is the reason — without this
        detach, larger KL coefficients spiked the gradient norm and diverged the LM loss within a few
        hundred steps, and even at stable coefficients short-context benchmarks regressed (the backbone
        can cheat by flattening its own attention instead of improving the indexer).

        ``rope_fn(q, k, cos, sin)`` is the BASE MODEL's own rotary function (e.g. transformers'
        ``apply_rotary_pos_emb`` for Qwen3), passed in so the index branch cannot drift from the
        attention it distills. vLLM does the same thing structurally:
        ``self.index_rotary_emb = self.rotary_emb``.

        Call this through ``__call__`` (i.e. ``indexer(...)``), never as an unbound method: a direct
        method call bypasses ``nn.Module.__call__``, so FSDP2's forward hooks on a separately-wrapped
        indexer unit never fire (no all-gather, no pre-backward gate, no grad reduce-scatter).
        """
        x = hidden_states.detach()  # Eq. 11
        b, t, _ = x.shape
        q = self.index_q_norm(self.index_q_proj(x).view(b, t, self.num_index_heads, self.index_dim)).transpose(1, 2)
        k = self.index_k_norm(self.index_k_proj(x).view(b, t, 1, self.index_dim)).transpose(1, 2)
        if rope_fn is not None:
            q, k = rope_fn(q, k, cos, sin)  # norm THEN rope, per the fused serving kernel
        return q, k

    def scores(self, q_idx: torch.Tensor, k_idx: torch.Tensor, attn_bias: Optional[torch.Tensor] = None):
        """Token-level index scores ``S_idx``: ``[b, H_kv, Tq, T]``.

        ``q_idx`` may be a query tile ``[b, H_kv, Tq, d]``; ``k_idx`` is always the full key sequence
        ``[b, 1, T, d]`` and broadcasts over the group axis (MQA). ``attn_bias`` is the additive
        causal/document/padding mask ``[b, Tq, T]`` (0 or ``-inf``), broadcast over groups.
        """
        s = torch.matmul(q_idx.float(), k_idx.float().transpose(-1, -2)) * self.softmax_scale
        if attn_bias is not None:
            s = s + attn_bias.unsqueeze(1)  # [b, 1, Tq, T] -> broadcast over H_kv
        return s

    def block_scores(self, token_scores: torch.Tensor) -> torch.Tensor:
        """Max-pool token scores into block scores ``M_idx``: ``[b, H_kv, Tq, n_blocks]`` (Eq. 6, right).

        The masking must already be baked into ``token_scores`` as ``-inf`` — vLLM applies the causal
        ``-inf`` inside the score kernel *before* ``tl.max``, so a masked token can never win a block
        that straddles the causal frontier or a document boundary. ``-inf`` blocks are then rewritten to
        the kernel's ``-1e30`` sentinel (its top-k loads masked entries as ``other=-1e30`` and maps NaN
        to the same value), which keeps them representable in fp32 and orderable.

        The block grid is anchored to ABSOLUTE key position: the last dim of ``token_scores`` is the full
        sequence, so block ``b`` is always keys ``[b*B_k, (b+1)*B_k)`` regardless of which query tile we
        are in. Anchoring to the tile instead would silently shift every block id.
        """
        b, h, tq, t = token_scores.shape
        bk = self.cfg.block_size
        n_blocks = (t + bk - 1) // bk
        pad = n_blocks * bk - t
        s = token_scores
        if pad:
            # Pad with -inf so a partial trailing block scores only over its real keys.
            s = torch.nn.functional.pad(s, (0, pad), value=float("-inf"))
        m = s.view(b, h, tq, n_blocks, bk).amax(dim=-1)
        # -inf (no visible key) / NaN -> the kernel's finite sentinel.
        return torch.where(torch.isfinite(m), m, torch.full_like(m, MASKED_SCORE))

    def select_blocks(self, block_scores: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        """Top-k block selection with forced blocks, replicating vLLM's sentinel mechanism.

        Args:
            block_scores: ``[b, H_kv, Tq, n_blocks]`` from ``block_scores`` (already sentinel-cleaned).
            query_pos: ``[Tq]`` ABSOLUTE key-space position of each query row in the tile.

        Returns ``[b, H_kv, Tq, k]`` int64 block ids, with ``-1`` in slots beyond the number of visible
        blocks (vLLM writes ``-1`` there too). Forced blocks consume slots INSIDE ``k``: the paper is
        explicit that the local block "reserves one block slot and leaves the remaining slots to be
        chosen by the Index Branch".

        ``valid_blocks = (pos + B_k) // B_k`` follows the kernel exactly, so the partial block CONTAINING
        the query counts as visible — that block is the forced "incomplete self block" of paper C.2.
        """
        cfg = self.cfg
        b, h, tq, n_blocks = block_scores.shape
        dev = block_scores.device
        bk, k = cfg.block_size, cfg.top_k

        blk_ids = torch.arange(n_blocks, device=dev)  # [n_blocks]
        valid_blocks = (query_pos.to(dev) + bk) // bk  # [Tq] -- includes the partial self block
        causal = blk_ids[None, :] < valid_blocks[:, None]  # [Tq, n_blocks]
        local = blk_ids[None, :] >= (valid_blocks[:, None] - cfg.local_blocks).clamp_min(0)
        init = blk_ids[None, :] < cfg.init_blocks

        s = block_scores.masked_fill(~causal[None, None], MASKED_SCORE)
        if cfg.init_blocks:
            s = torch.where((causal & init)[None, None], torch.full_like(s, FORCE_INIT_SCORE), s)
        if cfg.local_blocks:
            s = torch.where((causal & local)[None, None], torch.full_like(s, FORCE_LOCAL_SCORE), s)

        kk = min(k, n_blocks)
        idx = s.topk(kk, dim=-1).indices.to(torch.int64)  # [b, H_kv, Tq, kk] (unsorted, like the kernel)
        if kk < k:  # pad the slot axis so the shape is always [..., k]
            idx = torch.cat([idx, idx.new_full((b, h, tq, k - kk), -1)], dim=-1)
        # Slots beyond the visible-block count carry -1, matching the kernel's `store_mask & valid_mask`.
        slot = torch.arange(k, device=dev)
        return torch.where(slot[None, None, None, :] < valid_blocks[None, None, :, None], idx, torch.full_like(idx, -1))

    def block_token_mask(self, selected: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Expand selected block ids to a token mask ``[b, H_kv, Tq, T]`` (True = in the selected set).

        This is the Phase-2 KL support ``I_tok`` = the tokens induced by the selected blocks, and in
        Phase 1 it is only used by diagnostics (Phase 1's support is the full causal set).
        """
        b, h, tq, k = selected.shape
        bk = self.cfg.block_size
        n_blocks = (seq_len + bk - 1) // bk
        hit = selected.new_zeros((b, h, tq, n_blocks + 1), dtype=torch.bool)
        hit.scatter_(-1, selected.clamp_min(-1) + 1, True)  # id -1 lands in the throwaway column 0
        hit = hit[..., 1:]
        return hit.repeat_interleave(bk, dim=-1)[..., :seq_len]
