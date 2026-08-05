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
"""Unit test for ``_selected_token_index`` — the partial-final-block aliasing bug.

**The bug.** ``tok`` is built as ``block_id * B_k + offset`` and then ``clamp_max(seq_len - 1)``.
When the final block is only partly filled (``seq_len % B_k != 0``) its surplus slots are pinned
onto ``seq_len - 1``, aliasing onto the last real token. ``slot_ok`` used to test only "did this
slot come from a selected block", which is True for them, so nothing masked them. Every query
except the one at ``seq_len - 1`` is saved by the causal mask (the alias is in its future); that
last query is not, and attends to the final token ``B_k - (seq_len % B_k)`` times over.

**Why the existing suite missed it.** ``test_qwen3_msa_phase2.py`` D1 asserts dense equivalence,
which would have caught this — but at ``--seq-len 512``, and ``512 % 128 == 0``. A block-aligned
length is precisely the one case with zero surplus slots. This file therefore sweeps NON-aligned
lengths, which is the whole point.

CPU-only, no model, runs in under a second.

Run:
  cd <repo> && PYTHONPATH=$(pwd) python3 tests/msa/test_selected_token_index.py
"""

import sys
from types import SimpleNamespace

import torch

from verl.models.transformers.qwen3_msa import _selected_token_index

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_res = []


def check(name, cond, detail=""):
    _res.append(bool(cond))
    print(f"  [{OK if cond else BAD}] {name}" + (f"  ({detail})" if detail else ""))


def ix(bk):
    return SimpleNamespace(cfg=SimpleNamespace(block_size=bk))


def visible(tok, ok, q):
    """Token positions the query at `q` actually attends to (causal + slot_ok)."""
    return [int(p) for p, v in zip(tok.flatten().tolist(), ok.flatten().tolist()) if v and p <= q]


def main():
    print("\n-- 1. every slot marked real must hold a token that exists --")
    for bk in (8, 128):
        for seq_len in (1, 5, bk - 1, bk, bk + 1, 3 * bk - 7, 3 * bk):
            n_blocks = (seq_len + bk - 1) // bk
            sel = torch.arange(n_blocks).view(1, 1, 1, n_blocks)
            tok, ok = _selected_token_index(ix(bk), sel, seq_len)
            bad = int(((tok >= seq_len) & ok).sum())
            if bad:
                check(f"bk={bk} seq_len={seq_len}: no out-of-range slot marked real", False,
                      f"{bad} bad slots")
                return finish()
    check("no slot marked real holds an out-of-range token (all bk x seq_len combos)", True)

    print("\n-- 2. the LAST query sees each visible token EXACTLY once (no aliasing) --")
    for bk in (8, 128):
        worst = None
        for seq_len in range(1, 3 * bk + 1):
            n_blocks = (seq_len + bk - 1) // bk
            sel = torch.arange(n_blocks).view(1, 1, 1, n_blocks)
            tok, ok = _selected_token_index(ix(bk), sel, seq_len)
            seen = visible(tok, ok, seq_len - 1)
            if sorted(seen) != list(range(seq_len)):
                worst = (seq_len, seen[:8], len(seen))
                break
        check(f"bk={bk}: last query attends to 0..seq_len-1 once each, for every seq_len in "
              f"1..{3 * bk}", worst is None, "" if worst is None else f"seq_len={worst[0]} saw "
              f"{worst[2]} slots, head {worst[1]}")

    print("\n-- 3. unselected (-1) slots masked; partial block trimmed, full block untouched --")
    bk, seq_len = 8, 13  # blocks: 0 -> tokens 0..7 (FULL), 1 -> tokens 8..12 (PARTIAL, 3 surplus)
    tok, ok = _selected_token_index(ix(bk), torch.tensor([[[[0, -1]]]]), seq_len)
    o = ok.flatten().tolist()
    check("the -1 block contributes no live slot", not any(o[bk:]), f"tail = {o[bk:]}")
    check("a FULL block keeps all its slots", o[:bk] == [True] * bk, f"block 0 = {o[:bk]}")

    tok, ok = _selected_token_index(ix(bk), torch.tensor([[[[1, -1]]]]), seq_len)
    o = ok.flatten().tolist()
    check("a PARTIAL block keeps only its 5 real slots (was 8 before the fix)",
          o[:bk] == [True] * 5 + [False] * 3, f"block 1 = {o[:bk]}")
    check("the trimmed slots are exactly the ones clamped onto seq_len-1",
          [int(p) for p, v in zip(tok.flatten().tolist()[:bk], o[:bk]) if not v] == [seq_len - 1] * 3,
          f"tok block 1 = {tok.flatten().tolist()[:bk]}")

    print("\n-- 4. batch > 1 (micro_batch_size_per_gpu > 1): shape and per-row correctness --")
    bk, seq_len, b, h, tq, k = 128, 300, 4, 8, 6, 4  # 300 % 128 = 44 -> 84 surplus slots
    torch.manual_seed(0)
    n_blocks = (seq_len + bk - 1) // bk
    sel = torch.randint(-1, n_blocks, (b, h, tq, k))
    tok, ok = _selected_token_index(ix(bk), sel, seq_len)
    check("shapes", tuple(tok.shape) == (b, h, tq, k * bk) and tuple(ok.shape) == (b, h, tq, k * bk),
          f"{tuple(tok.shape)}")
    check("no live slot out of range, anywhere in the batch", int(((tok >= seq_len) & ok).sum()) == 0)
    check("every live slot came from a selected block",
          bool((ok.reshape(b, h, tq, k, bk).any(-1) <= (sel >= 0)).all()))
    # `seq_len` is the TENSOR width, shared by the batch, so the guard is row-independent: rows
    # shorter than the width are additionally protected by the padding mask in the bias gather.
    per_row = [int(((tok[i] >= seq_len) & ok[i]).sum()) for i in range(b)]
    check("guard is row-independent (tensor-width based)", per_row == [0] * b, f"{per_row}")

    return finish()


def finish():
    n = len(_res)
    print(f"\n{sum(_res)}/{n} checks passed")
    return 0 if all(_res) else 1


if __name__ == "__main__":
    sys.exit(main())
