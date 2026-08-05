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
"""Pre-tokenized, VARIABLE-LENGTH SFT rows for MSA/DSA Phase-2 (sparse) training.

Reads parquet with an ``input_ids`` column and an optional ``loss_mask`` column — the format emitted by
``scripts/dsa/trajectories_to_sft_parquet.py --emit-input-ids``. One row is one conversation.

Why this exists rather than reusing an existing loader:

* ``PackedPretrainDataset`` reads ``input_ids`` but (a) **forces ``loss_mask`` to all-ones**, so the
  prompt would be trained on instead of only the model's own completion, and (b) **skips every row
  shorter than ``max_length``** (its data is uniformly 32,768). Behaviour-cloning rows are
  variable-length — p50 ~7K — so it would silently discard essentially the whole dataset.
* ``MultiTurnSFTDataset`` requires a ``messages`` column and templates **one turn at a time**
  (``multiturn_sft_dataset.py:216``). For a Qwen3 *Thinking* model that hits the template's
  reasoning-stripping branch and deletes every ``<think>`` trace — measured on a real row: 2,876 → 1,305
  characters, 55% of the row, and ~97% of a median math row. Upstream verl detects the mismatch in
  ``sanity_check`` and raises unless ``ignore_input_ids_mismatch=True``. That is precisely why this data
  is stored pre-tokenized: the splice is the only path that preserves the trace, so the correct loader is
  one that does no templating at all.

Contract with the engine (``pad_mode=no_padding``, ``micro_batch_size_per_gpu=1``,
``use_remove_padding=False``): ``SFTTensorCollator`` builds one jagged nested tensor per key
(``dataset_utils.py:52-80``), and the engine then pads each micro-batch to *its own* longest row
(``fsdp/transformer_impl.py:1132-1141``). With one sample per micro-batch that is the row's own length, so
**no padding is ever materialised** and ``attention_mask`` is legitimately all-ones. A step's cost is
therefore set by the longest row across ranks, not by any padded width — see ``length_tiers`` below.

Memory: the flat token buffer is taken straight from arrow and cast to ``int32`` (~4 bytes/token), not
built as ``list[list[int]]``. That distinction matters at this scale — Python ints are ~28 bytes each, so
the list route costs ~28 GB per rank at 1B tokens, times every rank.
"""

import logging
import os
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__file__)


class MSASFTDataset(Dataset):
    """Config keys (all read off the ``data`` node):

    ``max_length``      hard cap; rows longer than this are truncated (and counted, loudly).
    ``input_ids_key``   default ``input_ids``.
    ``loss_mask_key``   default ``loss_mask``; all-ones if the column is absent.
    ``length_tiers``    0 (default) = keep parquet row order. >0 = reorder rows so each consecutive
                        group of ``train_batch_size`` shares a length tier, then shuffle the groups.
                        Requires ``sampler_shuffle=False``, else the sampler re-permutes and the
                        grouping is destroyed. See the note on step cost below.
    ``tier_width``      tier width in tokens, default 2048.
    ``train_batch_size``/``seed`` used only when ``length_tiers > 0``.
    """

    def __init__(self, parquet_files, tokenizer, config, processor: Optional[object] = None, max_samples: int = -1):
        self.tokenizer = tokenizer
        self.max_length = int(config.get("max_length", 32768))
        ids_key = config.get("input_ids_key", "input_ids")
        mask_key = config.get("loss_mask_key", "loss_mask")

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        files = []
        for spec in parquet_files:
            # Accept a directory (the builders emit sharded train-*.parquet) as well as explicit files.
            files.extend(sorted(f.path for f in os.scandir(spec) if f.name.endswith(".parquet"))
                         if os.path.isdir(spec) else [spec])
        assert files, f"no parquet files resolved from {parquet_files}"

        ids_chunks, mask_chunks, lens = [], [], []
        n_trunc = 0
        for path in files:
            tbl = pq.read_table(path)
            assert ids_key in tbl.column_names, (
                f"{path}: no '{ids_key}' column (cols={tbl.column_names}). This loader needs PRE-TOKENIZED "
                f"rows; for a `messages` parquet use MultiTurnSFTDataset instead."
            )
            ids_arr = tbl.column(ids_key).combine_chunks()
            flat = np.asarray(ids_arr.values)
            offs = np.asarray(ids_arr.offsets, dtype=np.int64)

            if mask_key in tbl.column_names:
                m_arr = tbl.column(mask_key).combine_chunks()
                m_flat = np.asarray(m_arr.values)
                m_offs = np.asarray(m_arr.offsets, dtype=np.int64)
                assert np.array_equal(offs - offs[0], m_offs - m_offs[0]), (
                    f"{path}: '{ids_key}' and '{mask_key}' have different row lengths"
                )
            else:
                # Real pre-training text: every token is a target. Explicit, not a silent default.
                logger.warning("%s has no '%s' column -- training on ALL tokens of every row", path, mask_key)
                m_flat, m_offs = None, None

            for i in range(len(offs) - 1):
                a, b = int(offs[i]), int(offs[i + 1])
                n = b - a
                if n > self.max_length:
                    n_trunc += 1
                    b = a + self.max_length
                    n = self.max_length
                if n == 0:
                    continue
                ids_chunks.append(flat[a:b].astype(np.int32, copy=True))
                if m_flat is None:
                    mask_chunks.append(np.ones(n, dtype=np.int8))
                else:
                    ma = int(m_offs[i])
                    mask_chunks.append(m_flat[ma: ma + n].astype(np.int8, copy=True))
                lens.append(n)

        assert ids_chunks, "dataset is empty after loading"
        self.lengths = np.asarray(lens, dtype=np.int64)
        self.ids = ids_chunks
        self.masks = mask_chunks
        if n_trunc:
            logger.warning("truncated %d rows to max_length=%d", n_trunc, self.max_length)

        order = self._tiered_order(config) if int(config.get("length_tiers", 0) or 0) > 0 else None
        if order is not None:
            self.order = order
        else:
            self.order = np.arange(len(self.ids), dtype=np.int64)

        if max_samples is not None and max_samples > 0 and max_samples < len(self.order):
            # Floor to a whole number of batches so truncation can never split a length-tiered group.
            bsz = int(config.get("train_batch_size", 1) or 1)
            keep = max(bsz, (max_samples // bsz) * bsz)
            self.order = self.order[:keep]

        tot = int(self.lengths[self.order].sum())
        trained = int(sum(int(self.masks[i].sum()) for i in self.order))
        logger.info(
            "MSASFTDataset: %d rows from %d file(s) | %.1fM tokens (%.1fM trained, %.1f%%) | "
            "len p50=%d max=%d | length_tiers=%s",
            len(self.order), len(files), tot / 1e6, trained / 1e6, 100 * trained / max(tot, 1),
            int(np.percentile(self.lengths[self.order], 50)), int(self.lengths[self.order].max()),
            config.get("length_tiers", 0),
        )

    def _tiered_order(self, config):
        """Group rows so each consecutive block of ``train_batch_size`` shares a length tier.

        A step is ``train_batch_size`` rows, one per rank, and the collective ends when the SLOWEST rank
        finishes — so wall-clock is paid on the longest row in the step while the other ranks idle. Drawing
        8 rows at random from a broad length distribution therefore wastes most of the GPU time. Restricting
        a step to one tier bounds that waste; the tier ORDER is shuffled so tiers still interleave randomly
        across steps.

        Deterministic given (row lengths, seed, tier params): the order is recomputed identically on every
        restart, which the resume path depends on — verl saves the dataloader iterator as a bare batch
        COUNTER with no dataset identity (``checkpoint_handler.py:116-124``), so a different order would
        silently resume onto different rows.
        """
        width = int(config.get("tier_width", 2048))
        bsz = int(config.get("train_batch_size", 8) or 8)
        n_tiers = int(config.get("length_tiers", 16))
        rng = np.random.default_rng(int(config.get("seed", 1234) or 1234))

        tiers = np.minimum(self.lengths // max(width, 1), n_tiers - 1)
        groups, leftover = [], []
        for t in range(n_tiers):
            idx = rng.permutation(np.where(tiers == t)[0])
            nb = len(idx) // bsz
            if nb:
                groups.extend(idx[: nb * bsz].reshape(nb, bsz))
            leftover.append(idx[nb * bsz:])  # rows that cannot fill a group WITHIN their own tier

        # EVERY row must be used. A tier whose size is not a multiple of bsz leaves up to bsz-1 rows over;
        # pooling those across tiers and sorting by length recovers them into batches that are as
        # length-homogeneous as the leftovers allow (they may straddle tiers, which is why they are sorted
        # rather than shuffled). There are at most n_tiers*(bsz-1) such rows, so the efficiency cost is
        # negligible next to never training on part of the data.
        left = np.concatenate(leftover) if leftover else np.array([], dtype=np.int64)
        left = left[np.argsort(self.lengths[left], kind="stable")]
        n_mixed = len(left) // bsz
        mixed = list(left[: n_mixed * bsz].reshape(n_mixed, bsz)) if n_mixed else []
        # Whatever still cannot fill a group goes LAST, because the dataloader's own drop_last will trim a
        # trailing partial batch. Placing it here makes the trimmed rows deterministic rather than arbitrary.
        tail = left[n_mixed * bsz:]

        assert groups or mixed, f"no full groups of {bsz} formed; check train_batch_size against the row count"
        allg = groups + mixed
        allg = [allg[i] for i in rng.permutation(len(allg))]
        order = np.concatenate(allg + ([tail] if len(tail) else []))
        assert len(order) == len(self.lengths), f"tiering lost rows: {len(order)} != {len(self.lengths)}"
        logger.info(
            "MSASFTDataset: length_tiers=%d width=%d bsz=%d -> %d in-tier groups + %d mixed-leftover "
            "groups; %d row(s) in a trailing partial batch (trimmed by the loader's drop_last). "
            "All %d rows are in the order.",
            n_tiers, width, bsz, len(groups), len(mixed), len(tail), len(order),
        )
        return order

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int) -> dict:
        r = int(self.order[idx])
        ids = torch.from_numpy(self.ids[r]).long()  # embedding requires int64
        return {
            "input_ids": ids,
            # No padding exists: the engine pads each micro-batch to its own longest row, and with
            # micro_batch_size_per_gpu=1 that is this row's length.
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(ids.shape[0], dtype=torch.long),  # one document per row
            "loss_mask": torch.from_numpy(self.masks[r]).long(),
        }
