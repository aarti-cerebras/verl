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
"""Tests for MSASFTDataset -- the Phase-2 (sparse) loader for pre-tokenized behaviour-cloning rows.

Each test pins a property whose violation is SILENT in training:

* ``loss_mask`` honoured -> otherwise the prompt is trained on as if the model had written it;
* short rows kept -> ``PackedPretrainDataset`` drops rows shorter than ``max_length``, which for
  variable-length BC data is the entire dataset;
* every row used, exactly once -> a tiering bug that quietly discards a slice of the data;
* **deterministic row order** -> the resume invariant. verl saves the dataloader iterator as a bare batch
  COUNTER with no dataset identity (``checkpoint_handler.py:116-124``), so if the order is not reproduced
  exactly, a restart resumes onto different rows and nothing reports it;
* length-tiered batches -> a step's cost is its longest row across ranks, so mixed-length batches idle most
  of the GPUs (measured 32.1% efficient vs ~100% tiered on the real data).

Synthetic fixtures keep this runnable in CI; ``test_real_bc_parquet_contract`` additionally checks the real
artifact when present and skips otherwise.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from verl.utils.dataset.msa_sft_dataset import MSASFTDataset

REAL_BC = (
    "/cb/ml-eng/aarti/msa/data/"
    "qwen3-4b-thinking-2507__dolci-think-rl-32b__ph2b_full93889_L32768_20260730_231930/bc_2b.parquet"
)
IM_END, THINK_OPEN, THINK_CLOSE = 151645, 151667, 151668


def _write(path, rows, masks=None):
    """rows: list[list[int]]; masks: list[list[int]] or None (column omitted)."""
    cols = {"input_ids": pa.array(rows, type=pa.list_(pa.int32()))}
    fields = [pa.field("input_ids", pa.list_(pa.int32()))]
    if masks is not None:
        cols["loss_mask"] = pa.array(masks, type=pa.list_(pa.int8()))
        fields.append(pa.field("loss_mask", pa.list_(pa.int8())))
    pq.write_table(pa.table(cols, schema=pa.schema(fields)), path)
    return str(path)


@pytest.fixture
def varlen_parquet(tmp_path):
    """40 rows of deliberately varied length, each with a masked 'prompt' prefix."""
    rng = np.random.default_rng(0)
    lens = [int(x) for x in rng.integers(50, 5000, size=40)]
    rows = [list(range(1, n + 1)) for n in lens]
    masks = [[0] * (n // 4) + [1] * (n - n // 4) for n in lens]
    return _write(tmp_path / "bc.parquet", rows, masks), lens, masks


def test_keeps_short_rows_and_honours_loss_mask(varlen_parquet):
    """The two behaviours PackedPretrainDataset gets wrong for this data."""
    path, lens, masks = varlen_parquet
    ds = MSASFTDataset(parquet_files=path, tokenizer=None, config={"max_length": 32768})

    # Every row survives even though all are far shorter than max_length.
    assert len(ds) == len(lens), "short rows were dropped"

    s = ds[0]
    assert set(s) == {"input_ids", "attention_mask", "position_ids", "loss_mask"}
    assert s["input_ids"].dtype == torch.int64, "embedding lookup requires int64"
    # loss_mask comes from the column, NOT forced to ones.
    assert s["loss_mask"].tolist() == masks[0]
    assert s["loss_mask"].sum() < s["loss_mask"].numel(), "prompt region is not masked"
    # No padding is materialised, so attention_mask is legitimately all ones, and each row is one document.
    assert s["attention_mask"].tolist() == [1] * lens[0]
    assert s["position_ids"].tolist() == list(range(lens[0]))


def test_missing_loss_mask_column_defaults_to_all_ones(tmp_path):
    """Real pre-training text has no loss_mask: every token is a target. Must be explicit, not a crash."""
    path = _write(tmp_path / "plain.parquet", [list(range(1, 101)), list(range(1, 51))], masks=None)
    ds = MSASFTDataset(parquet_files=path, tokenizer=None, config={"max_length": 4096})
    assert len(ds) == 2
    assert ds[0]["loss_mask"].tolist() == [1] * 100


def test_rows_longer_than_max_length_are_truncated(tmp_path):
    path = _write(tmp_path / "long.parquet", [list(range(1, 5001))], [[1] * 5000])
    ds = MSASFTDataset(parquet_files=path, tokenizer=None, config={"max_length": 1024})
    s = ds[0]
    assert len(s["input_ids"]) == 1024
    assert len(s["loss_mask"]) == 1024, "mask must be truncated with the ids, never left misaligned"


def test_messages_only_parquet_fails_loudly(tmp_path):
    """A `messages` parquet must raise here rather than silently produce nothing -- that data belongs to
    MultiTurnSFTDataset (and for a Thinking model, that path deletes the <think> trace)."""
    p = tmp_path / "messages.parquet"
    pq.write_table(pa.table({"messages": pa.array([["a", "b"]], type=pa.list_(pa.string()))}), p)
    with pytest.raises(AssertionError, match="PRE-TOKENIZED"):
        MSASFTDataset(parquet_files=str(p), tokenizer=None, config={"max_length": 128})


class TestLengthTiering:
    CFG = {"max_length": 32768, "length_tiers": 8, "tier_width": 512, "train_batch_size": 4, "seed": 7}

    def test_every_row_used_exactly_once(self, varlen_parquet):
        """Per-tier leftovers must be POOLED into mixed batches, not discarded. Regression test: an earlier
        version dropped `len(tier) % batch_size` rows per tier."""
        path, lens, _ = varlen_parquet
        ds = MSASFTDataset(parquet_files=path, tokenizer=None, config=dict(self.CFG))
        assert len(ds) == len(lens), "tiering changed the row count"
        assert np.array_equal(np.sort(ds.order), np.arange(len(lens))), "a row is missing or duplicated"

    def test_batches_are_length_homogeneous(self, varlen_parquet):
        path, _, _ = varlen_parquet
        cfg = dict(self.CFG)
        ds = MSASFTDataset(parquet_files=path, tokenizer=None, config=cfg)
        bsz, width = cfg["train_batch_size"], cfg["tier_width"]
        L = ds.lengths[ds.order]
        nb = len(L) // bsz
        spread = np.ptp(L[: nb * bsz].reshape(nb, bsz), axis=1)  # ndarray.ptp() was removed in numpy 2.0
        # Leftover-pool batches may straddle tiers; in-tier batches must not.
        assert (spread < width).mean() >= 0.5, f"too many batches exceed one tier: {spread}"
        # Untiered order should be markedly worse, i.e. the tiering is doing something.
        plain = MSASFTDataset(parquet_files=path, tokenizer=None, config={"max_length": 32768})
        pl = np.ptp(plain.lengths[plain.order][: nb * bsz].reshape(nb, bsz), axis=1)
        assert spread.mean() < pl.mean(), "tiering did not reduce intra-batch length spread"

    def test_order_is_deterministic(self, varlen_parquet):
        """THE resume invariant: same inputs -> byte-identical order, or a restart silently changes data."""
        path, _, _ = varlen_parquet
        a = MSASFTDataset(parquet_files=path, tokenizer=None, config=dict(self.CFG))
        b = MSASFTDataset(parquet_files=path, tokenizer=None, config=dict(self.CFG))
        assert np.array_equal(a.order, b.order)
        assert [a[i]["input_ids"].tolist() for i in range(len(a))] == \
               [b[i]["input_ids"].tolist() for i in range(len(b))]

    def test_seed_changes_order(self, varlen_parquet):
        path, _, _ = varlen_parquet
        a = MSASFTDataset(parquet_files=path, tokenizer=None, config=dict(self.CFG))
        b = MSASFTDataset(parquet_files=path, tokenizer=None, config={**self.CFG, "seed": 99})
        assert not np.array_equal(a.order, b.order), "seed had no effect -- order would not be reproducible"

    def test_max_samples_floors_to_whole_batches(self, varlen_parquet):
        """Truncating mid-group would hand the sampler a partial length-tiered batch."""
        path, _, _ = varlen_parquet
        ds = MSASFTDataset(parquet_files=path, tokenizer=None, config=dict(self.CFG), max_samples=10)
        assert len(ds) % self.CFG["train_batch_size"] == 0 and len(ds) == 8


@pytest.mark.skipif(not __import__("os").path.exists(REAL_BC), reason="real BC parquet not present")
def test_real_bc_parquet_contract():
    """The phase2_data_gen.md §6 splice contract, on the real artifact: prompt masked, exactly one
    </think> in the trained region, no re-opened <think>, terminal <|im_end|>."""
    ds = MSASFTDataset(parquet_files=REAL_BC, tokenizer=None, config={"max_length": 32768})
    assert len(ds) == 90230, f"expected 90,230 rows, got {len(ds)} -- artifact changed"
    assert (ds.lengths < 32768).all(), "a row reaches max_length; PackedPretrainDataset-style filtering would keep only those"

    for i in (0, 1, 100, 5000, 90229):
        s = ds[i]
        ids, lm = s["input_ids"].tolist(), s["loss_mask"].tolist()
        n0 = lm.count(0)
        assert lm[:n0] == [0] * n0 and lm[n0:] == [1] * (len(lm) - n0), f"row {i}: mask is not prompt-then-completion"
        assert 0 < n0 < len(lm), f"row {i}: degenerate mask"
        assert ids[-1] == IM_END, f"row {i}: does not end with <|im_end|>"
        # The served prefix ends INSIDE the think block, so the trace carries only the closing marker.
        assert ids[n0 - 1] == 198 and ids[n0 - 2] == THINK_OPEN, f"row {i}: prefix does not end '<think>\\n'"
        trained = ids[n0:]
        assert trained.count(THINK_CLOSE) == 1, f"row {i}: expected exactly one </think>"
        assert THINK_OPEN not in trained, f"row {i}: <think> re-opened mid-answer"
