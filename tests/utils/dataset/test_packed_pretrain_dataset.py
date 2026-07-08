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
"""CPU unit test for PackedPretrainDataset (contract + fixed-length chunking)."""

import pandas as pd
import pytest
import torch

from verl.utils.dataset.packed_pretrain_dataset import PackedPretrainDataset


class _StubTokenizer:
    """Whitespace tokenizer: word -> small int id; eos = 0."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [(abs(hash(w)) % 997) + 1 for w in str(text).split()]
        return type("Enc", (), {"input_ids": ids})()


def test_packed_dataset_contract(tmp_path):
    # 3 docs of 10 words each -> 30 words + 3 eos = 33 tokens; seq_len=8 -> 4 full windows (drop 1)
    df = pd.DataFrame({"text": [" ".join(["w%d" % i for i in range(10)]) for _ in range(3)]})
    p = tmp_path / "d.parquet"
    df.to_parquet(p)

    ds = PackedPretrainDataset(parquet_files=str(p), tokenizer=_StubTokenizer(), config={"max_length": 8})
    assert len(ds) == 33 // 8  # == 4, remainder dropped

    item = ds[0]
    assert set(item) == {"input_ids", "attention_mask", "position_ids", "loss_mask"}
    for k in item:
        assert item[k].shape == (8,) and item[k].dtype == torch.long
    assert torch.equal(item["attention_mask"], torch.ones(8, dtype=torch.long))
    assert torch.equal(item["loss_mask"], torch.ones(8, dtype=torch.long))
    assert torch.equal(item["position_ids"], torch.arange(8))
    # contiguity: window 0 then window 1 are consecutive slices of the stream
    assert torch.equal(ds[0]["input_ids"], ds.windows[0]) and torch.equal(ds[1]["input_ids"], ds.windows[1])


def test_max_samples_caps_windows(tmp_path):
    df = pd.DataFrame({"text": [" ".join(["w%d" % i for i in range(50)])]})
    p = tmp_path / "d.parquet"
    df.to_parquet(p)
    ds = PackedPretrainDataset(parquet_files=str(p), tokenizer=_StubTokenizer(), config={"max_length": 8}, max_samples=2)
    assert len(ds) == 2


# ---- pre-tokenized, one-document-per-row mode (input_ids column; the 6a real-data path) ----


def test_pretokenized_one_doc_per_row(tmp_path):
    # 3 long docs (>= seq_len=8) + 1 short doc (< seq_len, must be skipped). Rows have varying lengths.
    df = pd.DataFrame({"input_ids": [list(range(1, 13)), list(range(1, 9)), list(range(1, 21)), [1, 2, 3]]})
    p = tmp_path / "ids.parquet"
    df.to_parquet(p)

    ds = PackedPretrainDataset(parquet_files=str(p), tokenizer=None, config={"max_length": 8})
    assert len(ds) == 3  # the 3-token short doc is skipped

    item = ds[0]
    assert set(item) == {"input_ids", "attention_mask", "position_ids", "loss_mask"}
    for k in item:
        assert item[k].shape == (8,) and item[k].dtype == torch.long
    # each row is ONE doc truncated to seq_len (not concatenated across docs)
    assert torch.equal(ds[0]["input_ids"], torch.arange(1, 9))  # first 8 of the 12-token doc
    assert torch.equal(ds[2]["input_ids"], torch.arange(1, 9))  # first 8 of the 20-token doc
    assert torch.equal(item["position_ids"], torch.arange(8))  # single doc, contiguous
    assert torch.equal(item["attention_mask"], torch.ones(8, dtype=torch.long))
    assert torch.equal(item["loss_mask"], torch.ones(8, dtype=torch.long))


def test_pretokenized_takes_precedence_and_caps(tmp_path):
    # input_ids column wins even if a text column is also present; max_samples caps rows.
    df = pd.DataFrame({"input_ids": [list(range(20))] * 5, "text": ["ignored"] * 5})
    p = tmp_path / "both.parquet"
    df.to_parquet(p)
    ds = PackedPretrainDataset(parquet_files=str(p), tokenizer=None, config={"max_length": 8}, max_samples=3)
    assert len(ds) == 3


def test_pretokenized_all_short_raises(tmp_path):
    df = pd.DataFrame({"input_ids": [[1, 2, 3], [4, 5]]})  # none reach seq_len=8
    p = tmp_path / "short.parquet"
    df.to_parquet(p)
    with pytest.raises(AssertionError, match="no rows with >="):
        PackedPretrainDataset(parquet_files=str(p), tokenizer=None, config={"max_length": 8})
