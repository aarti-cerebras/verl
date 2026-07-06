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
