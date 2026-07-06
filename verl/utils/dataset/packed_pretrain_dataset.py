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
"""Fixed-length packed pre-training dataset for DSA Phase-1 indexer warm-up.

Reads parquet files with a raw-text column, tokenizes, concatenates into one token stream (documents
separated by ``eos``), and chunks into **fixed-length** windows of ``max_length`` tokens. Every window is
exactly ``max_length`` long, so with ``pad_mode=no_padding`` there is **no padding** — the engine's
``use_remove_padding=False`` path yields a clean ``[bsz, max_length]`` batch, which is what the DSA
``dense_warmup`` forward consumes.

Emits ``input_ids`` / ``attention_mask`` (all ones) / ``position_ids`` (arange) / ``loss_mask`` (all ones;
unused by the indexer KL but kept for the SFT collator contract).

NOTE (smoke scope): each window is one contiguous chunk with a single ``arange`` of positions — i.e. it is
treated as **one document** (no per-document ``position_ids`` reset, no cross-document masking). That is
fine for exercising the training loop; genuine multi-document 32K packing (per-doc resets + varlen
attention masking) is a later item. Instantiated by ``create_sft_dataset`` (sft_trainer.py) as
``PackedPretrainDataset(parquet_files=..., tokenizer=..., config=..., processor=..., max_samples=...)``.
"""

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class PackedPretrainDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor: Optional[object] = None, max_samples: int = -1):
        self.tokenizer = tokenizer
        self.seq_len = int(config.get("max_length", 4096))
        self.text_key = config.get("text_key", "text")
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        # concatenate all documents into one token stream, separated by eos
        eos = tokenizer.eos_token_id
        stream: list[int] = []
        for path in parquet_files:
            df = pd.read_parquet(path)
            assert self.text_key in df.columns, f"column '{self.text_key}' not in {path} (cols={list(df.columns)})"
            for text in df[self.text_key].tolist():
                ids = tokenizer(str(text), add_special_tokens=False).input_ids
                stream.extend(ids)
                if eos is not None:
                    stream.append(eos)

        # chunk into fixed-length windows; drop the remainder so every window is exactly seq_len
        n_windows = len(stream) // self.seq_len
        if max_samples is not None and max_samples > 0:
            n_windows = min(n_windows, max_samples)
        assert n_windows > 0, (
            f"not enough tokens ({len(stream)}) to form one window of {self.seq_len}; "
            f"use more data or a smaller data.max_length"
        )
        flat = torch.tensor(stream[: n_windows * self.seq_len], dtype=torch.long)
        self.windows = flat.view(n_windows, self.seq_len)  # [n_windows, seq_len]

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> dict:
        ids = self.windows[idx]  # [seq_len]
        ones = torch.ones_like(ids)
        return {
            "input_ids": ids,
            "attention_mask": ones,  # fixed length -> all real, no padding
            "position_ids": torch.arange(ids.shape[0], dtype=torch.long),  # single contiguous chunk
            "loss_mask": ones,  # unused by indexer_kl; kept for the SFT collator contract
        }
