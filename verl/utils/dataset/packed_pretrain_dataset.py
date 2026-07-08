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
"""Fixed-length pre-training dataset for DSA Phase-1 indexer warm-up.

Two input modes, auto-detected from the parquet columns:

1. **Pre-tokenized, one document per row** (preferred; ``input_ids`` column present). Each row is one real
   document, already tokenized with the *training model's* tokenizer and filtered to >= ``max_length``
   tokens (see ``examples/dsa/prepare_real_data.py``). We take each row as a single ``max_length`` window,
   truncating if longer. This is the 6a MVP (docs/dsa_train_indexer_plan.md): one doc per row => full-causal
   base attention is correct and the KL target ``p`` equals the true base attention, with NO cross-document
   packing / varlen wiring.

2. **Raw text, concatenate-and-chunk** (fallback; ``text`` column). Tokenizes, concatenates all documents
   into one stream (separated by ``eos``), and chunks into fixed ``max_length`` windows. A window may straddle
   doc boundaries, but since ``position_ids`` is a flat ``arange`` (no per-doc reset), base and target agree
   (both treat the window as one causal doc) — fine for a mechanics smoke on synthetic data.

Either way every window is exactly ``max_length`` long, so with ``pad_mode=no_padding`` there is **no
padding** — the engine's ``use_remove_padding=False`` path yields a clean ``[bsz, max_length]`` batch, which
is what the DSA ``dense_warmup`` forward consumes.

Emits ``input_ids`` / ``attention_mask`` (all ones) / ``position_ids`` (arange) / ``loss_mask`` (all ones;
unused by the indexer KL but kept for the SFT collator contract). Instantiated by ``create_sft_dataset``
(sft_trainer.py) as ``PackedPretrainDataset(parquet_files=..., tokenizer=..., config=..., processor=...,
max_samples=...)``. Genuine multi-document packing (per-doc ``position_ids`` resets + varlen masking) is a
later item.
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
        self.input_ids_key = config.get("input_ids_key", "input_ids")
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        frames = [pd.read_parquet(path) for path in parquet_files]
        df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

        if self.input_ids_key in df.columns:
            self.windows = self._load_pretokenized(df, max_samples)  # one doc per row
        else:
            self.windows = self._pack_from_text(df, parquet_files, max_samples)  # concatenate-and-chunk

    def _load_pretokenized(self, df: pd.DataFrame, max_samples: int) -> torch.Tensor:
        """One document per row: take each pre-tokenized row as a single ``seq_len`` window."""
        windows: list[list[int]] = []
        for ids in df[self.input_ids_key].tolist():
            ids = [int(t) for t in ids]
            if len(ids) < self.seq_len:
                continue  # prepare_real_data.py filters to >= seq_len; skip any short rows defensively
            windows.append(ids[: self.seq_len])  # one doc, truncated to exactly seq_len
            if max_samples is not None and max_samples > 0 and len(windows) >= max_samples:
                break
        assert len(windows) > 0, (
            f"no rows with >= {self.seq_len} tokens in the '{self.input_ids_key}' column; "
            f"regenerate the parquet at this seq_len or lower data.max_length"
        )
        return torch.tensor(windows, dtype=torch.long)  # [n_windows, seq_len]

    def _pack_from_text(self, df: pd.DataFrame, parquet_files, max_samples: int) -> torch.Tensor:
        """Fallback: concatenate all text into one token stream and chunk into fixed windows."""
        assert self.text_key in df.columns, (
            f"parquet has neither '{self.input_ids_key}' nor '{self.text_key}' column (cols={list(df.columns)})"
        )
        eos = self.tokenizer.eos_token_id
        stream: list[int] = []
        for text in df[self.text_key].tolist():
            stream.extend(self.tokenizer(str(text), add_special_tokens=False).input_ids)
            if eos is not None:
                stream.append(eos)

        n_windows = len(stream) // self.seq_len
        if max_samples is not None and max_samples > 0:
            n_windows = min(n_windows, max_samples)
        assert n_windows > 0, (
            f"not enough tokens ({len(stream)}) to form one window of {self.seq_len}; "
            f"use more data or a smaller data.max_length"
        )
        flat = torch.tensor(stream[: n_windows * self.seq_len], dtype=torch.long)
        return flat.view(n_windows, self.seq_len)  # [n_windows, seq_len]

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> dict:
        ids = self.windows[idx]  # [seq_len]
        ones = torch.ones_like(ids)
        return {
            "input_ids": ids,
            "attention_mask": ones,  # fixed length -> all real, no padding
            "position_ids": torch.arange(ids.shape[0], dtype=torch.long),  # single doc, contiguous
            "loss_mask": ones,  # unused by indexer_kl; kept for the SFT collator contract
        }
