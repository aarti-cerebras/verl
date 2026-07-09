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
"""CPU unit test for indexer_kl_loss global-batch normalization (mean over batch/seq/layers)."""

import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.utils.losses import indexer_kl_loss


class _FakeModel:
    """Stands in for the FSDP-wrapped module: exposes the hook-set KL + metrics attributes."""

    def __init__(self, kl_value: float):
        self._dsa_indexer_kl = torch.tensor(kl_value, requires_grad=True)
        self._dsa_metrics = {"indexer/kl_layer_mean": kl_value}


def _micro_batch(mb_tokens: int, global_tokens: int, dp_size: int) -> TensorDict:
    # normalization is by valid (non-pad) query rows: attention_mask all-ones => mb_valid == mb_tokens
    data = TensorDict(
        {
            "input_ids": torch.zeros(1, mb_tokens, dtype=torch.long),
            "attention_mask": torch.ones(1, mb_tokens),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(data, batch_num_valid_queries=global_tokens)
    tu.assign_non_tensor(data, dp_size=dp_size)
    return data


def test_accumulated_loss_is_global_mean_not_sum():
    # 8 uniform micro-batches (4096 tokens each) of a 32768-token global batch, single dp rank.
    kl_value, mb, n_micro, dp = 3.69, 4096, 8, 1
    global_tokens = mb * n_micro

    total = 0.0
    for _ in range(n_micro):
        loss, metrics = indexer_kl_loss(
            config=None, model_output={}, data=_micro_batch(mb, global_tokens, dp), model=_FakeModel(kl_value)
        )
        # each micro-batch contributes its token share, so backward accumulates to the mean (not n_micro * kl)
        assert abs(loss.item() - kl_value * mb / global_tokens) < 1e-5
        total += loss.item()

    assert abs(total - kl_value) < 1e-5  # sum over micro-batches == the global mean
    assert abs(metrics["indexer/kl"].item() - kl_value) < 1e-5  # metric logs the interpretable mean


def test_uneven_micro_batches_are_token_weighted():
    # unequal micro-batch sizes must still accumulate to the exact token-weighted global mean.
    kl_value, sizes, dp = 3.69, [4096, 2048, 1024], 1
    global_tokens = sum(sizes)
    total = sum(
        indexer_kl_loss(config=None, model_output={}, data=_micro_batch(s, global_tokens, dp), model=_FakeModel(kl_value))[0].item()
        for s in sizes
    )
    assert abs(total - kl_value) < 1e-5


def test_dp_size_cancels_ddp_gradient_averaging():
    # *dp_size multiplies the loss so that FSDP/DDP's later /dp_size mean recovers the true global mean.
    kl_value, mb, dp = 3.69, 4096, 4
    # with dp_size ranks, batch_num_tokens is the all-reduced (summed) count across ranks
    global_tokens = mb * dp
    loss, _ = indexer_kl_loss(
        config=None, model_output={}, data=_micro_batch(mb, global_tokens, dp), model=_FakeModel(kl_value)
    )
    # one micro-batch per rank: loss = kl * mb/global * dp; summing across dp ranks then DDP-averaging (/dp) -> kl
    assert abs(loss.item() - kl_value * mb / global_tokens * dp) < 1e-5
    assert abs(loss.item() * dp / dp - kl_value) < 1e-5  # DDP mean over the dp ranks recovers kl


def test_falls_back_to_raw_kl_without_batch_metadata():
    loss, metrics = indexer_kl_loss(config=None, model_output={}, data=None, model=_FakeModel(3.69))
    assert abs(loss.item() - 3.69) < 1e-5
    assert abs(metrics["indexer/kl"].item() - 3.69) < 1e-5
