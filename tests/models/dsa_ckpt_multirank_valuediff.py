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
"""#6 value-diff: after the real FSDP2 broadcast-load, every rank's indexer params must be bit-identical to
rank 0's (the freshly-created indexer params must propagate). This closes the value-level gap left by #10's
key-level 4-GPU check.

Method: build tiny MiniCPM3 + DSA on each rank; set rank-0's indexer to a KNOWN deterministic pattern and
CORRUPT every other rank's indexer with garbage; run the exact engine path
(`apply_fsdp2` + `fsdp2_load_full_state_dict`, which broadcasts `module.state_dict()` from rank 0); then
gather each indexer param's full tensor on EVERY rank and assert it equals the known pattern. If the
broadcast failed to carry the injected indexer params, non-rank-0 would still hold garbage -> mismatch.

Run (needs 4 GPUs + transformers 4.57.1):
    PYTHONPATH=.devlibs/tf457lib torchrun --standalone --nproc_per_node=4 \
        tests/models/dsa_ckpt_multirank_valuediff.py
"""

import os
import zlib

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

MODEL = "openbmb/MiniCPM3-4B"


def _expected(name: str, shape, dtype, device):
    """Deterministic constant fill, distinct per param, reproducible across processes (crc32, not hash())."""
    v = (zlib.crc32(name.encode()) % 1000) / 1000.0
    return torch.full(tuple(shape), v, dtype=dtype, device=device)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    dev = torch.device(f"cuda:{rank}")

    from transformers import AutoConfig, AutoModelForCausalLM

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except ImportError:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    cfg.num_hidden_layers = 2
    cfg.vocab_size = 1000
    cfg._attn_implementation = "flash_attention_2"
    cfg.dsa_enabled = True
    cfg.dsa_overrides = {"n_heads": 4, "head_dim": 64, "mode": "dense_warmup", "fp8": False}
    model = AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    ).to(dev).to(torch.bfloat16)
    apply_monkey_patch(model=model, use_remove_padding=False, ulysses_sp_size=1)  # attaches the indexers

    # rank 0: known pattern; other ranks: garbage. If the broadcast carries the indexer, garbage is overwritten.
    n_idx = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".indexer." in n:
                n_idx += 1
                p.copy_(_expected(n, p.shape, p.dtype, p.device) if rank == 0 else torch.full_like(p, -999.0))

    # sanity: non-rank-0 really is corrupted pre-load (so a PASS can't be a coincidence)
    if rank != 0:
        n0 = next(n for n, _ in model.named_parameters() if ".indexer." in n)
        p0 = dict(model.named_parameters())[n0]
        assert torch.equal(p0, torch.full_like(p0, -999.0)), "pre-load corruption not applied"

    full_state = model.state_dict()  # captured with indexers present; rank 0's is authoritative under broadcast

    # exact engine wrap + broadcast-load path
    mesh = init_device_mesh("cuda", (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
    fsdp_kwargs = {"mesh": mesh, "mp_policy": mp, "offload_policy": None, "reshard_after_forward": False}
    apply_fsdp2(model, fsdp_kwargs, {"wrap_policy": {}, "forward_prefetch": False})
    fsdp2_load_full_state_dict(model, full_state, mesh, None)

    # gather each indexer param's FULL tensor on THIS rank and compare to the known pattern
    mismatches = []
    checked = 0
    for n, v in model.state_dict().items():
        if ".indexer." not in n:
            continue
        full = v.full_tensor() if hasattr(v, "full_tensor") else v
        exp = _expected(n, full.shape, full.dtype, full.device)
        checked += 1
        if not torch.equal(full, exp):
            mismatches.append(n)

    local_bad = torch.tensor([len(mismatches)], device=dev)
    total_checked = torch.tensor([checked], device=dev)
    dist.all_reduce(local_bad)
    dist.all_reduce(total_checked)
    print(f"[rank{rank}] indexer params attached={n_idx} checked={checked} local_mismatches={len(mismatches)}"
          + ("" if not mismatches else f" -> {mismatches[:3]}"), flush=True)
    if rank == 0:
        status = "PASS" if local_bad.item() == 0 else "FAIL"
        print(f"[rank0] {status}: {total_checked.item()} indexer full-tensor checks across {world} ranks, "
              f"{local_bad.item()} mismatches", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
