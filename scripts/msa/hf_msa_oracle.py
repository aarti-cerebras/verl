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
"""The TRAINING-side oracle for P4 parity (docs/qwen3_4b_msa/serving_plan.md §6.3, §6.5).

Runs the *training* MSA forward — `verl/models/transformers/qwen3_msa.py`, the exact code that
produced the checkpoint — over given token ids and dumps the logits. The vLLM side dumps its
own for the same tokens; a third script compares them. If those agree, the served function
equals the trained function, which is the whole point of P4.

**Why a separate process.** This must run in the TRAINING environment (system python:
transformers 5.3.0, `ray` present), not the serving venv (transformers 5.14.1, no `ray`). The
training forward is a line-for-line mirror of `Qwen3Attention.forward` for a specific
transformers version, so running the oracle on a different one would compare against something
that is not what we trained. The two sides therefore exchange tensors on disk.

**Weights: use the `--no-norm-shift` export.** It holds the exact trained values in Qwen3's own
convention. The shifted export is for the fused kernel only; feeding `w - 1` to a standard
RMSNorm would compute `x*(w-1)` (§2.3). We rename `self_attn.index_*` back to
`self_attn.indexer.index_*`, undoing P2's strip, because the training module nests them.

**Prefill only, and that is sufficient.** The training module has no decode path — it recomputes
index keys from the current `hidden_states`, so with a KV cache at decode `k_idx` would have
length 1 (§1). But teacher forcing is a single full-sequence prefill, so this oracle covers both
the first-token check (§6.3) and per-position decode parity (§6.5). Only free-running HF
generation would need a decode path, and we never need that.

Run (training env, NOT the venv):
  /usr/bin/python3 scripts/msa/hf_msa_oracle.py \
      --model /cb/ml-eng/aarti/msa/serving/k8_step1400_noshift \
      --tokens /tmp/p4/tokens.json --out /tmp/p4/oracle.pt
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

INDEX_RE = re.compile(r"^(.*\.self_attn)\.(index_(?:q|k)_(?:proj|norm)\.weight)$")


def build_model(model_dir: str, device: str, dtype: torch.dtype, sparse: bool, attn_impl: str = "eager"):
    """Stock Qwen3 + the MSA monkey-patch + indexers, weights from `model_dir`.

    `attn_impl` selects the kernel for the DENSE-PREFIX layers only -- sparse layers return early
    from their own path and never reach `attention_interface`. Default "eager" keeps the P4 oracle
    maximally literal; pass "sdpa" for long inputs, where eager materialises a [1, H_q, T, T] score
    matrix (63 GiB at 32K -- an immediate OOM).
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen3 import modeling_qwen3

    from verl.models.transformers.qwen3_msa import (
        attach_indexers,
        build_msa_config,
        install_kl_accumulation,
        msa_overrides_from_config,
        qwen3_msa_attn_forward,
    )

    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.architectures = ["Qwen3ForCausalLM"]
    assert getattr(cfg, "msa_enabled", False), "config has no msa_enabled"
    assert not getattr(cfg, "msa_norm_shift_applied", False), (
        "this export has the w-1 shift applied; the training forward uses standard RMSNorm and "
        "would compute x*(w-1). Pass the --no-norm-shift export."
    )
    cfg._attn_implementation = attn_impl

    # Patch BEFORE constructing, so every layer gets the MSA forward.
    modeling_qwen3.Qwen3Attention.forward = qwen3_msa_attn_forward

    # Construct for REAL on the target device -- NOT meta + to_empty(). `to_empty()` leaves every
    # NON-PERSISTENT buffer uninitialised, and the rotary embedding's `inv_freq` is exactly that:
    # it is absent from the state dict, so load_state_dict cannot restore it and the model runs on
    # garbage RoPE frequencies. That is nearly invisible at position 0 and catastrophic at 1600 --
    # it produced a mean |Δlogprob| of 4.4 against vLLM before being found.
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = modeling_qwen3.Qwen3ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev)

    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    # Undo P2's `.indexer.` strip -- the training module nests the index branch one level deeper.
    renamed, n_idx = {}, 0
    for k, v in sd.items():
        m = INDEX_RE.match(k)
        if m:
            k = f"{m.group(1)}.indexer.{m.group(2)}"
            n_idx += 1
        renamed[k] = v.to(dtype)
    if cfg.tie_word_embeddings and "lm_head.weight" not in renamed:
        renamed["lm_head.weight"] = renamed["model.embed_tokens.weight"]

    # MUST pass the flat msa_* overrides through -- build_msa_config does NOT read them itself,
    # it only forces geometry. Calling it with just `mode` silently yields MSAConfig DEFAULTS
    # (top_k=16), which on a k8 checkpoint would compare the served model against a differently
    # configured oracle and look like a parity failure. Same call as the trainer's
    # monkey_patch.py:552.
    ov = msa_overrides_from_config(cfg)
    ov["mode"] = "sparse" if sparse else "dense_warmup"
    ov.pop("warmstart_path", None)  # weights come from --model, not the training warm start
    msa_cfg = build_msa_config(cfg, **ov)

    # Guard: the oracle's selection geometry must equal what the checkpoint was trained with.
    for field, key in (("top_k", "msa_top_k"), ("block_size", "msa_block_size"),
                       ("dense_prefix", "msa_dense_prefix"), ("local_blocks", "msa_local_blocks"),
                       ("init_blocks", "msa_init_blocks")):
        want = getattr(cfg, key, None)
        got = getattr(msa_cfg, field)
        assert want is None or int(want) == int(got), (
            f"oracle {field}={got} but config.{key}={want} -- the oracle would not match the "
            f"served model")
    attach_indexers(model, msa_cfg)
    install_kl_accumulation(model)

    missing, unexpected = model.load_state_dict(renamed, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not missing, f"missing keys: {missing[:5]}"
    print(f"[oracle] loaded {len(renamed)} tensors ({n_idx} index_*), mode={msa_cfg.mode}, "
          f"top_k={msa_cfg.top_k}, block={msa_cfg.block_size}, dense_prefix={msa_cfg.dense_prefix}")
    return model.to(dtype).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the --no-norm-shift serving dir (exact weights)")
    ap.add_argument("--tokens", required=True, help="JSON: {name: [token ids], ...}")
    ap.add_argument("--out", required=True, help="output .pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dense", action="store_true", help="run the DENSE (dense_warmup) forward instead")
    ap.add_argument("--topk", type=int, default=20, help="how many top logits to keep per position")
    args = ap.parse_args()

    with open(args.tokens) as fh:
        prompts = json.load(fh)

    model = build_model(args.model, args.device, torch.bfloat16, sparse=not args.dense)

    out = {}
    for name, ids in prompts.items():
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        am = torch.ones_like(x)
        pos = torch.arange(x.shape[1], device=args.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=x, attention_mask=am, position_ids=pos).logits[0].float()
        # last position -> what vLLM reports as the first generated step
        lp_last = torch.log_softmax(logits[-1], dim=-1)
        top = torch.topk(lp_last, args.topk)
        out[name] = {
            "n_tokens": len(ids),
            "last_top_ids": top.indices.cpu(),
            "last_top_logprobs": top.values.cpu(),
            # full per-position logprobs of the ACTUAL next token -- for teacher-forced parity (§6.5)
            "tf_logprobs": torch.log_softmax(logits[:-1], dim=-1)
            .gather(1, torch.tensor(ids[1:], device=args.device).unsqueeze(1))
            .squeeze(1)
            .cpu(),
        }
        print(f"[oracle] {name}: {len(ids)} tokens, top1={top.indices[0].item()} "
              f"({top.values[0].item():.4f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"mode": "dense" if args.dense else "sparse", "model": args.model, "data": out}, args.out)
    print(f"[oracle] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
