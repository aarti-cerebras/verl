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
"""Invert the `w - 1` Gemma shift on an existing serving dir -> a Route-B (`--no-norm-shift`) dir.

`build_msa_serving_dir.py --no-norm-shift` is the normal way to get this, but it needs the FSDP
`global_step_N` shards, which a retention policy may have pruned. The shifted export is a lossless
enough record to recover from: the shift is exactly `w -= 1` on MAIN `q_norm`/`k_norm`, sparse
layers only (build_msa_serving_dir.py:157-184), so `w += 1` inverts it.

**Accuracy.** The round trip is lossy by <= 1 bf16 ULP, and the ORIGINAL export already measured
and gated that error (`--max-shift-err`, default 2^-8; worst observed 3.799e-3). Re-adding 1 here
lands on the same value the export's own `_verify` step compared against, so this reproduces the
`--no-norm-shift` output to within that already-accepted floor -- not bit-exactly. Fine for the
indexer probes; do NOT use it as a parity oracle where the floor itself is the quantity of
interest.

Scope guard: index norms are Gemma ALREADY and dense-prefix layers never touch the kernel, so both
must be left alone -- shifting either would corrupt the model. We assert the count matches the
number of sparse layers.

Run:
  /usr/bin/python3 scripts/msa/unshift_serving_dir.py \
      --src /cb/ml-eng/aarti/msa/serving/k16v2_step5800 \
      --out /cb/ml-eng/aarti/msa/serving/k16v2_step5800_noshift
"""

import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

QK_NORM_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.(q_norm|k_norm)\.weight$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a serving dir built WITH the w-1 shift")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.src, "config.json")))
    assert cfg.get("msa_norm_shift_applied"), (
        f"{args.src} is not shifted (msa_norm_shift_applied is falsy) -- nothing to invert, and "
        f"un-shifting it would corrupt the weights")
    dense_prefix = int(cfg["msa_dense_prefix"])
    n_layers = int(cfg["num_hidden_layers"])
    sparse_ids = list(range(dense_prefix, n_layers))

    sd = load_file(os.path.join(args.src, "model.safetensors"))
    n = 0
    for k in list(sd):
        m = QK_NORM_RE.match(k)
        if m and int(m.group(1)) in sparse_ids:
            sd[k] = (sd[k].float() + 1.0).to(sd[k].dtype)
            n += 1
    exp = 2 * len(sparse_ids)
    assert n == exp, f"un-shifted {n} tensors, expected {exp} (2 x {len(sparse_ids)} sparse layers)"

    os.makedirs(args.out, exist_ok=True)
    for f in os.listdir(args.src):
        if f not in ("model.safetensors", "config.json") and os.path.isfile(os.path.join(args.src, f)):
            shutil.copy2(os.path.join(args.src, f), os.path.join(args.out, f))
    save_file(sd, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    cfg["msa_norm_shift_applied"] = False
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    # Sanity: a shifted Qwen3 q_norm sits near 0 (w-1 of a ~1 gain); un-shifted it must sit near 1.
    w = sd[f"model.layers.{sparse_ids[0]}.self_attn.q_norm.weight"].float()
    print(f"[unshift] {n} tensors, layers {sparse_ids[0]}..{sparse_ids[-1]}; "
          f"q_norm[{sparse_ids[0]}] mean now {w.mean():.4f} (expect ~1) -> {args.out}")
    assert 0.5 < w.mean() < 2.0, f"un-shifted q_norm mean {w.mean():.4f} is not near 1"
    return 0


if __name__ == "__main__":
    sys.exit(main())
