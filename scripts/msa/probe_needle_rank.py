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
"""Needle-block RANK probe -- `eval_plan` §6 items 4 and 5, never previously built for MSA.

**The question.** RULER 32K collapses on multikey_2/3 and qa_squad while `learned_coverage` reads
0.97. Coverage is attention MASS captured; it is dominated by sinks and the local window and can
sit at 0.97 while the one block holding the answer is unselected. So we ask the direct question
instead: for an item whose answer we KNOW the location of, where does that block RANK in the
indexer's ordering?

The remedy hinges on the answer, which is why rank and not just recall:

  * rank 17-40  -> NEAR MISS. The indexer scores the needle nearly right and a modest ranking
                   improvement (more Phase-1, `full_support_kl_prob`) recovers it.
  * rank >> k   -> BLIND. The needle is nowhere near the top; restricted-support Phase-2 KL can
                   never fix it (it gives no gradient to unselected blocks) and Phase-1 data that
                   rewards discrimination is required.

Binary recall cannot distinguish these, and vLLM cannot answer it at all -- its fused kernel emits
only the top-k INDICES. Hence the training-side forward, where `ix.block_scores()` is an explicit
`[b, H_kv, T_q, n_blocks]` tensor we can intercept before `select_blocks` discards all but `k`.

**Weights must be the `--no-norm-shift` export** (see `unshift_serving_dir.py`): this runs the
training module's standard RMSNorm, and the shifted export would compute `x*(w-1)`.

**Where we measure.** The last prompt token -- the position that generates the answer, and the one
whose selection decides whether the answer is reachable at all. RULER's prompts end mid-sentence
("... the special magic uuid for X is"), so this is exactly the retrieval step.

**Ground truth** is recovered from the prompt rather than dataset internals: OpenCompass records
`gold`, the gold string occurs verbatim in the haystack, and the needle is the line containing it.
Items where it does not occur (or occurs more than once) are SKIPPED and counted, never guessed.

Run (training env; ~1-2 min/item at 32K):
  CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/msa/probe_needle_rank.py \
      --model /cb/ml-eng/aarti/msa/serving/k16v2_step5800_noshift \
      --preds '<oc_workdir>/predictions/*/ruler_niah_multikey_3_32k_*.json' \
      --out /tmp/rank_mk3.json --shard 0/4
"""

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Filled by the patched select_blocks: [(layer_idx, block_scores[H,N], sel[H,k]), ...]
_CAP: dict = {"row": None, "out": []}


def _install_capture():
    """Intercept `select_blocks` -- it receives the full block scores AND the absolute query rows.

    Patching here rather than `block_scores` gets both halves in one place: the scores we rank over
    and the selection actually made, guaranteed consistent with the forward that produced them.
    """
    from verl.models.transformers.msa_indexer import MSAIndexer

    orig = MSAIndexer.select_blocks

    def patched(self, block_scores, query_pos):
        sel = orig(self, block_scores, query_pos)
        row = _CAP["row"]
        if row is not None:
            hit = (query_pos == row).nonzero()
            if hit.numel():  # this tile contains the target position
                r = int(hit[0, 0])
                _CAP["out"].append(
                    (self._probe_layer, block_scores[0, :, r, :].float().cpu(), sel[0, :, r, :].cpu())
                )
        return sel

    MSAIndexer.select_blocks = patched


def _needle_blocks(text, gold, tok, block_size):
    """Token-block ids of the LINE containing `gold`, plus its token span. None if not locatable."""
    n_occ = text.count(gold)
    if n_occ != 1:
        return None, n_occ  # 0 = gold absent (e.g. free-form QA); >1 = ambiguous location
    ci = text.index(gold)
    ls = text.rfind("\n", 0, ci) + 1
    le = text.find("\n", ci)
    le = len(text) if le < 0 else le
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    span = [i for i, (a, b) in enumerate(offs) if b > ls and a < le]
    if not span:
        return None, n_occ
    blocks = sorted({t // block_size for t in span})
    return {"ids": ids, "tok_lo": span[0], "tok_hi": span[-1], "blocks": blocks}, n_occ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the --no-norm-shift export")
    ap.add_argument("--preds", required=True, help="glob of OpenCompass prediction json files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=1000)
    ap.add_argument("--shard", default="0/1", help="i/N -- split items across GPUs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from scripts.msa.hf_msa_oracle import build_model

    si, sn = (int(x) for x in args.shard.split("/"))

    items = []
    for f in sorted(glob.glob(args.preds)):
        for key, rec in json.load(open(f)).items():
            op = rec["origin_prompt"]
            prompt = op[0]["prompt"] if isinstance(op, list) else op
            gold = rec["gold"]
            gold = gold[0] if isinstance(gold, list) else gold
            items.append({"key": f"{os.path.basename(f)}:{key}", "prompt": prompt,
                          "gold": str(gold), "pred": rec.get("prediction", "")})
    items = items[: args.max_items][si::sn]
    print(f"[probe] {len(items)} items (shard {si}/{sn})", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    bk, k = int(cfg["msa_block_size"]), int(cfg["msa_top_k"])

    _install_capture()
    # sdpa, not the oracle's default eager: the 3 dense-prefix layers would otherwise build a
    # [1, 32, 32K, 32K] score matrix (63 GiB). Sparse layers are unaffected -- they never reach it.
    model = build_model(args.model, args.device, torch.bfloat16, sparse=True, attn_impl="sdpa")
    n_idx = 0
    for li, layer in enumerate(model.model.layers):
        ix = getattr(layer.self_attn, "indexer", None)
        if ix is not None:
            ix._probe_layer = li
            n_idx += 1
    print(f"[probe] {n_idx} indexer layers, block_size={bk}, top_k={k}", flush=True)

    results, skipped = [], {"gold_absent": 0, "gold_ambiguous": 0, "no_span": 0}
    for n, it in enumerate(items):
        text = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        nb, n_occ = _needle_blocks(text, it["gold"], tok, bk)
        if nb is None:
            skipped["gold_absent" if n_occ == 0 else
                    ("gold_ambiguous" if n_occ > 1 else "no_span")] += 1
            continue

        ids = nb["ids"]
        row = len(ids) - 1
        _CAP["row"], _CAP["out"] = row, []
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        pos = torch.arange(len(ids), device=args.device).unsqueeze(0)
        with torch.no_grad():
            model(input_ids=x, attention_mask=torch.ones_like(x), position_ids=pos)

        n_vis = (row + bk) // bk  # blocks visible to this query, matching select_blocks
        needle = torch.tensor(nb["blocks"])
        per_layer = []
        for li, M, sel in _CAP["out"]:
            # Rank of the BEST-ranked needle block, per index head, among visible blocks only.
            # `descending` order -> rank 0 is the top-scoring block.
            order = M[:, :n_vis].argsort(dim=-1, descending=True)          # [H, n_vis]
            pos_of = order.argsort(dim=-1)                                  # rank by block id
            ranks = pos_of[:, needle].min(dim=-1).values                    # [H] best needle block
            hit = torch.isin(sel, needle).any(dim=-1)                       # [H] selected?
            per_layer.append({"layer": li, "rank_min": int(ranks.min()),
                              "rank_med": float(ranks.float().median()),
                              "n_heads_hit": int(hit.sum()), "selected": bool(hit.any())})

        correct = it["gold"].lower() in it["pred"].lower()
        results.append({"key": it["key"], "n_tokens": len(ids), "n_vis_blocks": n_vis,
                        "needle_blocks": nb["blocks"], "needle_depth": nb["tok_lo"] / max(row, 1),
                        "correct": correct, "layers": per_layer,
                        "layers_selected": sum(p["selected"] for p in per_layer),
                        "rank_min_over_layers": min(p["rank_min"] for p in per_layer)})
        r = results[-1]
        print(f"[probe] {n + 1}/{len(items)} tok={len(ids)} vis={n_vis} "
              f"needle_blk={nb['blocks']} depth={r['needle_depth']:.2f} "
              f"sel_layers={r['layers_selected']}/{len(per_layer)} "
              f"best_rank={r['rank_min_over_layers']} correct={correct}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"model": args.model, "preds": args.preds, "top_k": k, "block_size": bk,
               "skipped": skipped, "results": results}, open(args.out, "w"))
    print(f"[probe] wrote {args.out}  ({len(results)} items, skipped {skipped})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
