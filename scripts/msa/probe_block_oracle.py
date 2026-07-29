#!/usr/bin/env python3
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
"""MSA block-ORACLE probe on a STOCK model — the go/no-go on ``(B_k, k)`` before writing any code.

Measures, on the **unmodified** model, how much attention mass a *perfect* selector could capture:

    P[i, b, r]   = sum_{j in block b}  p_group[i, j, r]        (block mass; sums to 1 over b)
    oracle_block = sum over the top-k blocks BY TRUE MASS      <- ceiling for any block selector
    oracle_token = sum over the top-(k*B_k) tokens BY TRUE MASS <- ceiling for any token selector

    1.0
     |-- 1 - oracle_token            unreachable at this budget (attention too spread) -- irreducible
     |-- oracle_token - oracle_block GRANULARITY COST (forced to take whole B_k-token blocks)
     +-- oracle_block - captured     INDEX QUALITY GAP (measured later, after Phase 1)

``oracle_block`` is exactly ``sum_{b in I*} P_b`` — the *denominator* of the MSA paper's ``score
recall`` (arXiv 2606.13392 §5.2). Because score recall is oracle-*relative*, it cannot tell you
whether the budget is too small; that is what this probe is for. A trained index branch can never
beat the oracle, so a low ``oracle_block`` invalidates the configuration regardless of training.

Decision rule at ``B_k=128, k=16``, 32K (docs/qwen3_4b_msa/plan.md §10 item 2):
    >= 0.95   config well-suited; proceed
    0.85-0.95 workable; expect a lower captured_mass ceiling
    <= ~0.70  too coarse -- sweep k in {32, 64} before writing code

Teacher construction follows MSA Eq. 9: per-head softmax, then ``1/G`` average over the query heads
of each GQA group (probability-level averaging), then block-sum.

Example
-------
    python3 scripts/msa/probe_block_oracle.py \\
        --model /cb/ml-eng/aarti/models/qwen3_4b_thinking_2507 \\
        --input /cb/ml-eng/aarti/dsa/phase_a/long_docs.jsonl --text-key text \\
        --seq-len 32768 --num-docs 64 --num-queries 512 \\
        --ks 8 16 32 64 --block-sizes 64 128 \\
        --out /cb/ml-eng/aarti/msa/oracle/qwen3_4b_32k.json
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

log = logging.getLogger("oracle")

# Filled in by main(); read by the patched attention function.
_COLLECT = {"on": False, "args": None, "acc": {}, "idx": None}


# --------------------------------------------------------------------------------------- statistics


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[b, H_kv, T, d] -> [b, H_kv*n_rep, T, d] (interleaved, matching HF's repeat_kv)."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


@torch.no_grad()
def _oracle_for_layer(query, key, scaling, layer_idx):
    """Accumulate oracle statistics for one layer.

    ``query`` [b, H_q, T, d] and ``key`` [b, H_kv, T, d] are POST-RoPE (and post-QK-norm for Qwen3).
    Loops over GQA groups so peak memory is O(G * S * T) rather than O(H_q * S * T).
    """
    a = _COLLECT["args"]
    idx = _COLLECT["idx"]  # [S] sampled query positions (long, on device)
    b, h_q, t, d = query.shape
    h_kv = key.shape[1]
    g = h_q // h_kv
    s = idx.numel()
    dev = query.device

    q_sel = query.index_select(2, idx)  # [b, H_q, S, d]
    j = torch.arange(t, device=dev)
    visible = j[None, :] <= idx[:, None]  # [S, T] causal for the sampled rows

    acc = _COLLECT["acc"].setdefault(layer_idx, {})

    for r in range(h_kv):
        qr = q_sel[:, r * g : (r + 1) * g]  # [b, G, S, d]
        kr = key[:, r : r + 1].expand(b, g, t, d)  # [b, G, T, d]
        scores = torch.matmul(qr.float(), kr.float().transpose(-1, -2)) * scaling  # [b, G, S, T]
        scores.masked_fill_(~visible[None, None], float("-inf"))
        p_head = torch.softmax(scores, dim=-1)
        del scores
        p_group = p_head.mean(dim=1)  # [b, S, T]  <- MSA Eq. 9: 1/G average at probability level
        del p_head

        for bk in a.block_sizes:
            nb = math.ceil(t / bk)
            pad = nb * bk - t
            pg = F.pad(p_group, (0, pad)) if pad else p_group
            block_mass = pg.view(b, s, nb, bk).sum(-1)  # [b, S, nb]
            vis_blocks = torch.div(idx, bk, rounding_mode="floor") + 1  # [S]

            for k in a.ks:
                if k > nb:
                    continue
                # A position is only informative if it has MORE visible blocks than the budget;
                # otherwise the oracle is trivially 1.0 and would inflate the mean.
                valid = vis_blocks > k  # [S]
                if not bool(valid.any()):
                    continue
                ob = block_mass.topk(k, dim=-1).values.sum(-1)  # [b, S]
                ot = p_group.topk(min(k * bk, t), dim=-1).values.sum(-1)  # [b, S]
                m = valid.unsqueeze(0).expand_as(ob)  # [b, S]
                cell = acc.setdefault((bk, k), {"blk": 0.0, "tok": 0.0, "n": 0})
                cell["blk"] += float(ob[m].sum())
                cell["tok"] += float(ot[m].sum())
                cell["n"] += int(m.sum())
            del block_mass
        del p_group


# ------------------------------------------------------------------------------ attention interface


def _oracle_attention(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0, **kw):
    """Drop-in HF attention implementation: exact SDPA output + oracle stats as a side effect.

    Registered as ``ALL_ATTENTION_FUNCTIONS["msa_oracle"]``. The model's forward is unchanged --
    we compute the real output with SDPA so downstream layers see correct activations, and only
    *additionally* materialize scores for the sampled query rows.
    """
    if scaling is None:
        scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)

    n_rep = query.shape[1] // key.shape[1]
    k_rep, v_rep = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
    out = F.scaled_dot_product_attention(query, k_rep, v_rep, is_causal=True, scale=scaling)
    out = out.transpose(1, 2).contiguous()  # [b, T, H, d] -- HF convention

    if _COLLECT["on"]:
        _oracle_for_layer(query, key, scaling, int(getattr(module, "layer_idx", -1)))

    return out, None


# ---------------------------------------------------------------------------------------- data / io


def _iter_docs(path, text_key):
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line).get(text_key, "")
    elif path.endswith(".parquet"):
        import pyarrow.parquet as pq

        tbl = pq.read_table(path, columns=[text_key])
        for v in tbl.column(text_key).to_pylist():
            yield v
    else:
        raise ValueError(f"unsupported input: {path} (want .jsonl/.json/.parquet)")


def _build_batches(path, text_key, tok, seq_len, num_docs, shard_id=0, num_shards=1):
    """One doc per row, truncated to seq_len; docs shorter than seq_len are skipped.

    Sharding happens BEFORE tokenization so each rank only tokenizes its own slice -- tokenizing a
    62-doc x 40k-token corpus in all 8 ranks wastes minutes of CPU before any GPU work starts. Safe
    because ``dump_long_docs.py`` already pre-filtered the corpus with *this* tokenizer, so strided
    slices stay balanced.
    """
    texts = [t for t in _iter_docs(path, text_key) if t][:num_docs]
    mine = texts[shard_id::num_shards]
    out, skipped = [], 0
    for text in mine:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < seq_len:
            skipped += 1
            continue
        out.append(ids[:seq_len])
    log.info("shard %d/%d: kept %d of %d assigned docs at seq_len=%d (skipped %d shorter; corpus has %d)",
             shard_id, num_shards, len(out), len(mine), seq_len, skipped, len(texts))
    if not out:
        raise RuntimeError(
            f"shard {shard_id}/{num_shards} got 0 usable docs (assigned {len(mine)} of {len(texts)}); "
            f"lower --num-shards, raise --num-docs, or lower --seq-len={seq_len}"
        )
    return out


# --------------------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="stock HF model path or id (NOT an MSA checkpoint)")
    ap.add_argument("--input", required=True, help=".jsonl/.parquet of long documents")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--seq-len", type=int, default=32768)
    ap.add_argument("--num-docs", type=int, default=64)
    ap.add_argument("--num-queries", type=int, default=512, help="query positions sampled per doc")
    ap.add_argument("--min-pos", type=int, default=0,
                    help="lowest sampled query position; 0 => auto = 2 * max(ks) * max(block_sizes)")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64], help="block budgets to sweep")
    ap.add_argument("--block-sizes", type=int, nargs="+", default=[128], help="B_k values to sweep")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split docs across N processes (one per GPU); merge with merge_oracle.py")
    ap.add_argument("--shard-id", type=int, default=0, help="this process's shard in [0, num-shards)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(a.out + ".log")],
    )
    # Self-reproducing log: full invocation + the env that shaped it.
    log.info("CMD: %s", " ".join([sys.executable] + sys.argv))
    log.info("CWD: %s", os.getcwd())
    log.info("ENV: %s", {k: os.environ.get(k) for k in
                         ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "HF_HOME", "TRANSFORMERS_OFFLINE")})
    log.info("ARGS: %s", vars(a))

    if a.min_pos == 0:
        # Base the sampling floor on the SMALLEST budget so the window stays as wide as possible;
        # larger k are handled by the per-k `vis_blocks > k` validity mask inside _oracle_for_layer,
        # which simply uses fewer of the sampled positions.
        a.min_pos = 2 * min(a.ks) * max(a.block_sizes)
        log.info("min_pos auto-set to %d (= 2 * min(ks) * max(block_sizes))", a.min_pos)
        for k in sorted(a.ks):
            for bk in sorted(a.block_sizes):
                log.info("  effective floor for Bk=%d k=%d: positions >= %d", bk, k, k * bk)
    if a.min_pos >= a.seq_len:
        raise ValueError(f"min_pos={a.min_pos} >= seq_len={a.seq_len}: nothing to sample")

    random.seed(a.seed)
    torch.manual_seed(a.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["msa_oracle"] = _oracle_attention

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        a.model,
        dtype=getattr(torch, a.dtype),
        attn_implementation="msa_oracle",
        trust_remote_code=a.trust_remote_code,
    ).to(a.device).eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    log.info("model: %d layers, %d q heads, %d kv heads, head_dim %s",
             n_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
             getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))

    docs = _build_batches(a.input, a.text_key, tok, a.seq_len, a.num_docs)
    if a.num_shards > 1:
        if not 0 <= a.shard_id < a.num_shards:
            raise ValueError(f"shard-id {a.shard_id} out of range for num-shards {a.num_shards}")
        docs = docs[a.shard_id :: a.num_shards]  # strided => balanced regardless of doc order
        log.info("shard %d/%d: %d docs", a.shard_id, a.num_shards, len(docs))
        if not docs:
            raise RuntimeError("this shard got 0 docs; reduce --num-shards or raise --num-docs")

    # Strided query positions in [min_pos, seq_len) -- deterministic and evenly spread over depth.
    n_q = min(a.num_queries, a.seq_len - a.min_pos)
    positions = torch.linspace(a.min_pos, a.seq_len - 1, n_q).round().long().unique()
    _COLLECT["args"] = a
    _COLLECT["idx"] = positions.to(a.device)
    log.info("sampling %d query positions in [%d, %d)", positions.numel(), a.min_pos, a.seq_len)

    t0 = time.time()
    _COLLECT["on"] = True
    for n, ids in enumerate(docs):
        x = torch.tensor([ids], device=a.device)
        with torch.no_grad():
            model(input_ids=x, use_cache=False)
        if (n + 1) % 8 == 0 or n + 1 == len(docs):
            log.info("doc %d/%d  (%.1fs elapsed, peak %.1f GiB)",
                     n + 1, len(docs), time.time() - t0,
                     torch.cuda.max_memory_allocated() / 2**30 if a.device.startswith("cuda") else 0.0)
    _COLLECT["on"] = False

    # ------------------------------------------------------------------ reduce and report
    per_layer, agg, raw = {}, {}, {}
    for layer_idx, cells in sorted(_COLLECT["acc"].items()):
        per_layer[layer_idx] = {}
        raw[str(layer_idx)] = {}
        for (bk, k), c in sorted(cells.items()):
            # Raw sums + counts so shards can be merged EXACTLY (weighted), not by averaging means.
            raw[str(layer_idx)][f"Bk{bk}_k{k}"] = {"blk": c["blk"], "tok": c["tok"], "n": c["n"]}
            if c["n"] == 0:
                continue
            blk, tokm = c["blk"] / c["n"], c["tok"] / c["n"]
            per_layer[layer_idx][f"Bk{bk}_k{k}"] = {
                "oracle_block": blk,
                "oracle_token": tokm,
                "granularity_cost": tokm - blk,
                "unreachable": 1.0 - tokm,
                "n": c["n"],
            }
            s = agg.setdefault(f"Bk{bk}_k{k}", {"blk": [], "tok": []})
            s["blk"].append(blk)
            s["tok"].append(tokm)

    summary = {}
    for cfg_name, s in sorted(agg.items()):
        summary[cfg_name] = {
            "oracle_block_mean": sum(s["blk"]) / len(s["blk"]),
            "oracle_block_min_layer": min(s["blk"]),
            "oracle_token_mean": sum(s["tok"]) / len(s["tok"]),
            "granularity_cost_mean": sum(s["tok"]) / len(s["tok"]) - sum(s["blk"]) / len(s["blk"]),
            "layers": len(s["blk"]),
        }

    print("\n" + "=" * 96)
    print(f"MSA BLOCK-ORACLE PROBE  --  {a.model}  @  seq_len={a.seq_len}  ({len(docs)} docs)")
    print("=" * 96)
    print(f"{'config':>12} | {'oracle_block':>13} | {'min layer':>10} | {'oracle_token':>13} "
          f"| {'granularity':>11} | {'unreachable':>11}")
    print("-" * 96)
    for cfg_name, v in summary.items():
        print(f"{cfg_name:>12} | {v['oracle_block_mean']:13.4f} | {v['oracle_block_min_layer']:10.4f} "
              f"| {v['oracle_token_mean']:13.4f} | {v['granularity_cost_mean']:11.4f} "
              f"| {1.0 - v['oracle_token_mean']:11.4f}")
    print("-" * 96)
    print("Decision rule (docs/qwen3_4b_msa/plan.md §10.2):  >=0.95 proceed | 0.85-0.95 workable "
          "| <=~0.70 too coarse\nGate on oracle_block AND its per-layer min -- one bad layer matters.")
    print("=" * 96 + "\n")

    payload = {
        "invocation": " ".join([sys.executable] + sys.argv),
        "args": vars(a),
        "model_config": {
            "num_hidden_layers": n_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
            "rope_theta": getattr(cfg, "rope_theta", None),
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        },
        "num_docs": len(docs),
        "shard": {"id": a.shard_id, "of": a.num_shards},
        "query_positions": positions.tolist(),
        "summary": summary,
        "per_layer": per_layer,
        "raw": raw,  # merge_oracle.py consumes this
        "elapsed_s": time.time() - t0,
    }
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    log.info("wrote %s", a.out)


if __name__ == "__main__":
    main()
