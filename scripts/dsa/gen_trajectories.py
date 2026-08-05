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
"""DSA Phase-2 — TRAJECTORY GENERATION (see docs/dsa_phase2_plan.md, T5).

Self-generates behavior-cloning trajectories: serves the **stock dense MiniCPM3-4B** and, for each selected
prompt (from select_prompts.py), samples a response at **temperature 0.7** with a **per-domain
``max_new_tokens``** cap. The generated response is the SFT target.

Parallelism:
  * ``--data-parallel-size N`` (recommended for this 4B model): the launcher spawns N child processes, each
    pinned to its own GPU slice (``CUDA_VISIBLE_DEVICES``) running ``tensor_parallel_size`` GPUs, each
    processing a ``prompts[rank::N]`` shard; parts are merged at the end. **DP=8, TP=1 is optimal for a 4B
    model** — TP=8 alone is communication-bound and leaves the GPUs mostly idle.
  * ``--tensor-parallel-size K`` shards one replica across K GPUs (use only for models too big for one GPU).

Backends: ``vllm`` (default) or ``hf`` (transformers ``generate`` fallback).

Output JSONL row: {source_uid, source_dataset, source_config, domain, lang, prompt_sha256,
    messages:[user, {assistant: generated}], prompt_tokens, resp_tokens, total_tokens, finish_reason,
    temperature, top_p, seed, max_new_tokens}. Post-filter drops finish_reason==length (runaways).
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dsa_log import setup_logging  # noqa: E402
from _dsa_tok import chat_prefix_ids  # noqa: E402

# Per-domain max_new_tokens: math/code (incl. ZH math) = 16K ("cap high, filter after"); all others = 4K.
DEFAULT_CAPS = {
    "Math": 16384, "Code": 16384, "Multi-lang-Math": 16384,
    "Chinese-general": 4096, "IF": 4096, "Knowledge": 4096, "Multi-lang-Knowledge": 4096,
}
DEFAULT_CAP_FALLBACK = 4096


def _pctl(xs, q):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1))))]


def _log_dist(logger, rows):
    by = {}
    for r in rows:
        by.setdefault(r["domain"], []).append(r["resp_tokens"])
    for dom, rl in sorted(by.items()):
        logger.info(
            "[%s] n=%d resp p50=%d p90=%d p99=%d max=%d  (>=512:%d%% >=1024:%d%%)",
            dom, len(rl), _pctl(rl, 50), _pctl(rl, 90), _pctl(rl, 99), max(rl),
            round(100 * sum(x >= 512 for x in rl) / len(rl)),
            round(100 * sum(x >= 1024 for x in rl) / len(rl)),
        )


def _row(p, resp_text, prompt_tokens, resp_tokens, finish, args, caps, *,
         resp_token_ids=None, prefix_tokens=None, prefix_token_ids=None, max_new_tokens=None, sample_idx=0):
    """One trajectory row.

    ``prefix_tokens`` / ``resp_token_ids`` are what make the SFT conversion exact: the converter rebuilds
    the served prefix from ``messages`` and asserts its length equals ``prefix_tokens``, then splices
    ``resp_token_ids`` verbatim — no decode/re-encode round trip, which BPE does not guarantee.
    See docs/qwen3_4b_msa/phase2_data_gen.md §6.
    """
    row = {
        "source_uid": p.get("source_uid"),
        "source_dataset": p.get("source_dataset"),
        "source_config": p.get("source_config"),
        "original_dataset": p.get("original_dataset"),
        "domain": p["domain"],
        "lang": p.get("lang"),
        "prompt_sha256": p.get("prompt_sha256"),
        "messages": p["messages"] + [{"role": "assistant", "content": resp_text}],
        "prompt_tokens": prompt_tokens,
        "resp_tokens": resp_tokens,
        "total_tokens": prompt_tokens + resp_tokens,
        "finish_reason": finish,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "sample_idx": sample_idx,
        "max_new_tokens": max_new_tokens if max_new_tokens is not None
        else caps.get(p["domain"], DEFAULT_CAP_FALLBACK),
    }
    if prefix_tokens is not None:
        row["prefix_tokens"] = prefix_tokens
    if prefix_token_ids is not None:
        row["prefix_token_ids"] = list(prefix_token_ids)
    if resp_token_ids is not None:
        row["resp_token_ids"] = list(resp_token_ids)
    if args.fit_window:
        row["window"] = args.fit_window
    return row


def _load_done_shas(out_path, logger):
    """Resume support: prompt_sha256 already present in an existing (partial) output file."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("prompt_sha256"))
                except Exception:
                    pass
        if done:
            logger.info("resume: %d trajectories already in %s — skipping those prompts", len(done), out_path)
    return done


def run_vllm(prompts, tok, args, caps, logger):
    """Generate in chunks of --chunk-size, APPENDING each chunk to args.out (incremental checkpointing +
    resumable). If args.out already has trajectories, those prompts are skipped."""
    from vllm import LLM, SamplingParams

    done = _load_done_shas(args.out, logger)
    todo = [p for p in prompts if p.get("prompt_sha256") not in done]
    logger.info("to generate: %d (of %d; %d already done)", len(todo), len(prompts), len(prompts) - len(todo))
    results = []
    if not todo:
        logger.info("nothing to do — all prompts already generated")
        return results

    max_cap = max(list(caps.values()) + [DEFAULT_CAP_FALLBACK])
    max_model_len = args.max_model_len or (args.fit_window or (max_cap + 4096))
    logger.info("vLLM: tp=%d dtype=%s max_model_len=%d gpu_mem_util=%.2f chunk=%d fit_window=%s",
                args.tensor_parallel_size, args.dtype, max_model_len, args.gpu_memory_utilization,
                args.chunk_size, args.fit_window or "off (per-domain caps)")
    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size,
              dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max_model_len, seed=args.seed)

    written = 0
    n_skipped_nofit = 0
    n_prefix_warned = False
    with open(args.out, "a") as fout:  # append -> survives crash/kill; relaunch resumes from here
        for s in range(0, len(todo), args.chunk_size):
            batch = todo[s : s + args.chunk_size]
            reqs, sps, kept = [], [], []
            for p in batch:
                # Feed vLLM the TEMPLATED STRING, not prompt_token_ids: vLLM 0.20.2's
                # _validate_model_input does `max(tokenizer.max_token_id, vocab-1)`, and under
                # transformers 5.3.0 `max_token_id` is a str -> TypeError on the token-ids path only.
                # The prefix we record for training comes back from the engine itself as
                # `o.prompt_token_ids` (what generation actually conditioned on), which is a stronger
                # guarantee than tokenizing it ourselves. phase2_data_gen.md §6.
                text = tok.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=False)
                prefix_ids = chat_prefix_ids(tok, p["messages"])
                if args.fit_window:
                    # generation window == training window: leave room for the closing <|im_end|>
                    cap = args.fit_window - len(prefix_ids) - 1
                    if cap <= 0:
                        n_skipped_nofit += 1
                        continue
                else:
                    cap = caps.get(p["domain"], DEFAULT_CAP_FALLBACK)
                sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                    min_p=args.min_p, max_tokens=cap, n=args.n, seed=args.seed)
                if args.repetition_penalty and args.repetition_penalty != 1.0:
                    sp.repetition_penalty = args.repetition_penalty
                if args.presence_penalty:
                    sp.presence_penalty = args.presence_penalty
                reqs.append(text)
                sps.append(sp)
                kept.append((p, len(prefix_ids), cap))
            if not reqs:
                continue
            outs = llm.generate(reqs, sps)
            for (p, n_local, cap), o in zip(kept, outs, strict=True):
                served = [int(t) for t in o.prompt_token_ids]  # exactly what generation conditioned on
                if len(served) != n_local and not n_prefix_warned:
                    logger.warning("served prefix %d != locally tokenized %d — recording the SERVED prefix; "
                                   "the --fit-window cap used the local length", len(served), n_local)
                    n_prefix_warned = True
                for si, comp in enumerate(o.outputs):  # honour n>1: one row per sample
                    row = _row(p, comp.text, len(served), len(comp.token_ids), comp.finish_reason, args, caps,
                               resp_token_ids=comp.token_ids, prefix_tokens=len(served),
                               prefix_token_ids=served, max_new_tokens=cap, sample_idx=si)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    results.append(row)
            fout.flush()
            os.fsync(fout.fileno())  # durable checkpoint
            written += len(batch)
            logger.info("chunk done: %d/%d written (%.0f%%) -> %s",
                        written, len(todo), 100.0 * written / len(todo), args.out)
    if n_skipped_nofit:
        logger.warning("skipped %d prompts: no room left in --fit-window=%d", n_skipped_nofit, args.fit_window)
    return results


def run_hf(prompts, tok, args, caps, logger):
    import torch
    from transformers import AutoModelForCausalLM

    logger.info("HF backend: loading %s (device_map=auto, bf16)...", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    by_dom = {}
    for i, p in enumerate(prompts):
        by_dom.setdefault(p["domain"], []).append(i)
    results = [None] * len(prompts)
    for dom, idxs in by_dom.items():
        cap = caps.get(dom, DEFAULT_CAP_FALLBACK)
        for s in range(0, len(idxs), args.hf_batch_size):
            bi = idxs[s : s + args.hf_batch_size]
            texts = [tok.apply_chat_template(prompts[i]["messages"], add_generation_prompt=True, tokenize=False)
                     for i in bi]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=(args.max_model_len or cap + 4096)).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=cap, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(bi):
                in_len = int(enc["attention_mask"][j].sum())
                out_ids = gen[j][enc["input_ids"].shape[1]:]
                resp = tok.decode(out_ids, skip_special_tokens=True)
                resp_tokens = int((out_ids != tok.pad_token_id).sum())
                results[i] = _row(prompts[i], resp, in_len, resp_tokens,
                                  "length" if resp_tokens >= cap else "stop", args, caps)
            logger.info("[%s] generated %d/%d", dom, min(s + args.hf_batch_size, len(idxs)), len(idxs))
    return results


def launch_dp(args, logger):
    """Launcher role: spawn `data_parallel_size` child replicas, each on its own GPU slice + prompt shard."""
    dp, tp = args.data_parallel_size, args.tensor_parallel_size
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    devs = cvd.split(",") if cvd else [str(i) for i in range(dp * tp)]
    if len(devs) < dp * tp:
        logger.warning("only %d visible GPUs for data_parallel*tensor_parallel=%d", len(devs), dp * tp)
    procs, parts = [], []
    for r in range(dp):
        part = f"{args.out}.part{r}"
        parts.append(part)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devs[r * tp : (r + 1) * tp])
        # Each replica is an independent single-node vLLM engine that opens its own torch.distributed
        # TCPStore. Left to themselves, N children racing through get_open_port() collide -> the loser
        # dies with `EADDRINUSE`. Hand out disjoint port blocks (and stagger the spawns) instead.
        env["VLLM_PORT"] = str(args.vllm_port_base + r * 16)
        cmd = [sys.executable, os.path.abspath(__file__),
               "--prompts", args.prompts, "--out", part, "--log-dir", args.log_dir,
               "--model", args.model, "--backend", args.backend,
               "--temperature", str(args.temperature), "--top-p", str(args.top_p),
               "--top-k", str(args.top_k), "--min-p", str(args.min_p),
               "--presence-penalty", str(args.presence_penalty), "--fit-window", str(args.fit_window),
               "--n", str(args.n), "--seed", str(args.seed),
               "--repetition-penalty", str(args.repetition_penalty), "--chunk-size", str(args.chunk_size),
               "--tensor-parallel-size", str(tp), "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--num-nodes", str(args.num_nodes), "--node-rank", str(args.node_rank),
               "--dp-rank", str(r), "--dp-world", str(dp)]
        if args.max_new_tokens_json:
            cmd += ["--max-new-tokens-json", args.max_new_tokens_json]
        if args.max_model_len:
            cmd += ["--max-model-len", str(args.max_model_len)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        logger.info("DP rank %d/%d on GPUs [%s] VLLM_PORT=%s -> %s",
                    r, dp, env["CUDA_VISIBLE_DEVICES"], env["VLLM_PORT"], part)
        procs.append(subprocess.Popen(cmd, env=env))
        if args.launch_stagger and r + 1 < dp:
            time.sleep(args.launch_stagger)  # keep 8 engine startups from racing each other
    rc = 0
    for r, p in enumerate(procs):
        pr = p.wait()
        if pr != 0:
            logger.error("DP rank %d FAILED rc=%d", r, pr)
            rc = pr
    if rc:
        sys.exit(rc)
    if args.no_merge:
        # The merge duplicates every byte of the .partN files. trajectories_to_sft_parquet.py and
        # analyze_lengths.py both accept the parts directly, so on a tight filesystem the copy is pure
        # waste — skip it and read the glob instead.
        n = sum(1 for p in parts if os.path.exists(p) for _ in open(p))
        logger.info("--no-merge: leaving %d parts in place (%d rows). Consume them with the glob '%s.part*'",
                    len(parts), n, args.out)
        return
    results = []
    with open(args.out, "w") as fout:
        for part in parts:
            with open(part) as f:
                for line in f:
                    fout.write(line)
                    if line.strip():
                        results.append(json.loads(line))
    logger.info("merged %d DP parts -> %d trajectories -> %s", len(parts), len(results), args.out)
    _log_dist(logger, results)


def main():
    ap = argparse.ArgumentParser(description="Generate self-gen trajectories from dense MiniCPM3-4B.")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--model", default="openbmb/MiniCPM3-4B")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=-1, help="-1 = disabled; Qwen3-4B-Thinking-2507 wants 20")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="last resort for thinking-mode repetition loops; changes the cloned distribution")
    ap.add_argument("--fit-window", type=int, default=0,
                    help="generation window == training window (e.g. 32768): per-row max_tokens = "
                         "window - len(prefix) - 1, so no sample can exceed the training length. "
                         "Overrides the per-domain caps. See docs/qwen3_4b_msa/phase2_data_gen.md §5.1")
    ap.add_argument("--n", type=int, default=1, help="samples per prompt; each becomes its own row (sample_idx)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-new-tokens-json", default=None, help='override caps, e.g. \'{"Math":8192}\'')
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--vllm-port-base", type=int, default=51000,
                    help="each DP replica gets VLLM_PORT = base + rank*16; disjoint blocks avoid the "
                         "EADDRINUSE race between simultaneously starting engines")
    ap.add_argument("--launch-stagger", type=float, default=3.0,
                    help="seconds between spawning DP replicas (0 to disable)")
    ap.add_argument("--no-merge", action="store_true",
                    help="skip concatenating the .partN files into --out at the end (the copy doubles disk "
                         "usage; downstream scripts accept the '<out>.part*' glob)")
    ap.add_argument("--data-parallel-size", type=int, default=1, help="N replicas, each on its own GPU slice")
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="GPUs per replica (1 for a 4B model)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--hf-batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=512,
                    help="vLLM: generate + append to --out every this many prompts (checkpoint/resume)")
    ap.add_argument("--limit", type=int, default=0, help="cap total prompts (0=all)")
    ap.add_argument("--num-nodes", type=int, default=1, help="split prompts across this many nodes (disjoint)")
    ap.add_argument("--node-rank", type=int, default=0, help="this node's rank in [0, num-nodes)")
    ap.add_argument("--dp-rank", type=int, default=-1, help="internal: child replica rank")
    ap.add_argument("--dp-world", type=int, default=0, help="internal: child replica count")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    log_dir = args.log_dir or os.path.join(out_dir, "logs")
    is_child = args.dp_rank >= 0
    logger, _ = setup_logging(f"gen_trajectories_dp{args.dp_rank}" if is_child else "gen_trajectories", log_dir)
    logger.info("config: %s", vars(args))

    # Launcher role: fan out to child replicas and merge.
    if args.data_parallel_size > 1 and not is_child:
        launch_dp(args, logger)
        logger.info("DONE (data-parallel launcher)")
        return

    caps = dict(DEFAULT_CAPS)
    if args.max_new_tokens_json:
        caps.update(json.loads(args.max_new_tokens_json))
    logger.info("per-domain max_new_tokens: %s (fallback %d)", caps, DEFAULT_CAP_FALLBACK)

    prompts = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if args.limit:
        prompts = prompts[: args.limit]
    # Global shard across (nodes x dp-replicas): every worker gets a disjoint slice; union = all prompts.
    # global_world = num_nodes * dp_world ; global_rank = node_rank * dp_world + dp_rank.
    world = args.dp_world if is_child else 1
    rank = args.dp_rank if is_child else 0
    gw = args.num_nodes * world
    gr = args.node_rank * world + rank
    if gw > 1:
        n_before = len(prompts)
        prompts = prompts[gr::gw]
        logger.info("shard node=%d/%d dp=%d/%d -> global %d/%d : %d of %d prompts",
                    args.node_rank, args.num_nodes, rank, world, gr, gw, len(prompts), n_before)
    else:
        logger.info("loaded %d prompts from %s", len(prompts), args.prompts)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.backend == "vllm":
        results = run_vllm(prompts, tok, args, caps, logger)  # writes incrementally (append + resume)
    else:
        results = run_hf(prompts, tok, args, caps, logger)
        with open(args.out, "w") as fout:
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("done: %d new trajectories -> %s", len(results), args.out)
    if not is_child:
        _log_dist(logger, results)
    logger.info("DONE")


if __name__ == "__main__":
    main()
