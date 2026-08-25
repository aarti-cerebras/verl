#!/usr/bin/env python3
"""Offline bring-up for the Qwen3-DSA vLLM plugin (serving_eval_plan.md §4, P2/P3).

In-process ``LLM``, so a failure shows a real traceback instead of an EngineCore death notice.

Modes:
  DSA_SPARSE=0  -> every layer stock dense Qwen3Attention, indexer weights dropped. Answers
                   "does the scaffolding load and generate" independently of the sparse kernels.
  DSA_SPARSE=1  -> the sparse path.

``--top-k N`` overrides ``index_topk``/``dsa_top_k`` in a COPY of the serving dir's config (the dir
itself is never mutated), which is how the dense-equivalence control at ``top_k >= T`` is run.

Usage:
  DSA_SPARSE=0 .devlibs/vllm026/bin/python tests/dsa/qwen3_dsa_offline_smoke.py --model <dir>
"""

import argparse
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import scripts.dsa.vllm_qwen3_dsa  # noqa: F401,E402  (registers the arch + CUSTOM backend)


def maybe_override_topk(model_dir: str, top_k: int | None, tmpdir: str) -> str:
    """Return a dir whose config has ``index_topk == dsa_top_k == top_k``, symlinking the weights."""
    if top_k is None:
        return model_dir
    out = os.path.join(tmpdir, f"topk{top_k}")
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(model_dir):
        if name == "config.json":
            continue
        src, dst = os.path.join(model_dir, name), os.path.join(out, name)
        if not os.path.exists(dst) and os.path.isfile(src):
            os.symlink(src, dst)
    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["dsa_top_k"] = top_k
    cfg["index_topk"] = top_k
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/cb/ml-eng/aarti/dsa_qwen3/serving/p2_mix5050_k2048_step1200")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=None, help="override index_topk (dense-eq control)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--eager", action="store_true", default=True)
    ap.add_argument("--cudagraph", dest="eager", action="store_false")
    ap.add_argument("--cudagraph-mode", default=None,
                    help="FULL_AND_PIECEWISE (vLLM default) | PIECEWISE | FULL | NONE")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="prepend roughly N filler tokens to exercise a longer prefill")
    ap.add_argument("--out-json", default=None, help="dump {prompt_idx: text} for ladder diffing")
    ap.add_argument("--label", default="", help="tag recorded in --out-json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tmp = tempfile.mkdtemp(prefix="dsa_smoke_")
    try:
        model = maybe_override_topk(args.model, args.top_k, tmp)
        sparse = os.environ.get("DSA_SPARSE", "1") not in ("0", "", "false", "False")
        print(f"[smoke] model={model} DSA_SPARSE={int(sparse)} eager={args.eager}")

        llm = LLM(
            model=model,
            tokenizer=args.model,
            trust_remote_code=True,
            tensor_parallel_size=1,
            block_size=64,  # mandatory for the DSA path
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            dtype="bfloat16",
            enforce_eager=args.eager,
            **({"compilation_config": {"cudagraph_mode": args.cudagraph_mode}}
               if args.cudagraph_mode else {}),
            enable_prefix_caching=False,
            seed=1234,
        )
        # A long, non-repetitive filler: repeated text would let ANY selector succeed (every
        # position is interchangeable), which would make the sparsity ladder meaningless. Numbered
        # sentences also give the needle question a unique answer to retrieve.
        sents = [
            f"Fact {i}: the code word for item {i} is {(i * 7919) % 10007}."
            for i in range(max(args.prompt_tokens // 14, 1))
        ]
        filler = " ".join(sents) + "\n" if args.prompt_tokens else ""
        needle_i = max(len(sents) // 3, 0)
        prompts = [
            filler + "Q: What is 17 * 23? Think briefly, then answer.\nA:",
            filler + f"Q: Repeat the code word for item {needle_i} exactly.\nA:",
        ]
        outs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=1234),
        )
        rec = {}
        for i, o in enumerate(outs):
            text = o.outputs[0].text
            rec[str(i)] = {
                "n_prompt_tok": len(o.prompt_token_ids),
                "token_ids": list(o.outputs[0].token_ids),
                "text": text,
            }
            print(f"[smoke] prompt {i}: n_prompt_tok={len(o.prompt_token_ids)} -> "
                  f"{text.replace(chr(10), ' ')[:200]!r}")
        if args.out_json:
            with open(args.out_json, "w") as fh:
                json.dump({"label": args.label, "sparse": int(sparse), "top_k": args.top_k,
                           "outputs": rec}, fh, indent=2)
            print(f"[smoke] wrote {args.out_json}")
        if len(sents) > 1:
            want = str((needle_i * 7919) % 10007)
            got = want in rec["1"]["text"]
            print(f"[smoke] needle(item {needle_i} = {want}) retrieved: {got}")
        print("[smoke] OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
