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
import pathlib
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


# --- approximate-selector telemetry -------------------------------------------------------------
# Telemetry accumulates inside the EngineCore WORKER, a subprocess vLLM terminates rather than
# exiting cleanly, so the `atexit` dump registered by SelectorRuntime.configure never runs and the
# artifact is never written. (This file's docstring predates vLLM defaulting
# VLLM_ENABLE_V1_MULTIPROCESSING=1, so `LLM` is no longer in-process.) Ask the live worker instead,
# the way dsa-csx does. The methods live on a worker extension because vLLM 0.26 refuses to
# serialize a bare callable for collective_rpc without VLLM_ALLOW_INSECURE_SERIALIZATION=1; naming a
# method on the worker is the supported route. Both methods no-op unless the approximate plugin is
# the loaded one, so the exact-plugin arms are unaffected.

WORKER_EXTENSION_CLS = "scripts.dsa.selector_telemetry_rpc.SelectorTelemetryExtension"
BUCKET_WORKER_EXTENSION_CLS = "scripts.dsa.bucket_telemetry_rpc.BucketTelemetryExtension"


def telemetry_kind(model_dir: str) -> str | None:
    """Return the isolated telemetry family selected by the model architecture."""

    try:
        with open(os.path.join(model_dir, "config.json")) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return None
    architectures = config.get("architectures") or []
    if any("Bucketed" in str(name) for name in architectures):
        return "bucket"
    if any("Approx" in str(name) for name in architectures):
        return "selector"
    return None


def telemetry_worker_extension(model_dir: str) -> str:
    return BUCKET_WORKER_EXTENSION_CLS if telemetry_kind(model_dir) == "bucket" else WORKER_EXTENSION_CLS


def selector_telemetry_expected(model_dir: str) -> bool:
    """Whether THIS model should produce a selector artifact.

    Needed because "no worker returned telemetry" is ambiguous: legitimate for an exact-plugin arm,
    but for an approximate arm with telemetry on it means the artifact is missing -- and a missing
    artifact reads as "no violations" to the runner's gate (`any({})` is False). Without this the
    empty case fails open, which is the vacuous verdict we were trying to remove.
    """

    try:
        with open(os.path.join(model_dir, "config.json")) as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        return False
    architectures = config.get("architectures") or []
    if any("Bucketed" in str(name) for name in architectures):
        return str(config.get("dsa_bucket_telemetry", "off")) != "off"
    if any("Approx" in str(name) for name in architectures):
        return str(config.get("dsa_telemetry", "off")) != "off"
    return False


def reset_selector_telemetry(llm, *, kind: str | None = None) -> None:
    """Discard warmup observations so the safety gate only ever sees real traffic.

    A transport failure is FATAL, not a warning. The worker extension is attached on every arm, so
    the call reaching nothing means the wiring is broken -- and continuing would fold vLLM's
    synthetic autotune batches into the artifact while the run still reported success.
    """

    try:
        method = "bucket_telemetry_reset" if kind == "bucket" else "selector_telemetry_reset"
        done = llm.collective_rpc(method)
    except Exception as exc:
        raise RuntimeError(
            f"selector telemetry reset failed: {exc!r}. Warmup observations would be attributed "
            f"to served traffic, so this run cannot produce a trustworthy safety artifact."
        ) from exc
    if any(done):
        print(f"[smoke] {kind or 'selector'} telemetry reset on {sum(map(bool, done))} worker(s)")


def write_selector_artifact(
    llm,
    path: str | None,
    *,
    expected: bool,
    model: str = "",
    kind: str | None = None,
) -> None:
    """Pull the artifact out of the worker(s) and write it from THIS process.

    FATAL on transport failure, and also fatal on an EMPTY result when `expected` -- see
    `selector_telemetry_expected`. An absent artifact reads as "no violations" to the runner's gate,
    so neither a broken transport nor a silently inactive plugin may pass as a clean run.
    """

    if not path:
        return
    try:
        method = "bucket_telemetry_artifact" if kind == "bucket" else "selector_telemetry_artifact"
        payloads = llm.collective_rpc(method)
    except Exception as exc:
        raise RuntimeError(
            f"selector telemetry read failed: {exc!r}. DSA_SELECTOR_ARTIFACT={path} was requested "
            f"but no artifact was produced, and a missing artifact reads as 'no violations'."
        ) from exc
    payloads = [item for item in payloads if item]
    if not payloads:
        if expected:
            raise RuntimeError(
                f"no worker returned {kind or 'selector'} telemetry, but {model} enables it, so "
                f"an artifact was expected at {path}. The matching plugin/worker extension is "
                f"not active, and a missing artifact reads as 'no violations'."
            )
        print("[smoke] no selector telemetry to write (exact plugin: none expected)")
        return
    body = payloads[0] if len(payloads) == 1 else {"ranks": payloads}
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(body, indent=2, default=str) + "\n")
    temporary.replace(target)
    if len(payloads) == 1:
        rows = (body.get("safety") or {}).get("rows")
        if rows is None:
            rows = ((body.get("graph_replay") or {}).get("global") or {}).get("rows", "?")
    else:
        rows = "?"
    print(f"[smoke] wrote {kind or 'selector'} telemetry {target} (safety rows={rows})")


def build_prompts(prompt_tokens: int, decode_batch_size: int) -> tuple[list[str], list[str], int]:
    """Build a heterogeneous batch that keeps every request alive for decode validation."""

    if decode_batch_size < 2:
        raise ValueError(f"decode_batch_size must be at least 2, got {decode_batch_size}")
    sents = [
        f"Fact {i}: the code word for item {i} is {(i * 7919) % 10007}."
        for i in range(max(prompt_tokens // 14, 1))
    ]
    filler = " ".join(sents) + "\n" if prompt_tokens else ""
    needle_i = max(len(sents) // 3, 0)
    prompts = [
        filler + "Q: What is 17 * 23? Think briefly, then answer.\nA:",
        filler + f"Q: Repeat the code word for item {needle_i} exactly.\nA:",
    ]
    for request_index in range(2, decode_batch_size):
        # Drop a different number of complete facts from each extra request. This keeps the long
        # context content meaningful while guaranteeing heterogeneous request lengths. A batch of
        # seven therefore exercises vLLM's 7 -> 8 FULL_DECODE_ONLY graph padding instead of merely
        # replaying the historically covered, naturally captured two-request shape.
        if prompt_tokens:
            kept = max(1, len(sents) - request_index)
            request_filler = " ".join(sents[:kept]) + "\n"
        else:
            request_filler = " ".join(
                f"Check value {value}." for value in range(request_index + 1)
            ) + "\n"
        prompts.append(
            request_filler
            + f"Q: State the request marker {request_index}, then briefly summarize the final fact.\nA:"
        )
    return prompts, sents, needle_i


def serialize_step_logprobs(steps) -> list[dict[str, float]]:
    """Convert vLLM's per-step ``Logprob`` objects into stable JSON data."""

    return [
        {str(int(token_id)): float(item.logprob) for token_id, item in step.items()}
        for step in (steps or [])
    ]


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
    ap.add_argument(
        "--decode-batch-size",
        type=int,
        default=2,
        help=(
            "number of heterogeneous prompts submitted in one LLM.generate call; use 7 to force "
            "the FULL_DECODE_ONLY 7-to-8 padded capture bucket"
        ),
    )
    ap.add_argument(
        "--logprobs",
        type=int,
        default=0,
        help=(
            "save this many top token logprobs per generation step in --out-json; "
            "0 keeps the normal compact artifact"
        ),
    )
    ap.add_argument(
        "--architecture-override",
        default=None,
        help=(
            "diagnostic Hugging Face architecture override; permits loading the same serving "
            "directory through a different registered implementation"
        ),
    )
    ap.add_argument("--out-json", default=None, help="dump {prompt_idx: text} for ladder diffing")
    ap.add_argument("--label", default="", help="tag recorded in --out-json")
    args = ap.parse_args()
    if args.logprobs < 0:
        ap.error("--logprobs must be nonnegative")

    from vllm import LLM, SamplingParams

    tmp = tempfile.mkdtemp(prefix="dsa_smoke_")
    try:
        model = maybe_override_topk(args.model, args.top_k, tmp)
        telemetry = telemetry_kind(model)
        if telemetry == "bucket":
            # The reset RPC below marks the real-traffic boundary. Avoid running expensive
            # host-folded exact comparisons during synthetic profiling/autotune forwards.
            os.environ.setdefault("DSA_BUCKET_DEFER_HOST_TELEMETRY", "1")
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
            worker_extension_cls=telemetry_worker_extension(model),
            **(
                {"hf_overrides": {"architectures": [args.architecture_override]}}
                if args.architecture_override
                else {}
            ),
        )
        # A long, non-repetitive filler: repeated text would let ANY selector succeed (every
        # position is interchangeable), which would make the sparsity ladder meaningless. Numbered
        # sentences also give the needle question a unique answer to retrieve.
        prompts, sents, needle_i = build_prompts(
            args.prompt_tokens, args.decode_batch_size
        )
        # Everything observed so far came from vLLM's warmup/autotune dummy batches, whose
        # operands are synthetic; folding them into the artifact would attribute them to real
        # traffic and let the runner's hard-violation gate fire (or hide) on garbage.
        reset_selector_telemetry(llm, kind=telemetry)

        outs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0.0,
                max_tokens=args.max_tokens,
                min_tokens=args.max_tokens,
                ignore_eos=True,
                seed=1234,
                logprobs=args.logprobs or None,
            ),
        )
        if len(outs) != args.decode_batch_size:
            raise RuntimeError(
                f"concurrent decode fixture returned {len(outs)} outputs for "
                f"decode_batch_size={args.decode_batch_size}"
            )
        prompt_lengths = [len(output.prompt_token_ids) for output in outs]
        decode_lengths = [len(output.outputs[0].token_ids) for output in outs]
        if any(length != args.max_tokens for length in decode_lengths):
            raise RuntimeError(
                "concurrent decode fixture did not keep every request alive for the forced decode "
                f"window: expected={args.max_tokens}, observed={decode_lengths}"
            )
        distinct_lengths = len(set(prompt_lengths))
        if args.decode_batch_size > 2 and distinct_lengths < 3:
            raise RuntimeError(
                "concurrent decode fixture did not produce heterogeneous prompt lengths: "
                f"{prompt_lengths}"
            )
        print(
            f"[smoke] concurrent decode batch: requests={args.decode_batch_size} "
            f"distinct_prompt_lengths={distinct_lengths} min={min(prompt_lengths)} "
            f"max={max(prompt_lengths)} forced_decode_tokens={args.max_tokens}"
        )
        rec = {}
        for i, o in enumerate(outs):
            completion = o.outputs[0]
            text = completion.text
            rec[str(i)] = {
                "n_prompt_tok": len(o.prompt_token_ids),
                "token_ids": list(completion.token_ids),
                "text": text,
            }
            if args.logprobs:
                step_logprobs = serialize_step_logprobs(completion.logprobs)
                if len(step_logprobs) != len(completion.token_ids):
                    raise RuntimeError(
                        "requested generation logprobs are incomplete: "
                        f"prompt={i} tokens={len(completion.token_ids)} "
                        f"logprob_steps={len(step_logprobs)}"
                    )
                rec[str(i)]["step_logprobs"] = step_logprobs
            print(f"[smoke] prompt {i}: n_prompt_tok={len(o.prompt_token_ids)} -> "
                  f"{text.replace(chr(10), ' ')[:200]!r}")
        needle_retrieved = None
        if len(sents) > 1:
            want = str((needle_i * 7919) % 10007)
            needle_retrieved = want in rec["1"]["text"]
            print(f"[smoke] needle(item {needle_i} = {want}) retrieved: {needle_retrieved}")
            if not needle_retrieved:
                raise RuntimeError(f"long-context needle item {needle_i}={want} was not retrieved")
        if args.out_json:
            artifact = {
                "label": args.label,
                "sparse": int(sparse),
                "top_k": args.top_k,
                "decode_batch_size": args.decode_batch_size,
                "needle_retrieved": needle_retrieved,
                "outputs": rec,
            }
            if args.architecture_override:
                artifact["architecture_override"] = args.architecture_override
            if args.logprobs:
                artifact["logprobs"] = args.logprobs
            with open(args.out_json, "w") as fh:
                json.dump(artifact, fh, indent=2)
            print(f"[smoke] wrote {args.out_json}")
        write_selector_artifact(
            llm,
            os.environ.get("DSA_BUCKET_TELEMETRY_ARTIFACT")
            if telemetry == "bucket"
            else os.environ.get("DSA_SELECTOR_ARTIFACT"),
            expected=selector_telemetry_expected(model),
            model=model,
            kind=telemetry,
        )

        print("[smoke] OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
