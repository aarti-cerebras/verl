#!/usr/bin/env python3
"""Compare eager and FULL_DECODE_ONLY bucketed serving with one mode per process.

The GPU handoff runner invokes this file once per execution mode. Measurements exclude engine
construction and CUDA graph capture. Each measured batch reports whole-generation wall time and,
when vLLM exposes request timestamps, the decode window from first emitted token to completion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen3_dsa_offline_smoke as smoke  # noqa: E402


def parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not sizes or any(size < 2 for size in sizes):
        raise argparse.ArgumentTypeError("every batch size must be at least 2")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return sizes


def request_timestamp(output: Any, name: str) -> float | None:
    metrics = getattr(output, "metrics", None)
    value = getattr(metrics, name, None)
    return float(value) if value is not None else None


def run_batch(llm: Any, *, batch_size: int, prompt_tokens: int, max_tokens: int) -> dict[str, Any]:
    from vllm import SamplingParams

    prompts, sentences, needle_index = smoke.build_prompts(prompt_tokens, batch_size)
    started = time.perf_counter()
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            min_tokens=max_tokens,
            ignore_eos=True,
            seed=1234,
        ),
    )
    generation_wall_s = time.perf_counter() - started

    if len(outputs) != batch_size:
        raise RuntimeError(f"expected {batch_size} outputs, observed {len(outputs)}")
    prompt_lengths = [len(output.prompt_token_ids) for output in outputs]
    decode_lengths = [len(output.outputs[0].token_ids) for output in outputs]
    if any(length != max_tokens for length in decode_lengths):
        raise RuntimeError(f"expected {max_tokens} decode tokens per request, got {decode_lengths}")
    if batch_size > 2 and len(set(prompt_lengths)) < 3:
        raise RuntimeError(f"batch lacks heterogeneous prompt lengths: {prompt_lengths}")

    wanted_code = str((needle_index * 7919) % 10007)
    needle_retrieved = wanted_code in outputs[1].outputs[0].text if len(sentences) > 1 else None
    if needle_retrieved is False:
        raise RuntimeError(f"needle item {needle_index}={wanted_code} was not retrieved")

    first_token_times = [request_timestamp(output, "first_token_ts") for output in outputs]
    finished_times = [request_timestamp(output, "last_token_ts") for output in outputs]
    timestamps_available = all(value is not None for value in first_token_times + finished_times)
    decode_window_s = None
    decode_tok_per_s = None
    if timestamps_available:
        decode_window_s = max(finished_times) - min(first_token_times)  # type: ignore[arg-type]
        if decode_window_s <= 0:
            raise RuntimeError(f"invalid vLLM decode window: {decode_window_s}")
        # The first token occurs at the start of this window, so count only subsequent tokens.
        decode_tok_per_s = sum(length - 1 for length in decode_lengths) / decode_window_s

    output_tokens = sum(decode_lengths)
    record = {
        "batch_size": batch_size,
        "captured_batch_size_expected": next(
            size for size in (2, 4, 8, 16, 24, 32, 40, 48, 56, 64) if size >= batch_size
        ),
        "prompt_token_counts": prompt_lengths,
        "decode_token_counts": decode_lengths,
        "generation_wall_s": generation_wall_s,
        "generation_output_tok_per_s": output_tokens / generation_wall_s,
        "decode_window_s": decode_window_s,
        "decode_tok_per_s": decode_tok_per_s,
        "request_timestamps_available": timestamps_available,
        "needle_item": needle_index,
        "needle_code": int(wanted_code),
        "needle_retrieved": needle_retrieved,
        "completion_prefixes": [output.outputs[0].text[:120] for output in outputs],
    }
    print(
        "[decode-bench] "
        f"batch={batch_size} padded={record['captured_batch_size_expected']} "
        f"wall={generation_wall_s:.6f}s output={record['generation_output_tok_per_s']:.3f} tok/s "
        f"decode={decode_tok_per_s if decode_tok_per_s is not None else 'unavailable'} tok/s"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("EAGER", "FULL_DECODE_ONLY"), required=True)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("3,9,33"))
    parser.add_argument("--prompt-tokens", type=int, default=1600)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.80)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    from vllm import LLM

    eager = args.mode == "EAGER"
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        block_size=64,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        enforce_eager=eager,
        **(
            {}
            if eager
            else {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}}
        ),
        enable_prefix_caching=False,
        # Offline LLM otherwise defaults this to True and omits RequestOutput.metrics, which makes
        # first_token_ts/last_token_ts unavailable even though the engine records them internally.
        disable_log_stats=False,
        seed=1234,
        worker_extension_cls=smoke.telemetry_worker_extension(args.model),
    )

    # Do not bill first-inference initialization to either mode.
    run_batch(llm, batch_size=2, prompt_tokens=0, max_tokens=8)
    smoke.reset_selector_telemetry(llm, kind=smoke.telemetry_kind(args.model))

    samples = []
    for repeat in range(args.repeats):
        for batch_size in args.batch_sizes:
            record = run_batch(
                llm,
                batch_size=batch_size,
                prompt_tokens=args.prompt_tokens,
                max_tokens=args.max_tokens,
            )
            record["repeat"] = repeat
            samples.append(record)

    artifact = {
        "mode": args.mode,
        "model": args.model,
        "batch_sizes": args.batch_sizes,
        "prompt_tokens_requested": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "samples": samples,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"[decode-bench] wrote {args.out_json}")
    print("[decode-bench] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
