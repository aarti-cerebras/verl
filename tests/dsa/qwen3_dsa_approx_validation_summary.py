#!/usr/bin/env python3
"""Summarize and gate the concurrent Qwen3 DSA approximate-selector GPU fixture.

Cross-process greedy token equality is diagnostic: fresh exact-plugin processes can disagree at a
zero-margin decision despite identical inputs and seeds. Hard numerical comparison therefore uses
centered top-token logprobs only while the generated prefixes are identical. Fixed-input selector
and CUDA-replay tests remain the deterministic equality gates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


COMPARISONS = {
    "exact_vs_approx_topk_eager": ("exact_eager", "approx_topk_eager"),
    "approx_topk_eager_vs_graph": ("approx_topk_eager", "approx_topk_graph"),
    "ceil_eager_vs_graph": ("ceil_eager", "ceil_graph"),
    "floor_eager_vs_graph": ("floor_eager", "floor_graph"),
}
LABELS = (
    "exact_eager",
    "approx_topk_eager",
    "approx_topk_graph",
    "ceil_eager",
    "ceil_graph",
    "midpoint_eager",
    "floor_eager",
    "floor_graph",
)
TELEMETRY_LABELS = ("ceil_eager", "ceil_graph", "midpoint_eager", "floor_eager", "floor_graph")


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def compare_outputs(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    mean_tolerance: float,
    max_tolerance: float,
) -> dict[str, Any]:
    """Compare logits on identical generated prefixes; report greedy equality diagnostically."""

    if left is None or right is None:
        return {"available": False, "within_tolerance": False}
    left_outputs, right_outputs = left.get("outputs", {}), right.get("outputs", {})
    if set(left_outputs) != set(right_outputs) or not left_outputs:
        return {"available": False, "within_tolerance": False}

    token_matches = 0
    token_positions = 0
    first_differences: dict[str, int | None] = {}
    comparable_steps = 0
    relative_deltas: list[float] = []
    logprobs_complete = True

    for request in sorted(left_outputs, key=int):
        a, b = left_outputs[request], right_outputs[request]
        at, bt = list(a.get("token_ids", ())), list(b.get("token_ids", ()))
        width = min(len(at), len(bt))
        token_matches += sum(x == y for x, y in zip(at[:width], bt[:width]))
        token_positions += max(len(at), len(bt))
        first_differences[request] = next(
            (position for position, (x, y) in enumerate(zip(at, bt)) if x != y),
            None if len(at) == len(bt) else width,
        )

        al, bl = a.get("step_logprobs"), b.get("step_logprobs")
        if al is None or bl is None or len(al) != len(at) or len(bl) != len(bt):
            logprobs_complete = False
            continue
        for position in range(min(len(al), len(bl))):
            # Step p predicts token p from generated tokens [:p]. Once those prefixes differ, the
            # two distributions no longer describe the same model input and cannot be compared.
            if at[:position] != bt[:position]:
                break
            ax = {int(token): float(value) for token, value in al[position].items()}
            bx = {int(token): float(value) for token, value in bl[position].items()}
            shared = set(ax) & set(bx)
            if len(shared) < 2:
                continue
            # Logprobs may differ by an additive normalization constant. Center each distribution
            # on its best shared token so the comparison measures relative logits instead.
            ac = max(ax[token] for token in shared)
            bc = max(bx[token] for token in shared)
            relative_deltas.extend(
                abs((ax[token] - ac) - (bx[token] - bc)) for token in shared
            )
            comparable_steps += 1

    mean_delta = sum(relative_deltas) / len(relative_deltas) if relative_deltas else None
    max_delta = max(relative_deltas) if relative_deltas else None
    minimum_steps = len(left_outputs)
    within_tolerance = bool(
        logprobs_complete
        and comparable_steps >= minimum_steps
        and mean_delta is not None
        and max_delta is not None
        and mean_delta <= mean_tolerance
        and max_delta <= max_tolerance
    )
    return {
        "available": True,
        # Diagnostic only. Exact-plugin replicate r3 proves this is not a deterministic hard gate.
        "token_exact": all(
            left_outputs[key].get("token_ids") == right_outputs[key].get("token_ids")
            for key in left_outputs
        ),
        "token_agreement": token_matches / token_positions if token_positions else None,
        "token_matches": token_matches,
        "token_positions": token_positions,
        "first_differences": first_differences,
        "logprobs_complete": logprobs_complete,
        "comparable_logprob_steps": comparable_steps,
        "shared_logprob_values": len(relative_deltas),
        "mean_centered_logprob_delta": mean_delta,
        "max_centered_logprob_delta": max_delta,
        "mean_tolerance": mean_tolerance,
        "max_tolerance": max_tolerance,
        "within_tolerance": within_tolerance,
    }


def concurrent_fixture(payload: dict[str, Any] | None, *, batch_size: int, tokens: int) -> dict[str, Any]:
    if payload is None:
        return {"available": False, "passed": False}
    outputs = payload.get("outputs", {})
    prompt_lengths = [item.get("n_prompt_tok", 0) for item in outputs.values()]
    completion_lengths = [len(item.get("token_ids", ())) for item in outputs.values()]
    passed = bool(
        payload.get("decode_batch_size") == batch_size
        and len(outputs) == batch_size
        and len(set(prompt_lengths)) >= 3
        and min(prompt_lengths, default=0) > 2048
        and all(length == tokens for length in completion_lengths)
        and payload.get("needle_retrieved") is True
    )
    return {
        "available": True,
        "passed": passed,
        "decode_batch_size": payload.get("decode_batch_size"),
        "output_count": len(outputs),
        "distinct_prompt_lengths": len(set(prompt_lengths)),
        "min_prompt_tokens": min(prompt_lengths, default=0),
        "max_prompt_tokens": max(prompt_lengths, default=0),
        "forced_decode_tokens": tokens,
        "min_completion_tokens": min(completion_lengths, default=0),
        "max_completion_tokens": max(completion_lengths, default=0),
        "needle_retrieved": payload.get("needle_retrieved"),
    }


def telemetry_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if artifact is None:
        return {"available": False, "rows": 0, "hard_violation_counts": {"missing": 1}}
    safety = artifact.get("safety", {})
    counts = safety.get("counts", {})
    return {
        "available": True,
        "rows": safety.get("rows", 0),
        "hard_violation_counts": {key: value for key, value in counts.items() if value},
        "capacity_saturation": counts.get("capacity_saturation", 0),
        "rescued": counts.get("rescued", 0),
        "hook_symbols": artifact.get("provenance", {}).get("hooks", {}).get("symbols", []),
    }


def _constant(dist: dict[str, Any] | None, target: float) -> bool:
    if not dist or int(dist.get("n", 0)) <= 0:
        return False
    return all(
        math.isclose(float(dist.get(field, math.nan)), target, rel_tol=0.0, abs_tol=1e-7)
        for field in ("mean", "min", "max")
    )


def quality_comparison(
    selector: str,
    eager: dict[str, Any] | None,
    graph: dict[str, Any] | None,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    if eager is None or graph is None:
        return {"available": False, "means_within_tolerance": False, "semantic_invariants": False}
    if graph.get("meta", {}).get("selector", {}).get("telemetry") != "graph_verify_exact":
        return {"available": False, "means_within_tolerance": False, "semantic_invariants": False}
    fields = graph.get("graph_replay_quality", {}).get("fields", [])
    if not fields:
        return {"available": False, "means_within_tolerance": False, "semantic_invariants": False}

    results: dict[str, dict[str, bool]] = {}
    all_close = True
    semantics: dict[str, bool] = {}
    for phase in ("prefill", "decode"):
        phase_result: dict[str, bool] = {}
        eg = eager.get(phase, {}).get("global", {})
        gg = graph.get(phase, {}).get("global", {})
        for field in fields:
            a, b = eg.get(field), gg.get(field)
            close = bool(
                a
                and b
                and a.get("n") == b.get("n")
                and math.isclose(
                    float(a.get("mean", math.nan)),
                    float(b.get("mean", math.nan)),
                    rel_tol=relative_tolerance,
                    abs_tol=absolute_tolerance,
                )
            )
            phase_result[field] = close
            all_close &= close
        results[phase] = phase_result

        for mode, artifact in (("eager", eager), ("graph", graph)):
            global_metrics = artifact.get(phase, {}).get("global", {})
            if selector == "ceil":
                semantics[f"{mode}_{phase}_added_zero"] = _constant(
                    global_metrics.get("added"), 0.0
                )
                semantics[f"{mode}_{phase}_precision_one"] = _constant(
                    global_metrics.get("precision"), 1.0
                )
            else:
                semantics[f"{mode}_{phase}_dropped_zero"] = _constant(
                    global_metrics.get("dropped"), 0.0
                )
                semantics[f"{mode}_{phase}_exact_recall_one"] = _constant(
                    global_metrics.get("exact_recall"), 1.0
                )
    return {
        "available": True,
        "means_within_tolerance": all_close,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "fields": results,
        "semantic_invariants": bool(semantics) and all(semantics.values()),
        "semantic_checks": semantics,
    }


def build_summary(
    root: Path,
    *,
    batch_size: int,
    tokens: int,
    logprob_mean_tolerance: float,
    logprob_max_tolerance: float,
    quality_relative_tolerance: float,
    quality_absolute_tolerance: float,
) -> dict[str, Any]:
    results = {label: _read(root / "results" / f"{label}.json") for label in LABELS}
    telemetry = {label: _read(root / "telemetry" / f"{label}.json") for label in TELEMETRY_LABELS}
    summary: dict[str, Any] = {}
    for name, (left, right) in COMPARISONS.items():
        summary[name] = compare_outputs(
            results[left],
            results[right],
            mean_tolerance=logprob_mean_tolerance,
            max_tolerance=logprob_max_tolerance,
        )
    for label in LABELS:
        summary[f"concurrent_fixture_{label}"] = concurrent_fixture(
            results[label], batch_size=batch_size, tokens=tokens
        )
    for label in TELEMETRY_LABELS:
        summary[f"telemetry_{label}"] = telemetry_summary(telemetry[label])
    for selector in ("ceil", "floor"):
        summary[f"{selector}_eager_vs_graph_quality"] = quality_comparison(
            selector,
            telemetry[f"{selector}_eager"],
            telemetry[f"{selector}_graph"],
            relative_tolerance=quality_relative_tolerance,
            absolute_tolerance=quality_absolute_tolerance,
        )
    summary["hard_failure_reasons"] = hard_failure_reasons(summary)
    summary["passed"] = not summary["hard_failure_reasons"]
    return summary


def hard_failure_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for name in COMPARISONS:
        item = summary.get(name, {})
        if not item.get("within_tolerance", False):
            reasons.append(f"{name}: centered logprobs unavailable or outside tolerance")
    for label in LABELS:
        if not summary.get(f"concurrent_fixture_{label}", {}).get("passed", False):
            reasons.append(f"concurrent_fixture_{label}: fixture or needle gate failed")
    for label in TELEMETRY_LABELS:
        item = summary.get(f"telemetry_{label}", {})
        if not item.get("available") or int(item.get("rows", 0)) <= 0:
            reasons.append(f"telemetry_{label}: missing or empty")
        if item.get("hard_violation_counts"):
            reasons.append(f"telemetry_{label}: safety violations present")
    for selector in ("ceil", "floor"):
        eager = summary.get(f"telemetry_{selector}_eager", {})
        graph = summary.get(f"telemetry_{selector}_graph", {})
        if graph.get("rows") != eager.get("rows"):
            reasons.append(f"{selector}: eager/graph safety row counts differ")
        quality = summary.get(f"{selector}_eager_vs_graph_quality", {})
        if not quality.get("means_within_tolerance", False):
            reasons.append(f"{selector}: eager/graph quality means outside tolerance")
        if not quality.get("semantic_invariants", False):
            reasons.append(f"{selector}: containment/subset semantic invariant failed")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--logprob-mean-tolerance", type=float, default=0.20)
    parser.add_argument("--logprob-max-tolerance", type=float, default=1.0)
    parser.add_argument("--quality-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--quality-absolute-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    summary = build_summary(
        args.root,
        batch_size=args.batch_size,
        tokens=args.tokens,
        logprob_mean_tolerance=args.logprob_mean_tolerance,
        logprob_max_tolerance=args.logprob_max_tolerance,
        quality_relative_tolerance=args.quality_relative_tolerance,
        quality_absolute_tolerance=args.quality_absolute_tolerance,
    )
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
