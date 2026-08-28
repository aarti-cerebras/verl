import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "qwen3_dsa_approx_validation_summary.py"
    spec = importlib.util.spec_from_file_location("qwen3_dsa_approx_validation_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _output(tokens, logprobs):
    return {
        "outputs": {
            "0": {
                "token_ids": tokens,
                "step_logprobs": logprobs,
            }
        }
    }


def test_token_mismatch_is_diagnostic_when_centered_logprobs_are_close() -> None:
    summary = _module()
    left = _output(
        [382, 10],
        [
            {"382": -0.9, "271": -1.0, "13": -2.0},
            {"10": -0.1, "11": -1.1},
        ],
    )
    right = _output(
        [271, 12],
        [
            {"382": -0.95, "271": -0.95, "13": -2.05},
            {"12": -0.2, "11": -1.2},
        ],
    )

    result = summary.compare_outputs(left, right, mean_tolerance=0.20, max_tolerance=1.0)

    assert result["token_exact"] is False
    assert result["first_differences"] == {"0": 0}
    assert result["comparable_logprob_steps"] == 1
    assert result["within_tolerance"] is True


def test_large_centered_logprob_delta_fails() -> None:
    summary = _module()
    left = _output([1], [{"1": -0.1, "2": -0.2}])
    right = _output([1], [{"1": -0.1, "2": -2.2}])

    result = summary.compare_outputs(left, right, mean_tolerance=0.20, max_tolerance=1.0)

    assert result["token_exact"] is True
    assert result["within_tolerance"] is False


def test_concurrent_fixture_requires_needle_retrieval() -> None:
    summary = _module()
    payload = {
        "decode_batch_size": 3,
        "needle_retrieved": True,
        "outputs": {
            str(index): {"n_prompt_tok": 3000 + index, "token_ids": [1, 2]}
            for index in range(3)
        },
    }

    assert summary.concurrent_fixture(payload, batch_size=3, tokens=2)["passed"] is True
    payload["needle_retrieved"] = False
    assert summary.concurrent_fixture(payload, batch_size=3, tokens=2)["passed"] is False


def _dist(value: float) -> dict:
    return {"n": 10, "mean": value, "min": value, "max": value}


def _quality(selector: str, *, perturb: float = 0.0, violate: bool = False):
    fields = ["selected_count", "added", "dropped", "exact_recall", "precision"]
    artifact = {
        "meta": {"selector": {"telemetry": "graph_verify_exact"}},
        "graph_replay_quality": {"fields": fields},
    }
    for phase in ("prefill", "decode"):
        metrics = {
            "selected_count": _dist(2048.0 + perturb),
            "added": _dist(1.0 if selector == "ceil" and violate else 0.0),
            "dropped": _dist(1.0 if selector == "floor" and violate else 0.0),
            "exact_recall": _dist(0.9 if selector == "floor" and violate else 1.0),
            "precision": _dist(0.9 if selector == "ceil" and violate else 1.0),
        }
        artifact[phase] = {"global": metrics}
    return artifact


def test_quality_gate_uses_tolerance_but_keeps_semantics_exact() -> None:
    summary = _module()
    eager = _quality("floor")
    graph = _quality("floor", perturb=0.5)

    result = summary.quality_comparison(
        "floor", eager, graph, relative_tolerance=0.01, absolute_tolerance=0.001
    )

    assert result["means_within_tolerance"] is True
    assert result["semantic_invariants"] is True

    broken = summary.quality_comparison(
        "floor",
        eager,
        _quality("floor", violate=True),
        relative_tolerance=0.01,
        absolute_tolerance=0.001,
    )
    assert broken["semantic_invariants"] is False
