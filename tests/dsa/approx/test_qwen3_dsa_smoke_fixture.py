import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def _load_smoke_module():
    # Prompt construction is pure Python. Stub plugin registration so this regression remains a
    # CPU-only test and does not require importing vLLM or initializing a CUDA runtime.
    name = "scripts.dsa.vllm_qwen3_dsa"
    original = sys.modules.get(name)
    sys.modules[name] = types.ModuleType(name)
    try:
        path = Path(__file__).parents[1] / "qwen3_dsa_offline_smoke.py"
        spec = importlib.util.spec_from_file_location("qwen3_dsa_offline_smoke_fixture", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def test_stage_gate_builds_seven_heterogeneous_long_requests() -> None:
    smoke = _load_smoke_module()
    prompts, sentences, needle = smoke.build_prompts(6000, 7)

    assert len(prompts) == 7
    assert len(set(map(len, prompts))) == 7
    assert len(sentences) > 7
    assert 0 <= needle < len(sentences)
    assert all("Q:" in prompt and prompt.endswith("\nA:") for prompt in prompts)


def test_default_smoke_fixture_retains_the_two_original_prompts() -> None:
    smoke = _load_smoke_module()
    prompts, _, _ = smoke.build_prompts(0, 2)

    assert len(prompts) == 2
    assert "17 * 23" in prompts[0]
    assert "Repeat the code word" in prompts[1]


def test_step_logprobs_are_serialized_without_vllm_types() -> None:
    smoke = _load_smoke_module()
    steps = [
        {
            382: SimpleNamespace(logprob=-0.25),
            271: SimpleNamespace(logprob=-0.50),
        },
        {13: SimpleNamespace(logprob=-0.125)},
    ]

    assert smoke.serialize_step_logprobs(steps) == [
        {"382": -0.25, "271": -0.5},
        {"13": -0.125},
    ]
    assert smoke.serialize_step_logprobs(None) == []
