import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.dsa.vllm_qwen3_dsa.indexer import Qwen3DSAServingIndexer
from scripts.dsa.vllm_qwen3_dsa_approx.indexer import Qwen3DSAApproxServingIndexer


REPO = Path(__file__).resolve().parents[3]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_metadata_free_exact_topk_snapshot_parity() -> None:
    torch.manual_seed(23)
    exact = Qwen3DSAServingIndexer(
        hidden_size=64,
        n_heads=2,
        head_dim=32,
        rope_head_dim=32,
        top_k=4,
        fp8=False,
        rotate_activation=False,
    )
    approximate_copy = Qwen3DSAApproxServingIndexer(
        hidden_size=64,
        n_heads=2,
        head_dim=32,
        rope_head_dim=32,
        top_k=4,
        capacity=4,
        fp8=False,
        rotate_activation=False,
    )
    approximate_copy.load_state_dict(exact.state_dict(), strict=True)
    hidden = torch.randn(12, 64)
    positions = torch.arange(12)
    exact_scores = exact.torch_scores(hidden, positions)
    copied_scores = approximate_copy.torch_scores(hidden, positions)
    torch.testing.assert_close(copied_scores, exact_scores, rtol=0, atol=0)
    assert torch.equal(
        approximate_copy.select_topk(copied_scores),
        exact.select_topk(exact_scores),
    )


def test_derived_serving_dir_does_not_modify_exact_source(tmp_path: Path) -> None:
    source = tmp_path / "exact"
    output = tmp_path / "approx"
    source.mkdir()
    config = {
        "architectures": ["Qwen3DSAForCausalLM"],
        "dsa_enabled": True,
        "dsa_mode": "sparse",
        "dsa_top_k": 2048,
        "index_topk": 2048,
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"shared-weight-fixture")
    before = {path.name: _digest(path) for path in source.iterdir()}

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/dsa/build_qwen3_dsa_approx_serving_dir.py"),
            "--source",
            str(source),
            "--out",
            str(output),
            "--selector",
            "radix_floor",
            "--selector-margin",
            "256",
            "--telemetry",
            "verify_exact",
        ],
        check=True,
    )

    assert before == {path.name: _digest(path) for path in source.iterdir()}
    derived = json.loads((output / "config.json").read_text())
    assert derived["architectures"] == ["Qwen3DSAApproxForCausalLM"]
    assert derived["dsa_top_k"] == 2048
    assert derived["index_topk"] == 2304
    assert derived["dsa_selector_backend"] == "dsa_csx_reference"
    assert (output / "model.safetensors").is_symlink()
    assert (output / "model.safetensors").resolve() == source / "model.safetensors"


def test_protected_exact_snapshot_has_no_approx_imports() -> None:
    protected = [
        REPO / "scripts/dsa/vllm_qwen3_dsa/__init__.py",
        REPO / "scripts/dsa/serving/_pluginboot/sitecustomize.py",
        REPO / "scripts/dsa/serving/serve_qwen3_dsa_entry.py",
    ]
    for path in protected:
        assert "vllm_qwen3_dsa_approx" not in path.read_text()
