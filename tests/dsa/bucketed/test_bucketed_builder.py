import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from scripts.dsa.vllm_qwen3_dsa.indexer import Qwen3DSAServingIndexer
from scripts.dsa.vllm_qwen3_dsa_bucketed.indexer import Qwen3DSABucketedServingIndexer

REPO = Path(__file__).resolve().parents[3]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bucketed_indexer_snapshot_matches_exact_scores() -> None:
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
    bucketed = Qwen3DSABucketedServingIndexer(
        hidden_size=64,
        n_heads=2,
        head_dim=32,
        rope_head_dim=32,
        top_k=4,
        fp8=False,
        rotate_activation=False,
    )
    bucketed.load_state_dict(exact.state_dict(), strict=True)
    hidden = torch.randn(12, 64)
    positions = torch.arange(12)
    torch.testing.assert_close(
        bucketed.torch_scores(hidden, positions),
        exact.torch_scores(hidden, positions),
        rtol=0,
        atol=0,
    )


def test_builder_derives_isolated_fixed_budget_directory(tmp_path: Path) -> None:
    source = tmp_path / "exact"
    output = tmp_path / "bucketed"
    source.mkdir()
    config = {
        "architectures": ["Qwen3DSAForCausalLM"],
        "dsa_enabled": True,
        "dsa_mode": "sparse",
        "dsa_top_k": 16,
        "index_topk": 16,
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").write_bytes(b"shared-weight-fixture")
    before = {path.name: _digest(path) for path in source.iterdir()}

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/dsa/build_qwen3_dsa_bucketed_serving_dir.py"),
            "--source",
            str(source),
            "--out",
            str(output),
            "--bucket-count",
            "4",
        ],
        check=True,
    )

    assert before == {path.name: _digest(path) for path in source.iterdir()}
    derived = json.loads((output / "config.json").read_text())
    assert derived["architectures"] == ["Qwen3DSABucketedForCausalLM"]
    assert derived["dsa_selector"] == "modulo_bucket_topk"
    assert derived["dsa_selector_backend"] == "vllm_stock_per_bucket"
    assert derived["dsa_bucket_count"] == 4
    assert derived["dsa_bucket_top_k"] == 4
    assert derived["dsa_bucket_telemetry"] == "off"
    assert derived["index_topk"] == derived["dsa_top_k"] == 16
    assert (output / "model.safetensors").is_symlink()
    manifest = json.loads((output / "BUCKETED_BUILD_MANIFEST.json").read_text())
    assert manifest["exact_source_modified"] is False
    assert manifest["selector_speed_claim_valid"] is False
    assert manifest["cuda_graph_modes_supported"] == ["NONE", "FULL_DECODE_ONLY"]
    assert manifest["decode_cuda_graph_validated"] is False
    assert manifest["prefill_cuda_graph_validated"] is False
    assert manifest["cuda_graph_validation_evidence"] is None


def test_builder_records_eager_telemetry_and_restricts_graph_modes(tmp_path: Path) -> None:
    source = tmp_path / "exact"
    output = tmp_path / "bucketed"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSAForCausalLM"],
                "dsa_enabled": True,
                "dsa_mode": "sparse",
                "dsa_top_k": 16,
                "index_topk": 16,
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/dsa/build_qwen3_dsa_bucketed_serving_dir.py"),
            "--source",
            str(source),
            "--out",
            str(output),
            "--bucket-count",
            "4",
            "--telemetry",
            "verify_exact",
        ],
        check=True,
    )
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "BUCKETED_BUILD_MANIFEST.json").read_text())
    assert config["dsa_bucket_telemetry"] == "verify_exact"
    assert manifest["telemetry"] == "verify_exact"
    assert manifest["cuda_graph_modes_supported"] == ["NONE"]


def test_builder_records_only_the_gpu_validated_decode_graph_geometry(tmp_path: Path) -> None:
    source = tmp_path / "exact"
    output = tmp_path / "bucketed"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSAForCausalLM"],
                "dsa_enabled": True,
                "dsa_mode": "sparse",
                "dsa_top_k": 2048,
                "index_topk": 2048,
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/dsa/build_qwen3_dsa_bucketed_serving_dir.py"),
            "--source",
            str(source),
            "--out",
            str(output),
            "--bucket-count",
            "8",
            "--bucket-top-k",
            "256",
        ],
        check=True,
    )

    manifest = json.loads((output / "BUCKETED_BUILD_MANIFEST.json").read_text())
    assert manifest["decode_cuda_graph_validated"] is True
    assert manifest["prefill_cuda_graph_validated"] is False
    assert manifest["cuda_graph_validation_evidence"].endswith(
        "20260828T210745Z-qwen3-dsa-bucketed-decode-graph/result.md"
    )

    graph_output = tmp_path / "bucketed-graph-safety"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/dsa/build_qwen3_dsa_bucketed_serving_dir.py"),
            "--source",
            str(source),
            "--out",
            str(graph_output),
            "--bucket-count",
            "8",
            "--bucket-top-k",
            "256",
            "--telemetry",
            "graph_safety",
        ],
        check=True,
    )
    graph_manifest = json.loads((graph_output / "BUCKETED_BUILD_MANIFEST.json").read_text())
    assert graph_manifest["decode_cuda_graph_validated"] is True
    assert graph_manifest["cuda_graph_validation_evidence"].endswith(
        "20260828T232757Z-qwen3-dsa-bucket-graph-telemetry-fix/result.md"
    )


def test_builder_rejects_budget_change(tmp_path: Path) -> None:
    source = tmp_path / "exact"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3DSAForCausalLM"],
                "dsa_enabled": True,
                "dsa_mode": "sparse",
                "dsa_top_k": 16,
                "index_topk": 16,
            }
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/dsa/build_qwen3_dsa_bucketed_serving_dir.py"),
            "--source",
            str(source),
            "--out",
            str(tmp_path / "bucketed"),
            "--bucket-count",
            "3",
            "--bucket-top-k",
            "5",
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "fixed budget requires" in completed.stderr


def test_bucketed_plugin_does_not_import_radix_modules() -> None:
    bucketed = REPO / "scripts/dsa/vllm_qwen3_dsa_bucketed"
    text = "\n".join(path.read_text() for path in bucketed.glob("*.py"))
    assert "vllm_qwen3_dsa_approx" not in text
    assert "radix_rules" not in text
    assert "radix_selector_reference" not in text


def test_radix_plugin_does_not_import_bucketed_modules() -> None:
    approximate = REPO / "scripts/dsa/vllm_qwen3_dsa_approx"
    text = "\n".join(path.read_text() for path in approximate.glob("*.py"))
    assert "vllm_qwen3_dsa_bucketed" not in text
