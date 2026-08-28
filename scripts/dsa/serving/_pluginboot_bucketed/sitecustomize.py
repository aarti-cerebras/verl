"""Register the isolated bucketed architecture and selector hooks in vLLM subprocesses."""

import sys

try:
    import scripts.dsa.vllm_qwen3_dsa_bucketed  # noqa: F401
    from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_hooks import install_hooks

    install_hooks()
    sys.stderr.write("[sitecustomize] isolated Qwen3-DSA-bucketed plugin registered\n")
except Exception as exc:
    sys.stderr.write(f"[sitecustomize] Qwen3-DSA-bucketed plugin import FAILED: {exc!r}\n")
    raise
