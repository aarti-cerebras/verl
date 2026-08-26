# Auto-imported at EVERY interpreter startup (this dir goes FIRST on PYTHONPATH).
#
# Why this file exists (docs/qwen3_4b_dsa/serving_eval_plan.md §2): on the async server path vLLM
# spawns EngineCore as a SUBPROCESS, which does not inherit the parent's imports. Registering
# Qwen3DSAApproxForCausalLM in the entry script alone therefore leaves the child unable to resolve the
# architecture -- and, worse, unable to resolve the CUSTOM attention backend, whose registration
# lives in the same package.
#
# Proven pattern, shared with scripts/msa/serving/_pluginboot/sitecustomize.py.
import sys

try:
    import scripts.dsa.vllm_qwen3_dsa_approx  # noqa: F401
    from scripts.dsa.vllm_qwen3_dsa_approx.selector_hooks import install_hooks

    install_hooks()

    sys.stderr.write("[sitecustomize] isolated Qwen3-DSA-approx plugin registered\n")
except Exception as e:
    sys.stderr.write(f"[sitecustomize] Qwen3-DSA-approx plugin import FAILED: {e!r}\n")
    raise
