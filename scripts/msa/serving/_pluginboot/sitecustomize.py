# Auto-imported at EVERY interpreter startup (this dir goes FIRST on PYTHONPATH).
#
# Why this file exists (docs/qwen3_4b_msa/serving_plan.md §7): on the async server path vLLM
# spawns EngineCore as a SUBPROCESS, which does not inherit the parent's imports. Registering
# Qwen3MSAForCausalLM in the entry script alone therefore leaves the child unable to resolve the
# architecture. `sitecustomize` is imported automatically by every Python interpreter that has
# this directory on its path, so the registration is present in the parent AND the child.
#
# Proven pattern, copied from the DSA plugin:
#   /cb/ml-eng/aarti/dsa/evals/minicpm3-4B-dsa-k128/serving/_pluginboot/sitecustomize.py
import sys

try:
    import scripts.msa.vllm_qwen3_msa  # noqa: F401  (registers Qwen3MSAForCausalLM)

    sys.stderr.write("[sitecustomize] Qwen3-MSA plugin registered\n")
except Exception as e:  # never break an unrelated interpreter
    sys.stderr.write(f"[sitecustomize] Qwen3-MSA plugin import FAILED: {e!r}\n")
