"""Worker RPC surface for isolated modulo-bucket selector telemetry.

The module deliberately observes ``sys.modules`` instead of importing the bucket plugin. Attaching
the extension to a worker therefore cannot register or activate any selector as a side effect.
The separate module and method names also prevent bucket artifacts from being mistaken for radix
ceil/floor telemetry.
"""

import sys
from typing import Any

_RUNTIME_MODULE = "scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_runtime"


def _runtime() -> Any | None:
    module = sys.modules.get(_RUNTIME_MODULE)
    if module is None:
        return None
    runtime = getattr(module, "RUNTIME", None)
    config = getattr(runtime, "config", None)
    if runtime is None or config is None or config.telemetry == "off":
        return None
    return runtime


class BucketTelemetryExtension:
    """Mixed into the vLLM worker through ``worker_extension_cls``."""

    def bucket_telemetry_reset(self) -> bool:
        runtime = _runtime()
        if runtime is None:
            return False
        runtime.reset()
        return True

    def bucket_telemetry_artifact(self) -> dict[str, Any] | None:
        runtime = _runtime()
        if runtime is None:
            return None
        return runtime.artifact()
