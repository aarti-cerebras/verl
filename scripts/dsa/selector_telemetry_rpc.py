"""Worker-side RPC methods for reading approximate-selector telemetry.

Why this exists
---------------
The telemetry accumulates inside vLLM's EngineCore *worker*, which is a subprocess that vLLM
terminates rather than exiting cleanly. The ``atexit`` dump registered by
``SelectorRuntime.configure`` therefore never runs, and the artifact is never written -- so the
runner's hard-violation gate reads a missing file, finds no violations, and reports a safety verdict
it never computed.

The fix is to ask the live worker instead of relying on shutdown ordering, which is what dsa-csx
does (``glm_52/vllm_study/server_worker.py::selector_metadata``, reached via ``collective_rpc``).

Why a worker extension rather than a callable
---------------------------------------------
``collective_rpc`` accepts a callable, but vLLM 0.26 refuses to serialize arbitrary functions unless
``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` is set. The supported route is to name a method that exists
on the worker, which ``ParallelConfig.worker_extension_cls`` is built for: the class is dynamically
inherited by the worker class specifically "to inject new attributes and methods to the worker class
for use in collective_rpc calls".

Why sys.modules rather than an import
-------------------------------------
``import scripts.dsa.vllm_qwen3_dsa_approx`` REGISTERS the approximate architecture as a side
effect. This extension is attached to every arm, including the ones running the exact plugin, so it
must never cause that registration. Looking the module up in ``sys.modules`` observes the selector
only when the approximate plugin is genuinely the loaded one, and no-ops otherwise.
"""

import sys
from typing import Any

_RUNTIME_MODULE = "scripts.dsa.vllm_qwen3_dsa_approx.selector_runtime"


def _runtime() -> Any | None:
    """The live SelectorRuntime, or None when the approximate plugin is not loaded."""

    module = sys.modules.get(_RUNTIME_MODULE)
    if module is None:
        return None
    runtime = getattr(module, "RUNTIME", None)
    if runtime is None or runtime.config is None:
        return None
    return runtime


class SelectorTelemetryExtension:
    """Mixed into the vLLM worker via ``worker_extension_cls``."""

    def selector_telemetry_reset(self) -> bool:
        """Discard telemetry accumulated over vLLM's synthetic warmup/autotune batches.

        Those batches carry dummy operands, so their selections are meaningless; folded into the
        artifact they would be attributed to served traffic and could either trip the
        hard-violation gate or dilute a real violation below notice.
        """

        runtime = _runtime()
        if runtime is None:
            return False
        runtime.reset()
        return True

    def selector_telemetry_artifact(self) -> dict[str, Any] | None:
        """Return the telemetry artifact as a plain dict, or None if not applicable."""

        runtime = _runtime()
        if runtime is None:
            return None
        return runtime.artifact()
