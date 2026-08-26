# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serve the isolated Qwen3-4B DSA approximate-selector plugin.

Importing the plugin FIRST registers ``Qwen3DSAApproxForCausalLM`` (and the CUSTOM sparse attention
backend) before vLLM builds the engine. That covers THIS process; the EngineCore subprocess is
covered by ``_pluginboot_approx/sitecustomize.py`` being first on ``PYTHONPATH``.
"""

import asyncio
import json
import os
import pathlib
import runpy

# How often the periodic exporter refreshes the on-disk artifact. Short enough that a hard kill
# loses little, long enough that the RPC is not a load on the serving path.
EXPORT_INTERVAL_SECONDS = float(os.environ.get("DSA_SELECTOR_EXPORT_INTERVAL", "30"))

import scripts.dsa.vllm_qwen3_dsa_approx  # noqa: F401
from scripts.dsa.vllm_qwen3_dsa_approx.selector_hooks import install_hooks


def _telemetry_expected() -> bool:
    """Whether the served directory should produce a selector artifact at all.

    "No worker reset anything" is ambiguous: legitimate for an exact serving dir, but for an
    approximate one with telemetry on it means the plugin never loaded in the worker.
    """

    import sys

    for index, argument in enumerate(sys.argv):
        model = None
        if argument == "--model" and index + 1 < len(sys.argv):
            model = sys.argv[index + 1]
        elif argument.startswith("--model="):
            model = argument.split("=", 1)[1]
        if model is None:
            continue
        try:
            with open(os.path.join(model, "config.json")) as handle:
                config = json.load(handle)
        except (OSError, ValueError):
            return False
        architectures = config.get("architectures") or []
        if not any("Approx" in str(name) for name in architectures):
            return False
        return str(config.get("dsa_telemetry", "off")) != "off"
    return False


def _wire_selector_telemetry() -> None:
    """Drive the selector telemetry lifecycle automatically. No manual calls, no reset route.

    Telemetry accumulates in the EngineCore SUBPROCESS, which vLLM terminates rather than letting it
    exit, so the ``atexit`` dump registered by ``SelectorRuntime.configure`` never runs. Asking the
    live engine is the fix (as dsa-csx does via ``collective_rpc``) -- but an endpoint nobody calls
    is no better than an atexit that never fires, and a *reset* endpoint is worse: it is
    state-destructive and the launcher configures no API key, so any client reaching the port could
    erase safety observations mid-evaluation.

    So the lifecycle is bound to the server's own, not exposed:

    * **reset** runs once here, before uvicorn accepts a connection. ``serve_http`` is called after
      ``build_async_engine_client`` has completed engine init, so warmup and autotune -- whose
      synthetic batches would otherwise be attributed to served traffic -- are already done.
    * **export** runs in a ``finally`` after uvicorn stops. ``serve_http`` is invoked inside the
      engine client's ``async with``, so the engine is still alive at that point; waiting for
      process exit would be too late.

    ``GET /selector/telemetry`` remains for reading the artifact mid-run. It is read-only, so it
    cannot destroy observations, and it is the only route added.

    The seam is ``serve_http`` because vLLM's endpoint-plugin hook needs a packaged distribution
    plus a ``VLLM_PLUGINS`` allowlist. ``api_server`` does
    ``from vllm.entrypoints.launcher import serve_http``, a binding resolved when ``runpy``
    re-executes the module -- after this patch lands -- so the wrapper is picked up even though
    ``runpy`` rebuilds the module's own namespace.
    """

    import vllm.entrypoints.launcher as launcher

    original_serve_http = launcher.serve_http

    async def _artifact(engine) -> dict | None:
        payloads = await engine.collective_rpc("selector_telemetry_artifact")
        live = [item for item in payloads if item]
        if not live:
            return None
        return live[0] if len(live) == 1 else {"ranks": live}

    def _write(body: dict) -> str | None:
        path = os.environ.get("DSA_SELECTOR_ARTIFACT")
        if not path:
            return None
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(body, indent=2, default=str) + "\n")
        temporary.replace(target)
        return str(target)

    async def _export_periodically(engine) -> None:
        """Snapshot the artifact to disk on an interval, so no death mode loses it."""

        while True:
            await asyncio.sleep(EXPORT_INTERVAL_SECONDS)
            try:
                body = await _artifact(engine)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dead engine ends the loop, quietly
                print(
                    f"[serve_dsa_approx] selector telemetry export stopped: {exc!r}",
                    flush=True,
                )
                return
            if body is not None:
                _write(body)

    async def serve_http(app, *args, **kwargs):  # type: ignore[no-untyped-def]
        @app.get("/selector/telemetry")
        async def selector_telemetry():
            """Read the artifact mid-run, and refresh the on-disk copy. Read-only."""

            body = await _artifact(app.state.engine_client)
            if body is None:
                return {"available": False, "reason": "approximate selector plugin not active"}
            return {"available": True, "written": _write(body), "artifact": body}

        engine = app.state.engine_client
        # FATAL, not a warning. If the reset does not land, vLLM's synthetic warmup batches stay in
        # the accumulators and get attributed to served traffic -- the artifact would then be
        # evidence of nothing while still presenting as a safety verdict. Refusing to serve is the
        # correct outcome for a server whose purpose is producing that evidence.
        done = await engine.collective_rpc("selector_telemetry_reset")
        if any(done):
            print(
                f"[serve_dsa_approx] selector telemetry reset on {sum(map(bool, done))} "
                f"worker(s) after engine init; served traffic only from here",
                flush=True,
            )
        elif _telemetry_expected():
            raise RuntimeError(
                "no worker reset selector telemetry, but this serving directory is approximate "
                "with dsa_telemetry enabled. The plugin is not active in the worker (check the "
                "_pluginboot_approx PYTHONPATH); warmup observations would be attributed to "
                "served traffic."
            )

        # Export PERIODICALLY, not at shutdown. Measured 2026-08-26: exporting after
        # `original_serve_http` returns fails with `EngineDeadError` -- SIGTERM reaches the whole
        # process group, so EngineCore dies BEFORE uvicorn finishes its graceful shutdown, and no
        # in-process hook downstream of the signal can still reach the engine. Depending on
        # shutdown ordering is what made the original atexit dump useless; a periodic snapshot does
        # not depend on it at all, so whatever kills the server, a recent artifact is already on
        # disk. The evaluation driver should still GET /selector/telemetry before stopping the
        # server when it wants an exact end-of-run cut.
        exporter = asyncio.create_task(_export_periodically(engine))
        try:
            return await original_serve_http(app, *args, **kwargs)
        finally:
            exporter.cancel()

    launcher.serve_http = serve_http


install_hooks()
_wire_selector_telemetry()

runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
