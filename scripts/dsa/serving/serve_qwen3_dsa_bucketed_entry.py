# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Serve the isolated modulo-bucket plugin and export its worker telemetry."""

import asyncio
import json
import os
import pathlib
import runpy
import sys

import scripts.dsa.vllm_qwen3_dsa_bucketed  # noqa: F401
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_hooks import install_hooks

EXPORT_INTERVAL_SECONDS = float(os.environ.get("DSA_BUCKET_EXPORT_INTERVAL", "30"))
# Host-folded exact telemetry is intentionally dormant during EngineCore's synthetic profiling and
# kernel autotune forwards. The reset RPC below both clears device counters and arms host telemetry
# immediately before real traffic begins.
os.environ.setdefault("DSA_BUCKET_DEFER_HOST_TELEMETRY", "1")


def _model_config() -> dict | None:
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
                return json.load(handle)
        except (OSError, ValueError):
            return None
    return None


def _telemetry_expected() -> bool:
    config = _model_config()
    if config is None:
        return False
    architectures = config.get("architectures") or []
    return any("Bucketed" in str(name) for name in architectures) and str(
        config.get("dsa_bucket_telemetry", "off")
    ) != "off"


def _wire_bucket_telemetry() -> None:
    """Reset after warmup, periodically export, and expose one read-only endpoint."""

    import vllm.entrypoints.launcher as launcher

    original_serve_http = launcher.serve_http

    async def _artifact(engine) -> dict | None:
        payloads = await engine.collective_rpc("bucket_telemetry_artifact")
        live = [item for item in payloads if item]
        if not live:
            return None
        return live[0] if len(live) == 1 else {"ranks": live}

    def _write(body: dict) -> str | None:
        path = os.environ.get("DSA_BUCKET_TELEMETRY_ARTIFACT")
        if not path:
            return None
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(body, indent=2, default=str) + "\n")
        temporary.replace(target)
        return str(target)

    async def _export_periodically(engine) -> None:
        while True:
            await asyncio.sleep(EXPORT_INTERVAL_SECONDS)
            try:
                body = await _artifact(engine)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a dead engine ends export
                print(f"[serve_dsa_bucketed] telemetry export stopped: {exc!r}", flush=True)
                return
            if body is not None:
                _write(body)

    async def serve_http(app, *args, **kwargs):  # type: ignore[no-untyped-def]
        @app.get("/bucket/telemetry")
        async def bucket_telemetry():
            body = await _artifact(app.state.engine_client)
            if body is None:
                return {"available": False, "reason": "bucket telemetry is not active"}
            return {"available": True, "written": _write(body), "artifact": body}

        engine = app.state.engine_client
        done = await engine.collective_rpc("bucket_telemetry_reset")
        if any(done):
            print(
                f"[serve_dsa_bucketed] telemetry reset on {sum(map(bool, done))} worker(s) "
                "after engine warmup",
                flush=True,
            )
        elif _telemetry_expected():
            raise RuntimeError(
                "bucket telemetry is enabled in config, but no worker reset it. Check the "
                "bucket plugin boot path and BucketTelemetryExtension wiring."
            )

        exporter = asyncio.create_task(_export_periodically(engine)) if any(done) else None
        try:
            return await original_serve_http(app, *args, **kwargs)
        finally:
            if exporter is not None:
                exporter.cancel()

    launcher.serve_http = serve_http


install_hooks()
_wire_bucket_telemetry()
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
