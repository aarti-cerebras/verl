# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Serve the isolated modulo-bucket Qwen3 DSA plugin through vLLM's OpenAI API server."""

import runpy

import scripts.dsa.vllm_qwen3_dsa_bucketed  # noqa: F401
from scripts.dsa.vllm_qwen3_dsa_bucketed.bucket_selector_hooks import install_hooks

install_hooks()
runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
