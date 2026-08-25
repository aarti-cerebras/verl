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
"""Serve Qwen3-4B DSA through vLLM's OpenAI API server.

Importing the plugin FIRST registers ``Qwen3DSAForCausalLM`` (and the CUSTOM sparse attention
backend) before vLLM builds the engine. That covers THIS process; the EngineCore subprocess is
covered by ``_pluginboot/sitecustomize.py`` being first on ``PYTHONPATH``.
"""

import runpy

import scripts.dsa.vllm_qwen3_dsa  # noqa: F401  (registers Qwen3DSAForCausalLM)

runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__", alter_sys=True)
