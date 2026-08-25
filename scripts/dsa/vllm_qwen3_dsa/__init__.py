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
"""Out-of-tree vLLM plugin: Qwen3-4B + token-granular DSA (GQA, no MLA).

``import scripts.dsa.vllm_qwen3_dsa`` registers the architecture. Registration is by STRING so that
importing this package does not drag in vLLM's model layer (and so the EngineCore subprocess, which
does not inherit the parent's imports, can register it via ``serving/_pluginboot``).
"""

from vllm import ModelRegistry

ARCH = "Qwen3DSAForCausalLM"


def register() -> None:
    """Idempotent registration of ``Qwen3DSAForCausalLM``."""
    if ARCH in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(ARCH, "scripts.dsa.vllm_qwen3_dsa.model:Qwen3DSAForCausalLM")


register()
