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
"""vLLM plugin: ``Qwen3MSAForCausalLM`` — Qwen3-4B with MiniMax Sparse Attention.

P3 of docs/qwen3_4b_msa/serving_plan.md. Registration only; everything else lives in
``model.py``.

Contrast with ``scripts/dsa/vllm_minicpm3_dsa/__init__.py`` (207 lines): that plugin also
had to install a ``deep_gemm`` meta-path shim and monkeypatch vLLM's DeepSeek-family MLA
allowlist, because vLLM gates the MLA path on ``model_type``. Neither is needed here — the
M3 sparse backend is bound directly by the attention module
(``self.attn_backend = MiniMaxM3SparseBackend``), and the kernels are Triton, not deep_gemm.
So this file is just the two ``register_model`` calls (serving_plan §4.1).

Import this package BEFORE constructing the vLLM engine. On the async server path that is not
sufficient on its own: vLLM spawns EngineCore as a subprocess which does NOT inherit the
parent's imports, so put ``serving/_pluginboot`` first on ``PYTHONPATH`` as well
(serving_plan §7).
"""


def register() -> None:
    """Register ``Qwen3MSAForCausalLM`` with vLLM's ``ModelRegistry`` (idempotent).

    Uses the lazy ``"<module>:<class>"`` string form so importing this package does not drag
    in vLLM's model layer at import time — vLLM imports the class only when it builds the
    model. Mirrors ``scripts/dsa/vllm_minicpm3_dsa/__init__.py:181-201``.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3MSAForCausalLM",
        "scripts.msa.vllm_qwen3_msa.model:Qwen3MSAForCausalLM",
    )


register()
