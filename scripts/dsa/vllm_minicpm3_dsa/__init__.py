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
"""vLLM-side MiniCPM3 DSA lightning indexer package.

Importing this package installs a ``sys.meta_path`` finder that makes the
*top-level* ``deep_gemm`` package un-importable (``ImportError``). This forces
vLLM's ``vllm.utils.deep_gemm._import_deep_gemm()`` to fall back to the VENDORED
``vllm.third_party.deep_gemm`` copy, which is the only one that exposes the
unified ``fp8_fp4_mqa_logits`` FP8/FP4 indexer-logits kernel.

Rationale (see tests/dsa/check_vendored_deepgemm.py Q3): the externally installed
``deep_gemm`` (~/.local) predates ``fp8_fp4_mqa_logits`` and, because vLLM tries
the top-level import first, it shadows the vendored copy — leaving vLLM's public
``fp8_fp4_mqa_logits`` wrapper resolved to a ``_missing()`` stub. Blocking the
external top-level import (WITHOUT touching sys.path / ~/.local, so flashinfer /
fast_hadamard_transform / cupy stay importable) makes vLLM use the vendored copy
end-to-end. The dotted ``vllm.third_party.deep_gemm`` is never matched, so the
vendored copy still imports fine.
"""

import sys


class _BlockExternalDeepGemm:
    """meta_path finder that makes ONLY top-level ``import deep_gemm`` fail.

    Leaves sys.path and ~/.local untouched so every other ~/.local package
    (flashinfer / fast_hadamard_transform / cupy / sgl_kernel) stays importable.
    ``vllm.third_party.deep_gemm`` (dotted, different top-level name) is not
    matched, so the vendored copy still imports.
    """

    def find_spec(self, name, path, target=None):
        if name == "deep_gemm" or name.startswith("deep_gemm."):
            raise ImportError(
                f"blocked external top-level {name!r} so vLLM uses its vendored "
                "vllm.third_party.deep_gemm (has fp8_fp4_mqa_logits)"
            )
        return None


def install_deep_gemm_shim() -> None:
    """Install the meta_path blocker (idempotent) and, if vLLM's deep_gemm wrapper
    was already imported/resolved against the external copy, purge + reset it so it
    re-resolves against the vendored copy on next use."""
    # 1) install the blocker at the front of meta_path (idempotent).
    if not any(isinstance(f, _BlockExternalDeepGemm) for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockExternalDeepGemm())

    # 2) purge any already-cached EXTERNAL deep_gemm modules so the blocker bites.
    #    Leave a vendored alias (installed in step 3) in place.
    for mod in [m for m in sys.modules if m == "deep_gemm" or m.startswith("deep_gemm.")]:
        m = sys.modules[mod]
        if getattr(m, "__name__", "").startswith("vllm.third_party.deep_gemm"):
            continue
        del sys.modules[mod]

    # 3) Alias the top-level ``deep_gemm`` name to the VENDORED copy in sys.modules.
    #    This is essential in vLLM's serving path: ``has_deep_gemm()`` probes with
    #    ``importlib.util.find_spec("deep_gemm")`` WITHOUT a try/except, so the
    #    raising meta_path finder (step 1) would crash it. A sys.modules entry
    #    short-circuits BOTH ``find_spec`` and ``import deep_gemm`` to the vendored
    #    module (which exposes ``fp8_fp4_mqa_logits``), so the finder is never
    #    consulted for that name and vLLM resolves the good kernel end-to-end.
    #    If the vendored copy is unavailable, skip the alias and fall back to the
    #    finder + vLLM's own ImportError-caught fallback in ``_import_deep_gemm``.
    if "deep_gemm" not in sys.modules:
        try:
            import vllm.third_party.deep_gemm as _vendored_deep_gemm

            sys.modules["deep_gemm"] = _vendored_deep_gemm
        except Exception:
            pass

    # 3) if vllm.utils.deep_gemm is already loaded, reset its cached *_impl globals
    #    (they may point at the external/_missing stub) so _lazy_init re-resolves.
    dg = sys.modules.get("vllm.utils.deep_gemm")
    if dg is not None:
        for name in list(vars(dg)):
            if name.endswith("_impl"):
                setattr(dg, name, None)
        # some builds cache a bool "loaded" flag; reset it if present.
        for flag in ("_initialized", "_DEEP_GEMM_INITED", "_loaded"):
            if hasattr(dg, flag):
                try:
                    setattr(dg, flag, False)
                except Exception:
                    pass


# Install at import time so `from scripts.dsa.vllm_minicpm3_dsa.indexer import ...`
# is enough to guarantee the vendored kernel path.
install_deep_gemm_shim()


# --------------------------------------------------------------------------- #
# Stage 2a — force vLLM to route MiniCPM3DSA through the MLA path.
#
# vLLM only builds the MLA path for a DeepSeek-family allowlist:
#   ``ModelArchConfigConvertorBase.is_deepseek_mla()`` gates ``model_config.
#   use_mla`` (config/model.py) and ``get_head_size()`` (which returns the MLA
#   latent head size). ``model_type="minicpm3"`` is NOT in that allowlist, so
#   MiniCPM3 falls back to dense attention and never builds the MLA latent
#   kv-cache. We monkeypatch the (base) convertor to treat our architecture
#   (``MiniCPM3DSAForCausalLM``) as MLA and report the PADDED head size 576
#   (kv_lora 512 + rope 64) so the metadata validation
#   (MLACommonMetadata.__post_init__ / MLACommonBackend.get_supported_head_sizes
#   == [320, 576]) and the kv-cache spec all agree with the padded MLAAttention
#   op built in attention.py.
#
# Scoped to our architecture ONLY (by config.architectures), so the stock dense
# parity reference ``MiniCPM3StockRefForCausalLM`` (same model_type "minicpm3")
# is untouched and stays on the dense materialized-QKV path.
# --------------------------------------------------------------------------- #
_FORCE_MLA_ARCH = "MiniCPM3DSAForCausalLM"
# 512 (kv_lora pad) + 64 (rope pad); MUST match MLA_PAD_HEAD_SIZE in attention.py.
_MLA_PAD_HEAD_SIZE = 576


def _install_force_mla_patch() -> None:
    """Patch ``ModelArchConfigConvertorBase.{is_deepseek_mla,get_head_size}`` to
    route ``MiniCPM3DSAForCausalLM`` through vLLM's MLA path at the padded head
    size. Idempotent."""
    # Import ``vllm.config`` FIRST (via its package __init__) so the whole
    # config import chain resolves in vLLM's normal order. Importing the
    # convertor module directly as the first touch enters mid-cycle
    # (vllm.config.model <-> model_arch_config_convertor) and raises a circular
    # ImportError.
    import vllm.config  # noqa: F401
    import vllm.envs as envs
    import vllm.transformers_utils.model_arch_config_convertor as _mc

    base = _mc.ModelArchConfigConvertorBase
    if getattr(base, "_minicpm3dsa_mla_patched", False):
        return

    orig_is_mla = base.is_deepseek_mla
    orig_head = base.get_head_size

    def _is_dsa(self) -> bool:
        try:
            arch = self.get_architectures() or []
        except Exception:
            arch = []
        return _FORCE_MLA_ARCH in arch

    def is_deepseek_mla(self):  # noqa: ANN001
        if _is_dsa(self):
            # Same predicate the DeepSeek allowlist uses: MLA iff a kv_lora_rank
            # is present (it is, =256 for MiniCPM3).
            return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        return orig_is_mla(self)

    def get_head_size(self):  # noqa: ANN001
        if _is_dsa(self) and not envs.VLLM_MLA_DISABLE:
            # Padded MLA latent head size (kv_lora 512 + rope 64). This is what
            # the metadata head_dim is validated against and what the (also
            # padded) MLAAttention op / kv-cache spec use — keep them consistent.
            return _MLA_PAD_HEAD_SIZE
        return orig_head(self)

    base.is_deepseek_mla = is_deepseek_mla
    base.get_head_size = get_head_size
    base._minicpm3dsa_mla_patched = True


_install_force_mla_patch()


def register() -> None:
    """Register ``MiniCPM3DSAForCausalLM`` with vLLM's ``ModelRegistry``.

    Uses the lazy string form ``"<module>:<class>"`` so importing this package
    (and thus registering) does NOT drag in vLLM's model layer at import time —
    vLLM imports the class only when it actually builds the model. Idempotent.

    The deep_gemm shim above is installed at import time (before this runs), so
    the vendored kernel path is guaranteed by the time the model is built.
    """
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "MiniCPM3DSAForCausalLM",
        "scripts.dsa.vllm_minicpm3_dsa.model:MiniCPM3DSAForCausalLM",
    )
    # Stock dense parity reference (Stage-1 MLA parity test).
    ModelRegistry.register_model(
        "MiniCPM3StockRefForCausalLM",
        "scripts.dsa.vllm_minicpm3_dsa.model:MiniCPM3StockRefForCausalLM",
    )


# Register at import time so a single `import scripts.dsa.vllm_minicpm3_dsa`
# before constructing the vLLM engine wires up both the deep_gemm shim and the
# custom architecture.
register()
