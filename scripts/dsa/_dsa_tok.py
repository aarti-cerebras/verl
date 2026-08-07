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
"""Tokenizer helpers shared by the DSA/MSA data-generation scripts.

Exists because ``apply_chat_template(..., tokenize=True)`` returns a **different type per transformers
major version**, and getting it wrong is silent:

* transformers 4.x -> ``list[int]``
* transformers 5.x -> ``BatchEncoding`` (ids under ``["input_ids"]``)

Indexing a ``BatchEncoding`` positionally yields ``Encoding`` objects, so a naive ``len()`` returns the
number of *encodings* (e.g. 2) instead of the number of tokens (e.g. 406). That silently corrupted the
``--fit-window`` budget in gen_trajectories.py (caps computed as ``window - 2 - 1``), which let sequences
land one token past the training window. Always go through ``chat_prefix_ids``.
"""


def pin_template_kwargs(kwargs=None, pin_date=None):
    """Normalise the extra ``apply_chat_template`` kwargs, pinning any non-deterministic template state.

    gpt-oss / harmony templates build their system message with ``strftime_now("%Y-%m-%d")``, so the
    **generation date is baked into every prefix**. Left alone that means (a) a resumed or re-run
    generation produces different prefixes than the first leg, and (b) training and serving skew by one
    line as soon as the date rolls over. Jinja context variables shadow globals, so passing
    ``strftime_now`` as a kwarg pins it (verified against gpt-oss-20b, transformers 4.57).

    ``reasoning_effort`` is likewise a real template variable on harmony (default "medium"); pass it
    explicitly so the run log records what was actually served rather than a default that can move.
    """
    out = dict(kwargs or {})
    if pin_date:
        out["strftime_now"] = lambda fmt, _d=pin_date: _d if fmt == "%Y-%m-%d" else _d
    return out


def chat_prefix_ids(tok, messages, add_generation_prompt=True, **tpl_kwargs):
    """Token ids of the templated prefix, as a flat ``list[int]``, on transformers 4.x and 5.x alike."""
    out = tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True,
                                  **tpl_kwargs)
    if hasattr(out, "keys"):  # transformers 5.x BatchEncoding (or return_dict=True)
        assert "input_ids" in out, f"chat template output has no input_ids: {list(out.keys())}"
        out = out["input_ids"]
    if hasattr(out, "tolist"):  # torch/np tensor
        out = out.tolist()
    while out and isinstance(out[0], list | tuple):  # strip batch dim(s)
        out = out[0]
    assert out, "chat template produced no tokens"
    assert all(isinstance(t, int) for t in out[:8]), (
        f"chat template did not yield ints (got {type(out[0]).__name__}); "
        f"transformers version changed the apply_chat_template return type again"
    )
    return [int(t) for t in out]
