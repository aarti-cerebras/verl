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
"""Stage-3 gate: DECODE / multi-token generation correctness for the SPARSE
vLLM server (``MiniCPM3DSAForCausalLM`` + indexer + FLASHMLA_SPARSE backend).

Stage 2b established that the sparse server *builds* and matches HF on the FIRST
generated token. Stage 3 hardens that over a full multi-token decode: it checks
that vLLM's *incremental* decode (both kv-cache groups grow per step, indexer
scores incrementally, sparse impl gathers per step) stays token-for-token
correct against the golden HF sparse forward.

Primary metric = TEACHER-FORCED per-position agreement (NOT free-running sequence
equality — a single bf16 flip cascades in a free run, so free-run divergence is
expected and is only a secondary report):

  1. Greedy-generate G tokens from the vLLM SPARSE server for several prompts
     that are LONGER than top_k=256 (real sub-selection): prose / code / math.
  2. Teacher-force the full ``prompt + vLLM-generation`` through the HF SPARSE
     forward (verl ``minicpm_dsa._sparse_attn``, dsa_mode=sparse, top_k=256) in
     ONE pass, using the EXACT token ids vLLM emitted (no re-tokenization, so
     positions align perfectly).
  3. At each generated position i, vLLM's chosen token ``gen[i]`` was produced
     from prefix ``prompt + gen[:i]``; the HF forward predicts that same next
     token from logits row ``P-1+i`` (P = prompt length). Compare HF's argmax at
     row ``P-1+i`` to ``gen[i]`` (top-1 agreement), and compare HF vs vLLM
     logprobs evaluated at vLLM's reported top-K ids for that step.

Reports:
  * per-position top-1 agreement % (HF argmax == vLLM-generated token) per prompt
  * logprob deltas: max / mean |Δlogprob| over the shared top-K, all positions
  * SECONDARY: free-running HF greedy (full-forward loop, same code path as the
    teacher-forced pass) vs vLLM greedy — longest common prefix / divergence pt
  * DEGENERACY-over-gen: a SHORT prompt where prompt+gen stays <= top_k, so the
    indexer selects every key -> the SPARSE kernel must reduce to the DENSE (2a)
    kernel. Checked TEACHER-FORCED (per-position): feed the sparse-generated
    sequence back through the DENSE engine and compare dense's per-position
    argmax to the sparse-emitted token. (Free-running sparse-vs-dense is also
    reported, but it cascades on a single bf16 flip, so it is NOT the gate.)

PASS iff per-position top-1 agreement is high (>= 90%, the ~0.95-0.97 selection-
parity regime; a few borderline flips OK) AND logprob deltas are in the bf16
noise range consistent with 2a/2b (mean |Δlogprob| <= 0.40 nats).

Run:
  cd <repo> && export PYTHONPATH=$(pwd)/.devlibs/tf457lib:$(pwd)
  /usr/bin/python3 tests/dsa/test_stage3_decode_parity.py
"""

import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

SERVING = (
    "/cb/ml-eng/aarti/dsa/phase2_topk_sweep_20260717_223140/"
    "phase2_full_k256_1ep_20260717_223341/train/checkpoints/inference/"
    "global_step_2805/serving_stage0"
)
OPENBMB = "openbmb/MiniCPM3-4B"  # arch + trust-remote-code modeling (cached locally)
DSA_ARCH = "MiniCPM3DSAForCausalLM"

TOP_K = 256  # indexer top_k (config dsa_top_k / index_topk)

# ---- decode lengths ----
GEN_LONG = 96  # greedy tokens generated per LONG prompt (real multi-token decode).
GEN_SHORT = 64  # greedy tokens for the degeneracy prompt (prompt+gen must stay <=256).
FREE_RUN_STEPS = 48  # HF free-running greedy steps (secondary; full-forward loop).
TOP_LOGPROBS = 20  # per-step top-K logprobs requested from vLLM.

EXPECT_HEAD_SIZE = 576  # padded MLA head size (kv_lora 512 + rope 64).

# ---- gate thresholds ----
TOP1_AGREE_MIN = 0.90  # per-position top-1 agreement (primary metric).
LOGPROB_MEAN_TOL = 0.40  # mean |Δlogprob| over shared top-K (bf16 noise, per 2a/2b).
DEGEN_AGREE_MIN = 0.90  # teacher-forced sparse-vs-dense agreement (degenerate regime).

# --------------------------------------------------------------------------- #
# Prompts (LONG: > top_k=256 tokens after tokenization -> real sub-selection).
# Varied domains so selection is exercised on prose, code, and math.
# --------------------------------------------------------------------------- #
_PROSE_SEED = (
    "In the study of large language models, sparse attention mechanisms such as "
    "DeepSeek Sparse Attention select a subset of key-value pairs for each query "
    "token, which reduces the quadratic cost of full attention while aiming to "
    "preserve the model's quality on long documents. A lightning indexer scores "
    "every past token, keeps only the top-k highest scoring ones, and the main "
    "attention then attends over that selected set rather than the whole context. "
    "This matters because context lengths keep growing, and the memory and compute "
    "of dense attention grow with the square of the sequence length. "
)
_CODE_SEED = (
    "def merge_intervals(intervals):\n"
    "    intervals.sort(key=lambda x: x[0])\n"
    "    merged = []\n"
    "    for start, end in intervals:\n"
    "        if merged and start <= merged[-1][1]:\n"
    "            merged[-1][1] = max(merged[-1][1], end)\n"
    "        else:\n"
    "            merged.append([start, end])\n"
    "    return merged\n\n"
    "def binary_search(arr, target):\n"
    "    lo, hi = 0, len(arr) - 1\n"
    "    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n"
    "        if arr[mid] == target:\n"
    "            return mid\n"
    "        elif arr[mid] < target:\n"
    "            lo = mid + 1\n"
    "        else:\n"
    "            hi = mid - 1\n"
    "    return -1\n\n"
)
_MATH_SEED = (
    "We compute a running sum step by step. Start with 3. Add 7 to get 10. "
    "Multiply by 2 to get 20. Subtract 5 to get 15. Add 11 to get 26. "
    "Divide by 2 to get 13. Add 4 to get 17. Multiply by 3 to get 51. "
    "Subtract 1 to get 50. Add 6 to get 56. Divide by 8 to get 7. "
    "Now consider the sequence 2, 4, 8, 16, 32, and each term is double the last. "
)

LONG_PROMPTS = {
    "prose": (_PROSE_SEED * 4) + "\n\nSummarize the paragraph above in one sentence:",
    "code": (_CODE_SEED * 3) + "\n# Now write a function that reverses a linked list.\ndef",
    "math": (_MATH_SEED * 3) + "\n\nContinuing the pattern, the next three terms are",
}
# Short degeneracy prompt: prompt+GEN_SHORT stays <= top_k=256 -> indexer selects all.
DEGEN_PROMPT = "Once upon a time, in a small village by the sea, there lived"


def _ikeys(d: dict) -> dict:
    """json.dumps stringifies dict keys; normalize a {token_id: logprob} map back
    to int keys."""
    return {int(k): v for k, v in d.items()}


# --------------------------------------------------------------------------- #
# vLLM child: build engine (sparse or dense), greedy-generate, return tokens +
# per-step top-K logprobs + prompt token ids. Captures cache-group specs so we
# can assert the SPARSE decode backend is actually in play. The DENSE child, if
# STAGE3_DENSE_TF is set, ALSO teacher-forces the sparse-generated degeneracy
# sequence (via prompt_logprobs) for the per-position degeneracy check.
# --------------------------------------------------------------------------- #
def _gen_vllm(sparse: bool) -> None:
    import scripts.dsa.vllm_minicpm3_dsa  # noqa: F401  (register + shims + MLA patch)

    mla_specs: list[dict] = []
    idx_specs: list[dict] = []
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

    _orig_mla = MLAAttention.get_kv_cache_spec

    def _cap_mla(self, vllm_config):
        spec = _orig_mla(self, vllm_config)
        try:
            backend = self.attn_backend.get_name()
        except Exception:
            backend = "?"
        mla_specs.append(
            {
                "head_size": int(getattr(spec, "head_size", -1)),
                "backend": backend,
                "use_sparse": bool(getattr(self, "use_sparse", False)),
            }
        )
        return spec

    MLAAttention.get_kv_cache_spec = _cap_mla

    _orig_idx = DeepseekV32IndexerCache.get_kv_cache_spec

    def _cap_idx(self, vllm_config):
        spec = _orig_idx(self, vllm_config)
        idx_specs.append({"head_size": int(getattr(spec, "head_size", -1))})
        return spec

    DeepseekV32IndexerCache.get_kv_cache_spec = _cap_idx

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=SERVING,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("STAGE3_GPU_MEM_UTIL", "0.70")),
        block_size=64,
        seed=0,
        hf_overrides={"architectures": [DSA_ARCH]},
    )

    # The DENSE (2a) baseline is only exercised on the SHORT degeneracy prompt;
    # its FLASH_ATTN_MLA prefill kernel rejects padded MLA dims at long context
    # (a pre-existing dense-path limitation, orthogonal to the sparse decode we
    # gate). The SPARSE run does the LONG prompts and the short one.
    jobs = []  # (tag, prompt_text, max_tokens)
    if sparse:
        for tag, text in LONG_PROMPTS.items():
            jobs.append((tag, text, GEN_LONG))
    jobs.append(("DEGEN", DEGEN_PROMPT, GEN_SHORT))

    prompts = [t for _, t, _ in jobs]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=mt, logprobs=TOP_LOGPROBS, seed=0)
        for _, _, mt in jobs
    ]
    outs = llm.generate(prompts, sps)

    results = []
    for (tag, _, _), o in zip(jobs, outs):
        comp = o.outputs[0]
        step_lps = []
        for step in comp.logprobs or []:
            step_lps.append({int(tid): float(lp.logprob) for tid, lp in step.items()})
        results.append(
            {
                "tag": tag,
                "prompt_token_ids": list(o.prompt_token_ids),
                "n_prompt_tok": len(o.prompt_token_ids),
                "gen_token_ids": list(comp.token_ids),
                "text": comp.text,
                "step_logprobs": step_lps,
            }
        )

    # ---- DENSE teacher-forcing of the sparse-generated degeneracy sequence ----
    tf_degen = None
    if (not sparse) and os.environ.get("STAGE3_DENSE_TF"):
        with open(os.environ["STAGE3_DENSE_TF"]) as f:
            din = json.load(f)
        P = len(din["prompt_token_ids"])
        gen = din["gen_token_ids"]
        rep = din["reported_ids_per_step"]  # ints (list -> survives json)
        full = list(din["prompt_token_ids"]) + list(gen)
        sp_tf = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOP_LOGPROBS)
        o = llm.generate([{"prompt_token_ids": full}], sp_tf)[0]
        pl = o.prompt_logprobs  # aligned to `full`; pl[j] predicts token j from [:j]
        per_step = []
        for i in range(len(gen)):
            j = P + i
            dist = pl[j] if (pl is not None and j < len(pl) and pl[j] is not None) else {}
            argmax = max(dist, key=lambda k: dist[k].logprob) if dist else None
            ids_i = rep[i] if i < len(rep) else []
            at = {int(t): float(dist[t].logprob) for t in ids_i if t in dist}
            per_step.append(
                {"dense_argmax": (int(argmax) if argmax is not None else None), "dense_lp_at": at}
            )
        tf_degen = {"P": P, "G": len(gen), "per_step": per_step}

    print(
        "RESULT_JSON:"
        + json.dumps(
            {
                "mode": "sparse" if sparse else "dense",
                "mla_specs_n": len(mla_specs),
                "mla_head_sizes": sorted({s["head_size"] for s in mla_specs}),
                "mla_backends": sorted({s["backend"] for s in mla_specs}),
                "mla_use_sparse": sorted({s["use_sparse"] for s in mla_specs}),
                "idx_specs_n": len(idx_specs),
                "idx_head_sizes": sorted({s["head_size"] for s in idx_specs}),
                "results": results,
                "tf_degen": tf_degen,
            }
        )
    )


# --------------------------------------------------------------------------- #
# HF child: teacher-force the vLLM-generated sequences through the HF SPARSE
# forward, and (secondary) run a free-running greedy loop. Reads the vLLM
# sequences from a JSON file whose path is in STAGE3_TF_INPUT.
# --------------------------------------------------------------------------- #
def _gen_hf_ref() -> None:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    from verl.models.transformers.monkey_patch import apply_monkey_patch

    dev = "cuda"
    torch.manual_seed(0)

    cfg = AutoConfig.from_pretrained(OPENBMB, trust_remote_code=True)
    sc = json.load(open(os.path.join(SERVING, "config.json")))
    for k in (
        "dsa_enabled", "dsa_mode", "dsa_n_heads", "dsa_head_dim",
        "dsa_rope_head_dim", "dsa_top_k", "dsa_fp8", "index_topk",
    ):
        if k in sc:
            setattr(cfg, k, sc[k])
    cfg.dsa_enabled = True
    cfg.dsa_mode = "sparse"
    cfg.dsa_warmstart_path = None
    cfg._attn_implementation = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        OPENBMB, config=cfg, trust_remote_code=True, dtype=torch.bfloat16
    )
    apply_monkey_patch(model, use_remove_padding=False, ulysses_sp_size=1)

    sd = load_file(os.path.join(SERVING, "model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_idx = sum(1 for k in sd if ".indexer." in k)
    unexpected_real = [k for k in unexpected if ".indexer." in k or "self_attn." in k]
    model = model.to(dev).eval()

    with open(os.environ["STAGE3_TF_INPUT"]) as f:
        payload = json.load(f)

    def _logits_full(ids_list):
        ids = torch.tensor([[int(x) for x in ids_list]], device=dev, dtype=torch.long)
        pos = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        with torch.no_grad():
            return model(input_ids=ids, position_ids=pos).logits[0].float()  # [T, V]

    out = {"load": {"n_indexer_keys": n_idx, "n_missing": len(missing),
                    "unexpected_real": unexpected_real[:8]}, "prompts": {}}

    for item in payload["prompts"]:
        tag = item["tag"]
        P = item["n_prompt_tok"]
        gen_ids = [int(x) for x in item["gen_token_ids"]]
        step_ids = item["reported_ids_per_step"]  # vLLM top-K ids per step
        full_ids = [int(x) for x in item["prompt_token_ids"]] + gen_ids
        G = len(gen_ids)

        # ---- teacher forcing: one forward over prompt+gen ----
        logits = _logits_full(full_ids)  # [P+G, V]
        lp = torch.log_softmax(logits, dim=-1)  # [P+G, V]

        per_step = []
        for i in range(G):
            row = lp[P - 1 + i]  # predicts token at index P+i == gen_ids[i]
            hf_arg = int(row.argmax().item())
            ids_i = [int(t) for t in (step_ids[i] if i < len(step_ids) else [])]
            hf_at = {t: float(row[t].item()) for t in ids_i}
            per_step.append({"hf_argmax": hf_arg, "hf_lp_at": hf_at})
        out["prompts"][tag] = {"P": P, "G": G, "per_step": per_step}

        # ---- secondary: free-running HF greedy (full-forward loop) ----
        if item.get("free_run"):
            cur = [int(x) for x in item["prompt_token_ids"]]
            hf_free = []
            for _ in range(min(FREE_RUN_STEPS, G)):
                r = _logits_full(cur)[-1]
                nt = int(r.argmax().item())
                hf_free.append(nt)
                cur.append(nt)
            out["prompts"][tag]["hf_free_run"] = hf_free

    print("RESULT_JSON:" + json.dumps(out))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _run_child(kind: str, extra_env: dict) -> dict:
    env = dict(os.environ)
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", kind],
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON:")), None
    )
    if line is None:
        print(f"[{kind}] no RESULT_JSON (rc={proc.returncode}).")
        print("---- stdout tail ----\n" + proc.stdout[-3000:])
        print("---- stderr tail ----\n" + proc.stderr[-6000:])
        raise RuntimeError(f"child {kind} failed")
    return json.loads(line[len("RESULT_JSON:") :])


def _lcp(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _write_tmp(obj) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


def run_gate() -> bool:
    print("=" * 78)
    print("STAGE-3 GATE: MiniCPM3DSA SPARSE decode/generation correctness")
    print("=" * 78)

    print("\n[1/3] SPARSE server: greedy-generate LONG prompts + degeneracy prompt ...")
    sp = _run_child("sparse", {"DSA_SPARSE": "1"})

    sp_res = {r["tag"]: r for r in sp["results"]}
    long_tags = [t for t in LONG_PROMPTS]

    # ---- files handed to the teacher-forcing children ----
    tf_payload = {"prompts": []}
    for tag in long_tags:
        r = sp_res[tag]
        tf_payload["prompts"].append(
            {
                "tag": tag,
                "prompt_token_ids": r["prompt_token_ids"],
                "n_prompt_tok": r["n_prompt_tok"],
                "gen_token_ids": r["gen_token_ids"],
                "reported_ids_per_step": [[int(k) for k in d.keys()] for d in r["step_logprobs"]],
                "free_run": True,
            }
        )
    tf_file = _write_tmp(tf_payload)

    dr = sp_res["DEGEN"]
    dense_tf_file = _write_tmp(
        {
            "prompt_token_ids": dr["prompt_token_ids"],
            "gen_token_ids": dr["gen_token_ids"],
            "reported_ids_per_step": [[int(k) for k in d.keys()] for d in dr["step_logprobs"]],
        }
    )

    print("[2/3] DENSE (2a) baseline: free-run + teacher-force the degeneracy seq ...")
    dn = _run_child("dense", {"DSA_SPARSE": "0", "DSA_WIRE_INDEXER": "0",
                              "STAGE3_DENSE_TF": dense_tf_file})
    dn_res = {r["tag"]: r for r in dn["results"]}

    print("[3/3] HF SPARSE reference: teacher-force + free-run ...")
    try:
        hf = _run_child("hfref", {"STAGE3_TF_INPUT": tf_file})
        hf_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  HF reference unavailable: {e}")
        hf, hf_ok = None, False
    finally:
        for p in (tf_file, dense_tf_file):
            try:
                os.unlink(p)
            except OSError:
                pass

    ok = True

    # ---- BUILD sanity: sparse decode backend + two cache groups ----
    print("\n" + "-" * 78)
    print("BUILD) FLASHMLA_SPARSE decode backend + two kv-cache groups")
    print(f"  MLA group  : n={sp['mla_specs_n']}  head_sizes={sp['mla_head_sizes']}  "
          f"backends={sp['mla_backends']}  use_sparse={sp['mla_use_sparse']}")
    print(f"  indexer grp: n={sp['idx_specs_n']}  head_sizes={sp['idx_head_sizes']}")
    build_ok = (
        sp["mla_specs_n"] > 0
        and sp["mla_head_sizes"] == [EXPECT_HEAD_SIZE]
        and sp["mla_backends"] == ["FLASHMLA_SPARSE"]
        and sp["mla_use_sparse"] == [True]
        and sp["idx_specs_n"] == sp["mla_specs_n"]
    )
    print(f"  => BUILD {'OK' if build_ok else 'FAIL'}")
    ok = ok and build_ok

    # ---- A) TEACHER-FORCED per-position agreement (PRIMARY) ----
    print("\n" + "-" * 78)
    print("A) TEACHER-FORCED per-position agreement (HF sparse forward, one pass)")
    if hf_ok:
        print(f"  HF load: indexer_keys={hf['load']['n_indexer_keys']} "
              f"missing={hf['load']['n_missing']} "
              f"unexpected_real={hf['load']['unexpected_real']}")
        all_hits = all_pos = 0
        all_dmax = 0.0
        dsum = dcount = 0
        gateA = True
        for tag in long_tags:
            r = sp_res[tag]
            n_tok = r["n_prompt_tok"]
            sub = n_tok > TOP_K  # real sub-selection?
            steps = hf["prompts"][tag]["per_step"]
            gen = r["gen_token_ids"]
            vlp = r["step_logprobs"]
            hits = sum(1 for i, s in enumerate(steps) if s["hf_argmax"] == gen[i])
            npos = len(steps)
            miss_pos = [i for i, s in enumerate(steps) if s["hf_argmax"] != gen[i]]
            pmax = 0.0
            psum = pcnt = 0
            for i, s in enumerate(steps):
                hf_at = _ikeys(s["hf_lp_at"])
                v_at = _ikeys(vlp[i])
                for t, hlp in hf_at.items():
                    vv = v_at.get(t)
                    if vv is None:
                        continue
                    d = abs(hlp - vv)
                    pmax = max(pmax, d)
                    psum += d
                    pcnt += 1
            agree = hits / max(npos, 1)
            pmean = psum / max(pcnt, 1)
            all_hits += hits
            all_pos += npos
            all_dmax = max(all_dmax, pmax)
            dsum += psum
            dcount += pcnt
            flag = "" if sub else "  (WARN: n_tok<=top_k, NOT sub-selecting)"
            print(f"  [{tag:5s}] n_tok={n_tok:>4} gen={npos:>3}  top1 {hits}/{npos}"
                  f" = {agree:.2%}  |Δlp| max={pmax:.3e} mean={pmean:.3e}{flag}")
            if miss_pos:
                print(f"          mismatch steps: {miss_pos}")
            if sub and agree < TOP1_AGREE_MIN:
                gateA = False
        overall_agree = all_hits / max(all_pos, 1)
        overall_mean = dsum / max(dcount, 1)
        print(f"  ---- overall: top1 {all_hits}/{all_pos} = {overall_agree:.2%}  "
              f"|Δlogprob| max={all_dmax:.3e} mean={overall_mean:.3e}")
        gateA = gateA and overall_agree >= TOP1_AGREE_MIN and overall_mean <= LOGPROB_MEAN_TOL
        print(f"  thresholds: per-prompt & overall top1 >= {TOP1_AGREE_MIN:.0%}, "
              f"mean|Δlp| <= {LOGPROB_MEAN_TOL}")
        print(f"  => GATE A {'PASS' if gateA else 'FAIL'}")
        ok = ok and gateA
    else:
        print("  SKIPPED (HF reference could not be built) — see stderr above.")
        ok = False

    # ---- B) SECONDARY: free-running divergence (report only) ----
    print("\n" + "-" * 78)
    print("B) FREE-RUNNING greedy: vLLM vs HF full-forward loop (SECONDARY, report)")
    if hf_ok:
        for tag in long_tags:
            gen = sp_res[tag]["gen_token_ids"]
            hf_free = hf["prompts"][tag].get("hf_free_run", [])
            n = min(len(hf_free), len(gen))
            lcp = _lcp(gen[:n], hf_free[:n])
            status = "identical over compared range" if lcp == n else f"diverges at step {lcp}"
            print(f"  [{tag:5s}] compared {n} steps: longest common prefix = {lcp}  ({status})")
        print("  (free-run divergence is EXPECTED from bf16 flips; not a gate.)")
    else:
        print("  SKIPPED (HF reference unavailable).")

    # ---- C) DEGENERACY over generation: sparse kernel reduces to dense ----
    print("\n" + "-" * 78)
    print("C) DEGENERACY over generation (short prompt, prompt+gen <= top_k):")
    print("   sparse kernel must reduce to dense (2a) — TEACHER-FORCED per position")
    s = sp_res["DEGEN"]
    P = s["n_prompt_tok"]
    sg = s["gen_token_ids"]
    vlp = s["step_logprobs"]
    tfd = dn.get("tf_degen")
    # only positions where the running total (prompt + emitted) stays <= top_k are
    # guaranteed degenerate (indexer selects all keys).
    valid = [i for i in range(len(sg)) if P + i < TOP_K]
    if tfd is not None:
        steps = tfd["per_step"]
        match = 0
        pmax = 0.0
        psum = pcnt = 0
        first_div = None
        for i in valid:
            da = steps[i]["dense_argmax"]
            if da == sg[i]:
                match += 1
            elif first_div is None:
                first_div = i
            d_at = _ikeys(steps[i]["dense_lp_at"])
            v_at = _ikeys(vlp[i])
            for t, dlp in d_at.items():
                vv = v_at.get(t)
                if vv is None:
                    continue
                dd = abs(dlp - vv)
                pmax = max(pmax, dd)
                psum += dd
                pcnt += 1
        degen_agree = match / max(len(valid), 1)
        degen_mean = psum / max(pcnt, 1)
        print(f"  prompt_tok={P}  teacher-forced {len(valid)} positions with total<=top_k")
        print(f"  dense-argmax == sparse-token: {match}/{len(valid)} = {degen_agree:.2%}"
              + ("" if first_div is None else f"  (first mismatch at step {first_div})"))
        print(f"  |Δlogprob| (sparse vs dense) max={pmax:.3e} mean={degen_mean:.3e}")
        gateC = (
            len(valid) > 0
            and degen_agree >= DEGEN_AGREE_MIN
            and degen_mean <= LOGPROB_MEAN_TOL
        )
    else:
        print("  dense teacher-forcing unavailable (tf_degen is None).")
        gateC = False
    # free-running sparse-vs-dense: report only (cascades on a single flip).
    dg = dn_res["DEGEN"]["gen_token_ids"]
    fr_valid = [i for i in range(min(len(sg), len(dg))) if P + i < TOP_K]
    fr_match = sum(1 for i in fr_valid if sg[i] == dg[i])
    fr_lcp = _lcp(sg, dg)
    print(f"  [free-run, report only] sparse-vs-dense tokens: {fr_match}/{len(fr_valid)} "
          f"match while total<=top_k; longest common prefix={fr_lcp} "
          f"(free-run cascade expected, NOT gated)")
    print(f"  sparse text: {s['text']!r}")
    print(f"  => GATE C {'PASS' if gateC else 'FAIL'}  (teacher-forced agree >= "
          f"{DEGEN_AGREE_MIN:.0%}, mean|Δlp| <= {LOGPROB_MEAN_TOL})")
    ok = ok and gateC

    return ok


def main():
    if "--child" in sys.argv:
        kind = sys.argv[sys.argv.index("--child") + 1]
        if kind == "sparse":
            _gen_vllm(sparse=True)
        elif kind == "dense":
            _gen_vllm(sparse=False)
        elif kind == "hfref":
            _gen_hf_ref()
        else:
            raise ValueError(kind)
        return
    passed = run_gate()
    print("\n" + "#" * 78)
    print(f"# STAGE 3 GATE: {'PASS' if passed else 'FAIL'}")
    print("#" * 78)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
