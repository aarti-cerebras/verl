# DSA Phase-1 datasets — reproducible build

Scripts to (re)generate the indexer warm-up datasets and splits. **Reproducible by design:** a pinned HF
dataset commit SHA + a fixed `--seed` produce byte-identical windows. Each output gets a `MANIFEST.json`
recording the seed, resolved source SHA, tokenizer, counts, and git commit — copy the SHA back to reproduce
an exact past build.

All windows are **one document per row** (item 6a): each doc is retokenized with the **MiniCPM3-4B**
tokenizer (the InfLLM `token_ids` are CPM-5 vocab and unusable), filtered to `>= length`, truncated to
exactly `length`.

## Sources

| Split | Source | Column | Role |
|---|---|---|---|
| train + in-dist val | `openbmb/InfLLM-V2-data-5B` | `text` | long-context training corpus; disjoint seeded split |
| OOD val | `openbmb/Ultra-FineWeb` (en) | `content` | web corpus, different distribution; multiple lengths |

## One-command build (Phase A)

```bash
# needs the transformers-4.57.1 env for the MiniCPM3 tokenizer + network access to HF
examples/dsa/build_phase_a_datasets.sh
```

Defaults (override via env): `SEED=1234`, `SEQ_LEN=4096`, `TRAIN_WINDOWS=2048`, `VAL_WINDOWS=256`,
`OOD_LENGTHS=1024,2048,4096`, `OOD_PER_LEN=128`, `OOD_MAX_FILES=64`, `OOD_MIN_SCORE=0.9`,
`OUT_DIR=data/dsa/phase_a`.

Produces under `data/dsa/phase_a/`:
- `infllm_minicpm3_4096_train.parquet`  (+ `.MANIFEST.json`)
- `infllm_minicpm3_4096_val.parquet`     — in-distribution val, document-disjoint from train
- `ood_ultrafineweb_minicpm3_L1024.parquet`, `_L2048.parquet`, `_L4096.parquet` (+ `.MANIFEST.json`)

## Reproducing an exact past build

Read the SHAs from the manifest and pin them:

```bash
REV_INFLLM=<sha-from-manifest> REV_UFW=<sha-from-manifest> SEED=1234 \
  examples/dsa/build_phase_a_datasets.sh
```

## Individual scripts

```bash
# InfLLM train + disjoint in-dist val (seeded)
python examples/dsa/prepare_real_data.py \
  --out data/dsa/phase_a/infllm_minicpm3_4096_train.parquet \
  --val_out data/dsa/phase_a/infllm_minicpm3_4096_val.parquet \
  --num_windows 2048 --val_windows 256 --seq_len 4096 --seed 1234

# Ultra-FineWeb OOD val at multiple lengths (one parquet per length)
python examples/dsa/prepare_ood_data.py \
  --out_prefix data/dsa/phase_a/ood_ultrafineweb_minicpm3 \
  --lengths 1024,2048,4096 --per_len 128 --seed 1234 --max_files 64
```

## Consuming in the trainer

Each parquet is drop-in for `PackedPretrainDataset`. Point `data.train_files` at the train parquet and
`data.val_files` at a val parquet with `data.max_length` set to that file's length. The OOD files are
**per-length** so each is evaluated at its native length (recall-vs-length curve); evaluating several
val sets in one run needs the multi-named-val-set trainer support (see `docs/dsa_train_indexer_plan.md`,
"Training & Validation Runs" → remaining work).

## Quality filter (OOD)

Ultra-FineWeb carries a per-doc quality-classifier `score` (float in `[0.5, 1.0]`; the corpus is already
pre-filtered to ≥0.5). `prepare_ood_data.py --min_score` keeps only docs at/above the threshold
(default **0.9** ≈ top quartile: median ≈0.78, p75 ≈0.93). Docs below it (or with an unparseable score) are
dropped; the manifest records `min_score`, `rows_read`, and `score_dropped`. Set `OOD_MIN_SCORE=0`
(or `--min_score 0`) to disable. InfLLM has no score column, so its splits are length-filtered only.

Filtering order per doc: **quality (score) → length (≥ target) → largest-fillable length bucket** (each doc
used once). A stricter `min_score` shrinks the pool, so long buckets fill more slowly — raise
`OOD_MAX_FILES` if a long bucket comes up short (check `windows` vs `requested` in the manifest).

## Notes

- **Phase B (32K):** rerun with `SEQ_LEN=32768` and larger counts; Ultra-FineWeb is short, so high OOD
  lengths (16K/32K) will be low-yield — raise `OOD_MAX_FILES` or keep OOD lengths modest.
- `--oversample > 1.0` collects a wider pool before the shuffle/split (more decorrelated train vs val).
- `--max_files` (OOD) samples across shards after a seeded shuffle of the 2048-file list, so a small cap
  still draws from the whole corpus rather than only the first shard.
