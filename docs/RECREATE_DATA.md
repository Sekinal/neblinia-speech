# Recreating the NeblinIA-Speech dataset & benchmark from scratch

Everything here is rebuildable from **open sources** — no data is committed (heavy + the private
test must stay private), but every source is pinned and every step is scripted across the three
repos. The benchmark split is **deterministic** (fixed salt), so anyone who runs this gets the
*identical* train/dev/test partition and the same fingerprint registry.

## Repos
- **`Sekinal/speech-data-research`** — acquisition (download/materialize) + source & license docs.
- **`Sekinal/neblinia-speech`** (this repo) — manifest building + training.
- **`Sekinal/mexa-benchmark`** — deterministic split + fingerprinting + eval.

## Sources (all open; full detail in `speech-data-research/docs/data-sources.md` + `licensing.md`)
| Source | HF / access | License | Role |
|---|---|---|---|
| Omnilingual ASR corpus | `facebook/omnilingual-asr-corpus` (23 MX configs, `*_Latn`) | CC BY 4.0 | the 23 Indigenous langs + held-out test |
| Common Voice v26 (MX Indigenous) | Mozilla Data Collective slugs (in `data-sources.md`) | CC0 | broad pretrain (10 extra langs, train-only) |
| CIEMPIESS family | `ciempiess/*` (needs `datasets<3` + `trust_remote_code=True`) | CC BY-SA 4.0 | Mexican Spanish |

## One-time config
All scripts read a shared data root. Set it once (defaults to the repo's `data/` if unset):
```bash
export NEBLINIA_DATA=/path/to/shared/data    # materialized corpora + manifests land here
```
> Note: a few legacy scripts still hardcode `/root/foundational_asr` (e.g. `build_broad_manifest.py`);
> either run under that layout or edit the `ROOT=` line. Tracked as a portability cleanup.

## Step 1 — Acquire + materialize  (repo: `speech-data-research`)
```bash
# Omnilingual: download + decode the 23 MX configs into per-(lang,split) wav + manifest tsv
uv run python scripts/materialize.py omni --splits test,dev,train      # default = all 23 MX configs
# CIEMPIESS Spanish (ready test split)
uv run python scripts/materialize.py ciempiess
# Common Voice v26 (CC0) via Mozilla Data Collective (Playwright; clickwrap, not re-hosted)
uv run python scripts/mdc.py login                                     # interactive once
uv run python scripts/mdc.py download <slug> [<slug> ...]              # slugs in data-sources.md
```
→ produces `data/materialized/<source>_<lang>/{train,dev,test}/` + `.tsv`, and `data/mdc/<lang>/`.

## Step 2 — Build manifests  (repo: `neblinia-speech`)
```bash
# In-domain: word-align + segment long Omnilingual clips into <=30s utterances
python scripts/prep_indomain.py --splits train,dev
#   -> data/train/manifest_indomain.jsonl  (+ _dev.jsonl)   ~22.5k segs, 23 langs
# Broad pretrain: in-domain Omnilingual + the 10 CC0 Common Voice langs (train-only)
python scripts/build_broad_manifest.py --out data/train/manifest_broad.jsonl
#   -> ~97k clips
```

## Step 3 — Build the contamination-resistant benchmark  (repo: `mexa-benchmark`)
```bash
uv run python -m mexa.build_benchmark        # add --no-audio to skip test fingerprinting
```
- Splits each source **deterministically** with the fixed salt `neblinia-mexa-v0` (changing it
  reshuffles everything), preferring each corpus's official split, enforcing **speaker-disjoint**
  test vs train. → `data/benchmark/manifests/<source>_<lang>.jsonl`, `benchmark.json`
  (sizes + `benchmark_version` = hash of test fingerprints), `fingerprints.jsonl` (hashes only).
- Same sources + same salt ⇒ **identical** test set on any machine ⇒ reproducible leaderboard.

## Step 4 — Decontaminate training (guarantee)
Training uses CV + CIEMPIESS-**train** + Omnilingual-**train** (different splits than the test),
and is additionally filtered against the **public fingerprint registry** (`mexa-asr-fingerprints`,
hashes only) so no test clip/transcript can leak in. See this repo's README "Decontamination".

## Recap (clone → data → benchmark)
```bash
git clone https://github.com/Sekinal/speech-data-research && git clone https://github.com/Sekinal/neblinia-speech && git clone https://github.com/Sekinal/mexa-benchmark
# then: Step 1 (materialize) -> Step 2 (manifests) -> Step 3 (benchmark)
```
That regenerates the entire dataset + the exact benchmark. Models are on Hugging Face
(`Thermostatic/neblinia-speech*`); see this repo's `MODEL_CARD.md`.
