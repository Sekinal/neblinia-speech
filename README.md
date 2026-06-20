# MexicoSpeech — Training

Fine-tuning ASR models on **Mexican Indigenous speech**, starting from the gap the
`mexa-benchmark` exposed (every SOTA ≥99% WER on Indigenous languages). The goal is a
model — **MexicoSpeech-v0** — that actually transcribes these languages, evaluated on
the same held-out benchmark and added to the leaderboard.

## Approach (v0)

- **Base**: `openai/whisper-large-v3-turbo`
- **Method**: LoRA via **Unsloth** (fast, low-memory)
- **Data**: Common Voice validated Indigenous (~75k clips, 10 locales, CC0, local) +
  CIEMPIESS Spanish (CC BY-SA) to avoid Spanish forgetting
- **Eval**: the held-out Omnilingual benchmark (a *different* corpus → clean, tests
  cross-variety transfer)

## Layout

| Path | What |
|---|---|
| `scripts/prep_train.py` | Build the training manifest from local CV + CIEMPIESS (TRAIN only) |
| `scripts/train_unsloth.py` | Unsloth Whisper LoRA fine-tune → `models/mexicospeech-v0/` |
| `models/` | Checkpoints + adapters (gitignored) |

## Decontamination

The benchmark is Omnilingual/CIEMPIESS-**test**; training uses CV + CIEMPIESS-**train**
(different corpora/splits). Train data is still filtered against the benchmark
fingerprint registry (`mexa-asr-fingerprints`) as a guarantee.

## Usage

```bash
NEBLINIA_DATA=/path/to/shared/data
uv run scripts/prep_train.py                                   # build manifest
.venv-unsloth/bin/python scripts/train_unsloth.py --epochs 1   # fine-tune (Unsloth env)
```

> Note: Unsloth wants `torch>=2.11` for its compiled kernels; pin accordingly in the
> training env.
