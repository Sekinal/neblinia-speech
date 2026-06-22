---
language:
- es
- nah
- mix
- zap
- maz
- amu
license: cc-by-4.0
library_name: ctranslate2
tags:
- automatic-speech-recognition
- whisper
- mexican-indigenous-languages
- low-resource
- nahuatl
- mixtec
- zapotec
base_model: openai/whisper-large-v3
---

# NeblinIA-Speech preview-1.0 (whisper-large-v3)

A foundational ASR model for **Mexican Spanish + 23 Mexican Indigenous languages** (Nahuatl,
Mixtec, Zapotec, Mazatec, Zoque, Amuzgo, Chinantec, Tlapanec, Triqui, Totonac, Mixe, across 6+
language families), by the NeblinIA lab.

## Results (MEXA contamination-resistant benchmark)

| model | WER | CER |
|---|---|---|
| **preview-1.0 (whisper-large-v3, this model)** | **54.4** | **24.2** |
| preview-0.1 (whisper-large-v3-turbo + RL) | 59.0 | 26.5 |

Scored on a **private, held-out** test set (5,925 clips) the model cannot have trained on, with
the identical decoding protocol for every model (faster-whisper, beam 1, temperature fallback).

![Per-language WER and CER](figures/release_per_language.png)

![Architecture scoreboard](figures/release_scoreboard.png)

**Read CER, not just WER.** These languages have no standardized orthography, so references spell
the same spoken word several ways (tone marks, vowel length, tz/ts/z, word segmentation). An error
audit showed the model is acoustically strong (CER 24, about 76% of characters correct) while WER
is inflated by orthographic-convention mismatch, not mishearing. WER overstates the real error.

## What it is

- **Base**: `openai/whisper-large-v3` (the full 32-layer decoder). The decoder capacity was the
  key lever: it beat the 4-layer turbo base (with RL) by about 5 WER points, before any RL.
- **Training**: LoRA SFT on a broad multilingual manifest (about 97k clips) built only from
  open-licensed data (Omnilingual ASR CC BY 4.0, Common Voice v26 CC0, CIEMPIESS CC BY-SA).
- **Files**: `ct2/` (CTranslate2, for faster-whisper, the format these numbers come from) and
  `lora/` (the LoRA adapter, applies on top of `openai/whisper-large-v3`).

## Usage (faster-whisper, recommended)

```python
from faster_whisper import WhisperModel
m = WhisperModel("Thermostatic/neblinia-speech-preview-1.0", revision="main")  # or local ct2/ dir
segs, _ = m.transcribe("audio.wav", beam_size=1)
print("".join(s.text for s in segs))
```

## Reproducibility

- **Code + benchmark + data recreation**: https://github.com/Sekinal/neblinia-speech
  (see `docs/RECREATE_DATA.md` for the full open-source data pipeline) and
  https://github.com/Sekinal/mexa-benchmark (deterministic split + fingerprint registry).
- **Honest log** of everything tried, including the negative results (byte-level and ByT5
  speech-LLM both loop; GSPO RL helps weak models but over-generates on this strong base; the
  open-data ceiling): `docs/findings.md` in the GitHub repo.

## Limitations

- Zapotec and other data-dark languages stay weak: there is no open audio for them, so this is a
  data limit, not a model limit.
- WER near 20 is likely unreachable while the reference orthography itself is inconsistent. The
  honest content-accuracy metric is CER, and an orthographic-normalization protocol is in progress.
- Tone is phonemic in the Oto-Manguean languages; the model does not yet model it reliably.

## License

CC BY 4.0 (matching the dominant training source, Omnilingual ASR). Some training sources are
CC0 and CC BY-SA; see the GitHub `speech-data-research` repo for the full license audit.
