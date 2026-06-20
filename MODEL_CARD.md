# NeblinIA-Speech — preview-0.1

Foundational ASR for **Mexican Indigenous languages**, from the NeblinIA foundational lab.
First public preview. Internal id: `neblinia-preview-0.9-broadgspo`.

## What it is
A fine-tuned **Whisper-large-v3-turbo** (809M) that transcribes 23 Mexican Indigenous
languages spanning Oto-Manguean (Mixtec, Zapotec, Chinantec, Amuzgo, Mazatec, Cuicatec),
Uto-Aztecan (Nahuatl), Totonacan (Totonac), Mixe-Zoque (Zoque), and others, plus Mexican
Spanish. All audio is decoded under a single forced `es` language token (the low-resource
bucket trick); the model learns to map acoustics → the target orthography directly.

## Results (contamination-resistant MEXA benchmark, private test, n=5925)
Evaluated with the **identical fair faster-whisper protocol** every baseline gets
(beam 1, faster-whisper temperature-fallback, auto language detect for non-Spanish):

| model | WER | CER |
|---|---|---|
| **NeblinIA-Speech preview-0.1** | **58.99** | **26.45** |
| prior best (preview-0.3, GSPO) | 66.02 | 28.26 |
| Whisper-large-v3-turbo (baseline) | (much higher; loops on these langs) | — |

**−7.0 WER over our prior best, #1 on the fair leaderboard.**

Per-language is bimodal — strong on some, still hard on others:
- Strong: `spa` 18.8, `zor` 39, `zoh` 50, `tlp` 53, `amu` 56.
- Hard (loop-prone, data-starved): `zts` 106, `nhq` 100 (only 258 train samples), `xti` 89,
  `mig` 86. These cap the average.

## How it was trained (the recipe that worked)
1. **Broad multistage SFT base** — LoRA (all-linear modules, r=64) on **97k clips**:
   Omnilingual ASR (CC BY 4.0, the 23 target varieties, ~22.5k) **+ 75k Common Voice v26
   (CC0)** related-family clips (more Nahuatl, Zoque, Mazatec, Cuicatec, Purépecha, Yaqui,
   Seri, Tarahumara). **Stopped early (~0.9 epoch)** — full training overfits and amplifies
   repetition looping (teacher-forced metrics improve while real autoregressive WER worsens).
2. **GSPO RL post-training** (Group Sequence Policy Optimization, arxiv 2507.18071) — group
   of 8 samples/clip, verifiable reward = composite of −CER/−WER + anti-repetition penalties,
   sequence-level length-normalized importance ratio, KL to a frozen SFT anchor. RL is what
   fixes the exposure-bias looping that SFT alone cannot. Held-out greedy dev CER 0.544→0.422.

## Key lessons (see docs/findings.md for the full log)
- **Teacher-forced eval lies** for this task — it anti-correlates with autoregressive WER in
  late training. Always select checkpoints by raw-greedy autoregressive triage.
- **Data scale helps** (3× data cut looping 26%→16%) but **RL is decisive** (SFT alone, any
  capacity, loops).
- **Decode guards hurt** (`no_repeat_ngram`/`repetition_penalty`) — these languages use
  grammatical reduplication, so "no repetition" must come from RL training, not decode hacks.
- Dead ends: Unsloth full-finetune & label-smoothing crash Whisper; best-of-K
  self-distillation overfits; MGPO frontier-weighting fails on bimodal difficulty.

## Limitations & honest scope
- ~10 h/language of open data is the hard ceiling; the worst languages need more per-language
  data than currently exists openly. Average WER ≤20 is not reached at this data scale.
- No timestamps; ≤30 s segments; single `es` decode bucket (not a language identifier).
- The hard languages still produce occasional errors; not yet production-grade for those.

## License & data
- Model: intended open release. Training data: Omnilingual ASR (CC BY 4.0) + Common Voice
  v26 (CC0). No CC-BY-NC data was used in this checkpoint (kept commercial-clean).
- The MEXA benchmark test transcripts are a held-out private answer key (never published).

## Files
- `models/neblinia-preview-0.9-broadgspo/best/` — LoRA adapter (RL-tuned, on turbo).
- `models/neblinia-preview-0.9-broadgspo/ct2/` — CTranslate2 fp16 model for faster-whisper.
- Reproduce eval: `mexa-benchmark/scripts/run_gpu.sh .venv/bin/python scripts/eval_fw_ours.py <ct2_dir> auto`.
