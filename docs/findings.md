# Findings & campaign log — NeblinIA-Speech

Running log of experiments, errors, fixes, and direction changes. Newest first.
Goal (set 2026-06-20): **fair-eval WER ≤ 20 with no repetition** across the 23 MX
Indigenous languages. Current best **preview-0.3 (GSPO) = 66.0 WER / 28.3 CER**.

---

## 2026-06-20 — Campaign kickoff: drive WER 66 → ≤20

### State of play
- Best model: `preview-0.3-gspo` (GSPO RL on top of `preview-0.2` LoRA SFT) — **66.0 / 28.3**.
- Data: **all of Omnilingual MX** materialized = 22,549 train + 5,784 dev segments,
  23 languages, ~1,000 seg/lang (~10 h/lang). This is Omnilingual's ceiling for these
  langs (corpus has ~450–477 rows/lang of ~60 s, force-aligned to ~2 segs each).
- Core problem: **WER (66) ≫ CER (28)**. The acoustic→text mapping partly works; the WER
  gap is dominated by **autoregressive looping/repetition** on the hard languages.

### NEGATIVE RESULT — best-of-K self-distillation (RFT/STaR) — preview-0.4
- Generated best-of-K pseudo-labels (K=8, keep min-CER sample, cap 0.8): 22,088 kept,
  mean best-CER 0.287. Trained LoRA on them (tuned preview-0.2 recipe, r=128, lr 5.2e-4).
- **Overfit immediately**: held-out dev `eval_loss` best at step 100 (2.74) then blew up
  to ~7 while train loss kept falling. Optimum at **epoch 0.29** — recipe tuned for
  from-scratch SFT is far too hot for self-distillation (model already sits near targets).
- The recovered best (ckpt-100) **loops so heavily the fair faster-whisper eval didn't
  finish in 33 min** (clean eval ≈ 5 min) — temp-fallback fires its full 5-pass retry on
  most clips. Self-distillation on the model's own low-diversity outputs *amplified*
  looping rather than fixing it. **Abandoned.**

### New direction (ranked bets)
1. **All-linear-module LoRA** (preview-0.5, RUNNING): q,v-only adapter undertrains a
   language shift. Adding k/out/fc1/fc2 → 6.44% trainable (was 3.15%). r=64, lr 2e-4,
   4 epochs, early-stop patience 4.
2. **Full fine-tune** turbo (queued): tests whether LoRA capacity is the ceiling.
3. **Stronger base?** turbo has a 4-layer distilled decoder — likely a key cause of
   looping (too little decoder capacity for these languages). Test large-v3 (32-layer).
4. **Anti-repetition**: training-side (label smoothing, exposure-bias) + GSPO reward.
5. **Open data expansion** (see below).

### Open-data inventory (training fuel — open licenses only)
- Already using: **Omnilingual ASR corpus** (CC BY 4.0), all 23 MX langs.
- **Common Voice v26 (CC0, cleanest)**: ~10 h each for several MX Indigenous langs —
  Nahuatl `ncx`/`nlv`, Mazatec `mau`, Cuicatec `cut`/`cux`, Purépecha `pua`, Yaqui `yaq`,
  Seri `sei`, Tarahumara `tar`. Multilingual-transfer fuel (mostly different varieties
  than the 23 test langs — watch orthography mismatch).
- **OpenSLR SLR89 Mixtec / SLR92 Puebla-Nahuatl / SLR107 Totonac**: large, time-coded,
  but **CC BY-NC-SA** → non-commercial; flag before any release that bundles them.
- **CIEMPIESS Mexican Spanish (CC BY-SA, ~100 h)**: cheap WER win for Spanish test clips.

### Tooling / infra notes
- New laptop (macOS) drives the box `root@154.54.100.217:40299`. Repos synced locally
  (code+docs only) for codex/Read/Edit; rsync to box to run.
- `train_unsloth.py` now supports `--target {qv,all}`, `--full-finetune`, `--warmup`,
  `--patience` (EarlyStoppingCallback).
- `export_ct2.py` (new, reusable): merge adapter → CT2 float16 → tokenizer fix. Gotchas:
  setsid strips PATH (call converter by abs path); transformers 5.5 saves
  `processor_config.json` not `preprocessor_config.json` (copy the good one);
  config `dtype`→`torch_dtype` rename for the ct2 converter.
- `run_gpu.sh` (mexa-benchmark) restored after container wipe: faster-whisper/ct2 (CUDA-12
  build) needs `libcublas.so.12`+`libcudnn.so.9` on `LD_LIBRARY_PATH`, borrowed from the
  unsloth venv (the mexa venv ships CUDA-13 wheels → `libcublas.so.12 not found`).
- **In-training `eval_wer` is teacher-forced** (`predict_with_generate=False`) → optimistic
  and blind to looping. Always trust the fair faster-whisper autoregressive eval.
- Box is an **ephemeral container**: `/tmp` wiped on restart killed a staged launcher.
  Keep launchers + state under `/root`, not `/tmp`.
</content>
