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

### NEGATIVE — Parakeet-CTC head-to-head blocked by an HF backward bug (2026-06-21)
Attempted a direct Parakeet (FastConformer) vs Whisper comparison, fully in HF/safetensors
(no NeMo, per requirement). Got far but hit a wall. What was solved:
- **Tokenizer**: Parakeet's English subword tokenizer maps every diacritic/glottal stop to
  `<unk>` (cannot represent these orthographies). Fixed with a SentencePiece subword vocab
  (512 BPE, char-coverage 1.0) trained on our text + a fresh CTC head on the pretrained
  encoder. (Char-level is infeasible: 8x subsampling = ~12.5 frames/sec < chars/sec.)
- **torch version**: under torch 2.10 the Parakeet cuda cpp-extensions are skipped and the
  fallback backward crashes immediately. Built `.venv-parakeet` with **torch 2.11.0+cu128**;
  the short-clip backward then runs clean (no crash, no NaN).
- **THE WALL**: on real full-data training the backward still throws `CUDA error: illegal
  memory access` within ~16-40 steps. Systematically ruled out: precision (fp32 & bf16 both
  crash), sequence length (dummy sweep OK at all lengths to 28s), CTC alignment (filtered
  clips where label>frames, still crashes), and padding (length-grouped sampler pushed the
  crash from step 16 to 40 but did not fix it). It only survives the tiny 12-step debug.
- **Verdict**: the HF `ParakeetForCTC` integration (added 2025-09) has a backward CUDA bug
  on real training that needs upstream-level debugging. Not viable now; abandoned in favor
  of the data-scaling work that directly serves the WER goal. Scripts kept: train_parakeet.py,
  eval_parakeet.py, debug_parakeet.py, sweep_len.py.

### ✅ CONSOLIDATION — ship preview-0.9 as NeblinIA-Speech preview-0.1 (2026-06-20)
Decision (user): consolidate at the best result rather than chase uncertain incremental gains
against the data ceiling. **Deliverable = preview-0.9-broadgspo, 58.99 WER / 26.45 CER** (#1
fair, −7 vs prior). See MODEL_CARD.md. Model persisted to models/neblinia-preview-0.9-broadgspo/
{best,ct2} (out of ephemeral /tmp).
- balanced-broad (preview-0.11) confirmed the early-stop lesson: full 2-epoch SFT overfit →
  autoregressive WER 123/loop 24% (worse than early-stopped broad's 108/16%). Balancing (less
  CV data) also hurt. So the winning recipe stays: full 97k CV data + EARLY STOP + GSPO RL.
- preview-0.12 (RL from the overfit balanced base) killed — low EV, wouldn't beat 59.
- Untried lever left for a future round: large-v3 (32-layer decoder) capacity — slow RL cycle,
  deferred. Path to <50 WER likely needs more per-language data than exists openly.
- HF leaderboard push pending token ROTATION (token was pasted in plaintext 3x → burned).

### NEGATIVE — decode guards hurt (reduplication); continuation RL plateaus (2026-06-20)
- **Decode guards** (`no_repeat_ngram_size=3` + `repetition_penalty=1.15`) on preview-0.9:
  WER 61.61 vs unguarded 58.99 — **WORSE**. These polysynthetic langs use **reduplication
  grammatically**, so penalizing repetition suppresses CORRECT output. Helped the worst
  looper (nhq 100→82) but hurt Spanish (18.8→20.7) and reduplicating langs (vmj 66→78).
  → **"No repetition" must come from RL training, not decode hacks.** The RL'd model's
  normal temp-fallback decode (= the fair eval) is already loop-free. **Leaderboard model
  IS the product model** — no separate guarded version. (eval_fw_guarded.py kept for record.)
- **Continuation RL** (preview-0.10-rl2, GSPO from the already-RL'd preview-0.9 base, lower
  lr + stronger anti-rep): dev CER stuck 0.43–0.44, never beat the 0.414 starting point.
  More RL of the same kind = diminishing returns. Killed. → to beat 59, need a BETTER BASE
  or MORE DATA, not more RL. Next: balanced-broad base (CV capped 2500/lang so the 23 test
  langs get ~half the weight vs ¼) → fresh RL.

### 🎯 NEW BEST — preview-0.9-broadgspo (data + RL) = 58.99 / 26.45 (2026-06-20)
Fair faster-whisper eval (auto mode, strict leaderboard parity, n=5925):
- **WER 58.99 / CER 26.45** vs prior best preview-0.3 = 66.02 / 28.26 → **−7.0 WER**.
- Recipe: broad-pretrain (97k omni+CV) SFT base → GSPO RL (group8, 400 steps, dev-CER
  0.544→0.422). The **data + RL** combination, exactly as the evidence predicted.
- Evals fast (no catastrophic looping) — RL fixed the worst loops.
- **Bimodal per-language WER** (the remaining battle): usable-ish — spa 18.8, zor 39,
  zoh 50, tlp 53, amu 56; still failing — zts 106, nhq 100, xti 89, mig 86, nhg 80.
  The ~6-8 hard/starved langs (Mixtec/Zapotec tonal, nhq 258 samples) cap the average.
- Next: continuation RL (preview-0.10-rl2, lower lr 7e-7, stronger anti-rep w-rep0.8)
  from this base, targeting the loopers. HF leaderboard push pending token rotation;
  recorded here as the authoritative log.

### RESULT — data scale helps autoregression, but RL is decisive (2026-06-20)
Autoregressive raw-greedy dev triage (eval_dev_fast, 460 clips), the ONLY trustworthy metric:
| model | data | recipe | WER | CER | loop% |
|---|---|---|---|---|---|
| preview-0.3 | 22.5k | q,v SFT **+ GSPO** | **89** | 50 | **11.3** |
| broad ck1800 | **97k** | all-mod SFT, 0.9ep | 108 | 60 | 15.9 |
| p05 | 22.5k | all-mod SFT, ~4ep | 139 | 87 | 26.3 |
- **3× data cut looping 26%→16%, WER 139→108** — data scale clearly helps free-running gen.
- But broad-SFT (108) still loses to preview-0.3 (89) because **preview-0.3 had RL**.
  → **data + RL together** is the play. Stopped broad at 0.9ep (more epochs risk overfit-
  looping like p05) and launched **preview-0.9-broadgspo = GSPO on the broad-ck1800 base**
  (group8, 400 steps, lr1e-6, reward w-len0.4/w-rep0.6, autoregressive dev-CER val every 25).
- best langs already strong even pre-RL: ztp CER20/loop5%, vmj/zpv CER~29/loop0%.

### PIVOT — teacher-forced SFT amplifies looping; RL is the real lever (2026-06-20)
The biggest finding of the campaign. Hard evidence:
- **preview-0.5 (all-mod LoRA)**: teacher-forced eval_wer *improved* monotonically
  (78.6→64.5, eval_loss 1.95→1.33) — looked great. But the **autoregressive raw-greedy
  dev triage = WER 139.7 / CER 87.5 / LOOP 26.3%** — WORSE than the preview-0.3 baseline
  (89/50/11%). More adapter capacity → better teacher-forced fit → **worse free-running
  generation** (exposure bias). Per-lang: pmq 273%/loop65%, zoh 284%/loop40%, vmp 299%.
- **Implication 1**: the in-training `eval_wer` (teacher-forced, `predict_with_generate=
  False`) ANTI-correlates with real WER in late training. `load_best_model_at_end` on it
  saves the MOST-overfit/loopiest checkpoint. → must select checkpoints by autoregressive
  triage, and ideally switch eval to generation on a dev subset.
- **Implication 2 (the pivot)**: preview-0.3 reached 66 *because GSPO (RL) optimizes
  free-running sampled output* — directly fixing exposure bias. **More SFT ≠ the answer.
  The path to WER≤20 is RL (GSPO/MGPO) on a good base + more data**, with a strong
  anti-repetition reward. Likely best SFT base = LIGHT adapter (less overfit-looping),
  then heavy RL. (q,v-only may beat all-mod as an RL base — to verify.)

### ERROR + FIX — LABEL SMOOTHING breaks Unsloth+Whisper (2026-06-20)
- `--label-smoothing 0.1` crashes at step 0:
  `ValueError: cannot specify both decoder_input_ids and decoder_inputs_embeds`
  (unsloth_compiled_module_whisper.py, WhisperDecoder_forward). HF's label-smoother path
  pops `labels` and feeds the model differently, conflicting with Unsloth's patched
  Whisper forward.
- **Misdiagnosed twice first**: blamed full-FT, then a "poisoned cache". The real cause is
  **label smoothing** — proof: p05 (LoRA, NO label smoothing) trained fine; both full-FT
  AND broad-pretrain used `--label-smoothing 0.1` and crashed identically; deleting the
  cache did NOT fix broad (it recrashed). Removing `--label-smoothing` fixed it.
- **Fix**: don't pass `--label-smoothing` with Unsloth+Whisper (removed from broad/largev3
  launchers). For anti-repetition, use RL reward shaping instead.
- full_finetuning=True is separately untested without label smoothing and deprioritized
  (RL pivot). The compiled cache lives in `unsloth_compiled_cache/`; `rm -rf` it if a run
  leaves it in a bad state.

### ERROR + FIX — 5h GPU idle from pgrep self-match (2026-06-20)
- **What happened**: preview-0.5 finished cleanly at 10:03 but the chain watcher never
  fired full-FT — **GPU sat idle ~5 hours**. Same class of waste flagged before.
- **Root cause**: watchers used `while pgrep -f "0.5-allmod"; do sleep; done` to detect
  "training still running". `pgrep -f PATTERN` matches the full command line of *every*
  process — including (a) the watcher's own inline command if PATTERN is in it, and (b)
  any *other* concurrent SSH command of mine that contains PATTERN (every status check
  did). So the loop kept "seeing" the run alive and never advanced. A stuck background
  monitor with `0.5-allmod` in its cmdline kept the chain blocked indefinitely.
  Bonus footgun: `pkill -f "while pgrep"` killed its *own* shell (pattern in cmdline).
- **FIX**: detect completion via the launcher's **EXIT marker in the log**, not pgrep:
  `while ! grep -q "^EXIT" "$LOG"; do sleep 30; done`. Each launcher appends
  `echo "EXIT $? at $(date)"` when training ends. Robust, no cmdline matching.
  General rule: never put a match/kill PATTERN in a command whose own command line then
  contains PATTERN. For "is it on the GPU?", trust `nvidia-smi --query-compute-apps`, not
  `pgrep` on a name. Keep launchers/state under /root (container wipes /tmp).

### DATA DISCOVERY — 75k CV clips already on disk (multistage fuel)
`data/mdc/` already holds **Common Voice v26 (CC0)** for 10 related MX Indigenous langs,
extracted, standard format, **soundfile reads the mp3s directly** (no align/materialize):
cux 9016, zoc 8886, ncx 8644, sei 8006, tar 7895, pua 7537, yaq 6925, nlv 6667,
mau 6040, cut 5481 = **~75k validated clips** (3x our omni 22.5k). `zoc` (Zoque) is the
same family as test `zoh`; ncx/nlv are more Nahuatl. `ciempiess_spa` (Mexican Spanish,
CC BY-SA) is also materialized.
- `build_broad_manifest.py` → `manifest_broad.jsonl` = omni-23 (22.5k) + CV-10 (75k) =
  **97,646 clips**. CV langs are TRAIN-ONLY transfer fuel; dev stays omni-23 (the test).
- **Plan = multistage** (CV dominates the mix, so don't just blend): Stage-1 broad-pretrain
  on 97k → save merged → Stage-2 specialize on omni-22.5k via `--base <stage1>`. Deploy if
  recipe levers (0.5/0.6/0.7) plateau. Manifest ready; trainer `--base` supports the resume.

### BASELINE — preview-0.3 raw-greedy dev triage (the bar to beat)
`eval_dev_fast.py`, 23×20=460 dev clips, raw greedy (no temp-fallback → looping visible):
- **Overall WER 89.2 / CER 50.5 / LOOP 11.3%** (vs fair-test 66/28 — fallback masks loops).
- **Catastrophic loopers** (drive the average): nhq 187/loop30% (only 258 train samples!),
  pmq 157/25%, trq 142/15%, xti 140, mig 118/25%, zts 111/20%, tcf loop30%, amu loop30%.
- **Already near-usable** (proof WER≤20/lang is reachable): ztp 50.7/CER20/loop5%,
  ztn 55, vmj 55/CER21, vmc 64/CER19.7/loop0%, zpv 70/CER23/loop0%.
- **Read**: the average is wrecked by ~6-8 looping languages, several data-starved.
  Killing the loops + feeding the starved langs is the path. Good langs already prove
  the per-language floor is low.

### Research notes (2026-06-20)
- **WER ≤20 is achievable for this language class.** Reported low-resource agglutinative
  results (e.g. SeTswana) go from ~223% WER (catastrophic looping) to ~13% under proper
  fine-tuning. Our 66 has large headroom; looping collapses with enough adaptation.
- **turbo = 4 decoder layers vs large-v3 = 32.** turbo shows "larger degradation on some
  languages"; the shallow decoder is a prime suspect for repetition collapse on the hard
  langs. → queued `preview-0.7-largev3` (all-mod LoRA on large-v3) as the capacity test.
- **Label smoothing 0.1**: standard mitigation for overconfident repetition in seq2seq
  ASR. Added as `--label-smoothing`; enabled on full-FT (0.6) and large-v3 (0.7) runs.
- **Multistage fine-tuning** (broad related-family + Spanish pretrain → exact-variety
  fine-tune) is the main DATA lever if recipe levers plateau — deferred until we see
  whether full-FT breaks the plateau (don't invest hours in data before knowing it's the
  bottleneck).
- In-training `eval_wer` is teacher-forced; for the real looping signal use
  `eval_dev_fast.py` (raw-greedy, no temp-fallback, reports loop%).

### Experiment matrix (this campaign)
| id | base | method | key knobs | status |
|---|---|---|---|---|
| 0.5-allmod | turbo | LoRA all-linear | r64, lr2e-4, 4ep, ES4 | RUNNING |
| 0.6-fullft | turbo | full fine-tune | lr1e-5, 3ep, ls0.1, ES4 | queued (auto after 0.5) |
| 0.7-largev3 | large-v3 | LoRA all-linear | r64, lr2e-4, 3ep, ls0.1 | staged (conditional) |
| (later) | best | GSPO/MGPO RL | anti-repetition reward | pending |
| (later) | best | + open-data multistage | CV CC0 + CIEMPIESS | pending |

Decision rule: eval each on dev triage (loop% + WER) first, full fair eval on winners.
Kill any run whose dev eval_loss climbs for 4 evals (early-stop handles automatically).

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
