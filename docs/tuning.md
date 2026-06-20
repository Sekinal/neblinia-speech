# LoRA tuning notes — MexicoSpeech

Hyperparameter intuitions for fine-tuning whisper-large-v3-turbo on Mexican
Indigenous speech (low-resource: ~10h/language). Captured from lab experience +
what the v1 run showed.

## v1 baseline (the bar to beat)
- `r=64`, `lora_alpha=64` (α/r = **1.0** scaling), `lora_dropout=0`
- `target_modules=["q_proj","v_proj"]`, gradient checkpointing on (Whisper needs it)
- `lr=1e-4`, linear schedule, warmup 0.05, **2 epochs**, effective batch 128
- 74,797 train + 300 held-out val clips (Common Voice, CC0)
- val WER trajectory: 85 → 74 → 67 → 62 → … (healthy, no plateau)

## The dials, and how they interact

### Alpha (scaling = α/r)
- v1 is α/r = 1.0. **Try α=128 (α=2r → scaling 2.0)** — a stronger adapter update,
  helpful when data is in-domain but distributionally far (our case).
- Alpha is effectively a *second LR knob* → sweep it **together** with the LR, not
  independently. Too-high alpha → instability / explosion.

### Learning rate
- LoRA tolerates much higher LR than full FT (only the adapter moves). v1's 1e-4 is
  conservative. **Sweep {1e-4, 3e-4, 5e-4}** (up to ~1e-3 can work with tuning).
- Judge by val WER; the loss curve flags divergence fast.

### Rank
- Conventional fear: high rank overfits on small data. In practice **r up to 128
  holds up decently even without extensive data** — ASR has lots of per-token signal
  for the extra rank to use, and the α/r scaling (not raw rank) gates capacity.
- **De-risked by validation**: if r=128 overfits, the val WER turns up and
  `load_best_model_at_end` grabs the pre-turn weights automatically. Cheap to try.

### rsLoRA — handle with care
- Scales by **α/√r** instead of α/r. At r=64 that's α/8 (vs α/64) → **8× bigger**
  update for the same alpha; at r=128 it's α/√128 ≈ α/11.3. So enabling rsLoRA
  **without dropping alpha hard will explode.** Use rsLoRA *with a much smaller alpha*
  (or lower rank), then re-tune.

### DoRA
- Magnitude/direction decomposition; often edges LoRA at *low* rank, ~1.5–2× slower.
  Worth one arm of the sweep, **but plain LoRA usually still wins** at this scale —
  not the headline.

## Proposed v2 sweep (cheap: ~300 steps each, judged on val WER)
| arm | r | α | lr | variant |
|---|---|---|---|---|
| baseline (=v1) | 64 | 64 | 1e-4 | LoRA |
| 2α | 64 | 128 | 1e-4 | LoRA |
| higher-lr | 64 | 64 | 3e-4 / 5e-4 | LoRA |
| high-rank | 128 | 128 | 1e-4 | LoRA |
| rsLoRA | 64 | **16–32** | 1e-4 | rsLoRA (α dropped!) |
| DoRA | 64 | 64 | 1e-4 | DoRA |

Pick the lowest val-WER arm → one full run → eval on the MEXA benchmark.
The `--eval-steps` + best-checkpoint plumbing in `train_unsloth.py` makes this a clean grid.
(Unsloth `get_peft_model` exposes `use_rslora=` and `use_dora=` flags.)
