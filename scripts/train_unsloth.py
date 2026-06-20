"""Fine-tune whisper-large-v3-turbo on Mexican Indigenous speech with Unsloth (LoRA).

Trains on the Common Voice indigenous manifest (data/train/manifest.jsonl). All
clips share a single Whisper language token ("es", the standard low-resource trick:
the token becomes a 'Mexican-languages' bucket; the model just learns to transcribe).
The held-out benchmark is a *different* corpus (Omnilingual), so eval is clean.

  .venv-unsloth/bin/python scripts/train_unsloth.py [--max-samples N] [--epochs E]

Saves the LoRA adapter + a merged fp16 model under models/mexicospeech-v0/.
"""

from __future__ import annotations

# Unsloth must be imported before transformers/peft so its patches apply.
import unsloth  # noqa: F401  (import order matters)

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "train" / "manifest.jsonl"
OUTDIR = ROOT / "models" / "mexicospeech-v0"
BASE = "openai/whisper-large-v3-turbo"
LANG = "es"


def build_dataset(max_samples, val_per_lang=30):
    """Lightweight {audio_path, text} datasets, split into train + a held-out
    validation set (val_per_lang clips per language). Features are extracted lazily
    in the collator (parallel workers) so we never precompute ~75 GB of log-mels.

    NB: validation is a held-out slice of the *training* corpus (Common Voice) — for
    monitoring overfitting / picking the best checkpoint. The MEXA benchmark
    (Omnilingual, a different corpus) remains the untouched final test."""
    from collections import defaultdict

    from datasets import Dataset
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["language"]].append(r)
    if max_samples:
        per = max(1, max_samples // max(1, len(by)))
        by = {k: v[:per] for k, v in by.items()}

    train, val = [], []
    for v in by.values():
        val += v[:val_per_lang]           # held-out validation slice
        train += v[val_per_lang:]         # everything else trains
    print(f"train clips: {len(train)} | val clips: {len(val)}", flush=True)
    mk = lambda rs: Dataset.from_list([{"audio": r["audio"], "text": r["text"]} for r in rs])
    return mk(train), mk(val)


def build_indomain_dataset(max_train=0, dev_cap=0, train_path=None):
    """In-domain datasets from the segmented Omnilingual manifests (the 24 BENCHMARK
    languages, force-aligned to <=30 s — see scripts/prep_indomain.py). Train = Omni
    train segments (or train_path for best-of-K self-distillation), val = Omni *dev*
    segments with REAL references (honest validation). max_train caps train clips
    (balanced across languages); 0 = all."""
    from collections import defaultdict
    from pathlib import Path as _P

    from datasets import Dataset
    tr_path = _P(train_path) if train_path else ROOT / "data" / "train" / "manifest_indomain.jsonl"
    dv_path = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"
    train = [json.loads(l) for l in open(tr_path, encoding="utf-8")]
    val = [json.loads(l) for l in open(dv_path, encoding="utf-8")]

    # Whisper's decoder caps at 448 tokens. These languages are OOV for Whisper's BPE,
    # so dense ~28s segments can tokenize past that and crash the forward. Drop the few
    # over-long segments up front (cheap, audio not touched).
    from transformers import WhisperTokenizer
    wt = WhisperTokenizer.from_pretrained(BASE)
    cap = 440
    keep = lambda rows: [r for r in rows if len(wt(r["text"]).input_ids) <= cap]
    n0t, n0v = len(train), len(val)
    train, val = keep(train), keep(val)
    print(f"dropped over-{cap}-token segs: train {n0t - len(train)}, val {n0v - len(val)}", flush=True)

    if max_train:
        by = defaultdict(list)
        for r in train:
            by[r["language"]].append(r)
        per = max(1, max_train // max(1, len(by)))
        train = [r for v in by.values() for r in v[:per]]
    if dev_cap:
        by = defaultdict(list)
        for r in val:
            by[r["language"]].append(r)
        per = max(1, dev_cap // max(1, len(by)))
        val = [r for v in by.values() for r in v[:per]]

    print(f"in-domain train clips: {len(train)} | val (dev) clips: {len(val)}", flush=True)
    mk = lambda rs: Dataset.from_list([{"audio": r["audio"], "text": r["text"]} for r in rs])
    return mk(train), mk(val)


class Collator:
    """Lazy: decode + log-mel + tokenize per batch (runs in DataLoader workers).
    soundfile decodes mp3 natively (~6ms, no ffmpeg subprocess), so worker
    parallelism is fork-safe."""

    def __init__(self, processor):
        self.fe = processor.feature_extractor
        self.tok = processor.tokenizer

    def __call__(self, batch):
        import librosa
        import soundfile as sf
        feats, labels = [], []
        for item in batch:
            arr, sr = sf.read(item["audio"], dtype="float32")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16000:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            feats.append({"input_features": self.fe(arr, sampling_rate=16000).input_features[0]})
            labels.append({"input_ids": self.tok(item["text"]).input_ids})
        inp = self.fe.pad(feats, return_tensors="pt")
        lab = self.tok.pad(labels, return_tensors="pt")
        labels_t = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels_t[:, 0] == self.tok.bos_token_id).all().item():
            labels_t = labels_t[:, 1:]
        labels_t = labels_t[:, :448]  # safety net: Whisper decoder caps at 448 tokens
        inp["labels"] = labels_t
        return inp


def main():
    import math

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=150)
    # in-domain mode + tunable hyperparameters (defaults reproduce the old run)
    ap.add_argument("--indomain", action="store_true", help="train on Omni in-domain segments")
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0, help="effective LoRA scale -> alpha")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--use-dora", action="store_true")
    ap.add_argument("--target", default="qv", choices=["qv", "all"],
                    help="LoRA target modules: qv (attn q,v only) or all (q,k,v,out,fc1,fc2)")
    ap.add_argument("--full-finetune", action="store_true",
                    help="full fine-tune all params instead of LoRA (use a low lr ~1e-5)")
    ap.add_argument("--train-manifest", default=None, help="custom train manifest (e.g. best-of-K)")
    ap.add_argument("--warmup", type=float, default=0.05, help="warmup_ratio")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stopping patience in eval steps (0 = off)")
    ap.add_argument("--base", default=BASE,
                    help="base model id (e.g. openai/whisper-large-v3 for the 32-layer decoder)")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="label_smoothing_factor (0.1 mitigates overconfident repetition)")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    # alpha from effective scale: scale*sqrt(r) for rsLoRA, scale*r for plain (matches sweep)
    alpha = args.scale * (math.sqrt(args.r) if args.use_rslora else args.r)
    TARGETS = {"qv": ["q_proj", "v_proj"],
               "all": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]}
    print(f"config: full_ft={args.full_finetune} r={args.r} alpha={alpha:.1f} lr={args.lr} "
          f"dropout={args.dropout} rslora={args.use_rslora} dora={args.use_dora} "
          f"target={args.target} indomain={args.indomain}", flush=True)

    from transformers import (AutoModelForSpeechSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        args.base, auto_model=AutoModelForSpeechSeq2Seq, whisper_language="Spanish",
        whisper_task="transcribe", load_in_4bit=False, dtype=None,
        full_finetuning=args.full_finetune)
    if not args.full_finetune:
        model = FastModel.get_peft_model(
            model, r=args.r, target_modules=TARGETS[args.target], lora_alpha=alpha,
            lora_dropout=args.dropout, bias="none",
            # Whisper's long encoder sequence makes activations huge -> gradient
            # checkpointing is REQUIRED (without it, even batch 64 OOMs at 80 GB). It
            # keeps memory low, so we spend the headroom on a big batch instead.
            use_gradient_checkpointing="unsloth",
            use_rslora=args.use_rslora, use_dora=args.use_dora,
            random_state=3407,
            task_type=None)  # ** task_type=None is REQUIRED for Whisper (per Unsloth) **

    # decode config for the fine-tuned model (single "es" bucket)
    model.generation_config.language = "<|es|>"
    model.generation_config.task = "transcribe"
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None

    if args.indomain:
        train_ds, val_ds = build_indomain_dataset(max_train=args.max_samples,
                                                  train_path=args.train_manifest)
    else:
        train_ds, val_ds = build_dataset(args.max_samples)
    collator = Collator(processor)
    tok = processor.tokenizer

    import jiwer
    import numpy as np

    def preprocess_logits_for_metrics(logits, labels):
        # argmax in the eval loop so we don't accumulate huge [N, T, 51866] logits
        lg = logits[0] if isinstance(logits, tuple) else logits
        return lg.argmax(dim=-1)

    def compute_metrics(pred):
        # the trainer pads gathered preds/labels with -100; batch_decode chokes on
        # negatives -> replace with pad_token_id in BOTH before decoding.
        pred_ids = np.where(pred.predictions == -100, tok.pad_token_id, pred.predictions)
        label_ids = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
        pred_str = tok.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tok.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(r, h) for r, h in zip(label_str, pred_str) if r.strip()]
        if not pairs:
            return {"wer": 100.0}
        refs, hyps = zip(*pairs)
        return {"wer": round(jiwer.wer(list(refs), list(hyps)) * 100, 2)}

    outdir.mkdir(parents=True, exist_ok=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(outdir), per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=1,
        learning_rate=args.lr, warmup_ratio=args.warmup, num_train_epochs=args.epochs, bf16=True,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=25, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False,
        report_to="none", remove_unused_columns=False, dataloader_num_workers=8,
        dataloader_prefetch_factor=4, predict_with_generate=False, label_names=["labels"])
    callbacks = []
    if args.patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks)
    trainer.train()

    # save: full-FT writes the model directly; LoRA writes adapter + a merged fp16 copy
    if args.full_finetune:
        model.save_pretrained(str(outdir / "merged"))
        processor.save_pretrained(str(outdir / "merged"))
        print("saved full-finetuned model ->", outdir / "merged")
    else:
        model.save_pretrained(str(outdir / "lora"))
        processor.save_pretrained(str(outdir / "lora"))
        try:
            model.save_pretrained_merged(str(outdir / "merged"), processor, save_method="merged_16bit")
            print("saved merged model ->", outdir / "merged")
        except Exception as e:  # noqa: BLE001
            print("merge save skipped:", type(e).__name__, str(e)[:80])
    print("DONE training ->", outdir)


if __name__ == "__main__":
    main()
