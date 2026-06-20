"""Fine-tune whisper-large-v3-turbo on Mexican Indigenous speech with PEFT LoRA.

The canonical transformers + PEFT path (Unsloth 2026.6.7's Whisper patch is broken
against the transformers it bundles: WhisperDecoder.forward() gets duplicate
'input_ids'). Same LoRA result, robust.

All clips share one Whisper language token ("es", the standard low-resource trick).
The held-out benchmark is a *different* corpus (Omnilingual) so eval is clean.

  python scripts/train_peft.py [--max-samples N] [--epochs E] [--batch B]

Saves the LoRA adapter under models/mexicospeech-v0/lora/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "train" / "manifest.jsonl"
OUTDIR = ROOT / "models" / "mexicospeech-v0"
BASE = "openai/whisper-large-v3-turbo"
LANG = "es"


def build_dataset(max_samples, processor):
    from datasets import Dataset
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    if max_samples:
        from collections import defaultdict
        by = defaultdict(list)
        for r in rows:
            by[r["language"]].append(r)
        per = max(1, max_samples // max(1, len(by)))
        rows = [r for langrows in by.values() for r in langrows[:per]]
    print(f"training clips: {len(rows)}", flush=True)
    ds = Dataset.from_list([{"audio": r["audio"], "text": r["text"]} for r in rows])
    fe, tok = processor.feature_extractor, processor.tokenizer

    def prepare(b):
        # soundfile (libsndfile) decodes mp3 fast (~6ms, no ffmpeg subprocess);
        # resample 32k->16k for Whisper.
        import librosa
        import soundfile as sf
        arr, sr = sf.read(b["audio"], dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        b["input_features"] = fe(arr, sampling_rate=16000).input_features[0]
        b["labels"] = tok(b["text"]).input_ids
        return b

    return ds.map(prepare, remove_columns=ds.column_names, desc="extract features")


class Collator:
    def __init__(self, processor):
        self.p = processor

    def __call__(self, feats):
        inp = self.p.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.p.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.p.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        inp["labels"] = labels
        return inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor)

    processor = WhisperProcessor.from_pretrained(BASE, language=LANG, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = LANG
    model.generation_config.task = "transcribe"

    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "v_proj", "k_proj", "out_proj",
                                      "fc1", "fc2"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False

    ds = build_dataset(args.max_samples, processor)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(OUTDIR), per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1, learning_rate=1e-4, warmup_ratio=0.05,
        num_train_epochs=args.epochs, bf16=True, logging_steps=20, save_strategy="epoch",
        report_to="none", remove_unused_columns=False, dataloader_num_workers=4,
        gradient_checkpointing=True, label_names=["labels"])
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds,
                             data_collator=Collator(processor))
    trainer.train()

    model.save_pretrained(str(OUTDIR / "lora"))
    processor.save_pretrained(str(OUTDIR / "lora"))
    print("DONE training ->", OUTDIR / "lora", flush=True)


if __name__ == "__main__":
    main()
