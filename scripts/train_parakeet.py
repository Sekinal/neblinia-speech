"""Fine-tune NVIDIA Parakeet-CTC (FastConformer encoder) on Mexican Indigenous speech via
Hugging Face Transformers + safetensors. NO NeMo. Head-to-head comparison vs our Whisper.

Parakeet's English subword tokenizer cannot represent these orthographies (every diacritic
and glottal stop becomes <unk>), so we train a SUBWORD (SentencePiece) tokenizer on the
target text and swap in a fresh CTC head, keeping the pretrained FastConformer encoder (the
valuable acoustic part). Subword (not char) is required: the encoder subsamples 8x to
~12.5 frames/sec, and these polysynthetic languages exceed that in chars/sec, so char-level
CTC cannot align. Subwords (~3 chars each) fit the frame budget.

  .venv-unsloth/bin/python scripts/train_parakeet.py [--smoke] [--vocab 512] [--epochs E]
      [--lr 3e-4] [--batch 16] [--base nvidia/parakeet-ctc-0.6b] [--outdir ...]
"""
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train" / "manifest_indomain.jsonl"
DEV = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"


def build_spm(texts, prefix: Path, vocab_size: int):
    """Train a SentencePiece BPE model on the target text (full char coverage). The CTC
    blank is appended as a new index after all SentencePiece pieces."""
    import sentencepiece as spm
    txt = prefix.with_suffix(".txt")
    txt.write_text("\n".join(texts), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(txt), model_prefix=str(prefix), vocab_size=vocab_size,
        character_coverage=1.0, model_type="bpe",
        bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1,
        train_extremely_large_corpus=False, normalization_rule_name="identity")
    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    return sp


def load_audio(path):
    import librosa, soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


@dataclass
class CTCCollator:
    fe: object
    sp: object

    def __call__(self, batch):
        feats, labels = [], []
        for item in batch:
            a = load_audio(item["audio"])
            f = self.fe(a, sampling_rate=16000, return_tensors="pt")
            feats.append({"input_features": f["input_features"][0]})
            ids = self.sp.encode(item["text"], out_type=int)
            labels.append(ids if ids else [0])
        inp = self.fe.pad(feats, return_tensors="pt")
        maxlen = max(len(l) for l in labels)
        lab = torch.full((len(labels), maxlen), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, :len(l)] = torch.tensor(l, dtype=torch.long)
        inp["labels"] = lab
        return inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="nvidia/parakeet-ctc-0.6b")
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--train-manifest", default=None)
    ap.add_argument("--outdir", default=str(ROOT / "models" / "neblinia-parakeet-ctc"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import (ParakeetForCTC, ParakeetFeatureExtractor,
                              Trainer, TrainingArguments)
    from datasets import Dataset
    import jiwer

    tr_path = Path(args.train_manifest) if args.train_manifest else TRAIN
    train_rows = [json.loads(l) for l in open(tr_path, encoding="utf-8")]
    dev_rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
    if args.max_samples:
        train_rows = train_rows[:args.max_samples]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sp = build_spm([r["text"] for r in train_rows], outdir / "spm", args.vocab)
    blank = sp.get_piece_size()              # CTC blank = new index after all pieces
    vocab_size = blank + 1
    print(f"sentencepiece pieces: {blank} + blank = {vocab_size}", flush=True)
    print(f"train {len(train_rows)} | dev {len(dev_rows)}", flush=True)

    fe = ParakeetFeatureExtractor.from_pretrained(args.base)
    model = ParakeetForCTC.from_pretrained(args.base)
    hidden = model.config.encoder_config.hidden_size
    model.ctc_head = nn.Conv1d(hidden, vocab_size, kernel_size=1)
    model.config.vocab_size = vocab_size
    model.config.pad_token_id = blank
    # The bf16 BACKWARD path crashes (CUDA illegal access) because Parakeet's cuda cpp
    # extensions are skipped under torch 2.10 (needs 2.11). fp32 backward is correct and
    # stable, so force fp32 weights and train in fp32 (bf16=False below).
    model.float()
    model.to("cuda")
    print(f"swapped ctc_head -> Conv1d({hidden}, {vocab_size})", flush=True)

    collator = CTCCollator(fe, sp)
    mk = lambda rs: Dataset.from_list([{"audio": r["audio"], "text": r["text"]} for r in rs])
    train_ds, dev_ds = mk(train_rows), mk(dev_rows)

    if args.smoke:
        batch = collator([train_rows[i] for i in range(8)])
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        print("SMOKE loss:", float(out.loss), "logits:", tuple(out.logits.shape), flush=True)
        out.loss.backward()
        print("SMOKE backward OK", flush=True)
        return

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    def ctc_decode(ids):
        out, prev = [], None
        for i in ids:
            i = int(i)
            if i != blank and i != prev and i >= 0:
                out.append(i)
            prev = i
        return sp.decode(out)

    def compute_metrics(pred):
        hyps = [ctc_decode(p) for p in pred.predictions]
        refs = [sp.decode([int(i) for i in lab if int(i) >= 0]) for lab in pred.label_ids]
        pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
        if not pairs:
            return {"wer": 100.0}
        rr, hh = zip(*pairs)
        return {"wer": round(jiwer.wer(list(rr), list(hh)) * 100, 2),
                "cer": round(jiwer.cer(list(rr), list(hh)) * 100, 2)}

    targs = TrainingArguments(
        output_dir=str(outdir), per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=1,
        learning_rate=args.lr, warmup_ratio=args.warmup, num_train_epochs=args.epochs,
        bf16=False, fp16=False, max_grad_norm=1.0,   # fp32: bf16 backward crashes (see above)
        logging_steps=25, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        report_to="none", remove_unused_columns=False, dataloader_num_workers=8,
        dataloader_prefetch_factor=4, label_names=["labels"])
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collator, compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    trainer.train()
    trainer.save_model(str(outdir / "best"))
    fe.save_pretrained(str(outdir / "best"))
    print("DONE ->", outdir, flush=True)


if __name__ == "__main__":
    main()
