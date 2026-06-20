"""Shared in-domain data plumbing (NO unsloth import) so both the Unsloth trainer and
the vanilla-PEFT method comparison can use the exact same dataset + collator. Keeping it
unsloth-free is what makes the PEFT-method comparison a fair, apples-to-apples test."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "openai/whisper-large-v3-turbo"
TR_PATH = ROOT / "data" / "train" / "manifest_indomain.jsonl"
DV_PATH = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"
TOKEN_CAP = 440   # Whisper decoder caps at 448; drop denser segments up front


def build_indomain_dataset(max_train=0, dev_cap=0):
    """Train = Omni-train segments, val = Omni-dev segments (in-domain). Drops segments
    whose transcript exceeds TOKEN_CAP Whisper tokens. max_train / dev_cap cap clip
    counts (balanced across languages); 0 = all."""
    from datasets import Dataset
    from transformers import WhisperTokenizer

    train = [json.loads(l) for l in open(TR_PATH, encoding="utf-8")]
    val = [json.loads(l) for l in open(DV_PATH, encoding="utf-8")]

    wt = WhisperTokenizer.from_pretrained(BASE)
    keep = lambda rows: [r for r in rows if len(wt(r["text"]).input_ids) <= TOKEN_CAP]
    n0t, n0v = len(train), len(val)
    train, val = keep(train), keep(val)
    print(f"dropped over-{TOKEN_CAP}-token segs: train {n0t - len(train)}, val {n0v - len(val)}",
          flush=True)

    def cap(rows, n):
        if not n:
            return rows
        by = defaultdict(list)
        for r in rows:
            by[r["language"]].append(r)
        per = max(1, n // max(1, len(by)))
        return [r for v in by.values() for r in v[:per]]

    train, val = cap(train, max_train), cap(val, dev_cap)
    print(f"in-domain train clips: {len(train)} | val (dev) clips: {len(val)}", flush=True)
    mk = lambda rs: Dataset.from_list([{"audio": r["audio"], "text": r["text"]} for r in rs])
    return mk(train), mk(val)


class Collator:
    """Lazy decode + log-mel + tokenize per batch (soundfile, fork-safe). Truncates
    labels to 448 (Whisper decoder cap) as a safety net."""

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
        labels_t = labels_t[:, :448]
        inp["labels"] = labels_t
        return inp
