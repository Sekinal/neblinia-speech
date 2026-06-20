"""GPU forced-alignment segmenter (fast path). Same job as prep_indomain.py — cut the
~60 s Omnilingual clips into <=28 s chunks with aligned text — but on the GPU via
torchaudio's MMS_FA forced-aligner, fp16, with BATCHED emissions for high GPU util.

Why a second script: the ONNX aligner (prep_indomain.py) is CPU-only here
(onnxruntime-gpu links CUDA 13, unimportable on this cu128 box). torchaudio's MMS_FA
runs the same MMS alignment model on CUDA — ~20-50x faster.

Original orthography is preserved: we romanize each word (unidecode) only to pick
alignment tokens, then map the resulting word spans back onto the ORIGINAL words.

  .venv-align/bin/python scripts/prep_indomain_gpu.py [--splits train,dev] [--batch 16] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from torchaudio.pipelines import MMS_FA as BUNDLE
from unidecode import unidecode

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAT = DATA / "materialized"
OUT_AUDIO = DATA / "train_indomain"
SR = 16_000
MAX_DUR = 28.0
PAD = 0.10
DEVICE = "cuda"

DICT = BUNDLE.get_dict()                  # char -> token id (MMS_FA vocab, blank=0)
MODEL = BUNDLE.get_model(with_star=False).to(DEVICE).eval()


def clean_word(w: str) -> str:
    """Romanize to the MMS_FA latin vocab; keep only in-vocab chars."""
    r = unidecode(w).lower()
    return "".join(c for c in r if c in DICT and DICT[c] != 0)


def unflatten(seq, lengths):
    i, out = 0, []
    for l in lengths:
        out.append(seq[i:i + l]); i += l
    return out


@torch.inference_mode()
def align_batch(waves):
    """waves: list of 1D float32 tensors (16 kHz). Returns list of emissions (on cpu)
    and the per-item frame counts. Batched forward for GPU throughput."""
    lengths = [w.numel() for w in waves]
    maxlen = max(lengths)
    batch = torch.zeros(len(waves), maxlen, dtype=torch.float32)
    for i, w in enumerate(waves):
        batch[i, :w.numel()] = w
    batch = batch.to(DEVICE)
    emissions, _ = MODEL(batch)                       # [B, T, C]
    T = emissions.size(1)
    emissions = emissions.float().log_softmax(-1).cpu()
    frames = [max(1, round(T * (l / maxlen))) for l in lengths]
    return emissions, frames


def align_clip(emission, n_frames, n_samples, orig_words):
    """Forced-align one clip; return [{start,end,text}] in ORIGINAL orthography."""
    pairs = [(o, clean_word(o)) for o in orig_words]
    pairs = [(o, c) for o, c in pairs if c]
    if not pairs:
        return None
    o_words, c_words = zip(*pairs)
    tokens = [DICT[ch] for w in c_words for ch in w]
    if not tokens:
        return None
    emission = emission[:n_frames].unsqueeze(0)        # [1, t, C]
    targets = torch.tensor([tokens], dtype=torch.int32)
    aligned, scores = AF.forced_align(emission, targets, blank=0)
    spans = AF.merge_tokens(aligned[0], scores[0].exp())
    word_spans = unflatten(spans, [len(w) for w in c_words])
    ratio = n_samples / n_frames
    out = []
    for ow, ws in zip(o_words, word_spans):
        out.append({"start": ws[0].start * ratio / SR,
                    "end": ws[-1].end * ratio / SR, "text": ow})
    return out


def chunk_words(wts, max_dur):
    chunks, cur = [], []
    for w in wts:
        if w["end"] <= w["start"]:
            continue
        if cur and (w["end"] - cur[0]["start"] > max_dur):
            chunks.append(cur); cur = [w]
        else:
            cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    print("device:", DEVICE, "| cuda:", torch.cuda.is_available(), flush=True)

    for split in args.splits.split(","):
        manifest = (DATA / "train" / ("manifest_indomain.jsonl" if split == "train"
                                      else f"manifest_indomain_{split}.jsonl"))
        manifest.parent.mkdir(parents=True, exist_ok=True)
        # gather every clip for the split first (so we can batch)
        items = []
        for tsv in sorted(glob.glob(str(MAT / f"omni_*/{split}.tsv"))):
            iso = Path(tsv).parent.name.split("_", 1)[1]
            rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
            if args.limit:
                rows = rows[:args.limit]
            for row in rows:
                src = MAT / f"omni_{iso}" / split / f"{row['uid']}.wav"
                if row["text"].strip() and src.exists():
                    items.append((iso, row["uid"], str(src), row["text"].strip()))
        print(f"{split}: {len(items)} clips", flush=True)

        n_pass = n_seg = n_fail = 0
        with open(manifest, "w", encoding="utf-8") as out:
            # sort by audio length so batches are length-homogeneous (less padding waste)
            metas = []
            for iso, uid, src, text in items:
                info = sf.info(src)
                metas.append((info.frames / info.samplerate, iso, uid, src, text))
            metas.sort()

            long_items = [m for m in metas if m[0] > 30.0]
            for (dur, iso, uid, src, text) in (m for m in metas if m[0] <= 30.0):
                out.write(json.dumps({"audio": src, "text": text, "language": iso,
                                      "split": split}, ensure_ascii=False) + "\n")
                n_pass += 1

            for b in range(0, len(long_items), args.batch):
                batch = long_items[b:b + args.batch]
                waves, raws = [], []
                for (dur, iso, uid, src, text) in batch:
                    a, _ = sf.read(src, dtype="float32")
                    if a.ndim > 1:
                        a = a.mean(axis=1)
                    raws.append(a)
                    waves.append(torch.from_numpy(a))
                try:
                    emissions, frames = align_batch(waves)
                except Exception as e:  # noqa: BLE001
                    n_fail += len(batch)
                    print(f"  batch fail @{b}: {type(e).__name__} {str(e)[:60]}", flush=True)
                    continue
                for j, (dur, iso, uid, src, text) in enumerate(batch):
                    wav_full = raws[j]
                    try:
                        wts = align_clip(emissions[j], frames[j], len(wav_full), text.split())
                    except Exception as e:  # noqa: BLE001
                        wts = None
                    if not wts:
                        n_fail += 1
                        continue
                    seg_dir = OUT_AUDIO / f"omni_{iso}"
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    for k, g in enumerate(chunk_words(wts, MAX_DUR)):
                        a0 = max(0.0, g[0]["start"] - PAD)
                        b0 = min(len(wav_full) / SR, g[-1]["end"] + PAD)
                        seg = wav_full[int(a0 * SR):int(b0 * SR)]
                        if len(seg) < SR * 0.2:
                            continue
                        seg_text = " ".join(w["text"] for w in g).strip()
                        if not seg_text:
                            continue
                        spath = seg_dir / f"{uid}_{k}.wav"
                        sf.write(str(spath), seg, SR)
                        out.write(json.dumps({"audio": str(spath), "text": seg_text,
                                              "language": iso, "split": split},
                                             ensure_ascii=False) + "\n")
                        n_seg += 1
                if (b // args.batch) % 20 == 0:
                    print(f"  {split}: {b + len(batch)}/{len(long_items)} long clips, "
                          f"{n_seg} segs, {n_fail} fails", flush=True)
        print(f"=== {split}: {n_pass} whole + {n_seg} segments ({n_fail} fails) -> {manifest.name}",
              flush=True)


if __name__ == "__main__":
    main()
