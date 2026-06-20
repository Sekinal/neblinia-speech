"""Build the in-domain training data: segment long Omnilingual clips into <=30s
chunks via forced alignment (ctc-forced-aligner, ONNX MMS), keeping text aligned.

Omnilingual clips average ~60s but Whisper's window is 30s; feeding them whole
truncates audio while keeping the full transcript (label/audio mismatch). We force-
align each transcript to its audio, then cut on the silences between words into
<=MAX_DUR chunks with the matching text slice. Clips already <=30s pass through whole.

Output:
  data/train_indomain/<source_key>/<uid>_<k>.wav      segmented audio (16 kHz mono)
  data/train/manifest_indomain.jsonl                  {audio, text, language, split}
  data/train/manifest_indomain_dev.jsonl              in-domain validation segments

  python scripts/prep_indomain.py [--splits train,dev] [--limit N] [--provider cpu|cuda]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAT = DATA / "materialized"
OUT_AUDIO = DATA / "train_indomain"
SR = 16_000
MAX_DUR = 28.0   # target max chunk seconds (margin under Whisper's 30 s)
PAD = 0.10       # seconds of audio padding around a chunk


def chunk_words(wts, max_dur):
    """Greedily pack aligned words into <=max_dur groups, splitting between words."""
    chunks, cur = [], []
    for w in wts:
        if w["end"] <= w["start"]:
            continue
        if cur and (w["end"] - cur[0]["start"] > max_dur):
            chunks.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def boundary(prev_end, next_start):
    """Audio cut point = midpoint of the silence gap between two words."""
    return (prev_end + next_start) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,dev")
    ap.add_argument("--limit", type=int, default=0, help="cap clips per split (0=all)")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--configs", default="", help="comma iso list to process (for sharding); default all")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime intra-op threads (0=default)")
    ap.add_argument("--shard", default="", help="shard id -> writes shards/<manifest>.<shard>.jsonl")
    args = ap.parse_args()

    import onnxruntime
    import ctc_forced_aligner as c

    mp = os.path.expanduser("~/ctc_forced_aligner/model.onnx")
    c.ensure_onnx_model(mp, c.MODEL_URL)
    provs = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda"
             else ["CPUExecutionProvider"])
    so = onnxruntime.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads  # cap threads so parallel shards don't thrash
    sess = onnxruntime.InferenceSession(mp, sess_options=so, providers=provs)
    print("ONNX providers:", sess.get_providers(), flush=True)
    tok = c.Tokenizer()

    want = set(args.configs.split(",")) if args.configs else None
    for split in args.splits.split(","):
        base = "manifest_indomain.jsonl" if split == "train" else f"manifest_indomain_{split}.jsonl"
        if args.shard:
            manifest = DATA / "train" / "shards" / f"{base}.{args.shard}.jsonl"
        else:
            manifest = DATA / "train" / base
        manifest.parent.mkdir(parents=True, exist_ok=True)
        tsvs = sorted(glob.glob(str(MAT / f"omni_*/{split}.tsv")))
        if want:
            tsvs = [t for t in tsvs if Path(t).parent.name.split("_", 1)[1] in want]
        n_clip = n_seg = n_pass = n_fail = 0
        with open(manifest, "w", encoding="utf-8") as out:
            for tsv in tsvs:
                iso = Path(tsv).parent.name.split("_", 1)[1]
                rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
                if args.limit:
                    rows = rows[:args.limit]
                seg_dir = OUT_AUDIO / f"omni_{iso}"
                seg_dir.mkdir(parents=True, exist_ok=True)
                for row in rows:
                    n_clip += 1
                    src = MAT / f"omni_{iso}" / split / f"{row['uid']}.wav"
                    text = row["text"].strip()
                    try:
                        dur = float(row["duration"])
                    except ValueError:
                        dur = 0.0
                    if not text or not src.exists():
                        continue
                    # short clips: pass through whole, no alignment needed
                    if dur <= 30.0:
                        out.write(json.dumps({"audio": str(src), "text": text,
                                              "language": iso, "split": split},
                                             ensure_ascii=False) + "\n")
                        n_pass += 1
                        continue
                    try:
                        audio = c.load_audio(str(src), ret_type="np")
                        emis, stride = c.generate_emissions(sess, audio, batch_size=8)
                        toks, texts = c.preprocess_text(text, romanize=True, language=iso,
                                                        split_size="word")
                        segs, scores, blank = c.get_alignments(emis, toks, tok)
                        spans = c.get_spans(toks, segs, blank)
                        wts = c.postprocess_results(texts, spans, stride, scores)
                        wts = [w for w in wts if w["text"].strip()]
                    except Exception as e:  # noqa: BLE001
                        n_fail += 1
                        print(f"  align fail {iso}/{row['uid'][:8]}: {type(e).__name__} {str(e)[:60]}",
                              flush=True)
                        continue
                    groups = chunk_words(wts, MAX_DUR)
                    wav_full, _ = sf.read(str(src), dtype="float32")
                    if wav_full.ndim > 1:
                        wav_full = wav_full.mean(axis=1)
                    for k, g in enumerate(groups):
                        a = max(0.0, g[0]["start"] - PAD)
                        b = min(len(wav_full) / SR, g[-1]["end"] + PAD)
                        seg = wav_full[int(a * SR):int(b * SR)]
                        if len(seg) < SR * 0.2:        # drop <0.2s fragments
                            continue
                        seg_text = " ".join(w["text"] for w in g).strip()
                        if not seg_text:
                            continue
                        spath = seg_dir / f"{row['uid']}_{k}.wav"
                        sf.write(str(spath), seg, SR)
                        out.write(json.dumps({"audio": str(spath), "text": seg_text,
                                              "language": iso, "split": split},
                                             ensure_ascii=False) + "\n")
                        n_seg += 1
                print(f"  {split}/omni_{iso}: {len(rows)} clips done "
                      f"(passthrough {n_pass}, segs {n_seg}, fails {n_fail})", flush=True)
        print(f"=== {split}: {n_clip} clips -> {n_pass} whole + {n_seg} segments "
              f"({n_fail} align fails) -> {manifest.name}", flush=True)


if __name__ == "__main__":
    main()
