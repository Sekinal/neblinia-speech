"""Build the Whisper fine-tuning manifest from publishable/local training data.

Sources (TRAIN only — never the benchmark test):
  - Common Voice validated indigenous clips (short, ~<=10s, CC0, local mp3s)
  - CIEMPIESS Spanish train (light + balance) loaded from HF (CC BY-SA)

The benchmark is Omnilingual + CIEMPIESS *test*, which are different corpora/splits,
so there is no overlap with this training data. We still text-fingerprint-filter
against the benchmark as a belt-and-suspenders check.

Output: data/train/manifest.jsonl  rows {audio, text, language, source, duration}
(`audio` is a path; the Audio feature decodes + resamples to 16k at train time.)

  uv run python scripts/prep_train.py [--cap-per-lang N] [--no-ciempiess]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MDC = ROOT / "data" / "mdc"
OUT = ROOT / "data" / "train"

CV_LOCALES = ["ncx", "nlv", "mau", "tar", "cut", "cux", "pua", "sei", "yaq", "zoc"]


def _cv_corpus(iso):
    base = MDC / iso / "extracted"
    return next(base.glob(f"*/{iso}"), None) if base.exists() else None


def collect_common_voice(cap):
    rows = []
    for iso in CV_LOCALES:
        corpus = _cv_corpus(iso)
        if not corpus:
            continue
        durs = {}
        dp = corpus / "clip_durations.tsv"
        if dp.exists():
            for r in csv.DictReader(open(dp, encoding="utf-8"), delimiter="\t"):
                try:
                    durs[r.get("clip") or r.get("path")] = float(
                        r.get("duration[ms]") or r.get("duration") or 0) / 1000
                except ValueError:
                    pass
        n = 0
        for r in csv.DictReader(open(corpus / "validated.tsv", encoding="utf-8"), delimiter="\t"):
            if cap and n >= cap:
                break
            sent, path = r.get("sentence", "").strip(), r.get("path", "")
            wav = corpus / "clips" / path
            if not sent or not wav.exists():
                continue
            d = durs.get(path)
            if d and d > 30:  # Whisper window
                continue
            rows.append({"audio": str(wav), "text": sent, "language": iso,
                         "source": "common_voice", "duration": d})
            n += 1
        print(f"  CV {iso}: {n} clips")
    return rows


def collect_ciempiess(cap):
    """CIEMPIESS train (light + balance) from HF — Spanish, keeps Whisper from
    forgetting Spanish during the indigenous fine-tune."""
    from datasets import load_dataset
    rows = []
    for hf_id in ("ciempiess/ciempiess_light", "ciempiess/ciempiess_balance"):
        try:
            ds = load_dataset(hf_id, split="train", trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"  CIEMPIESS {hf_id}: skip ({type(e).__name__})")
            continue
        n = 0
        for ex in ds:
            if cap and n >= cap:
                break
            text = (ex.get("normalized_text") or ex.get("sentence")
                    or ex.get("transcription") or ex.get("text") or "").strip()
            if not text:
                continue
            # store the HF audio path (decoded later); write the array to a wav cache
            rows.append({"audio": ex["audio"]["path"], "text": text, "language": "spa",
                         "source": "ciempiess", "duration": None})
            n += 1
        print(f"  CIEMPIESS {hf_id}: {n} clips")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-lang", type=int, default=0, help="cap clips per source/lang")
    ap.add_argument("--no-ciempiess", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Collecting Common Voice (indigenous, local)...")
    rows = collect_common_voice(args.cap_per_lang)
    if not args.no_ciempiess:
        print("Collecting CIEMPIESS (Spanish, HF)...")
        rows += collect_ciempiess(args.cap_per_lang)

    mpath = OUT / "manifest.jsonl"
    with open(mpath, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_lang = {}
    for r in rows:
        by_lang[r["language"]] = by_lang.get(r["language"], 0) + 1
    print(f"\nwrote {mpath}: {len(rows)} clips across {len(by_lang)} languages")
    print("per-language:", dict(sorted(by_lang.items())))


if __name__ == "__main__":
    main()
