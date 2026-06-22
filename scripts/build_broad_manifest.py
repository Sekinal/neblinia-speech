"""Build a BROAD training manifest = omni in-domain (the 23 test varieties) + open
Common Voice MX Indigenous languages already on disk under data/mdc/ (CC0). The CV
langs are RELATED-family transfer fuel (more Nahuatl, Zoque, Mazatec, Cuicatec, ...),
trained under the same single "es" bucket. This scales train data ~3x to attack the
looping that wrecks the hard languages.

Dev/eval is UNCHANGED (omni dev, the exact test varieties) — CV langs are train-only.

  python scripts/build_broad_manifest.py --out data/train/manifest_broad.jsonl \
      [--max-per-cv 0] [--include-spa]

CV clips are short single utterances -> no forced alignment, no materialization;
soundfile reads the mp3s directly (the collator resamples 32k->16k).
"""
from __future__ import annotations
import argparse, csv, glob, json, os, random
from pathlib import Path

# portable: set NEBLINIA_DATA to your shared data root (see docs/RECREATE_DATA.md)
ROOT = Path(os.environ.get("NEBLINIA_DATA", "/root/foundational_asr"))
OMNI = ROOT / "data/train/manifest_indomain.jsonl"
MDC = ROOT / "data/mdc"
CV_LANGS = ["cut", "cux", "mau", "ncx", "nlv", "pua", "sei", "tar", "yaq", "zoc"]


def cv_rows(lang, max_per):
    base = glob.glob(str(MDC / lang / "extracted" / "*" / lang))
    if not base:
        print(f"  !! {lang}: no extracted dir"); return []
    d = Path(base[0])
    tsv = d / "validated.tsv"
    clips = d / "clips"
    if not tsv.exists():
        print(f"  !! {lang}: no validated.tsv"); return []
    out = []
    with open(tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sent = (r.get("sentence") or "").strip()
            p = (r.get("path") or "").strip()
            if not sent or not p:
                continue
            ap = clips / p
            out.append({"audio": str(ap), "text": sent, "language": lang})
    random.shuffle(out)
    if max_per:
        out = out[:max_per]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/train/manifest_broad.jsonl")
    ap.add_argument("--max-per-cv", type=int, default=0, help="cap clips per CV lang (0=all)")
    ap.add_argument("--include-spa", action="store_true",
                    help="also add materialized CIEMPIESS Spanish if a manifest exists")
    args = ap.parse_args()
    random.seed(3407)

    rows = [json.loads(l) for l in open(OMNI, encoding="utf-8")]
    print(f"omni in-domain: {len(rows)}")
    for lang in CV_LANGS:
        r = cv_rows(lang, args.max_per_cv)
        print(f"  cv {lang}: +{len(r)}")
        rows += r

    random.shuffle(rows)
    outp = ROOT / args.out if not args.out.startswith("/") else Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(r["language"] for r in rows)
    print(f"\nTOTAL {len(rows)} clips -> {outp}")
    print("per-language:", dict(sorted(c.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
