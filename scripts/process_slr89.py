"""Process SLR89 (Yoloxochitl Mixtec, ISO xty) ELAN .eaf transcriptions into an ASR manifest.
Parses each .eaf for the surface-orthography tier (time-aligned), cuts the linked narrative
wav into 16kHz mono utterance segments, writes manifest_slr89.jsonl compatible with our broad
manifest ({audio, text, language, split}).

  python scripts/process_slr89.py [--max-eaf N] [--dry]
"""
import argparse, json, re, sys, hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path("/root/data_pull/slr89/Yoloxochitl-Mixtec-for-ASR")
TRANS = ROOT / "Transcriptions-for-ASR"
OUT_AUDIO = Path("/root/data_pull/slr89_segments")
OUT_MANIFEST = Path("/root/neblinia-asr/mexicospeech-training/data/train/manifest_slr89.jsonl")
LANG = "xty"
MIN_S, MAX_S = 1.0, 28.0


def parse_eaf(p):
    root = ET.parse(p).getroot()
    ts = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE", 0))
          for t in root.iter("TIME_SLOT") if t.get("TIME_VALUE")}
    media = None
    for md in root.iter("MEDIA_DESCRIPTOR"):
        url = md.get("MEDIA_URL") or md.get("RELATIVE_MEDIA_URL") or ""
        if url.endswith(".wav"):
            media = Path(url.replace("file://", "")).name
            break
    tiers = {}
    for tier in root.iter("TIER"):
        anns = []
        for aa in tier.iter("ALIGNABLE_ANNOTATION"):
            t1, t2 = ts.get(aa.get("TIME_SLOT_REF1")), ts.get(aa.get("TIME_SLOT_REF2"))
            val = (aa.findtext("ANNOTATION_VALUE") or "").strip()
            if t1 is not None and t2 is not None and val:
                anns.append((t1, t2, val))
        if anns:
            tiers[tier.get("TIER_ID")] = anns
    return media, tiers


def pick_surface_tier(tiers):
    # prefer a tier whose name signals surface orthography; else the one with most total text
    for pat in (r"surface|surf|ortog|transcri|practical|palabra", r".*"):
        cand = {k: v for k, v in tiers.items() if re.search(pat, k, re.I)}
        if cand:
            return max(cand.items(), key=lambda kv: sum(len(a[2]) for a in kv[1]))[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eaf", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    import soundfile as sf, librosa
    import numpy as np

    wav_index = {p.name: p for p in ROOT.rglob("*.wav")}
    eafs = sorted(TRANS.rglob("*.eaf"))
    if args.max_eaf:
        eafs = eafs[:args.max_eaf]
    print(f"{len(eafs)} eaf | {len(wav_index)} wav", flush=True)

    OUT_AUDIO.mkdir(parents=True, exist_ok=True)
    n_seg = n_skip_nowav = n_skip_dur = 0
    tier_names = {}
    audio_cache = {}
    rows = []
    for i, ef in enumerate(eafs):
        media, tiers = parse_eaf(ef)
        for k in tiers:
            tier_names[k] = tier_names.get(k, 0) + 1
        anns = pick_surface_tier(tiers)
        if not anns:
            continue
        wav = wav_index.get(media) if media else None
        if wav is None:  # fall back to matching by eaf stem
            stem = ef.stem.split("_ed-")[0]
            wav = next((wav_index[n] for n in wav_index if stem in n), None)
        if wav is None:
            n_skip_nowav += 1
            continue
        if args.dry:
            n_seg += len(anns)
            continue
        if wav not in audio_cache:
            a, sr = sf.read(str(wav), dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            if sr != 16000:
                a = librosa.resample(a, orig_sr=sr, target_sr=16000)
            audio_cache = {wav: (a, 16000)}  # keep only current (memory)
        a, sr = audio_cache[wav]
        for (t1, t2, text) in anns:
            dur = (t2 - t1) / 1000.0
            if dur < MIN_S or dur > MAX_S:
                n_skip_dur += 1
                continue
            seg = a[int(t1 / 1000 * sr):int(t2 / 1000 * sr)]
            if len(seg) < int(MIN_S * sr):
                n_skip_dur += 1
                continue
            h = hashlib.md5(f"{wav.name}{t1}{t2}".encode()).hexdigest()[:16]
            outp = OUT_AUDIO / f"{h}.wav"
            sf.write(str(outp), seg, sr)
            rows.append({"audio": str(outp), "text": text, "language": LANG, "split": "train"})
            n_seg += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(eafs)} eaf -> {n_seg} segs", flush=True)

    print(f"TIER NAMES seen: {dict(sorted(tier_names.items(), key=lambda x:-x[1])[:10])}", flush=True)
    if not args.dry:
        OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE: {n_seg} segments | no-wav {n_skip_nowav} | bad-dur {n_skip_dur} -> {OUT_MANIFEST}", flush=True)


if __name__ == "__main__":
    main()
