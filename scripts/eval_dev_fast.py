"""Fast checkpoint triage on the in-domain DEV set (NOT the private test — keep that
clean for the final fair eval). Uses RAW GREEDY decoding (single temperature, no
temp-fallback) so the model's intrinsic LOOPING is exposed rather than masked.

Reports per-language WER/CER + a repetition rate (fraction of hyps that loop) so we
can rank checkpoints in ~1-2 min and only run the full faster-whisper eval on winners.

  run_gpu.sh <mexa_venv_python> eval_dev_fast.py <ct2_dir> [n_per_lang=25]
"""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiwer
import soundfile as sf
from faster_whisper import WhisperModel

CT2 = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25
DEV = Path("/root/foundational_asr/data/train/manifest_indomain_dev.jsonl")
WORKERS = 8


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def max_ngram_repeat(words, n=3):
    """Max number of consecutive repeats of any n-gram (loop detector)."""
    if len(words) < n * 2:
        return 1
    best = 1
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    i = 0
    while i < len(grams):
        c = 1
        while i + c * n < len(grams) and grams[i + c * n] == grams[i]:
            c += 1
        best = max(best, c)
        i += 1
    return best


rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
by = defaultdict(list)
for r in rows:
    by[r["language"]].append(r)
sample = [r for v in by.values() for r in v[:N]]
print(f"dev triage: {len(sample)} clips ({len(by)} langs x {N}) model={CT2}", flush=True)

model = WhisperModel(CT2, device="cuda", compute_type="float16", num_workers=WORKERS)


def work(r):
    audio, _ = sf.read(r["audio"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # raw greedy: single temperature, NO fallback -> intrinsic looping is visible
    segs, _ = model.transcribe(audio, language="es", beam_size=1, temperature=0.0,
                               condition_on_previous_text=False)
    return r, "".join(s.text for s in segs)


with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    res = list(ex.map(work, sample))

by_l = defaultdict(lambda: {"ref": [], "hyp": [], "loops": 0, "n": 0})
for r, hyp in res:
    ref = norm(r["text"])
    if not ref:
        continue
    h = norm(hyp)
    d = by_l[r["language"]]
    d["ref"].append(ref); d["hyp"].append(h); d["n"] += 1
    if max_ngram_repeat(h.split(), 3) >= 3:   # a 3-gram repeated >=3x in a row = loop
        d["loops"] += 1

per, ar, ah, tot_loop, tot_n = {}, [], [], 0, 0
for lang, d in sorted(by_l.items()):
    per[lang] = {"n": d["n"], "wer": round(jiwer.wer(d["ref"], d["hyp"]) * 100, 1),
                 "cer": round(jiwer.cer(d["ref"], d["hyp"]) * 100, 1),
                 "loop%": round(100 * d["loops"] / max(1, d["n"]), 1)}
    ar += d["ref"]; ah += d["hyp"]; tot_loop += d["loops"]; tot_n += d["n"]

ow = round(jiwer.wer(ar, ah) * 100, 2)
oc = round(jiwer.cer(ar, ah) * 100, 2)
lp = round(100 * tot_loop / max(1, tot_n), 2)
print(f"\n=== DEV TRIAGE: WER {ow} | CER {oc} | LOOP {lp}% (n={tot_n}) ===", flush=True)
print("per-lang:", json.dumps(per, ensure_ascii=False), flush=True)
json.dump({"overall_wer": ow, "overall_cer": oc, "loop_pct": lp, "n": tot_n,
           "per_language": per, "model": CT2},
          open("/tmp/dev_triage.json", "w"), ensure_ascii=False, indent=2)
