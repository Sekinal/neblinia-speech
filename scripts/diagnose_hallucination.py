"""Deep diagnosis of the >100% WER (hallucination) languages in preview-0.2.

Three hypotheses for why xti/nhq/zts blow past 100% WER:
  (D) DATA: forced alignment mis-segmented those langs -> trained on (audio,text)
      mismatches -> the model learned to hallucinate. Signal: weird chars/sec.
  (X) DECODING: model is OK but greedy decoding loops/repeats. Signal: guards fix it.
  (U) UNLEARNED: genuinely too little/hard data; samples are uniform garbage.

  --data   per-language audit: counts, chars/sec (misalignment), align-fail rate
  --infer  transcribe benchmark test clips for target langs, dump ref/hyp + the
           hallucination signals (len ratio, zlib compression ratio, n-gram repeats),
           with DEFAULT decoding vs hallucination GUARDS (repetition_penalty + no_repeat).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("MEXA_SRC", str(ROOT.parent / "mexa-benchmark" / "src")))
DATA = ROOT / "data"
MAT = DATA / "materialized"
BASE = "openai/whisper-large-v3-turbo"
ADAPTER = ROOT / "models" / "neblinia-preview-0.2" / "lora"
BROKEN = ["xti", "nhq", "zts"]
CONTROL = ["tlp", "zoh", "vmp"]          # the good ones (51-58% WER)
TARGETS = BROKEN + CONTROL


def comp_ratio(text):
    """Whisper's repetition signal: len / len(zlib(text)). >2.4 ~ repetitive."""
    b = text.encode("utf-8")
    if not b:
        return 0.0
    return len(b) / max(1, len(zlib.compress(b)))


def max_ngram_repeat(words, n=3):
    """Longest run of an immediately-repeating n-gram (repetition-loop detector)."""
    if len(words) < 2 * n:
        return 0
    best = 0
    for i in range(len(words) - n):
        run = 1
        j = i
        while j + n < len(words) and words[j:j + n] == words[j + n:j + 2 * n]:
            run += 1; j += n
        best = max(best, run)
    return best


def audit_data():
    print("=== DATA AUDIT: per-language segments + chars/sec (misalignment signal) ===")
    import soundfile as sf
    rows = [json.loads(l) for l in open(DATA / "train" / "manifest_indomain.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["language"]].append(r)
    # align-fail counts per lang from the prep log
    fails = defaultdict(int)
    plog = ROOT / "prep_gpu.log"
    if plog.exists():
        for line in open(plog, errors="ignore"):
            if "align fail" in line or "fail" in line.lower():
                for t in TARGETS:
                    if f"/{t}/" in line or f"omni_{t}" in line:
                        fails[t] += 1
    print(f"{'lang':5} {'segs':>6} {'cps_med':>8} {'cps_p95':>8} {'short<2s':>9}  tag")
    for lang in TARGETS:
        rs = by.get(lang, [])
        cps = []           # chars per second
        short = 0
        import random
        random.seed(0)
        for r in random.sample(rs, min(120, len(rs))):
            try:
                d = sf.info(r["audio"]).duration
                if d < 2.0:
                    short += 1
                if d > 0:
                    cps.append(len(r["text"]) / d)
            except Exception:
                pass
        cps.sort()
        med = cps[len(cps) // 2] if cps else 0
        p95 = cps[int(len(cps) * 0.95)] if cps else 0
        tag = "BROKEN" if lang in BROKEN else "ok"
        print(f"{lang:5} {len(rs):>6} {med:>8.1f} {p95:>8.1f} {short:>9}  {tag}")
    print("\n(Normal speech ~10-20 chars/sec. Very high cps or many <2s clips with long "
          "text => mis-alignment. Compare BROKEN vs ok rows.)")


def run_infer(limit):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor, pipeline

    from mexa.evaluate import load_test
    from mexa.normalize import normalize
    import jiwer
    import soundfile as sf

    print("merging preview-0.2 adapter...", flush=True)
    base = AutoModelForSpeechSeq2Seq.from_pretrained(BASE, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload()
    proc = WhisperProcessor.from_pretrained(str(ADAPTER), language="es", task="transcribe")
    proc.tokenizer.padding_side = "right"; proc.feature_extractor.padding_side = "right"
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                    feature_extractor=proc.feature_extractor, device=0, torch_dtype=torch.float16,
                    chunk_length_s=30, batch_size=16)

    rows = [r for r in load_test() if r["language"] in TARGETS]
    if limit:
        capped = defaultdict(int); keep = []
        for r in rows:
            if capped[r["language"]] < limit:
                keep.append(r); capped[r["language"]] += 1
        rows = keep
    arrs = {}
    for r in rows:
        a, sr = sf.read(r["audio_ref"][1], dtype="float32")
        arrs[r["audio_ref"][1]] = a.mean(axis=1) if a.ndim > 1 else a

    DECODES = {
        "default":   {"language": "es", "task": "transcribe"},
        "mild":      {"language": "es", "task": "transcribe", "repetition_penalty": 1.2,
                      "no_repeat_ngram_size": 4},
        "med":       {"language": "es", "task": "transcribe", "repetition_penalty": 1.3,
                      "no_repeat_ngram_size": 3},
        "strong":    {"language": "es", "task": "transcribe", "repetition_penalty": 1.5,
                      "no_repeat_ngram_size": 3},
    }
    for name, gen in DECODES.items():
        print(f"\n######## DECODING = {name}  ({gen}) ########", flush=True)
        per = defaultdict(lambda: {"ref": [], "hyp": [], "lr": [], "cr": [], "rep": []})
        examples = defaultdict(list)
        for r in rows:
            res = pipe({"raw": arrs[r["audio_ref"][1]], "sampling_rate": 16000},
                       generate_kwargs=gen)
            ref = normalize(r["raw_text"]); hyp = normalize(res.get("text", ""))
            if not ref:
                continue
            rw, hw = ref.split(), hyp.split()
            d = per[r["language"]]
            d["ref"].append(ref); d["hyp"].append(hyp)
            d["lr"].append(len(hw) / max(1, len(rw)))
            d["cr"].append(comp_ratio(hyp))
            d["rep"].append(max_ngram_repeat(hw))
            if len(examples[r["language"]]) < 2:
                examples[r["language"]].append((ref, hyp))
        print(f"{'lang':5} {'WER':>7} {'len_ratio':>9} {'comp_r':>7} {'max_rep':>7}  tag")
        for lang in TARGETS:
            d = per.get(lang)
            if not d or not d["ref"]:
                continue
            wer = round(jiwer.wer(d["ref"], d["hyp"]) * 100, 1)
            lr = sum(d["lr"]) / len(d["lr"]); cr = sum(d["cr"]) / len(d["cr"])
            rep = max(d["rep"])
            tag = "BROKEN" if lang in BROKEN else "ok"
            print(f"{lang:5} {wer:>7} {lr:>9.2f} {cr:>7.2f} {rep:>7}  {tag}")
        print("\n--- sample ref/hyp (broken langs) ---")
        for lang in BROKEN:
            for ref, hyp in examples.get(lang, [])[:1]:
                print(f"[{lang}] REF: {ref[:140]}")
                print(f"[{lang}] HYP: {hyp[:200]}")


def run_fallback(limit):
    """Whisper's native long-form decoding with temperature fallback: retries a segment
    at higher temperature ONLY when compression_ratio/avg_logprob flags a hallucination,
    so healthy segments are untouched (unlike the blunt global guards)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor

    from mexa.evaluate import load_test
    from mexa.normalize import normalize
    import jiwer
    import soundfile as sf

    print("merging preview-0.2 adapter (fallback decode)...", flush=True)
    base = AutoModelForSpeechSeq2Seq.from_pretrained(BASE, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(ADAPTER)).merge_and_unload().to("cuda").eval()
    model.generation_config.forced_decoder_ids = None
    proc = WhisperProcessor.from_pretrained(str(ADAPTER), language="es", task="transcribe")

    rows = [r for r in load_test() if r["language"] in TARGETS]
    if limit:
        capped = defaultdict(int); keep = []
        for r in rows:
            if capped[r["language"]] < limit:
                keep.append(r); capped[r["language"]] += 1
        rows = keep

    per = defaultdict(lambda: {"ref": [], "hyp": [], "rep": []})
    for r in rows:
        a, sr = sf.read(r["audio_ref"][1], dtype="float32")
        a = a.mean(axis=1) if a.ndim > 1 else a
        inputs = proc(a, sampling_rate=16000, return_tensors="pt", truncation=False,
                      return_attention_mask=True)
        inputs = {k: v.to("cuda", torch.float16 if v.dtype == torch.float32 else v.dtype)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            gen = model.generate(**inputs, language="es", task="transcribe",
                                 condition_on_prev_tokens=False,
                                 compression_ratio_threshold=2.0, logprob_threshold=-1.0,
                                 no_speech_threshold=0.6,
                                 temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                 return_timestamps=True)
        hyp = normalize(proc.batch_decode(gen, skip_special_tokens=True)[0])
        ref = normalize(r["raw_text"])
        if not ref:
            continue
        per[r["language"]]["ref"].append(ref); per[r["language"]]["hyp"].append(hyp)
        per[r["language"]]["rep"].append(max_ngram_repeat(hyp.split()))
    print(f"\n######## DECODING = fallback (compression+logprob+temp fallback) ########")
    print(f"{'lang':5} {'WER':>7} {'max_rep':>7}  tag")
    for lang in TARGETS:
        d = per.get(lang)
        if not d or not d["ref"]:
            continue
        wer = round(jiwer.wer(d["ref"], d["hyp"]) * 100, 1)
        tag = "BROKEN" if lang in BROKEN else "ok"
        print(f"{lang:5} {wer:>7} {max(d['rep']):>7}  {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--limit", type=int, default=15, help="clips per lang")
    args = ap.parse_args()
    if args.data:
        audit_data()
    if args.infer:
        run_infer(args.limit)
    if args.fallback:
        run_fallback(args.limit)


if __name__ == "__main__":
    main()
