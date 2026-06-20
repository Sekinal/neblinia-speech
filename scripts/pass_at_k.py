"""Measure Pass@1 (greedy CER) vs Pass@K (best-of-K sampled CER) per language for an
adapter. Tests the VibeThinker SPECTRUM premise: is the correct transcription REACHABLE
by sampling (Pass@K << Pass@1 => rich spectrum, RL/merge can amplify) or thin (the
spectrum genuinely lacks the answer => needs SFT diversity / data)?

  .venv-unsloth/bin/python scripts/pass_at_k.py [--adapter DIR] [--k 8] [--per-lang 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("MEXA_SRC", str(ROOT.parent / "mexa-benchmark" / "src")))
BASE = "openai/whisper-large-v3-turbo"
DEV = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"
DEVICE = "cuda"


def load_audio(p):
    import librosa
    import soundfile as sf
    a, sr = sf.read(p, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "models" / "neblinia-preview-0.2" / "lora"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--per-lang", type=int, default=6)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    import jiwer
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from mexa.normalize import normalize

    proc = WhisperProcessor.from_pretrained(args.adapter, language="es", task="transcribe")
    fe, tok = proc.feature_extractor, proc.tokenizer
    base = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter).to(DEVICE).eval()
    model.generation_config.language = "<|es|>"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["language"]].append(r)
    clips = [r for v in by.values() for r in v[:args.per_lang]]
    print(f"adapter={args.adapter}\nPass@{args.k} over {len(clips)} clips ({len(by)} langs), temp={args.temp}", flush=True)

    p1, pk = defaultdict(list), defaultdict(list)
    for r in clips:
        feat = fe(load_audio(r["audio"]), sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, torch.bfloat16)
        ref = normalize(r["text"])
        if not ref:
            continue
        with torch.inference_mode():
            g = model.generate(input_features=feat, do_sample=False, num_beams=1,
                               max_new_tokens=128, language="es", task="transcribe",
                               pad_token_id=tok.pad_token_id)
            gk = model.generate(input_features=feat, do_sample=True, temperature=args.temp,
                                num_return_sequences=args.k, max_new_tokens=128,
                                language="es", task="transcribe", pad_token_id=tok.pad_token_id)
        cer1 = jiwer.cer(ref, normalize(tok.batch_decode(g, skip_special_tokens=True)[0]))
        cers = [jiwer.cer(ref, normalize(h)) for h in tok.batch_decode(gk, skip_special_tokens=True)]
        p1[r["language"]].append(cer1)
        pk[r["language"]].append(min(cers))

    print(f"\n{'lang':5} {'pass@1':>7} {'pass@k':>7} {'gap':>6}  spectrum")
    a1, ak = [], []
    for lang in sorted(p1):
        m1 = sum(p1[lang]) / len(p1[lang])
        mk = sum(pk[lang]) / len(pk[lang])
        a1 += p1[lang]; ak += pk[lang]
        flag = "RICH" if (m1 - mk) > 0.1 else ("thin" if (m1 - mk) < 0.04 else "")
        print(f"{lang:5} {m1:>7.3f} {mk:>7.3f} {m1-mk:>6.3f}  {flag}")
    o1 = sum(a1) / len(a1); ok = sum(ak) / len(ak)
    print(f"\nOVERALL  pass@1 {o1:.3f}  pass@{args.k} {ok:.3f}  gap {o1-ok:.3f}")
    print("(big gap = greedy misses an answer the model CAN sample -> RL/merge headroom;")
    print(" small gap on broken langs = spectrum lacks the answer -> needs SFT diversity/data)")


if __name__ == "__main__":
    main()
