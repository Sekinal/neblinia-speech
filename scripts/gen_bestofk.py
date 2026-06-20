"""Generate best-of-K pseudo-labels from preview-0.2 for self-distillation (RFT/STaR).

Pass@K showed the model can SAMPLE good transcriptions it can't greedy-decode (greedy
loops). So: sample K per clip, keep the lowest-CER sample as the distillation target, then
SFT on (audio, best-sample) -> the rare-but-reachable good output becomes the greedy one.

  .venv-unsloth/bin/python scripts/gen_bestofk.py [--k 8] [--batch 12] [--max-cer 0.8]

Writes data/train/manifest_bestofk.jsonl: {audio, text: best-of-K sample, language, base_cer}.
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
ADAPTER = ROOT / "models" / "neblinia-preview-0.2" / "lora"
MANIFEST = ROOT / "data" / "train" / "manifest_indomain.jsonl"
OUT = ROOT / "data" / "train" / "manifest_bestofk.jsonl"
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
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-cer", type=float, default=0.8, help="drop clips whose best sample is still worse")
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()

    import jiwer
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from mexa.normalize import normalize

    proc = WhisperProcessor.from_pretrained(str(ADAPTER), language="es", task="transcribe")
    fe, tok = proc.feature_extractor, proc.tokenizer
    base = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(ADAPTER)).to(DEVICE).eval()
    model.generation_config.language = "<|es|>"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    if args.max_clips:
        rows = rows[:args.max_clips]
    print(f"best-of-{args.k} over {len(rows)} clips, batch={args.batch}", flush=True)

    K, kept, dropped = args.k, 0, 0
    impr = []
    with open(OUT, "w", encoding="utf-8") as out:
        for b in range(0, len(rows), args.batch):
            batch = rows[b:b + args.batch]
            feats = fe([load_audio(r["audio"]) for r in batch], sampling_rate=16000,
                       return_tensors="pt").input_features.to(DEVICE, torch.bfloat16)
            with torch.inference_mode():
                gk = model.generate(input_features=feats, do_sample=True, temperature=args.temp,
                                    num_return_sequences=K, max_new_tokens=128, language="es",
                                    task="transcribe", pad_token_id=tok.pad_token_id)
            hyps = tok.batch_decode(gk, skip_special_tokens=True)            # [B*K]
            for i, r in enumerate(batch):
                ref = normalize(r["text"])
                cands = [(jiwer.cer(ref, normalize(hyps[i * K + j])) if ref else 1.0, hyps[i * K + j])
                         for j in range(K)]
                cands.sort(key=lambda c: c[0])
                best_cer, best_text = cands[0]
                if not ref or best_cer > args.max_cer or not best_text.strip():
                    dropped += 1
                    continue
                out.write(json.dumps({"audio": r["audio"], "text": best_text.strip(),
                                      "language": r["language"], "best_cer": round(best_cer, 3)},
                                     ensure_ascii=False) + "\n")
                kept += 1
                impr.append(best_cer)
            if (b // args.batch) % 25 == 0:
                mc = sum(impr) / len(impr) if impr else 0
                print(f"  {b + len(batch)}/{len(rows)} | kept {kept} dropped {dropped} | "
                      f"mean best-CER {mc:.3f}", flush=True)
    print(f"DONE: kept {kept}, dropped {dropped} -> {OUT.name} | mean best-CER "
          f"{sum(impr)/max(1,len(impr)):.3f}", flush=True)


if __name__ == "__main__":
    main()
