"""Evaluate the fine-tuned MexicoSpeech model on the held-out MEXA benchmark.

Merges the LoRA adapter into whisper-large-v3-turbo, runs it over the benchmark test
split (decoding with the "es" token it was trained with), scores WER/CER per language,
and appends a leaderboard row.

Imports the benchmark's eval helpers from the sibling mexa-benchmark repo (set
MEXA_SRC, default ../mexa-benchmark/src).

  python scripts/eval_model.py [--limit N]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEXA_SRC = os.environ.get("MEXA_SRC", str(ROOT.parent / "mexa-benchmark" / "src"))
sys.path.insert(0, MEXA_SRC)
import jiwer  # noqa: E402
from mexa.evaluate import load_test  # noqa: E402
from mexa.normalize import normalize  # noqa: E402

ADAPTER = ROOT / "models" / "mexicospeech-v0" / "lora"
BASE = "openai/whisper-large-v3-turbo"
BENCH = ROOT / "data" / "benchmark"
# Preview iterations: 0.1 = first LoRA prototype, bump for each v2-sweep winner -> ... -> v1.0
PREVIEW_VERSION = "0.1"
HF_REPO = "Thermostatic/NeblinIA-Speech-preview"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adapter", default=str(ADAPTER), help="LoRA adapter dir to eval")
    ap.add_argument("--version", default=PREVIEW_VERSION, help="preview version label")
    ap.add_argument("--tag", default="whisper-large-v3-turbo+LoRA", help="arch tag in model_id")
    args = ap.parse_args()
    adapter = args.adapter
    model_id = f"NeblinIA-Speech (preview-{args.version}) — {args.tag}"

    import torch
    from peft import PeftModel
    from transformers import (AutoModelForSpeechSeq2Seq, WhisperProcessor, pipeline)

    print(f"merging adapter into base... ({adapter})", flush=True)
    base = AutoModelForSpeechSeq2Seq.from_pretrained(BASE, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter))
    model = model.merge_and_unload()
    processor = WhisperProcessor.from_pretrained(str(adapter), language="es", task="transcribe")
    # the HF ASR pipeline's batch collator requires tokenizer + feature extractor to
    # agree on padding side; Whisper's feature extractor pads right, so force the tokenizer.
    processor.tokenizer.padding_side = "right"
    processor.feature_extractor.padding_side = "right"

    pipe = pipeline("automatic-speech-recognition", model=model,
                    tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
                    device=0, torch_dtype=torch.float16, chunk_length_s=30, batch_size=24)
    # Mild anti-repetition guards: low-resource langs occasionally fall into decoder
    # repetition loops (-> WER >100%). repetition_penalty 1.2 + no_repeat_ngram 4 breaks
    # them with ZERO regression on healthy langs (diagnosed: every lang improves or holds;
    # stronger guards and Whisper's temp-fallback both hurt the good langs). See docs.
    gen = {"language": "es", "task": "transcribe",
           "repetition_penalty": 1.2, "no_repeat_ngram_size": 4}

    rows = load_test()
    if args.limit:
        rows = rows[:args.limit]
    print(f"benchmark test clips: {len(rows)}", flush=True)

    # Decode with soundfile and feed the pipeline pre-decoded arrays via a generator:
    # avoids transformers' (broken here) torchcodec backend AND holding all audio in RAM.
    import soundfile as sf

    def audio_gen():
        for r in rows:
            a, sr = sf.read(r["audio_ref"][1], dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            yield {"raw": a, "sampling_rate": sr}

    by = defaultdict(lambda: {"ref": [], "hyp": []})
    for r, res in zip(rows, pipe(audio_gen(), generate_kwargs=gen, batch_size=24)):
        ref = normalize(r["raw_text"])
        if not ref:
            continue
        by[r["language"]]["ref"].append(ref)
        by[r["language"]]["hyp"].append(normalize(res.get("text", "")))
    per, ar, ah = {}, [], []
    for lang, d in sorted(by.items()):
        per[lang] = {"n": len(d["ref"]),
                     "wer": round(jiwer.wer(d["ref"], d["hyp"]) * 100, 2),
                     "cer": round(jiwer.cer(d["ref"], d["hyp"]) * 100, 2)}
        ar += d["ref"]; ah += d["hyp"]
    version = json.loads((BENCH / "benchmark.json").read_text()).get("benchmark_version", "")
    row = {"model_id": model_id, "kind": "ours", "submitter": "neblinia",
           "date": datetime.date.today().isoformat(), "per_language": per,
           "overall_wer": round(jiwer.wer(ar, ah) * 100, 2) if ar else None,
           "overall_cer": round(jiwer.cer(ar, ah) * 100, 2) if ar else None,
           "benchmark_version": version, "eval_n": sum(v["n"] for v in per.values()),
           "params_m": 809, "arch": "enc-dec transformer + LoRA", "family": "NeblinIA-Speech",
           "scope": "MX indigenous + Spanish (fine-tuned)"}
    seed = BENCH / "leaderboard_seed.jsonl"
    out = [json.loads(l) for l in open(seed)] if seed.exists() else []
    out = [r for r in out if r["model_id"] != model_id] + [row]
    with open(seed, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n{model_id}: overall WER {row['overall_wer']} CER {row['overall_cer']}", flush=True)
    print("per-language WER:", {k: v["wer"] for k, v in per.items()}, flush=True)


if __name__ == "__main__":
    main()
