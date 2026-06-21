"""Diagnose ByT5 speech-LLM: load best checkpoint, generate a few dev clips, print hyp vs ref
(the true read past the rambler-inflated aggregate WER)."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_byt5_speech import SpeechByT5, load_audio, WHISPER
from transformers import WhisperFeatureExtractor, AutoTokenizer

ck = torch.load("/root/neblinia-asr/mexicospeech-training/models/neblinia-byt5-speech/best.pt", map_location="cpu")
model = SpeechByT5(ck["byt5_id"])
model.proj.load_state_dict(ck["proj"])
model.byt5.load_state_dict(ck["byt5"])
model = model.to("cuda").eval()
fe = WhisperFeatureExtractor.from_pretrained(WHISPER)
tok = AutoTokenizer.from_pretrained(ck["byt5_id"])
print("loaded byt5 speech-llm best")

dv = [json.loads(l) for l in open("/root/foundational_asr/data/train/manifest_indomain_dev.jsonl")][:6]
feats = torch.stack([fe(load_audio(r["audio"]), sampling_rate=16000, return_tensors="pt").input_features[0] for r in dv]).to("cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    gen = model.generate(feats, max_new=200)
for r, g in zip(dv, gen):
    h = tok.decode(g, skip_special_tokens=True)
    print(f"\nREF ({len(r['text'])}c): {r['text'][:90]!r}")
    print(f"HYP ({len(h)}c): {h[:90]!r}")
