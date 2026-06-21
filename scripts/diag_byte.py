"""Diagnose byte-whisper: load best checkpoint, greedy-decode a few dev clips, print
hyp vs ref + lengths, and whether output depends on the audio (feed clip A's audio with
clip B to check if it's ignoring the encoder)."""
import json, sys
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_byte_whisper import load_audio, greedy, BOS, EOS, BASE

M = "/root/neblinia-asr/mexicospeech-training/models/neblinia-byte-whisper/best"
fe = WhisperFeatureExtractor.from_pretrained(BASE)
model = WhisperForConditionalGeneration.from_pretrained(M, dtype=torch.float32).to("cuda").eval()
print("loaded", M, "vocab", model.config.vocab_size, "maxpos", model.config.max_target_positions)

dv = [json.loads(l) for l in open("/root/foundational_asr/data/train/manifest_indomain_dev.jsonl")][:4]
feats = torch.stack([fe(load_audio(r["audio"]), sampling_rate=16000, return_tensors="pt").input_features[0] for r in dv]).to("cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    hyps = greedy(model, feats, max_len=400)
for r, h in zip(dv, hyps):
    print(f"\nREF ({len(r['text'])}c): {r['text'][:90]!r}")
    print(f"HYP ({len(h)}c): {h[:90]!r}")
# does it depend on audio? feed the SAME audio (clip 0) for all -> outputs should be identical
with torch.autocast("cuda", dtype=torch.bfloat16):
    same = greedy(model, feats[:1].repeat(2, 1, 1), max_len=400)
print("\nsame-audio check (should be identical):", same[0][:50] == same[1][:50])
