"""Isolate the Parakeet-CTC backward crash: try fp32 vs bf16-autocast, and gradient
checkpointing, over several real batches. Prints which step (if any) crashes."""
import json, sys
from pathlib import Path
import torch, torch.nn as nn
import sentencepiece as spm
from transformers import ParakeetForCTC, ParakeetFeatureExtractor
import soundfile as sf, librosa

MODE = sys.argv[1] if len(sys.argv) > 1 else "fp32"   # fp32 | bf16 | gradckpt
ROOT = Path("/root/neblinia-asr/mexicospeech-training")
rows = [json.loads(l) for l in open("/root/foundational_asr/data/train/manifest_indomain.jsonl")][:200]

prefix = Path("/tmp/pk_dbg2/spm"); prefix.parent.mkdir(parents=True, exist_ok=True)
prefix.with_suffix(".txt").write_text("\n".join(r["text"] for r in rows), encoding="utf-8")
spm.SentencePieceTrainer.train(input=str(prefix)+".txt", model_prefix=str(prefix), vocab_size=512,
    character_coverage=1.0, model_type="bpe", bos_id=-1, eos_id=-1, unk_id=0, pad_id=-1,
    normalization_rule_name="identity")
sp = spm.SentencePieceProcessor(model_file=str(prefix)+".model")
blank = sp.get_piece_size(); V = blank + 1

fe = ParakeetFeatureExtractor.from_pretrained("nvidia/parakeet-ctc-0.6b")
model = ParakeetForCTC.from_pretrained("nvidia/parakeet-ctc-0.6b")
model.ctc_head = nn.Conv1d(model.config.encoder_config.hidden_size, V, 1)
model.config.vocab_size = V; model.config.pad_token_id = blank
if MODE == "fp32":
    model.float()                      # force all weights to fp32 (checkpoint is bf16)
if MODE == "gradckpt":
    model.gradient_checkpointing_enable()
model.to("cuda").train()
print(f"MODE={MODE} vocab={V}", flush=True)

def load(p):
    a, srr = sf.read(p, dtype="float32")
    if a.ndim > 1: a = a.mean(1)
    if srr != 16000: a = librosa.resample(a, orig_sr=srr, target_sr=16000)
    return a

def collate(items):
    feats, labels = [], []
    for it in items:
        f = fe(load(it["audio"]), sampling_rate=16000, return_tensors="pt")
        feats.append({"input_features": f["input_features"][0]})
        labels.append(sp.encode(it["text"], out_type=int) or [0])
    inp = fe.pad(feats, return_tensors="pt")
    m = max(len(l) for l in labels)
    lab = torch.full((len(labels), m), -100, dtype=torch.long)
    for i, l in enumerate(labels): lab[i, :len(l)] = torch.tensor(l)
    inp["labels"] = lab
    return inp

opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for step in range(12):
    b = collate(rows[step*4:(step+1)*4])
    b = {k: v.to("cuda") for k, v in b.items()}
    try:
        if MODE == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**b)
        else:
            out = model(**b)
        out.loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        print(f"step {step} loss {float(out.loss):.3f} T={out.logits.shape[1]} lab={b['labels'].shape[1]} OK", flush=True)
    except Exception as e:
        print(f"step {step} CRASH: {type(e).__name__}: {str(e)[:120]} (T={b['input_features'].shape})", flush=True)
        raise
print("ALL STEPS OK", flush=True)
