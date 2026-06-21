"""ByT5 speech-LLM: Whisper-turbo encoder (FROZEN, acoustics) -> linear projector ->
ByT5 byte-native decoder (the byte-LM prior). Tokenizer-free, zero OOV, and unlike
byte-Whisper the decoder is byte-fluent FROM PRETRAINING (no negative transfer). T5 uses
relative position bias -> no 448-token decoder cap, so long byte sequences just work.

  python scripts/train_byt5_speech.py [--smoke] [--steps N] [--byt5 google/byt5-small]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train" / "manifest_indomain.jsonl"
DEV = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"
WHISPER = "openai/whisper-large-v3-turbo"
DEVICE = "cuda"


def load_audio(path):
    import librosa, soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


class SpeechByT5(nn.Module):
    def __init__(self, byt5_id):
        super().__init__()
        from transformers import WhisperForConditionalGeneration, T5ForConditionalGeneration
        # load fp32 (turbo checkpoint is fp16); bf16 autocast handles mixed precision at train
        self.wenc = WhisperForConditionalGeneration.from_pretrained(WHISPER, dtype=torch.float32).model.encoder
        self.byt5 = T5ForConditionalGeneration.from_pretrained(byt5_id, dtype=torch.float32)
        self.proj = nn.Linear(self.wenc.config.d_model, self.byt5.config.d_model)
        for p in self.wenc.parameters():
            p.requires_grad_(False)
        for p in self.byt5.encoder.parameters():           # ByT5 encoder unused (we feed speech)
            p.requires_grad_(False)

    def encode(self, feats):
        with torch.no_grad():
            h = self.wenc(feats).last_hidden_state          # [B, T, dWhisper]
        return self.proj(h)                                 # [B, T, dByt5]

    def forward(self, feats, labels):
        from transformers.modeling_outputs import BaseModelOutput
        mem = self.encode(feats)
        return self.byt5(encoder_outputs=BaseModelOutput(last_hidden_state=mem), labels=labels).loss

    @torch.no_grad()
    def generate(self, feats, max_new=256):
        # NOTE: no_repeat_ngram in HF generate is a slow CPU op that stalls eval over many
        # clips; rely on a bounded max_new instead (a few ramblers, but eval stays fast).
        from transformers.modeling_outputs import BaseModelOutput
        mem = self.encode(feats)
        return self.byt5.generate(encoder_outputs=BaseModelOutput(last_hidden_state=mem),
                                  max_new_tokens=max_new, num_beams=1)


def main():
    import jiwer
    from transformers import WhisperFeatureExtractor, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--byt5", default="google/byt5-small")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--dev-clips", type=int, default=300)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "models" / "neblinia-byt5-speech"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tr = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    dv = [json.loads(l) for l in open(DEV, encoding="utf-8")][:args.dev_clips]
    if args.max_samples:
        tr = tr[:args.max_samples]
    print(f"train {len(tr)} | dev {len(dv)}", flush=True)

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER)
    tok = AutoTokenizer.from_pretrained(args.byt5)

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            r = self.rows[i]
            feat = fe(load_audio(r["audio"]), sampling_rate=16000, return_tensors="pt").input_features[0]
            ids = tok(r["text"], return_tensors="pt").input_ids[0]
            return feat, ids

    def collate(batch):
        feats = torch.stack([b[0] for b in batch])
        L = max(b[1].size(0) for b in batch)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            labels[i, :b[1].size(0)] = b[1]
        return feats, labels

    model = SpeechByT5(args.byt5).to(DEVICE)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ByT5 speech-LLM: dWhisper={model.wenc.config.d_model} dByt5={model.byt5.config.d_model} "
          f"| trainable {ntr/1e6:.0f}M (whisper enc + byt5 enc frozen)", flush=True)

    loader = DataLoader(DS(tr), batch_size=args.batch, shuffle=True, num_workers=8,
                        collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=4)
    # precompute dev features ONCE (a second live DataLoader deadlocks the persistent train
    # workers and stalls the GPU). Eval iterates this cache, no DataLoader.
    print("caching dev features...", flush=True)
    dev_cache = [(fe(load_audio(r["audio"]), sampling_rate=16000, return_tensors="pt").input_features[0],
                  tok(r["text"], return_tensors="pt").input_ids[0]) for r in dv]

    if args.smoke:
        feats, labels = next(iter(loader))
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(feats, labels)
        loss.backward()
        gen = model.generate(feats[:2], max_new=60)
        print(f"SMOKE loss {float(loss):.3f} feats{tuple(feats.shape)} OK | gen:",
              [tok.decode(g, skip_special_tokens=True) for g in gen], flush=True)
        return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    def lr_at(s):
        return min(s / args.warmup, (args.warmup / max(s, 1)) ** 0.5)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def dev_eval():
        model.eval()
        refs, hyps = [], []
        for i in range(0, len(dev_cache), args.batch):
            chunk = dev_cache[i:i + args.batch]
            feats = torch.stack([c[0] for c in chunk]).to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = model.generate(feats)
            for g, c in zip(gen, chunk):
                hyps.append(tok.decode(g, skip_special_tokens=True).lower().strip())
                refs.append(tok.decode(c[1], skip_special_tokens=True).lower().strip())
        model.train()
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        rr, hh = zip(*pairs)
        return jiwer.wer(list(rr), list(hh)) * 100, jiwer.cer(list(rr), list(hh)) * 100

    step, best = 0, 1e9
    model.train()
    while step < args.steps:
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(feats, labels)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step + 1)
            opt.step(); step += 1
            if step % 50 == 0:
                print(f"step{step} loss {float(loss):.3f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
            if step % args.eval_every == 0:
                w, c = dev_eval()
                print(f"=== DEV step{step}: WER {w:.2f} CER {c:.2f} ===", flush=True)
                if w < best:
                    best = w
                    torch.save({"proj": model.proj.state_dict(),
                                "byt5": model.byt5.state_dict(), "byt5_id": args.byt5}, outdir / "best.pt")
                    print(f"  saved best (WER {w:.2f})", flush=True)
            if step >= args.steps:
                break
    print(f"DONE best dev WER {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
