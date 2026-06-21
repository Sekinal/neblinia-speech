"""NeblinIA-mini: a SIMPLE from-scratch autoregressive attention encoder-decoder (AED) for
Mexican Indigenous ASR. Char-level (no OOV), small (~20M), trained from random init on our
data. Baked from the campaign findings: autoregressive decoder (beats CTC), char tokenizer
(beats OOV BPE/subword), SpecAugment + label smoothing, small model for tiny data.

Goal: a from-scratch FLOOR + a clean test of "autoregressive > CTC" independent of
pretraining. Compare vs Whisper (59) and Parakeet-CTC (~93).

  .venv-unsloth/bin/python scripts/train_scratch.py [--smoke] [--epochs E] [--d 256] ...
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train" / "manifest_indomain.jsonl"
DEV = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"
PAD, BOS, EOS = 0, 1, 2
DEV_DEVICE = "cuda"


def build_vocab(texts):
    chars = sorted(set("".join(texts)))
    stoi = {c: i + 3 for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, len(chars) + 3


def load_audio(path):
    import librosa, soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=5000):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def conv_out_len(L):                       # two conv2d stride2 pad1 k3 -> 4x downsample
    for _ in range(2):
        L = (L - 1) // 2 + 1
    return L


class AED(nn.Module):
    def __init__(self, V, d=256, nhead=4, enc=6, dec=4, n_mel=80):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, d, 3, 2, 1), nn.ReLU(),
                                  nn.Conv2d(d, d, 3, 2, 1), nn.ReLU())
        self.proj = nn.Linear(d * (n_mel // 4), d)
        self.encpos = PositionalEncoding(d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True, norm_first=True), enc)
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.decpos = PositionalEncoding(d)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nhead, 4 * d, batch_first=True, norm_first=True), dec)
        self.head = nn.Linear(d, V)

    def encode(self, mel, mel_len):
        x = self.conv(mel.unsqueeze(1))                       # [B,d,T',F']
        B, C, Tp, Fp = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, Tp, C * Fp)
        x = self.encpos(self.proj(x))
        sub = conv_out_len(mel_len)
        mmask = torch.arange(Tp, device=x.device)[None, :] >= sub[:, None]
        return self.enc(x, src_key_padding_mask=mmask), mmask

    def forward(self, mel, mel_len, tin, tout, tpad):
        mem, mmask = self.encode(mel, mel_len)
        y = self.decpos(self.emb(tin))
        cmask = torch.triu(torch.ones(tin.size(1), tin.size(1), device=mel.device), 1).bool()
        out = self.dec(y, mem, tgt_mask=cmask, tgt_key_padding_mask=tpad,
                       memory_key_padding_mask=mmask)
        logits = self.head(out)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tout.reshape(-1),
                               ignore_index=PAD, label_smoothing=0.1)
        return loss, logits

    @torch.no_grad()
    def greedy(self, mel, mel_len, itos, max_len=180):
        mem, mmask = self.encode(mel, mel_len)
        B = mel.size(0)
        ys = torch.full((B, 1), BOS, device=mel.device, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=mel.device)
        for _ in range(max_len):
            y = self.decpos(self.emb(ys))
            cmask = torch.triu(torch.ones(ys.size(1), ys.size(1), device=mel.device), 1).bool()
            out = self.dec(y, mem, tgt_mask=cmask, memory_key_padding_mask=mmask)
            nxt = self.head(out[:, -1]).argmax(-1)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        texts = []
        for row in ys.tolist():
            s = []
            for t in row[1:]:
                if t == EOS:
                    break
                s.append(itos.get(t, ""))
            texts.append("".join(s))
        return texts


def main():
    import torchaudio
    import jiwer
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--enc", type=int, default=6)
    ap.add_argument("--dec", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--dev-clips", type=int, default=400)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "models" / "neblinia-mini"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tr = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    dv = [json.loads(l) for l in open(DEV, encoding="utf-8")]
    if args.max_samples:
        tr = tr[:args.max_samples]
    stoi, itos, V = build_vocab([r["text"] for r in tr])
    print(f"char vocab V={V} | train {len(tr)} | dev {len(dv)}", flush=True)

    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=400, hop_length=160, n_mels=80).to(DEV_DEVICE)
    specaug = nn.Sequential(torchaudio.transforms.FrequencyMasking(27),
                            torchaudio.transforms.TimeMasking(80)).to(DEV_DEVICE)

    def feats(paths, train=False):
        mels, lens = [], []
        for p in paths:
            a = torch.tensor(load_audio(p), device=DEV_DEVICE)
            m = (melspec(a) + 1e-6).log()                    # [80, T]
            mels.append(m.transpose(0, 1)); lens.append(m.size(1))
        T = max(lens)
        x = torch.zeros(len(mels), T, 80, device=DEV_DEVICE)
        for i, m in enumerate(mels):
            x[i, :m.size(0)] = m
        x = (x - x.mean()) / (x.std() + 1e-5)
        if train:
            x = specaug(x.transpose(1, 2)).transpose(1, 2)
        return x, torch.tensor(lens, device=DEV_DEVICE)

    def labels(texts):
        seqs = [[BOS] + [stoi[c] for c in t if c in stoi] + [EOS] for t in texts]
        L = max(len(s) for s in seqs)
        tin = torch.full((len(seqs), L - 1), PAD, dtype=torch.long)
        tout = torch.full((len(seqs), L - 1), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            tin[i, :len(s) - 1] = torch.tensor(s[:-1])
            tout[i, :len(s) - 1] = torch.tensor(s[1:])
        tpad = tin == PAD
        return tin.to(DEV_DEVICE), tout.to(DEV_DEVICE), tpad.to(DEV_DEVICE)

    model = AED(V, args.d, 4, args.enc, args.dec).to(DEV_DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparams/1e6:.1f}M", flush=True)

    if args.smoke:
        b = tr[:6]
        x, xl = feats([r["audio"] for r in b], train=True)
        tin, tout, tpad = labels([r["text"] for r in b])
        loss, _ = model(x, xl, tin, tout, tpad)
        loss.backward()
        print(f"SMOKE loss {float(loss):.3f} | mel{tuple(x.shape)} OK", flush=True)
        hyp = model.greedy(x[:2], xl[:2], itos)
        print("SMOKE greedy sample:", hyp, flush=True)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    def lr_at(step):
        return min(step / args.warmup, (args.warmup / max(step, 1)) ** 0.5)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    json.dump({"stoi": stoi}, open(outdir / "vocab.json", "w"), ensure_ascii=False)

    @torch.no_grad()
    def dev_eval():
        model.eval()
        sub = dv[:args.dev_clips]
        refs, hyps = [], []
        for i in range(0, len(sub), args.batch):
            ch = sub[i:i + args.batch]
            x, xl = feats([r["audio"] for r in ch])
            for h, r in zip(model.greedy(x, xl, itos), ch):
                hyps.append(h.lower().strip()); refs.append(r["text"].lower().strip())
        model.train()
        pairs = [(r, h) for r, h in zip(refs, hyps) if r]
        rr, hh = zip(*pairs)
        return jiwer.wer(list(rr), list(hh)) * 100, jiwer.cer(list(rr), list(hh)) * 100

    step, best = 0, 1e9
    steps_per = max(1, len(tr) // args.batch)
    print(f"steps/epoch {steps_per} | total {steps_per*args.epochs}", flush=True)
    for ep in range(args.epochs):
        random.shuffle(tr)
        for i in range(0, len(tr) - args.batch, args.batch):
            ch = tr[i:i + args.batch]
            x, xl = feats([r["audio"] for r in ch], train=True)
            tin, tout, tpad = labels([r["text"] for r in ch])
            loss, _ = model(x, xl, tin, tout, tpad)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step + 1)
            opt.step(); step += 1
            if step % 50 == 0:
                print(f"ep{ep} step{step} loss {float(loss):.3f} lr {opt.param_groups[0]['lr']:.2e}", flush=True)
            if step % args.eval_every == 0:
                w, c = dev_eval()
                print(f"=== DEV step{step}: WER {w:.2f} CER {c:.2f} ===", flush=True)
                if w < best:
                    best = w
                    torch.save({"model": model.state_dict(), "stoi": stoi,
                                "cfg": {"d": args.d, "enc": args.enc, "dec": args.dec, "V": V}},
                               outdir / "best.pt")
                    print(f"  saved best (WER {w:.2f})", flush=True)
    print(f"DONE best dev WER {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
