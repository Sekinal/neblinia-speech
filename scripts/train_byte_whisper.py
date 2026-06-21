"""Byte-level Whisper-turbo: swap Whisper's BPE vocab for a tokenizer-free UTF-8 BYTE vocab
(256 bytes + pad/bos/eos), keep the pretrained encoder (frozen) and decoder layers, re-init
only the decoder token-embeddings + output head, and fine-tune on byte targets.

Why: byte output covers ANY Mexican Indigenous orthography with zero OOV (Whisper's BPE is
OOV for these scripts), while keeping all of Whisper's acoustic + sequence pretraining.
Custom greedy byte-decode for honest autoregressive eval (no teacher-forced trap).

  python scripts/train_byte_whisper.py [--smoke] [--steps N] [--batch 16] [--lr 1e-4]
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train" / "manifest_indomain.jsonl"
DEV = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"
BASE = "openai/whisper-large-v3-turbo"
PAD, BOS, EOS = 256, 257, 258
VOCAB = 259
MAXPOS = 1024          # extend Whisper's 448 decoder positions: byte labels run longer than BPE
DEVICE = "cuda"


def load_audio(path):
    import librosa, soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


class WDS(Dataset):
    def __init__(self, rows, fe):
        self.rows = rows
        self.fe = fe

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        feat = self.fe(load_audio(r["audio"]), sampling_rate=16000,
                       return_tensors="pt").input_features[0]      # [128, 3000]
        ids = list(r["text"].encode("utf-8")) + [EOS]
        return feat, torch.tensor(ids, dtype=torch.long)


def collate(batch):
    feats = torch.stack([b[0] for b in batch])                    # uniform [128,3000]
    L = max(b[1].size(0) for b in batch)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :b[1].size(0)] = b[1]
    return feats, labels


@torch.no_grad()
def greedy(model, feats, max_len=600, no_repeat=4):
    enc = model.model.encoder(feats).last_hidden_state
    B = feats.size(0)
    ids = torch.full((B, 1), BOS, device=feats.device, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool, device=feats.device)
    for _ in range(max_len):
        h = model.model.decoder(input_ids=ids, encoder_hidden_states=enc).last_hidden_state
        logits = model.proj_out(h[:, -1])
        if no_repeat and ids.size(1) >= no_repeat:        # block repeating n-grams (loop guard)
            seqs = ids.tolist()
            for b in range(B):
                pre = tuple(seqs[b][-(no_repeat - 1):])
                for i in range(len(seqs[b]) - no_repeat + 1):
                    if tuple(seqs[b][i:i + no_repeat - 1]) == pre:
                        logits[b, seqs[b][i + no_repeat - 1]] = -1e9
        nxt = logits.argmax(-1)
        ids = torch.cat([ids, nxt[:, None]], 1)
        done |= nxt == EOS
        if done.all():
            break
    outs = []
    for row in ids.tolist():
        b = []
        for t in row[1:]:
            if t == EOS:
                break
            if 0 <= t < 256:
                b.append(t)
        outs.append(bytes(b).decode("utf-8", errors="ignore"))
    return outs


def main():
    import jiwer
    from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=600)
    ap.add_argument("--dev-clips", type=int, default=300)
    ap.add_argument("--unfreeze-encoder", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "models" / "neblinia-byte-whisper"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tr = [json.loads(l) for l in open(TRAIN, encoding="utf-8")]
    dv = [json.loads(l) for l in open(DEV, encoding="utf-8")][:args.dev_clips]
    if args.max_samples:
        tr = tr[:args.max_samples]
    nbyte = lambda r: len(r["text"].encode("utf-8")) + 1
    tr = [r for r in tr if nbyte(r) <= MAXPOS]
    dv = [r for r in dv if nbyte(r) <= MAXPOS]
    print(f"train {len(tr)} | dev {len(dv)}", flush=True)

    from transformers.models.whisper.modeling_whisper import WhisperPositionalEmbedding
    fe = WhisperFeatureExtractor.from_pretrained(BASE)
    model = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.float32)
    d = model.config.d_model
    # swap decoder token embeddings + output head to the byte vocab (re-init)
    model.model.decoder.embed_tokens = nn.Embedding(VOCAB, d, padding_idx=PAD)
    model.proj_out = nn.Linear(d, VOCAB, bias=False)
    # extend decoder positional embeddings (byte sequences exceed Whisper's 448 cap).
    # INTERPOLATE the learned positions to MAXPOS so every position is distinct + smooth
    # (copying the last position makes all extra positions identical -> the decoder loses
    # track of position on long sequences and rambles).
    op = model.model.decoder.embed_positions.weight.data            # [448, d]
    npos = WhisperPositionalEmbedding(MAXPOS, d)
    interp = torch.nn.functional.interpolate(
        op.t().unsqueeze(0).float(), size=MAXPOS, mode="linear", align_corners=True
    ).squeeze(0).t().contiguous()                                  # [MAXPOS, d]
    npos.weight.data.copy_(interp.to(op.dtype))
    model.model.decoder.embed_positions = npos
    model.config.max_target_positions = MAXPOS
    model.model.decoder.max_target_positions = MAXPOS     # decoder caches this at init (was 448)
    model.max_target_positions = MAXPOS                   # the forward's label-length check reads THIS
    model.generation_config.max_length = MAXPOS
    model.config.vocab_size = VOCAB
    model.config.pad_token_id = PAD
    model.config.bos_token_id = BOS
    model.config.eos_token_id = EOS
    model.config.decoder_start_token_id = BOS
    model.config.suppress_tokens = []
    model.config.begin_suppress_tokens = None
    if not args.unfreeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad_(False)
    model.to(DEVICE)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"byte-whisper: vocab {VOCAB} | trainable {ntr/1e6:.0f}M | encoder frozen={not args.unfreeze_encoder}", flush=True)

    loader = DataLoader(WDS(tr, fe), batch_size=args.batch, shuffle=True, num_workers=8,
                        collate_fn=collate, drop_last=True, persistent_workers=True, prefetch_factor=4)
    dev_loader = DataLoader(WDS(dv, fe), batch_size=args.batch, shuffle=False, num_workers=8,
                            collate_fn=collate)

    if args.smoke:
        feats, labels = next(iter(loader))
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_features=feats, labels=labels)
        out.loss.backward()
        print(f"SMOKE loss {float(out.loss):.3f} feats{tuple(feats.shape)} OK | greedy:",
              greedy(model, feats[:2]), flush=True)
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
        for feats, labels in dev_loader:
            feats = feats.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hh = greedy(model, feats)
            for h, lab in zip(hh, labels):
                ids = [int(t) for t in lab if 0 <= int(t) < 256]
                refs.append(bytes(ids).decode("utf-8", errors="ignore").lower().strip())
                hyps.append(h.lower().strip())
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
                loss = model(input_features=feats, labels=labels).loss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    model.save_pretrained(str(outdir / "best"))
                    print(f"  saved best (WER {w:.2f})", flush=True)
            if step >= args.steps:
                break
    print(f"DONE best dev WER {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
