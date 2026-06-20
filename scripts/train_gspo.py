"""GSPO (Group Sequence Policy Optimization) RL post-training for NeblinIA-Speech.

Why: SFT (teacher-forced) never sees the model's own outputs, so it can't punish the
repetition loops that still inflate WER on a few clips. RL on a *verifiable* reward
(-CER vs the reference) optimizes the actual sampled transcriptions — a looping output
racks up insertions -> awful CER -> negative advantage -> suppressed.

Method (GSPO, arxiv 2507.18071): sample G hyps per clip, reward = -CER, group-relative
advantage, and a SEQUENCE-level importance ratio exp((logp_theta - logp_old)/|y|)
(length-normalized geometric mean) — more stable than GRPO's token-level ratio for long
sequences. Small KL to the frozen SFT anchor.

Whisper takes AUDIO, so this is a hand-rolled loop (TRL's GRPO is text-only).

  .venv-unsloth/bin/python scripts/train_gspo.py [--clips-per-step 8] [--group 8]
      [--steps 300] [--lr 1e-6] [--temp 1.0] [--kl 0.04] [--clip 0.2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.environ.get("MEXA_SRC", str(ROOT.parent / "mexa-benchmark" / "src")))
BASE = "openai/whisper-large-v3-turbo"
ADAPTER = ROOT / "models" / "neblinia-preview-0.2" / "lora"
MANIFEST = ROOT / "data" / "train" / "manifest_indomain.jsonl"
DEV_MANIFEST = ROOT / "data" / "train" / "manifest_indomain_dev.jsonl"
OUTDIR = ROOT / "models" / "neblinia-preview-0.3-gspo"
DEVICE = "cuda"


def load_audio(path):
    import librosa
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


def max_ngram_repeat(words, n=2):
    """Longest run of an immediately-repeating n-gram (loop detector)."""
    if len(words) < 2 * n:
        return 0
    best = 0
    for i in range(len(words) - n):
        run, j = 1, i
        while j + n < len(words) and words[j:j + n] == words[j + n:j + 2 * n]:
            run += 1; j += n
        best = max(best, run)
    return best


def compute_reward(hyp_norm, ref_norm, w, jiwer):
    """Composite reward: metric-aligned (CER+WER blend) MINUS explicit loop penalties
    (over-generation length + n-gram repetition). Pure -CER = {cer:1, wer:0, len:0, rep:0}."""
    rw, hw = ref_norm.split(), hyp_norm.split()
    cer = jiwer.cer(ref_norm, hyp_norm) if ref_norm else 1.0
    wer = jiwer.wer(ref_norm, hyp_norm) if ref_norm else 1.0
    len_ratio = len(hw) / max(1, len(rw))
    over = max(0.0, len_ratio - 1.2)                        # over-generation (loops/inserts)
    rep = max_ngram_repeat(hw, 2) / max(1, len(hw))         # repetition fraction
    return -(w["cer"] * min(cer, 2.0) + w["wer"] * min(wer, 2.0)) - w["len"] * over - w["rep"] * rep


def encode(model, feats):
    """Run the (LoRA-augmented) encoder ONCE per clip -> [B, T, D]. Reused across the G
    samples so we don't re-encode the 1500-frame audio G times (the OOM culprit)."""
    return model.get_base_model().get_encoder()(feats).last_hidden_state


def seq_logprob(model, enc_rep, sequences, prefix_len, pad_id, eot_id):
    """Summed log-prob of the GENERATED tokens of each sequence, given precomputed encoder
    outputs (enc_rep = [N, T, D]). Returns (seq_logprob[N], gen_len[N])."""
    from transformers.modeling_outputs import BaseModelOutput
    dec_in = sequences[:, :-1]
    labels = sequences[:, 1:]
    logits = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc_rep),
                   decoder_input_ids=dec_in).logits
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)          # [N, L-1]
    pos = torch.arange(labels.size(1), device=labels.device).unsqueeze(0)
    gen = pos >= (prefix_len - 1)                                      # generated region
    not_pad = labels != pad_id
    ie = (labels == eot_id).long()
    after_eot = (ie.cumsum(1) - ie) > 0                                # tokens past first eot
    mask = gen & not_pad & (~after_eot)
    return (tok_lp * mask).sum(1), mask.sum(1).clamp(min=1)


@torch.no_grad()
def eval_dev(model, val_feats, val_refs, tok, normalize, jiwer, pad_id):
    """Held-out GREEDY dev CER — the honest RL signal (vs noisy on-policy training reward)."""
    model.eval()
    cers = []
    for i in range(0, len(val_refs), 16):
        fb = val_feats[i:i + 16].to(DEVICE, torch.bfloat16)
        gen = model.generate(input_features=fb, do_sample=False, num_beams=1,
                             max_new_tokens=128, language="es", task="transcribe",
                             pad_token_id=pad_id)
        for h, r in zip(tok.batch_decode(gen, skip_special_tokens=True), val_refs[i:i + 16]):
            cers.append(jiwer.cer(r, normalize(h)) if r else 1.0)
    return sum(cers) / max(1, len(cers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-per-step", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=0.04)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-clips", type=int, default=0, help="cap manifest (0=all)")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=25, help="held-out greedy dev CER cadence")
    ap.add_argument("--val-clips", type=int, default=72, help="fixed held-out dev clips")
    # composite reward weights (pure -CER = 1,0,0,0)
    ap.add_argument("--w-cer", type=float, default=0.7)
    ap.add_argument("--w-wer", type=float, default=0.3)
    ap.add_argument("--w-len", type=float, default=0.3, help="over-generation (loop) penalty")
    ap.add_argument("--w-rep", type=float, default=0.5, help="n-gram repetition penalty")
    ap.add_argument("--outdir", default=str(OUTDIR), help="checkpoint output dir")
    # MGPO (VibeThinker): weight each clip's advantage by exp(-gamma*KL(p_c||0.5)) so the
    # gradient focuses on the learnable frontier (clips solved ~half the time).
    ap.add_argument("--mgpo", action="store_true", help="enable MaxEnt-Guided weighting")
    ap.add_argument("--mgpo-gamma", type=float, default=2.0)
    ap.add_argument("--mgpo-tau", type=float, default=0.3, help="CER<tau counts as a 'correct' sample")
    args = ap.parse_args()
    rw_w = {"cer": args.w_cer, "wer": args.w_wer, "len": args.w_len, "rep": args.w_rep}
    outdir = Path(args.outdir) if getattr(args, "outdir", None) else OUTDIR
    print(f"GSPO config: {vars(args)}", flush=True)

    import jiwer
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from mexa.normalize import normalize

    proc = WhisperProcessor.from_pretrained(str(ADAPTER), language="es", task="transcribe")
    tok, fe = proc.tokenizer, proc.feature_extractor
    pad_id = tok.pad_token_id
    eot_id = tok.eos_token_id

    def mk(trainable):
        base = WhisperForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=trainable).to(DEVICE)
        m.generation_config.language = "<|es|>"
        m.generation_config.task = "transcribe"
        m.generation_config.forced_decoder_ids = None
        m.config.suppress_tokens = []
        return m

    print("loading policy + reference...", flush=True)
    policy = mk(True)
    ref = mk(False).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    # forced prefix length: <sot><lang><task><notimestamps>
    prefix_ids = [tok.convert_tokens_to_ids(t) for t in
                  ["<|startoftranscript|>", "<|es|>", "<|transcribe|>", "<|notimestamps|>"]]
    prefix_len = len(prefix_ids)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    if args.max_clips:
        rows = rows[:args.max_clips]
    print(f"RL prompts: {len(rows)} clips", flush=True)

    # fixed held-out dev set (balanced across languages) for greedy validation
    dev_rows = [json.loads(l) for l in open(DEV_MANIFEST, encoding="utf-8")]
    by_lang = defaultdict(list)
    for r in dev_rows:
        by_lang[r["language"]].append(r)
    per_lang = max(1, args.val_clips // max(1, len(by_lang)))
    val_rows = [r for v in by_lang.values() for r in v[:per_lang]][:args.val_clips]
    val_feats = fe([load_audio(r["audio"]) for r in val_rows], sampling_rate=16000,
                   return_tensors="pt").input_features
    val_refs = [normalize(r["text"]) for r in val_rows]
    print(f"held-out dev: {len(val_rows)} clips ({len(by_lang)} langs)", flush=True)
    best_vcer = float("inf")

    G = args.group
    gen_kwargs = dict(do_sample=True, temperature=args.temp, num_return_sequences=G,
                      max_new_tokens=args.max_new, language="es", task="transcribe",
                      pad_token_id=pad_id)
    outdir.mkdir(parents=True, exist_ok=True)
    # baseline dev CER (the SFT preview-0.2 starting point) before any RL
    v0 = eval_dev(policy, val_feats, val_refs, tok, normalize, jiwer, pad_id)
    print(f"VAL step    0 (SFT baseline): greedy dev CER {v0:.4f}", flush=True)
    best_vcer = v0
    idx = 0
    for step in range(args.steps):
        batch = rows[idx:idx + args.clips_per_step]
        idx = (idx + args.clips_per_step) % max(1, len(rows) - args.clips_per_step)
        if not batch:
            break
        feats = fe([load_audio(b["audio"]) for b in batch], sampling_rate=16000,
                   return_tensors="pt").input_features.to(DEVICE, torch.bfloat16)  # [B,80,3000]
        refs = [normalize(b["text"]) for b in batch]

        # 1) rollout: sample G hyps per clip (old policy = current weights, no grad)
        policy.eval()
        with torch.no_grad():
            gen = policy.generate(input_features=feats, **gen_kwargs)            # [B*G, L]
        # 2) reward = -CER (group-relative advantage)
        hyps = tok.batch_decode(gen, skip_special_tokens=True)
        rewards = torch.zeros(len(batch) * G, device=DEVICE)
        cers = torch.zeros(len(batch) * G, device=DEVICE)
        for i in range(len(batch)):
            for g in range(G):
                h = normalize(hyps[i * G + g])
                rewards[i * G + g] = compute_reward(h, refs[i], rw_w, jiwer)
                cers[i * G + g] = jiwer.cer(refs[i], h) if refs[i] else 1.0
        rew = rewards.view(len(batch), G)
        adv = (rew - rew.mean(1, keepdim=True)) / (rew.std(1, keepdim=True) + 1e-4)
        adv = adv.view(-1)                                                       # [B*G]

        mgpo_frac = 0.0
        if args.mgpo:
            # p_c = per-clip pass rate (CER<tau); weight ~1 at p_c=0.5, ->0 at 0/1 extremes
            p_c = (cers.view(len(batch), G) < args.mgpo_tau).float().mean(1).clamp(1e-3, 1 - 1e-3)
            d_me = p_c * torch.log(p_c / 0.5) + (1 - p_c) * torch.log((1 - p_c) / 0.5)
            w_me = torch.exp(-args.mgpo_gamma * d_me)                            # [B]
            adv = adv * w_me.repeat_interleave(G)
            mgpo_frac = ((p_c > 0.15) & (p_c < 0.85)).float().mean().item()      # frac on frontier

        # 3) logprobs under policy (grad) + ref (no grad); old = policy detached (mu=1).
        #    Encode ONCE per clip, repeat the encoder output across the G samples.
        policy.train()
        enc_pol = encode(policy, feats).repeat_interleave(G, dim=0)              # [B*G,T,D] grad
        lp, glen = seq_logprob(policy, enc_pol, gen, prefix_len, pad_id, eot_id)
        with torch.no_grad():
            enc_ref = encode(ref, feats).repeat_interleave(G, dim=0)
            lp_ref, _ = seq_logprob(ref, enc_ref, gen, prefix_len, pad_id, eot_id)
        lp_old = lp.detach()

        # 4) GSPO: sequence-level (length-normalized) importance ratio + clip + KL
        log_ratio = (lp - lp_old) / glen.float()
        ratio = log_ratio.exp()
        unclipped = ratio * adv
        clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * adv
        pg = -torch.min(unclipped, clipped).mean()
        # k3 KL estimator KL(policy||ref), per sequence (length-normalized)
        d = (lp_ref - lp) / glen.float()
        kl = (d.exp() - d - 1).mean()
        loss = pg + args.kl * kl

        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()

        if step % args.log_every == 0:
            mtag = f" | frontier {mgpo_frac:.2f}" if args.mgpo else ""
            print(f"step {step:4d} | reward {rewards.mean().item():+.3f} | "
                  f"KL {kl.item():.4f} | adv|{adv.abs().mean():.3f} | "
                  f"genlen {glen.float().mean():.1f} | loss {loss.item():+.4f} | "
                  f"gnorm {gn:.2f}{mtag}", flush=True)
        if step and step % args.val_every == 0:
            vcer = eval_dev(policy, val_feats, val_refs, tok, normalize, jiwer, pad_id)
            tag = ""
            if vcer < best_vcer:
                best_vcer = vcer
                policy.save_pretrained(str(outdir / "best"))
                tag = "  <- new best (saved)"
            print(f"VAL step {step:4d}: greedy dev CER {vcer:.4f} (best {best_vcer:.4f}){tag}",
                  flush=True)
        if step and step % args.save_every == 0:
            policy.save_pretrained(str(outdir / f"step{step}"))
            print(f"  saved -> {outdir}/step{step}", flush=True)

    policy.save_pretrained(str(outdir / "final"))
    proc.save_pretrained(str(outdir / "final"))
    print(f"DONE GSPO -> {outdir}/final", flush=True)


if __name__ == "__main__":
    main()
