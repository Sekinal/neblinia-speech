"""Fair PEFT-method comparison for the in-domain ASR task, inspired by HF's
'Beyond LoRA' (https://huggingface.co/blog/peft-beyond-lora).

Same base / data / eval / harness for every method (vanilla transformers + PEFT, no
Unsloth). The blog's key caveat: a method only loses to LoRA if you under-tune its LR,
so EACH method gets its own small LR sweep. Ranks methods by in-domain dev WER.

Methods (all in peft 0.19): LoRA, rsLoRA, DoRA, OFT, BOFT, VeRA, LoHa, LoKr, AdaLoRA.

  .venv-unsloth/bin/python scripts/sweep_methods.py [--dry-run] [--steps 150] [--subset 8000]

Writes methods_results.json (incrementally) + prints a ranked table.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
BASE = "openai/whisper-large-v3-turbo"
TARGETS = ["q_proj", "v_proj"]
RESULTS = ROOT / "methods_results.json"

# method -> (config factory taking total_steps, list of LRs to try)
# LR ranges are method-aware: orthogonal/projection methods and VeRA want higher LRs.
def method_configs(steps):
    from peft import (AdaLoraConfig, BOFTConfig, LoHaConfig, LoKrConfig, LoraConfig,
                      OFTConfig, VeraConfig)
    r = 32
    return {
        "lora":    (lambda: LoraConfig(r=r, lora_alpha=2 * r, target_modules=TARGETS, lora_dropout=0.05), [1e-4, 3e-4]),
        "rslora":  (lambda: LoraConfig(r=r, lora_alpha=2 * r, target_modules=TARGETS, lora_dropout=0.05, use_rslora=True), [1e-4, 3e-4]),
        "dora":    (lambda: LoraConfig(r=r, lora_alpha=2 * r, target_modules=TARGETS, lora_dropout=0.05, use_dora=True), [1e-4, 3e-4]),
        "loha":    (lambda: LoHaConfig(r=r, alpha=2 * r, target_modules=TARGETS, module_dropout=0.05), [1e-4, 3e-4]),
        "lokr":    (lambda: LoKrConfig(r=r, alpha=2 * r, target_modules=TARGETS, module_dropout=0.05), [1e-4, 3e-4]),
        "oft":     (lambda: OFTConfig(r=0, oft_block_size=32, target_modules=TARGETS, module_dropout=0.0), [1e-4, 5e-4]),
        "boft":    (lambda: BOFTConfig(boft_block_size=8, target_modules=TARGETS, boft_dropout=0.0), [1e-4, 5e-4]),
        "vera":    (lambda: VeraConfig(r=256, target_modules=TARGETS, vera_dropout=0.0), [1e-3, 5e-3]),
        "adalora": (lambda: AdaLoraConfig(init_r=16, target_r=8, target_modules=TARGETS, total_step=steps), [1e-4, 3e-4]),
    }


def build_model_with_peft(cfg):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForSpeechSeq2Seq
    model = AutoModelForSpeechSeq2Seq.from_pretrained(BASE, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, cfg)
    model.generation_config.language = "<|es|>"
    model.generation_config.task = "transcribe"
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    return model


def run_one(method, lr, cfg_factory, train_ds, val_ds, processor, steps, batch):
    import jiwer
    import numpy as np
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    from data_indomain import Collator

    cfg = cfg_factory()
    model = build_model_with_peft(cfg)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tok = processor.tokenizer

    def preprocess_logits_for_metrics(logits, labels):
        lg = logits[0] if isinstance(logits, tuple) else logits
        return lg.argmax(dim=-1)

    def compute_metrics(pred):
        pi = np.where(pred.predictions == -100, tok.pad_token_id, pred.predictions)
        li = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
        ps = tok.batch_decode(pi, skip_special_tokens=True)
        ls = tok.batch_decode(li, skip_special_tokens=True)
        pairs = [(a, b) for a, b in zip(ls, ps) if a.strip()]
        if not pairs:
            return {"wer": 100.0}
        refs, hyps = zip(*pairs)
        return {"wer": round(jiwer.wer(list(refs), list(hyps)) * 100, 2)}

    outdir = ROOT / "models" / f"method_{method}_{lr:.0e}"
    targs = Seq2SeqTrainingArguments(
        output_dir=str(outdir), per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch, learning_rate=lr, warmup_ratio=0.05,
        max_steps=steps, bf16=True, logging_steps=50, eval_strategy="steps",
        eval_steps=steps, save_strategy="no", report_to="none",
        remove_unused_columns=False, dataloader_num_workers=8,
        dataloader_prefetch_factor=4, predict_with_generate=False, label_names=["labels"])
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(processor), compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    wer = trainer.evaluate().get("eval_wer", 100.0)
    vram = torch.cuda.max_memory_allocated() / 1e9
    del model, trainer
    gc.collect(); torch.cuda.empty_cache()
    return {"method": method, "lr": lr, "dev_wer": wer, "trainable_params": n_train,
            "peak_vram_gb": round(vram, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="construct + apply each config, no training")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--subset", type=int, default=8000)
    ap.add_argument("--dev-cap", type=int, default=480)
    ap.add_argument("--only", default="", help="comma list of methods to run")
    args = ap.parse_args()

    from transformers import WhisperProcessor

    from data_indomain import build_indomain_dataset
    processor = WhisperProcessor.from_pretrained(BASE, language="es", task="transcribe")
    cfgs = method_configs(args.steps)
    if args.only:
        cfgs = {k: v for k, v in cfgs.items() if k in args.only.split(",")}

    if args.dry_run:
        print("=== DRY RUN: construct + apply each method ===", flush=True)
        for name, (factory, _) in cfgs.items():
            try:
                m = build_model_with_peft(factory())
                n = sum(p.numel() for p in m.parameters() if p.requires_grad)
                print(f"  OK  {name:8} trainable={n:,}", flush=True)
                del m; gc.collect()
                import torch; torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {name:8} {type(e).__name__}: {str(e)[:90]}", flush=True)
        return

    train_ds, val_ds = build_indomain_dataset(max_train=args.subset, dev_cap=args.dev_cap)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {(r["method"], r["lr"]) for r in results}
    for name, (factory, lrs) in cfgs.items():
        for lr in lrs:
            if (name, lr) in done:
                print(f"skip {name} lr={lr} (done)", flush=True); continue
            print(f"\n=== {name} lr={lr} ===", flush=True)
            try:
                res = run_one(name, lr, factory, train_ds, val_ds, processor, args.steps, args.batch)
            except Exception as e:  # noqa: BLE001
                import torch; gc.collect(); torch.cuda.empty_cache()
                res = {"method": name, "lr": lr, "dev_wer": None, "error": f"{type(e).__name__}: {str(e)[:120]}"}
                print(f"  ERROR {name} lr={lr}: {res['error']}", flush=True)
            results.append(res)
            RESULTS.write_text(json.dumps(results, indent=2))
            if res.get("dev_wer") is not None:
                print(f"  -> dev WER {res['dev_wer']} | vram {res.get('peak_vram_gb')}GB | params {res.get('trainable_params'):,}", flush=True)

    ok = [r for r in results if r.get("dev_wer") is not None]
    ok.sort(key=lambda r: r["dev_wer"])
    print("\n=== RANKED (best dev WER) ===", flush=True)
    for r in ok:
        print(f"  {r['dev_wer']:6.2f}  {r['method']:8} lr={r['lr']:.0e}  "
              f"{r.get('peak_vram_gb','?')}GB  {r.get('trainable_params',0):,} params", flush=True)


if __name__ == "__main__":
    main()
