"""Optuna hyperparameter search for the NeblinIA-Speech LoRA, judged on in-domain
Omnilingual *dev* WER (the same 24 languages as the benchmark — NOT a Common Voice
proxy). Each trial runs a short fine-tune and reports dev WER; a MedianPruner kills
weak trials early via intermediate evals.

Search space (the dials from docs/tuning.md, made safe):
  r            in {16, 32, 64, 128}
  scale        loguniform[0.5, 4.0]  -> the EFFECTIVE LoRA scaling magnitude. alpha is
                                        derived from it: alpha = scale*r (plain) or
                                        scale*sqrt(r) (rsLoRA). This keeps the update
                                        magnitude comparable across rank/variant, so
                                        rsLoRA can't silently explode.
  lr           loguniform[3e-5, 8e-4]
  dropout      in {0.0, 0.05, 0.1}
  use_rslora   bool
  use_dora     bool

  uv run --python .venv-unsloth python scripts/sweep_optuna.py \
      --trials 24 --trial-steps 350 --eval-steps 175 --subset 9000

Writes the study to optuna_neblinia.db (SQLite) and the best params to
models/optuna_best.json.
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must precede transformers/peft)

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "openai/whisper-large-v3-turbo"
BEST_OUT = ROOT / "models" / "optuna_best.json"
STUDY_DB = f"sqlite:///{ROOT / 'optuna_neblinia.db'}"

# reuse the data + collator plumbing from the trainer (single source of truth)
from train_unsloth import Collator, build_indomain_dataset  # noqa: E402


def make_objective(args):
    import jiwer
    import numpy as np
    import optuna
    from transformers import (AutoModelForSpeechSeq2Seq, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments, TrainerCallback)
    from unsloth import FastModel

    # cap val: evaluating all ~5.8k dev segments every eval step would dominate sweep
    # wall-clock. A balanced subset is plenty for ranking trials.
    train_ds, val_ds = build_indomain_dataset(max_train=args.subset, dev_cap=args.dev_cap)

    def objective(trial: optuna.Trial) -> float:
        r = trial.suggest_categorical("r", [16, 32, 64, 128])
        scale = trial.suggest_float("scale", 0.5, 4.0, log=True)
        lr = trial.suggest_float("lr", 3e-5, 8e-4, log=True)
        dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
        use_rslora = trial.suggest_categorical("use_rslora", [True, False])
        use_dora = trial.suggest_categorical("use_dora", [True, False])
        # derive alpha from the effective scale so rsLoRA (alpha/sqrt(r)) and plain
        # LoRA (alpha/r) land at the same update magnitude for a given `scale`.
        alpha = scale * (math.sqrt(r) if use_rslora else r)

        model, processor = FastModel.from_pretrained(
            BASE, auto_model=AutoModelForSpeechSeq2Seq, whisper_language="Spanish",
            whisper_task="transcribe", load_in_4bit=False, dtype=None)
        model = FastModel.get_peft_model(
            model, r=r, target_modules=["q_proj", "v_proj"], lora_alpha=alpha,
            lora_dropout=dropout, bias="none", use_gradient_checkpointing="unsloth",
            use_rslora=use_rslora, use_dora=use_dora, random_state=3407, task_type=None)
        model.generation_config.language = "<|es|>"
        model.generation_config.task = "transcribe"
        model.config.suppress_tokens = []
        model.generation_config.forced_decoder_ids = None

        tok = processor.tokenizer

        def preprocess_logits_for_metrics(logits, labels):
            lg = logits[0] if isinstance(logits, tuple) else logits
            return lg.argmax(dim=-1)

        def compute_metrics(pred):
            pi = np.where(pred.predictions == -100, tok.pad_token_id, pred.predictions)
            li = np.where(pred.label_ids == -100, tok.pad_token_id, pred.label_ids)
            ps = tok.batch_decode(pi, skip_special_tokens=True)
            ls = tok.batch_decode(li, skip_special_tokens=True)
            pairs = [(r_, h) for r_, h in zip(ls, ps) if r_.strip()]
            if not pairs:
                return {"wer": 100.0}
            refs, hyps = zip(*pairs)
            return {"wer": round(jiwer.wer(list(refs), list(hyps)) * 100, 2)}

        # report intermediate dev WER to Optuna so the pruner can stop weak trials
        class PruneCb(TrainerCallback):
            def on_evaluate(self, a, state, control, metrics=None, **kw):
                if metrics and "eval_wer" in metrics:
                    trial.report(metrics["eval_wer"], step=state.global_step)
                    if trial.should_prune():
                        control.should_training_stop = True

        outdir = ROOT / "models" / f"optuna_trial_{trial.number}"
        targs = Seq2SeqTrainingArguments(
            output_dir=str(outdir), per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch, learning_rate=lr, warmup_ratio=0.05,
            max_steps=args.trial_steps, bf16=True, logging_steps=50,
            eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="no",
            report_to="none", remove_unused_columns=False, dataloader_num_workers=8,
            dataloader_prefetch_factor=4, predict_with_generate=False,
            label_names=["labels"])
        trainer = Seq2SeqTrainer(
            model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
            data_collator=Collator(processor), compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[PruneCb()])
        trainer.train()
        final = trainer.evaluate().get("eval_wer", 100.0)

        del model, trainer
        import gc

        import torch
        gc.collect(); torch.cuda.empty_cache()
        return final

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--trial-steps", type=int, default=350)
    ap.add_argument("--eval-steps", type=int, default=175)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--subset", type=int, default=9000, help="cap train clips per trial for speed")
    ap.add_argument("--dev-cap", type=int, default=480, help="cap val clips (balanced) for fast evals")
    args = ap.parse_args()

    import optuna
    study = optuna.create_study(
        direction="minimize", study_name="neblinia-lora",
        storage=STUDY_DB, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=3407))
    study.optimize(make_objective(args), n_trials=args.trials)

    print("\n=== BEST ===")
    print("dev WER:", study.best_value)
    print("params:", study.best_params)
    BEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    BEST_OUT.write_text(json.dumps(
        {"dev_wer": study.best_value, "params": study.best_params}, indent=2))
    print("wrote", BEST_OUT)


if __name__ == "__main__":
    main()
