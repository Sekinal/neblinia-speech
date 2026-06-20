"""Export a fine-tuned Whisper model to CTranslate2 for the fair faster-whisper eval.
Reusable, no ad-hoc shell. Accepts EITHER:
  - a LoRA/DoRA adapter dir (has adapter_config.json) -> merges into base, or
  - an already-merged HF dir (full fine-tune output) -> converts directly.

Usage:
  python scripts/export_ct2.py <adapter_or_merged_dir> <out_base_dir> [--good-tokenizer <tokenizer.json>]

Produces <out_base_dir>/ct2 (float16) and, for the adapter path, <out_base_dir>/merged_hf.

Gotchas handled:
  - transformers 5.5 WhisperProcessor.save_pretrained writes processor_config.json, NOT
    the preprocessor_config.json the ct2 converter needs -> we write it from the feature
    extractor explicitly.
  - ct2 converter wants config `torch_dtype`, tf5 saves `dtype` -> rename.
  - setsid strips PATH -> call the converter by absolute path next to sys.executable.
  - faster-whisper stub-tokenizer crash -> carry over a known-good tokenizer.json.
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

src = Path(sys.argv[1]).resolve()
outbase = Path(sys.argv[2]).resolve()
good_tok = None
if "--good-tokenizer" in sys.argv:
    good_tok = Path(sys.argv[sys.argv.index("--good-tokenizer") + 1]).resolve()

ct2 = outbase / "ct2"
is_adapter = (src / "adapter_config.json").exists()

if is_adapter:
    from peft import PeftModel
    base_id = json.loads((src / "adapter_config.json").read_text())["base_model_name_or_path"]
    merged = outbase / "merged_hf"
    merged.mkdir(parents=True, exist_ok=True)
    print(f"[adapter] base={base_id} adapter={src} -> {merged}", flush=True)
    base = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(src)).merge_and_unload()
    model.save_pretrained(str(merged), safe_serialization=True)
    proc = WhisperProcessor.from_pretrained(base_id)
else:
    merged = src   # already-merged HF dir (full fine-tune); convert in place
    print(f"[merged-hf] {merged}", flush=True)
    proc = WhisperProcessor.from_pretrained(str(merged))

# write tokenizer + BOTH config files (preprocessor_config.json is what ct2 needs)
proc.save_pretrained(str(merged))
proc.feature_extractor.save_pretrained(str(merged))   # -> preprocessor_config.json

cf = merged / "config.json"
d = json.loads(cf.read_text())
if "dtype" in d and "torch_dtype" not in d:
    d["torch_dtype"] = d["dtype"]
    cf.write_text(json.dumps(d, indent=2))
    print("renamed config dtype -> torch_dtype", flush=True)

if good_tok and good_tok.exists():
    shutil.copy(good_tok, merged / "tokenizer.json")
    print(f"copied good tokenizer from {good_tok}", flush=True)

if ct2.exists():
    shutil.rmtree(ct2)
print("converting to CT2 float16...", flush=True)
conv = str(Path(sys.executable).parent / "ct2-transformers-converter")
subprocess.run([
    conv, "--model", str(merged),
    "--output_dir", str(ct2), "--quantization", "float16",
    "--copy_files", "tokenizer.json", "preprocessor_config.json",
], check=True)

if good_tok and good_tok.exists():
    shutil.copy(good_tok, ct2 / "tokenizer.json")
print(f"DONE -> {ct2}", flush=True)
