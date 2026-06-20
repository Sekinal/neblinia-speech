"""Merge a LoRA/DoRA adapter into Whisper-turbo and export a CTranslate2 model
for the fair faster-whisper eval. Reusable, no ad-hoc shell.

Usage:
  python scripts/export_ct2.py <adapter_dir> <out_base_dir> [--good-tokenizer <tokenizer.json>]

Produces <out_base_dir>/merged_hf and <out_base_dir>/ct2 (float16).
Carries over the known-good 3.9MB tokenizer.json so faster-whisper does not
choke on a stub (the WhisperProcessor-from-base gotcha).
"""
from __future__ import annotations
import json, shutil, subprocess, sys
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

adapter = Path(sys.argv[1]).resolve()
outbase = Path(sys.argv[2]).resolve()
good_tok = None
if "--good-tokenizer" in sys.argv:
    good_tok = Path(sys.argv[sys.argv.index("--good-tokenizer") + 1]).resolve()

cfg = json.loads((adapter / "adapter_config.json").read_text())
base_id = cfg["base_model_name_or_path"]
print(f"base={base_id}\nadapter={adapter}\nout={outbase}", flush=True)

merged = outbase / "merged_hf"
ct2 = outbase / "ct2"
merged.mkdir(parents=True, exist_ok=True)

print("loading base + adapter, merging...", flush=True)
base = WhisperForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, str(adapter))
model = model.merge_and_unload()
model.save_pretrained(str(merged), safe_serialization=True)

proc = WhisperProcessor.from_pretrained(base_id)
proc.save_pretrained(str(merged))

# config dtype gotcha: ct2 converter wants torch_dtype, tf5 saves dtype
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

# belt-and-suspenders: ensure good tokenizer in ct2 dir too
if good_tok and good_tok.exists():
    shutil.copy(good_tok, ct2 / "tokenizer.json")
print(f"DONE -> {ct2}", flush=True)
