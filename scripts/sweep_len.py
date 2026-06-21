"""Find the sequence length where Parakeet's backward crashes. Forward+backward on dummy
features of increasing mel-frame length (100 fps), bf16 autocast (torch 2.11 venv)."""
import torch, torch.nn as nn
from transformers import ParakeetForCTC
m = ParakeetForCTC.from_pretrained("nvidia/parakeet-ctc-0.6b")
m.ctc_head = nn.Conv1d(m.config.encoder_config.hidden_size, 513, 1)
m.config.vocab_size = 513; m.config.pad_token_id = 512
m.to("cuda").train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-5)
for sec in [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28]:
    T = sec * 100
    feats = torch.randn(1, T, 80, device="cuda")
    am = torch.ones(1, T, dtype=torch.long, device="cuda")
    lab = torch.randint(0, 512, (1, max(2, sec)), device="cuda")
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(input_features=feats, attention_mask=am, labels=lab)
        out.loss.backward(); opt.zero_grad(); torch.cuda.synchronize()
        print(f"sec={sec:2d} T={T} loss={float(out.loss):.2f} OK", flush=True)
    except Exception as e:
        print(f"sec={sec:2d} T={T} CRASH: {type(e).__name__} {str(e)[:80]}", flush=True)
        break
print("done", flush=True)
