"""Run upstream torch codec (CPU) on texts + random ids; dump ids/codes/zhats/decoded to w/ref/<task>.npz."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
from ref_torch import load_ref
from texts import *
from transformers import AutoTokenizer
task = sys.argv[1]; d = f"{WORK}/up/{task}"
torch.set_grad_enabled(False); torch.manual_seed(0)
m, cfg = load_ref(d)
tok = AutoTokenizer.from_pretrained(f"{WORK}/tok/{'gpt2' if task=='xsum' else 'Qwen3-0.6B'}")
pt = cfg.tokenizer.pre_tokenizer
EOS = tok.convert_tokens_to_ids(pt.special_tokens.eos_token)
PAD = cfg.model.prefix_pad_token_id
L = cfg.tokenizer.vqvae.context_length
texts = (NEWS if task == "xsum" else PLAIN)
ids = [tok.encode(t)[: L - 1] + [EOS] for t in texts]
ids = [i + [PAD] * (L - len(i)) for i in ids]
rnd = torch.randint(0, 50000, (4, L)).tolist()
x = torch.tensor(ids + rnd)
z = m.encode_z(x)
idx, zh = m.vq.z_to_idx(z, return_zhats=True)
dec = [m.decoder(h).argmax(-1) for h in zh]
out = dict(ids=x.numpy(), z=z.numpy(), idx=idx.numpy(), zh=torch.stack(zh).numpy(), dec=torch.stack(dec).numpy(),
           dec_full=m.decode(idx).argmax(-1).numpy())
np.savez(f"{WORK}/ref_{task}.npz", **out)
print(task, "idx", idx.shape, "EOS", EOS, "PAD", PAD, "L", L)
print("torch roundtrip exact tokens (text rows):", (out["dec"][-1][:len(texts)] == out["ids"][:len(texts)]).mean())
