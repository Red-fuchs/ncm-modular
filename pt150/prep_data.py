"""Stream FineWeb-Edu from HF, tokenize with GPT-2 BPE (+EOS per doc), write uint16 shards: val.bin first, then train_XXX.bin.
usage: python prep_data.py --out data/fineweb_edu_gpt2 --max-tokens 2e8 [--val-tokens 2.6e5]"""
import argparse, os, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--max-tokens", type=float, default=2e8)
ap.add_argument("--val-tokens", type=float, default=262144); ap.add_argument("--shard-tokens", type=float, default=1e8)
ap.add_argument("--name", default="HuggingFaceFW/fineweb-edu"); ap.add_argument("--config", default="sample-10BT")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained("gpt2"); EOS = tok.eos_token_id
ds = load_dataset(a.name, a.config, split="train", streaming=True)
buf, total, shard, tgt = [], 0, 0, int(a.val_tokens)
name = lambda: os.path.join(a.out, "val.bin" if shard < 0 else f"train_{shard:03d}.bin")
shard = -1
def flush(n):
    global buf, shard, total
    arr = np.array(buf[:n], dtype=np.uint16); buf = buf[n:]
    arr.tofile(name()); total += n; print(name(), n, flush=True)
for ex in ds:
    buf += tok.encode(ex["text"]) + [EOS]
    while len(buf) >= tgt:
        flush(tgt)
        if shard == -1: shard, tgt = 0, int(a.shard_tokens)
        else: shard += 1
    if total >= a.max_tokens: break
