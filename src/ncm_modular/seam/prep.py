"""Build masked, sentence-blocked TinyStories splits. out: data/{train,dev,test}.json (list of stories) + data/blocks_*.npz."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories import *
from ncm_mlx.official.bpe import BPETokenizer
L, EOS, PAD = 128, 0, 1
bpe = BPETokenizer.load(f"{WORK}/ts/bpe.json"); enc = lambda t: bpe.encode(t)
print({s: enc(" " + s) for s in SUR}, {s: enc(s) for s in SUR})
raw = load_stories(); idx_tr, idx_dv, idx_te = split(raw)
def build(ids, min_blocks=2):
    out = []
    for i in ids:
        m = mask_story(raw[i])
        if not m: continue
        txt, ents, mp = m
        bl = blocks(txt, enc, L)
        if len(bl) < min_blocks: continue
        out.append(dict(sid=i, blocks=bl, table=ents, nraw=len(mp)))
    return out
import collections
tr = build(idx_tr)
# dev/test: long stories only (>=5 blocks => gold block 0 + 4 generated), dev small
dv_all, te_all = build(idx_dv), build(idx_te)
dv = [s for s in dv_all if len(s["blocks"]) >= 5][:150]; te = [s for s in te_all if len(s["blocks"]) >= 5][:250]
dvp = [s for s in dv_all if len(s["blocks"]) >= 2]      # dev pairs for teacher-forced dev loss (any length)
print("blocks/story hist (train)", sorted(collections.Counter(len(s["blocks"]) for s in tr).items()))
print("train stories", len(tr), "pairs", sum(len(s["blocks"]) - 1 for s in tr), "dev(long)", len(dv), "dev(all)", len(dvp), "test(long)", len(te), "of", len(te_all))
print("avg block toks", np.mean([len(enc(b)) for s in tr[:500] for b in s["blocks"]]))
for n, d in [("train", tr), ("dev", dvp), ("devgen", dv), ("test", te)]:
    json.dump(d, open(f"{DATA}/{n}.json", "w"))
