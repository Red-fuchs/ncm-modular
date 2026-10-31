"""round trip on masked story blocks for both codecs. usage: rt.py N"""
import sys, os, json, numpy as np, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stories import *
mx.set_default_device(mx.cpu)
N = int(sys.argv[1]); st = [mask_story(s) for s in load_stories()[:3000]]; st = [x[0] for x in st if x][:300]
import difflib
def score(name, blk, enc, dec, codes_fn, dec_fn, L, EOS, PAD):
    ids = [enc(b)[:L-1] + [EOS] for b in blk]; x = mx.array([i + [PAD]*(L-len(i)) for i in ids])
    ex = tk = 0; tot = 0; ent = entn = 0; sims = []
    for i in range(0, len(ids), 64):
        out = dec_fn(codes_fn(x[i:i+64])); mx.eval(out)
        for r, ref, b in zip(out.tolist(), ids[i:i+64], blk[i:i+64]):
            r = r[:r.index(EOS)] if EOS in r else r; ref = ref[:-1]
            ex += r == ref; tk += sum(a == c for a, c in zip(r, ref)); tot += len(ref)
            t = dec(r); sims.append(difflib.SequenceMatcher(None, t, b).ratio())
            for n in SRE.findall(b): entn += 1; ent += n in t
    print(f"{name}: blocks={len(ids)} exact={ex/len(ids):.1%} tokacc={tk/tot:.1%} charsim={np.mean(sims):.3f} surrogate_kept={ent/max(entn,1):.1%}", flush=True)
# ts
from ncm_mlx.official import spec as specs
from ncm_mlx.official.bpe import BPETokenizer
from ncm_mlx.official.model import Codec
D = WORK+"/ts"; task = specs.from_json(json.load(open(f"{D}/spec.json"))["task"])
cd = Codec(task.codec); cd.load_weights(f"{D}/codec.safetensors", strict=True); cd.eval()
bpe = BPETokenizer.load(f"{D}/bpe.json")
enc2 = lambda t: bpe.encode(t)
bl2 = [b for s in st for b in blocks(s, enc2, 128)][:N]
print("ts L=128 avg block tokens", np.mean([len(enc2(b)) for b in bl2]))
score("ts", bl2, enc2, lambda r: bpe.decode(r, skip_special=True), lambda x: cd.encode(x), lambda cs: mx.argmax(cd.decode_logits(cd.cumulative(cs)[-1]), -1), 128, task.eos_id, task.pad_id)
