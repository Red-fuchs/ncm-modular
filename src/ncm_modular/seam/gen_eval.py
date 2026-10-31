"""generate NB blocks per held-out story, conditioning on gold block 0. usage: gen_eval.py --runs runs --temps 0 0.7 ; writes runs/gen_{v}_t{T}.json (+ gold)"""
import os, sys, json, time, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mlx.core as mx
from gen2 import *
from data import tok_block
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default=f"{RUNS}"); ap.add_argument("--temps", type=float, nargs="+", default=[0.0])
ap.add_argument("--nb", type=int, default=4); ap.add_argument("--variants", default="abc"); ap.add_argument("--ckpt", default="best"); ap.add_argument("--bs", type=int, default=50); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--cpu", action="store_true")
a = ap.parse_args()
if a.cpu: mx.set_default_device(mx.cpu)
SR = re.compile(r"(?<![\w'])(?:John|Mary|David|Sarah|James|Peter|Emma|Mark)(?![\w'])")
fc = FrozenCodec()
S = json.load(open(f"{DATA}/test.json"))
if a.limit: S = S[:a.limit]
def dec(ids): return bpe.decode([t for t in ids if t not in (EOS, PAD)], skip_special=True)
def trim(r): r = r[:r.index(EOS)] if EOS in r else r; return [t for t in r if t != PAD]
def enc_codes(ids_list):
    x = mx.array([i[:L - 1] + [EOS] + [PAD] * (L - 1 - len(i[:L - 1])) for i in ids_list]); cs = fc.encode(x[:, :L]); mx.eval(cs)
    return np.array(mx.concatenate(cs[:2], 1))          # (B, 9)
def run(v, temp):
    gen = Gen(fc, v, drop=0).set_compute(mx.float16); gen.load_weights(f"{a.runs}/{v}/{a.ckpt}.safetensors"); gen.set_dtype(mx.float16); gen.eval()
    res = []; tms = []
    for i0 in range(0, len(S), a.bs):
        B = S[i0:i0 + a.bs]
        ids = [[tok_block(st["blocks"][0])[:-1]] for st in B]            # per story: list of block id-lists
        codes = [[c] for c in enc_codes([x[0] for x in ids])]
        for n in range(a.nb):
            t0 = time.time()
            P = np.stack([make_prefix(v, ids[j][-1], list(dict.fromkeys(m for blk in ids[j] for m in SR.findall(dec(blk)))), codes[j]) for j in range(len(B))])
            toks, _ = generate(gen, fc, mx.array(P), temp); mx.eval(toks); tms.append((time.time() - t0) * 1000)
            new = [trim(r) for r in toks.tolist()]
            nc = enc_codes([x if x else [PAD] for x in new])
            for j in range(len(B)): ids[j].append(new[j]); codes[j].append(nc[j])
        for j, st in enumerate(B): res.append(dict(sid=st["sid"], gold=st["blocks"][:1 + a.nb], gen=[dec(b) for b in ids[j]], table=st["table"]))
    # bs=1 latency
    lat = []
    for j in range(2, min(22, len(S))):
        st = S[j]; pid = tok_block(st["blocks"][0])[:-1]; h = enc_codes([pid])
        t0 = time.time(); p = make_prefix(v, pid, [m for m in dict.fromkeys(SR.findall(st["blocks"][0]))], [h[0]])
        toks, _ = generate(gen, fc, mx.array(p[None]), temp); mx.eval(toks); lat.append((time.time() - t0) * 1000)
    json.dump(dict(v=v, temp=temp, stories=res, batch_ms_per_block=float(np.mean(tms[1:])), lat_b1_ms=float(np.median(lat[2:])), NFE=fc.K, P=gen.P), open(f"{a.runs}/gen_{v}_t{temp}.json", "w"))
    print(v, temp, "done; b1 ms", np.median(lat[2:]), "batch ms/block", np.mean(tms[1:]), flush=True)
for t in a.temps:
    for v in a.variants: run(v, t)
