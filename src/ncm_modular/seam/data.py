"""tokenise blocks, codec codes (cached), pair/prefix arrays per variant."""
import os, json, numpy as np, mlx.core as mx
from gen2 import *
def tok_block(t): return (bpe.encode(t)[:L - 1] + [EOS])
def load_split(name, fc, cache=True, limit=0):
    S = json.load(open(f"{DATA}/{name}.json"))
    if limit: S, cache = S[:limit], False
    ids, sidx, bidx = [], [], []
    for si, s in enumerate(S):
        for bi, b in enumerate(s["blocks"]): ids.append(tok_block(b)); sidx.append(si); bidx.append(bi)
    arr = np.array([i + [PAD] * (L - len(i)) for i in ids], np.int32)
    f = f"{DATA}/codes_{name}.npy"
    if cache and os.path.exists(f) and len(np.load(f, mmap_mode='r')) == len(arr): codes = np.load(f)
    else:
        out = []
        for i in range(0, len(arr), 128):
            cs = fc.encode(mx.array(arr[i:i + 128])); out.append(np.array(mx.concatenate(cs, 1)).astype(np.int16)); mx.clear_cache()
        codes = np.concatenate(out)
        if cache: np.save(f, codes)
    return S, arr, codes, np.array(sidx), np.array(bidx)
def pairs(v, S, arr, codes, sidx, bidx, ntable_from="gold"):
    """one pair per (story, block n>=1): prefix array (P,), target block row."""
    start = {}
    for r, (s, b) in enumerate(zip(sidx, bidx)):
        if b == 0: start[s] = r
    import re
    SR = re.compile(r"(?<![\w'])(?:John|Mary|David|Sarah|James|Peter|Emma|Mark)(?![\w'])")
    P, T = [], []
    for s, st in enumerate(S):
        seen, r0 = [], start[s]
        for n in range(1, len(st["blocks"])):
            for m in SR.findall(st["blocks"][n - 1]):
                if m not in seen: seen.append(m)
            prev = [t for t in arr[r0 + n - 1].tolist() if t not in (EOS, PAD)]
            hist = [codes[r0 + j][:CPB] for j in range(max(0, n - MB), n)]
            P.append(make_prefix(v, prev, seen, hist)); T.append(r0 + n)
    return np.stack(P), np.array(T)
