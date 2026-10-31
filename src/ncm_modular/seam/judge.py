"""LM judge (LFM2-1.2B, mlx_lm; path/HF id via NCM_JUDGE_MODEL). Scores seam windows vs within-block windows.
Window = 16 judge-tokens | 16 judge-tokens. seam: last16(block n) | first16(block n+1). within: 32 contiguous tokens from the middle of one block.
Reports: win_ppl (tokens 1..31), cond_ppl (second half given first half; i.e. seam NLL of the next block's opening given the previous block's end)."""
import sys, json, glob, os, numpy as np, mlx.core as mx
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from paths import *
from mlx_lm import load
M = os.path.expanduser(JUDGE_MODEL)
model, tok = load(M)
enc = lambda t: tok.encode(t, add_special_tokens=False)
HALF = 16
def score(wins, bs=64):
    out = []
    for i in range(0, len(wins), bs):
        x = mx.array(wins[i:i + bs]); lg = model(x[:, :-1]).astype(mx.float32)
        nll = -(mx.take_along_axis(lg - mx.logsumexp(lg, -1, keepdims=True), x[:, 1:, None], -1)[..., 0]); mx.eval(nll)
        out.append(np.array(nll))
    return np.concatenate(out) if out else np.zeros((0, 2 * HALF - 1))
def seam_win(a, b):
    A, B = enc(a), enc(" " + b)
    return A[-HALF:] + B[:HALF] if len(A) >= HALF and len(B) >= HALF else None
def within_win(b):
    B = enc(b)
    if len(B) < 2 * HALF + 4: return None
    s = max(0, len(B) // 2 - HALF); return B[s:s + 2 * HALF]
def stats(nll): 
    return dict(n=int(len(nll)), win_ppl=float(np.exp(nll.mean())), cond_ppl=float(np.exp(nll[:, HALF - 1:].mean())), first_half_ppl=float(np.exp(nll[:, :HALF - 1].mean())))
res = {}
def do(name, blocks_list):
    """blocks_list: list of stories (list of block texts)"""
    sw = [[], [], [], []]; ww = []
    for bl in blocks_list:
        for n in range(len(bl) - 1):
            w = seam_win(bl[n], bl[n + 1])
            if w: sw[min(n, 3)].append(w)
        for b in bl[1:]:
            w = within_win(b)
            if w: ww.append(w)
    allseam = [w for s in sw for w in s]
    res[name] = dict(seam_all=stats(score(allseam)), seam0=stats(score(sw[0])), within=stats(score(ww)))
    print(name, json.dumps(res[name]), flush=True)
files = sorted(glob.glob(f"{RUNS}/gen_*.json"))
first = json.load(open(files[0]))
do("gold", [s["gold"] for s in first["stories"]])
# control: shuffled seam (block n followed by the opening of a block from ANOTHER story) = worst-case seam
import random; r = random.Random(0); gl = [s["gold"] for s in first["stories"]]
do("gold_shuffled_next", [[g[0], r.choice(gl)[1]] for g in gl for _ in range(4)])
for f in files:
    d = json.load(open(f)); do(f"{d['v']}_t{d['temp']}", [s["gen"] for s in d["stories"]])
json.dump(res, open(f"{RUNS}/judge.json", "w"), indent=1)
