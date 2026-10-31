"""usage: train_seam.py --v a|b|c --out DIR [--steps N] [--n 64 overfit]."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mlx.core as mx, mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_map
from gen2 import *
from data import *
ap = argparse.ArgumentParser()
ap.add_argument("--v", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--minutes", type=float, default=1e9)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--corrupt", type=float, default=0.1); ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--fp16", type=int, default=1); ap.add_argument("--scale", type=float, default=1024.)
ap.add_argument('--cpu', action='store_true'); ap.add_argument('--train', default='train'); ap.add_argument('--init', default=''); ap.add_argument('--warm', type=int, default=100)
a = ap.parse_args(); V_ = a.v
if a.cpu: mx.set_default_device(mx.cpu)
os.makedirs(a.out, exist_ok=True)
fc = FrozenCodec()
def split_levels(flat):
    o, out = 0, []
    for l in fc.levels: out.append(flat[:, o:o + l]); o += l
    return out
def loss_fn(gen, flat, p, corrupt):
    codes = split_levels(flat); seen = codes
    if corrupt > 0:
        seen = []
        for k, c in enumerate(codes):
            if k < fc.K - 1: c = mx.where(mx.random.uniform(shape=c.shape) < corrupt, mx.random.randint(0, 4096, c.shape), c)
            seen.append(c)
    inputs = [mx.stop_gradient(x) for x in fc.teacher_inputs(seen)]
    logits = gen(inputs, p)
    ce = [nn.losses.cross_entropy(lg, c, reduction="sum") for lg, c in zip(logits[1:], codes[1:])]
    return sum(ce) / (flat.shape[0] * flat.shape[1])
def dev_losses(gen, flat, p, bs=64):
    """teacher-forced CE per code, total and levels 0-1 only."""
    tot = c01 = 0.0; n = 0
    for i in range(0, len(flat), bs):
        f, pp = mx.array(flat[i:i + bs]), mx.array(p[i:i + bs]); codes = split_levels(f)
        inputs = fc.teacher_inputs(codes); lg = gen(inputs, pp)
        ce = [nn.losses.cross_entropy(l, c, reduction="sum") for l, c in zip(lg[1:], codes[1:])]
        t = sum(ce) / f.shape[1]; q = ce[0] / 8
        mx.eval(t, q); tot += t.sum().item(); c01 += q.sum().item(); n += f.shape[0]
    return tot / n, c01 / n
def exact(gen, p, tgt_ids, c0, n=128):
    ex = 0; n = min(n, len(p))
    for i in range(0, n, 64):
        toks, _ = generate(gen, fc, mx.array(p[i:i + 64][:n - i]), 0.0, mx.array(c0[i:i + 64][:n - i, :1].astype(np.int32))); mx.eval(toks)
        for r, ref in zip(toks.tolist(), tgt_ids[i:i + 64]):
            r = r[:r.index(EOS)] if EOS in r else r; ref = list(ref); ref = ref[:ref.index(EOS)]
            ex += [x for x in r if x != PAD] == ref
    return ex / n
def main():
    mx.random.seed(0); np.random.seed(0)
    S, arr, codes, sidx, bidx = load_split(a.train, fc, limit=a.n)
    ptr, ttr = pairs(V_, S, arr, codes, sidx, bidx)
    Sd, ad, cd_, sd_, bd_ = load_split("dev", fc, limit=40 if a.n else 0); pdv, tdv = pairs(V_, Sd, ad, cd_, sd_, bd_)
    rs = np.random.RandomState(1).permutation(len(pdv))[:400]; pdv, tdv = pdv[rs], tdv[rs]
    if a.n: ptr, ttr = ptr[:a.n], ttr[:a.n]
    ctr = codes.astype(np.int32)
    gen = Gen(fc, V_, drop=a.drop)
    if a.init: gen.load_weights(a.init); print('resumed from', a.init, flush=True)
    print(f"variant {V_} P={gen.P} params {count_params(gen)/1e6:.2f}M pairs {len(ptr)}", flush=True)
    sched = optim.join_schedules([optim.linear_schedule(0, a.lr, a.warm), optim.cosine_decay(a.lr, max(a.steps - a.warm, 1), a.lr * 0.05)], [a.warm])
    opt = optim.AdamW(sched, weight_decay=0.05, betas=(0.9, 0.98))
    g16 = Gen(fc, V_, drop=a.drop).set_compute(mx.float16 if a.fp16 else mx.float32)
    SC = a.scale if a.fp16 else 1.0
    vg = nn.value_and_grad(g16, lambda g, f, p: loss_fn(g, f, p, a.corrupt) * SC)
    cast = (lambda t: t.astype(mx.float16)) if a.fp16 else (lambda t: t)
    skipped = [0]
    def step(f, p):
        g16.update(tree_map(cast, gen.parameters()))
        loss, grads = vg(g16, f, p)
        grads = tree_map(lambda t: t.astype(mx.float32) / SC, grads)
        grads, gn = optim.clip_grad_norm(grads, 1.0); ok = mx.isfinite(gn)
        grads = tree_map(lambda t: mx.where(ok, t, 0.0), grads)
        opt.update(gen, grads); return loss / SC, ok
    state = [gen.state, opt.state, mx.random.state, g16.state]
    N = len(ptr); t0 = time.time(); ema = None; best = 1e9; curve = []
    gen.train()
    for s in range(a.steps):
        idx = np.random.randint(0, N, a.bs)
        loss, ok = step(mx.array(ctr[ttr[idx]]), mx.array(ptr[idx])); mx.eval(state, loss, ok); skipped[0] += (not ok.item())
        l = loss.item(); ema = l if ema is None else .98 * ema + .02 * l
        if (s + 1) % 25 == 0:
            el = time.time() - t0; print(f"step {s+1} loss {ema:.4f} {a.bs*(s+1)/el:.1f} ex/s skipped {skipped[0]} {el/60:.1f}m", flush=True)
        last = (s + 1) == a.steps or time.time() - t0 > a.minutes * 60
        if (s + 1) % a.eval_every == 0 or last:
            g16.update(tree_map(cast, gen.parameters())); g16.eval()
            if a.n: r = dict(step=s + 1, loss=ema, train_exact=exact(g16, ptr, arr[ttr], codes[ttr], a.n))
            else:
                dl, d01 = dev_losses(g16, cd_[tdv].astype(np.int32), pdv); r = dict(step=s + 1, loss=ema, dev_loss=dl, dev_c1=d01, dev_exact128_gold_l0=exact(g16, pdv, ad[tdv], cd_[tdv]))
            g16.train(); curve.append(r); print("  ", r, flush=True); json.dump(curve, open(f"{a.out}/curve.json", "w"))
            if not a.n and r["dev_loss"] < best: best = r["dev_loss"]; gen.save_weights(f"{a.out}/best.safetensors"); json.dump(r, open(f"{a.out}/best.json", "w"))
        if last: break
    gen.save_weights(f"{a.out}/last.safetensors")
    print("done", (time.time() - t0) / 60, "min")
if __name__ == "__main__": main()
