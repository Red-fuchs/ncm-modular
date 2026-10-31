"""Train prefix-conditioned NCM generator on MASSIVE de-DE pairs (frozen wmt14 codec).
usage: train_gen.py --data w/massive_de --out w/ncm_run --steps N [--n 64 (overfit)] [--minutes M]"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
import numpy as np, mlx.core as mx, mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from transformers import AutoTokenizer
from gen_mlx import *

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=WORK+"/massive_de"); ap.add_argument("--out", default=WORK+"/ncm_run")
ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--minutes", type=float, default=1e9)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--P", type=int, default=48); ap.add_argument("--D", type=int, default=512)
ap.add_argument("--H", type=int, default=8); ap.add_argument("--nl", type=int, default=8)
ap.add_argument("--n", type=int, default=0, help="overfit on first n examples"); ap.add_argument("--drop", type=float, default=0.1)
ap.add_argument("--corrupt", type=float, default=0.0); ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--ckpt_every", type=int, default=500); ap.add_argument("--resume", action="store_true")
ap.add_argument("--compile", type=int, default=0); ap.add_argument("--fp16", type=int, default=1); ap.add_argument("--scale", type=float, default=1024.)
ap.add_argument("--bench", action="store_true", help="throughput only")
a = ap.parse_args()
EOS, PAD, L = 151645, 151643, 64
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(WORK+"/tok/Qwen3-0.6B")
fc = FrozenCodec()

def load(sp): return [json.loads(l) for l in open(f"{a.data}/{sp}.jsonl")]
def pad(ids, n): ids = ids[:n]; return ids + [PAD] * (n - len(ids))
def prefix_arr(D): return np.array([pad(tok.encode(d["prefix"])[:a.P - 1] + [EOS], a.P) for d in D], np.int32)
def target_arr(D): return np.array([pad(tok.encode(d["target"])[:L - 1] + [EOS], L) for d in D], np.int32)
def codes_of(T, bs=128):
    out = []
    for i in range(0, len(T), bs):
        cs = fc.encode(mx.array(T[i:i + bs])); out.append(np.array(mx.concatenate(cs, 1))); mx.clear_cache()
    return np.concatenate(out)

def get_split(sp, n=0):
    cache = f"{a.out}/../cache_{sp}_P{a.P}.npz" if not n else None
    D = load(sp)[: n or None]
    if cache and os.path.exists(cache): z = np.load(cache); return D, z["p"], z["t"], z["c"]
    p, t = prefix_arr(D), target_arr(D); c = codes_of(t)
    if cache: np.savez(cache, p=p, t=t, c=c)
    return D, p, t, c

def split_levels(flat):
    o, out = 0, []
    for l in fc.levels: out.append(flat[:, o:o + l]); o += l
    return out

def loss_fn(gen, flat, ptoks, corrupt):
    codes = split_levels(flat)
    seen = codes
    if corrupt > 0:
        seen = []
        for k, c in enumerate(codes):
            if k < fc.K - 1:
                flip = mx.random.uniform(shape=c.shape) < corrupt
                c = mx.where(flip, mx.random.randint(0, 4096, c.shape), c)
            seen.append(c)
    inputs = [mx.stop_gradient(x) for x in fc.teacher_inputs(seen)]
    logits = gen(inputs, ptoks)
    ce = [nn.losses.cross_entropy(lg, c, reduction="sum") for lg, c in zip(logits, codes)]
    return sum(ce) / (flat.shape[0] * flat.shape[1])

def main():
    mx.random.seed(0); np.random.seed(0)
    Dtr, ptr, ttr, ctr = get_split("train", a.n)
    gen = Gen(fc, a.P, a.D, a.H, a.nl, PAD, drop=a.drop)
    print("params", count_params(gen) / 1e6, "M; train", len(Dtr), flush=True)
    steps0, ck = 0, f"{a.out}/ckpt.safetensors"
    if a.resume and os.path.exists(ck):
        gen.load_weights(ck); steps0 = json.load(open(f"{a.out}/ckpt.json"))["step"]; print("resumed", steps0)
    sched = optim.join_schedules([optim.linear_schedule(0, a.lr, 200), optim.cosine_decay(a.lr, max(a.steps - 200, 1), a.lr * 0.05)], [200])
    opt = optim.AdamW(sched, weight_decay=0.05, betas=(0.9, 0.98)); opt.state["step"] = mx.array(steps0)
    # master fp32 weights `gen` (optimizer); compute copy `g16` (fp16) re-cast from master each step; static-ish loss scale
    g16 = Gen(fc, a.P, a.D, a.H, a.nl, PAD, drop=a.drop).set_compute(mx.float16 if a.fp16 else mx.float32)
    SC = a.scale if a.fp16 else 1.0
    vg = nn.value_and_grad(g16, lambda g, f, p: loss_fn(g, f, p, a.corrupt) * SC)
    cast = (lambda t: t.astype(mx.float16)) if a.fp16 else (lambda t: t)
    from mlx.utils import tree_map
    skipped = [0]
    def step(f, p):
        g16.update(tree_map(cast, gen.parameters()))
        loss, grads = vg(g16, f, p)
        grads = tree_map(lambda t: t.astype(mx.float32) / SC, grads)
        grads, gn = optim.clip_grad_norm(grads, 1.0)
        ok = mx.isfinite(gn)
        grads = tree_map(lambda t: mx.where(ok, t, 0.0), grads)
        opt.update(gen, grads); return loss / SC, ok
    state = [gen.state, opt.state, mx.random.state, g16.state]
    Ddev = load('dev')[:300]; pdev, tdev = prefix_arr(Ddev), target_arr(Ddev); cdev = codes_of(tdev); best = [-1.0]; curve = []
    def dev_loss():
        g16.eval(); l = loss_fn(g16, mx.array(cdev), mx.array(pdev), 0.0).item(); g16.train(); return l
    N = len(Dtr); t0 = time.time(); ema = None; toks = 0
    gen.train()
    for s in range(steps0, a.steps):
        idx = np.random.randint(0, N, a.bs)
        loss, ok = step(mx.array(ctr[idx]), mx.array(ptr[idx])); mx.eval(state, loss, ok); skipped[0] += (not ok.item())
        toks += a.bs * (L + a.P)
        l = loss.item(); ema = l if ema is None else .98 * ema + .02 * l
        if (s + 1) % 50 == 0:
            el = time.time() - t0
            print(f"step {s+1} loss {ema:.4f} {toks/el:.0f} tok/s(raw P+L) {a.bs*(s+1-steps0)/el:.1f} ex/s skipped {skipped[0]} elapsed {el/60:.1f}m", flush=True)
        if a.bench and time.time() - t0 > a.minutes * 60: break
        if (s + 1) % a.ckpt_every == 0 or (s + 1) == a.steps or time.time() - t0 > a.minutes * 60:
            gen.save_weights(ck); json.dump({"step": s + 1, "loss": ema, "args": vars(a)}, open(f"{a.out}/ckpt.json", "w"))
        if (s + 1) % a.eval_every == 0 or (s + 1) == a.steps:
            g16.update(tree_map(cast, gen.parameters())); g16.eval(); acc = quick_eval(g16, Dtr, ptr, ttr); accd = quick_eval(g16, Ddev, pdev, tdev, 300); g16.train(); dl = dev_loss()
            r = dict(step=s + 1, train_loss_ema=ema, train_exact256=acc, dev_exact300=accd, dev_loss=dl); curve.append(r); print("  ", r, flush=True)
            json.dump(curve, open(f"{a.out}/dev_curve.json", "w"))
            if accd - 1e-3 * dl > best[0]: best[0] = accd - 1e-3 * dl; gen.save_weights(f"{a.out}/best.safetensors"); json.dump(r, open(f"{a.out}/best.json", "w"))
        if time.time() - t0 > a.minutes * 60: print("time budget reached"); break
    gen.save_weights(ck); json.dump({"step": s + 1, "loss": ema, "args": vars(a)}, open(f"{a.out}/ckpt.json", "w"))

def quick_eval(gen, D, p, t, n=256):
    n = min(n, len(D)); ex = 0
    for i in range(0, n, 128):
        toks, _ = generate(gen, fc, mx.array(p[i:i + 128][: n - i])); mx.eval(toks)
        for r, ref in zip(toks.tolist(), t[i:i + 128]):
            r = r[:r.index(EOS)] if EOS in r else r; ref = list(ref); ref = ref[:ref.index(EOS)]
            ex += [x for x in r if x != PAD] == ref
    return ex / n

def partial_compile(state):
    import functools
    return lambda f: mx.compile(f, inputs=state, outputs=state)

if __name__ == "__main__": main()
