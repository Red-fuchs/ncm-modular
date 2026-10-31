"""Train AR baseline (--mode ar) or intent classifier (--mode cls) on MASSIVE."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
import numpy as np, mlx.core as mx, mlx.optimizers as optim
from mlx import nn
from transformers import AutoTokenizer
from gen_mlx import FrozenCodec
from ar_mlx import *
ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="ar"); ap.add_argument("--data", default=WORK+"/massive_de"); ap.add_argument("--out", default=WORK+"/ar_run")
ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--P", type=int, default=48); ap.add_argument("--T", type=int, default=48); ap.add_argument("--n", type=int, default=0)
ap.add_argument("--minutes", type=float, default=1e9); ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--eval_every", type=int, default=500)
a = ap.parse_args()
EOS, PAD = 151645, 151643
os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(WORK+"/tok/Qwen3-0.6B"); fc = FrozenCodec()
load = lambda sp: [json.loads(l) for l in open(f"{a.data}/{sp}.jsonl")]
def pad(ids, n): ids = ids[:n]; return ids + [PAD] * (n - len(ids))
def prefix_arr(D): return np.array([pad(tok.encode(d["prefix"])[:a.P - 1] + [EOS], a.P) for d in D], np.int32)
Dtr, Ddev = load("train")[: a.n or None], load("dev")
INTS = sorted({d["intent"] for d in load("train")})
ttr = [tok.encode(d["target"]) for d in Dtr]
vocab = sorted({t for x in ttr for t in x}); c2 = {t: i + 3 for i, t in enumerate(vocab)}   # 0 BOS, 1 EOS, 2 UNK
V = len(vocab) + 3; inv_v = {v: k for k, v in c2.items()}
json.dump({"vocab": vocab, "ints": INTS}, open(f"{a.out}/vocab.json", "w"))
ptr = prefix_arr(Dtr); pdev = prefix_arr(Ddev)
def tgt_arr(ids):
    return np.array([[0] + [c2.get(t, 2) for t in x][: a.T - 1] + [1] + [1] * (a.T - 1 - min(len(x), a.T - 1)) for x in ids], np.int32)   # (N, T+1): BOS..EOS, padded w/ EOS
def tmask(ids): return np.array([[1] * (min(len(x), a.T - 1) + 1) + [0] * (a.T - 1 - min(len(x), a.T - 1)) for x in ids], np.float32)
if a.mode == "ar":
    Ttr = tgt_arr(ttr); Mtr = tmask(ttr)
    model = ARGen(fc.tok, V, a.P, a.T, pad_id=PAD, drop=a.drop)
    def loss_fn(m, p, t, w):
        lg = m(p, t[:, :-1]); ce = nn.losses.cross_entropy(lg, t[:, 1:], reduction="none")
        return (ce * w).sum() / w.sum()
else:
    Ytr = np.array([INTS.index(d["intent"]) for d in Dtr], np.int32); Ydev = np.array([INTS.index(d["intent"]) if d["intent"] in INTS else -1 for d in Ddev])
    model = IntentCls(fc.tok, len(INTS), a.P, pad_id=PAD, drop=a.drop)
    def loss_fn(m, p, y, w): return nn.losses.cross_entropy(m(p), y, reduction="mean")
from gen_mlx import count_params
print("params", count_params(model) / 1e6, "M V", V, "N", len(Dtr), flush=True)
mx.random.seed(0); np.random.seed(0)
sched = optim.join_schedules([optim.linear_schedule(0, a.lr, 200), optim.cosine_decay(a.lr, max(a.steps - 200, 1), a.lr * 0.05)], [200])
opt = optim.AdamW(sched, weight_decay=0.05, betas=(0.9, 0.98))
vg = nn.value_and_grad(model, loss_fn); state = [model.state, opt.state, mx.random.state]
@mx.compile
def _s(p, t, w): pass
def step(p, t, w):
    loss, g = vg(model, p, t, w); g, _ = optim.clip_grad_norm(g, 1.0); opt.update(model, g); return loss
step = mx.compile(step, inputs=state, outputs=state)
Pdv = mx.array(pdev[:300])
def dev_loss():
    if a.mode == "cls": return nn.losses.cross_entropy(model(mx.array(pdev)), mx.array(np.maximum(Ydev, 0)), reduction="mean").item()
    Td, Md = tgt_arr([tok.encode(d["target"]) for d in Ddev[:300]]), tmask([tok.encode(d["target"]) for d in Ddev[:300]])
    return loss_fn(model, Pdv, mx.array(Td), mx.array(Md)).item()
best = [-1.0]; curve = []
def evaluate():
    model.eval()
    if a.mode == "cls":
        hit = 0
        for i in range(0, len(Ddev), 256): hit += (mx.argmax(model(mx.array(pdev[i:i+256])), -1) == mx.array(Ydev[i:i+256])).sum().item()
        r = {"dev_intent_acc": hit / len(Ddev), "dev_loss": dev_loss()}; model.train(); return r
    n = min(512, len(Ddev)); ex = 0
    for i in range(0, n, 128):
        out, _ = model.generate(mx.array(pdev[i:i+128][: n - i]), 1); mx.eval(out)
        for r, d in zip(out.tolist(), Ddev[i:i+128]):
            r = r[:r.index(1)] if 1 in r else r
            ex += tok.decode([inv_v.get(c, 0) for c in r if c > 2]) == d["target"]
    n2 = n; r = {"dev_target_exact_512": ex / n, "dev_loss": dev_loss()}; model.train(); return r
t0 = time.time(); ema = None; model.train()
for s in range(a.steps):
    idx = np.random.randint(0, len(Dtr), a.bs)
    if a.mode == "ar": loss = step(mx.array(ptr[idx]), mx.array(Ttr[idx]), mx.array(Mtr[idx]))
    else: loss = step(mx.array(ptr[idx]), mx.array(Ytr[idx]), mx.zeros(1))
    mx.eval(state, loss); l = loss.item(); ema = l if ema is None else .98 * ema + .02 * l
    if (s + 1) % 100 == 0: print(f"step {s+1} loss {ema:.4f} {(time.time()-t0)/60:.1f}m", flush=True)
    if (s + 1) % a.eval_every == 0 or s + 1 == a.steps:
        r = evaluate(); r["step"] = s + 1; curve.append(r); print("  ", r, flush=True)
        sc = list(r.values())[0] - 1e-3 * r["dev_loss"]
        if sc > best[0]: best[0] = sc; model.save_weights(f"{a.out}/best.safetensors"); json.dump(r, open(f"{a.out}/best.json", "w"))
        json.dump(curve, open(f"{a.out}/dev_curve.json", "w"))
        model.save_weights(f"{a.out}/model.safetensors")
        json.dump({"step": s + 1, "args": vars(a), "V": V}, open(f"{a.out}/ckpt.json", "w"))
    if time.time() - t0 > a.minutes * 60: print("time budget"); break
model.save_weights(f"{a.out}/model.safetensors"); json.dump({"step": s + 1, "args": vars(a), "V": V}, open(f"{a.out}/ckpt.json", "w"))
print("train time min", (time.time() - t0) / 60)
