"""Test-set metrics + latency for NCM generator, AR baseline, intent classifier."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
os.environ.setdefault("MFMT", "C")
import numpy as np, mlx.core as mx
from transformers import AutoTokenizer
from gen_mlx import *
from ar_mlx import *
from massive_eval import score
ap = argparse.ArgumentParser()
ap.add_argument("--ncm", default=WORK+"/ncm_pilot"); ap.add_argument("--ar", default=WORK+"/ar_pilot"); ap.add_argument("--cls", default=WORK+"/cls_pilot")
ap.add_argument("--split", default="test"); ap.add_argument("--out", default=WORK+"/massive_de/results.json"); ap.add_argument("--P", type=int, default=40)
ap.add_argument("--nreq", type=int, default=200)
a = ap.parse_args()
EOS, PAD = 151645, 151643
tok = AutoTokenizer.from_pretrained(WORK+"/tok/Qwen3-0.6B"); fc = FrozenCodec()
D = [json.loads(l) for l in open(f"{WORK}/massive_de/{a.split}.jsonl")]
def pad(ids, n): ids = ids[:n]; return ids + [PAD] * (n - len(ids))
PT = np.array([pad(tok.encode(d["prefix"])[:a.P - 1] + [EOS], a.P) for d in D], np.int32)
def cut(r):
    r = list(r); r = r[:r.index(EOS)] if EOS in r else r
    return tok.decode([t for t in r if t != PAD], skip_special_tokens=True)
def lat(fn, bs, n):
    ts = []
    rng = np.random.RandomState(0)
    for i in range(n + 3):
        idx = rng.randint(0, len(D), bs); t = time.perf_counter(); fn(mx.array(PT[idx])); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts[3:]); return dict(p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)), mean=float(ts.mean()))
res = {}
# ---- NCM
cfg = json.load(open(f"{a.ncm}/ckpt.json"))["args"]
master = Gen(fc, cfg["P"], cfg["D"], cfg["H"], cfg["nl"], PAD, drop=0.0); master.load_weights(f"{a.ncm}/best.safetensors")
g = Gen(fc, cfg["P"], cfg["D"], cfg["H"], cfg["nl"], PAD, drop=0.0).set_compute(mx.float16)
from mlx.utils import tree_map
g.update(tree_map(lambda t: t.astype(mx.float16), master.parameters())); g.eval(); mx.eval(g.parameters())
def ncm_gen(p): t, _ = generate(g, fc, p); mx.eval(t); return t
preds = []
for i in range(0, len(D), 64): preds += [cut(r) for r in ncm_gen(mx.array(PT[i:i + 64])).tolist()]
r = score(preds, D); r["NFE"] = g.K; r["params_M"] = count_params(master) / 1e6
r["lat_b1_ms"] = lat(ncm_gen, 1, a.nreq); r["lat_b16_ms"] = lat(ncm_gen, 16, max(a.nreq // 5, 20)); res["ncm"] = r
res["ncm_codec_ceiling_exact"] = None
json.dump({"preds_ncm": preds[:50]}, open(a.out.replace(".json", "_samples_ncm.json"), "w"), ensure_ascii=False, indent=1)
print("NCM", r, flush=True)
# ---- AR
if os.path.exists(f"{a.ar}/model.safetensors"):
    voc = json.load(open(f"{a.ar}/vocab.json")); ck = json.load(open(f"{a.ar}/ckpt.json")); inv_v = {i + 3: t for i, t in enumerate(voc["vocab"])}
    m = ARGen(fc.tok, ck["V"], ck["args"]["P"], ck["args"]["T"], pad_id=PAD, drop=0.0); m.load_weights(f"{a.ar}/best.safetensors")
    m.set_dtype(mx.float16); m.eval(); mx.eval(m.parameters())
    nfe = []
    def ar_gen(p): o, n = m.generate(p, 1); mx.eval(o); nfe.append(n); return o
    def dec(row):
        row = row[:row.index(1)] if 1 in row else row
        return tok.decode([inv_v[c] for c in row if c in inv_v], skip_special_tokens=True)
    preds = []
    for i in range(0, len(D), 64): preds += [dec(r) for r in ar_gen(mx.array(PT[i:i + 64])).tolist()]
    r = score(preds, D); r["NFE_mean_b64"] = float(np.mean(nfe)); r["params_M"] = count_params(m) / 1e6
    nfe.clear(); r["lat_b1_ms"] = lat(ar_gen, 1, a.nreq); r["NFE_mean_b1"] = float(np.mean(nfe)); nfe.clear()
    r["lat_b16_ms"] = lat(ar_gen, 16, max(a.nreq // 5, 20)); r["NFE_mean_b16"] = float(np.mean(nfe)); res["ar"] = r
    json.dump({"preds_ar": preds[:50]}, open(a.out.replace(".json", "_samples_ar.json"), "w"), ensure_ascii=False, indent=1)
    print("AR", r, flush=True)
# ---- classifier
if os.path.exists(f"{a.cls}/model.safetensors"):
    ints = json.load(open(f"{a.cls}/vocab.json"))["ints"]
    m = IntentCls(fc.tok, len(ints), a.P, pad_id=PAD, drop=0.0); m.load_weights(f"{a.cls}/best.safetensors"); m.set_dtype(mx.float16); m.eval()
    def cl(p): o = mx.argmax(m(p), -1); mx.eval(o); return o
    hit = 0
    for i in range(0, len(D), 256): hit += sum(ints[j] == d["intent"] for j, d in zip(cl(mx.array(PT[i:i + 256])).tolist(), D[i:i + 256]))
    res["cls"] = dict(intent_acc=hit / len(D), params_M=count_params(m) / 1e6, NFE=1, lat_b1_ms=lat(cl, 1, a.nreq), lat_b16_ms=lat(cl, 16, max(a.nreq // 5, 20)))
    print("CLS", res["cls"], flush=True)
json.dump(res, open(a.out, "w"), indent=1)
