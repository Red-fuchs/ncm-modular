"""Long-output eval. python -m ncm150.eval_long --ckpt runs/ncm/last.pt --ref runs/ar/last.pt --modes none mask slot --out res.json
Metrics (all over generated blocks only; names = single-token entries of the English name list):
 invented_rate   name mentions whose name was not in the table (prompt + earlier blocks) / all name mentions
 invented_texts  share of texts with >=1 invented name
 carry_over      share of prompt-table names that reappear in the generated text (texts with a non-empty prompt table)
 seam_doubling   share of block seams (and prompt->block1) where last word == first word of the next block
 ppl_ref         within-text PPL of the generated tokens under the reference AR model (sliding window, same K)
 name_hit_rate   share of decode positions where the unconstrained argmax was a disallowed name"""
import argparse, json, math
import numpy as np, torch, torch.nn.functional as F
from ncm150.generate import GenCfg, generate_long, load_model
from ncm150.safe import build_nameset, synthetic_nameset, update_table

def invented(table0, blocks, ns):
    """-> (invented mentions, mentions) over generated blocks; table grows block by block"""
    tab, inv, tot = list(table0), 0, 0
    for b in blocks:
        for t in b:
            k = ns.tok2key.get(int(t))
            if k is None: continue
            tot += 1; inv += k not in tab
        tab = update_table(tab, b, ns)
    return inv, tot

def carry_over(table0, blocks, ns):
    seen = set(update_table([], [t for b in blocks for t in b], ns))
    return (sum(k in seen for k in table0), len(table0))

def seam_doubling(prompt, blocks, words):
    chunks = [prompt] + blocks; d = n = 0
    for a, b in zip(chunks, chunks[1:]):
        wa, wb = words(a), words(b)
        if wa and wb: n += 1; d += wa[-1].lower() == wb[0].lower()
    return d, n

@torch.no_grad()
def ppl_ar(ar, cfg, prompt, blocks, K, W):
    """NLL of every generated token given [BOS] + last K tokens before the block (+ earlier tokens of the same block, window W)"""
    dev = next(ar.parameters()).device; hist, nll, n = list(prompt), 0.0, 0
    for b in blocks:
        if not b: continue
        ctx = hist[-K:] if K else []; x = torch.tensor([[cfg.data.eos_id] + ctx + b[:-1]], device=dev)[:, -(K + W):]
        lg = ar(x)[0, -len(b):].float()
        nll += F.cross_entropy(lg, torch.tensor(b, device=dev), reduction="sum").item(); n += len(b); hist += b
    return nll, n

def words_fn(vocab):
    try:
        import tiktoken; enc = tiktoken.get_encoding("gpt2")
        return lambda ids: enc.decode([t for t in ids if t < 50257]).split()
    except Exception: return lambda ids: [str(t) for t in ids]

def get_prompts(cfg, n, plen, source, seed):
    if source == "random":
        r = np.random.default_rng(seed); return r.integers(0, cfg.data.vocab - 2, (n, plen)).tolist()
    from ncm150.data import WindowStream
    s = WindowStream(cfg.data.dir, plen, seed, "val"); return s.batch(0, min(n, s.n)).tolist()

def evaluate(net, codec, cfg, prompts, ns, g, ref=None, ref_cfg=None, words=None):
    r = generate_long(net, codec, cfg, prompts, ns, g); K, W = int(cfg.data.get("prefix") or 0), int(cfg.data.window)
    words = words or (lambda ids: [str(t) for t in ids])
    iv = np.array([invented(t0, b, ns) for t0, b in zip(r["table0"], r["blocks"])])
    co = np.array([carry_over(t0, b, ns) for t0, b in zip(r["table0"], r["blocks"])]); co = co[co[:, 1] > 0]
    sd = np.array([seam_doubling(p, b, words) for p, b in zip(prompts, r["blocks"])])
    out = dict(mode=g.mode, n_texts=len(prompts), n_blocks=g.n_blocks, tokens=int(sum(len(x) for b in r["blocks"] for x in b)),
               invented_rate=float(iv[:, 0].sum() / max(iv[:, 1].sum(), 1)), name_mentions=int(iv[:, 1].sum()),
               invented_texts=float((iv[:, 0] > 0).mean()),
               carry_over=float(co[:, 0].sum() / co[:, 1].sum()) if len(co) else None,
               seam_doubling=float(sd[:, 0].sum() / max(sd[:, 1].sum(), 1)),
               name_hit_rate=r["hits"] / max(len(prompts) * g.n_blocks * W, 1))
    if ref is not None:
        s = np.array([ppl_ar(ref, ref_cfg, p, b, int(ref_cfg.data.get("prefix") or 0), int(ref_cfg.data.window)) for p, b in zip(prompts, r["blocks"])])
        out["ppl_ref"] = float(math.exp(s[:, 0].sum() / max(s[:, 1].sum(), 1)))
    return out, r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--config", nargs="+"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ref", help="AR checkpoint for ppl_ref (default: the model itself if it is AR)"); ap.add_argument("--ref-config", nargs="+")
    ap.add_argument("--modes", nargs="+", default=["none", "mask", "slot"]); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=4); ap.add_argument("--prompt-len", type=int); ap.add_argument("--prompts", default="val", choices=["val", "random"])
    ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--table-slots", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-names", action="store_true", help="tests/tiny vocab: ids 10.. are names"); ap.add_argument("--device", default="auto")
    ap.add_argument("--out"); ap.add_argument("--dump", help="also write generated token ids here")
    a = ap.parse_args()
    cfg, net, codec = load_model(a.ckpt, a.config, a.set, a.device)
    ref = ref_cfg = None
    if a.ref or a.ref_config: ref_cfg, ref, _ = load_model(a.ref, a.ref_config, [*a.set, "arm=ar"] if not a.ref else [], a.device)
    elif cfg.arm == "ar": ref_cfg, ref = cfg, net
    if cfg.data.vocab >= 50257 and not a.synthetic_names:
        import tiktoken; enc = tiktoken.get_encoding("gpt2"); ns = build_nameset(lambda s: enc.encode_ordinary(s), cfg.data.vocab)
    else: ns = synthetic_nameset(cfg.data.vocab)
    K = int(cfg.data.get("prefix") or 0); prompts = get_prompts(cfg, a.n, a.prompt_len or max(K, cfg.data.window), a.prompts, a.seed)
    res = dict(arm=cfg.arm, ckpt=a.ckpt, names=len(ns.ids), runs=[], dump={})
    for m in a.modes:
        g = GenCfg(n_blocks=a.blocks, mode=m, temp=a.temp, table_slots=a.table_slots, seed=a.seed)
        o, r = evaluate(net, codec, cfg, prompts, ns, g, ref, ref_cfg, words_fn(cfg.data.vocab)); res["runs"].append(o); res["dump"][m] = r["blocks"]
        print(json.dumps(o), flush=True)
    if a.dump: json.dump(res["dump"], open(a.dump, "w"))
    del res["dump"]
    if a.out: json.dump(res, open(a.out, "w"), indent=1)
    return res

if __name__ == "__main__": main()
