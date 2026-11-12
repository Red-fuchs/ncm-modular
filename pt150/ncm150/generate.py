"""Long-text generation by sliding window, block by block, for both arms.
Row j: hist (prompt + generated tokens), table (ordered name keys seen). Each step the model sees
[BOS] + table tokens (table_slots) + last K - table_slots raw tokens, and produces one block of `window` tokens:
ncm = level-0 codes (uniform prior) -> NCM levels -> codec decode -> safe-masked argmax; ar = sampled token by token (same mask).
table_slots=0 is checkpoint-compatible (models trained without a table); >0 needs a model trained with table tokens in the prefix."""
import contextlib
from dataclasses import dataclass
import torch
from omegaconf import OmegaConf
from ncm150.config import levels_of, sample_len
from ncm150.models import build_codec, build_model
from ncm150.safe import choose, table_masks, update_table
from ncm150.train import pick_device

@dataclass
class GenCfg:
    n_blocks: int = 4
    mode: str = "none"        # none | mask | slot
    temp: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    table_slots: int = 0
    seed: int = 0

def window_tail(hist, n): return list(hist[-n:]) if n > 0 else []

def build_prefix(hist, table, ns, K, slots, bos, pad=None):
    """[BOS] + table tokens + raw tail. pad=None -> unpadded (AR); else NCM fixed length 1+K: table right-padded, tail left-padded."""
    tab = [ns.canon[k] for k in table][:slots]
    tail = window_tail(hist, K - slots)
    if pad is None: return [bos] + tab + tail
    return [bos] + tab + [pad] * (slots - len(tab)) + [pad] * (K - slots - len(tail)) + tail

def top_kp(lg, temp, top_k, top_p):
    if temp == 0: return lg.argmax(-1)
    lg = lg / temp
    if top_k > 0: lg = lg.masked_fill(lg < lg.topk(min(top_k, lg.shape[-1]), -1).values[..., -1:], -torch.inf)
    sl, si = lg.sort(-1, descending=True); p = sl.softmax(-1)
    sl = sl.masked_fill(p.cumsum(-1) - p > top_p, -torch.inf)
    return torch.multinomial(torch.zeros_like(lg).scatter(-1, si, sl).softmax(-1), 1).squeeze(-1)

def _ncm_block(net, codec, cfg, prefs, bad, ok, g, dev, ac):
    L0 = levels_of(cfg)[0]
    with ac():
        idx = net.generate(torch.randint(0, cfg.codec.vocab, (len(prefs), L0), device=dev), torch.tensor(prefs, device=dev),
                           temperature=g.temp, top_k=min(g.top_k, cfg.codec.vocab), top_p=g.top_p)
        lg = codec.decode(idx).float()
    return choose(lg, g.mode, bad, ok)

def _ar_block(net, cfg, prefs, bad, ok, g, dev, ac):
    pick = lambda l: top_kp(l, g.temp, g.top_k, g.top_p)
    res, hits = [None] * len(prefs), 0
    for n in sorted({len(p) for p in prefs}):  # ragged prefixes (short prompts) run as separate sub-batches; no pad masking in AR
        rows = [j for j, p in enumerate(prefs) if len(p) == n]; x = torch.tensor([prefs[j] for j in rows], device=dev); r = bad[rows], ok[rows]; out = []
        for _ in range(int(cfg.data.window)):
            with ac(): lg = net(x)[:, -1].float()
            t, h = choose(lg, g.mode, *r, pick); hits += h
            out.append(t); x = torch.cat([x, t[:, None]], 1)
        for j, o in zip(rows, torch.stack(out, 1)): res[j] = o
    return torch.stack(res), hits

@torch.no_grad()
def generate_long(net, codec, cfg, prompts, ns, g, dev=None):
    """prompts: list of token-id lists. Returns dict(blocks[j][n] token lists, table0, tables, hits)."""
    dev = dev or next(net.parameters()).device; net.eval()
    ac = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if (cfg.precision == "bf16" and dev.type == "cuda") else contextlib.nullcontext
    torch.manual_seed(g.seed)
    K, bos, pad, eos = int(cfg.data.get("prefix") or 0), cfg.data.eos_id, cfg.data.vocab - 1, cfg.data.eos_id
    assert g.table_slots <= K, "table_slots must fit in the prefix window"
    hist = [list(p) for p in prompts]; tabs = [update_table([], p, ns) for p in hist]; table0 = [list(t) for t in tabs]
    blocks = [[] for _ in hist]; hits = 0
    for _ in range(g.n_blocks):
        bad, ok = table_masks(tabs, ns)
        prefs = [build_prefix(h, t, ns, K, g.table_slots, bos, pad if cfg.arm == "ncm" else None) for h, t in zip(hist, tabs)]
        if cfg.arm == "ncm": ids, h = _ncm_block(net, codec, cfg, prefs, bad, ok, g, dev, ac)
        else: ids, h = _ar_block(net, cfg, prefs, bad, ok, g, dev, ac)
        hits += h
        for j, r in enumerate(ids.tolist()):
            r = [t for t in r if t != eos]; hist[j] += r; blocks[j].append(r); tabs[j] = update_table(tabs[j], r, ns)
    return dict(blocks=blocks, table0=table0, tables=tabs, hits=hits)

def load_model(ckpt=None, config=None, overrides=(), device="auto"):
    """ckpt (runs/<tag>/last.pt) -> (cfg, net, codec). Or config files (+overrides) for random-init weights."""
    if ckpt:
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = OmegaConf.merge(OmegaConf.create(st["cfg"]), OmegaConf.from_dotlist(list(overrides)))
    else:
        from ncm150.config import load_cfg
        cfg, st = load_cfg(*config, overrides=overrides), None
    dev = pick_device(device if device != "auto" else cfg.device)
    torch.manual_seed(cfg.seed)  # random-init codec is reproducible: same RNG order as train.Task (codec, then net)
    codec = build_codec(cfg).to(dev) if cfg.arm == "ncm" else None
    net = build_model(cfg, codec).to(dev)
    if st is not None: net.load_state_dict(st["model"])
    return cfg, net.eval(), codec
