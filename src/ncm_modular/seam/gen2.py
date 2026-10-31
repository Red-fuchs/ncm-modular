"""Seam experiment: prefix-conditioned NCM generator over the frozen TinyStories codec (ncm_mlx official Codec, 12 levels, L=128).
Prefix = [codes(c) | entity table(b,c) | text tail(a,b,c)], one int array of ids: <8192 BPE token, 8192.. = level0/1 codebook codes."""
import os, sys, json, math
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from paths import *
import numpy as np, mlx.core as mx
from mlx import nn
from ncm_mlx.official import spec as specs
from ncm_mlx.official.bpe import BPETokenizer
from ncm_mlx.official.model import Codec, Block, rope_tables
H = os.path.dirname(os.path.abspath(__file__)); W = f"{WORK}/ts"
EOS, PAD, L, V = 0, 1, 128, 8192
KTXT, MB, TAB, CPB = 32, 6, 8, 9          # text tail tokens, max previous blocks as codes, table slots, codes/block (1 + 8)
SEG = {"a": (0, 0, KTXT), "b": (0, TAB, KTXT), "c": (MB * CPB, TAB, KTXT)}   # (code slots, table slots, text slots)
def plen(v): return sum(SEG[v])
bpe = BPETokenizer.load(f"{W}/bpe.json")
SURID = {s: bpe.encode(" " + s)[0] for s in ["John","Mary","David","Sarah","James","Peter","Emma","Mark"]}

def make_prefix(v, prev_ids, table, hist):
    """prev_ids: token ids of previous block (no EOS/pad); table: ordered surrogate names; hist: list of (9,) code arrays oldest..newest."""
    nc, nt, nx = SEG[v]; out = []
    if nc:
        c = []
        for h in hist[-MB:]: c += [V + int(h[0])] + [V + 4096 + int(x) for x in h[1:]]
        out += [PAD] * (nc - len(c)) + c
    if nt: t = [SURID[s] for s in table][:nt]; out += t + [PAD] * (nt - len(t))
    t = list(prev_ids)[-nx:]; out += t + [PAD] * (nx - len(t))
    return np.array(out, np.int32)

class FrozenCodec:
    def __init__(self):
        task = specs.from_json(json.load(open(f"{W}/spec.json"))["task"])
        c = Codec(task.codec); c.load_weights(f"{W}/codec.safetensors", strict=True); c.eval(); self.c = c
        self.levels, self.K, self.dim = c.levels, len(c.levels), task.codec.dim
        self.cb = list(c.codebooks); self.tok = c.tok.weight
        nrm = lambda E: E / mx.maximum(mx.sqrt((E * E).sum(-1, keepdims=True)), 1e-12) * 23.4   # cb0 has ~1e10 norms: rescale code rows to token-embedding scale
        self.emb = mx.concatenate([self.tok, nrm(self.cb[0]), nrm(self.cb[1])], 0)       # (8192+8192, 384)
    def encode(self, ids): return self.c.encode(ids)
    def teacher_inputs(self, codes):
        z, outs = None, []
        for k in range(self.K - 1):
            q = self.c.level_contribution(codes[k], k); z = q if z is None else z + q
            outs.append(self.c.pool(z, k + 1))
        return outs
    def next_input(self, z, k): return self.c.pool(z, k)
    def contrib(self, c, k): return self.c.level_contribution(c, k)
    def decode(self, z): return mx.argmax(self.c.decode_logits(z), -1)

class Gen(nn.Module):
    def __init__(self, fc, v, D=512, H=8, nl=8, drop=0.1):
        super().__init__()
        self.v, self.P, self.D, self.levels, self.K = v, plen(v), D, fc.levels, fc.K
        P, cd = self.P, fc.dim
        self.in_proj = nn.Linear(cd, D); self.level_emb = nn.Embedding(self.K - 1, D); self.seg_emb = nn.Embedding(3, D)
        self.blocks = [Block(D, H) for _ in range(nl)]
        self.ln_f = nn.LayerNorm(D, eps=1e-6); self.drop = nn.Dropout(drop)
        self.proj = [nn.Linear(D, cd, bias=False) for _ in range(self.K)]
        self.log_inv_t = [mx.array(0.4) for _ in range(self.K)]
        self.cdt = mx.float32
        hd = D // H
        self._cb = fc.cb; self._emb = fc.emb
        self._esq = [(E * E).sum(-1) for E in fc.cb]; self._scale = [mx.maximum(e.mean(), 1e-12) for e in self._esq]
        nc, nt, nx = SEG[v]
        self._seg = mx.array(np.array([0] * nc + [1] * nt + [2] * nx))
        pos = np.concatenate([np.arange(P)] + [np.arange(l) for l in self.levels[1:]])
        self._rope = rope_tables(pos, hd, 10000.0)
        self._lvl = mx.array(np.concatenate([np.full(l, k) for k, l in enumerate(self.levels[1:])]))
        grp = np.concatenate([np.full(P, -1)] + [np.full(l, k) for k, l in enumerate(self.levels[1:])])
        kv_p = grp[None, :] == -1
        self._base = mx.array(np.where(grp[:, None] == -1, kv_p, kv_p | (grp[:, None] == grp[None, :])))[None, None]
        self._rope_lv = [rope_tables(np.concatenate([np.arange(P), np.arange(l)]), hd, 10000.0) for l in self.levels[1:]]
        self._rope_p = rope_tables(np.arange(P), hd, 10000.0)
    def set_compute(self, dt): self.set_dtype(dt); self.cdt = dt; return self
    def head(self, k, h):
        E = self._cb[k]; z = self.proj[k](h).astype(mx.float32)
        return (2.0 * (z @ E.T) - self._esq[k]) / self._scale[k] * mx.exp(8.0 * self.log_inv_t[k].astype(mx.float32))
    def prefix_emb(self, p): return self.in_proj(self._emb[p].astype(self.cdt)) + self.seg_emb(self._seg)
    def rp(self, r): return (r[0].astype(self.cdt), r[1].astype(self.cdt))
    def keep(self, p): return p != PAD
    def _stack(self, h, mask, rope):
        for b in self.blocks: h = b(h, mask, rope)
        return self.ln_f(h)
    def l0_from(self, hp, p):
        # text = last segment, left-aligned: last non-pad position = query for level 0
        last = self.P - 1 - mx.argmax(self.keep(p)[:, ::-1], 1)
        return self.head(0, mx.take_along_axis(hp, last[:, None, None], axis=1))
    def __call__(self, inputs, p):
        x = mx.concatenate(inputs, axis=1).astype(self.cdt)
        h = self.drop(self.in_proj(x) + self.level_emb(self._lvl))
        h = mx.concatenate([self.drop(self.prefix_emb(p)), h], axis=1)
        keep = mx.concatenate([self.keep(p), mx.ones((x.shape[0], x.shape[1]), dtype=mx.bool_)], 1)
        h = self._stack(h, self._base & keep[:, None, None, :], self.rp(self._rope))
        out = [None]; o = self.P   # level 0 is not modelled (upstream: sampled from a unigram prior)
        for k in range(1, self.K):
            l = self.levels[k]; out.append(self.head(k, h[:, o:o + l])); o += l
        return out
    def encode_prefix(self, p):
        pe = self.prefix_emb(p); keep = self.keep(p)
        return pe, keep, None
    def level_logits(self, k, inp, pe, keep):
        h = self.in_proj(inp.astype(self.cdt)) + self.level_emb.weight[k - 1]
        h = mx.concatenate([pe, h], axis=1); n = inp.shape[1]
        kk = mx.concatenate([keep, mx.ones((keep.shape[0], n), dtype=mx.bool_)], 1)
        T = self.P + n; rows = mx.arange(T)[:, None]; cols = mx.arange(T)[None, :]
        allowed = ~((rows < self.P) & (cols >= self.P))
        h = self._stack(h, allowed[None, None] & kk[:, None, None, :], self.rp(self._rope_lv[k - 1]))
        return self.head(k, h[:, self.P:])

def pick(lg, temp):
    return mx.argmax(lg, -1) if temp <= 0 else mx.random.categorical(lg / temp)
UNI = mx.load(f"{W}/inference.safetensors")["level0_unigram"]
def generate(gen, fc, p, temp=0.0, c0=None):
    """c0: (B,1) level-0 codes (gold for teacher-level0 evals); None -> sampled from the codec's level-0 unigram."""
    pe, keep, _ = gen.encode_prefix(p)
    c = c0 if c0 is not None else mx.random.categorical(mx.log(UNI), shape=(p.shape[0], 1))
    z = fc.contrib(c, 0); codes = [c]
    for k in range(1, gen.K):
        c = pick(gen.level_logits(k, fc.next_input(z, k), pe, keep), temp); codes.append(c); z = z + fc.contrib(c, k)
    return fc.decode(z), codes
def count_params(m):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(m.parameters()))
