"""Prefix-conditioned NCM generator (MLX) over the frozen wmt14 codec. Matches upstream wmt14 layout:
prefix attends only to itself (pad-masked); level k>=1 positions see prefix + own level (block_diagonal);
RoPE restarts per level; learned level emb; codebook-tied heads; level 0 predicted from last non-pad prefix position."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
import numpy as np, mlx.core as mx
from mlx import nn
import codec_mlx
from ncm_mlx.official.model import Block, rope_tables

class FrozenCodec:
    """thin wrapper: stop-gradient teacher inputs / sampling inputs from codes."""
    def __init__(self, d=WORK+"/mlx/wmt14"):
        self.c = codec_mlx.load(d)
        self.levels, self.K, self.L = self.c.levels, self.c.K, self.c.L
        self.cb = [self.c.W[f"cb.{k}"] for k in range(self.K)]
        self.tok = self.c.W["wte"]                      # (vocab, 384) frozen token embedding
        self.dim = self.tok.shape[1]
    def encode(self, ids): return self.c.encode(ids)
    def teacher_inputs(self, codes):
        """codes: list K of (B,l_k) -> inputs for levels 1..K-1: area_k @ cum recon through level k-1."""
        z, outs = None, []
        for k in range(self.K - 1):
            q = self.c.contrib(codes[k], k); z = q if z is None else z + q
            outs.append(self.c.area[k + 1] @ z)
        return outs
    def next_input(self, z, k): return self.c.area[k] @ z
    def contrib(self, c, k): return self.c.contrib(c, k)
    def decode(self, z): return mx.argmax(self.c.decode_logits(z), -1)

class Gen(nn.Module):
    def __init__(self, fc: FrozenCodec, P=48, D=512, H=8, nl=8, pad_id=151643, rope_base=10000.0, drop=0.1):
        super().__init__()
        self.P, self.D, self.pad_id, self.levels, self.K = P, D, pad_id, fc.levels, fc.K
        cd = fc.dim
        self.in_proj = nn.Linear(cd, D)
        self.level_emb = nn.Embedding(self.K - 1, D)
        self.blocks = [Block(D, H) for _ in range(nl)]
        self.ln_f = nn.LayerNorm(D, eps=1e-6)
        self.drop = nn.Dropout(drop)
        self.proj = [nn.Linear(D, cd, bias=False) for _ in range(self.K)]      # tied-codebook heads
        self.log_inv_t = [mx.array(0.4) for _ in range(self.K)]   # effective inverse temperature = exp(8*u), init ~24
        self.cdt = mx.float32
        hd = D // H
        self._cb = fc.cb; self._tok = fc.tok
        self._esq = [(E * E).sum(-1) for E in fc.cb]; self._scale = [mx.maximum(e.mean(), 1e-12) for e in self._esq]
        pos = np.concatenate([np.arange(P)] + [np.arange(l) for l in self.levels[1:]])
        self._rope = rope_tables(pos, hd, rope_base)
        self._lvl = mx.array(np.concatenate([np.full(l, k) for k, l in enumerate(self.levels[1:])]))
        grp = np.concatenate([np.full(P, -1)] + [np.full(l, k) for k, l in enumerate(self.levels[1:])])
        S = len(grp); kv_p = grp[None, :] == -1
        base = np.where(grp[:, None] == -1, kv_p, kv_p | (grp[:, None] == grp[None, :]))
        self._base = mx.array(base)[None, None]
        # single-level (inference) tables
        self._rope_lv = [rope_tables(np.concatenate([np.arange(P), np.arange(l)]), hd, rope_base) for l in self.levels[1:]]
        self._rope_p = rope_tables(np.arange(P), hd, rope_base)

    def set_compute(self, dt):
        self.set_dtype(dt); self.cdt = dt; return self
    def head(self, k, h):
        E = self._cb[k]; z = self.proj[k](h).astype(mx.float32)
        return (2.0 * (z @ E.T) - self._esq[k]) / self._scale[k] * mx.exp(8.0 * self.log_inv_t[k].astype(mx.float32))

    def prefix_emb(self, ptoks): return self.in_proj(self._tok[ptoks].astype(self.cdt))
    def rp(self, r): return (r[0].astype(self.cdt), r[1].astype(self.cdt))
    def keep(self, ptoks): return ptoks != self.pad_id

    def _stack(self, h, mask, rope):
        for b in self.blocks: h = b(h, mask, rope)
        return self.ln_f(h)

    def l0_logits_from(self, hp, ptoks):
        last = mx.maximum(self.keep(ptoks).sum(1) - 1, 0)
        h0 = mx.take_along_axis(hp, last[:, None, None], axis=1)            # (B,1,D)
        return self.head(0, h0)

    def __call__(self, inputs, ptoks):
        """teacher-forced: inputs = list of K-1 arrays (B,l_k,cd) for levels 1..K-1 -> list of K logits."""
        x = mx.concatenate(inputs, axis=1).astype(self.cdt)
        h = self.drop(self.in_proj(x) + self.level_emb(self._lvl))
        h = mx.concatenate([self.drop(self.prefix_emb(ptoks)), h], axis=1)
        keep = mx.concatenate([self.keep(ptoks), mx.ones((x.shape[0], x.shape[1]), dtype=mx.bool_)], 1)
        h = self._stack(h, self._base & keep[:, None, None, :], self.rp(self._rope))
        out = [self.l0_logits_from(h[:, :self.P], ptoks)]
        o = self.P
        for k in range(1, self.K):
            l = self.levels[k]; out.append(self.head(k, h[:, o:o + l])); o += l
        return out

    # ---- inference
    def encode_prefix(self, ptoks):
        pe = self.prefix_emb(ptoks)
        keep = self.keep(ptoks)
        hp = self._stack(pe, keep[:, None, None, :] & mx.ones((1, 1, self.P, 1), dtype=mx.bool_), self.rp(self._rope_p))
        return pe, keep, self.l0_logits_from(hp, ptoks)

    def level_logits(self, k, inp, pe, keep):
        h = self.in_proj(inp.astype(self.cdt)) + self.level_emb.weight[k - 1]
        h = mx.concatenate([pe, h], axis=1)
        n = inp.shape[1]
        kk = mx.concatenate([keep, mx.ones((keep.shape[0], n), dtype=mx.bool_)], 1)
        T = self.P + n
        rows = mx.arange(T)[:, None]; cols = mx.arange(T)[None, :]
        allowed = ~((rows < self.P) & (cols >= self.P))
        h = self._stack(h, allowed[None, None] & kk[:, None, None, :], self.rp(self._rope_lv[k - 1]))
        return self.head(k, h[:, self.P:])

def generate(gen: Gen, fc: FrozenCodec, ptoks):
    """greedy; NFE = 1 (prefix/level0) + K-1 level forwards."""
    pe, keep, l0 = gen.encode_prefix(ptoks)
    c = mx.argmax(l0, -1); z = fc.contrib(c, 0); codes = [c]
    for k in range(1, gen.K):
        c = mx.argmax(gen.level_logits(k, fc.next_input(z, k), pe, keep), -1)
        codes.append(c); z = z + fc.contrib(c, k)
    return fc.decode(z), codes

def count_params(m):
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(m.parameters()))
