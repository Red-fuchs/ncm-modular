"""Baselines (MLX): AR prefix-LM decoder (same 8L/512d/8H as NCM gen, KV cache) + intent classifier encoder."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mlx.core as mx
from mlx import nn
from ncm_mlx.official.model import rope_tables, _rotate, LN_EPS

class CAttn(nn.Module):
    def __init__(self, D, H):
        super().__init__(); self.H, self.hd = H, D // H
        self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.q_norm = nn.RMSNorm(self.hd, eps=LN_EPS); self.k_norm = nn.RMSNorm(self.hd, eps=LN_EPS)
    def __call__(self, x, mask, rope, cache=None):
        B, T, C = x.shape
        q, k, v = (t.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3) for t in mx.split(self.qkv(x), 3, -1))
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = _rotate(q, *rope), _rotate(k, *rope)
        if cache is not None:
            if cache[0] is not None: k = mx.concatenate([cache[0], k], 2); v = mx.concatenate([cache[1], v], 2)
            cache[0], cache[1] = k, v
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask=mask)
        return self.proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))

class CBlock(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, eps=LN_EPS); self.attn = CAttn(D, H); self.ln2 = nn.LayerNorm(D, eps=LN_EPS)
        self.fc = nn.Linear(D, 4 * D); self.out = nn.Linear(4 * D, D)
    def __call__(self, x, mask, rope, cache=None):
        x = x + self.attn(self.ln1(x), mask, rope, cache)
        return x + self.out(nn.gelu(self.fc(self.ln2(x))))

class ARGen(nn.Module):
    """prefix (frozen codec token emb -> in_proj, bidirectional, pad-masked) + target tokens (compact vocab, causal)."""
    def __init__(self, tok_table, V, P=48, T=48, D=512, H=8, nl=8, pad_id=151643, drop=0.1):
        super().__init__()
        self.P, self.T, self.V, self.pad_id, self.D = P, T, V, pad_id, D
        self._tok = tok_table
        self.in_proj = nn.Linear(tok_table.shape[1], D)
        self.emb = nn.Embedding(V, D)               # target tokens; 0 = BOS; head tied
        self.blocks = [CBlock(D, H) for _ in range(nl)]
        self.ln_f = nn.LayerNorm(D, eps=LN_EPS); self.drop = nn.Dropout(drop)
        self._rope = rope_tables(np.arange(P + T), D // H, 10000.0)
        rows = np.arange(P + T)[:, None]; cols = np.arange(P + T)[None, :]
        self._base = mx.array(np.where(rows < P, cols < P, (cols < P) | (cols <= rows)))[None, None]
    def logits(self, h): return h @ self.emb.weight.T
    def __call__(self, ptoks, tin):
        """tin: (B,T') input tokens (BOS + target[:-1]) -> logits (B,T',V)"""
        B, Tn = tin.shape
        keep = ptoks != self.pad_id
        h = mx.concatenate([self.drop(self.in_proj(self._tok[ptoks])), self.drop(self.emb(tin))], 1)
        S = self.P + Tn
        kk = mx.concatenate([keep, mx.ones((B, Tn), dtype=mx.bool_)], 1)
        mask = self._base[:, :, :S, :S] & kk[:, None, None, :]
        rope = (self._rope[0][:S], self._rope[1][:S])
        for b in self.blocks: h = b(h, mask, rope)
        return self.logits(self.ln_f(h)[:, self.P:])
    def generate(self, ptoks, eos_c, maxlen=None):
        """greedy with KV cache; returns list of compact-id lists; NFE = steps run."""
        B = ptoks.shape[0]; maxlen = maxlen or self.T
        keep = ptoks != self.pad_id
        caches = [[None, None] for _ in self.blocks]
        # prefill: prefix (bidirectional) + BOS
        h = mx.concatenate([self.in_proj(self._tok[ptoks]), self.emb(mx.zeros((B, 1), dtype=mx.int32))], 1)
        S = self.P + 1
        kk = mx.concatenate([keep, mx.ones((B, 1), dtype=mx.bool_)], 1)
        mask = self._base[:, :, :S, :S] & kk[:, None, None, :]
        rope = (self._rope[0][:S], self._rope[1][:S])
        for b, c in zip(self.blocks, caches): h = b(h, mask, rope, c)
        out = []; done = mx.zeros((B,), dtype=mx.bool_); nfe = 1
        nxt = mx.argmax(self.logits(self.ln_f(h)[:, -1]), -1)
        for i in range(maxlen):
            out.append(nxt); done = done | (nxt == eos_c)
            if i == maxlen - 1 or bool(done.all().item()): break
            pos = self.P + 1 + i
            h = self.emb(nxt[:, None]); rope = (self._rope[0][pos:pos + 1], self._rope[1][pos:pos + 1])
            kk = mx.concatenate([keep, mx.ones((B, 1 + i + 1), dtype=mx.bool_)], 1)
            mask = kk[:, None, None, :]
            for b, c in zip(self.blocks, caches): h = b(h, mask, rope, c)
            nxt = mx.argmax(self.logits(self.ln_f(h)[:, -1]), -1); nfe += 1
        return mx.stack(out, 1), nfe

class IntentCls(nn.Module):
    def __init__(self, tok_table, n_cls, P=48, D=256, H=4, nl=4, pad_id=151643, drop=0.1):
        super().__init__()
        self.P, self.pad_id, self._tok = P, pad_id, tok_table
        self.in_proj = nn.Linear(tok_table.shape[1], D)
        self.pos = nn.Embedding(P, D)
        self.blocks = [CBlock(D, H) for _ in range(nl)]
        self.ln_f = nn.LayerNorm(D, eps=LN_EPS); self.drop = nn.Dropout(drop); self.head = nn.Linear(D, n_cls)
        self._rope = rope_tables(np.arange(P), D // H, 10000.0)
    def __call__(self, ptoks):
        keep = ptoks != self.pad_id
        h = self.drop(self.in_proj(self._tok[ptoks]) + self.pos.weight[None])
        mask = keep[:, None, None, :]
        for b in self.blocks: h = b(h, mask, self._rope)
        h = self.ln_f(h); m = keep[..., None].astype(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1))
