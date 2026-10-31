"""MLX port of the NCM multiscale-residual VQ-VAE codec (XSum/WMT14 variant). Encode + decode only."""
import json, math
import numpy as np
import mlx.core as mx

GN_EPS, LN_EPS = 1e-5, 1e-6


def area_matrix(src, dst, dt=np.float32):
    m = np.zeros((dst, src), dt)
    for i in range(dst):
        s, e = (i * src) // dst, math.ceil((i + 1) * src / dst)
        m[i, s:e] = 1.0 / (e - s)
    return m


def linear_matrix(src, dst, dt=np.float32):
    m = np.zeros((dst, src), dt)
    if src == 1:
        m[:, 0] = 1
        return m
    sc = src / dst
    for j in range(dst):
        x = max((j + .5) * sc - .5, 0.0)
        i0 = min(math.floor(x), src - 1); i1 = min(i0 + 1, src - 1); w = x - i0
        m[j, i0] += 1 - w; m[j, i1] += w
    return m


def plan_strides(total, base=4):
    S, out = int(total), []
    while S > 1:
        f = min(base, S)
        while f > 1 and S % f: f -= 1
        if f <= 1: out.append(S); S = 1
        else: out.append(f); S //= f
    return out or [1]


def phi_index(level, K):
    t = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
    return int(np.argmin(np.abs(t - level / max(K - 1, 1))))


def cascade_ops(total, base, inter_norm, final_norm):
    """Op list for level-0 down/up. ('d'|'u', stride, seq_idx) / ('n',). seq_idx = index inside upstream nn.Sequential."""
    st = plan_strides(total, base)
    down, i = [], 0
    for k, s in enumerate(st):
        if inter_norm and k > 0: down.append(("n",)); i += 1
        down.append(("d", s, i)); i += 1
    if final_norm: down.append(("n",))
    up, i = [], 0
    for k, s in enumerate(reversed(st)):
        if inter_norm and k > 0: up.append(("n",)); i += 1
        up.append(("u", s, i)); i += 1
    return down, up


def group_norm(x, G):
    """x (B,L,C): torch GroupNorm(G, C, affine=False) on (B,C,L) -> stats over (L, C/G) per group."""
    B, L, C = x.shape
    g = x.reshape(B, L, G, C // G)
    mu = g.mean(axis=(1, 3), keepdims=True)
    var = ((g - mu) ** 2).mean(axis=(1, 3), keepdims=True)
    return ((g - mu) * mx.rsqrt(var + GN_EPS)).reshape(B, L, C)


def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) * mx.rsqrt(v + LN_EPS) * w + b


def rms(x, w):
    return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + LN_EPS) * w


class Codec:
    def __init__(self, W: dict, cfg: dict, dtype=mx.float32):
        self.cfg, self.dt = cfg, dtype
        self.W = {k: v.astype(dtype) for k, v in W.items()}
        self.levels = cfg["levels"]; self.K = len(self.levels); self.L = self.levels[-1]
        self.G = cfg["groupnorm"]; self.H = cfg["n_head"]
        self.ratio = cfg["phi_ratio"]
        self.down_ops, self.up_ops = cascade_ops(self.L // self.levels[0], cfg["cascade_base"], cfg["cascade_norm"], cfg["cascade_norm_final"])
        nd = np.float64 if dtype == mx.float64 else np.float32
        self.area = [mx.array(area_matrix(self.L, n, nd), dtype=dtype) for n in self.levels]
        self.lin = [mx.array(linear_matrix(n, self.L, nd), dtype=dtype) for n in self.levels]
        self.n_extra = cfg.get("dec_layers", 1) - 1

    # ---- encoder
    def embed(self, tok):
        z = self.W["wte"][tok] + self.W["wpe"][: tok.shape[1]]
        return group_norm(z, self.G)

    def _down0(self, x):
        for op in self.down_ops:
            if op[0] == "n": x = group_norm(x, self.G); continue
            s, i = op[1], op[2]
            B, L, C = x.shape
            x = x.reshape(B, L // s, s * C) @ self.W[f"down.{i}.w"] + self.W[f"down.{i}.b"]
        return x

    def _up0(self, x):
        for op in self.up_ops:
            if op[0] == "n": x = group_norm(x, self.G); continue
            s, i = op[1], op[2]
            B, L, _ = x.shape
            x = (x @ self.W[f"up.{i}.w"]).reshape(B, L * s, -1) + self.W[f"up.{i}.b"]
        return x

    def contrib(self, codes, k):
        q = self.W[f"cb.{k}"][codes]
        if k == 0: q = self._up0(q)
        elif k != self.K - 1: q = self.lin[k] @ q
        j = phi_index(k, self.K)
        c = mx.conv1d(q, self.W[f"phi.{j}.w"], padding=1) + self.W[f"phi.{j}.b"]
        return q * (1 - self.ratio) + c * self.ratio

    def encode(self, tok):
        rest = self.embed(tok); codes = []
        for k in range(self.K):
            cb = self.W[f"cb.{k}"]
            if k == 0: x = self._down0(rest)
            elif k != self.K - 1: x = self.area[k] @ rest
            else: x = rest
            d = (x * x).sum(-1, keepdims=True) + (cb * cb).sum(-1) - 2 * (x @ cb.T)
            idx = mx.argmin(d, axis=-1)
            codes.append(idx)
            rest = rest - self.contrib(idx, k)
        return codes

    def cumulative(self, codes):
        out, z = [], None
        for k, c in enumerate(codes):
            q = self.contrib(c, k)
            z = q if z is None else z + q
            out.append(z)
        return out

    # ---- decoder
    def _block(self, x, p):
        W, H = self.W, self.H
        B, T, C = x.shape; hd = C // H
        h = layer_norm(x, W[f"{p}.ln1.w"], W[f"{p}.ln1.b"])
        qkv = h @ W[f"{p}.attn.w"].T + W[f"{p}.attn.b"]
        q, k, v = [t.reshape(B, T, H, hd).transpose(0, 2, 1, 3) for t in mx.split(qkv, 3, axis=-1)]
        q, k = rms(q, W[f"{p}.qn"]), rms(k, W[f"{p}.kn"])
        y = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * hd ** -0.5, axis=-1) @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C) @ W[f"{p}.proj.w"].T + W[f"{p}.proj.b"]
        x = x + y
        h = layer_norm(x, W[f"{p}.ln2.w"], W[f"{p}.ln2.b"])
        h = h @ W[f"{p}.fc.w"].T + W[f"{p}.fc.b"]
        h = 0.5 * h * (1 + mx.erf(h / math.sqrt(2)))
        return x + h @ W[f"{p}.fcp.w"].T + W[f"{p}.fcp.b"]

    def decode_logits(self, z):
        x = self._block(z, "blk0")
        for i in range(self.n_extra): x = self._block(x, f"blk{i+1}")
        x = layer_norm(x, self.W["lnf.w"], self.W["lnf.b"])
        return x @ self.W["head"].T

    def decode_codes(self, codes):
        return mx.argmax(self.decode_logits(self.cumulative(codes)[-1]), -1)


def load(d, dtype=mx.float32):
    cfg = json.load(open(f"{d}/spec.json"))
    return Codec(mx.load(f"{d}/codec.safetensors"), cfg, dtype)
