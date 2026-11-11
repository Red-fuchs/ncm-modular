"""Explicit FLOP accounting. Convention: 1 multiply-add = 2 FLOPs; train = 3 x forward (fwd + 2x bwd, no activation
recompute counted); only matmul-like work is counted (linears, conv, tied-codebook head matmul, attention QK^T/AV).
Norms/softmax/embedding lookups are ignored (<1%). Attention is counted over *allowed* (unmasked) pairs ("algorithmic"
FLOPs, what flex/block-sparse attention executes); dense_attn=True counts the full SxS product (plain SDPA with a mask).
Compute matching uses the algorithmic number. Codec encode (frozen, fwd only, run online) is charged to the ncm arm."""
import torch
from torch.utils.flop_counter import FlopCounterMode
from ncm150.ext import ffn_active_params

def ar_pairs(T): return T * (T + 1) // 2
def ncm_pairs(P, levels):  # prefix rows see prefix; level-k rows see prefix + own level (block_diagonal)
    return P * P + sum(l * (P + l) for l in levels[1:])

def ar_fwd_flops(m, vocab, T, dense_attn=False):
    d = int(m.n_embd); mm = m.n_layer * (4 * d * d + ffn_active_params(m)) + d * vocab
    pairs = T * T if dense_attn else ar_pairs(T)
    return 2 * mm * T + 4 * m.n_layer * d * pairs

def ncm_fwd_flops(m, levels, cd, cv, P=1, conv=True, dense_attn=False):
    d = int(m.n_embd); T = sum(levels[1:]); S = P + T
    blocks = 2 * m.n_layer * (4 * d * d + ffn_active_params(m)) * S
    wte = 2 * d * cd * S
    convs = 2 * 3 * cd * cd * T if conv else 0
    heads = 2 * (d * cd + cd * cv) * T
    pairs = S * S if dense_attn else ncm_pairs(P, levels)
    return blocks + wte + convs + heads + 4 * m.n_layer * d * pairs

def measure_fwd(fn):
    with FlopCounterMode(display=False) as fc, torch.no_grad(): fn()
    return fc.get_total_flops()

def train_flops_per_window(cfg, codec=None, dense_attn=False):
    """FLOPs to train on one sample (data.prefix raw tokens + one data.window codec block), incl. codec encode (block only) for ncm."""
    from ncm150.config import levels_of, sample_len
    T, P = int(cfg.data.window), 1 + int(cfg.data.get("prefix") or 0)
    if cfg.arm == "ar":
        return 3 * ar_fwd_flops(cfg.model, cfg.data.vocab, sample_len(cfg), dense_attn)
    f = 3 * ncm_fwd_flops(cfg.model, levels_of(cfg), cfg.codec.embed_dim, cfg.codec.vocab, P, cfg.ncm.post_upsample_conv, dense_attn)
    if codec is not None:
        x = torch.zeros(1, T, dtype=torch.long, device=next(codec.parameters()).device)
        f += measure_fwd(lambda: codec.encode(x))
    return f

def plan_steps(cfg, flops_per_window):
    """-> (steps, flops_per_step, tokens_per_step). mode=flops: steps = floor(budget / flops_per_step)."""
    from ncm150.config import sample_len
    bs, T = int(cfg.train.global_bs), sample_len(cfg)
    fps = flops_per_window * bs
    steps = int(cfg.budget.flops // fps) if cfg.budget.mode == "flops" else int(cfg.budget.tokens // (bs * T))
    return max(steps, 1), fps, bs * T
