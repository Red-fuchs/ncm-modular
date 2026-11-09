"""Extension points for later arms (MoE FFN, fixed per-level experts). Nothing here changes the AR/NCM baselines.
To add an arm: register an FFN builder; it receives the dense MLP's config namespace and must return an nn.Module
with the same (B,T,d)->(B,T,d) contract. Per-level experts: NCM target positions carry a level id (model.lvl_1L,
shape (1,T)); a level-aware FFN can read it via the `level_ids` attribute set by `bind_level_ids`.
flops.py asks `ffn_active_params(cfg)` for the per-position FFN matmul params, so active-FLOP matching stays honest.
"""
from typing import Callable
import torch.nn as nn

FFN_REGISTRY: dict[str, Callable] = {}

def register_ffn(name):
    def deco(fn): FFN_REGISTRY[name] = fn; return fn
    return deco

def ffn_active_params(mcfg) -> int:
    d, r = int(mcfg.n_embd), float(mcfg.mlp_ratio)
    if str(mcfg.ffn) == "dense": return int(2 * r * d * d)
    raise NotImplementedError(f"register active-param rule for ffn={mcfg.ffn}")

def apply_ffn_variant(model: nn.Module, mcfg, blocks):
    """Swap block.mlp for the registered variant. dense = no-op."""
    name = str(mcfg.ffn)
    if name == "dense": return model
    if name not in FFN_REGISTRY: raise NotImplementedError(f"ffn={name!r} not registered (see ext.py, moe-btm.md)")
    for b in blocks: b.mlp = FFN_REGISTRY[name](mcfg)
    return model

def bind_level_ids(model, ffn_modules):
    for m in ffn_modules: m.level_ids = model.lvl_1L
