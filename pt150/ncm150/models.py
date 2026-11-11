import math, os
import torch, torch.nn as nn, torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint
from ncm150.config import levels_of
from ncm150.ext import apply_ffn_variant
from ncm150.vendor.models.common import Block, LayerNorm, Transformer, precompute_rope_cos_sin
from ncm150.vendor.models.next_concept import NextConceptTransformer
from ncm150.vendor.models.vqvae import VQVAE
from ncm150.vendor.utils.misc import count_params

def _block_cfg(m):
    return OmegaConf.create(dict(n_embd=m.n_embd, n_head=m.n_head, n_layer=m.n_layer, bias=m.bias, dropout=m.dropout,
                                 use_qk_norm=m.use_qk_norm, mlp_ratio=m.mlp_ratio))

def enable_grad_ckpt(blocks):
    for b in blocks:
        f = b.forward
        b.forward = (lambda f: lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k))(f)

class ARTransformer(Transformer):
    """Plain causal LM from the same vendored Block (RoPE, QK-norm, pre-LN, GELU MLP). Tied embedding/head."""
    def __init__(self, mcfg, vocab, window):
        super().__init__()
        self.mcfg, self.vocab = mcfg, vocab
        bc = _block_cfg(mcfg)
        self.wte = nn.Embedding(vocab, mcfg.n_embd)
        self.drop = nn.Dropout(mcfg.dropout)
        self.blocks = nn.ModuleList([Block(bc) for _ in range(mcfg.n_layer)])
        self.ln_f = LayerNorm(mcfg.n_embd, bias=mcfg.bias)
        cos, sin = precompute_rope_cos_sin(torch.arange(window), mcfg.n_embd // mcfg.n_head, int(mcfg.rope_base))
        self.register_buffer("rope_cos", cos, persistent=False); self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"): nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * mcfg.n_layer))
        apply_ffn_variant(self, mcfg, self.blocks)

    def forward(self, ids):  # ids (B,T) -> logits (B,T,V)
        T = ids.shape[1]
        x = self.drop(self.wte(ids)); rope = (self.rope_cos[:T], self.rope_sin[:T])
        for b in self.blocks: x = b(x, rope_cos_sin=rope)
        return F.linear(self.ln_f(x), self.wte.weight)

    def nonembedding_params(self):
        return count_params(self, exclude_params=["wte"])[1]

def build_codec(cfg):
    c, L = cfg.codec, levels_of(cfg)
    assert L[-1] == cfg.data.window, "codec levels must end at the window length"
    vq = dict(clss="multiscale_residual_vector_quantizer", decay=0.99, epsilon=1e-5, commitment_cost=0.25,
              learned_l1_sampling=True, learned_l1_cascade=True, learned_l1_cascade_base=4, learned_all_sampling=False,
              quant_resi=dict(enabled=True, ratio=0.5, share_mode=0, learnable_ratio=False),
              aux=dict(fine_drop=dict(prob=0.5, min_keep=1)),
              levels=dict(use_manual_levels=True, manual_levels=L),
              learned_l1_cascade_norm=False, learned_l1_cascade_norm_final=False, ema_scale_by_world_size=False)
    vq_cfg = OmegaConf.merge(OmegaConf.create(vq), c.get("vq") or {})
    m = VQVAE(in_vocab_size=cfg.data.vocab, context_length=cfg.data.window, embed_dim=c.embed_dim,
              vqvae_vocab_size=[c.vocab] * len(L), n_layers=c.n_layers, vector_quantizer_config=vq_cfg,
              compression_factor=1, dropout=c.dropout, pre_quant_groupnorm=c.pre_quant_groupnorm,
              pre_quant_dropout=float(c.get("pre_quant_dropout") or 0.0), use_qk_norm=True)
    if not c.random_init:
        if not os.path.exists(c.checkpoint):
            raise FileNotFoundError(f"codec {c.get('name')}: {c.checkpoint} missing, run ./fetch_codec.sh {c.get('name') or ''}".strip())
        sd = torch.load(c.checkpoint, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict(sd, strict=True)
    m.eval().requires_grad_(False)
    return m

def build_ncm(cfg, codec):
    m, L = cfg.model, levels_of(cfg)
    bc = _block_cfg(m)
    mc = OmegaConf.merge(bc, OmegaConf.create(dict(
        context_length=cfg.data.window + 1, prefix_len=1, raw_token_prefix=True, prefix_pad_token_id=cfg.data.vocab - 1,
        vocab_size=[cfg.codec.vocab] * (len(L) - 1), predict_level0=False, attn_pattern="block_diagonal",
        use_positional_encoding=True, use_level_encoding=True, use_rope=True, rope_base=int(m.rope_base),
        ctx_encoder=dict(enabled=False), use_flex_attention=bool(cfg.ncm.flex_attention), query_level0=False,
        post_upsample_conv=dict(enabled=bool(cfg.ncm.post_upsample_conv), kernel_size=3),
        tie_heads_to_codebook=True, tie_heads_learn_scale=True, tie_heads_bias=True)))
    net = NextConceptTransformer(vqvae_local=codec, config=mc)
    # SDPA fast path: bool mask (True = attend) instead of an fp32 -inf/0 mask (identical semantics, 4x less memory)
    net.attn_mask_for_masking = (net.attn_mask_for_masking == 0)
    apply_ffn_variant(net, m, net.transformer.blocks)
    return net

def ncm_nonembedding_params(net):
    return count_params(net, exclude_params=["wte", "wle", "wpe"])[1]

def build_model(cfg, codec=None):
    if cfg.arm == "ar":
        return ARTransformer(cfg.model, cfg.data.vocab, cfg.data.window)
    return build_ncm(cfg, codec)

def blocks_of(net):
    return net.transformer.blocks if hasattr(net, "transformer") else net.blocks
