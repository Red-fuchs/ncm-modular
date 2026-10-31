"""Upstream PyTorch codec (CPU) loader. Needs torch, omegaconf, pyyaml and NCM_UPSTREAM=<HazyResearch/ncm checkout>."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import UPSTREAM
sys.path.insert(0, UPSTREAM)   # checkout of github.com/HazyResearch/ncm (not vendored); provides models.vqvae
import torch, yaml
from omegaconf import OmegaConf


def load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("state_dict", "model", "vqvae", "tokenizer"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict) and not torch.is_tensor(sd[k]): sd = sd[k]
    return sd


def load_ref(d):
    from models.vqvae import VQVAE
    cfg = OmegaConf.load(f"{d}/config.yaml")
    kw = OmegaConf.to_container(cfg.tokenizer.vqvae, resolve=True)
    kw.pop("checkpoint_path", None)
    kw["vector_quantizer_config"] = cfg.tokenizer.vqvae.vector_quantizer_config
    m = VQVAE(**kw)
    sd = load_state(f"{d}/tokenizer.pt")
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    print("load:", m.load_state_dict(sd, strict=False))
    return m.eval(), cfg
