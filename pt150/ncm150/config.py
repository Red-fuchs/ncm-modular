import os
from omegaconf import OmegaConf

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

def load_cfg(*files, overrides=()):
    """Merge yaml files in order (names resolved against configs/), then dotlist overrides."""
    parts = [OmegaConf.load(f if os.path.exists(f) else os.path.join(CFG_DIR, f)) for f in ("base.yaml", *files)]
    parts.append(OmegaConf.from_dotlist(list(overrides)))
    return OmegaConf.merge(*parts)

def levels_of(cfg):
    c = cfg.codec
    return [int(l) for l in c.levels] if c.get("levels") else [i * i for i in range(1, int(c.n_levels) + 1)]
