# Vendored from https://github.com/HazyResearch/ncm @ c70274273f489abc73832b2b98584a8483b6b203 (utils/registry.py)
# MIT License, Copyright (c) 2025 Hermann Kumbong. See LICENSE.upstream.
# Local change: absolute imports re-rooted to ncm150.vendor.* (nothing else, unless marked "LOCAL").
import math
from ncm150.vendor.utils.misc import round_levels

# LOCAL: tokenizer registry dropped (tokenization/ not vendored)
tokenizers = {}

vector_quantizers = {
    "multiscale_residual_vector_quantizer": "ncm150.vendor.models.quant.MultiscaleResidualVectorQuantizer",
}

level_schedulers = {
    "linear": lambda start, end, n: round_levels([start + (end - start) * (i-1)/(n-1) for i in range(1, n+1)], start, end),
    "cubic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**3/(n-1)**3 for i in range(1, n+1)], start, end),
    "exp": lambda start, end, n: round_levels([start + (end - start) * (math.exp(i-1))/(math.exp(n-1)) for i in range(1, n+1)], start, end),
    "sqrt": lambda start, end, n: round_levels([start + (end - start) * math.sqrt(i-1)/math.sqrt(n-1) for i in range(1, n+1)], start, end),
    "sine": lambda start, end, n: round_levels([start + (end - start) * math.sin((math.pi/2) * ((i-1)/(n-1))) for i in range(1, n+1)], start, end),
    "log": lambda start, end, n: round_levels([start + (end - start) * math.log(i)/math.log(n) for i in range(1, n+1)], start, end),
    "quadratic": lambda start, end, n: round_levels([start + (end - start) * (i-1)**2/(n-2)**2 for i in range(1, n+1)], start, end),
}
