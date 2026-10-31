"""Work-dir layout. Everything heavy (checkpoints, caches, runs) lives under NCM_WORK (default ./w), never in the repo.
  $NCM_WORK/up/<task>/{config.yaml,tokenizer.pt}   official torch checkpoints (xsum, wmt14)
  $NCM_WORK/mlx/<task>/{codec.safetensors,spec.json}  converted MLX codecs
  $NCM_WORK/tok/{gpt2,Qwen3-0.6B}                    HF tokenizers
  $NCM_WORK/ts/                                      official TinyStories checkpoint converted by `python -m ncm_mlx official convert`
  $NCM_WORK/seam_data, $NCM_WORK/seam_runs           seam experiment data / outputs
  NCM_UPSTREAM = checkout of github.com/HazyResearch/ncm (only for convert.py / ref_dump.py / parity refs)"""
import os
WORK = W = os.environ.get("NCM_WORK", "w")
UPSTREAM = os.environ.get("NCM_UPSTREAM", "upstream")
DATA = os.environ.get("NCM_SEAM_DATA", f"{W}/seam_data")
RUNS = os.environ.get("NCM_SEAM_RUNS", f"{W}/seam_runs")
JUDGE_MODEL = os.environ.get("NCM_JUDGE_MODEL", "LiquidAI/LFM2-1.2B")   # any mlx_lm-loadable LM (path or HF id)
