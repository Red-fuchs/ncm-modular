#!/bin/zsh
# Fetch the OFFICIAL HazyResearch checkpoints and convert the XSum / WMT14 codecs to MLX safetensors.
# Download only (no upload). Sizes: xsum tokenizer.pt ~300 MB, wmt14 ~646 MB (torch) -> MLX codec 231 MB / 549 MB (fp32).
# Cap bandwidth if your link is small, e.g. HF_HUB_ENABLE_HF_TRANSFER=0 and --max-workers 1.
set -e
export NCM_WORK=${NCM_WORK:-w}; export NCM_UPSTREAM=${NCM_UPSTREAM:-upstream}
[ -d $NCM_UPSTREAM ] || git clone https://github.com/HazyResearch/ncm $NCM_UPSTREAM      # MIT, needed for models.vqvae (torch reference only)
for t in xsum wmt14; do
  hf download hazyresearch/ncm-checkpoints --include "$t/config.yaml" "$t/tokenizer.pt" --local-dir $NCM_WORK/up_raw   # repo layout verified: <task>/{config.yaml,generator.pt,tokenizer.pt,README.md}; generator.pt not needed
  mkdir -p $NCM_WORK/up/$t; cp $NCM_WORK/up_raw/$t/{config.yaml,tokenizer.pt} $NCM_WORK/up/$t/ 2>/dev/null || echo "adjust paths: need config.yaml + tokenizer.pt in $NCM_WORK/up/$t"
  python src/ncm_modular/convert.py $t
done
# tokenizers (HF): gpt2 for xsum, Qwen/Qwen3-0.6B for wmt14
hf download openai-community/gpt2 --local-dir $NCM_WORK/tok/gpt2 --include "*.json" "*.txt"
hf download Qwen/Qwen3-0.6B --local-dir $NCM_WORK/tok/Qwen3-0.6B --include "*.json" "*.txt"
# reference dump + parity (torch reference vs MLX)
for t in xsum wmt14; do python src/ncm_modular/ref_dump.py $t && python src/ncm_modular/parity.py $t; done
