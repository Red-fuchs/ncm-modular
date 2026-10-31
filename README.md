---
license: mit
library_name: mlx
tags:
  - mlx
  - next-concept-modeling
  - apple-silicon
  - text-generation
  - diffusion-lm
  - research-code
  - negative-results
base_model: hazyresearch/ncm-checkpoints
datasets:
  - AmazonScience/massive
  - roneneldan/TinyStories
language:
  - en
  - de
paper:
  - arxiv:2610.14162
---

# ncm-modular

> "Think of it as a Linux-style idea for local LLMs: a shared core (the codec), swappable modules, and anyone can contribute one." (the author)

Tools and honest experiment logs for **Next-Concept Modeling (NCM)** on Apple Silicon (MLX). No model weights are included; everything is built from the official checkpoints.

> Independent third-party work. NCM is by Kumbong et al. / HazyResearch (Stanford). Paper: [arXiv:2610.14162](https://arxiv.org/abs/2610.14162), code: [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT), checkpoints: [hazyresearch/ncm-checkpoints](https://huggingface.co/hazyresearch/ncm-checkpoints). The MLX package [janmus1c/ncm-mlx](https://github.com/janmus1c/ncm-mlx) (MIT, TinyStories and Sudoku) is used as a dependency. See `NOTICE` and `LICENSES.md`.

## What is in here

| Component | Files | What it does |
|---|---|---|
| MLX port of the XSum and WMT14 codecs | `src/ncm_modular/codec_mlx.py`, `convert.py`, `ref_torch.py`, `ref_dump.py`, `parity.py` | multiscale-residual VQ-VAE encode + decode in MLX (no torch at inference). Parity check against the official PyTorch codec. |
| Codec round-trip probes | `roundtrip_news.py`, `texts.py` | how many facts (names, numbers, dates) survive encode/decode |
| Surrogate masking library | `masking.py`, `mask_stress.py` | replace rare entities by common "surrogate" words (Tom, Lily, three, Monday ...) before the codec, per-text table, restore afterwards |
| Prefix-conditioned NCM generator (MLX training) | `gen_mlx.py`, `train_gen.py`, `ar_mlx.py`, `train_ar.py`, `eval_all.py` | train a small generator whose input is a text prefix and whose output are codec codes (coarse to fine, `NFE = 1 + K-1`), plus same-size AR and classifier baselines |
| MASSIVE-de pilot | `massive_build.py`, `massive_eval.py`, `results/massive/` | German intent/slot task, NCM vs same-size AR, trained from scratch (11.5k examples) |
| Block-seam experiment | `src/ncm_modular/seam/`, `scripts/seam_pipeline.sh`, `results/seam/` | generate long text block by block (sliding window) with a surrogate **entity table** and **coarse-code memory** of earlier blocks in the prefix |

Not included (see `LICENSES.md`): the JANG to GGUF lossless converter and the Gemma speed benchmarks. Their licensing is unresolved; they are listed as a separate candidate package.

## Results (short; full tables and caveats in `RESULTS.md`)

All numbers are single runs on one M1 Max (64 GB), small models, no confidence intervals. Several of them are negative.

**Codec port.** Matches the PyTorch codec on the test inputs (CPU, fp32 and fp64; logs in `results/codec/parity_*.txt`): XSum codes 100 % identical; WMT14 codes 100 % identical except one code at level 9 (99.89 % there); decoded ids 100 % identical at every level for both. MLX codec sizes: XSum 231 MB, WMT14 549 MB (fp32).

**Codec is lossy for rare strings** (round trip, raw rare names/places/numbers): TinyStories codec 62 % of entities kept, XSum 89 %, WMT14 96 %. With surrogate masking and a per-text table, the stress test (XSum and WMT14 codecs, gold entity spans) restores 100 % of facts with 0 swaps (TinyStories codec: 98.9 % recall, 76 % of texts exact, target missed). The limit moves to entity detection: the regex detector alone reaches 86 to 91 % span recall. Structured output works as sentences ("The intent is ... The date is John."), not as pipes or JSON (99.9 % vs 8 % round trip).

**MASSIVE-de pilot: NCM lost** against an AR decoder of the same size (test, 2974 utterances):

| model | params | intent acc | slot F1 | exact match | b=1 latency |
|---|---|---|---|---|---|
| NCM (WMT14 codec, 8L/512d, NFE 11) | 27.6 M | 68.2 % | 47.6 % | 51.0 % | 304 ms* |
| AR (same blocks, KV cache) | 25.5 M | 73.5 % | 59.1 % | 57.6 % | 102 ms* |
| classifier (intent only) | 3.3 M | 74.7 % | - | - | 3.5 ms* |

*Measured before 2026-10-31 10:45, probably throttled (see below). The bottleneck looks like data volume (the generator must learn a codec latent space from 11.5k examples), not the codec (target round trip 99.97 %).

**Block-seam experiment** (TinyStories, official TinyStories codec (frozen) + our 8L/512d generator, three prefix variants a/b/c, ~66 min training each, strongly undertrained): the entity table and the coarse-code memory help with names, but the seam smoothing is **not shown**.

| variant | prefix | invented names (T=0.7) | entity consistency (T=0.7) | seam cond. PPL (T=0) |
|---|---|---|---|---|
| a | text tail only | 8.1 % | 91.6 % | 39.0 |
| b | + entity table | 3.2 % | 96.6 % | 40.0 |
| c | + table + coarse codes of 6 earlier blocks | 2.2 % | 97.7 % | 43.2 |

Character carry-over from block 0 into later blocks: 41.8 % (a) to 60.2 % (c); gold is 82.4 %. At T=0 all models degenerate into repetition (cross-block 4-gram repeat 32 to 37 %), seam perplexities at T=0 are therefore not comparable with T=0.7 and the gold (repetitive text is easy to predict, so T=0 PPL looks better than gold). Read `RESULTS.md` before quoting any of this.

**Speed on M1 Max.** Generation latency for one 128-token block, seam generator (8L/512d, 12 codec levels, fp16, NFE 12): 49 to 69 ms at batch 1, i.e. roughly 1900 to 2600 tok/s, measured on 2026-10-31 around 15:37 (after the Low Power Mode problem was fixed). Batching hardly helps (about 1.1 to 1.9 s per batch of 50 blocks; the GPU is already busy). Earlier pilot numbers (TinyStories 89M: 128 tokens in 60 to 110 ms; MASSIVE latencies above) were taken **before 2026-10-31 10:45 with macOS Low Power Mode active on AC power, which cost about 46 % throughput**; treat them as lower bounds on speed. Training throughput of the MASSIVE pilot was affected as well.

## Limitations

- Tiny models, tiny data, single seeds, no confidence intervals. The "negative" results are about this setup (from scratch, 11.5k examples, short training), not about NCM in general.
- Surrogate masking needs an entity detector. The regex detector (86 to 91 % span recall) is the weak link; perfect gold spans were used for the stress tests.
- Masked outputs are restored by table lookup, so correctness of facts depends on the table, not on the model.
- The seam experiment does not prove smoothing; generated text at T=0 is degenerate; the LM judge (a 1.2B model) scores windows only.
- English-centric codecs; the WMT14 codec (Qwen tokenizer) can encode German but was not trained for it.
- Apple Silicon only (MLX 0.32.3). Not tested on other backends.

## Roadmap

1. **Stage 0 (compute-matched pilot, 150M):** AR vs NCM vs NCM-MoE vs NCM with level experts, same compute, FineWeb-Edu. The question the small pilots cannot answer.
2. Shared multilingual codec (EN + DE).
3. **Modular 1B NCMs** branched from one base NCM (BTM/BTX style), sharing one codec as lingua franca so that modules can be swapped per block or split by level (plan coarse, language fine).
4. Target use: fast local text renderer for prose (stories, game text, descriptions). Facts, code, tools and chat stay autoregressive.

## Reproduce

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[torch]"                      # mlx==0.32.3, numpy, safetensors, transformers, pyarrow, ncm-mlx==0.2.0 (+ torch for conversion)
export NCM_WORK=$PWD/w                        # all heavy files live here, never in the repo
scripts/fetch_and_convert.sh                  # downloads official checkpoints (~0.95 GB torch) + tokenizers, converts, runs parity (CPU)

python src/ncm_modular/roundtrip_news.py xsum 40       # codec fact round trip
python src/ncm_modular/mask_stress.py wmt14 150        # surrogate masking stress test (writes $NCM_WORK/mask_wmt14.txt)

# MASSIVE-de pilot: put de-DE train/validation/test .parquet into $NCM_WORK/massive_raw/
python src/ncm_modular/massive_build.py C
python src/ncm_modular/train_gen.py --steps 8000 --minutes 330 --out $NCM_WORK/ncm_pilot    # GPU, hours
python src/ncm_modular/train_ar.py  --mode ar  --steps 6000 --out $NCM_WORK/ar_pilot
python src/ncm_modular/train_ar.py  --mode cls --steps 3000 --out $NCM_WORK/cls_pilot
python src/ncm_modular/eval_all.py

# Seam experiment (needs the official TinyStories codec converted to $NCM_WORK/ts via `python -m ncm_mlx official ...`
# and TinyStories validation text as $NCM_SEAM_DATA/valid.txt)
scripts/seam_pipeline.sh                      # GPU, ~3.5 h on an M1 Max
```

Env vars: `NCM_WORK` (default `./w`), `NCM_UPSTREAM` (HazyResearch/ncm checkout, only for conversion/reference), `NCM_SEAM_DATA`, `NCM_SEAM_RUNS`, `NCM_JUDGE_MODEL`. Commands mirror the reported runs (MASSIVE NCM pilot: `--P 40 --bs 64 --lr 4e-4`, best checkpoint at step 3500; AR best at step 4000; seam: 66 min cap, ~2340 steps, bs 32).

## Related work & credit

This builds on other people's work, and we are grateful for it. NCM itself (method, architecture, checkpoints, reference code) is by Kumbong et al. at HazyResearch / Stanford ([HazyResearch/ncm](https://github.com/HazyResearch/ncm)). The MLX implementation of the TinyStories codec and its building blocks, which we depend on, is [janmus1c/ncm-mlx](https://github.com/janmus1c/ncm-mlx). If you are one of these authors and would like something changed or credited differently, please open an issue; we are happy to adjust.

## How this was made

This is a human-AI collaboration ("meta-cyborg"). The human author (GitHub / Hugging Face: Red-fuchs) set the direction, the ideas (surrogate masking as cross-block context, sliding window over blocks, modular NCMs) and the decisions. Claude (Anthropic) agents wrote most of the code, ran the experiments and wrote the docs. Results were checked against stored logs, but treat the code as research-grade.

## Contributing / collaboration

Repos: [github.com/Red-fuchs/ncm-modular](https://github.com/Red-fuchs/ncm-modular) and [huggingface.co/Red-fuchs/ncm-modular](https://huggingface.co/Red-fuchs/ncm-modular).

We can fund the small stage-0 comparison (150M, about 100 EUR of compute) ourselves. We lack the means to train the 1B base NCM and the modules. So we publish everything openly, in the hope that the right people (compute partners, researchers, module contributors) find it. If that is you, please open an issue or a discussion.

Ways to help, following the [roadmap](#roadmap): run or review stage 0 (AR vs NCM vs NCM-MoE vs level experts at equal compute); work on a shared multilingual (EN + DE) codec; train or donate compute for the 1B base and modules branched from it; contribute a module for a domain or language; reproduce or break the results above. Negative results and corrections are welcome.

## Citation

Please cite the NCM paper (arXiv:2610.14162) and, if you use the MLX stack, janmus1c/ncm-mlx. This repo has no paper. It is at https://github.com/Red-fuchs/ncm-modular.
