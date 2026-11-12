# pt150: compute-matched 150M comparison, AR vs NCM (PyTorch, single CUDA GPU)

Branch `torch-150m`. Upstream HazyResearch/ncm ships no training code, so this is our own loop around their model files
(vendored, see NOTICE). Arms: `ar` (causal LM from the same `Block`), `ncm` (upstream `NextConceptTransformer`, frozen upstream XSum codec
for targets; the OWT codec is not published upstream yet, see Codec). Extension points for MoE / per-level experts: `ncm150/ext.py` (`model.ffn`, `model.experts` in config).

## What is compared
- Data: FineWeb-Edu (sample-10BT), GPT-2 BPE, 48-token windows (= XSum codec context), uint16 shards. Sample `j` of the stream is a pure function of
  `(seed, j)`; both arms read the same windows in the same order (`train.global_bs` windows per step, identical by default).
- AR: input `[BOS] + w[:-1]`, loss on all 48 tokens (nats/token), same 48-token context as NCM. BOS = EOS id 50256.
- NCM: prefix = BOS; codec(window) -> 9 levels (4,6,10,14,20,26,32,40,48 codes, 4096-way); inputs = cumulative reconstruction pooled
  to the next level; loss = CE over 196 target codes (levels 2-9) (`ncm.level_weighting`: `uniform` default | `per_level_mean`). Logged `code_nats_per_text_token`
  = summed code CE / 48, NOT the same quantity as AR nats/token. Cross-arm quality needs generation eval (upstream `evaluate.py`, gen-ppl; not wired here).
- Size: non-embedding params by upstream `count_params` (excludes wte/wle/wpe): AR 12L d1024 = 151.2M; NCM 11L d1024 = 145.3M
  (8 heads + 8 post-upsample convs included). `ncm_150m_blocks12.yaml`: same 12 blocks as AR, 157.9M.
- **`--ctx1024` (LLM-shaped, the comparison that matters for scaling):** samples are 1024 tokens = 976 raw-token prefix + one
  48-token codec block. NCM: prefix = BOS + 976 tokens through the codec's token embedding (upstream `raw_token_prefix` path, so
  upstream `generate()` can use the trained model), predicts the block's 196 codes. AR: causal over all 1024, trains on all of
  them (its normal regime); `block_nats_per_token` = AR loss on the same 48 block positions. Per sample NCM supervises 48 text
  tokens, AR 1024: that data-efficiency gap is part of what is measured. Per 1024-token sample: AR 1.32 TFLOP, NCM 1.14 TFLOP
  (incl. codec encode; dense SDPA: 1.40 / 1.17). Default (no flag): 48-token samples, BOS-only prefix (plumbing smoke).
- Compute matching: `ncm150/flops.py`. 6ND-style: train = 3 x fwd, 2 FLOPs per MAC, matmul work + attention over *allowed* pairs;
  ncm also pays the online codec encode. Per 48-token window (computed, not measured): AR 58.5 GFLOP, NCM 168.8 GFLOP incl. codec encode (2.9x;
  plain masked SDPA counts dense attention: 173.2 GFLOP, only 2.6 % more at this length, so `ncm.flex_attention=false` is a safe fallback).
  `budget.mode=flops` sets steps per arm = floor(budget / flops_per_step); `mode=tokens` gives both arms the same steps. Logged each step: `flops`, `text_tokens`.

## Setup
Linux, NVIDIA driver + CUDA 12.x/13.x, Python >= 3.10, torch >= 2.5 (flex attention), single GPU. H100: bf16 autocast, ~80 GB. 4080 Super 16 GB: use smoke overlay.
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # then the CUDA build of torch for your driver
./fetch_codec.sh                           # ncm only: XSum codec -> artifacts/xsum/tokenizer.pt (~300 MB, sha256-checked, MIT, not redistributed)
python prep_data.py --out data/fineweb_edu_gpt2 --max-tokens 2e8     # smoke: 2e8 is plenty; main run: size to budget
python -m pytest -q                        # CPU, ~25 s, includes full-size param count (needs ~6 GB RAM); -m "not slow" skips that
```
## Codec
Default is the upstream **XSum** codec (`hazyresearch/ncm-checkpoints` `xsum/tokenizer.pt`, sha256 pinned in `configs/base.yaml`),
the only published codec with GPT-2 BPE. The OWT codec (1024 ctx) the harness was first written for is not published yet
(`owt/` holds only `TODO.md`). Codec shape is config-driven (`codec.levels`, `codec.vq`), loaded with `strict=True`.

## Run
```
./run.sh ncm --smoke       # plumbing/throughput smoke, 48-token samples (flex-attention path)
./run.sh ncm --ctx1024 --smoke   # LLM-shaped smoke: 976-token prefix + 48-token block, 16 x 1024 tok/step, micro_bs 2
./run.sh ar --ctx1024 --smoke    # same for AR; main runs: ./run.sh <arm> --ctx1024 [budget.flops=...]
./run.sh ar --smoke        # each smoke: ~1 h cap (train.max_minutes=70), flops budget 2e16, global_bs 336 windows (16k tok)
./run.sh ar                # main: edit budget.flops (default 1e18) or pass budget.flops=3e18 ...
./run.sh ncm budget.flops=3e18
./run.sh ncm --cpu-tiny data.dir=... # local plumbing check, random-init codec
```
Any `key=value` is an OmegaConf override (`train.micro_bs=2`, `ncm.flex_attention=false`, `wandb.enabled=true`).
Resume is automatic from `runs/<tag>/last.pt` (model, optimizer, RNG, step; data position = step * global_bs; exact, tested on CPU).
`train.max_minutes` stops cleanly with a checkpoint; rerun the same command to continue. Schedule length is fixed by the budget, not by the cap.

## Logs and sending them back
`runs/<tag>/log.jsonl`: `start` record (params, FLOPs, planned steps, full config), then per-`log_every` train records
(`loss, lr, grad_norm, text_tokens, flops, text_tok_s, tflops_s, mem_gb`) and `eval` records (`val_loss`, ncm: `val_code_nats_per_text_token`).
Send back: `tar czf logs_<arm>_<host>.tgz runs/*/log.jsonl && nvidia-smi -q > gpu.txt && pip freeze > pip.txt` (no data, no checkpoints, no paths beyond the config echoed in `start`; check it before sending).

## Throughput placeholders (fill from first real runs)
| GPU | arm | micro_bs | text tok/s | TFLOP/s achieved | VRAM |
|---|---|---|---|---|---|
| H100 80GB | ar | TBD | TBD | TBD | TBD |
| H100 80GB | ncm (flex) | TBD | TBD | TBD | TBD |
| RTX 4080S 16GB | ar | TBD | TBD | TBD | TBD |
| RTX 4080 16GB | ncm (compiled flex, no grad_ckpt) | 56 | ~10.9k | ~38 | 9.7 GB |

## Known gaps
- CUDA: community run on an RTX 4080 (sweep + logs: https://gist.github.com/granman3/15d49ea081464c8bf73321b496d99901): compiled flex_attention +
  `micro_bs=56`, no grad_ckpt -> ncm smoke 9 min instead of 54, val_loss unchanged (5.876 vs 5.875). flex_attention is now compiled on CUDA
  automatically. `train.compile=true` (whole-model `net.compile()`) is wired but still unmeasured. Fallback: plain SDPA with a bool mask (+2.6 % NCM FLOPs at 48 tokens).
- Codec not retrained on FineWeb-Edu: uses the upstream XSum codec (GPT-2 tokenizer, 48 ctx, built for XSum news summaries of <= 48 tokens). Exact round trip
  on FineWeb-Edu text is not measured here; a 2-sentence news check decodes 100 %.
- Default mode is 48-token context for both arms (short-context only); use `--ctx1024` for the long-prefix comparison.
  One codec block per sample (as upstream); multi-block samples (block-causal across blocks) would give NCM more targets per sample. When upstream publishes `owt/tokenizer.pt`:
  `./fetch_codec.sh owt`, then see the header of `configs/codec_owt.yaml` (1024 ctx, 32 levels, ncm `model.n_layer=10`).
- MPS (Apple Silicon): codec encode + FLOP setup tested on MPS (pytest); train loop and throughput not measured there.
- No generation-quality eval; no MoE / per-level experts yet (hooks only); no multi-GPU.
