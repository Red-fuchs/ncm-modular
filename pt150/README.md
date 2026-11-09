# pt150: compute-matched 150M comparison, AR vs NCM (PyTorch, single CUDA GPU)

Branch `torch-150m`. Upstream HazyResearch/ncm ships no training code, so this is our own loop around their model files
(vendored, see NOTICE). Arms: `ar` (causal LM from the same `Block`), `ncm` (upstream `NextConceptTransformer`, frozen OWT codec
for targets). Extension points for MoE / per-level experts: `ncm150/ext.py` (`model.ffn`, `model.experts` in config).

## What is compared
- Data: FineWeb-Edu (sample-10BT), GPT-2 BPE, 1024-token windows, uint16 shards. Sample `j` of the stream is a pure function of
  `(seed, j)`; both arms read the same windows in the same order (`train.global_bs` windows per step, identical by default).
- AR: input `[BOS] + w[:-1]`, loss on all 1024 tokens (nats/token). BOS = EOS id 50256.
- NCM: prefix = BOS; codec(window) -> 32 levels (1,4,...,1024 codes, 8192-way); inputs = cumulative reconstruction pooled to the next
  level; loss = CE over 11439 target codes (`ncm.level_weighting`: `uniform` default | `per_level_mean`). Logged `code_nats_per_text_token`
  = summed code CE / 1024, NOT the same quantity as AR nats/token. Cross-arm quality needs generation eval (upstream `evaluate.py`, gen-ppl; not wired here).
- Size: non-embedding params by upstream `count_params` (excludes wte/wle/wpe): AR 12L d1024 = 151.2M; NCM 10L d1024 = 152.1M
  (heads + post-upsample convs are ~26M of that). `ncm_150m_blocks12.yaml`: same 12 blocks as AR, 177M.
- Compute matching: `ncm150/flops.py`. 6ND-style: train = 3 x fwd, 2 FLOPs per MAC, matmul work + attention over *allowed* pairs;
  ncm also pays the online codec encode. Per 1024-token window (computed, not measured): AR 1.32 TFLOP, NCM 9.93 TFLOP (7.5x;
  with plain masked SDPA the NCM attention is dense and costs 25.1 TFLOP, 2.5x more, hence flex attention for the real run).
  `budget.mode=flops` sets steps per arm = floor(budget / flops_per_step); `mode=tokens` gives both arms the same steps. Logged each step: `flops`, `text_tokens`.

## Setup
Linux, NVIDIA driver + CUDA 12.x, Python >= 3.10, torch >= 2.5 (flex attention), single GPU. H100: bf16 autocast, ~80 GB. 4080 Super 16 GB: use smoke overlay.
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # then the CUDA build of torch for your driver
./fetch_codec.sh                           # ncm only: OWT codec -> artifacts/owt/tokenizer.pt (MIT, not redistributed)
python prep_data.py --out data/fineweb_edu_gpt2 --max-tokens 2e8     # smoke: 2e8 is plenty; main run: size to budget
python -m pytest -q                        # CPU, ~25 s, includes full-size param count (needs ~6 GB RAM); -m "not slow" skips that
```
## Run
```
./run.sh ar --smoke        # ~1 h cap (train.max_minutes=70), flops budget 2e16, global_bs 16
./run.sh ncm --smoke
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
| RTX 4080S 16GB | ncm (flex, grad_ckpt) | TBD | TBD | TBD | TBD |

## Known gaps
- Nothing here ran on CUDA. Flex attention path (`ncm.flex_attention=true`) and `train.compile` are untested; fallback is plain SDPA with a bool mask (2.5x NCM FLOPs, 16 GB cards will need `micro_bs=1` + `grad_ckpt`).
- Codec not retrained on FineWeb-Edu: uses the upstream OWT codec (GPT-2 tokenizer, 1024 ctx).
- No generation-quality eval; no MoE / per-level experts yet (hooks only); no multi-GPU.
