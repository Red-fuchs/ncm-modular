# Results

Hardware: Apple M1 Max, 64 GB, MLX 0.32.3. Date of all runs: 2026-10-29 to 2026-10-31. One run per setting, no confidence intervals.

**Throttling caveat.** macOS Low Power Mode was active on AC power until 2026-10-31 10:45 and cost about 46 % throughput. Anything measured before that time is marked `[throttled]`. Quality metrics (accuracy, F1, perplexity) are not affected; latencies and tok/s are.

| Result | Run time (local time) | Throttled? |
|---|---|---|
| codec parity, round trips, masking stress (`results/codec`, `results/masking`) | re-run 10-31 evening on the final sources | n/a (no timing used) |
| MASSIVE pilot quality (`results/massive/results.json`) | 10-31 05:13 | quality no, latencies `[throttled]` |
| MASSIVE training throughput, TinyStories NCM 89M "128 tokens in 60 to 110 ms" | 10-31 00:00 to 05:00 | `[throttled]` |
| Seam training | 10-31 11:30 to 15:04 | no |
| Seam generation, metrics, judge, latencies | 10-31 15:37 to 15:45 | no |

## 1. Codec port (`codec_mlx.py`)

- Official XSum (GPT-2 tokenizer, L=48, 9 levels) and WMT14 (Qwen3-0.6B tokenizer, L=64) codecs ported. `parity.py` compares against a dump from the official PyTorch model: pre-quantisation latent, codes per level, cumulative z-hat, decoded ids per level, and decode from torch codes. Result (logs in `results/codec/parity_{xsum,wmt14}.txt`, CPU, fp32 and fp64, test texts plus random id rows, reference = PyTorch fp32 dump): XSum 100 % identical codes at all 9 levels; WMT14 100 % at 10 of 11 levels and 99.89 % at level 9 (one code in the batch differs, same in fp32 and fp64, cause not investigated), with the decoded ids still 100 % identical at every level and from torch codes. Max |diff| of z-hat is about 1e-5 except level 9/10 of WMT14 (6e-2, caused by that one code).
- Converted sizes (fp32 safetensors): XSum 231 MB, WMT14 549 MB. Torch originals: ~300 MB and ~646 MB. Both are fetched from upstream by the user; nothing is redistributed.

## 2. Codec fidelity and surrogate masking (`results/codec/rt_*.txt`, `results/masking/mask_*.txt`)

Entities kept after encode/decode of texts with rare names, places, numbers and dates:

| codec | entities kept (raw rare entities) |
|---|---|
| TinyStories | 62 % (earlier run; output not stored) |
| XSum | 89 % (in-distribution news summaries: 92 % exact, 97 % entities) |
| WMT14 | 96 % |

Typical failures are character-level damage inside rare names ("Anneliese Kouöder", "6.2%" for "7.2%"). Special placeholder tokens (Qwen special / spare slots) are *not* preserved by the codec (0 % identical placeholder sequences), so placeholders must be real common words.

Masking with word surrogates (per-text table, slot sets tuned on a calibration split: PER Tom/Lily/Sam..., LOC London/Paris..., NUM two/three..., DATE weekdays, AMT "ten dollars"...). Stress test: 150 texts with 5 to 8 entities and repeats, gold spans, exact-match of the restored text:

| codec | raw entities | surrogates (tuned) | span recall | binding errors | regex detector alone (span recall / precision) |
|---|---|---|---|---|---|
| XSum | 80.7 % exact | **100 %** exact | 100 % | 0 | 90.8 % / 93.9 % |
| WMT14 | 63.3 % exact | **100 %** exact | 100 % | 0 | 86.1 % / 92.4 % |
| TinyStories | 1.3 % exact | 76 % exact (36/150 fail) | 98.9 % (target 99.5 % missed) | 0 (23 texts with count errors, 10 spurious surrogate outputs) | 86.3 % / 91.6 % |

So masking is solved for XSum/WMT14 *given gold spans* and not for the TinyStories codec (cause not investigated). The weak link overall is detection (regex span recall 86 to 91 %, types PER/LOC conflated). A German control set is part of the stress test. Per-condition numbers (1/2/5 entities, repeats, mixed types, collisions) are in the `mask_*.txt` files.

Structured output as natural sentences instead of pipes/JSON: 99.9 % vs 8.1 % target round trip (MASSIVE labels, `results/massive/REPORT_stage1.md`).

## 3. MASSIVE-de pilot (`results/massive/`)

Data: MASSIVE de-DE (CC BY 4.0, Amazon Science), 11514 train / 2033 dev / 2974 test. Each slot value is replaced by a surrogate name; prefix = masked utterance; target = sentence format; WMT14 codec frozen. All models from scratch.

| | NCM | AR | intent classifier |
|---|---|---|---|
| params | 27.59 M | 25.49 M | 3.28 M |
| intent acc | 0.682 | 0.735 | 0.747 |
| slot F1 | 0.476 | 0.591 | - |
| exact match | 0.510 | 0.576 | - |
| fact restoration | 0.475 | 0.591 | - |
| invalid surrogate rate | 1.5 % | 0.07 % | - |
| forwards (NFE) | 11 | 12.9 mean (b=1) | 1 |
| latency b=1 p50 `[throttled]` | 304 ms | 102 ms | 3.5 ms |
| latency b=16 p50 `[throttled]` | 3687 ms | 503 ms | 29.8 ms |

Training: NCM best checkpoint at step 3500 of a run capped at 330 min (dev exact 0.597 on 300 dev examples), AR best at step 4000 (dev target exact 0.625 on 512). **Verdict: NCM lost at equal size, in quality and in latency.** Target round trip through the codec is 99.97 % (gate 99.5 %), so the codec is not the bottleneck; data amount and short training are the likely limits. Not a statement about NCM at scale.

## 4. Block-seam experiment (`results/seam/`)

Question: when long text is generated in blocks of 128 tokens, can a prefix with (b) a surrogate entity table and (c) a coarse-code memory of the last 6 blocks reduce invented names and smooth the join between blocks, versus (a) text tail only?

Setup: TinyStories (CDLA-Sharing-1.0), names replaced by 8 surrogates (John, Mary, ...) with a corpus-statistics name detector, sentence-packed blocks, official TinyStories codec frozen (12 levels), our 8L/512d generator. Test: 250 held-out stories with at least 5 blocks, gold block 0 given, 4 blocks generated. Training: 66 min cap per variant (~2340 steps, bs 32), loss still falling, so strongly undertrained. Judge: LFM2-1.2B perplexity on 16|16-token windows across a seam vs inside a block.

| | gold | a T=0.7 | b T=0.7 | c T=0.7 | a T=0 | b T=0 | c T=0 |
|---|---|---|---|---|---|---|---|
| spurious (invented) names | 0 | 8.1 % | 3.2 % | 2.2 % | 9.5 % | 6.8 % | 3.5 % |
| mention consistent | 98.2 % | 91.6 % | 96.6 % | 97.7 % | 89.5 % | 93.2 % | 96.5 % |
| stories with any spurious | 0 | 20.0 % | 10.0 % | 10.8 % | 4.8 % | 3.6 % | 2.0 % |
| block-0 entity persists | 82.4 % | 41.8 % | 47.5 % | 60.2 % | 11.6 % | 17.6 % | 26.4 % |
| cross-block 4-gram repeat | 1.2 % | 6.3 % | 4.1 % | 4.5 % | 34.9 % | 31.6 % | 36.7 % |
| seam cond. PPL (judge) | 165.8 | 225.7 | 227.5 | 241.5 | 39.0 | 40.0 | 43.2 |
| within-block cond. PPL | 154.7 | 153.1 | 185.6 | 215.5 | 12.6 | 13.1 | 14.5 |
| gold with shuffled next block | seam 291.6 | | | | | | |
| b=1 latency (1 block) | - | 58 ms | 69 ms | 63 ms | 52 ms | 49 ms | 67 ms |

Reading:
- **Entity gains are real in this setup**: table and code memory lower invented names and raise carry-over of block-0 characters (a to c), at roughly 1.6x prefix length cost (P=32/40/94).
- **Seam smoothing is inconclusive**: seam PPL minus within-block PPL is not clearly better for b/c than for a, and c is worse on within-block PPL at T=0.7. T=0 outputs collapse into repetition (4-gram repeat ~35 %), which makes the T=0 PPLs meaningless as quality measures. Models need much longer training and several seeds before a statement is possible.
- Sampled stories are in `samples.md` and look poor (repeated tokens), as expected for 66-minute models. Gold excerpts in that file are from TinyStories.

## 5. Side notes not backed by files in this package

(Gemma-4 26B-A4B JANG 4-bit on the same M1 Max; the converter and bench scripts are excluded for licensing reasons, see `LICENSES.md`.) vMLX 50 tok/s decode vs llama.cpp 41 tok/s on the same weights converted JANG to GGUF without loss; continuous batching on the MoE model gave vMLX 1.2x and llama.cpp 1.85x aggregate speed-up. These were measured after the Low Power Mode fix (benchmark JSONs from 11:57 and 15:23); an earlier partial run (10:56) is throttled and not used. Treat as anecdotal.

## 6. Release edits

For the public release, a few test-pool names in `texts.py` and `mask_stress.py` were replaced by fictional names. The CPU-only scripts (`roundtrip_news.py`, `mask_stress.py`, `parity.py`) were re-run afterwards with the final sources, so all stored outputs in `results/codec` and `results/masking` come from the code in this repo. Changes versus the first run: XSum raw round trip entities kept 90 % to 89 %; masking stress raw exact XSum 84.0 % to 80.7 %, TinyStories 2.0 % to 1.3 %, regex-detected-span exact TinyStories 38.0 % to 40.7 %; all other headline numbers unchanged. The TinyStories codec round trip (62 % entities kept) was not re-run and no output of it is stored.
