# NCM block-seam v2 report (2026-11-01)

## Setup
- Data: +1 TinyStories-train parquet shard (248 MB, curl 450K). train2 = old 18.7k valid-derived stories + 60k new (258k pairs; 100 exact duplicates of valid removed, 2.7k unmaskable skipped). Same names.json, masking and block cutting. Dev and test unchanged (test = same 250 stories).
- Training: a, b and c resumed from runs/{a,b,c}/best, identical 4500 steps, bs 32, seed 0 (about 0.55 epoch of train2). a 108 min, b 108 min, c 149 min. Best-dev checkpointing kept (best = step 4500 for all, so still improving).
- LR: fresh AdamW state (not saved before), 100-step warmup to 2e-4 (half of v1 peak), cosine to 5%. Not a true continuation, a warm restart.
- Eval: 250 test stories, block 0 gold, 4 blocks generated. T=0.7 (primary) and greedy. LM judge LFM2-1.2B, same windows. CIs: 95% bootstrap over stories (2000 resamples). "prev" = old gen files re-scored with the new script.
- Seam ratio = conditional PPL of seam windows / within-block windows (gold is about 1.07). Seam0 / random-next = seam-0 cond PPL divided by the control (gold block 0 followed by a block from another story; control PPL 305 [284, 326]; gold seam0 135 [119, 153]).

## Bug found and fixed
The old train_seam.py computed dev loss with TRAIN codes at dev row indices (codes[tdv] instead of cd_[tdv]). The old "dev loss" 1.38 and the best-checkpoint choice were against mismatched targets. Fixed. Correct dev loss (teacher-forced, true dev):
| variant | v1 checkpoint | v2 (4500 more steps) |
|---|---|---|
| a | 1.266 | 1.002 |
| b | 1.261 | 1.005 |
| c | 1.294 | 1.001 |
Level-1 CE: a 3.108 to 3.031, b 3.112 to 3.026, c 3.042 to 2.976. Dev loss is the same for all three variants within 0.004. Only c has a visibly lower level-1 CE.

## Results

### t0.7

| run | spurious | consistency | b0 carry-over | xblock 4gram rep | seam0 cond PPL | seam ratio (seam/within) | seam0 / random-next | within PPL |
|---|---|---|---|---|---|---|---|---|
| prev_a_t0.7 | 0.081 [0.060, 0.105] | 0.916 [0.891, 0.937] | 0.418 [0.380, 0.457] | 0.063 [0.048, 0.079] | 287.5 [247.7, 333.3] | 1.474 [1.308, 1.662] | 0.943 [0.811, 1.101] | 153.1 [134.9, 175.2] |
| prev_b_t0.7 | 0.032 [0.019, 0.048] | 0.966 [0.950, 0.979] | 0.475 [0.440, 0.515] | 0.041 [0.029, 0.055] | 272.3 [240.0, 309.3] | 1.226 [1.083, 1.396] | 0.893 [0.777, 1.032] | 185.6 [161.3, 213.8] |
| prev_c_t0.7 | 0.022 [0.014, 0.033] | 0.977 [0.966, 0.985] | 0.602 [0.568, 0.637] | 0.045 [0.033, 0.058] | 291.3 [253.1, 336.9] | 1.121 [1.003, 1.251] | 0.956 [0.819, 1.120] | 215.5 [193.1, 240.0] |
| v2_a_t0.7 | 0.060 [0.042, 0.080] | 0.933 [0.912, 0.952] | 0.490 [0.448, 0.531] | 0.030 [0.023, 0.037] | 225.6 [195.2, 261.6] | 1.017 [0.891, 1.163] | 0.740 [0.639, 0.862] | 192.6 [170.7, 217.9] |
| v2_b_t0.7 | 0.023 [0.014, 0.033] | 0.977 [0.967, 0.986] | 0.607 [0.574, 0.640] | 0.029 [0.022, 0.037] | 277.1 [239.7, 320.1] | 1.234 [1.091, 1.390] | 0.909 [0.780, 1.065] | 181.7 [161.8, 204.1] |
| v2_c_t0.7 | 0.019 [0.011, 0.028] | 0.980 [0.971, 0.989] | 0.626 [0.592, 0.662] | 0.035 [0.027, 0.043] | 239.0 [207.2, 274.0] | 1.223 [1.094, 1.361] | 0.784 [0.673, 0.910] | 165.3 [148.7, 184.4] |

### t0.0

| run | spurious | consistency | b0 carry-over | xblock 4gram rep | seam0 cond PPL | seam ratio (seam/within) | seam0 / random-next | within PPL |
|---|---|---|---|---|---|---|---|---|
| prev_a_t0.0 | 0.095 [0.050, 0.167] | 0.895 [0.820, 0.944] | 0.116 [0.094, 0.140] | 0.349 [0.316, 0.380] | 119.7 [101.2, 141.2] | 3.097 [2.694, 3.554] | 0.393 [0.330, 0.465] | 12.6 [10.8, 14.5] |
| prev_b_t0.0 | 0.068 [0.017, 0.106] | 0.932 [0.894, 0.983] | 0.176 [0.150, 0.204] | 0.316 [0.284, 0.347] | 129.7 [109.7, 153.4] | 3.061 [2.687, 3.466] | 0.425 [0.355, 0.512] | 13.1 [11.5, 15.0] |
| prev_c_t0.0 | 0.035 [0.006, 0.070] | 0.965 [0.930, 0.994] | 0.264 [0.233, 0.297] | 0.367 [0.334, 0.396] | 132.4 [114.2, 153.9] | 2.986 [2.624, 3.406] | 0.434 [0.370, 0.506] | 14.5 [12.4, 17.0] |
| v2_a_t0.0 | 0.095 [0.058, 0.136] | 0.894 [0.839, 0.939] | 0.190 [0.160, 0.221] | 0.318 [0.288, 0.346] | 110.6 [94.6, 130.4] | 2.543 [2.187, 2.963] | 0.363 [0.309, 0.427] | 15.8 [13.2, 19.3] |
| v2_b_t0.0 | 0.007 [0.000, 0.018] | 0.993 [0.982, 1.000] | 0.230 [0.195, 0.266] | 0.317 [0.287, 0.348] | 106.7 [90.7, 126.8] | 2.537 [2.194, 2.913] | 0.350 [0.295, 0.422] | 16.6 [14.0, 20.0] |
| v2_c_t0.0 | 0.019 [0.004, 0.043] | 0.981 [0.957, 0.996] | 0.306 [0.268, 0.346] | 0.327 [0.296, 0.356] | 107.3 [92.4, 126.2] | 2.790 [2.471, 3.133] | 0.352 [0.297, 0.421] | 15.3 [13.3, 17.9] |
| gold | 0 | 0.982 [0.971, 0.991] | 0.824 [0.795, 0.852] | 0.012 [0.010, 0.014] | | | | |

v2_gold: seam0 cond PPL 135.0 [118.9, 153.3], seam_ratio 1.072 [0.987, 1.168]

v2_control_random_next: seam0 cond PPL 304.8 [283.6, 326.3]

prev_gold: seam0 cond PPL 135.0 [118.9, 153.3], seam_ratio 1.072 [0.987, 1.168]

prev_control_random_next: seam0 cond PPL 304.8 [283.6, 326.3]

(Full CIs in metrics_v2.json and judge_v2.json. Samples in samples_v2.md.)

## Verdict
- Entities (conclusive): the entity table helps. At T=0.7 spurious-name rate is 6.0% for a (CI 4.2 to 8.0) vs 2.3% for b and 1.9% for c, with non-overlapping CIs. Block-0 carry-over is a 0.49, b 0.61, c 0.63 (gold 0.82); a vs b/c is significant, b vs c is not. Greedy agrees on spurious (a 9.5%, b 0.7%, c 1.9%) and on carry-over (c 0.31 > a 0.19, b 0.23). b vs c: no reliable difference. Training longer improved a (spurious 8.1% to 6.0%) and the cross-block 4-gram repeat (6% to 3% at T=0.7); b and c were already near their floor.
- Coarse codes (c vs b): no significant gain in entity metrics. Only a small dev level-1 CE gain (2.98 vs 3.03) and some carry-over gain (greedy 0.31 vs 0.23, CIs just touching). c costs about 37% more training time and about 2x inference latency.
- Seam (the open question): still not demonstrated. Seam0 / random-next is 0.74 [0.64, 0.86] for a, 0.91 [0.78, 1.07] for b and 0.78 [0.67, 0.91] for c at T=0.7. All are below 1, but the variant ordering is not the expected one (a, the text-only baseline, is best), and b has CIs overlapping 1. The seam ratio (seam/within) is about 1.0 to 1.2 at T=0.7, near gold 1.07 (CI 0.99 to 1.17). The gap to random-next is dominated by overall text quality, not by cross-block conditioning: within-block PPL itself varies 165 to 193 across variants, and greedy samples repeat. The judge cannot separate seam coherence from degenerate local text. Prefix conditioning on the previous block's text is enough for the metric to beat random; coarse codes and the table add nothing measurable on it.
- Text quality is still poor: samples show word doubling ("bee bee", "said said") and drifting story. Dev loss still falling at 4500 steps (best = last), so the models remain undertrained (about 0.55 epoch of the new data), but the doubling pattern suggests it is not just data volume; it should be checked in the level-wise sampler (generate() with the codec, temperature on levels) before spending more compute.
- Honest bottom line: entity side conclusive (table: yes; codes: no extra); seam side still not conclusive because the quality floor of the generator swamps the judge. To make seam conclusive, fix word doubling or use a seam metric normalised for quality, for example score teacher-forced gold continuations with the generator itself (cond NLL of gold block n+1 given the real prefix vs a shuffled prefix), which needs no sampling.

## Files
runs/v2/{metrics_v2.json, judge_v2.json, samples_v2.md, REPORT_v2.table.md, judge_raw_*.json, gen_*.json, a|b|c/ (checkpoints), prev_*/dev_eval.json, job.log}. Code: prep_v2.py, run_v2.sh, judge_v2.py, report_v2.py (train_seam.py gained --train/--init/--warm/--eval_only and the dev fix). Data: data/train2.json, data/codes_train2.npy; download dl/0000.parquet (249 MB, deletable).
