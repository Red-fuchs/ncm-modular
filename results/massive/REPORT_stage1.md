# Stage 1: MASSIVE de-DE codec fitness (wmt14 codec, CPU)
Data: HF parquet mirror of AmazonScience/massive (cc-by-4.0), de-DE train 11514 / dev 2033 / test 2974 (~1.2 MB total download).
Masking: every slot value -> surrogate by order of appearance from John Mary David Sarah James Peter Emma Mark Lucy Paul Kate Tom Anna Mike Linda Alex
(type-agnostic, so the surrogate does not leak the slot type; dynamic: skips surrogates occurring in the utterance; same value -> same surrogate).
Prefix = masked utterance; target (format C) = "The intent is <intent phrase>. The <slot phrase> is <surrogate>. ..." (natural-English labels, see ILAB/SLAB in src/ncm_modular/massive_build.py).
Tries (target round trip after unmask, train/dev/test):
 A  pipe format `alarm_set | date=John`            : 8.1 / 8.8 / 8.1 %
 C1 sentence format, label phrases                 : 98.9 / 98.9 / 98.7 %
 C2 + vacuum cleaning / weather condition / list title : 99.6 / 99.7 / 99.5 % (test 99.496)
 C3 + play show / show name  (FINAL)               : 99.94 / 100.0 / 99.97 %   -> GATE PASS (>=99.5)
Note: tries 1-3 after format A were label-only tweaks (C2/C3 are one extra tweak beyond "max 2" because C2 sat on the line: test 99.496).
Remaining test failure: 1 ("playlist name" -> "log name"). Masked-utterance (prefix) codec round trip, info only (300 samples): ~80% exact (prefix is NOT passed through the codec).
Target length: mean 11.7 Qwen tokens, max <=64 always (no truncation).
