#!/bin/zsh
# Seam experiment end to end (GPU: train 3 variants ~66 min each on M1 Max). Run from repo root with the venv active.
# Prereqs: $NCM_WORK/ts = converted official TinyStories codec; $NCM_SEAM_DATA/valid.txt = TinyStories validation text.
# On a shared machine wrap in your own GPU queue. Nothing here uploads anything.
set -e
export NCM_WORK=${NCM_WORK:-w} PYTHONUNBUFFERED=1
export NCM_SEAM_DATA=${NCM_SEAM_DATA:-$NCM_WORK/seam_data} NCM_SEAM_RUNS=${NCM_SEAM_RUNS:-$NCM_WORK/seam_runs}
S=src/ncm_modular/seam; mkdir -p $NCM_SEAM_DATA $NCM_SEAM_RUNS
[ -f $NCM_SEAM_DATA/names.json ] || python $S/names.py $NCM_SEAM_DATA/valid.txt $NCM_SEAM_DATA/names.json
[ -f $NCM_SEAM_DATA/train.json ] || python $S/prep.py
for v in a b c; do echo "=== train $v $(date)"; python $S/train_seam.py --v $v --out $NCM_SEAM_RUNS/$v --steps 2800 --bs 32 --minutes 66 --eval_every 400; done
echo "=== gen $(date)"; python $S/gen_eval.py --variants abc --temps 0 0.7
python $S/metrics.py
echo "=== judge $(date)"; python $S/judge.py     # needs mlx_lm and NCM_JUDGE_MODEL
echo "=== done $(date)"
