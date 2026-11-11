#!/usr/bin/env bash
# usage: ./run.sh <ar|ncm|ncm_blocks12> [--smoke|--cpu-tiny] [key=value ...]
# resumes automatically from <out_dir>/last.pt. Data: python prep_data.py ...; codec: ./fetch_codec.sh (ncm only, XSum codec).
set -euo pipefail
cd "$(dirname "$0")"; export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
arm="${1:?arm: ar | ncm | ncm_blocks12}"; shift
case "$arm" in ar) cfgs=(ar_150m.yaml);; ncm) cfgs=(ncm_150m.yaml);; ncm_blocks12) cfgs=(ncm_150m_blocks12.yaml);; *) echo "bad arm"; exit 2;; esac
tag="$arm"; extra=()
if [[ "${1:-}" == "--smoke" ]]; then cfgs+=(smoke_4080s.yaml); tag="${arm}_smoke"; shift
elif [[ "${1:-}" == "--cpu-tiny" ]]; then cfgs+=(cpu_tiny.yaml); tag="${arm}_tiny"; shift; fi
exec python -m ncm150.train --config "${cfgs[@]}" --set "out_dir=runs/${tag}" "$@"
