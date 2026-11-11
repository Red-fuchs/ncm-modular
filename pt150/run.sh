#!/usr/bin/env bash
# usage: ./run.sh <ar|ncm|ncm_blocks12> [--ctx1024] [--smoke|--cpu-tiny] [key=value ...]
# --ctx1024: 976-token raw prefix + 48-token codec block (LLM-shaped comparison); default: 48-token blocks, BOS-only prefix.
# resumes automatically from <out_dir>/last.pt. Data: python prep_data.py ...; codec: ./fetch_codec.sh (ncm only, XSum codec).
set -euo pipefail
cd "$(dirname "$0")"; export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
arm="${1:?arm: ar | ncm | ncm_blocks12}"; shift
case "$arm" in ar) cfgs=(ar_150m.yaml);; ncm) cfgs=(ncm_150m.yaml);; ncm_blocks12) cfgs=(ncm_150m_blocks12.yaml);; *) echo "bad arm"; exit 2;; esac
tag="$arm"; ctx=""
if [[ "${1:-}" == "--ctx1024" ]]; then cfgs+=(ctx1024.yaml); ctx=_ctx1024; tag+="$ctx"; shift; fi
if [[ "${1:-}" == "--smoke" ]]; then
  if [[ -n "$ctx" ]]; then cfgs+=(smoke_ctx1024_4080s.yaml); else cfgs+=(smoke_4080s.yaml); fi; tag+="_smoke"; shift
elif [[ "${1:-}" == "--cpu-tiny" ]]; then cfgs+=(cpu_tiny.yaml); tag+="_tiny"; shift; fi
exec python -m ncm150.train --config "${cfgs[@]}" --set "out_dir=runs/${tag}" "$@"
