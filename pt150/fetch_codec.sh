#!/usr/bin/env bash
# usage: ./fetch_codec.sh [xsum|owt]   downloads an upstream codec (MIT, hazyresearch/ncm-checkpoints). Not redistributed here.
# xsum (default): GPT-2 BPE, 48-token codec, ~300 MB, pinned to the sha256 in configs/base.yaml.
# owt: 1024-token codec, NOT published upstream yet (owt/ only holds TODO.md) -> fails with a clear message.
set -euo pipefail
cd "$(dirname "$0")"
t="${1:-xsum}"; f="artifacts/$t/tokenizer.pt"
if ! hf download hazyresearch/ncm-checkpoints "$t/tokenizer.pt" --local-dir artifacts; then
  [[ "$t" == owt ]] && echo "OWT codec not published upstream yet; use the default: ./fetch_codec.sh xsum" >&2
  exit 1
fi
if [[ "$t" == xsum ]]; then
  want=$(sed -n 's/^  sha256: \([0-9a-f]*\).*/\1/p' configs/base.yaml)
  got=$( (sha256sum "$f" 2>/dev/null || shasum -a 256 "$f") | cut -d' ' -f1)
  [[ "$got" == "$want" ]] || { echo "sha256 mismatch for $f: $got (want $want)" >&2; exit 1; }
  echo "ok: $f sha256 $got"
fi
