#!/usr/bin/env bash
# downloads the upstream OWT codec (MIT, hazyresearch/ncm-checkpoints). Not redistributed here.
set -euo pipefail
cd "$(dirname "$0")"
hf download hazyresearch/ncm-checkpoints owt/tokenizer.pt --local-dir artifacts
