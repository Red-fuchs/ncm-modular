# Security policy

This is research code that runs locally (MLX on Apple Silicon). It ships no model weights and no network service.

## Reporting

Please report vulnerabilities privately through GitHub: **Security tab → "Report a vulnerability"** on this repository. Do not open a public issue for security problems.

Relevant scope: code execution through the conversion or loading scripts (`convert.py`, `ref_dump.py`, checkpoint/`safetensors`/`.npz` loading), the fetch script (`scripts/fetch_and_convert.sh`), and dependency pins in `pyproject.toml`. Upstream components (HazyResearch/ncm, janmus1c/ncm-mlx, MLX) should be reported to their own projects.

Only the latest commit on `main` is supported. This is a spare-time project; expect a reply within about two weeks.
