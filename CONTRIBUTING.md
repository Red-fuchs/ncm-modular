# Contributing

Thanks for your interest. Directions are in the README ([Roadmap](README.md#roadmap), [Contributing / collaboration](README.md#contributing--collaboration)). Reproductions, negative results and corrections are as welcome as code.

## Ways to contribute

- **Reproduce or break a result.** Open an issue with the *Result reproduction / correction* template: command, commit, hardware, macOS + MLX version, your numbers vs ours.
- **Bug reports.** *Bug report* template, with the full traceback.
- **Collaboration / compute** (stage 0, multilingual codec, 1B base, modules): *Collaboration* template.
- **Pull requests.** Small and focused. Open an issue first for anything larger than a fix.

## Ground rules for code and results

- No weights, datasets or large binaries in the repo. Heavy files live in `$NCM_WORK` (`w/`, git-ignored); fetch scripts download them from upstream.
- Keep licensing clean: do not vendor upstream source (HazyResearch/ncm) or redistribute data. Add any new third-party item to `LICENSES.md`.
- Every reported number needs a stored log or JSON under `results/` and the exact command that produced it. State seeds, steps, hardware and whether power/thermal state was checked (Low Power Mode cost ~46 % in our runs).
- Report negative results and caveats in `RESULTS.md`; fix mistakes with an erratum, do not silently rewrite numbers.
- Match the existing style: plain scripts under `src/ncm_modular/`, env vars for paths (`NCM_WORK`, `NCM_UPSTREAM`, ...), no absolute or personal paths.
- Parity first: changes to `codec_mlx.py` must keep `parity.py` at the levels reported in the README.

## Setup

See [Reproduce](README.md#reproduce). Python 3.11+, Apple Silicon, `mlx==0.32.3`.

## Conduct and license

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Contributions are accepted under the MIT license (`LICENSE`).
