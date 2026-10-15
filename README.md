# eci-benchmaxxing

Work in progress from Emre Barut. We extend the [Epoch Capabilities Index](https://epoch.ai/benchmarks/eci) (ECI) to estimate how much individual models over-perform on specific benchmarks relative to their general capability.

ECI fits `performance(m, b) = sigmoid(alpha_b * (C_m - D_b))` across models `m` and benchmarks `b` ([Ho et al., "A Rosetta Stone for AI Benchmarks", arXiv 2512.00193](https://arxiv.org/abs/2512.00193)). We add a model-level term to that fit.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/get_data.py --check    # verifies data/benchmark_data.zip
.venv/bin/python analysis/00_reproduce.py       # refits Epoch's baseline ECI
```

Epoch republishes `benchmark_data.zip` as new results arrive, so this repo includes the snapshot our results use (`data/benchmark_data.zip`, retrieved 2026-10-15, sha256 in `data/SNAPSHOT.json`). The earlier snapshots of 2026-09-29 and 2026-10-01 are kept in `data/snapshot-<date>/`, and `analysis/32_snapshot_refit.py` refits the headline on either. `scripts/get_data.py` without flags fetches the latest version instead.

The scripts in `analysis/` are numbered in the order they were written; each one's docstring says what it reads, what it writes to `out/` and how to run it. Run `analysis/08_data_check.py` first, because the later analyses read its `out/data_check08.csv`. Run `analysis/23_appendix.py` (one study per run) before `analysis/24_speccurve.py`, and `analysis/28_movement.py` before `analysis/29_launch.py`.

## Credits

- Benchmark data: [Epoch AI](https://epoch.ai/benchmarks), CC-BY 4.0, redistributed unmodified in `data/`.
- Baseline fitting code: [epoch-research/eci-public](https://github.com/epoch-research/eci-public), MIT, pinned in `requirements.txt`.

Code in this repository is MIT licensed (see `LICENSE`). This project is not affiliated with Epoch AI.
