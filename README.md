# eci-benchmaxxing

Work in progress from QKV Labs. We extend the [Epoch Capabilities Index](https://epoch.ai/benchmarks/eci) (ECI) to estimate how much individual models over-perform on specific benchmarks relative to their general capability.

ECI fits `performance(m, b) = sigmoid(alpha_b * (C_m - D_b))` across models `m` and benchmarks `b` ([Ho et al., "A Rosetta Stone for AI Benchmarks", arXiv 2512.00193](https://arxiv.org/abs/2512.00193)). We add a model-level term to that fit.

## Reproduce

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/get_data.py            # downloads Epoch's benchmark_data.zip
.venv/bin/python analysis/00_reproduce.py       # refits Epoch's baseline ECI
```

Epoch republishes `benchmark_data.zip` as new results arrive. `data/SNAPSHOT.json` records the sha256 of the copy our results use; `scripts/get_data.py --check` compares a local copy against it.

## Credits

- Benchmark data: [Epoch AI](https://epoch.ai/benchmarks), CC-BY 4.0.
- Baseline fitting code: [epoch-research/eci-public](https://github.com/epoch-research/eci-public), MIT, pinned in `requirements.txt`.

This project is not affiliated with Epoch AI.
