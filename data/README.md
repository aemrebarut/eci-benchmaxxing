# Data

`benchmark_data.zip` is a snapshot of Epoch AI's benchmark data, retrieved from https://epoch.ai/data/benchmark_data.zip on 2026-09-29 (see `SNAPSHOT.json` for the sha256). It is redistributed unmodified under CC-BY 4.0; credit: Epoch AI, https://epoch.ai/benchmarks. The MIT license of this repository covers our code, not this data.

`snapshot-2026-10-01/benchmark_data.zip` is a later snapshot of the same file, retrieved on 2026-10-01 (sha256 and the 51 cells it adds in its `SNAPSHOT.json`), redistributed unmodified under the same license and credit. Only `analysis/32_snapshot_refit.py` uses it.

`benchmark_filtering.csv` is ours: for each of the 60 benchmarks, whether its items were kept or dropped based on whether AI models could solve them (`yes`, `partial`, `no`), the models used, the mechanism, primary sources and a supporting quote, and our confidence. Compiled from the cited papers and pages on 2026-09-30; several classifications are judgment calls, reflected in the `confidence` column.
