"""The headline step model refitted on Epoch's data of 2026-10-01 (data/snapshot-2026-10-01/, 51 cells more than the
2026-09-29 snapshot the other analyses use; see its SNAPSHOT.json). Google merged, near-unpenalized (lam 1e-3), both
structures, Epoch's ("epoch") and with benchmark drift ("drift", mu 0.1), as in analysis/23_appendix.py. The two z
values of the contrast are the "data of 2026-10-01" rows of analysis/24_speccurve.py.

The rows per cell and the unclipped maximum that analysis/08_data_check.py writes for the main snapshot are rebuilt
here from this snapshot's own rows (out/data_check08_20261001.csv), so out/data_check08.csv is not touched.
Writes out/app32_snapshot.csv. Usage: python analysis/32_snapshot_refit.py"""
import importlib, os, sys
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import cm
import robust as R
from robust import STRUCT

A = importlib.import_module("23_appendix")
SNAP = "data/snapshot-2026-10-01/benchmark_data.zip"
CHECK = "out/data_check08_20261001.csv"


def main():
    os.makedirs("out", exist_ok=True)
    s = R.rows(SNAP)
    s.groupby(["model_group", "benchmark"]).agg(n_rows=("performance", "size"), raw_max=("performance", "max")) \
        .reset_index().rename(columns={"model_group": "Model"}).to_csv(CHECK, index=False)
    load = cm.load
    cm.load = lambda: load(src=SNAP, check=CHECK)
    try:
        df = R.load()
    finally:
        cm.load = load
    out = pd.DataFrame([A.fit_summary(df, struct) | {"cells": len(df)} for struct in STRUCT])
    out.to_csv("out/app32_snapshot.csv", index=False)
    print(out[["struct", "cells", "status", "g_est", "g_se_mb", "g_z_mb", "c_est", "c_se_mb", "c_z_mb"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
