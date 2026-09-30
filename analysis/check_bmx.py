"""Sanity checks: baseline matches Epoch's fit; analytic Jacobian matches finite differences."""
import numpy as np, pandas as pd
from scipy.optimize._numdiff import approx_derivative
import bmx
from eci import prepare_benchmark_data, fit_eci_model

df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
base = bmx.fit(df, with_psi=False)
eci_epoch, _, _ = fit_eci_model(df, bootstrap_samples=0)
m = base["models"].merge(eci_epoch[["Model", "eci"]], on="Model", suffixes=("_ours", "_epoch"))
print("baseline vs epoch: max abs diff", np.abs(m.eci_ours - m.eci_epoch).max().round(6), "sse", round(base["sse"], 4))

# Jacobian check on a subsample, at a random point
sub = df.sample(400, random_state=0)
sub = sub[sub.model_id.isin(sub.model_id.value_counts()[lambda s: s >= 1].index)]
for anchor in ["Winogrande"]:
    assert anchor in set(sub.benchmark), "anchor missing in subsample"
