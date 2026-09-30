"""Compare the analytic Jacobian in bmx.fit with finite differences at a random point."""
import numpy as np
from scipy.optimize._numdiff import approx_derivative
import scipy.optimize
import bmx
from eci import prepare_benchmark_data

df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
captured = {}
orig = scipy.optimize.least_squares
def spy(fun, x0, jac=None, **kw):
    captured.update(fun=fun, jac=jac, x0=x0)
    raise StopIteration
bmx.least_squares = spy
import zipfile, pandas as pd
meta = pd.read_csv(zipfile.ZipFile("data/benchmark_data.zip").open("model_metadata.csv"))
org = meta.drop_duplicates("model_group").set_index("model_group")["organization"].fillna("unknown").to_dict()
for label, kw in [("per-model", {}), ("per-model lam_psi=0.3", {"lam_psi": 0.3}), ("per_param rows exact", {"ridge_rows": "per_param", "tr_solver": "exact", "lam_psi": 0.3}), ("per_param rows sparse org", {"ridge_rows": "per_param", "psi_group": org}),
                  ("per-org zero_below", {"psi_group": org, "zero_below": True, "lam_psi": 0.05}),
                  ("penalty per-model", {"scale": "penalty", "kappa": 0.1}),
                  ("penalty per-org", {"scale": "penalty", "kappa": 1.0, "psi_group": org}),
                  ("penalty zero_below", {"scale": "penalty", "kappa": 0.01, "zero_below": True})]:
    try:
        bmx.fit(df, with_psi=True, **kw)
    except StopIteration:
        pass
    rng = np.random.default_rng(1)
    x = captured["x0"] + rng.normal(0, 0.3, captured["x0"].size)
    x[len(x) - 3:] = rng.normal(0, 0.5, 3)
    J = captured["jac"](x); Ja = J if isinstance(J, np.ndarray) else J.toarray()
    Jn = approx_derivative(captured["fun"], x, method="3-point")
    err = np.abs(Ja - Jn)
    print(label, "max abs err", err.max(), "rel", err.max() / np.abs(Jn).max())
