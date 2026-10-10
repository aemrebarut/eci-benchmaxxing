"""Residual profile by benchmark date: each cell's residual from Epoch's structure without steps, in ECI points
(logit(y) - logit(yhat), divided by the benchmark slope a_j and multiplied by the unit b). Writes out/app25_profile.csv
for Figure A.5 of the appendix. Run from the repo root."""
import os, sys
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import robust as R

df = R.load()
PP, base, res, lam, mu = R.fit(df, mu=None)
x = base.xf
C, D, a, *_ = PP.unpack(x)
_, b = PP.scale(x)
yhat = np.clip(PP.predict(x), 1e-3, 1 - 1e-3)
y = np.clip(PP.y, 1e-3, 1 - 1e-3)
lg = lambda p: np.log(p / (1 - p))
aj = a[PP.bi]
out = df[["Model", "group", "benchmark", "bdate", "gap"]].copy()
out["resid_eci"] = b * (lg(y) - lg(yhat)) / aj
out["clipped"] = (PP.y <= 1e-3 + 1e-12) | (PP.y >= 1 - 1e-3 - 1e-12)
out.to_csv("out/app25_profile.csv", index=False)
g = out[out.group == R.GOOG]
print("b", round(b, 2), "cells", len(out), "google", len(g))
print(g.assign(post=g.gap > 0, era=pd.cut(g.bdate, [2015, 2023, 2024.5, 2025.5, 2027])).groupby(["era", "post"], observed=True).resid_eci.agg(["mean", "median", "size"]).round(2))
o = out[(out.group != R.GOOG)]
print(o.assign(post=o.gap > 0, era=pd.cut(o.bdate, [2015, 2023, 2024.5, 2025.5, 2027])).groupby(["era", "post"], observed=True).resid_eci.agg(["mean", "median", "size"]).round(2))
