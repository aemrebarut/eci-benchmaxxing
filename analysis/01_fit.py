"""First fit of the benchmaxxing model: multi-start, compare with baseline, print f and psi."""
import time, zipfile, numpy as np, pandas as pd
import bmx
from eci import prepare_benchmark_data

pd.set_option("display.width", 200)
df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
meta = pd.read_csv(zipfile.ZipFile("data/benchmark_data.zip").open("model_metadata.csv"))
org = meta.drop_duplicates("model_group").set_index("model_group")["organization"]

base = bmx.fit(df, with_psi=False)
fits = []
for seed in range(6):
    t = time.time()
    o = bmx.fit(df, with_psi=True, seed=seed)
    fits.append(o)
    print(f"seed {seed}: sse {o['sse']:.4f} cost {o['cost']:.5f} nfev {o['nfev']} ok {o['success']} {time.time()-t:.1f}s")
best = min(fits, key=lambda o: o["cost"])
print(f"baseline sse {base['sse']:.4f} ({base['n_params']} params); psi model sse {best['sse']:.4f} ({best['n_params']} params); n_obs {best['n_obs']}")
psis = np.array([o["models"].psi_eci for o in fits])
print("max psi_eci spread across starts", np.abs(psis - psis[np.argmin([o['cost'] for o in fits])]).max().round(3))

sp = best["spline"]
print("knots (years)", sp.knots.round(2))
grid = np.array([-3, -2, -1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7])
print("f(gap):", dict(zip(grid, sp(grid, best["beta"]).round(3))))

m = best["models"].merge(base["models"][["Model", "eci"]], on="Model", suffixes=("", "_base"))
st = df.groupby("Model").agg(n=("gap", "size"), n_neg=("gap", lambda g: (g < 0).sum()),
                             gap_max=("gap", "max"), date=("date", "first"))
m = m.join(st, on="Model")
m["org"] = m.Model.map(org)
print("psi_eci quantiles", m.psi_eci.quantile([0, .05, .25, .5, .75, .95, 1]).round(2).to_dict())
print("mean psi_eci", m.psi_eci.mean().round(3), "share at bound", (m.psi.abs() > 9.99).mean())
cols = ["Model", "org", "date", "n", "n_neg", "eci_base", "eci", "psi_eci"]
print("\nTop 20 psi (n >= 8):"); print(m[m.n >= 8].sort_values("psi_eci", ascending=False)[cols].head(20).round(2).to_string(index=False))
print("\nBottom 10 psi (n >= 8):"); print(m[m.n >= 8].sort_values("psi_eci")[cols].head(10).round(2).to_string(index=False))
m.to_csv("out/fit01_models.csv", index=False)
