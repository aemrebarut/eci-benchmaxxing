"""10-fold cross-validation over observed cells: baseline vs psi (free f) vs psi (f = 0 for gap <= 0)."""
import sys, time, numpy as np, pandas as pd
from multiprocessing import Pool
import bmx
from eci import prepare_benchmark_data

df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
IDX = bmx._index(df)
SPECS = {"baseline": dict(with_psi=False), "free_f": dict(with_psi=True), "zero_below": dict(with_psi=True, zero_below=True)}
rng = np.random.default_rng(7)
folds = rng.integers(0, 10, len(df))

def run(args):
    name, k = args
    train, test = df[folds != k], df[folds == k]
    # keep test rows whose model and benchmark still appear in training
    ok = test.model_id.isin(train.model_id) & test.benchmark_id.isin(train.benchmark_id)
    kw = dict(SPECS[name])
    if kw.get("with_psi"):
        g = train.gap.to_numpy()
        sp = bmx.Spline.from_quantiles(g[g > 0] if kw.get("zero_below") else g, 5)
        sp.zero_below = kw.get("zero_below", False)
        kw["spline"] = sp
    o = bmx.fit(train, index=bmx._index(train), **kw)
    pred = bmx.predict(o, test[ok])
    y = test[ok].performance.clip(1e-3, 1 - 1e-3).to_numpy()
    return name, k, np.sum((pred - y) ** 2), ok.sum(), (~ok).sum()

if __name__ == "__main__":
    t = time.time()
    with Pool(4) as pool:
        rows = pool.map(run, [(n, k) for n in SPECS for k in range(10)])
    r = pd.DataFrame(rows, columns=["spec", "fold", "sse", "n", "dropped"])
    s = r.groupby("spec").agg(sse=("sse", "sum"), n=("n", "sum"), dropped=("dropped", "sum"))
    s["rmse"] = np.sqrt(s.sse / s.n)
    print(s.round(4).to_string()); print(f"{time.time()-t:.0f}s")
    per = r.pivot(index="fold", columns="spec", values="sse")
    print("folds where free_f beats baseline", (per.free_f < per.baseline).sum(), "/10; zero_below beats baseline", (per.zero_below < per.baseline).sum(), "/10")
    r.to_csv("out/cv02.csv", index=False)
