"""10-fold CV of the scale-controlled fits (same folds as 02 and 03). Each config and fold gets
several random starts; the start with the lowest training cost is the fit, as it would be on full data."""
import sys, time, zipfile, numpy as np, pandas as pd
from multiprocessing import Pool
import bmx
from eci import prepare_benchmark_data

df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
meta = pd.read_csv(zipfile.ZipFile("data/benchmark_data.zip").open("model_metadata.csv"))
ORG = meta.drop_duplicates("model_group").set_index("model_group")["organization"].fillna("unknown").to_dict()
folds = np.random.default_rng(7).integers(0, 10, len(df))  # same folds as 02_cv.py and 03_cv_sweep.py
MAX_NFEV = 400
SEEDS = range(4)
ZB = "--zero-below" in sys.argv  # f(gap) = 0 for gap <= 0
OUT = "out/cv07_zb.csv" if ZB else "out/cv07.csv"

SPECS = {"baseline": dict(with_psi=False)}
for unit in ["model", "org"]:
    for kappa in [0.01, 0.1, 1.0]:
        SPECS[f"{unit} kappa={kappa}"] = dict(with_psi=True, scale="penalty", kappa=kappa, zero_below=ZB,
                                              psi_group=ORG if unit == "org" else None)
COLS = ["spec", "fold", "seed", "train_cost", "sse", "n", "status", "secs", "nfev"]


def run(args):
    name, k, seed = args
    train, test = df[folds != k], df[folds == k]
    ok = test.model_id.isin(train.model_id) & test.benchmark_id.isin(train.benchmark_id)
    t = time.time()
    o = bmx.fit(train, index=bmx._index(train), seed=seed, max_nfev=MAX_NFEV, **SPECS[name])
    pred = bmx.predict(o, test[ok])
    y = test[ok].performance.clip(1e-3, 1 - 1e-3).to_numpy()
    return name, k, seed, float(o["cost"]), float(np.sum((pred - y) ** 2)), int(ok.sum()), int(o["status"]), time.time() - t, int(o["nfev"])


def summarize(r):
    best = r.loc[r.groupby(["spec", "fold"]).train_cost.idxmin()]
    base = best[best.spec == "baseline"].set_index("fold").sse
    rows = []
    for spec, g in best.groupby("spec", sort=False):
        a = r[r.spec == spec]
        per_start = a.groupby("seed").apply(lambda d: np.sqrt(d.sse.sum() / d.n.sum()), include_groups=False)
        rows.append(dict(spec=spec, folds=g.fold.nunique(), rmse=np.sqrt(g.sse.sum() / g.n.sum()),
                         folds_beating_baseline=int((g.set_index("fold").sse < base.reindex(g.fold)).sum()),
                         rmse_start_min=per_start.min(), rmse_start_max=per_start.max(),
                         hit_max_nfev=int((a.status == 0).sum()), fits=len(a),
                         mean_secs=a.secs.mean(), mean_nfev=a.nfev.mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = [(n, k, s) for k in range(10) for s in SEEDS for n in SPECS if not (n == "baseline" and s > 0)]
    t = time.time()
    rows = []
    print(f"{len(jobs)} fits, {workers} workers, max_nfev {MAX_NFEV}", flush=True)
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run, jobs), 1):
            rows.append(row)
            name, k, seed, cost, sse, n, status, secs, nfev = row
            el = time.time() - t
            eta = el / i * (len(jobs) - i)
            print(f"[{i:3d}/{len(jobs)}] {time.strftime('%H:%M:%S')} {name:18s} fold {k} seed {seed} "
                  f"{secs:6.1f}s nfev {nfev:4d} st {status}  elapsed {el/60:5.1f}m eta {eta/60:5.1f}m", flush=True)
            pd.DataFrame(rows, columns=COLS).to_csv(OUT, index=False)
    s = summarize(pd.DataFrame(rows, columns=COLS))
    print(s.round(4).to_string(index=False))
    print(f"total {time.time()-t:.0f}s", flush=True)
