"""10-fold CV sweep: ridge weight on psi, per-model vs per-lab psi, free f vs f = 0 below gap 0."""
import time, zipfile, numpy as np, pandas as pd
from multiprocessing import Pool
import bmx
from eci import prepare_benchmark_data

df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
meta = pd.read_csv(zipfile.ZipFile("data/benchmark_data.zip").open("model_metadata.csv"))
ORG = meta.drop_duplicates("model_group").set_index("model_group")["organization"].fillna("unknown").to_dict()
folds = np.random.default_rng(7).integers(0, 10, len(df))  # same folds as 02_cv.py

SPECS = {"baseline": dict(with_psi=False)}
for lam in [None, 0.01, 0.1, 1.0]:
    SPECS[f"model free lam={lam}"] = dict(lam_psi=lam)
for lam in [None, 0.1]:
    SPECS[f"org free lam={lam}"] = dict(lam_psi=lam, psi_group=ORG)
for lam in [None, 0.1]:
    SPECS[f"model zero_below lam={lam}"] = dict(lam_psi=lam, zero_below=True)

def run(args):
    name, k = args
    train, test = df[folds != k], df[folds == k]
    ok = test.model_id.isin(train.model_id) & test.benchmark_id.isin(train.benchmark_id)
    kw = dict(with_psi=True, **SPECS[name]) if "with_psi" not in SPECS[name] else dict(SPECS[name])
    t = time.time()
    o = bmx.fit(train, index=bmx._index(train), **kw)
    pred = bmx.predict(o, test[ok])
    y = test[ok].performance.clip(1e-3, 1 - 1e-3).to_numpy()
    return name, k, float(np.sum((pred - y) ** 2)), int(ok.sum()), bool(o["success"]), time.time() - t, int(o["nfev"])

if __name__ == "__main__":
    import sys
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = [(n, k) for k in range(10) for n in SPECS]  # fold-major: first len(SPECS) results time every spec
    t = time.time()
    rows = []
    print(f"{len(jobs)} fits, {workers} workers", flush=True)
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run, jobs), 1):
            rows.append(row)
            name, k, sse, n, ok, secs, nfev = row
            el = time.time() - t
            eta = el / i * (len(jobs) - i)
            print(f"[{i:3d}/{len(jobs)}] {time.strftime('%H:%M:%S')} {name:28s} fold {k} "
                  f"{secs:6.1f}s nfev {nfev:5d} ok {ok}  elapsed {el/60:5.1f}m eta {eta/60:5.1f}m", flush=True)
            pd.DataFrame(rows, columns=["spec", "fold", "sse", "n", "success", "secs", "nfev"]).to_csv("out/cv03.csv", index=False)
    r = pd.DataFrame(rows, columns=["spec", "fold", "sse", "n", "success", "secs", "nfev"])
    base = r[r.spec == "baseline"].set_index("fold").sse
    s = r.groupby("spec", sort=False).agg(sse=("sse", "sum"), n=("n", "sum"), all_ok=("success", "all"),
                                          secs=("secs", "mean"), nfev=("nfev", "mean"))
    s["rmse"] = np.sqrt(s.sse / s.n)
    s["folds_beating_baseline"] = [int((r[r.spec == sp].set_index("fold").sse < base).sum()) for sp in s.index]
    print(s.round(4).to_string())
    print(f"total {time.time()-t:.0f}s", flush=True)
