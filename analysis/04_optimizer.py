"""Does the optimizer reach the minimum? Same objective, different optimizer settings, 3 starts each."""
import time, numpy as np, pandas as pd
from multiprocessing import Pool
import bmx
from eci import prepare_benchmark_data

df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
CONFIGS = {
    "A epoch (single row, lsmr)": dict(),
    "B single row, exact": dict(tr_solver="exact"),
    "C per-param rows, lsmr": dict(ridge_rows="per_param"),
    "D per-param rows, exact": dict(ridge_rows="per_param", tr_solver="exact"),
    "E per-param, exact, x_scale=jac": dict(ridge_rows="per_param", tr_solver="exact", x_scale="jac"),
    "F per-param, exact, tol 1e-12": dict(ridge_rows="per_param", tr_solver="exact", tol=1e-12),
}
MODELS = {"baseline": dict(with_psi=False), "psi": dict(with_psi=True)}

def run(args):
    mname, cname, seed = args
    t = time.time()
    o = bmx.fit(df, seed=seed, **MODELS[mname], **CONFIGS[cname])
    psi = o["models"]["psi_eci"].to_numpy() if "psi_eci" in o["models"] else None
    return dict(model=mname, config=cname, seed=seed, cost=o["cost"], sse=o["sse"], nfev=o["nfev"],
                status=o["status"], optimality=o["optimality"], active=o["n_active"], secs=time.time() - t,
                eci=o["models"]["eci"].to_numpy(), psi=psi)

if __name__ == "__main__":
    jobs = [(m, c, s) for m in MODELS for c in CONFIGS for s in range(3)]
    with Pool(4) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    r = pd.DataFrame(rows)
    best = r.groupby("model").cost.min()
    r["excess_cost"] = r.cost - r.model.map(best)
    pd.set_option("display.width", 220)
    print(r.drop(columns=["eci", "psi"]).round({"cost": 6, "sse": 4, "optimality": 6, "secs": 1, "excess_cost": 6}).to_string(index=False))
    for m in MODELS:
        rb = r[r.model == m].reset_index(drop=True)
        ib = rb.cost.idxmin()
        print(f"\n{m}: max |ECI - best| per config (over seeds):")
        for c in CONFIGS:
            sub = rb[rb.config == c]
            d_eci = max(np.abs(e - rb.eci[ib]).max() for e in sub.eci)
            line = f"  {c:35s} eci {d_eci:8.3f}"
            if m == "psi":
                d_psi = max(np.abs(p - rb.psi[ib]).max() for p in sub.psi)
                line += f"  psi {d_psi:8.3f}"
            print(line)
    r.drop(columns=["eci", "psi"]).to_csv("out/opt04.csv", index=False)
