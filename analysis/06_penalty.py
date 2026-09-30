"""Scale-controlled fits: kappa * (mean psi^2 + mean f^2), per-model and per-org psi, several starts each."""
import sys, time, zipfile, numpy as np, pandas as pd
from multiprocessing import Pool
import bmx
from eci import prepare_benchmark_data

df = bmx.add_gap(prepare_benchmark_data(source="data/benchmark_data.zip"))
meta = pd.read_csv(zipfile.ZipFile("data/benchmark_data.zip").open("model_metadata.csv"))
ORG = meta.drop_duplicates("model_group").set_index("model_group")["organization"].fillna("unknown").to_dict()
N_CELLS = df.groupby("Model").size()
GRID = [-3, -1, 0, 0.5, 1, 2, 4, 7]
ZB = "--zero-below" in sys.argv  # f(gap) = 0 for gap <= 0
OUT = "out/pen06_zb" if ZB else "out/pen06"
KAPPAS = [0.01, 0.1, 1.0]
SEEDS = range(8)
CONFIGS = [(u, k) for u in ["model", "org"] for k in KAPPAS]


def run(job):
    (unit, kappa), seed = job
    t = time.time()
    o = bmx.fit(df, with_psi=True, scale="penalty", kappa=kappa, seed=seed, zero_below=ZB,
                psi_group=ORG if unit == "org" else None)
    fg = o["spline"](np.array(GRID, float), o["beta"])
    eff = o["b"] * o["models"].psi_raw.to_numpy()[o["mi"]] * o["spline"](df.gap.to_numpy(), o["beta"])
    return dict(unit=unit, kappa=kappa, seed=seed, cost=o["cost"], sse=o["sse"], nfev=o["nfev"],
                status=o["status"], optimality=o["optimality"], n_active=o["n_active"],
                rms_psi=o["rms_psi"], rms_f=o["rms_f"], f1=o["f1"], secs=time.time() - t,
                f_raw=fg, eff=np.abs(eff), eci=o["models"].eci.to_numpy(),
                groups=o["psi_groups"], models=o["models"][["Model", "eci", "psi", "psi_eci", "psi_raw"]])


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = [(c, s) for s in SEEDS for c in CONFIGS]
    t0 = time.time()
    res = []
    print(f"{len(jobs)} fits, {workers} workers", flush=True)
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            res.append(r)
            el = time.time() - t0
            print(f"[{i:2d}/{len(jobs)}] {r['unit']:5s} kappa {r['kappa']:<5} seed {r['seed']} cost {r['cost']:.6f} "
                  f"sse {r['sse']:.4f} nfev {r['nfev']:4d} st {r['status']} {r['secs']:5.1f}s "
                  f"eta {el / i * (len(jobs) - i) / 60:4.1f}m", flush=True)

    keep = ["unit", "kappa", "seed", "cost", "sse", "nfev", "status", "optimality", "n_active", "rms_psi", "rms_f", "f1", "secs"]
    runs = pd.DataFrame([{k: r[k] for k in keep} for r in res]).sort_values(["unit", "kappa", "seed"])
    runs.to_csv(f"{OUT}_runs.csv", index=False)

    lines = ["# Scale-controlled fits (analysis/06_penalty.py)", "",
             f"Penalty kappa * (mean psi^2 + mean f^2), {'f = 0 for gap <= 0' if ZB else 'f(0) = 0 only'}, {len(SEEDS)} random starts per config. "
             "f is shown normalized to f(1) = 1 and psi rescaled to match (psi_eci = effect at a 1-year gap, ECI). "
             "Effects |psi f| are over the observed cells, in ECI points. 'optima' counts distinct final costs (gap > 1e-6).", "",
             "| unit | kappa | best SSE | best cost | nfev | optima | starts at best | rms psi = rms f | "
             + " | ".join(f"f({g})" for g in GRID) + " | eff p50 | p90 | p99 | max | max dECI (best-cost starts) | max dpsi_eci (best-cost starts) | max dECI (all starts) |",
             "|" + "---|" * 23]
    fcurves, models = [], []
    for (unit, kappa) in CONFIGS:
        rs = sorted([r for r in res if r["unit"] == unit and r["kappa"] == kappa], key=lambda r: r["cost"])
        best = rs[0]
        costs = np.array([r["cost"] for r in rs])
        distinct = int(np.sum(np.diff(costs) > 1e-6) + 1)
        at_best = [r for r in rs if r["cost"] - best["cost"] < 1e-6]
        d_eci_best = max(np.abs(r["eci"] - best["eci"]).max() for r in at_best)
        d_psi_best = max(np.abs(r["groups"].psi_eci.to_numpy() - best["groups"].psi_eci.to_numpy()).max() for r in at_best)
        d_eci_all = max(np.abs(r["eci"] - best["eci"]).max() for r in rs)
        fn = best["f_raw"] / best["f1"]
        q = np.quantile(best["eff"], [.5, .9, .99])
        lines.append(f"| {unit} | {kappa} | {best['sse']:.4f} | {best['cost']:.5f} | {best['nfev']} | {distinct} | {len(at_best)}/{len(rs)} | "
                     f"{best['rms_psi']:.3f} = {best['rms_f']:.3f} | " + " | ".join(f"{v:.2f}" for v in fn)
                     + f" | {q[0]:.1f} | {q[1]:.1f} | {q[2]:.1f} | {best['eff'].max():.1f} | {d_eci_best:.3g} | {d_psi_best:.3g} | {d_eci_all:.3g} |")
        for r in rs:
            fcurves.append(dict(unit=unit, kappa=kappa, seed=r["seed"], cost=r["cost"],
                                **{f"f_norm({g})": v for g, v in zip(GRID, r["f_raw"] / r["f1"])},
                                **{f"f_over_rms({g})": v for g, v in zip(GRID, r["f_raw"] / r["rms_f"])}))
        m = best["models"].copy() if unit == "model" else best["groups"].rename(columns={"group": "Model"}).copy()
        m.insert(0, "unit", unit); m.insert(1, "kappa", kappa)
        if unit == "model":
            m["n_cells"] = m.Model.map(N_CELLS)
        models.append(m)
    pd.DataFrame(fcurves).to_csv(f"{OUT}_f.csv", index=False)
    mm = pd.concat(models)
    mm.to_csv(f"{OUT}_models.csv", index=False)

    lines += ["", "Top and bottom 10 psi_eci at the best-cost start (per-model rows need at least 8 cells):", ""]
    for (unit, kappa) in CONFIGS:
        m = mm[(mm.unit == unit) & (mm.kappa == kappa)]
        if unit == "model":
            m = m[m.n_cells >= 8]
        s = m.sort_values("psi_eci", ascending=False)
        fmt = lambda d: ", ".join(f"{a} {b:+.1f}" for a, b in zip(d.Model, d.psi_eci))
        lines += [f"- {unit}, kappa {kappa}, top: {fmt(s.head(10))}", f"- {unit}, kappa {kappa}, bottom: {fmt(s.tail(10)[::-1])}"]
    open(f"{OUT}.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"total {time.time() - t0:.0f}s", flush=True)
