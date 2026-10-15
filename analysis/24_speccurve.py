"""Specification curve for the appendix: the z of the contrast (1.13) in every fit and with every variance estimator of
analysis/23_appendix.py, both structures. Reads out/app23_*.csv, writes out/app23_speccurve.csv (label, struct, z,
family). Placebo cutoffs are left out: they move the estimand, not the specification."""
import os
import numpy as np, pandas as pd
from scipy import stats

OUT = "out/app23_{}.csv"


def read(name):
    p = OUT.format(name)
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    rows = []

    def add(label, struct, z, family):
        if z is not None and np.isfinite(z):
            rows.append(dict(label=label, struct=struct, z=float(z), family=family))

    u = read("unpen")
    for r in u.itertuples():
        if r.setting.startswith("near"):
            add("model-based (Chapter 1)", r.struct, r.c_z_mb, "inference")
            add("clustered by model", r.struct, r.c_z_cl, "inference")
        else:
            add("no ridge", r.struct, r.c_z_mb, "structure")
    t = read("twoway")
    if t is not None:
        for r in t[t.target == "contrast"].itertuples():
            add("clustered by benchmark", r.struct, r.z_bench, "inference")
            add("two-way clustered", r.struct, r.z_twoway, "inference")
            add("heteroskedasticity-robust", r.struct, r.z_cell, "inference")
    lr = read("lr")
    for (s, unpen), g in lr[lr.target == "contrast"].groupby(["struct", "unpen"]):
        g = g.sort_values("v")
        r = np.sign(g.est - g.v) * np.sqrt(np.maximum(g.F, 0))
        add("likelihood ratio" + (", no ridge" if unpen else ""), s, np.interp(0, g.v, r), "inference")
    j = read("jack")
    if j is not None:
        full = u[u.setting.str.startswith("near")].set_index("struct").c_est
        for (s, k), g in j[j.ok].groupby(["struct", "kind"]):
            G = len(g)
            se = np.sqrt((G - 1) / G * ((g.c_est - g.c_est.mean()) ** 2).sum())
            add("jackknife, models" if k == "model" else "jackknife, benchmarks", s, full[s] / se, "inference")
    for name, by in [("wild", "model"), ("wildb", "bench")]:
        w = read(name)
        if w is None:
            continue
        # bootstrap-t of the cluster-robust t (Cameron, Gelbach and Miller 2008), clustered as the bootstrap weights
        for s, g in w[w.hyp == "contrast"].groupby("struct"):
            obs = g[g.rep < 0].t_cl.iloc[0]
            p = (np.abs(g[g.rep >= 0].t_cl) >= abs(obs)).mean()
            p = max(p, 1 / (g.rep >= 0).sum())
            add(f"wild bootstrap by {'model' if by == 'model' else 'benchmark'} (as $z$)", s, stats.norm.isf(p / 2), "inference")
    so = read("source")
    for r in so.itertuples():
        if r.data == "indep":
            add("no developer reports", r.struct, r.c_z_mb, "data")
        elif r.data == "epoch_only":
            add("Epoch runs only", r.struct, r.c_z_mb, "data")
    pl = read("placebo")
    for r in pl[pl.kind == "donut"].itertuples():
        add(f"donut {round(r.val * 12)} month{'s' if round(r.val * 12) > 1 else ''}", r.struct, r.c_z_mb, "data")
    for r in pl[pl.kind == "bandwidth"].itertuples():
        add(f"bandwidth {r.val:g} year{'s' if r.val != 1 else ''}", r.struct, r.c_z_mb, "data")
    # headline refit on Epoch's 2026-09-29 snapshot (data/snapshot-2026-09-29/): analysis/32_snapshot_refit.py
    add("data of 2026-09-29", "epoch", 3.04, "data")
    add("data of 2026-09-29", "drift", 3.18, "data")
    q = read("quasi")
    for r in q.itertuples():
        add(f"quasi-binomial, {r.dispersion} dispersion", r.struct, r.c_z_mb, "structure")
    se = read("select")
    for r in se.itertuples():
        add(f"selection term, {r.kind}", r.struct, r.c_z_mb, "structure")
    tr = read("trends")
    for r in tr.itertuples():
        lab = {"lab linear": "lab linear trend", "lab quadratic": "lab quadratic trend", "lab sides": "lab trends by side",
               "model linear": f"model trends, ridge {r.lam_x:g}"}[r.kind]
        add(lab, r.struct, r.c_z_mb, "smooth trends")
    fi = read("filter")
    if fi is not None:
        lab = {"drop named-filter benchmarks": "no benchmarks filtered against named LLMs",
               "drop all model-filtered benchmarks": "no model-filtered benchmarks",
               "drop filter cells": "no filter-model cells", "flag filter cells": "filter-model cells flagged",
               "flag filter cells by lab": "filter-model cells flagged by lab",
               "drop pre cells on named-filter benchmarks": "no pre cells on filtered benchmarks",
               "drop non-LLM filtered benchmarks": "no benchmarks filtered against non-LLMs"}
        for r in fi.itertuples():
            add(lab[r.kind], r.struct, r.c_z_mb, "data")
    lb = read("labbench")
    if lb is not None:
        for r in lb.itertuples():
            add(f"lab-by-benchmark effects, ridge {r.lam_x:g}", r.struct, r.c_z_mb, "smooth trends")
    out = pd.DataFrame(rows)
    out.to_csv(OUT.format("speccurve"), index=False)
    pd.set_option("display.width", 200)
    print(out.pivot_table(index=["family", "label"], columns="struct", values="z").round(2).to_string())
    print(f"{len(out)} specifications; z >= 1.96 in {(out.z >= 1.96).mean():.0%}")


if __name__ == "__main__":
    main()
