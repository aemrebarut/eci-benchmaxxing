"""One step shared by every model: Epoch's model plus a single beta on the cells where the benchmark was public when
the model was released,

    y_ij ~ sigmoid(a_j (C_i + beta post_ij [+ kappa_j z_ij] - D_j)),

identified through the models with cells on both sides of their release. The first rung of the ladder in the blog
Part 1 (blogpost/part1.md). Near-unpenalized (lam 1e-3), without and with the benchmark drift (mu 0.1, as in
analysis/19_drift.py), on the 2026-10-15 snapshot; model-based and clustered SEs as in analysis/17_prepost.py. The
held-out change against Epoch's model uses the 10 folds of 17 and 19 (seed 7).
Usage: python analysis/30_common_step.py. Writes out/p1_common_step.csv (estimates) and out/p1_common_step_cv.csv."""
import os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
import cm
from prepost import PrePost

LAM_FE, MU = 1e-3, 0.1
N_FOLDS, WORKERS = 10, 4
SPECS = [("Epoch", False, None), ("common step", True, None), ("drift", False, MU), ("drift and common step", True, MU)]


def load():
    df, ORG = cm.load()
    df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    return df


def init():
    global PP, XB, FOLDS
    df = load()
    PP = PrePost(df, np.full(len(df), "all"), trend=df.z)
    XB = PP.fit(PP.ones, 0.0, PP.init_x(), use_delta=False).xf
    FOLDS = np.random.default_rng(7).integers(0, N_FOLDS, PP.n)


def run(spec):
    name, ud, mu = spec
    xf = PP.fit(PP.ones, LAM_FE, XB, ud, mu).xf
    out = []
    for f in range(N_FOLDS):
        w = (FOLDS != f).astype(float)
        r = PP.fit(w, LAM_FE, xf, ud, mu)
        te = FOLDS == f
        out.append(dict(spec=name, fold=f, sse=float(((PP.predict(r.xf)[te] - PP.y[te]) ** 2).sum()), status=r.status))
    return out


if __name__ == "__main__":
    t0 = time.time()
    init()
    rows = []
    for name, mu in [("no drift", None), ("drift, mu 0.1", MU)]:
        pp = PrePost(PP.df, np.full(PP.n, "all"), trend=PP.df.z if mu else None)
        _, res = pp.fit_full(LAM_FE, mu=mu)
        d, Vc, Vm, b, edf = pp.analyse(res.xf, LAM_FE, mu)
        se_cl, se_mb = float(np.sqrt(Vc[0, 0])), float(np.sqrt(Vm[0, 0]))
        se = max(se_cl, se_mb)
        both = int(pp.both.sum())
        rows.append(dict(structure=name, beta=float(d[0]), se_cl=se_cl, se_mb=se_mb, se=se, z=float(d[0]) / se,
                         p=2 * stats.norm.sf(abs(float(d[0]) / se)), b=b, models_both_sides=both, status=res.status, nfev=res.nfev))
        print(f"{name:14s} status {res.status} nfev {res.nfev}: common beta {d[0]:+.3f} ECI points, SE clustered "
              f"{se_cl:.3f}, model-based {se_mb:.3f}, z {d[0] / se:.2f} (larger SE); b {b:.3f}; models with cells on "
              f"both sides {both}", flush=True)
    pd.DataFrame(rows).round(4).to_csv("out/p1_common_step.csv", index=False)

    with Pool(WORKERS, initializer=init) as pool:
        cv = pd.DataFrame([r for o in pool.map(run, SPECS) for r in o])
    cv.to_csv("out/p1_common_step_cv.csv", index=False)
    tot = cv.groupby("spec").sse.sum()
    ref = cv[cv.spec == "Epoch"].set_index("fold").sse
    print(f"\n10-fold CV over cells (folds of 17 and 19), held-out SSE; Epoch's model {tot['Epoch']:.3f}")
    for name, _, _ in SPECS[1:]:
        s = cv[cv.spec == name].set_index("fold").sse
        print(f"  {name:22s} {tot[name]:.3f} ({tot[name] - tot['Epoch']:+.3f} against Epoch's, better in {(s < ref).sum()} of 10 folds)")
    s1 = cv[cv.spec == "drift and common step"].set_index("fold").sse
    s0 = cv[cv.spec == "drift"].set_index("fold").sse
    print(f"  common step given drift: {s1.sum() - s0.sum():+.3f}, better in {(s1 < s0).sum()} of 10 folds")
    print(f"non-positive solver status: {sorted(set(cv.spec[cv.status <= 0])) or 'none'}")
    print(f"done in {time.time() - t0:.0f}s")
