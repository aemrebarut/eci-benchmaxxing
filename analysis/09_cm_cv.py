"""Tune the rival penalties of the common-mechanism null model (no gap term) by 10-fold CV over cells.

lam_g penalizes the per-model date slopes gamma, lam_r the per-model log discriminations rho; "off" drops
the term. Same folds as analysis/07_cv_penalty.py. Held-out cells get weight 0; the metric is RMSE on
the fitted score (Epoch's clipped y_fit by default; pass y for the unclipped score)."""
import os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
import cm

Y = sys.argv[1] if len(sys.argv) > 1 else "y_fit"
OUT = f"out/cv09_{Y}.csv"
GAMMA = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, None]
RHO = [0.1, 0.3, 1.0, 3.0, 10.0, None]
df, ORG = cm.load()
FOLD = np.random.default_rng(7).integers(0, 10, len(df))


def run(job):
    lg, lr, k = job
    spec = cm.Spec(gap=False, gamma=lg is not None, rho=lr is not None, lam_g=lg or 0.0, lam_r=lr or 0.0)
    d = cm.Data(df, y_col=Y, org=ORG, w=(FOLD != k).astype(float))
    t = time.time()
    o = cm.fit(d, spec)
    test = FOLD == k
    p = cm.predict(o["x"], d, spec, o["lay"])
    return dict(lam_g=lg, lam_r=lr, fold=k, sse_test=float(((p - d.y)[test] ** 2).sum()), n_test=int(test.sum()),
                status=o["status"], iters=o["iters"], gn_dp=o["gn_dp"], gn_step=o["gn_step"], secs=time.time() - t)


if __name__ == "__main__":
    jobs = [(g, r, k) for g in GAMMA for r in RHO for k in range(10)]
    t0, res = time.time(), []
    with Pool(3) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs), 1):
            res.append(r)
            if i % 60 == 0:
                print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    cv = pd.DataFrame(res)
    cv.to_csv(OUT, index=False)
    cv["lam_g"], cv["lam_r"] = cv.lam_g.fillna(np.inf), cv.lam_r.fillna(np.inf)  # inf = term off
    s = cv.groupby(["lam_g", "lam_r"]).agg(sse=("sse_test", "sum"), n=("n_test", "sum"),
                                           worst_gn_dp=("gn_dp", "max"), not_conv=("status", lambda x: x.isin(["stalled", "max_iter"]).sum()))
    s["rmse"] = np.sqrt(s.sse / s.n)
    print(s.rmse.unstack().round(5).to_string())
    base = cv[(cv.lam_g == np.inf) & (cv.lam_r == np.inf)].set_index("fold").sse_test.sort_index()
    best = s.rmse.idxmin()
    bf = cv[(cv.lam_g == best[0]) & (cv.lam_r == best[1])].set_index("fold").sse_test.sort_index()
    print(f"best lam_g {best[0]} lam_r {best[1]} rmse {s.rmse[best]:.5f}; base (both off) {s.rmse[(np.inf, np.inf)]:.5f}; "
          f"best beats base on {(bf < base).sum()}/10 folds; worst certificate (change in a fitted value) {s.worst_gn_dp.max():.1e}; "
          f"not converged {int(s.not_conv.sum())}; total {time.time() - t0:.0f}s")
