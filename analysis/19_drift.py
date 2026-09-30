"""Organization pre/post model with benchmark-specific drift (analysis/18_bench_resid.py found residuals that trend
with model release date within benchmarks):

    y_ij ~ sigmoid(a_j (C_i + delta_org(i) post_ij + kappa_j z_ij - D_j)),  z_ij = t_ij - mean over benchmark j of t,

t_ij the cell's model release date in years (benchmark release + gap), so z_ij is the gap centred within the
benchmark and kappa_j is benchmark j's drift: ability per year of model release relative to the model's C_i. A kappa
common to all benchmarks is absorbed by C and D (except for the few Models whose cells have different dates), so
only differences between benchmarks matter and the ridge fixes the common part. With the drift in, delta is
identified by the step at each benchmark's release date against its linear trend (pooled over benchmarks) and by the
within-model contrast between benchmarks released before and after the model.

Ridge 0.5 lam sum delta^2 + 0.5 mu sum kappa^2, (lam, mu) chosen jointly by 10-fold CV over cells (the folds of
analysis/17_prepost.py; lam or mu = inf drops the term). Model code in analysis/prepost.py; SEs as in 17.
Usage: python analysis/19_drift.py [org | gsplit]; "gsplit" merges Google's organization names and splits them into
Gemini and Gemma (prepost.org_groups). Writes out/drift19<suffix>.md, _cv.csv, _groups.csv, _kappa.csv and
out/figs/drift19<suffix>_cv.png, _delta.png, _kappa.png, suffix "" for org and "_gsplit" for gsplit."""
import os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
import cm
from prepost import PrePost, wmean, wald_equal, org_groups

LAMS = [1e-3, 0.1, 1, 3, 10, 30, 100, 1e3, np.inf]
MUS = [1e-3, 1e-2, 0.1, 1, 3, 10, 30, 100, 1e3, np.inf]
LAM_FE = 1e-3
N_FOLDS, WORKERS = 10, 4
MODE = sys.argv[1] if len(sys.argv) > 1 else "org"
assert MODE in ("org", "gsplit"), MODE
SUF = "" if MODE == "org" else "_gsplit"
GOOG = ["Google DeepMind"] if MODE == "org" else ["Google DeepMind (Gemini)", "Google DeepMind (Gemma)"]
MAIN = ["OpenAI", "Anthropic"] + GOOG + ["Alibaba", "Meta AI", "DeepSeek", "xAI"]


def load(mode):
    df, ORG = cm.load()
    df["org"] = df.Model.map(ORG).fillna("unknown")
    df["group"] = org_groups(df, mode)
    df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    return df


def init(mode):
    global PP, XB, FOLDS
    df = load(mode)
    PP = PrePost(df, df.group, trend=df.z)
    XB = PP.fit(PP.ones, 0.0, PP.init_x(), use_delta=False).xf
    FOLDS = np.random.default_rng(7).integers(0, N_FOLDS, PP.n)


def args(lam, mu):
    return (lam if np.isfinite(lam) else 0.0), bool(np.isfinite(lam)), (mu if np.isfinite(mu) else None)


def run(combo):
    lam, mu = combo
    t = time.time()
    la, ud, m = args(lam, mu)
    xf = PP.fit(PP.ones, la, XB, ud, m).xf
    sse, st = [], []
    for f in range(N_FOLDS):
        w = (FOLDS != f).astype(float)
        r = PP.fit(w, la, xf, ud, m)
        te = FOLDS == f
        sse.append(float(((PP.predict(r.xf)[te] - PP.y[te]) ** 2).sum()))
        st.append(r.status)
    return lam, mu, xf, sse, min(st), time.time() - t


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    combos = [(lam, mu) for mu in MUS for lam in LAMS]
    with Pool(WORKERS, initializer=init, initargs=(MODE,)) as pool:
        out = []
        for k, o in enumerate(pool.imap_unordered(run, combos)):
            out.append(o)
            if k % 10 == 9:
                print(f"{k + 1} of {len(combos)} combinations, {time.time() - t0:.0f}s", flush=True)
    init(MODE)
    full = {(lam, mu): xf for lam, mu, xf, _, _, _ in out}
    cv = pd.DataFrame([dict(lam=lam, mu=mu, fold=f, sse=s) for lam, mu, _, sse, _, _ in out for f, s in enumerate(sse)])
    cv.to_csv(f"out/drift19{SUF}_cv.csv", index=False)
    bad = [(lam, mu, st) for lam, mu, _, _, st, _ in out if st <= 0]
    tab = cv.groupby(["mu", "lam"]).sse.sum().unstack("lam")
    ref = tab.at[np.inf, np.inf]
    fold_ref = cv[(cv.lam == np.inf) & (cv.mu == np.inf)].set_index("fold").sse
    better = cv.set_index("fold").groupby(["mu", "lam"]).sse.apply(lambda s: int((s < fold_ref).sum())).unstack("lam")
    mu_star, lam_star = tab.stack().idxmin()
    lam_by_mu = tab.drop(columns=np.inf).idxmin(axis=1)
    c17 = pd.read_csv("out/prepost17_org_cv.csv").groupby("lam").sse.sum()
    check17 = float(np.abs(tab.loc[np.inf, LAMS].to_numpy() - c17.reindex(LAMS).to_numpy()).max()) if MODE == "org" else np.nan
    print(f"CV choice lam {lam_star:g}, mu {mu_star:g}; no-drift row vs 17: max |CV SSE diff| {check17:.2e}; "
          f"non-converged: {bad or 'none'}", flush=True)

    # Deltas along mu, near-unpenalized and at the CV lam for that mu
    rows = []
    res = {}
    for mu in MUS:
        for tag, lam in [("near-unpenalized", LAM_FE), ("CV lam given mu", lam_by_mu[mu])]:
            la, ud, m = args(lam, mu)
            d, Vc, Vm, b, edf = PP.analyse(full[(lam, mu)], la, m)
            res[(tag, mu)] = (lam, d, Vc, Vm, b, edf)
            for k, g in enumerate(PP.ident):
                rows.append(dict(mu=mu, fit=tag, lam=lam, org=g, delta=d[k], se_cl=np.sqrt(Vc[k, k]), se_mb=np.sqrt(Vm[k, k])))
    gt = pd.DataFrame(rows)
    gt["se"] = np.maximum(gt.se_cl, gt.se_mb)
    gt.round(4).to_csv(f"out/drift19{SUF}_groups.csv", index=False)

    # kappa at the CV choice (drift in ECI per year), with the drift of the no-drift residuals from 18 for comparison
    la, ud, m = args(lam_star, mu_star)
    xs = full[(lam_star, mu_star)]
    kap = None
    if m is not None:
        idx, Vc, Vm, b, _ = PP.cov(xs, la, m)
        ks = slice(len(idx) - PP.Jk, len(idx))
        brel = PP.df.groupby("benchmark").benchmark_release_date.first()
        kap = pd.DataFrame(dict(benchmark=PP.benches, released=brel.reindex(PP.benches).to_numpy(),
                                n=PP.df.benchmark.value_counts().reindex(PP.benches).to_numpy(),
                                z_sd=PP.df.groupby("benchmark").z.std().reindex(PP.benches).to_numpy(),
                                kappa_eci=b * xs[PP.n_ep + PP.K:], se_cl=b * np.sqrt(np.diag(Vc)[ks]), se_mb=b * np.sqrt(np.diag(Vm)[ks])))
        kap["se"] = np.maximum(kap.se_cl, kap.se_mb)
        b18 = pd.read_csv("out/bench18_bench.csv").set_index("benchmark")
        kap["drift18_score"] = kap.benchmark.map(b18.drift_slope)
        kap = kap.sort_values("released").reset_index(drop=True)
        kap.round(4).to_csv(f"out/drift19{SUF}_kappa.csv", index=False)

    # Residual drift after the fit (same regression as 18, raw)
    x = PP.df.copy()
    x["r"] = PP.y - PP.predict(xs)
    x0 = full[(lam_by_mu[np.inf], np.inf)]
    x["r0"] = PP.y - PP.predict(x0)
    def drift_z(col):
        zs = []
        for _, g in x.groupby("benchmark"):
            if len(g) < 12 or g.gap.std() < 0.1:
                continue
            X = np.column_stack([np.ones(len(g)), g.gap])
            beta = np.linalg.lstsq(X, g[col].to_numpy(), rcond=None)[0]
            e = g[col].to_numpy() - X @ beta
            XtXi = np.linalg.inv(X.T @ X)
            zs.append(beta[1] / np.sqrt((XtXi @ (X.T * e ** 2) @ X @ XtXi)[1, 1]))
        return np.array(zs)
    z0, z1 = drift_z("r0"), drift_z("r")

    # Report
    L = [f"# Organization pre/post model with benchmark drift (analysis/19_drift.py {MODE})", "",
         f"{PP.K} organizations with a delta, {PP.B} benchmarks with a drift kappa_j, {PP.n} cells. z_ij = gap centred "
         f"within the benchmark (sd over cells {PP.df.z.std():.2f} years).", "",
         (f"No-drift row (mu = inf) against analysis/17_prepost.py org: max |CV SSE difference| {check17:.1e}. " if MODE == "org" else
          f"Groups: organizations with Google's names merged and split by family ({', '.join(GOOG)}). ") +
         f"Fits with a non-positive solver status: {bad or 'none'}.", "",
         "## Joint CV over (lam, mu): CV SSE minus the no-delta, no-drift model (Epoch's), folds better of 10", "",
         "| mu \\ lam | " + " | ".join("none" if not np.isfinite(l) else f"{l:g}" for l in LAMS) + " |",
         "|---" * (len(LAMS) + 1) + "|"]
    for mu in MUS:
        L.append(f"| {'none' if not np.isfinite(mu) else f'{mu:g}'} | "
                 + " | ".join(f"{tab.at[mu, l] - ref:+.3f} ({better.at[mu, l]})" for l in LAMS) + " |")
    L += ["", f"CV choice: lam = {lam_star:g}, mu = {mu_star:g} (CV SSE {tab.at[mu_star, lam_star] - ref:+.3f} against Epoch's model, "
          f"{tab.at[mu_star, lam_star] - tab.at[mu_star, np.inf]:+.3f} against the same drift without delta, "
          f"{tab.at[mu_star, lam_star] - tab.loc[np.inf].min():+.3f} against the best no-drift fit).", "",
          f"Residual drift (slope of the residual on the gap within benchmark, |z| >= 2 of {len(z1)}): "
          f"{int((np.abs(z0) >= 2).sum())} without drift terms (lam {lam_by_mu[np.inf]:g}), {int((np.abs(z1) >= 2).sum())} at the CV choice.", ""]
    for tag in ["near-unpenalized", "CV lam given mu"]:
        for mu in [np.inf, mu_star] if np.isfinite(mu_star) else [np.inf]:
            lam, d, Vc, Vm, b, edf = res[(tag, mu)]
            L.append(f"### {tag}: lam = {lam:g}, mu = {'none' if not np.isfinite(mu) else f'{mu:g}'} (b = {b:.2f}, edf {edf:.1f})")
            L.append("")
            for Vname, V in [("clustered", Vc), ("model-based", Vm)]:
                mm, ss = wmean(d, V, np.arange(PP.K))
                W, k = wald_equal(d, V, np.arange(PP.K))
                L.append(f"- SE {Vname}: weighted mean delta {mm:+.2f} +- {ss:.2f} (z {mm / ss:+.2f}); all equal: Wald {W:.1f} on {k} df, p {stats.chi2.sf(W, k):.3f}.")
            L.append("")
    L += ["## Main organizations: delta (ECI) +- SE (larger of clustered and model-based) by mu", "",
          "| organization | fit | " + " | ".join("none" if not np.isfinite(m) else f"mu {m:g}" for m in MUS) + " |",
          "|---|---" + "|---" * len(MUS) + "|"]
    for o in MAIN:
        for tag in ["near-unpenalized", "CV lam given mu"]:
            s = gt[(gt.org == o) & (gt.fit == tag)].set_index("mu")
            L.append(f"| {o} | {tag} | " + " | ".join(f"{s.at[m, 'delta']:+.2f} +- {s.at[m, 'se']:.2f}" for m in MUS) + " |")
    bs = {mu: res[("near-unpenalized", mu)][4] for mu in MUS}
    L += ["", "z = delta / SE, near-unpenalized fits (scale-free: b, the ECI points per raw unit, changes with mu because the drift "
          "moves the anchor models' abilities; b by mu: " + ", ".join(f"{'none' if not np.isfinite(m) else f'{m:g}'} {b:.1f}" for m, b in bs.items()) + ")", "",
          "| organization | " + " | ".join("none" if not np.isfinite(m) else f"mu {m:g}" for m in MUS) + " |", "|---" * (len(MUS) + 1) + "|"]
    for o in MAIN:
        s_ = gt[(gt.org == o) & (gt.fit == "near-unpenalized")].set_index("mu")
        L.append(f"| {o} | " + " | ".join(f"{s_.at[m, 'delta'] / s_.at[m, 'se']:+.2f}" for m in MUS) + " |")
    if kap is not None:
        L += ["", f"Mean kappa at the CV choice {kap.kappa_eci.mean():+.2f} ECI per year (cell-weighted {np.average(kap.kappa_eci, weights=kap.n):+.2f}): "
              "the common drift, which C and D could absorb, is near 0."]
        top = kap.reindex(kap.kappa_eci.abs().sort_values(ascending=False).index).head(12)
        L += ["", f"## Largest drifts at the CV choice (ECI per year of model release; SE larger of the two)", "",
              "| benchmark | released | n | sd of z | kappa | SE | drift of no-drift residual in 18 (score per year) |", "|---|---|---|---|---|---|---|"]
        L += [f"| {r.benchmark} | {r.released:%Y-%m} | {r.n} | {r.z_sd:.2f} | {r.kappa_eci:+.2f} | {r.se:.2f} | {r.drift18_score:+.3f} |".replace("+nan", "")
              for r in top.itertuples()]
        L += ["", f"Correlation of kappa with the score-scale drift from 18 over {kap.drift18_score.notna().sum()} benchmarks: "
              f"{kap[['kappa_eci', 'drift18_score']].corr().iat[0, 1]:+.2f}."]
    open(f"out/drift19{SUF}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)

    # Figures
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    A = (tab.loc[MUS, LAMS] - ref).to_numpy()
    lim = np.abs(A).max()
    im = ax.imshow(A, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(MUS)):
        for j in range(len(LAMS)):
            ax.text(j, i, f"{A[i, j]:+.2f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if (MUS[i], LAMS[j]) == (mu_star, lam_star) else "normal")
    ax.set_xticks(range(len(LAMS)), ["none" if not np.isfinite(l) else f"{l:g}" for l in LAMS])
    ax.set_yticks(range(len(MUS)), ["none" if not np.isfinite(m) else f"{m:g}" for m in MUS])
    ax.set_xlabel("lam (ridge on the organization deltas)")
    ax.set_ylabel("mu (ridge on the benchmark drifts)")
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("CV SSE minus Epoch's model")
    ax.set_title(f"10-fold CV; bold: choice (lam {lam_star:g}, mu {mu_star:g})", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"out/figs/drift19{SUF}_cv.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(MAIN), figsize=(2.5 * len(MAIN), 3.6), sharey=True)
    xp = np.arange(len(MUS))
    for ax, o in zip(axes, MAIN):
        for tag, off, mk, c in [("near-unpenalized", -0.12, "o", "0.45"), ("CV lam given mu", 0.12, "s", "#C44E52")]:
            s = gt[(gt.org == o) & (gt.fit == tag)].set_index("mu").reindex(MUS)
            ax.errorbar(xp + off, s.delta, yerr=2 * s.se, fmt=mk, ms=3.5, color=c, lw=1, label=tag)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axvline(len(MUS) - 1.5, color="0.7", lw=0.8, ls=":")
        ax.set_xticks(xp, ["none" if not np.isfinite(m) else f"{m:g}" for m in MUS], rotation=90, fontsize=7)
        ax.set_title(o, fontsize=9)
        ax.set_xlabel("mu (drift ridge)", fontsize=8)
    axes[0].set_ylabel("delta (ECI), +-2 SE")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Organization deltas as the benchmark drift is freed (small mu = free drift; none = no drift term). "
                 f"CV choice mu = {mu_star:g}.\nECI units of each fit: b = {bs[np.inf]:.1f} ECI per raw unit without drift, {bs[mu_star]:.1f} at mu = {mu_star:g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"out/figs/drift19{SUF}_delta.png", dpi=130)
    plt.close(fig)

    if kap is not None:
        fig, ax = plt.subplots(figsize=(7.5, 0.19 * len(kap) + 1.6))
        yy = np.arange(len(kap))[::-1]
        ax.errorbar(kap.kappa_eci, yy, xerr=2 * kap.se, fmt="o", ms=3.5, color="#4C72B0", lw=1)
        ax.axvline(0, color="0.4", lw=0.8)
        ax.set_yticks(yy, [f"{r.benchmark[:30]} ({r.released:%y-%m}, n {r.n})" for r in kap.itertuples()], fontsize=6.5)
        ax.set_xlabel("kappa_j: ability drift, ECI per year of model release, +-2 SE")
        ax.set_title(f"Benchmark drift at the CV choice (lam {lam_star:g}, mu {mu_star:g}), benchmarks in release order", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"out/figs/drift19{SUF}_kappa.png", dpi=130)
        plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s")
