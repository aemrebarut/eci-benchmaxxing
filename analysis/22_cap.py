"""Benchmark ceilings as the alternative to benchmark drift. The benchmarks with the strongest downward drift in
analysis/18_bench_resid.py (Winogrande, HellaSwag, LAMBADA, TriviaQA, ARC AI2) are old ones whose scores stop
rising well below 1, which a sigmoid that tends to 1 reads as new models doing worse than their ability predicts.
The cap model gives each benchmark a ceiling,

    y_ij ~ (1 - c_j) sigmoid(a_j (C_i + delta_org(i) post_ij [+ kappa_j z_ij] - D_j)),  c_j in [0, 0.9],

with the ridge 0.5 nu sum c^2 (on a benchmark whose scores all sit low, c trades off with D, so the ridge decides).
(lam, mu, nu) are chosen jointly by 10-fold CV over cells (the folds of analysis/19_drift.py; inf drops the term),
so the CV compares Epoch's model, cap, drift and both, each with and without the organization deltas.
Usage: python analysis/22_cap.py [org | gmerge | gsplit]. Writes out/cap22_<mode>.md, _cv.csv, _groups.csv,
_caps.csv and out/figs/cap22_<mode>_curves.png, _delta.png."""
import os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
import cm
from prepost import PrePost, wmean, wald_equal, org_groups

LAMS = [1e-3, 3, 10, 30, 100, np.inf]
MUS = [1e-2, 0.1, 1, 10, np.inf]
NUS = [1e-2, 0.1, 1, 10, 100, np.inf]
LAM_FE = 1e-3
N_FOLDS, WORKERS = 10, 4
MODE = sys.argv[1] if len(sys.argv) > 1 else "org"
assert MODE in ("org", "gmerge", "gsplit"), MODE
GOOG = {"org": ["Google DeepMind"], "gmerge": ["Google DeepMind (all)"],
        "gsplit": ["Google DeepMind (Gemini)", "Google DeepMind (Gemma)"]}[MODE]
MAIN = ["OpenAI", "Anthropic"] + GOOG + ["Alibaba", "Meta AI", "DeepSeek", "xAI"]
SHOW = ["Winogrande", "HellaSwag", "LAMBADA", "TriviaQA", "ARC AI2", "GSM8K", "APEX-Agents", "DeepSWE"]


def load(mode):
    df, ORG = cm.load()
    df["org"] = df.Model.map(ORG).fillna("unknown")
    df["group"] = org_groups(df, mode)
    df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    return df


def init(mode):
    global PP, XB, FOLDS
    df = load(mode)
    PP = PrePost(df, df.group, trend=df.z, cap=True)
    XB = PP.fit(PP.ones, 0.0, PP.init_x(), use_delta=False).xf
    FOLDS = np.random.default_rng(7).integers(0, N_FOLDS, PP.n)


def args(lam, mu, nu):
    f = lambda v: v if np.isfinite(v) else None
    return (lam if np.isfinite(lam) else 0.0), bool(np.isfinite(lam)), f(mu), f(nu)


def run(combo):
    t = time.time()
    la, ud, m, n_ = args(*combo)
    xf = PP.fit(PP.ones, la, XB, ud, m, n_).xf
    sse, st = [], []
    for f in range(N_FOLDS):
        w = (FOLDS != f).astype(float)
        r = PP.fit(w, la, xf, ud, m, n_)
        te = FOLDS == f
        sse.append(float(((PP.predict(r.xf)[te] - PP.y[te]) ** 2).sum()))
        st.append(r.status)
    return combo, xf, sse, min(st), time.time() - t


def drift_z(x, col):
    """Slope of the residual on the gap within each benchmark (as in analysis/18_bench_resid.py), HC0 z."""
    zs = {}
    for b, g in x.groupby("benchmark"):
        if len(g) < 12 or g.gap.std() < 0.1:
            continue
        X = np.column_stack([np.ones(len(g)), g.gap])
        beta = np.linalg.lstsq(X, g[col].to_numpy(), rcond=None)[0]
        e = g[col].to_numpy() - X @ beta
        XtXi = np.linalg.inv(X.T @ X)
        zs[b] = beta[1] / np.sqrt((XtXi @ (X.T * e ** 2) @ X @ XtXi)[1, 1])
    return pd.Series(zs)


def lab(v):
    return "none" if not np.isfinite(v) else f"{v:g}"


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = time.time()
    combos = [(lam, mu, nu) for nu in NUS for mu in MUS for lam in LAMS]
    with Pool(WORKERS, initializer=init, initargs=(MODE,)) as pool:
        out = []
        for k, o in enumerate(pool.imap_unordered(run, combos)):
            out.append(o)
            if k % 10 == 9:
                print(f"{k + 1} of {len(combos)} combinations, {time.time() - t0:.0f}s", flush=True)
    init(MODE)
    full = {c: xf for c, xf, _, _, _ in out}
    cv = pd.DataFrame([dict(lam=c[0], mu=c[1], nu=c[2], fold=f, sse=s) for c, _, sse, _, _ in out for f, s in enumerate(sse)])
    cv.to_csv(f"out/cap22_{MODE}_cv.csv", index=False)
    bad = [c for c, _, _, st, _ in out if st <= 0]
    tot = cv.groupby(["lam", "mu", "nu"]).sse.sum()
    ref = tot[(np.inf, np.inf, np.inf)]
    fold_ref = cv[(cv.lam == np.inf) & (cv.mu == np.inf) & (cv.nu == np.inf)].set_index("fold").sse

    # The four structures (Epoch's, cap, drift, both), each with its CV choice of the other ridges
    def best(mu_ok, nu_ok, lam_ok=lambda l: True):
        s = tot[[mu_ok(m) and nu_ok(n) and lam_ok(l) for l, m, n in tot.index]]
        return s.idxmin(), s.min()
    fin, inf_ = np.isfinite, lambda v: not np.isfinite(v)
    STRUCT = {"Epoch": (inf_, inf_), "cap": (inf_, fin), "drift": (fin, inf_), "cap + drift": (fin, fin)}
    choice = {}
    for s, (mo, no) in STRUCT.items():
        (l, m, n), v = best(mo, no)
        (l0, m0, n0), v0 = best(mo, no, inf_)
        choice[s] = dict(lam=l, mu=m, nu=n, sse=v, lam_none=(l0, m0, n0), sse_none=v0)
    print(f"non-converged: {bad or 'none'}; choices: " + "; ".join(f"{s} lam {lab(c['lam'])} mu {lab(c['mu'])} nu {lab(c['nu'])}"
                                                            for s, c in choice.items()), flush=True)

    # Organization deltas under each structure, near-unpenalized and at the CV lam
    rows, res = [], {}
    for s, c in choice.items():
        for tag, lam in [("near-unpenalized", LAM_FE), ("CV lam", c["lam"])]:
            if not np.isfinite(lam):
                continue
            key = (lam, c["mu"], c["nu"])
            la, ud, m, n_ = args(*key)
            d, Vc, Vm, b, edf = PP.analyse(full[key], la, m, n_)
            res[(s, tag)] = (key, d, Vc, Vm, b, edf)
            for k, g in enumerate(PP.ident):
                rows.append(dict(structure=s, fit=tag, lam=lam, mu=c["mu"], nu=c["nu"], org=g, delta=d[k],
                                 se_cl=np.sqrt(Vc[k, k]), se_mb=np.sqrt(Vm[k, k])))
    gt = pd.DataFrame(rows)
    gt["se"] = np.maximum(gt.se_cl, gt.se_mb)
    gt["z"] = gt.delta / gt.se
    gt.round(4).to_csv(f"out/cap22_{MODE}_groups.csv", index=False)

    # Caps at the CV choices of "cap" and "cap + drift", with the observed maximum score
    x = PP.df.copy()
    mx = x.groupby("benchmark").performance.max()
    caps = pd.DataFrame(dict(benchmark=PP.benches, released=x.groupby("benchmark").benchmark_release_date.first().reindex(PP.benches).to_numpy(),
                             n=x.benchmark.value_counts().reindex(PP.benches).to_numpy(), max_score=mx.reindex(PP.benches).to_numpy()))
    for s in ["cap", "cap + drift"]:
        c = choice[s]
        caps[f"ceiling|{s}"] = PP.ceiling(full[(c["lam"], c["mu"], c["nu"])])
    caps = caps.sort_values("ceiling|cap").reset_index(drop=True)
    caps.round(4).to_csv(f"out/cap22_{MODE}_caps.csv", index=False)

    # Residual drift left after each structure's no-delta fit at its CV ridges
    rz = {}
    for s, c in choice.items():
        x[f"r|{s}"] = PP.y - PP.predict(full[c["lam_none"]])
        rz[s] = drift_z(x, f"r|{s}")
    rz = pd.DataFrame(rz)

    # Kappas with and without the cap (ECI per year, each fit's own b)
    kap = {}
    for s in ["drift", "cap + drift"]:
        key = choice[s]["lam_none"]
        _, b = PP.scale(full[key])
        kap[s] = pd.Series(b * full[key][PP.n_ep + PP.K:PP.n_ep + PP.K + PP.Jk], index=PP.benches)
    kap = pd.DataFrame(kap)

    # Report
    L = [f"# Benchmark ceilings against benchmark drift (analysis/22_cap.py {MODE})", "",
         f"{PP.K} organizations with a delta, {PP.B} benchmarks, {PP.n} cells. Ceiling 1 - c_j with c_j in [0, 0.9] and "
         f"ridge nu; drift kappa_j z_ij with ridge mu; organization deltas with ridge lam; all chosen by 10-fold CV "
         f"(inf / none drops the term). Fits with a non-positive solver status: {bad or 'none'}.", "",
         "## CV by structure (CV SSE minus Epoch's model; folds better than Epoch's of 10)", "",
         "| structure | no delta: ridges | CV SSE | with delta: ridges | CV SSE | delta's own gain |", "|---|---|---|---|---|---|"]
    for s, c in choice.items():
        l0, m0, n0 = c["lam_none"]
        better = lambda l, m, n: int((cv[(cv.lam == l) & (cv.mu == m) & (cv.nu == n)].set_index("fold").sse < fold_ref).sum())
        L.append(f"| {s} | mu {lab(m0)}, nu {lab(n0)} | {c['sse_none'] - ref:+.3f} ({better(l0, m0, n0)}) | "
                 f"lam {lab(c['lam'])}, mu {lab(c['mu'])}, nu {lab(c['nu'])} | {c['sse'] - ref:+.3f} ({better(c['lam'], c['mu'], c['nu'])}) | "
                 f"{c['sse'] - c['sse_none']:+.4f} |")
    L += ["", f"Epoch's model: CV SSE {ref:.3f}.", "",
          "### CV SSE minus Epoch's, no delta (rows nu, columns mu)", "",
          "| nu \\ mu | " + " | ".join(lab(m) for m in MUS) + " |", "|---" * (len(MUS) + 1) + "|"]
    for n in NUS:
        L.append(f"| {lab(n)} | " + " | ".join(f"{tot[(np.inf, m, n)] - ref:+.3f}" for m in MUS) + " |")
    L += ["", "### CV SSE minus Epoch's, best lam for each (mu, nu)", "",
          "| nu \\ mu | " + " | ".join(lab(m) for m in MUS) + " |", "|---" * (len(MUS) + 1) + "|"]
    for n in NUS:
        L.append(f"| {lab(n)} | " + " | ".join(f"{min(tot[(l, m, n)] for l in LAMS) - ref:+.3f}" for m in MUS) + " |")

    L += ["", f"## Residual drift left (slope of the residual on the gap within benchmark, |z| >= 2 of {len(rz)}; no-delta fits)", "",
          "| structure | benchmarks with abs z >= 2 | which |", "|---|---|---|"]
    for s in rz.columns:
        hit = rz[s][rz[s].abs() >= 2].sort_values()
        L.append(f"| {s} | {len(hit)} | " + ", ".join(f"{b} {v:+.1f}" for b, v in hit.items()) + " |")
    L += ["", "z by benchmark for the benchmarks flagged by any structure:", "",
          "| benchmark | " + " | ".join(rz.columns) + " |", "|---" * (len(rz.columns) + 1) + "|"]
    flag = rz.index[(rz.abs() >= 2).any(axis=1)]
    for b in rz.loc[flag].sort_values("Epoch").index:
        L.append(f"| {b} | " + " | ".join(f"{rz.at[b, s]:+.1f}" for s in rz.columns) + " |")

    L += ["", "## Organization deltas by structure (ECI points of each fit; SE larger of clustered and model-based)", ""]
    for tag in ["near-unpenalized", "CV lam"]:
        L += [f"### {tag}", "", "| organization | " + " | ".join(s for s in choice if (s, tag) in res) + " |",
              "|---" * (1 + sum((s, tag) in res for s in choice)) + "|"]
        for o in MAIN:
            cells = []
            for s in choice:
                if (s, tag) not in res:
                    continue
                r = gt[(gt.structure == s) & (gt.fit == tag) & (gt.org == o)].iloc[0]
                cells.append(f"{r.delta:+.2f} +- {r.se:.2f} (z {r.z:+.2f})")
            L.append(f"| {o} | " + " | ".join(cells) + " |")
        L.append("")
        for s in choice:
            if (s, tag) not in res:
                continue
            key, d, Vc, Vm, b, edf = res[(s, tag)]
            zz = gt[(gt.structure == s) & (gt.fit == tag)].set_index("org").z
            W, k = wald_equal(d, Vc, np.arange(PP.K))
            W2, _ = wald_equal(d, Vm, np.arange(PP.K))
            others = zz.drop(GOOG).abs()
            L.append(f"- {s} (lam {lab(key[0])}, mu {lab(key[1])}, nu {lab(key[2])}, b {b:.2f}): max abs z of the other "
                     f"{len(others)} organizations {others.max():.2f} ({others.idxmax()}); all equal: p {stats.chi2.sf(W, k):.3f} "
                     f"clustered, {stats.chi2.sf(W2, k):.3f} model-based.")
        L.append("")

    L += ["## Ceilings (1 - c_j), lowest 20 at the cap choice, with the highest observed score", "",
          "| benchmark | released | n | max score | ceiling, cap | ceiling, cap + drift |", "|---|---|---|---|---|---|"]
    for r in caps.head(20).rename(columns={"ceiling|cap": "u_cap", "ceiling|cap + drift": "u_both"}).itertuples():
        L.append(f"| {r.benchmark} | {r.released:%Y-%m} | {r.n} | {r.max_score:.2f} | {r.u_cap:.3f} | {r.u_both:.3f} |")
    L += ["", f"Benchmarks with a ceiling below 0.99: {int((caps['ceiling|cap'] < 0.99).sum())} (cap), "
          f"{int((caps['ceiling|cap + drift'] < 0.99).sum())} (cap + drift)."]
    L += ["", "## Drift with and without the cap (kappa, ECI per year of model release, no-delta fits)", "",
          "| benchmark | drift | cap + drift |", "|---|---|---|"]
    for b in kap["drift"].abs().sort_values(ascending=False).index[:12]:
        L.append(f"| {b} | {kap.at[b, 'drift']:+.2f} | {kap.at[b, 'cap + drift']:+.2f} |")
    open(f"out/cap22_{MODE}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)

    # Figure: scores against ability on the drifting benchmarks, Epoch's curve and the capped curve
    xE, xC = full[choice["Epoch"]["lam_none"]], full[choice["cap"]["lam_none"]]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharey=True)
    mdate = x.bdate + x.gap
    for ax, bname in zip(axes.flat, SHOW):
        j = PP.benches.index(bname)
        sel = (PP.bi == j)
        for xx, col, name in [(xE, "0.35", "Epoch's model"), (xC, "#C44E52", "with ceiling")]:
            C, D, a, _, _ = PP.unpack(xx)
            u = PP.ceiling(xx)[j]
            a0, b = PP.scale(xx)
            grid = np.linspace(C.min() - 0.5, C.max() + 0.5, 300)
            ax.plot(a0 + b * grid, u / (1 + np.exp(-a[j] * (grid - D[j]))), color=col, lw=1.6,
                    label=f"{name}" + (f" (ceiling {u:.2f})" if col != "0.35" else ""))
        C, D, a, _, _ = PP.unpack(xC)
        a0, b = PP.scale(xC)
        sc = ax.scatter(a0 + b * C[PP.mi[sel]], PP.y[sel], c=mdate[sel], cmap="viridis", s=14, vmin=2019, vmax=2026.8, zorder=3)
        ax.set_title(f"{bname} ({x.benchmark_release_date[sel].iloc[0]:%Y})", fontsize=10)
        ax.set_xlim(a0 + b * C[PP.mi[sel]].min() - 8, a0 + b * C[PP.mi[sel]].max() + 8)
        ax.axhline(1, color="0.8", lw=0.8)
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("model ability (ECI, fit with ceilings)")
    for ax in axes[:, 0]:
        ax.set_ylabel("score (Epoch's normalization)")
    fig.colorbar(sc, ax=axes, fraction=0.02).set_label("model release date")
    fig.suptitle("Scores against ability on the benchmarks that drift most: Epoch's curve (grey, its own ECI scale) and the "
                 f"curve with a ceiling (red, nu {lab(choice['cap']['nu'])})", fontsize=11)
    fig.savefig(f"out/figs/cap22_{MODE}_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Figure: organization deltas under the four structures
    fig, axes = plt.subplots(1, len(MAIN), figsize=(2.4 * len(MAIN), 3.6), sharey=True)
    S = list(choice)
    for ax, o in zip(axes, MAIN):
        for tag, off, mk, c in [("near-unpenalized", -0.12, "o", "0.45"), ("CV lam", 0.12, "s", "#C44E52")]:
            s = gt[(gt.org == o) & (gt.fit == tag)].set_index("structure").reindex(S)
            ax.errorbar(np.arange(len(S)) + off, s.delta, yerr=2 * s.se, fmt=mk, ms=3.5, color=c, lw=1, label=tag)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_xticks(range(len(S)), S, rotation=60, fontsize=7.5)
        ax.set_title(o, fontsize=9)
    axes[0].set_ylabel("delta (ECI), +-2 SE")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Organization deltas under Epoch's model, ceilings, drift and both (each fit's own ECI scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"out/figs/cap22_{MODE}_delta.png", dpi=130)
    plt.close(fig)
    print(f"done in {time.time() - t0:.0f}s")
