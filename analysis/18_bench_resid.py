"""Benchmark-level look at the organization pre/post model (analysis/17_prepost.py org).

0. Residuals of the fitted model by benchmark (strip plot), pre and post cells separately, with means; and the
   within-benchmark drift, the OLS slope of the residual on model release date (raw and controlling for the
   logit fitted score), HC0 SEs.
1. Residuals of the fitted model (ridge lam chosen by CV in 17, read from out/prepost17_org_cv.csv), one panel per
   benchmark in release order, against the gap, points colored by organization, loess (local linear, span
   max(0.4, min(1, 20 / n))) with +-2 SE from a bootstrap over the benchmark's cells (B_BOOT draws; each cell is a
   different model). Residual = observed - fitted score.
2. Heatmap of the mean residual by organization and benchmark.
3. Leave one benchmark out: refit the near-unpenalized model (lam LAM_FE) without each benchmark and record every
   organization's delta (ECI points). When Winogrande (Epoch's slope anchor) is dropped, HellaSwag's slope is
   pinned instead; ECI-scale quantities do not depend on that choice except through Epoch's tiny ridge.

Writes out/figs/bench18_strip.png, out/figs/bench18_drift.png, out/figs/bench18_panels.png, out/figs/bench18_heatmap.png, out/figs/bench18_lobo.png, out/bench18.md,
out/bench18_resid.csv, out/bench18_bench.csv and out/bench18_lobo.csv."""
import os
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
import cm
from scipy.special import logit
from prepost import PrePost

LAM_FE = 1e-3
B_BOOT, SEED = 200, 5
YLIM = 0.3
MIN_LOESS = 10
DRIFT_MIN = 12
HEAT_MIN_CELLS = 40
ORG_COL = {"OpenAI": "#4C72B0", "Anthropic": "#DD8452", "Google DeepMind": "#C44E52", "Alibaba": "#55A868",
           "Meta AI": "#8172B3", "DeepSeek": "#937860", "xAI": "#DA8BC3"}
FOCUS = "Google DeepMind"


def load():
    df, ORG = cm.load()
    df["org"] = df.Model.map(ORG).fillna("unknown")
    return df


def lobo(bench):
    df = load()
    d = df[df.benchmark != bench]
    pp = PrePost(d, d.org, anchor="Winogrande" if bench != "Winogrande" else "HellaSwag")
    _, f = pp.fit_full(LAM_FE)
    dd, _, Vm, _, _ = pp.analyse(f.x, LAM_FE)
    return bench, dict(zip(pp.ident, dd)), dict(zip(pp.ident, np.sqrt(np.diag(Vm)))), f.status


def span(n):
    return max(0.4, min(1.0, 20 / n))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    cvt = pd.read_csv("out/prepost17_org_cv.csv")
    tab = cvt.groupby("lam").sse.sum()
    LAM_CV = float(tab.drop(np.inf).idxmin())
    tags = pd.read_csv("data/benchmark_filtering.csv").set_index("benchmark").filtered
    df = load()
    pp = PrePost(df, df.org)
    base, fcv = pp.fit_full(LAM_CV)
    ffe = pp.fit(pp.ones, LAM_FE, fcv.x)
    d_cv, _, _, _, _ = pp.analyse(fcv.x, LAM_CV)
    d_fe, _, Vm_fe, _, _ = pp.analyse(ffe.x, LAM_FE)
    full_fe = dict(zip(pp.ident, d_fe))
    se_fe = dict(zip(pp.ident, np.sqrt(np.diag(Vm_fe))))
    x = pp.df.copy()
    x["mu"] = pp.predict(fcv.x)
    x["r"] = x.y_fit - x.mu
    x["post"] = pp.post
    x[["Model", "org", "benchmark", "gap", "post", "y_fit", "mu", "r"]].round(5).to_csv("out/bench18_resid.csv", index=False)
    print(f"fits: cv lam {LAM_CV:g} status {fcv.status}, fe lam {LAM_FE:g} status {ffe.status}; "
          f"{FOCUS} delta cv {d_cv[pp.ident.index(FOCUS)]:+.2f}, fe {full_fe[FOCUS]:+.2f} +- {se_fe[FOCUS]:.2f}", flush=True)

    brel = x.groupby("benchmark").benchmark_release_date.first().sort_values()
    order = list(brel.index)

    # 0. Residuals by benchmark: pre and post cells, with their means (SE: one cell per model within a benchmark)
    def mse(v):
        return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (np.nan, np.nan)
    rows = []
    for b in order:
        g = x[x.benchmark == b]
        (mp, sp), (mq, sq) = mse(g[g.post == 0].r), mse(g[g.post == 1].r)
        rows.append(dict(benchmark=b, released=f"{brel[b]:%Y-%m}", filtered=tags[b], n=len(g), n_pre=int((g.post == 0).sum()),
                         n_post=int((g.post == 1).sum()), sd_r=g.r.std(), mean_pre=mp, se_pre=sp, mean_post=mq, se_post=sq,
                         corr_r_mu=np.corrcoef(g.r, g.mu)[0, 1] if len(g) > 2 else np.nan))
    bt = pd.DataFrame(rows)
    bt["diff"] = bt.mean_post - bt.mean_pre
    bt["se_diff"] = np.sqrt(bt.se_pre ** 2 + bt.se_post ** 2)
    bt["z"] = bt["diff"] / bt.se_diff
    # Drift: within-benchmark OLS slope of the residual on the gap (= model release date, years), raw and with logit
    # fitted score as a control (level misfit); HC0 SEs (one cell per model within a benchmark)
    x["lmu"] = logit(x.mu.clip(1e-3, 1 - 1e-3))
    dr = []
    for b in order:
        g = x[x.benchmark == b]
        if len(g) < DRIFT_MIN or g.gap.std() < 0.1:
            continue
        v = {}
        for nm, cols in [("raw", ["gap"]), ("ctl", ["gap", "lmu"])]:
            X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy() for c in cols])
            beta = np.linalg.lstsq(X, g.r.to_numpy(), rcond=None)[0]
            e = g.r.to_numpy() - X @ beta
            XtXi = np.linalg.inv(X.T @ X)
            v[nm] = beta[1], np.sqrt((XtXi @ (X.T * e ** 2) @ X @ XtXi)[1, 1])
        dr.append(dict(benchmark=b, n=len(g), slope=v["raw"][0], se=v["raw"][1], slope_ctl=v["ctl"][0], se_ctl=v["ctl"][1]))
    dr = pd.DataFrame(dr)
    dr["z"], dr["z_ctl"] = dr.slope / dr.se, dr.slope_ctl / dr.se_ctl
    bt = bt.merge(dr.drop(columns="n").add_prefix("drift_").rename(columns={"drift_benchmark": "benchmark"}), on="benchmark", how="left")
    bt.round(4).to_csv("out/bench18_bench.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 0.2 * len(dr) + 1.6))
    yy = np.arange(len(dr))[::-1]
    ax.errorbar(dr.slope, yy + 0.15, xerr=2 * dr.se, fmt="o", ms=3.5, color="#4C72B0", lw=1, label="raw")
    ax.errorbar(dr.slope_ctl, yy - 0.15, xerr=2 * dr.se_ctl, fmt="o", ms=3.5, color="#C44E52", mfc="white", lw=0.8,
                label="controlling for logit fitted score")
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(yy, [f"{b[:30]} ({brel[b]:%y-%m}, n {k})" for b, k in zip(dr.benchmark, dr.n)], fontsize=7)
    ax.set_xlim(-0.45, 0.45)
    for k, r in enumerate(dr.itertuples()):
        if abs(r.slope) > 0.45:
            ax.text(0.44 * np.sign(r.slope), yy[k] - 0.45, f"{r.slope:+.2f} (SE {r.se:.2f}), off scale", fontsize=6,
                    ha="right" if r.slope > 0 else "left", va="center", color="#4C72B0")
    ax.set_xlabel("slope of the residual on model release date (score per year), +-2 SE (HC0)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(f"Within-benchmark drift of the residuals (lam = {LAM_CV:g});\nbenchmarks with at least {DRIFT_MIN} cells, in release order", fontsize=9)
    fig.tight_layout()
    fig.savefig("out/figs/bench18_drift.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(19, 6.5))
    jit = np.random.default_rng(1)
    for j, b in enumerate(order):
        g = x[x.benchmark == b]
        for pv, c, off in [(0, "#DD8452", -0.18), (1, "#4C72B0", 0.18)]:
            h = g[g.post == pv]
            ax.scatter(j + off + jit.uniform(-0.1, 0.1, len(h)), h.r.clip(-YLIM + 0.01, YLIM - 0.01), s=5, color=c, alpha=0.45, lw=0)
            r = bt.iloc[j]
            m, se = (r.mean_pre, r.se_pre) if pv == 0 else (r.mean_post, r.se_post)
            if np.isfinite(m):
                ax.errorbar(j + off, m, yerr=2 * se, fmt="_", color="black", ms=9, lw=1.2, mew=1.8, zorder=4)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(range(len(order)), [f"{b[:26]} ({brel[b]:%y-%m}){' *' if tags[b] != 'no' else ''}" for b in order], rotation=90, fontsize=7)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(-YLIM, YLIM)
    ax.set_ylabel("residual (observed - fitted score)")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#DD8452", label="model released before the benchmark (gap <= 0)"),
                       plt.Line2D([], [], marker="o", ls="", color="#4C72B0", label="model released after the benchmark (gap > 0)"),
                       plt.Line2D([], [], marker="_", ms=9, mew=1.8, ls="", color="black", label="mean +-2 SE")],
              loc="upper left", fontsize=8, ncol=3, frameon=False)
    ax.set_title(f"Residuals of the organization pre/post model (lam = {LAM_CV:g}) by benchmark, in release order; "
                 "* adversarially filtered (yes or partial)", fontsize=10)
    fig.tight_layout()
    fig.savefig("out/figs/bench18_strip.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # 1. Panels by benchmark
    rng = np.random.default_rng(SEED)
    ncol = 6
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(19, 2.55 * nrow + 0.8), sharey=True)
    for k, ax in enumerate(axes.flat):
        if k >= len(order):
            ax.set_visible(False)
            continue
        bname = order[k]
        g = x[x.benchmark == bname]
        other = g[~g.org.isin(ORG_COL)]
        ax.scatter(other.gap, other.r.clip(-YLIM + 0.01, YLIM - 0.01), s=9, color="0.65", lw=0)
        for o, c in ORG_COL.items():
            h = g[g.org == o]
            big = o == FOCUS
            ax.scatter(h.gap, h.r.clip(-YLIM + 0.01, YLIM - 0.01), s=24 if big else 11, color=c, lw=0.6 if big else 0,
                       edgecolor="black" if big else None, zorder=3 if big else 2)
        if len(g) >= MIN_LOESS:
            grid = np.linspace(g.gap.min(), g.gap.max(), 80)
            fit = lowess(g.r.to_numpy(), g.gap.to_numpy(), frac=span(len(g)), it=0, xvals=grid)
            draws = np.full((B_BOOT, len(grid)), np.nan)
            for b in range(B_BOOT):
                s = g.iloc[rng.integers(0, len(g), len(g))]
                v = lowess(s.r.to_numpy(), s.gap.to_numpy(), frac=span(len(s)), it=0, xvals=grid)
                v[(grid < s.gap.min()) | (grid > s.gap.max())] = np.nan
                draws[b] = v
            se = np.nanstd(draws, axis=0, ddof=1)
            se[np.isnan(draws).mean(axis=0) > 0.5] = np.nan
            ax.fill_between(grid, fit - 2 * se, fit + 2 * se, color="black", alpha=0.12, lw=0)
            ax.plot(grid, fit, color="black", lw=1.2)
        ax.axhline(0, color="0.6", lw=0.7)
        ax.axvline(0, color="0.3", lw=0.8, ls="--")
        ax.set_ylim(-YLIM, YLIM)
        ax.tick_params(labelsize=6.5)
        gf = g[g.org == FOCUS]
        foc = f"{FOCUS[:6]} mean r {gf.r.mean():+.3f} (n {len(gf)})" if len(gf) else f"no {FOCUS} cells"
        ax.set_title(f"{bname[:34]}\n{brel[bname]:%Y-%m}, filtered: {tags[bname]}, n = {len(g)}; {foc}", fontsize=7, loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("residual", fontsize=7.5)
    items = [plt.Line2D([], [], marker="o", ls="", color=c, mec="black" if o == FOCUS else c, label=o) for o, c in ORG_COL.items()]
    items += [plt.Line2D([], [], marker="o", ls="", color="0.65", label="other organizations"),
              plt.Line2D([], [], color="black", lw=1.2, label=f"loess, +-2 SE (bootstrap over cells, B = {B_BOOT})")]
    fig.legend(handles=items, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.998))
    fig.suptitle(f"Residuals of the organization pre/post model (lam = {LAM_CV:g}) by benchmark, against the gap "
                 "(model release - benchmark release, years; benchmarks in release order)", fontsize=12, y=1.008)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig("out/figs/bench18_panels.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    # 2. Heatmap: mean residual by organization and benchmark
    oc = x.org.value_counts()
    orgs = list(oc[oc >= HEAT_MIN_CELLS].index)
    H = x[x.org.isin(orgs)].pivot_table(index="org", columns="benchmark", values="r", aggfunc="mean").reindex(index=orgs, columns=order)
    N = x[x.org.isin(orgs)].pivot_table(index="org", columns="benchmark", values="r", aggfunc="size").reindex(index=orgs, columns=order)
    fig, ax = plt.subplots(figsize=(19, 0.42 * len(orgs) + 2.6))
    im = ax.imshow(H.to_numpy(float), cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto")
    for i in range(len(orgs)):
        for j in range(len(order)):
            if not np.isnan(N.iat[i, j]):
                ax.text(j, i, f"{int(N.iat[i, j])}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(H.iat[i, j]) > 0.06 else "0.15")
    ax.set_yticks(range(len(orgs)), [f"{o} ({oc[o]})" for o in orgs], fontsize=8)
    ax.set_xticks(range(len(order)), [f"{b[:26]} ({brel[b]:%y-%m})" for b in order], rotation=90, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("mean residual (score)", fontsize=8)
    ax.set_title(f"Mean residual of the organization pre/post model (lam = {LAM_CV:g}) by organization and benchmark; "
                 f"numbers are cells; organizations with at least {HEAT_MIN_CELLS} cells", fontsize=10)
    fig.tight_layout()
    fig.savefig("out/figs/bench18_heatmap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("panels and heatmap written", flush=True)

    # 3. Leave one benchmark out
    with Pool(3) as pool:
        out = pool.map(lobo, order)
    lob = pd.DataFrame({b: dd for b, dd, _, _ in out}).T.reindex(order)
    lob.index.name = "dropped"
    lob.round(4).to_csv("out/bench18_lobo.csv")
    bad = [b for b, _, _, st in out if st <= 0]
    main = [o for o in ORG_COL if o in full_fe]
    ch = lob[main] - pd.Series(full_fe)[main]

    fig, axes = plt.subplots(1, len(main), figsize=(2.6 * len(main), 11), sharey=True)
    for ax, o in zip(axes, main):
        v = ch[o].to_numpy()
        ax.barh(np.arange(len(order))[::-1], v, color=np.where(v < 0, "#C44E52", "#4C72B0"), height=0.8)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_title(f"{o}\nfull {full_fe[o]:+.2f} +- {se_fe[o]:.2f}", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("delta without the benchmark\nminus full delta (ECI)", fontsize=7.5)
    axes[0].set_yticks(np.arange(len(order))[::-1], [f"{b[:28]} ({brel[b]:%y-%m})" for b in order], fontsize=6.5)
    fig.suptitle(f"Leave one benchmark out: change in each organization's delta (lam = {LAM_FE:g}); "
                 "positive bar: the benchmark pulls the delta down", fontsize=10)
    fig.tight_layout()
    fig.savefig("out/figs/bench18_lobo.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Report
    L = ["# Benchmark-level look at the organization pre/post model (analysis/18_bench_resid.py)", "",
         f"Residuals from the CV fit (lam {LAM_CV:g}); leave-one-benchmark-out deltas from the near-unpenalized fit (lam {LAM_FE:g}). "
         f"Refits with a non-positive solver status: {bad or 'none'}.", "",
         "## Leave one benchmark out, main organizations (ECI points)", "",
         "| organization | full delta | SE (model-based) | range without one benchmark | benchmarks that move it most (change when dropped) |",
         "|---|---|---|---|---|"]
    for o in main:
        top = ch[o].abs().sort_values(ascending=False).index[:5]
        L.append(f"| {o} | {full_fe[o]:+.2f} | {se_fe[o]:.2f} | {lob[o].min():+.2f} to {lob[o].max():+.2f} | "
                 + "; ".join(f"{b} {ch.at[b, o]:+.2f}" for b in top) + " |")
    L += ["", "## Pre and post cells by benchmark (residuals of the CV fit, score points; |z| >= 2 or sd >= 0.08 listed)", "",
          f"Benchmarks with |post - pre| z >= 2: {int((bt.z.abs() >= 2).sum())} of {bt.z.notna().sum()} "
          f"(post higher in {int((bt.z >= 2).sum())}). Median residual sd {bt.sd_r.median():.3f}.", "",
          "| benchmark | released | filtered | n pre | n post | mean pre | mean post | post - pre | z | sd | corr(r, fitted) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in bt[(bt.z.abs() >= 2) | (bt.sd_r >= 0.08)].iterrows():
        L.append(f"| {r.benchmark} | {r.released} | {r.filtered} | {r.n_pre} | {r.n_post} | {r.mean_pre:+.3f} | {r.mean_post:+.3f} | "
                 f"{r['diff']:+.3f} | {r.z:+.1f} | {r.sd_r:.3f} | {r.corr_r_mu:+.2f} |".replace("+nan", "").replace("nan", ""))
    L += ["", f"## Within-benchmark drift (slope of the residual on model release date, score per year; HC0 SEs; |z| >= 2 in either listed)", "",
          f"{len(dr)} benchmarks with at least {DRIFT_MIN} cells: |z| >= 2 raw in {int((dr.z.abs() >= 2).sum())} "
          f"({int((dr.z >= 2).sum())} up), with the control in {int((dr.z_ctl.abs() >= 2).sum())} ({int((dr.z_ctl >= 2).sum())} up).", "",
          "| benchmark | released | n | slope | SE | z | slope (control) | SE | z |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in dr[(dr.z.abs() >= 2) | (dr.z_ctl.abs() >= 2)].sort_values("z").iterrows():
        L.append(f"| {r.benchmark} | {brel[r.benchmark]:%Y-%m} | {r.n} | {r.slope:+.3f} | {r.se:.3f} | {r.z:+.1f} | "
                 f"{r.slope_ctl:+.3f} | {r.se_ctl:.3f} | {r.z_ctl:+.1f} |")
    gf = x[x.org == FOCUS]
    L += ["", f"## {FOCUS} by benchmark (residuals of the CV fit, score points)", "",
          "| benchmark | released | filtered | cells pre | cells post | mean r pre | mean r post | delta change when dropped |",
          "|---|---|---|---|---|---|---|---|"]
    for b in order:
        h = gf[gf.benchmark == b]
        if not len(h):
            continue
        pre, post = h[h.post == 0].r, h[h.post == 1].r
        L.append(f"| {b} | {brel[b]:%Y-%m} | {tags[b]} | {len(pre)} | {len(post)} | "
                 f"{pre.mean():+.3f} | {post.mean():+.3f} | {ch.at[b, FOCUS]:+.2f} |".replace("+nan", "").replace("nan", ""))
    # Largest org by benchmark mean residuals in the heatmap (at least 3 cells)
    Hs = H.where(N >= 3).stack().dropna().sort_values()
    L += ["", "## Largest organization-by-benchmark mean residuals (at least 3 cells)", "",
          "Most negative: " + "; ".join(f"{o} on {b} {v:+.3f} (n {int(N.at[o, b])})" for (o, b), v in Hs.head(8).items()) + ".", "",
          "Most positive: " + "; ".join(f"{o} on {b} {v:+.3f} (n {int(N.at[o, b])})" for (o, b), v in Hs.tail(8)[::-1].items()) + "."]
    open("out/bench18.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
