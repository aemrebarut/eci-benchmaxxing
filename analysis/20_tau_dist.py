"""Organization shifts delta = tau_post - tau_pre without and with the benchmark drift (analysis/19_drift.py; drift at
its CV choice mu), for the near-unpenalized fit (lam 1e-3) and the CV ridge. Each organization's estimate is shown as
a point with +-1 and +-2 SE and as its normal sampling distribution N(delta, SE^2), SE the larger of the clustered
and model-based SEs. ECI points in each fit's own scale (b differs with the drift); z = delta / SE is scale-free.
Reads out/drift19_groups.csv, out/drift19_cv.csv and out/prepost17_org_groups.csv; writes out/tau20.md and
out/figs/tau20_near.png, out/figs/tau20_cv.png."""
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XLIM = {"near-unpenalized": 6, "CV lam given mu": 1.6}
COL = {"no drift": "0.35", "drift": "#C44E52"}

g = pd.read_csv("out/drift19_groups.csv")
cv = pd.read_csv("out/drift19_cv.csv").groupby(["mu", "lam"]).sse.sum()
MU = float(cv.idxmin()[0])
info = pd.read_csv("out/prepost17_org_groups.csv").set_index("org")
g = g[g.mu.isin([MU, np.inf])].copy()
g["model"] = np.where(np.isfinite(g.mu), "drift", "no drift")
g["z"] = g.delta / g.se

L = [f"# Organization shifts without and with the benchmark drift (analysis/20_tau_dist.py)", "",
     f"delta = tau_post - tau_pre in ECI points of each fit; drift at the CV choice mu = {MU:g}. SE: larger of clustered "
     "and model-based; se_cl and se_mb listed. Organizations ordered by the no-drift near-unpenalized SE.", ""]
for fit, fname in [("near-unpenalized", "near"), ("CV lam given mu", "cv")]:
    t = g[g.fit == fit].pivot(index="org", columns="model", values=["lam", "delta", "se", "se_cl", "se_mb", "z"])
    order = t[("se", "no drift")].sort_values().index if fit == "near-unpenalized" else order
    t = t.reindex(order)
    lab = [f"{o[:24]} ({info.at[o, 'models_both']}/{info.at[o, 'models']}, {info.at[o, 'cells_pre']}/{info.at[o, 'cells_post']})" for o in order]
    lams = {m: t[("lam", m)].iloc[0] for m in ("no drift", "drift")}
    xl = XLIM[fit]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.42 * len(order) + 1.8), sharey=True, gridspec_kw=dict(width_ratios=[1, 1.1]))
    yy = np.arange(len(order))[::-1]
    for m, off in [("no drift", 0.14), ("drift", -0.14)]:
        d, s = t[("delta", m)].to_numpy(), t[("se", m)].to_numpy()
        c = COL[m]
        ax.hlines(yy + off, np.clip(d - 2 * s, -xl, xl), np.clip(d + 2 * s, -xl, xl), color=c, lw=0.9)
        ax.hlines(yy + off, np.clip(d - s, -xl, xl), np.clip(d + s, -xl, xl), color=c, lw=2.6)
        inside = np.abs(d) <= xl
        ax.plot(d[inside], yy[inside] + off, "o", color=c, mfc="white", ms=4.5, mew=1.3,
                label=f"{m} (lam {lams[m]:g}{'' if m == 'no drift' else f', mu {MU:g}'})")
        for k in np.flatnonzero(~inside):
            ax.annotate("", (np.sign(d[k]) * xl, yy[k] + off), (np.sign(d[k]) * xl * 0.88, yy[k] + off),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        for k in range(len(order)):
            ax.text(xl * 1.02, yy[k] + off, f"{d[k]:+.2f} +- {s[k]:.2f}", fontsize=6.3, color=c, va="center", ha="left",
                    clip_on=False)
        # Sampling distributions, each scaled to the same peak height
        grid = np.linspace(-xl, xl, 600)
        for k in range(len(order)):
            dens = stats.norm.pdf(grid, d[k], s[k])
            h = 0.8 * dens / stats.norm.pdf(0, 0, s[k])
            if m == "no drift":
                ax2.fill_between(grid, yy[k] - 0.4, yy[k] - 0.4 + h, color=c, alpha=0.25, lw=0)
            else:
                ax2.plot(grid, yy[k] - 0.4 + h, color=c, lw=1.3)
    for a in (ax, ax2):
        a.axvline(0, color="0.5", lw=0.8)
        a.set_xlim(-xl, xl)
    ax.set_yticks(yy, lab, fontsize=7.5)
    ax.set_xlabel("delta (ECI points), thick +-1 SE, thin +-2 SE")
    ax.legend(fontsize=7.5, loc="lower left")
    ax2.set_xlabel("N(delta, SE^2), each scaled to the same peak height; grey no drift, red drift")
    ax2.tick_params(labelleft=False)
    ax2.yaxis.set_tick_params(length=0)
    fig.suptitle(f"Organization shift delta = tau_post - tau_pre, {fit} fits, without and with the benchmark drift\n"
                 "labels: (models with cells on both sides / models, cells pre / post); numbers: delta +- SE; arrows: point off scale",
                 fontsize=10)
    fig.tight_layout(w_pad=7)
    fig.savefig(f"out/figs/tau20_{fname}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    L += [f"## {fit} (no drift: lam {lams['no drift']:g}; drift: lam {lams['drift']:g}, mu {MU:g})", "",
          "| organization | models both / models | cells pre / post | no drift: delta | SE | se_cl | se_mb | z | drift: delta | SE | se_cl | se_mb | z |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in order:
        L.append(f"| {o} | {info.at[o, 'models_both']}/{info.at[o, 'models']} | {info.at[o, 'cells_pre']}/{info.at[o, 'cells_post']} | "
                 + " | ".join(f"{t.at[o, ('delta', m)]:+.2f} | {t.at[o, ('se', m)]:.2f} | {t.at[o, ('se_cl', m)]:.2f} | "
                              f"{t.at[o, ('se_mb', m)]:.2f} | {t.at[o, ('z', m)]:+.2f}" for m in ("no drift", "drift")) + " |")
    zz = t[[("z", "no drift"), ("z", "drift")]]
    L += ["", f"|z| >= 2: {int((zz[('z', 'no drift')].abs() >= 2).sum())} without drift, {int((zz[('z', 'drift')].abs() >= 2).sum())} with. "
          f"Positive estimates: {int((t[('delta', 'no drift')] > 0).sum())} of {len(t)} without drift, {int((t[('delta', 'drift')] > 0).sum())} with.", ""]
open("out/tau20.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
