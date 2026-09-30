"""Residuals of Epoch's ECI fit against the gap, split by the model's release year, by organization, with a
loess curve and a +-2 SE band.

Residuals as in analysis/14_resid_plots.py: one fit of Epoch's model to all years (fit_eci_model, unchanged),
r = clipped score - sigmoid(slope (ECI - EDI)). Rows: all models, then each organization with at least
MIN_ORG_CELLS cells, then the remaining organizations pooled. Columns: model release year.

Loess: local linear (statsmodels lowess, no robustness iterations), span max(0.4, min(1, 20 / n)), drawn when a
panel has at least MIN_CELLS cells. SE: bootstrap over models within the panel (B draws of the panel's models
with replacement, all cells of a drawn model kept), because cells of the same model are correlated; pointwise
sd of the resampled curves, where at least half of the draws cover that gap. The band is drawn only when the
panel has at least MIN_MODELS models; with fewer, the curve is dashed and has no band.

Writes out/figs/resid_org_year.png (grid), out/figs/resid_org_year_overlay.png (years overlaid for the larger
organizations) and out/resid15_curves.csv (curve, SE and band by panel and gap)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from eci import prepare_benchmark_data, fit_eci_model
from eci.fitting import sigmoid
import cm

ZIP = "data/benchmark_data.zip"
YLIM, XLIM = 0.3, (-3.5, 8.0)
MIN_ORG_CELLS, MIN_CELLS, MIN_MODELS = 40, 10, 4
OVERLAY_MIN_CELLS = 100
B, SEED = 300, 11
C_BEFORE, C_AFTER = "#4C72B0", "#DD8452"
YEAR_COL = {2023: "#8172B3", 2024: "#55A868", 2025: "#C44E52", 2026: "#4C72B0"}
XLAB = "gap: model release - benchmark release (years)"

os.makedirs("out/figs", exist_ok=True)
df, ORG = cm.load(ZIP)
eci, edi, _ = fit_eci_model(prepare_benchmark_data(source=ZIP), bootstrap_samples=0)
x = df.merge(eci[["model_id", "eci"]], on="model_id", validate="many_to_one") \
      .merge(edi[["benchmark_id", "edi", "discriminability_scaled"]], on="benchmark_id", validate="many_to_one")
assert len(x) == len(df)
x["r"] = x.y_fit - sigmoid(x.discriminability_scaled * (x.eci - x.edi))
x["org"] = x.Model.map(ORG).fillna("unknown")
x["year"] = x.date.dt.year
YEARS = sorted(x.year.unique())

big = x.groupby("org").size()
big = list(big[big >= MIN_ORG_CELLS].sort_values(ascending=False).index)
rows = [("All models", x)] + [(o, x[x.org == o]) for o in big]
rest = x[~x.org.isin(big)]
rows.append((f"Other organizations ({rest.org.nunique()}, pooled)", rest))


def span(n):
    return max(0.4, min(1.0, 20 / n))


def curve(g, grid, rng):
    """Loess on grid, and its pointwise SE from the model bootstrap (NaN where not estimable)."""
    f = span(len(g))
    fit = lowess(g.r.to_numpy(), g.gap.to_numpy(), frac=f, it=0, xvals=grid)
    mods = g.Model.unique()
    if len(mods) < MIN_MODELS:
        return fit, np.full(len(grid), np.nan)
    by = {m: h for m, h in g.groupby("Model")}
    draws = np.full((B, len(grid)), np.nan)
    for b in range(B):
        s = pd.concat([by[m] for m in rng.choice(mods, len(mods), replace=True)])
        v = lowess(s.r.to_numpy(), s.gap.to_numpy(), frac=span(len(s)), it=0, xvals=grid)
        v[(grid < s.gap.min()) | (grid > s.gap.max())] = np.nan
        draws[b] = v
    se = np.nanstd(draws, axis=0, ddof=1)
    se[np.isnan(draws).mean(axis=0) > 0.5] = np.nan
    return fit, se


rng = np.random.default_rng(SEED)
curves, out = {}, []
for name, g in rows:
    for yr in YEARS:
        h = g[g.year == yr]
        if len(h) < MIN_CELLS:
            continue
        grid = np.linspace(h.gap.min(), h.gap.max(), 120)
        fit, se = curve(h, grid, rng)
        curves[(name, yr)] = (grid, fit, se)
        out.append(pd.DataFrame(dict(panel=name, year=yr, gap=grid, loess=fit, se=se, models=h.Model.nunique(), cells=len(h))))
pd.concat(out).round(5).to_csv("out/resid15_curves.csv", index=False)


def draw(ax, grid, fit, se, col, lw=1.4):
    if np.isnan(se).all():
        ax.plot(grid, fit, color=col, lw=lw, ls="--")
        return
    ax.fill_between(grid, fit - 2 * se, fit + 2 * se, color=col, alpha=0.2, lw=0)
    ax.plot(grid, fit, color=col, lw=lw)


def frame(ax):
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.3", lw=0.9, ls="--")
    ax.set_xlim(*XLIM); ax.set_ylim(-YLIM, YLIM)
    ax.set_xticks([-2, 0, 2, 4, 6])
    ax.tick_params(labelsize=7)


# Grid: rows are organizations, columns are release years
fig, axes = plt.subplots(len(rows), len(YEARS), figsize=(4.2 * len(YEARS), 2.35 * len(rows) + 0.8), sharex=True, sharey=True)
for i, (name, g) in enumerate(rows):
    for j, yr in enumerate(YEARS):
        ax = axes[i, j]
        h = g[g.year == yr]
        ms = 5 if len(h) > 300 else 9
        for flag, col in [(True, C_BEFORE), (False, C_AFTER)]:
            k = h[(h.gap > 0) == flag]
            ax.scatter(k.gap, k.r.clip(-YLIM + 0.01, YLIM - 0.01), s=ms, color=col, alpha=0.6, lw=0)
        if (name, yr) in curves:
            draw(ax, *curves[(name, yr)], "black")
        frame(ax)
        ax.set_title(f"{name[:40]}, {yr}: {h.Model.nunique()} models, {len(h)} cells", fontsize=7.5, loc="left")
    axes[i, 0].set_ylabel("residual", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel(XLAB, fontsize=8)
items = [plt.Line2D([], [], marker="o", ls="", color=C_BEFORE, label="benchmark released before the model (gap > 0)"),
         plt.Line2D([], [], marker="o", ls="", color=C_AFTER, label="benchmark released after the model (gap <= 0)"),
         plt.Line2D([], [], color="black", lw=1.4, label=f"loess with +-2 SE (bootstrap over models, B = {B})"),
         plt.Line2D([], [], color="black", lw=1.4, ls="--", label=f"loess, fewer than {MIN_MODELS} models: no band")]
fig.legend(handles=items, loc="upper center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.997))
fig.suptitle("Residuals of Epoch's ECI fit against the gap, by organization and model release year "
             "(one ECI fit to all years)", fontsize=12, y=1.005)
fig.tight_layout(rect=(0, 0, 1, 0.99))
fig.savefig("out/figs/resid_org_year.png", dpi=100, bbox_inches="tight")
plt.close(fig)

# Overlay: the years on one panel, for all models and the larger organizations
ov = [(n, g) for n, g in rows if len(g) >= OVERLAY_MIN_CELLS and not n.startswith("Other")]
ncol = 3
nrow = int(np.ceil(len(ov) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.3 * nrow + 0.6), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat):
    if k >= len(ov):
        ax.set_visible(False)
        continue
    name, g = ov[k]
    for yr in YEARS:
        if (name, yr) in curves:
            draw(ax, *curves[(name, yr)], YEAR_COL[yr], lw=1.6)
    frame(ax)
    ax.set_ylim(-0.15, 0.15)
    ax.set_title(f"{name}: " + ", ".join(f"{yr} {g[g.year == yr].Model.nunique()}" for yr in YEARS) + " models",
                 fontsize=8.5, loc="left")
for ax in axes[:, 0]:
    ax.set_ylabel("residual (loess)", fontsize=8)
for c in range(ncol):
    col = [ax for ax in axes[:, c] if ax.get_visible()]
    col[-1].set_xlabel(XLAB, fontsize=8)
    col[-1].tick_params(labelbottom=True)
items = [plt.Line2D([], [], color=YEAR_COL[yr], lw=2, label=f"models released in {yr}") for yr in YEARS] + \
        [plt.Line2D([], [], color="0.3", lw=1.4, ls="--", label=f"fewer than {MIN_MODELS} models: no band")]
fig.legend(handles=items, loc="upper center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Loess of the residuals against the gap by model release year, +-2 SE (bootstrap over models); "
             "y axis zoomed to +-0.15", fontsize=11, y=1.035)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig("out/figs/resid_org_year_overlay.png", dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"rows {len(rows)} ({', '.join(n for n, _ in rows)}), years {YEARS}, panels with a curve {len(curves)}")
for yr in YEARS:
    grid, fit, se = curves[("All models", yr)]
    pick = [np.argmin(np.abs(grid - v)) for v in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0) if grid.min() <= v <= grid.max()]
    print(f"All models {yr}: " + ", ".join(f"gap {grid[p]:+.1f}: {fit[p]:+.3f} +- {2 * se[p]:.3f}" for p in pick))
