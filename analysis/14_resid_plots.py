"""Residuals of Epoch's own ECI fit against the gap (model release date minus benchmark release date, in
years): one panel per model and one per organization, each with a loess smoother.

The fit is Epoch's fit_eci_model (the installed eci package, unchanged, no bootstrap) on the data snapshot.
Fitted score mu = sigmoid(slope_j (ECI_i - EDI_j)); residual r = y - mu in score points, y the score Epoch
fits (clipped to [0.001, 0.999]). Positive r: the model did better on that benchmark than its ECI predicts.
Positive gap: the benchmark was public before the model was released.

Each model's ECI absorbs its average residual and each benchmark's difficulty absorbs its own, so what the
panels can show is a tilt within a model: residuals above zero at positive gaps and below zero at negative
gaps. Org panels pool the cells of the org's models. The loess is local linear (statsmodels lowess, no
robustness iterations), span FRAC_ORG (0.5) for panels with at least 40 cells, else FRAC_SMALL (1, all
cells, since model panels have 4 to 30). Points beyond +-YLIM are drawn at the edge as triangles.

Writes out/figs/resid_models.pdf (all model panels) with PNG pages out/figs/resid_models_pNN.png,
out/figs/resid_orgs.png, and out/resid14_models.csv and out/resid14_orgs.csv with the mean residual at
positive and at negative gaps."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.nonparametric.smoothers_lowess import lowess
from eci import prepare_benchmark_data, fit_eci_model
from eci.fitting import sigmoid
import cm

ZIP = "data/benchmark_data.zip"
YLIM = 0.3
FRAC_ORG, FRAC_SMALL = 0.5, 1.0
XLIM = (-3.5, 8.0)
C_BEFORE, C_AFTER = "#4C72B0", "#DD8452"
NCOL, NROW = 5, 6
XLAB = "gap: model release - benchmark release (years)"
YLAB = "residual (observed - fitted score)"

os.makedirs("out/figs", exist_ok=True)
df, ORG = cm.load(ZIP)
eci, edi, _ = fit_eci_model(prepare_benchmark_data(source=ZIP), bootstrap_samples=0)
x = df.merge(eci[["model_id", "eci"]], on="model_id", validate="many_to_one") \
      .merge(edi[["benchmark_id", "edi", "discriminability_scaled"]], on="benchmark_id", validate="many_to_one")
assert len(x) == len(df)
x["mu"] = sigmoid(x.discriminability_scaled * (x.eci - x.edi))
x["r"] = x.y_fit - x.mu
x["org"] = x.Model.map(ORG).fillna("unknown")
x["before"] = x.gap > 0  # benchmark released before the model
assert x.gap.between(*XLIM).all()


def smooth(ax, g):
    n = len(g)
    if n < 4:
        return
    frac = FRAC_ORG if n >= 40 else FRAC_SMALL
    s = lowess(g.r.to_numpy(), g.gap.to_numpy(), frac=frac, it=0, return_sorted=True)
    ax.plot(s[:, 0], s[:, 1], color="black", lw=1.3)


def points(ax, g, ms):
    for flag, col in [(True, C_BEFORE), (False, C_AFTER)]:
        h = g[g.before == flag]
        inside = h.r.abs() <= YLIM
        ax.scatter(h.gap[inside], h.r[inside], s=ms, color=col, alpha=0.75, lw=0)
        for sgn, mk in [(1, "^"), (-1, "v")]:
            out = h[sgn * h.r > YLIM]
            ax.scatter(out.gap, np.full(len(out), sgn * (YLIM - 0.015)), s=ms * 1.6, color=col, marker=mk, lw=0)


def frame(ax, title, fs):
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axvline(0, color="0.3", lw=0.9, ls="--")
    ax.set_xlim(*XLIM); ax.set_ylim(-YLIM, YLIM)
    ax.set_title(title, fontsize=fs, loc="left")
    ax.tick_params(labelsize=6.5)
    ax.set_xticks([-2, 0, 2, 4, 6])


def contrast(g):
    b, a = g.r[g.before], g.r[~g.before]
    diff = b.mean() - a.mean() if len(b) and len(a) else np.nan
    se = np.sqrt(b.var(ddof=1) / len(b) + a.var(ddof=1) / len(a)) if len(b) > 1 and len(a) > 1 else np.nan
    return pd.Series(dict(cells=len(g), n_pos_gap=len(b), n_neg_gap=len(a), r_pos_gap=b.mean(), r_neg_gap=a.mean(),
                          diff=diff, se_diff=se))


# Order: organizations by number of models, models by release date within each
org_size = x.drop_duplicates("Model").org.value_counts()
mods = x.drop_duplicates("Model")[["Model", "org", "date"]].copy()
mods["org_rank"] = mods.org.map({o: k for k, o in enumerate(org_size.index)})
mods = mods.sort_values(["org_rank", "date", "Model"]).reset_index(drop=True)

legend_items = [plt.Line2D([], [], marker="o", ls="", color=C_BEFORE, label="benchmark released before the model (gap > 0)"),
                plt.Line2D([], [], marker="o", ls="", color=C_AFTER, label="benchmark released after the model (gap <= 0)"),
                plt.Line2D([], [], color="black", lw=1.3, label="loess")]
per = NCOL * NROW
pages = int(np.ceil(len(mods) / per))
for f in os.listdir("out/figs"):
    if f.startswith("resid_models_p") and f.endswith(".png"):
        os.remove(os.path.join("out/figs", f))
with PdfPages("out/figs/resid_models.pdf") as pdf:
    for p in range(pages):
        fig, axes = plt.subplots(NROW, NCOL, figsize=(15, 17.5), sharex=True, sharey=True)
        for k, ax in enumerate(axes.flat):
            i = p * per + k
            if i >= len(mods):
                ax.set_visible(False)
                continue
            m = mods.iloc[i]
            g = x[x.Model == m.Model]
            points(ax, g, 14)
            smooth(ax, g)
            frame(ax, f"{m.Model[:40]}\n{m.org[:30]}, {m.date:%Y-%m}, n = {len(g)}", 7)
        for c in range(NCOL):
            col = [ax for ax in axes[:, c] if ax.get_visible()]
            col[-1].set_xlabel(XLAB, fontsize=7)
            col[-1].tick_params(labelbottom=True)
        for ax in axes[:, 0]:
            ax.set_ylabel(YLAB, fontsize=7.5)
        fig.legend(handles=legend_items, loc="upper center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(f"Residuals of Epoch's ECI fit by model, against the gap (page {p + 1} of {pages}; "
                     f"ordered by organization, then model release date)", fontsize=11, y=1.01)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(f"out/figs/resid_models_p{p + 1:02d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

# Organizations: all cells pooled first, then each organization
orgs = list(org_size.index)
n_panels = len(orgs) + 1
ncol = 6
nrow = int(np.ceil(n_panels / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(18, 3.0 * nrow + 0.6), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat):
    if k >= n_panels:
        ax.set_visible(False)
        continue
    if k == 0:
        g, title = x, f"All models\n{x.Model.nunique()} models, {len(x)} cells"
    else:
        o = orgs[k - 1]
        g = x[x.org == o]
        title = f"{o[:38]}\n{g.Model.nunique()} models, {len(g)} cells"
    points(ax, g, 6 if len(g) > 200 else 10)
    smooth(ax, g)
    frame(ax, title, 8)
for ax in axes[:, 0]:
    ax.set_ylabel(YLAB, fontsize=8)
for c in range(ncol):
    col = [ax for ax in axes[:, c] if ax.get_visible()]
    col[-1].set_xlabel(XLAB, fontsize=8)
    col[-1].tick_params(labelbottom=True)
fig.legend(handles=legend_items, loc="upper center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
fig.suptitle("Residuals of Epoch's ECI fit by organization, against the gap "
             "(organization names as in Epoch's metadata)", fontsize=12, y=1.01)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig("out/figs/resid_orgs.png", dpi=120, bbox_inches="tight")
plt.close(fig)

tm = x.groupby("Model").apply(contrast, include_groups=False).join(mods.set_index("Model")[["org", "date"]])
tm["date"] = tm.date.dt.strftime("%Y-%m-%d")
tm.round(4).to_csv("out/resid14_models.csv")
to = x.groupby("org").apply(contrast, include_groups=False).sort_values("cells", ascending=False)
to.round(4).to_csv("out/resid14_orgs.csv")
print(f"cells {len(x)}, models {len(mods)}, orgs {len(orgs)}, model pages {pages}")
print(f"residual sd {x.r.std():.4f}; beyond +-{YLIM}: {(x.r.abs() > YLIM).sum()} cells")
ok = tm.dropna(subset=["diff"])
print(f"models with cells at both signs of the gap: {len(ok)}; mean of (r at gap > 0 minus r at gap <= 0) "
      f"{ok['diff'].mean():+.4f}, median {ok['diff'].median():+.4f}, share positive {(ok['diff'] > 0).mean():.2f}")
print(to.round(3).to_string())
