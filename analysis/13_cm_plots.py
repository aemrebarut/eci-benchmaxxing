"""Figures for the common-mechanism model: the fitted gap curve and the organization coefficients, each with
2 standard error bars. Writes out/figs/cm_curve.png, out/figs/cm_orgs.png and out/cm13.md.

Curve: effect(gap) = m h_tau(gap) in ECI at the primary fit (tau = 4, kink at 0). Its band is
+- 2 SE(m) h_tau(gap), conditional on tau and the kink. SE(m) is the sd of m over the bootstrap under the
fitted effect (analysis/11_cm_boot.py alt_primary, which includes the uncertainty of the ECI scale); the
Hessian SE, sigma * sqrt((H^-1)_mm), is reported next to it. The fit with the kink 1 year before
release is overlaid (the placebo).

Orgs: m + u_org in ECI per year from the stage 2 fit (org and model deviations as ridge-shrunk random
effects, ridges tuned by Fellner-Schall). SE from sigma^2 H^-1 of the penalized fit (the prediction
error variance of a shrunken random effect). Open markers: the same fit with the org ridge removed
(unshrunk org effects), for comparison."""
import os, sys
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cm

TAU = 4.0
os.makedirs("out/figs", exist_ok=True)
df, ORG = cm.load()
d = cm.Data(df, y_col="y_fit", org=ORG)
LAM_G = float(open("out/cm10_lam_g.txt").read())
BASE = dict(rho=False, w_loc=1e-6, lam_g=LAM_G)


def cov(o):
    """sigma^2 H^-1 with sigma^2 = SSE / (n - edf)."""
    Hinv = np.linalg.inv(o["H"])
    edf = float(np.trace(Hinv @ (o["H"] - o["lay"].P)))
    return o["sse"] / (d.n - edf) * Hinv


# Curve
o1 = cm.fit(d, cm.Spec(gap=True, tau=TAU, **BASE))
_, b1 = cm.eci_scale(o1, d)
im = o1["lay"].sl["m"].start
se_hess = b1 * np.sqrt(cov(o1)[im, im])
boot = pd.read_csv("out/boot11_alt_primary.csv")
se_boot = boot.m_tau4.std()
m1 = o1["m"] * b1
ok = cm.fit(d, cm.Spec(gap=True, tau=TAU, x0=-1.0, **BASE))
_, bk = cm.eci_scale(ok, d)
g = np.linspace(-3, 6, 400)
hg = cm.h(g, TAU)

fig, (ax, axh) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[4, 1], hspace=0.06))
ax.fill_between(g, (m1 - 2 * se_boot) * hg, (m1 + 2 * se_boot) * hg, color="#4C72B0", alpha=0.22, lw=0,
                label=f"+-2 SE (bootstrap SE of m = {se_boot:.2f})")
ax.plot(g, m1 * hg, color="#4C72B0", lw=2.2, label=f"fitted: kink at release, tau = {TAU:g} years, m = {m1:.2f} ECI/yr")
ax.plot(g, ok["m"] * bk * cm.h(g, TAU, -1.0), color="#C44E52", lw=1.6, ls="--",
        label=f"kink 1 year before release (better fit: T 18.6 vs 9.4)")
ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0, color="0.5", lw=0.8, ls=":")
ax.set_ylabel("effect on the model's score (ECI points)")
ax.set_title("Shared gap effect, common-mechanism model (primary fit)", fontsize=11)
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
ax.text(0.99, 0.02, "band conditional on tau and the kink location", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="0.35")
axh.hist(df.gap.clip(-3, 6), bins=np.arange(-3, 6.25, 0.25), color="0.6")
axh.axvline(0, color="0.5", lw=0.8, ls=":")
axh.text(0.99, 0.92, "gaps beyond -3 or 6 years are counted in the end bins", transform=axh.transAxes, ha="right", va="top", fontsize=7.5, color="0.35")
axh.set_ylabel("cells", fontsize=9)
axh.set_xlabel("gap: model release minus benchmark release (years)")
fig.savefig("out/figs/cm_curve.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# Orgs
spec2 = cm.Spec(gap=True, tau=TAU, org=True, model_dev=True, lam_u=1.0, lam_v=10.0, **BASE)
o2, spec2, _ = cm.tune(d, spec2, blocks=("u", "v"), max_rounds=300)
o3 = cm.fit(d, spec2.replace(lam_u=1e-6), x0=o2["x"])  # org ridge removed


def org_table(o):
    lay = o["lay"]
    _, b = cm.eci_scale(o, d)
    C = cov(o)
    im, us = lay.sl["m"].start, lay.sl["u"].start
    est, se = [], []
    for k in range(len(d.orgs)):
        c = np.zeros(lay.n); c[im] = 1; c[us + k] = 1
        est.append(b * (o["x"] @ c)); se.append(b * np.sqrt(c @ C @ c))
    u = b * o["x"][lay.sl["u"]]  # deviation from the common m alone, without its shared uncertainty
    u_se = b * np.sqrt(np.diag(C)[lay.sl["u"]])
    return np.array(est), np.array(se), b * o["m"], b * np.sqrt(C[im, im]), u, u_se


est, se, mm, mse, dev, dev_se = org_table(o2)
est_u, se_u, _, _, _, _ = org_table(o3)
n_models = np.bincount(d.oi, minlength=len(d.orgs))
n_cells = np.bincount(d.oi[d.mi], minlength=len(d.orgs))
t = pd.DataFrame(dict(org=d.orgs, models=n_models, cells=n_cells, eff=est, se=se, dev=dev, dev_se=dev_se,
                      eff_unshrunk=est_u, se_unshrunk=se_u))
t = t.sort_values("eff").reset_index(drop=True)
t.to_csv("out/cm13_orgs.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 0.26 * len(t) + 1.4))
y = np.arange(len(t))
ax.errorbar(t.eff_unshrunk, y + 0.18, xerr=2 * t.se_unshrunk, fmt="o", mfc="white", mec="0.55", ecolor="0.75", ms=4, lw=1,
            label="org ridge removed (unshrunk)")
ax.errorbar(t.eff, y - 0.1, xerr=2 * t.se, fmt="o", color="#4C72B0", ms=5, lw=1.6, label="shrunken (random effect), primary")
ax.axvline(mm, color="#4C72B0", lw=0.9, ls="--", label=f"common m = {mm:.2f}")
ax.axvline(0, color="0.5", lw=0.8)
ax.set_yticks(y, [f"{o[:34]} ({mo}, {c})" for o, mo, c in zip(t.org, t.models, t.cells)], fontsize=7.5)
ax.set_xlabel("gap effect per year just after release (ECI points), +-2 SE")
ax.set_title(f"Organization coefficients m + u_org (tau = {TAU:g}); labels: (models, cells)", fontsize=10)
ax.legend(loc="lower right", fontsize=8, frameon=True)
lo = min((t.eff_unshrunk - 2 * t.se_unshrunk).quantile(0.05), (t.eff - 2 * t.se).min()) - 1
hi = max((t.eff_unshrunk + 2 * t.se_unshrunk).quantile(0.95), (t.eff + 2 * t.se).max()) + 1
ax.set_xlim(lo, hi)
fig.savefig("out/figs/cm_orgs.png", dpi=160, bbox_inches="tight")
plt.close(fig)

L = ["# Figures (analysis/13_cm_plots.py)", "",
     f"Curve: m = {m1:+.2f} ECI/yr at tau = {TAU:g}; SE(m) bootstrap {se_boot:.2f} (B = {len(boot)}), Hessian {se_hess:.2f}. "
     f"Effect at 1 year {m1 * cm.h(1.0, TAU):+.2f} +- {2 * se_boot * cm.h(1.0, TAU):.2f}, at 2 years {m1 * cm.h(2.0, TAU):+.2f} +- {2 * se_boot * cm.h(2.0, TAU):.2f} (2 SE).",
     f"Kink -1 fit: m = {ok['m'] * bk:+.2f} ECI/yr.", "",
     f"Orgs (stage 2, lam_u {spec2.lam_u:.3g}, lam_v {spec2.lam_v:.3g}): common m {mm:+.2f} (Hessian SE {mse:.2f}).", "",
     f"Deviations u alone: {(np.abs(t.dev) > 2 * t.dev_se).sum()} of {len(t)} orgs beyond 2 SE of the common m.", "",
     "| organization | models | cells | m + u shrunken | 2 SE | u alone | 2 SE | unshrunk | 2 SE |", "|---|---|---|---|---|---|---|---|---|"]
L += [f"| {r.org} | {r.models} | {r.cells} | {r.eff:+.2f} | {2 * r.se:.2f} | {r.dev:+.2f} | {2 * r.dev_se:.2f} | "
      f"{r.eff_unshrunk:+.2f} | {2 * r.se_unshrunk:.2f} |" for r in t[::-1].itertuples()]
open("out/cm13.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
