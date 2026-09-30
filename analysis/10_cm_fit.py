"""Common-mechanism fits on the full data: null (no gap term) against one shared gap effect, with rivals.

For each variant: the null fit, the alternative for each saturation tau, the statistic
T_tau = 2 (cost_null - cost_alt) / sigma2_null and its signed root, m in ECI per year of gap just after the
kink, and the effect at gaps of 1 and 2 years. Then the kink-location profile (placebo) and the org and
model heterogeneity (stage 2, random-effect style ridges tuned by Fellner-Schall).
Calibration of T is in analysis/11_cm_boot.py; nothing here is a p-value."""
import os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
import cm

TAUS = [0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
X0S = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
# Location ridge 1e-6 (effectively none, Fable's recommendation): 1e-4 pulls strong models down and the gap term
# compensates, inflating the effect; it is kept as a sensitivity. rho is off: its variance estimate is zero.
BASE = dict(rho=False, w_loc=1e-6)
df, ORG = cm.load()
# The date-slope ridge is set as a random-effect variance (Fellner-Schall) on the null model.
_, _tuned, _ = cm.tune(cm.Data(df, y_col="y_fit", org=ORG), cm.Spec(gap=False, **BASE), ("gamma",), max_rounds=300)
LAM_G = round(_tuned.lam_g, 2)
BASE["lam_g"] = LAM_G
VARIANTS = {  # name: (y column, Spec overrides)
    "primary": ("y_fit", {}),
    "lam_g=1": ("y_fit", dict(lam_g=1.0)), "lam_g=100": ("y_fit", dict(lam_g=100.0)),
    "no date slopes": ("y_fit", dict(gamma=False)),
    "rho on (lam_r=3)": ("y_fit", dict(rho=True, lam_r=3.0)),
    "no selection term": ("y_fit", dict(sel=False)),
    "unclipped y": ("y", {}),
    "w_loc=1e-4": ("y_fit", dict(w_loc=1e-4)),
    "no rivals, no selection": ("y_fit", dict(gamma=False, sel=False)),
}


def data(ycol):
    return cm.Data(df, y_col=ycol, org=ORG)


def run_variant(name):
    ycol, over = VARIANTS[name]
    d = data(ycol)
    spec0 = cm.Spec(gap=False, **{**BASE, **over})
    t = time.time()
    o0 = cm.fit(d, spec0)
    n = float(d.n)
    edf0 = cm.edf(o0, d)
    sigma2 = o0["sse"] / (n - edf0)
    _, b0 = cm.eci_scale(o0, d)
    csel = o0["x"][o0["lay"].sl["csel"]][0] * b0 if "csel" in o0["lay"].sl else np.nan
    rows = []
    for tau in TAUS:
        spec1 = spec0.replace(gap=True, tau=tau)
        x0 = np.zeros(cm.Layout(d, spec1).n)
        L0, L1 = o0["lay"], cm.Layout(d, spec1)
        for k, s in L0.sl.items():
            x0[L1.sl[k]] = o0["x"][s]
        o1 = cm.fit(d, spec1, x0=x0)
        o1c = cm.fit(d, spec1)  # cold start as a multi-start check
        _, b1 = cm.eci_scale(o1, d)
        T = 2 * (o0["cost"] - o1["cost"]) / sigma2
        rows.append(dict(variant=name, tau=tau, m=o1["m"], m_eci_per_yr=o1["m"] * b1,
                         eff_1yr=o1["m"] * b1 * cm.h(1.0, tau), eff_2yr=o1["m"] * b1 * cm.h(2.0, tau),
                         eff_max=o1["m"] * b1 * (tau if np.isfinite(tau) else np.nan),
                         T=T, signed_root=np.sign(o1["m"]) * np.sqrt(max(T, 0.0)),
                         cold_cost_diff=o1c["cost"] - o1["cost"], gn_dp=max(o1["gn_dp"], o0["gn_dp"]),
                         gn_m=o1["gn_step_m"], status=o1["status"], csel_eci=csel, sigma=np.sqrt(sigma2),
                         edf_null=edf0, cost_null=o0["cost"], cost_alt=o1["cost"]))
    return rows, time.time() - t


def run_kink(job):
    name, tau, x0k = job
    ycol, over = VARIANTS[name]
    d = data(ycol)
    spec0 = cm.Spec(gap=False, **{**BASE, **over})
    o0 = cm.fit(d, spec0)
    sigma2 = o0["sse"] / (d.n - cm.edf(o0, d))
    o1 = cm.fit(d, spec0.replace(gap=True, tau=tau, x0=x0k))
    _, b1 = cm.eci_scale(o1, d)
    T = 2 * (o0["cost"] - o1["cost"]) / sigma2
    return dict(variant=name, tau=tau, kink=x0k, m=o1["m"], m_eci_per_yr=o1["m"] * b1, T=T,
                signed_root=np.sign(o1["m"]) * np.sqrt(max(T, 0.0)), gn_dp=o1["gn_dp"], status=o1["status"])


def stage2(tau):
    """Org and model deviations of the gap coefficient: psi_i = m + u_org + v_i, ridges tuned as random effects."""
    d = data("y_fit")
    spec = cm.Spec(gap=True, tau=tau, org=True, model_dev=True, lam_u=1.0, lam_v=10.0, **BASE)
    o, spec, rnd = cm.tune(d, spec, blocks=("u", "v"), max_rounds=200)
    lay = o["lay"]
    _, b = cm.eci_scale(o, d)
    u, v = o["x"][lay.sl["u"]], o["x"][lay.sl["v"]]
    n_models = pd.Series(d.oi).value_counts().reindex(range(len(d.orgs))).fillna(0).astype(int)
    n_cells = pd.Series(d.oi[d.mi]).value_counts().reindex(range(len(d.orgs))).fillna(0).astype(int)
    orgs = pd.DataFrame(dict(org=d.orgs, n_models=n_models.values, n_cells=n_cells.values,
                             u_eci_per_yr=u * b, total_eci_per_yr=(o["m"] + u) * b)).sort_values("total_eci_per_yr")
    sd_u = np.sqrt(o["sigma2"] / spec.lam_u) * b
    sd_v = np.sqrt(o["sigma2"] / spec.lam_v) * b
    info = dict(tau=tau, lam_u=spec.lam_u, lam_v=spec.lam_v, rounds=rnd, m_eci_per_yr=o["m"] * b,
                sd_org_eci_per_yr=sd_u, sd_model_eci_per_yr=sd_v, max_abs_v_eci=float(np.abs(v).max() * b),
                status=o["status"], gn_dp=o["gn_dp"], cost=o["cost"])
    return info, orgs


if __name__ == "__main__":
    t0 = time.time()
    with Pool(3) as pool:
        res = pool.map(run_variant, list(VARIANTS))
    rows = [r for rr, _ in res for r in rr]
    fits = pd.DataFrame(rows)
    fits.to_csv("out/cm10_fits.csv", index=False)
    open("out/cm10_lam_g.txt", "w").write(f"{LAM_G}\n")
    print(f"variants done {time.time() - t0:.0f}s", flush=True)
    best_tau = fits[fits.variant == "primary"].sort_values("T")["tau"].iloc[-1]
    with Pool(3) as pool:
        kink = pd.DataFrame(pool.map(run_kink, [(v, best_tau, k) for v in ["primary", "no date slopes"] for k in X0S]))
    kink.to_csv("out/cm10_kink.csv", index=False)
    print(f"kink profile done {time.time() - t0:.0f}s", flush=True)
    info, orgs = stage2(best_tau)
    orgs.to_csv("out/cm10_orgs.csv", index=False)
    print(f"stage 2 done {time.time() - t0:.0f}s", flush=True)

    f = lambda x: f"{x:.3g}" if isinstance(x, float) else str(x)
    L = ["# Common-mechanism fits (analysis/10_cm_fit.py)", "",
         f"Cells {len(df)}, models {df.Model.nunique()}, benchmarks {df.benchmark.nunique()}. y is Epoch's fit target (clipped to [0.001, 0.999]) unless stated. "
         f"Primary: date-slope rival with lam_g = {LAM_G} (Fellner-Schall), rho off, selection term on, w_loc = 1e-6. "
         "T = 2 (cost_null - cost_alt) / sigma2_null; not calibrated here (see the bootstrap).", "",
         "## Best tau per variant", "",
         "| variant | best tau | m (ECI/yr) | effect at 1 yr (ECI) | effect at 2 yr | T | signed root | T at tau=inf | selection term (ECI per unit E[max]) | worst certificate | cold start cost diff |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, g in fits.groupby("variant", sort=False):
        r = g.sort_values("T").iloc[-1]
        ti = g[np.isinf(g.tau)]["T"].iloc[0]
        L.append(f"| {v} | {f(r.tau)} | {r.m_eci_per_yr:+.2f} | {r.eff_1yr:+.2f} | {r.eff_2yr:+.2f} | {r['T']:.2f} | {r.signed_root:+.2f} | "
                 f"{ti:.2f} | {r.csel_eci:+.2f} | {g.gn_dp.max():.1e} | {g.cold_cost_diff.abs().max():.1e} |")
    L += ["", "## Primary, every tau", "", "| tau | m (ECI/yr) | effect at 1 yr | effect at 2 yr | T | status |", "|---|---|---|---|---|---|"]
    for r in fits[fits.variant == "primary"].itertuples():
        L.append(f"| {f(r.tau)} | {r.m_eci_per_yr:+.2f} | {r.eff_1yr:+.2f} | {r.eff_2yr:+.2f} | {r.T:.2f} | {r.status} |")
    L += ["", f"## Kink-location profile at tau = {f(best_tau)} (placebo)", "",
          "If the effect comes from benchmarks that already existed when the model was trained, T should peak at a kink of 0. "
          "A peak below 0 means the curvature is not tied to benchmark release.", "",
          "| variant | kink (years) | m (ECI/yr) | T |", "|---|---|---|---|"]
    for r in kink.itertuples():
        L.append(f"| {r.variant} | {r.kink:+.1f} | {r.m_eci_per_yr:+.2f} | {r.T:.2f} |")
    L += ["", f"## Stage 2: organization and model deviations (tau = {f(best_tau)})", "",
          f"psi_i = m + u_org + v_i with ridges tuned as random effects (Fellner-Schall, {info['rounds']} rounds): "
          f"lam_u = {info['lam_u']:.3g}, lam_v = {info['lam_v']:.3g}. Common m = {info['m_eci_per_yr']:+.2f} ECI/yr; "
          f"implied sd across organizations {info['sd_org_eci_per_yr']:.2f} ECI/yr, across models within organization "
          f"{info['sd_model_eci_per_yr']:.2f} ECI/yr (largest shrunken model deviation {info['max_abs_v_eci']:.2f}). "
          f"Status {info['status']}, certificate {info['gn_dp']:.1e}. Shrunken values, not a ranking.", "",
          "| organization | models | cells | m + u (ECI/yr) |", "|---|---|---|---|"]
    for r in orgs[orgs.n_models >= 3].itertuples():
        L.append(f"| {r.org} | {r.n_models} | {r.n_cells} | {r.total_eci_per_yr:+.2f} |")
    L += ["", f"Organizations with fewer than 3 models are omitted from the table (they are in out/cm10_orgs.csv). Total {time.time() - t0:.0f}s."]
    open("out/cm10.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
