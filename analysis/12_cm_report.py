"""Calibrated results for the common-mechanism test: observed statistics (analysis/10_cm_fit.py) against
their bootstrap distributions (analysis/11_cm_boot.py). Writes out/cm12.md."""
import os
import numpy as np, pandas as pd

fits = pd.read_csv("out/cm10_fits.csv")
kink = pd.read_csv("out/cm10_kink.csv")
KINKS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0]
NULL_RUNS = [("null_primary", "primary", "parametric"), ("null_primary_wild", "primary", "wild, clustered by model"),
             ("null_lam_g1", "lam_g=1", "parametric"), ("null_wloc1e-4", "w_loc=1e-4", "parametric")]


def load(run):
    p = f"out/boot11_{run}.csv"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    b = pd.read_csv(p)
    return b if len(b) else None


def observed(variant):
    g = fits[fits.variant == variant]
    return g.signed_root.max(), g["T"].max(), g.set_index("tau")


def kcol(k):
    return "T_tau4" if k == 0.0 else f"T_kink{k:+g}"


def pval(null, obs):
    return (1 + int((null >= obs - 1e-12).sum())) / (len(null) + 1)


L = ["# Common-mechanism test, calibrated (analysis/12_cm_report.py)", "",
     "S = max over tau of sign(m) sqrt(T_tau) (one-sided: the gap helps). Tmax = max over tau of T_tau (two-sided). "
     "p-values are (1 + #{null >= observed}) / (B + 1).", "",
     "## Test of m = 0", "",
     "| null run | variant | noise | B | observed S | p (S) | observed Tmax | p (Tmax) | null 95% of S | null 99% of S | fits with cost certificate > 1e-7 | worst cost certificate |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for run, variant, noise in NULL_RUNS:
    b = load(run)
    if b is None:
        L.append(f"| {run} | {variant} | {noise} | not run | | | | | | | | |")
        continue
    S, Tm, _ = observed(variant)
    L.append(f"| {run} | {variant} | {noise} | {len(b)} | {S:+.2f} | {pval(b.S, S):.4f} | {Tm:.2f} | {pval(b.Tmax, Tm):.4f} | "
             f"{b.S.quantile(0.95):.2f} | {b.S.quantile(0.99):.2f} | {int((b.cert_dec > 1e-7).sum())} | {b.cert_dec.max():.1e} |")

b0 = load("null_primary")
if b0 is not None:
    L += ["", "Null distribution of m at tau = 4 (ECI per year, primary, parametric): "
          f"mean {b0.m_tau4.mean():+.2f}, sd {b0.m_tau4.std():.2f}. Observed "
          f"{fits[(fits.variant == 'primary') & (fits.tau == 4.0)].m_eci_per_yr.iloc[0]:+.2f}."]
    cold = b0.cold_diff.dropna()
    L += [f"Cold-start check on {len(cold)} replicates: largest |cost difference| {cold.abs().max():.1e}."]

L += ["", "## Kink location (placebo)", "",
      "Observed T by kink location (tau = 4, primary): " +
      ", ".join(f"{k:+g}: {kink[(kink.variant == 'primary') & (kink.kink == k)]['T'].iloc[0]:.2f}" for k in KINKS) + ".", ""]
obs = {k: kink[(kink.variant == "primary") & (kink.kink == k)]["T"].iloc[0] for k in KINKS}
obs_arg = max(obs, key=obs.get)
obs_gap = obs[-1.0] - obs[0.0]
for run, label in [("null_primary", "no gap effect (null)"), ("alt_primary", "the fitted gap effect with its kink at 0 (tau = 4)")]:
    b = load(run)
    if b is None:
        L.append(f"- Simulated from {label}: not run.")
        continue
    T = b[[kcol(k) for k in KINKS]].to_numpy()
    arg = np.array(KINKS)[T.argmax(axis=1)]
    share = {k: float((arg == k).mean()) for k in KINKS}
    diff = b[kcol(-1.0)] - b[kcol(0.0)]
    L.append(f"- Simulated from {label} (B = {len(b)}): the profile peaks at " +
             ", ".join(f"{k:+g} in {share[k]:.0%}" for k in KINKS) +
             f". T(kink -1) minus T(kink 0) is at least the observed {obs_gap:.2f} in {(diff >= obs_gap).mean():.1%} of replicates "
             f"(mean {diff.mean():+.2f}, 95th percentile {diff.quantile(0.95):+.2f}).")
    if run == "alt_primary":
        m_true = fits[(fits.variant == "primary") & (fits.tau == 4.0)].m_eci_per_yr.iloc[0]
        L.append(f"- Under the fitted effect (true m {m_true:+.2f} ECI/yr), the estimate at tau = 4 averages "
                 f"{b.m_tau4.mean():+.2f} (sd {b.m_tau4.std():.2f}); the test rejects at 5% in "
                 f"{(b.S > load('null_primary').S.quantile(0.95)).mean():.0%} of replicates (power at this effect size)."
                 if load("null_primary") is not None else "")
L += ["", f"Observed peak: kink {obs_arg:+g}."]
open("out/cm12.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
