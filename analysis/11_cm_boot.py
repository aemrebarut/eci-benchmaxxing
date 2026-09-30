"""Bootstrap calibration of the common-mechanism test (analysis/10_cm_fit.py).

Null runs simulate from a fitted no-gap model and refit null and alternatives exactly as on the real data,
giving the null distribution of the one-sided statistic S = max over tau of sign(m) sqrt(T_tau) and of
the kink-location profile. The placebo run simulates from the fitted alternative (kink at 0, tau = 4) and
asks how often the profile would peak before benchmark release, as it does on the real data.

Simulated scores are clipped to [0.001, 0.999] like the real ones (cm.CLIP).
Noise, "param": y* = mu + e, e ~ N(0, phi_j max(mu (1 - mu), 0.01)), phi_j the benchmark's Pearson
dispersion from the fitted model, shrunk toward the pooled value (10 pseudo-cells), times n / (n - edf).
Noise, "wild": y* = mu + s_i r_ij sqrt(n / (n - edf)), s_i = +-1 per model (keeps within-model residual
correlation, which the parametric version assumes away).

Usage: python analysis/11_cm_boot.py RUN B   (RUN in RUNS below). Appends one row per replicate to
out/boot11_<RUN>.csv, so a partial run keeps its replicates."""
import os, sys, time, csv
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
import cm

TAUS = [0.25, 0.5, 1.0, 2.0, 4.0, np.inf]
X0S = [-2.0, -1.0, -0.5, 0.5, 1.0]  # the kink at 0 with tau = 4 is already in the tau loop
KINK_TAU = 4.0
PRIMARY = dict(rho=False, lam_g=float(open("out/cm10_lam_g.txt").read()), w_loc=1e-6)  # as in 10_cm_fit.py
RUNS = {  # name: (spec overrides, noise, simulate from)
    "null_primary": ({}, "param", "null"),
    "null_primary_wild": ({}, "wild", "null"),
    "null_lam_g1": (dict(lam_g=1.0), "param", "null"),
    "null_wloc1e-4": (dict(w_loc=1e-4), "param", "null"),
    "alt_primary": ({}, "param", "alt"),
}
df, ORG = cm.load()
RUN = sys.argv[1]
OVER, NOISE, FROM = RUNS[RUN]
D = cm.Data(df, y_col="y_fit", org=ORG)
SPEC0 = cm.Spec(gap=False, **{**PRIMARY, **OVER})
FIT0 = cm.fit(D, SPEC0)
FIT1 = cm.fit(D, SPEC0.replace(gap=True, tau=KINK_TAU))
GEN = FIT0 if FROM == "null" else FIT1
MU = cm.predict(GEN["x"], D, GEN["spec"], GEN["lay"])
R = D.y - MU
INFL = D.n / (D.n - cm.edf(GEN, D))
V = np.maximum(MU * (1 - MU), 0.01)
pear = pd.Series(R ** 2 / V).groupby(D.bi)
phi_pool = float((R ** 2 / V).mean())
phi = ((pear.sum() + 10 * phi_pool) / (pear.size() + 10)).reindex(range(D.B)).to_numpy()
SD = np.sqrt(phi[D.bi] * V * INFL)


def embed(x_from, lay_from, lay_to):
    x = np.zeros(lay_to.n)
    for k, s in lay_from.sl.items():
        x[lay_to.sl[k]] = x_from[s]
    return x


def replicate(b):
    rng = np.random.default_rng(1000 + b)
    if NOISE == "param":
        ys = MU + rng.normal(0, SD)
    else:
        s = rng.choice([-1.0, 1.0], D.M)
        ys = MU + s[D.mi] * R * np.sqrt(INFL)
    ys = np.clip(ys, cm.CLIP, 1 - cm.CLIP)
    d = cm.Data(df.assign(ysim=ys), y_col="ysim", org=ORG)
    t = time.time()
    o0 = cm.fit(d, SPEC0, x0=FIT0["x"])
    sigma2 = o0["sse"] / (d.n - cm.edf(o0, d))
    row = dict(rep=b, cost0=o0["cost"], sigma2=sigma2, cert=o0["gn_dp"], cert_m=0.0, status0=o0["status"],
               cert_dec=o0["gn_dec"], max_iters=o0["iters"])
    stats = []
    jobs = [(tau, 0.0) for tau in TAUS] + [(KINK_TAU, k) for k in X0S]
    for tau, k in jobs:
        spec1 = SPEC0.replace(gap=True, tau=tau, x0=k)
        o1 = cm.fit(d, spec1, x0=embed(o0["x"], o0["lay"], cm.Layout(d, spec1)))
        _, b1 = cm.eci_scale(o1, d)
        T = 2 * (o0["cost"] - o1["cost"]) / sigma2
        sr = np.sign(o1["m"]) * np.sqrt(max(T, 0.0))
        lab = f"tau{tau:g}" if k == 0.0 else f"kink{k:+g}"
        row[f"T_{lab}"], row[f"m_{lab}"] = T, o1["m"] * b1
        row["cert"], row["cert_m"] = max(row["cert"], o1["gn_dp"]), max(row["cert_m"], o1["gn_step_m"])
        row["cert_dec"], row["max_iters"] = max(row["cert_dec"], o1["gn_dec"]), max(row["max_iters"], o1["iters"])
        if k == 0.0:
            stats.append(sr)
    row["S"] = max(stats)
    row["Tmax"] = max(row[f"T_tau{t:g}"] for t in TAUS)
    if b < 10:  # cold-start check at tau = 4
        oc = cm.fit(d, SPEC0.replace(gap=True, tau=KINK_TAU))
        o1 = cm.fit(d, SPEC0.replace(gap=True, tau=KINK_TAU), x0=embed(o0["x"], o0["lay"], cm.Layout(d, SPEC0.replace(gap=True))))
        row["cold_diff"] = oc["cost"] - o1["cost"]
    row["secs"] = time.time() - t
    return row


if __name__ == "__main__":
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out = f"out/boot11_{RUN}.csv"
    done = set(pd.read_csv(out).rep) if os.path.exists(out) else set()
    todo = [b for b in range(B) if b not in done]
    print(f"{RUN}: noise {NOISE}, from {FROM}, {len(todo)} replicates to run ({len(done)} done)", flush=True)
    t0, header = time.time(), not os.path.exists(out)
    cols = None
    with Pool(3) as pool, open(out, "a", newline="") as fh:
        for i, row in enumerate(pool.imap_unordered(replicate, todo), 1):
            if cols is None:
                cols = list(pd.read_csv(out, nrows=0).columns) if not header else \
                    ["rep", "S", "Tmax", "cost0", "sigma2", "cert", "cert_m", "cert_dec", "max_iters", "status0", "cold_diff", "secs"] + \
                    sorted(k for k in row if k.startswith(("T_", "m_")))
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                if header:
                    w.writeheader()
            w.writerow(row); fh.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
