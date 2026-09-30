"""Basic pre/post model: Epoch's ECI model plus one shift per (model release year, organization) on the cells
where the benchmark was public when the model was released.

    y_ij ~ sigmoid(a_j (C_i + delta_g(i) post_ij - D_j)),  post_ij = 1(gap_ij > 0),  g(i) = (year_i, org_i)

This is tau_{year,org,pre} 1(pre) + tau_{year,org,post} 1(post) with tau_pre absorbed by the model's own C_i, so
only delta = tau_post - tau_pre is identified, and only for groups with at least one model that has cells on
both sides of its release. Other groups get no delta. Year is the year of the model's modal release date.

Objective: Epoch's (least squares on the clipped score, Winogrande slope pinned to 1, L2 of 0.1 / n_params on
C, D and slopes, bounds as in eci.fitting) plus the ridge 0.5 lam sum delta^2. lam is chosen by 10-fold CV over
cells (same folds as analysis/09_cm_cv.py). Solver: scipy least_squares, trf with the exact trust-region solver
on the dense Jacobian. delta is reported on the ECI scale (times Epoch's affine map slope b).

SEs: sandwich clustered by model, H^-1 (sum over models of s_g s_g') H^-1 with H = J'J + penalty; and the
model-based sigma^2 H^-1. With the ridge these are SEs of shrunken estimates (prediction error variance).
Usage: python analysis/17_prepost.py [year_org | org]. "org" drops the year: one delta per organization.
Writes out/prepost17<suffix>.md, _groups.csv, _cv.csv and out/figs/prepost17<suffix>.png, suffix "" for
year_org and "_org" for org."""
import os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy import stats
from eci.fitting import (sigmoid, DEFAULT_ANCHOR_BENCHMARK, DEFAULT_ANCHOR_DISCRIMINABILITY, DEFAULT_ANCHOR_MODEL_LOW,
                         DEFAULT_ANCHOR_ECI_LOW, DEFAULT_ANCHOR_MODEL_HIGH, DEFAULT_ANCHOR_ECI_HIGH)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cm

LAMS = [1e-3, 1e-2, 0.1, 0.3, 1, 3, 10, 30, 100, 1e3]
MODE = sys.argv[1] if len(sys.argv) > 1 else "year_org"
assert MODE in ("year_org", "org"), MODE
SUF = "" if MODE == "year_org" else "_org"
GLABEL = "(year, organization)" if MODE == "year_org" else "organization"
EPOCH_REG = 0.1

df, ORG = cm.load()
df["org"] = df.Model.map(ORG).fillna("unknown")
mdate = df.groupby("Model").date.agg(lambda s: s.mode().iloc[0])
df["year"] = df.Model.map(mdate).dt.year
df["post"] = (df.gap > 0).astype(float)
models = list(dict.fromkeys(df.Model)); benches = list(dict.fromkeys(df.benchmark))
M, B, n = len(models), len(benches), len(df)
mi = df.Model.map({m: i for i, m in enumerate(models)}).to_numpy()
bi = df.benchmark.map({b: j for j, b in enumerate(benches)}).to_numpy()
y = df.y_fit.to_numpy()
anchor = benches.index(DEFAULT_ANCHOR_BENCHMARK)
free_a = np.array([j for j in range(B) if j != anchor])
lo_i, hi_i = models.index(DEFAULT_ANCHOR_MODEL_LOW), models.index(DEFAULT_ANCHOR_MODEL_HIGH)

both = df.groupby("Model").post.agg(lambda s: 0 < s.mean() < 1)
df["group"] = df.year.astype(str) + " | " + df.org if MODE == "year_org" else df.org
ident = sorted(df[df.Model.map(both)].group.unique())
gi_of = {g: k for k, g in enumerate(ident)}
gi = df.group.map(gi_of).fillna(-1).astype(int).to_numpy()
K = len(ident)
n_ep = M + B + (B - 1)
P = n_ep + K
has = gi >= 0


def unpack(x):
    C, D = x[:M], x[M:M + B]
    a = np.full(B, DEFAULT_ANCHOR_DISCRIMINABILITY); a[free_a] = x[M + B:n_ep]
    return C, D, a, x[n_ep:]


def eta_parts(x):
    C, D, a, d = unpack(x)
    shift = np.where(has, d[np.maximum(gi, 0)], 0.0) * df.post.to_numpy() if len(d) else 0.0
    lin = C[mi] + shift - D[bi]
    return a, lin, a[bi] * lin


def resid(x, w, lam, use_delta=True):
    _, _, e = eta_parts(x)
    r = np.sqrt(w) * (sigmoid(e) - y)
    parts = [r, np.sqrt(EPOCH_REG / n_ep) * x[:n_ep]]
    if use_delta:
        parts.append(np.sqrt(lam) * x[n_ep:])
    return np.concatenate(parts)


def jac(x, w, lam, use_delta=True):
    a, lin, e = eta_parts(x)
    s = sigmoid(e); ds = np.sqrt(w) * s * (1 - s)
    rows = np.arange(n)
    J = np.zeros((n + n_ep + (K if use_delta else 0), P if use_delta else n_ep))
    J[rows, mi] = ds * a[bi]
    J[rows, M + bi] = -ds * a[bi]
    pos = np.full(B, -1); pos[free_a] = np.arange(B - 1)
    ok = pos[bi] >= 0
    J[rows[ok], M + B + pos[bi][ok]] = ds[ok] * lin[ok]
    if use_delta:
        hk = has & (df.post.to_numpy() > 0)
        J[rows[hk], n_ep + gi[hk]] = ds[hk] * a[bi][hk]
    J[n + np.arange(n_ep), np.arange(n_ep)] = np.sqrt(EPOCH_REG / n_ep)
    if use_delta:
        J[n + n_ep + np.arange(K), n_ep + np.arange(K)] = np.sqrt(lam)
    return J


LB = np.concatenate([np.full(M + B, -10.0), np.full(B - 1, 0.1), np.full(K, -10.0)])
UB = np.concatenate([np.full(M + B, 10.0), np.full(B - 1, 10.0), np.full(K, 10.0)])


def fit(w, lam, x0, use_delta=True):
    p = P if use_delta else n_ep
    r = least_squares(resid, x0[:p], jac=jac, bounds=(LB[:p], UB[:p]), method="trf", tr_solver="exact",
                      args=(w, lam, use_delta), ftol=1e-13, xtol=1e-13, gtol=1e-11, max_nfev=5000)
    return r


def scale(x):
    C = x[:M]
    b = (DEFAULT_ANCHOR_ECI_HIGH - DEFAULT_ANCHOR_ECI_LOW) / (C[hi_i] - C[lo_i])
    return DEFAULT_ANCHOR_ECI_LOW - b * C[lo_i], b


rng0 = np.random.default_rng(42)
x_init = np.concatenate([rng0.normal(0, 0.1, M), rng0.normal(0, 0.1, B), np.ones(B - 1), np.zeros(K)])
ones = np.ones(n)
t0 = time.time()
base = fit(ones, 0.0, x_init, use_delta=False)
a0, b0 = scale(base.x)
rep = pd.read_csv("out/eci_repro.csv").set_index("Model").eci
eci0 = pd.Series(a0 + b0 * base.x[:M], index=models)
print(f"no-delta fit: status {base.status}, cost {base.cost:.6f}, {time.time() - t0:.1f}s; max |ECI - Epoch's lsmr fit| "
      f"{(eci0 - rep.reindex(models)).abs().max():.3f}", flush=True)
x_base = np.concatenate([base.x, np.zeros(K)])

# CV over lam, with the no-delta model as the reference
folds = np.random.default_rng(7).integers(0, 10, n)
cv = []
full = {}
for lam in LAMS + [np.inf]:
    t = time.time()
    if np.isfinite(lam):
        full[lam] = fit(ones, lam, x_base).x
    for f in range(10):
        w = (folds != f).astype(float)
        if np.isfinite(lam):
            xf = fit(w, lam, full[lam]).x
        else:
            xf = np.concatenate([fit(w, 0.0, base.x, use_delta=False).x, np.zeros(K)])
        _, _, e = eta_parts(xf)
        te = folds == f
        cv.append(dict(lam=lam, fold=f, sse=float(((sigmoid(e[te]) - y[te]) ** 2).sum()), n=int(te.sum())))
    print(f"lam {lam:g}: cv sse {sum(c['sse'] for c in cv if c['lam'] == lam):.5f} ({time.time() - t:.1f}s)", flush=True)
cv = pd.DataFrame(cv)
cv.to_csv(f"out/prepost17{SUF}_cv.csv", index=False)
tab = cv.groupby("lam").sse.sum()
ref = cv[cv.lam == np.inf].set_index("fold").sse
better = {lam: int((cv[cv.lam == lam].set_index("fold").sse < ref).sum()) for lam in LAMS}
lam_cv = float(tab.drop(np.inf).idxmin())
print(f"CV choice lam {lam_cv:g}", flush=True)


def analyse(x, lam):
    """ECI-scale deltas with clustered and model-based SEs."""
    _, b = scale(x)
    J = jac(x, ones, lam)
    r = resid(x, ones, lam)
    H = J.T @ J
    Hi = np.linalg.inv(H)
    Jd, rd = J[:n], r[:n]
    S = np.zeros((P, len(models)))
    np.add.at(S.T, mi, Jd * rd[:, None])
    Vc = Hi @ (S @ S.T) @ Hi
    edf = float(np.trace(Hi @ (Jd.T @ Jd)))
    Vm = float(rd @ rd) / (n - edf) * Hi
    sl = slice(n_ep, P)
    return b * x[sl], b ** 2 * Vc[sl, sl], b ** 2 * Vm[sl, sl], b, edf


def wmean(d, V, idx):
    """Inverse-variance weighted mean over idx (weights from the diagonal), SE from the full covariance."""
    w = np.zeros(len(d)); w[idx] = 1 / np.diag(V)[idx]; w /= w.sum()
    return float(w @ d), float(np.sqrt(w @ V @ w))


def wald_equal(d, V, idx):
    idx = np.asarray(idx)
    if len(idx) < 2:
        return np.nan, 0
    Cm = np.zeros((len(idx) - 1, len(d)))
    for k in range(len(idx) - 1):
        Cm[k, idx[k]], Cm[k, idx[-1]] = 1, -1
    c = Cm @ d
    W = float(c @ np.linalg.solve(Cm @ V @ Cm.T, c))
    return W, len(idx) - 1


info = pd.DataFrame(dict(group=ident))
info["year"] = [int(g.split(" | ")[0]) for g in ident] if MODE == "year_org" else "all"
info["org"] = [g.split(" | ")[1] for g in ident] if MODE == "year_org" else ident
g_df = df[has]
info["models"] = [g_df[g_df.group == g].Model.nunique() for g in ident]
info["models_both"] = [int(both[g_df[g_df.group == g].Model.unique()].sum()) for g in ident]
info["cells_pre"] = [int(((g_df.group == g) & (g_df.post == 0)).sum()) for g in ident]
info["cells_post"] = [int(((g_df.group == g) & (g_df.post == 1)).sum()) for g in ident]

L = [f"# Pre/post shift by {GLABEL} (analysis/17_prepost.py {MODE})", "",
     f"{K} {GLABEL} groups identified (at least one model with cells on both sides of its release), "
     f"{int(has.sum())} of {n} cells in them. Unidentified groups: "
     f"{sorted(set(df.group) - set(ident))}.", "",
     f"No-delta fit (exact solver) against Epoch's published-code fit: max ECI difference {(eci0 - rep.reindex(models)).abs().max():.3f}.", "",
     "## Ridge on delta, 10-fold CV", "",
     "| lam | CV SSE | minus no-delta CV SSE | folds better than no-delta |", "|---|---|---|---|"]
L += [f"| {lam:g} | {tab[lam]:.5f} | {tab[lam] - tab[np.inf]:+.5f} | {better[lam]} of 10 |" for lam in LAMS]
L += [f"| no delta | {tab[np.inf]:.5f} | 0 | |", "", f"CV choice: lam = {lam_cv:g}.", ""]

res = {}
for label, lam in [("CV ridge", lam_cv), ("near-unpenalized", LAMS[0])]:
    d, Vc, Vm, b, edf = analyse(full[lam], lam)
    res[label] = (d, Vc, Vm)
    se_c, se_m = np.sqrt(np.diag(Vc)), np.sqrt(np.diag(Vm))
    info[f"delta_{label}"], info[f"se_cl_{label}"], info[f"se_mb_{label}"] = d, se_c, se_m
    L += [f"## {label} (lam = {lam:g}), ECI points (b = {b:.2f} ECI per raw unit, edf {edf:.1f})", ""]
    for Vname, V in [("clustered by model", Vc), ("model-based", Vm)]:
        m_all, s_all = wmean(d, V, np.arange(K))
        W, k = wald_equal(d, V, np.arange(K))
        L.append(f"- SE {Vname}: weighted mean delta {m_all:+.2f} +- {s_all:.2f} (z {m_all / s_all:+.2f}, "
                 f"p {2 * stats.norm.sf(abs(m_all / s_all)):.3f}); all groups equal: Wald {W:.1f} on {k} df, "
                 f"p {stats.chi2.sf(W, k):.3f}.")
        if MODE != "year_org":
            continue
        ym = []
        for yr in sorted(info.year.unique()):
            idx = np.flatnonzero(info.year == yr)
            my, sy = wmean(d, V, idx)
            Wy, ky = wald_equal(d, V, idx)
            ym.append(f"{yr}: {my:+.2f} +- {sy:.2f} (orgs equal: p {stats.chi2.sf(Wy, ky):.3f})" if ky else f"{yr}: {my:+.2f} +- {sy:.2f}")
        L.append("  - by year: " + "; ".join(ym) + ".")
    L.append("")

L += ["## Groups (ECI points; se_cl clustered by model, se_mb model-based)", "",
      "| year | organization | models | with both | cells pre | cells post | delta (CV ridge) | se_cl | se_mb | delta (near-unpenalized) | se_cl | se_mb |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in info.sort_values(["year", "org"]).iterrows():
    L.append(f"| {r.year} | {r.org} | {r.models} | {r.models_both} | {r.cells_pre} | {r.cells_post} | "
             + " | ".join(f"{r['delta_' + k]:+.2f} | {r['se_cl_' + k]:.2f} | {r['se_mb_' + k]:.2f}" for k in ("CV ridge", "near-unpenalized")) + " |")
info.round(4).to_csv(f"out/prepost17{SUF}_groups.csv", index=False)
open(f"out/prepost17{SUF}.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# Figure: delta by group with 2 SE, both fits. The clustered SE is too small with one or two models per group,
# so the bars use the larger of the clustered and model-based SEs
yrs = sorted(info.year.unique())
cols = {2023: "#8172B3", 2024: "#55A868", 2025: "#C44E52", 2026: "#4C72B0", "all": "#4C72B0"}
t = info.sort_values(["year", "cells_post"], ascending=[True, False]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 0.28 * len(t) + 1.5))
yy = np.arange(len(t))[::-1]
for label, off, mk in [("near-unpenalized", 0.17, "o"), ("CV ridge", -0.12, "s")]:
    d = t[f"delta_{label}"]; s = np.maximum(t[f"se_cl_{label}"], t[f"se_mb_{label}"])
    for yr in yrs:
        k = (t.year == yr).to_numpy()
        ax.errorbar(d[k], yy[k] + off, xerr=2 * s[k], fmt=mk, ms=4 if mk == "o" else 4.5,
                    color=cols[yr], mfc="white" if label == "near-unpenalized" else cols[yr], lw=1.1 if label == "CV ridge" else 0.8,
                    alpha=1 if label == "CV ridge" else 0.6)
ax.axvline(0, color="0.4", lw=0.8)
ax.set_yticks(yy, [f"{r.year if MODE == 'year_org' else ''} {r.org[:26]} ({r.models_both}/{r.models}, {r.cells_pre}/{r.cells_post})" for r in t.itertuples()], fontsize=7.5)
ax.set_xlabel("delta = tau_post - tau_pre (ECI points), +-2 SE (larger of clustered and model-based)")
ax.set_title(f"Pre/post shift by {GLABEL}. Filled: ridge, lam = {lam_cv:g} (CV); open: lam = {LAMS[0]:g}\n"
             "labels: (models with cells on both sides / models, cells pre / post)", fontsize=9)
ax.set_xlim(-25, 25)
fig.tight_layout()
fig.savefig(f"out/figs/prepost17{SUF}.png", dpi=150)
print(f"wrote out/figs/prepost17{SUF}.png")
