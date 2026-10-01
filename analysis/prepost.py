"""Pre/post shift model (analysis/17_prepost.py): Epoch's ECI model plus a shift delta_g on the cells where the
benchmark was public when the model was released,

    y_ij ~ sigmoid(a_j (C_i + delta_g(i) post_ij + kappa_j z_ij - D_j)),  post_ij = 1(gap_ij > 0).

delta_g is identified only for groups with at least one model that has cells on both sides of its release; other
cells get no delta. The drift term kappa_j z_ij (analysis/19_drift.py) is optional: z_ij is a per-cell covariate
(the model's release date centred within the benchmark) and kappa_j one coefficient per benchmark. So is a ceiling
per benchmark (analysis/22_cap.py): y_ij ~ (1 - c_j) sigmoid(...), c_j in [0, CAP_MAX]. Objective: Epoch's (least
squares on the clipped score, the anchor benchmark's slope pinned to 1, L2 of 0.1 / n_params on C, D and slopes,
bounds as in eci.fitting) plus the ridges 0.5 lam sum delta^2, 0.5 mu sum kappa^2 and 0.5 nu sum c^2. Parameters
are ordered C, D, free slopes, delta, kappa, c; a fit frees Epoch's parameters, delta if use_delta, kappa if mu is
given and c if nu is given. Solver: scipy least_squares, trf with the exact trust-region solver on the dense
Jacobian."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
import scipy.optimize._lsq.trf as _trf
from eci.fitting import (sigmoid, DEFAULT_ANCHOR_BENCHMARK, DEFAULT_ANCHOR_DISCRIMINABILITY, DEFAULT_ANCHOR_MODEL_LOW,
                         DEFAULT_ANCHOR_ECI_LOW, DEFAULT_ANCHOR_MODEL_HIGH, DEFAULT_ANCHOR_ECI_HIGH)

EPOCH_REG = 0.1
CAP_MAX = 0.9
_trf_svd = _trf.svd


def _svd_fallback(a, full_matrices=True, **kw):
    """trf's SVD with a fallback: LAPACK gesdd sometimes fails to converge on a well-conditioned Jacobian (seen in a
    cap fit, smallest singular value 0.016); gesvd, slower but robust, is used only then."""
    try:
        return _trf_svd(a, full_matrices=full_matrices, **kw)
    except np.linalg.LinAlgError:
        return _trf_svd(a, full_matrices=full_matrices, lapack_driver="gesvd", **kw)


_trf.svd = _svd_fallback


class PrePost:
    def __init__(self, df, group, y_col="y_fit", anchor=DEFAULT_ANCHOR_BENCHMARK, trend=None, cap=False):
        """df: cells with Model, benchmark, gap and y_col. group: one label per cell (the delta it may get).
        trend: optional per-cell covariate z, which adds kappa_j z_ij. cap: add the ceiling 1 - c_j."""
        d = df.reset_index(drop=True)
        self.df = d
        self.models = list(dict.fromkeys(d.Model))
        self.benches = list(dict.fromkeys(d.benchmark))
        self.M, self.B, self.n = len(self.models), len(self.benches), len(d)
        self.mi = d.Model.map({m: i for i, m in enumerate(self.models)}).to_numpy()
        self.bi = d.benchmark.map({b: j for j, b in enumerate(self.benches)}).to_numpy()
        self.y = d[y_col].to_numpy(float)
        self.post = (d.gap > 0).to_numpy(float)
        self.anchor = self.benches.index(anchor)
        self.free_a = np.array([j for j in range(self.B) if j != self.anchor])
        self.pos = np.full(self.B, -1)
        self.pos[self.free_a] = np.arange(self.B - 1)
        self.lo_i, self.hi_i = self.models.index(DEFAULT_ANCHOR_MODEL_LOW), self.models.index(DEFAULT_ANCHOR_MODEL_HIGH)
        self.both = pd.Series(self.post).groupby(d.Model).agg(lambda s: 0 < s.mean() < 1)
        self.group = pd.Series(np.asarray(group), index=d.index)
        self.ident = sorted(self.group[d.Model.map(self.both).to_numpy()].unique())
        gi_of = {g: k for k, g in enumerate(self.ident)}
        self.gi = self.group.map(gi_of).fillna(-1).astype(int).to_numpy()
        self.has = self.gi >= 0
        self.K = len(self.ident)
        self.z = None if trend is None else np.asarray(trend, float)
        self.Jk = 0 if trend is None else self.B
        self.Jc = self.B if cap else 0
        self.n_ep = self.M + self.B + (self.B - 1)
        self.P = self.n_ep + self.K + self.Jk + self.Jc
        self.LB = np.concatenate([np.full(self.M + self.B, -10.0), np.full(self.B - 1, 0.1), np.full(self.K + self.Jk, -10.0), np.zeros(self.Jc)])
        self.UB = np.concatenate([np.full(self.M + self.B, 10.0), np.full(self.B - 1, 10.0), np.full(self.K + self.Jk, 10.0), np.full(self.Jc, CAP_MAX)])
        self.ones = np.ones(self.n)

    def init_x(self, seed=42):
        """Epoch's initialization (normal(0, 0.1) abilities and difficulties, slopes 1), deltas, kappas and c 0."""
        r = np.random.default_rng(seed)
        return np.concatenate([r.normal(0, 0.1, self.M), r.normal(0, 0.1, self.B), np.ones(self.B - 1), np.zeros(self.K + self.Jk + self.Jc)])

    def blocks(self, use_delta=True, mu=None, nu=None):
        """Indices of the free parameters: Epoch's, then delta if use_delta, kappa if mu is given, c if nu is given."""
        parts = [np.arange(self.n_ep)]
        if use_delta:
            parts.append(self.n_ep + np.arange(self.K))
        if self.Jk and mu is not None:
            parts.append(self.n_ep + self.K + np.arange(self.Jk))
        if self.Jc and nu is not None:
            parts.append(self.n_ep + self.K + self.Jk + np.arange(self.Jc))
        return np.concatenate(parts)

    def full(self, xs, idx):
        x = np.zeros(self.P)
        x[idx] = xs
        return x

    def unpack(self, x):
        C, D = x[:self.M], x[self.M:self.M + self.B]
        a = np.full(self.B, DEFAULT_ANCHOR_DISCRIMINABILITY)
        a[self.free_a] = x[self.M + self.B:self.n_ep]
        return C, D, a, x[self.n_ep:self.n_ep + self.K], x[self.n_ep + self.K:self.n_ep + self.K + self.Jk]

    def ceiling(self, x):
        """Per-benchmark ceiling 1 - c_j (ones without the cap)."""
        return 1 - x[self.n_ep + self.K + self.Jk:] if self.Jc else np.ones(self.B)

    def eta_parts(self, x):
        C, D, a, d, k = self.unpack(x)
        shift = np.where(self.has, d[np.maximum(self.gi, 0)], 0.0) * self.post if self.K else 0.0
        lin = C[self.mi] + shift - D[self.bi]
        if self.Jk:
            lin = lin + k[self.bi] * self.z
        return a, lin, a[self.bi] * lin

    def predict(self, x):
        """x: the full parameter vector."""
        return self.ceiling(x)[self.bi] * sigmoid(self.eta_parts(x)[2])

    def resid(self, xs, w, lam, use_delta=True, mu=None, nu=None):
        x = self.full(xs, self.blocks(use_delta, mu, nu))
        parts = [np.sqrt(w) * (self.predict(x) - self.y), np.sqrt(EPOCH_REG / self.n_ep) * x[:self.n_ep]]
        if use_delta:
            parts.append(np.sqrt(lam) * x[self.n_ep:self.n_ep + self.K])
        if self.Jk and mu is not None:
            parts.append(np.sqrt(mu) * x[self.n_ep + self.K:self.n_ep + self.K + self.Jk])
        if self.Jc and nu is not None:
            parts.append(np.sqrt(nu) * x[self.n_ep + self.K + self.Jk:])
        return np.concatenate(parts)

    def jac(self, xs, w, lam, use_delta=True, mu=None, nu=None):
        idx = self.blocks(use_delta, mu, nu)
        x = self.full(xs, idx)
        a, lin, e = self.eta_parts(x)
        s = sigmoid(e)
        ds = np.sqrt(w) * s * (1 - s)
        if self.Jc:
            ds = ds * self.ceiling(x)[self.bi]
        n, M, B, n_ep, K, Jk, Jc = self.n, self.M, self.B, self.n_ep, self.K, self.Jk, self.Jc
        rows = np.arange(n)
        J = np.zeros((n + n_ep, self.P))
        J[rows, self.mi] = ds * a[self.bi]
        J[rows, M + self.bi] = -ds * a[self.bi]
        ok = self.pos[self.bi] >= 0
        J[rows[ok], M + B + self.pos[self.bi][ok]] = ds[ok] * lin[ok]
        hk = self.has & (self.post > 0)
        J[rows[hk], n_ep + self.gi[hk]] = ds[hk] * a[self.bi][hk]
        if Jk:
            J[rows, n_ep + K + self.bi] = ds * a[self.bi] * self.z
        if Jc:
            J[rows, n_ep + K + Jk + self.bi] = -np.sqrt(w) * s
        J[n + np.arange(n_ep), np.arange(n_ep)] = np.sqrt(EPOCH_REG / n_ep)
        pen = []
        if use_delta:
            E = np.zeros((K, self.P))
            E[np.arange(K), n_ep + np.arange(K)] = np.sqrt(lam)
            pen.append(E)
        if Jk and mu is not None:
            E = np.zeros((Jk, self.P))
            E[np.arange(Jk), n_ep + K + np.arange(Jk)] = np.sqrt(mu)
            pen.append(E)
        if Jc and nu is not None:
            E = np.zeros((Jc, self.P))
            E[np.arange(Jc), n_ep + K + Jk + np.arange(Jc)] = np.sqrt(nu)
            pen.append(E)
        return np.vstack([J] + pen)[:, idx]

    def fit(self, w, lam, x0, use_delta=True, mu=None, nu=None):
        """x0: the full parameter vector, or a prefix of it (the rest 0). Returns scipy's result, with .x the free
        parameters and .xf the full vector."""
        idx = self.blocks(use_delta, mu, nu)
        x0 = np.concatenate([x0, np.zeros(self.P - len(x0))])
        res = least_squares(self.resid, x0[idx], jac=self.jac, bounds=(self.LB[idx], self.UB[idx]), method="trf",
                            tr_solver="exact", args=(w, lam, use_delta, mu, nu), ftol=1e-13, xtol=1e-13, gtol=1e-11, max_nfev=5000)
        res.xf = self.full(res.x, idx)
        return res

    def fit_full(self, lam, x0=None, mu=None, nu=None):
        """No-delta, no-kappa, no-cap fit from Epoch's initialization, then the full fit warm-started from it."""
        base = self.fit(self.ones, 0.0, self.init_x() if x0 is None else x0, use_delta=False)
        return base, self.fit(self.ones, lam, base.xf, mu=mu, nu=nu)

    def scale(self, x):
        C = x[:self.M]
        b = (DEFAULT_ANCHOR_ECI_HIGH - DEFAULT_ANCHOR_ECI_LOW) / (C[self.hi_i] - C[self.lo_i])
        return DEFAULT_ANCHOR_ECI_LOW - b * C[self.lo_i], b

    def cov(self, x, lam, mu=None, nu=None):
        """Raw-scale covariances of the free parameters of the full fit (delta, kappa if mu is given, c if nu is
        given; c at its lower bound 0 is held fixed): the sandwich clustered by model and the model-based
        sigma^2 H^-1. Returns their indices, Vc, Vm, b and edf."""
        _, b = self.scale(x)
        idx = self.blocks(True, mu, nu)
        J = self.jac(x[idx], self.ones, lam, True, mu, nu)
        r = self.resid(x[idx], self.ones, lam, True, mu, nu)
        keep = ~((idx >= self.n_ep + self.K + self.Jk) & (x[idx] <= 1e-9))
        idx, J = idx[keep], J[:, keep]
        Hi = np.linalg.inv(J.T @ J)
        n = self.n
        Jd, rd = J[:n], r[:n]
        S = np.zeros((len(idx), self.M))
        np.add.at(S.T, self.mi, Jd * rd[:, None])
        Vc = Hi @ (S @ S.T) @ Hi
        edf = float(np.trace(Hi @ (Jd.T @ Jd)))
        Vm = float(rd @ rd) / (n - edf) * Hi
        return idx, Vc, Vm, b, edf

    def analyse(self, x, lam, mu=None, nu=None):
        """ECI-scale deltas, their sandwich covariance clustered by model, the model-based sigma^2 H^-1, b and edf."""
        _, Vc, Vm, b, edf = self.cov(x, lam, mu, nu)
        sl = slice(self.n_ep, self.n_ep + self.K)
        return b * x[sl], b ** 2 * Vc[sl, sl], b ** 2 * Vm[sl, sl], b, edf


GOOGLE = {"Google DeepMind", "Google DeepMind,Google", "Google"}


def org_groups(df, mode="org"):
    """Group label per cell: the organization ("org"); ("gmerge") the same with Google's three organization names in
    Epoch's data merged into Google DeepMind (all); or ("gsplit") merged and split by family into Google DeepMind
    (Gemini) and Google DeepMind (Gemma)."""
    if mode == "org":
        return df.org
    if mode == "gmerge":
        return pd.Series(np.where(df.org.isin(GOOGLE), "Google DeepMind (all)", df.org), index=df.index)
    fam = np.select([df.Model.str.contains("Gemma"), df.Model.str.contains("Gemini")], ["Gemma", "Gemini"], "other")
    return pd.Series(np.where(df.org.isin(GOOGLE), "Google DeepMind (" + pd.Series(fam, index=df.index) + ")", df.org), index=df.index)


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
