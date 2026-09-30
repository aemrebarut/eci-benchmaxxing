"""Pre/post shift model (analysis/17_prepost.py): Epoch's ECI model plus a shift delta_g on the cells where the
benchmark was public when the model was released,

    y_ij ~ sigmoid(a_j (C_i + delta_g(i) post_ij - D_j)),  post_ij = 1(gap_ij > 0).

delta_g is identified only for groups with at least one model that has cells on both sides of its release; other
cells get no delta. Objective: Epoch's (least squares on the clipped score, the anchor benchmark's slope pinned
to 1, L2 of 0.1 / n_params on C, D and slopes, bounds as in eci.fitting) plus the ridge 0.5 lam sum delta^2.
Solver: scipy least_squares, trf with the exact trust-region solver on the dense Jacobian."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from eci.fitting import (sigmoid, DEFAULT_ANCHOR_BENCHMARK, DEFAULT_ANCHOR_DISCRIMINABILITY, DEFAULT_ANCHOR_MODEL_LOW,
                         DEFAULT_ANCHOR_ECI_LOW, DEFAULT_ANCHOR_MODEL_HIGH, DEFAULT_ANCHOR_ECI_HIGH)

EPOCH_REG = 0.1


class PrePost:
    def __init__(self, df, group, y_col="y_fit", anchor=DEFAULT_ANCHOR_BENCHMARK):
        """df: cells with Model, benchmark, gap and y_col. group: one label per cell (the delta it may get)."""
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
        self.n_ep = self.M + self.B + (self.B - 1)
        self.P = self.n_ep + self.K
        self.LB = np.concatenate([np.full(self.M + self.B, -10.0), np.full(self.B - 1, 0.1), np.full(self.K, -10.0)])
        self.UB = np.concatenate([np.full(self.M + self.B, 10.0), np.full(self.B - 1, 10.0), np.full(self.K, 10.0)])
        self.ones = np.ones(self.n)

    def init_x(self, seed=42):
        """Epoch's initialization (normal(0, 0.1) abilities and difficulties, slopes 1), deltas 0."""
        r = np.random.default_rng(seed)
        return np.concatenate([r.normal(0, 0.1, self.M), r.normal(0, 0.1, self.B), np.ones(self.B - 1), np.zeros(self.K)])

    def unpack(self, x):
        C, D = x[:self.M], x[self.M:self.M + self.B]
        a = np.full(self.B, DEFAULT_ANCHOR_DISCRIMINABILITY)
        a[self.free_a] = x[self.M + self.B:self.n_ep]
        return C, D, a, x[self.n_ep:]

    def eta_parts(self, x):
        C, D, a, d = self.unpack(x)
        shift = np.where(self.has, d[np.maximum(self.gi, 0)], 0.0) * self.post if len(d) else 0.0
        lin = C[self.mi] + shift - D[self.bi]
        return a, lin, a[self.bi] * lin

    def predict(self, x):
        return sigmoid(self.eta_parts(x)[2])

    def resid(self, x, w, lam, use_delta=True):
        r = np.sqrt(w) * (self.predict(x) - self.y)
        parts = [r, np.sqrt(EPOCH_REG / self.n_ep) * x[:self.n_ep]]
        if use_delta:
            parts.append(np.sqrt(lam) * x[self.n_ep:])
        return np.concatenate(parts)

    def jac(self, x, w, lam, use_delta=True):
        a, lin, e = self.eta_parts(x)
        s = sigmoid(e)
        ds = np.sqrt(w) * s * (1 - s)
        n, M, B, n_ep, K = self.n, self.M, self.B, self.n_ep, self.K
        rows = np.arange(n)
        J = np.zeros((n + n_ep + (K if use_delta else 0), self.P if use_delta else n_ep))
        J[rows, self.mi] = ds * a[self.bi]
        J[rows, M + self.bi] = -ds * a[self.bi]
        ok = self.pos[self.bi] >= 0
        J[rows[ok], M + B + self.pos[self.bi][ok]] = ds[ok] * lin[ok]
        if use_delta:
            hk = self.has & (self.post > 0)
            J[rows[hk], n_ep + self.gi[hk]] = ds[hk] * a[self.bi][hk]
        J[n + np.arange(n_ep), np.arange(n_ep)] = np.sqrt(EPOCH_REG / n_ep)
        if use_delta:
            J[n + n_ep + np.arange(K), n_ep + np.arange(K)] = np.sqrt(lam)
        return J

    def fit(self, w, lam, x0, use_delta=True):
        p = self.P if use_delta else self.n_ep
        return least_squares(self.resid, x0[:p], jac=self.jac, bounds=(self.LB[:p], self.UB[:p]), method="trf",
                             tr_solver="exact", args=(w, lam, use_delta), ftol=1e-13, xtol=1e-13, gtol=1e-11, max_nfev=5000)

    def fit_full(self, lam, x0=None):
        """No-delta fit from Epoch's initialization, then the delta fit warm-started from it."""
        base = self.fit(self.ones, 0.0, self.init_x() if x0 is None else x0, use_delta=False)
        x_base = np.concatenate([base.x, np.zeros(self.K)])
        return base, self.fit(self.ones, lam, x_base)

    def scale(self, x):
        C = x[:self.M]
        b = (DEFAULT_ANCHOR_ECI_HIGH - DEFAULT_ANCHOR_ECI_LOW) / (C[self.hi_i] - C[self.lo_i])
        return DEFAULT_ANCHOR_ECI_LOW - b * C[self.lo_i], b

    def analyse(self, x, lam):
        """ECI-scale deltas, their sandwich covariance clustered by model, the model-based sigma^2 H^-1, b and edf."""
        _, b = self.scale(x)
        J = self.jac(x, self.ones, lam)
        r = self.resid(x, self.ones, lam)
        Hi = np.linalg.inv(J.T @ J)
        n = self.n
        Jd, rd = J[:n], r[:n]
        S = np.zeros((self.P, self.M))
        np.add.at(S.T, self.mi, Jd * rd[:, None])
        Vc = Hi @ (S @ S.T) @ Hi
        edf = float(np.trace(Hi @ (Jd.T @ Jd)))
        Vm = float(rd @ rd) / (n - edf) * Hi
        sl = slice(self.n_ep, self.P)
        return b * x[sl], b ** 2 * Vc[sl, sl], b ** 2 * Vm[sl, sl], b, edf
