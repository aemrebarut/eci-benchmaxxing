"""Tools for the appendix studies (analysis/23_appendix.py): reviewer objections to the step model and the analyses
that answer them.

PPX extends prepost.PrePost with
  1. extra covariates X (n x q) in the linear predictor, a_j (C_i + delta post + kappa_j z_ij + X_ij beta - D_j), with
     free coefficients beta and an optional ridge lam_x (lab-specific trends, the selection term);
  2. a linear constraint on the ECI scale, b(C) c'delta = v with b(C) = 20 / (C_high - C_low), imposed by a stiff
     penalty row (likelihood-ratio tests, profile likelihoods, restricted fits for the wild bootstrap);
  3. gauge rows, so that Epoch's ridge on C, D and slopes can be removed (set prepost.EPOCH_REG = 0): mean D = 0 fixes
     the location, which is otherwise free (C + t, D + t gives the same fit); with an unpenalized drift (mu = 0) also
     mean kappa = 0, the common drift that C and D absorb except through within-model date differences;
  4. covariances with parameters at their bounds held fixed and a pseudo-inverse, and the frequentist sandwich
     s^2 H^-1 (J'J) H^-1 of a penalized fit beside the model-based s^2 H^-1.
The ceiling (cap) of PrePost is not supported together with X."""
import numpy as np, pandas as pd
import prepost
from prepost import PrePost, org_groups
from eci.fitting import sigmoid, DEFAULT_ANCHOR_ECI_HIGH, DEFAULT_ANCHOR_ECI_LOW

ECI_SPAN = DEFAULT_ANCHOR_ECI_HIGH - DEFAULT_ANCHOR_ECI_LOW
GOOG = "Google DeepMind (all)"
MAIN = ["Google DeepMind (all)", "OpenAI", "Anthropic", "Alibaba", "Meta AI", "DeepSeek", "xAI"]
CON_W = 1e8


class PPX(PrePost):
    def __init__(self, df, group, X=None, trend=None, anchor=None, gauge=False, lam_x=0.0):
        kw = {} if anchor is None else {"anchor": anchor}
        super().__init__(df, group, trend=trend, **kw)
        self.X = np.zeros((self.n, 0)) if X is None else np.asarray(X, float)
        self.q = self.X.shape[1]
        self.P0 = self.P
        self.P = self.P0 + self.q
        self.LB = np.concatenate([self.LB, np.full(self.q, -50.0)])
        self.UB = np.concatenate([self.UB, np.full(self.q, 50.0)])
        self.lam_x, self.gauge, self.con = lam_x, gauge, None

    def set_con(self, c=None, v=0.0):
        """Constrain b(C) c'delta = v (ECI scale); c a K-vector over the identified groups, None to lift it."""
        self.con = None if c is None else (np.asarray(c, float), float(v))

    def blocks(self, use_delta=True, mu=None, nu=None):
        return np.concatenate([super().blocks(use_delta, mu, nu), self.P0 + np.arange(self.q)])

    def init_x(self, seed=42):
        return np.concatenate([super().init_x(seed), np.zeros(self.q)])

    def eta_parts(self, x):
        a, lin, _ = super().eta_parts(x)
        if self.q:
            lin = lin + self.X @ x[self.P0:self.P0 + self.q]
        return a, lin, a[self.bi] * lin

    def _extra(self, x, use_delta, mu):
        """Residual rows beyond PrePost's and their Jacobian rows (full width)."""
        r, J = [], []
        n_ep, K, Jk, M, B = self.n_ep, self.K, self.Jk, self.M, self.B
        if self.q and self.lam_x > 0:
            for k in range(self.q):
                row = np.zeros(self.P); row[self.P0 + k] = np.sqrt(self.lam_x)
                r.append(row @ x); J.append(row)
        if self.gauge:
            row = np.zeros(self.P); row[M:M + B] = 1.0 / B
            r.append(row @ x); J.append(row)
            if Jk and mu == 0:
                row = np.zeros(self.P); row[n_ep + K:n_ep + K + Jk] = 1.0 / Jk
                r.append(row @ x); J.append(row)
        if self.con is not None and use_delta:
            c, v = self.con
            d = x[n_ep:n_ep + K]
            span = x[self.hi_i] - x[self.lo_i]
            cd = float(c @ d)
            sw = np.sqrt(CON_W)
            r.append(sw * (ECI_SPAN * cd / span - v))
            row = np.zeros(self.P)
            row[n_ep:n_ep + K] = sw * ECI_SPAN * c / span
            row[self.hi_i] += -sw * ECI_SPAN * cd / span ** 2
            row[self.lo_i] += sw * ECI_SPAN * cd / span ** 2
            J.append(row)
        return np.array(r, float), np.array(J, float).reshape(-1, self.P)

    def resid(self, xs, w, lam, use_delta=True, mu=None, nu=None):
        base = super().resid(xs, w, lam, use_delta, mu, nu)
        r, _ = self._extra(self.full(xs, self.blocks(use_delta, mu, nu)), use_delta, mu)
        return np.concatenate([base, r])

    def jac(self, xs, w, lam, use_delta=True, mu=None, nu=None):
        idx = self.blocks(use_delta, mu, nu)
        J = super().jac(xs, w, lam, use_delta, mu, nu)
        x = self.full(xs, idx)
        if self.q:
            a, lin, e = self.eta_parts(x)
            s = sigmoid(e)
            ds = np.sqrt(w) * s * (1 - s)
            J[:self.n, len(idx) - self.q:] = (ds * a[self.bi])[:, None] * self.X
        _, Je = self._extra(x, use_delta, mu)
        return np.vstack([J, Je[:, idx]])

    def rss(self, x, w=None):
        w = self.ones if w is None else w
        return float((w * (self.predict(x) - self.y) ** 2).sum())

    def cov2(self, x, lam, mu=None, w=None, drop_bounds=False):
        """Covariances of the free parameters of the unconstrained fit: clustered (Vc; by model unless the attribute
        cluster is "bench" or "cell"), model-based
        s^2 H^-1 (Vm) and the sandwich s^2 H^-1 J'J H^-1 (Vs), H including penalty and gauge rows. Parameters at a
        bound are kept, as in PrePost.cov (more conservative), unless drop_bounds. Returns the parameter indices, Vc,
        Vm, Vs, b and edf."""
        w = self.ones if w is None else w
        con, self.con = self.con, None
        idx = self.blocks(True, mu)
        J = self.jac(x[idx], w, lam, True, mu)
        r = self.resid(x[idx], w, lam, True, mu)
        self.con = con
        xi = x[idx]
        if drop_bounds:
            free = (xi > self.LB[idx] + 1e-7) & (xi < self.UB[idx] - 1e-7)
            idx, J = idx[free], J[:, free]
        Hi = np.linalg.pinv(J.T @ J, rcond=1e-13, hermitian=True)
        n = self.n
        Jd, rd = J[:n], r[:n]
        by = getattr(self, "cluster", "model")
        cl, G = {"model": (self.mi, self.M), "bench": (self.bi, self.B), "cell": (np.arange(n), n)}[by]
        S = np.zeros((len(idx), G))
        np.add.at(S.T, cl, Jd * rd[:, None])
        Vc = Hi @ (S @ S.T) @ Hi
        JJ = Jd.T @ Jd
        edf = float(np.trace(Hi @ JJ))
        s2 = float(rd @ rd) / (n - edf)
        _, b = self.scale(x)
        return idx, Vc, s2 * Hi, s2 * Hi @ JJ @ Hi, b, edf

    def deltas(self, x, lam, mu=None, w=None):
        """ECI-scale deltas (K) and their K x K covariances Vc, Vm, Vs (NaN rows for deltas at a bound), b, edf."""
        idx, Vc, Vm, Vs, b, edf = self.cov2(x, lam, mu, w)
        pos = {p: k for k, p in enumerate(idx)}
        sel = [pos.get(self.n_ep + g, -1) for g in range(self.K)]
        out = []
        for V in (Vc, Vm, Vs):
            W = np.full((self.K, self.K), np.nan)
            ok = [g for g in range(self.K) if sel[g] >= 0]
            W[np.ix_(ok, ok)] = V[np.ix_([sel[g] for g in ok], [sel[g] for g in ok])]
            out.append(b ** 2 * W)
        return b * x[self.n_ep:self.n_ep + self.K], out[0], out[1], out[2], b, edf


def contrast_vec(ident, Vm_diag, target=GOOG, labs=None):
    """target minus the inverse-variance weighted mean of the other groups (weights from the model-based diagonal);
    labs restricts the comparison groups."""
    g = ident.index(target)
    oth = [k for k, lab in enumerate(ident) if k != g and (labs is None or lab in labs) and np.isfinite(Vm_diag[k])]
    c = np.zeros(len(ident))
    w = 1 / Vm_diag[oth]
    c[oth] = -w / w.sum()
    c[g] = 1.0
    return c


def lin_stat(c, d, V):
    est = float(c @ d)
    se = float(np.sqrt(c @ V @ c))
    return est, se, est / se


def load(mode="gmerge"):
    """The cells with organization, group label and the drift covariate, as analysis/19_drift.py loads them."""
    import cm
    df, ORG = cm.load()
    df["org"] = df.Model.map(ORG).fillna("unknown")
    df["group"] = org_groups(df, mode)
    df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    return df


STRUCT = {"epoch": None, "drift": 0.1}


def fit(df, mu=None, lam=1e-3, X=None, lam_x=0.0, unpen=False, anchor=None, x0=None, w=None):
    """Fit the step model; unpen removes every ridge (Epoch's, delta's and, with drift, kappa's: mu 0) and adds gauge
    rows. Returns PP, the base (no delta) result and the full result. prepost.EPOCH_REG is restored afterwards."""
    old = prepost.EPOCH_REG
    try:
        if unpen:
            prepost.EPOCH_REG = 0.0
            lam, mu = 0.0, (0.0 if mu is not None else None)
        PP = PPX(df, df.group, X=X, trend=df.z if mu is not None else None, anchor=anchor, gauge=unpen, lam_x=lam_x)
        w = PP.ones if w is None else w
        base = PP.fit(w, 0.0, PP.init_x() if x0 is None else x0, use_delta=False)
        res = PP.fit(w, lam, base.xf, mu=mu)
        return PP, base, res, lam, mu
    finally:
        prepost.EPOCH_REG = old


def refit(PP, x0, lam, mu, unpen=False, w=None):
    """Refit PP (with its current constraint) from x0, honouring unpen."""
    old = prepost.EPOCH_REG
    try:
        if unpen:
            prepost.EPOCH_REG = 0.0
        return PP.fit(PP.ones if w is None else w, lam, x0, mu=mu)
    finally:
        prepost.EPOCH_REG = old


def cov_unpen(PP, x, lam, mu, unpen=False, w=None):
    old = prepost.EPOCH_REG
    try:
        if unpen:
            prepost.EPOCH_REG = 0.0
        return PP.deltas(x, lam, mu, w)
    finally:
        prepost.EPOCH_REG = old


# ---- score sources -------------------------------------------------------------------------------------------------
INDEP_KEYS = ("http", "eaderboard", "Dashboard", "ithub", "Stanford", "METR", "OS World", "PapersWithCode", "Tweet")


def source_kind(src, epoch_file):
    """epoch: Epoch's own run; third party: a leaderboard or maintainer's table; developer: a report or model card
    from a model developer (the model's own or another's)."""
    if epoch_file:
        return "epoch"
    if pd.isna(src) or any(k in src for k in INDEP_KEYS):
        return "third party"
    return "developer"


def rows(src="data/benchmark_data.zip"):
    """Pre-aggregation rows as eci.dataloader.prepare_benchmark_data builds them (as in analysis/08_data_check.py),
    with the source kind of each row."""
    from eci import dataloader as L
    dfs = L.download_benchmark_data(src)
    meta = L.load_benchmark_metadata(dfs)
    meta = meta[meta["in_eci"].fillna(True).astype(bool)]
    epoch_files = {f for f, d in dfs.items() if "Log viewer" in d.columns}
    fm = meta.set_index("benchmark")["source_file"]
    models = L.load_model_metadata(dfs)[["model_version", "model_group", "date"]]
    s = L._load_scores(dfs, meta)
    s["performance"] = s["performance"].replace([np.inf, -np.inf], np.nan)
    s = s[s["performance"].notna()].copy()
    s["clipped"] = s["performance"].clip(0.0, 1.0)
    s = s.merge(models, on="model_version", how="inner")
    s = s[s["date"] >= pd.Timestamp("2023-01-01")]
    sup = meta.dropna(subset=["superseded_by"])
    sup = sup[sup["superseded_by"].astype(str) != ""]
    for old, new in zip(sup["benchmark"], sup["superseded_by"]):
        g = set(s.loc[s["benchmark"] == new, "model_group"])
        s = s[~((s["benchmark"] == old) & s["model_group"].isin(g))]
    keep = s.groupby("model_group")["benchmark"].nunique()
    s = s[s["model_group"].isin(keep[keep >= 4].index)]
    s["kind"] = [source_kind(x, fm.get(b) in epoch_files) for x, b in zip(s["source"], s["benchmark"])]
    return s
