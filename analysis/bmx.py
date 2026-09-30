"""ECI with a model-level benchmaxxing term.

    perf(m, b) = sigmoid(alpha_b * (C_m - D_b + psi_m * f(gap_mb)))

gap_mb is the model's release date minus the benchmark's release date, in
years. f is a restricted cubic spline shared by every model and benchmark,
pinned by f(0) = 0 (a constant part of f would be absorbed into C_m) and
f(1) = 1 (psi and f would otherwise trade scale). psi_m is then the
capability-equivalent boost model m gets on a benchmark that was one year old
when m was released.

The fit follows Epoch's eci-public: nonlinear least squares on normalized
scores clipped to [eps, 1 - eps], box bounds, the anchor benchmark's slope
pinned to 1, one ridge row, and an affine map onto the ECI scale that puts
Claude 3.5 Sonnet at 130 and GPT-5 at 150. The ridge keeps Epoch's weight per
parameter (lam / n_params of the baseline model) and extends it to psi, so
psi = 0 reproduces Epoch's objective exactly. Spline coefficients are not
penalized.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

ANCHOR_BENCHMARK = "Winogrande"
ANCHOR_LOW, ECI_LOW = "Claude 3.5 Sonnet", 130.0
ANCHOR_HIGH, ECI_HIGH = "GPT-5", 150.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def add_gap(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["benchmark_release_date"] = pd.to_datetime(df["benchmark_release_date"])
    df["gap"] = (df["date"] - df["benchmark_release_date"]).dt.days / 365.25
    return df


class Spline:
    """Restricted cubic spline (linear beyond the boundary knots) with
    f(0) = 0 and f(1) = 1 imposed by eliminating one coefficient."""

    def __init__(self, knots):
        self.knots = np.asarray(knots, float)
        v = self._basis(np.array([1.0]))[0] - self._basis(np.array([0.0]))[0]
        self.pivot = int(np.argmax(np.abs(v)))
        self.v = v
        self.n_free = len(v) - 1

    @classmethod
    def from_quantiles(cls, x, n_knots=5):
        # Harrell's default knot quantiles
        q = {3: [.10, .50, .90], 4: [.05, .35, .65, .95], 5: [.05, .275, .50, .725, .95],
             6: [.05, .23, .41, .59, .77, .95], 7: [.025, .1833, .3417, .50, .6583, .8167, .975]}[n_knots]
        return cls(np.quantile(x, q))

    def _basis(self, x):
        t, k = self.knots, len(self.knots)
        cube = lambda u: np.maximum(u, 0.0) ** 3
        scale = (t[-1] - t[0]) ** 2
        cols = [x]
        for j in range(k - 2):
            cols.append((cube(x - t[j])
                         - cube(x - t[-2]) * (t[-1] - t[j]) / (t[-1] - t[-2])
                         + cube(x - t[-1]) * (t[-2] - t[j]) / (t[-1] - t[-2])) / scale)
        return np.column_stack(cols)

    zero_below = False

    def design(self, x):
        """Return (f0, G) with f(x) = f0 + G @ beta_free."""
        x = np.asarray(x, float)
        if self.zero_below:
            x = np.maximum(x, 0.0)
        bc = self._basis(x) - self._basis(np.array([0.0]))[0]
        j, v = self.pivot, self.v
        others = [i for i in range(len(v)) if i != j]
        f0 = bc[:, j] / v[j]
        G = bc[:, others] - np.outer(bc[:, j], v[others] / v[j])
        return f0, G

    def __call__(self, x, beta_free):
        f0, G = self.design(x)
        return f0 + G @ beta_free


class FreeSpline(Spline):
    """The same restricted cubic spline with only f(0) = 0 imposed, so psi and
    f trade scale and a penalty must fix it (fit(scale="penalty")). The
    coefficients are rotated so the basis is orthonormal on the fit cells:
    mean over the fit cells of f^2 equals ||beta||^2. The rotation is learned
    once, on the fit cells, and reused on new cells."""

    def __init__(self, knots, x_fit, zero_below=False):
        super().__init__(knots)
        self.zero_below = zero_below
        q, r = np.linalg.qr(self._raw(x_fit))
        self.T = np.linalg.inv(r) * np.sqrt(len(q))  # raw @ T = sqrt(n) * q on the fit cells
        self.n_free = self.T.shape[1]

    def _raw(self, x):
        x = np.asarray(x, float)
        if self.zero_below:
            x = np.maximum(x, 0.0)
        return self._basis(x) - self._basis(np.array([0.0]))[0]

    def design(self, x):
        G = self._raw(x) @ self.T
        return np.zeros(len(G)), G


def _index(df):
    model_ids = df["model_id"].unique()
    bench_ids = df["benchmark_id"].unique()
    m_of = {m: i for i, m in enumerate(model_ids)}
    b_of = {b: i for i, b in enumerate(bench_ids)}
    names_m = df.drop_duplicates("model_id").set_index("model_id")["Model"]
    names_b = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark"]
    return (model_ids, bench_ids, m_of, b_of,
            [names_m[m] for m in model_ids], [names_b[b] for b in bench_ids])


def fit(df, with_psi=True, spline=None, n_knots=5, lam=0.1, clip_eps=1e-3,
        x0=None, seed=42, max_nfev=None, index=None, psi_bound=10.0, zero_below=False,
        lam_psi=None, psi_group=None, ridge_rows="per_param", tr_solver="exact",
        x_scale=1.0, tol=1e-10, scale="pin", kappa=0.1):
    """Fit the model. df needs model_id, benchmark_id, Model, benchmark,
    performance, and gap (see add_gap). Returns a dict of results; with
    with_psi=False it is Epoch's baseline. zero_below=True sets f(gap) = 0
    for gap <= 0 (no credit for benchmarks released after the model).
    lam_psi is the ridge weight on psi (default: Epoch's per-parameter weight).
    psi_group maps Model name to a group label (e.g. organization) so that
    models in a group share one psi; default is one psi per model.

    Optimizer options. ridge_rows="single" is Epoch's form: one residual
    sqrt(w * sum(theta^2)), whose Gauss-Newton curvature is rank one.
    ridge_rows="per_param" appends sqrt(w) * theta_i for each parameter: the
    same objective, with full curvature w * I. tr_solver="exact" solves each
    trust-region subproblem exactly on a dense Jacobian (default: lsmr on the
    sparse one). x_scale and tol pass through to least_squares. Defaults are
    per_param, exact, tol 1e-10: with psi, Epoch's settings (single, lsmr,
    tol 1e-8) stop up to 5 ECI points from the minimum (analysis/04_optimizer.py).

    Scale. scale="pin" fixes f(1) = 1 and leaves f unpenalized. Then a ridge
    on psi is evaded by inflating f. scale="penalty" imposes only f(0) = 0
    (FreeSpline) and adds kappa * (mean over groups of psi^2 + mean over fit
    cells of f^2) to the sum of squares, replacing lam_psi. Over the psi/f
    scale trade-off this equals 2 * kappa * rms(psi) * rms(f), so at the
    optimum rms(psi) = rms(f). The results then also carry f normalized to
    f(1) = 1, with psi rescaled to match, so they compare with the pinned
    fits."""
    penalized = with_psi and scale == "penalty"
    if penalized and (lam_psi is not None or ridge_rows != "per_param"):
        raise ValueError("scale='penalty' uses kappa and per_param ridge rows")
    if with_psi and scale not in ("pin", "penalty"):
        raise ValueError(f"unknown scale {scale!r}")
    y = df["performance"].clip(clip_eps, 1 - clip_eps).to_numpy()
    model_ids, bench_ids, m_of, b_of, mnames, bnames = index or _index(df)
    M, B = len(model_ids), len(bench_ids)
    mi = np.array([m_of[m] for m in df["model_id"]])
    bi = np.array([b_of[b] for b in df["benchmark_id"]])
    a_idx = bnames.index(ANCHOR_BENCHMARK)
    n_base = M + B + (B - 1)
    w = lam / n_base

    if with_psi:
        g = df["gap"].to_numpy()
        if penalized:
            if not isinstance(spline, FreeSpline):
                knots = spline.knots if spline is not None else \
                    Spline.from_quantiles(g[g > 0] if zero_below else g, n_knots).knots
                spline = FreeSpline(knots, g, zero_below)
        elif zero_below:
            spline = spline or Spline.from_quantiles(g[g > 0], n_knots)
            spline.zero_below = True
        spline = spline or Spline.from_quantiles(g, n_knots)
        f0, G = spline.design(g)
        K = spline.n_free
    else:
        K = 0
    if psi_group is not None:
        labels = [psi_group[m] for m in mnames]
        groups = list(dict.fromkeys(labels))
        grp = np.array([groups.index(l) for l in labels])
    else:
        groups, grp = list(mnames), np.arange(M)
    P = len(groups) if with_psi else 0
    n = n_base + P + K
    w_psi = w if lam_psi is None else lam_psi  # pinned: spline coefficients are not penalized
    if penalized:
        w_psi = kappa / P  # kappa * mean(psi^2)
    w_beta = kappa if penalized else 0.0  # kappa * mean f^2 = kappa * ||beta||^2 (FreeSpline)
    gi = grp[mi]

    def unpack(th):
        C = th[:M]
        D = th[M:M + B]
        al = np.insert(th[M + B:n_base], a_idx, 1.0)
        psi = th[n_base:n_base + P] if with_psi else None
        beta = th[n_base + P:] if with_psi else None
        return C, D, al, psi, beta

    def resid(th):
        C, D, al, psi, beta = unpack(th)
        lin = C[mi] - D[bi]
        if with_psi:
            lin = lin + psi[gi] * (f0 + G @ beta)
        r = sigmoid(al[bi] * lin) - y
        if ridge_rows == "per_param":
            pens = [np.sqrt(w) * th[:n_base]]
            if with_psi:
                pens.append(np.sqrt(w_psi) * psi)
            if penalized:
                pens.append(np.sqrt(w_beta) * beta)
            return np.concatenate([r] + pens)
        pens = [np.sqrt(w * np.sum(th[:n_base] ** 2))]
        if with_psi:
            pens.append(np.sqrt(w_psi * np.sum(psi ** 2)))
        return np.append(r, pens)

    obs = np.arange(len(y))
    free = bi != a_idx
    a_col = n_base - (B - 1) + np.where(bi < a_idx, bi, bi - 1)

    def jac(th):
        C, D, al, psi, beta = unpack(th)
        f = (f0 + G @ beta) if with_psi else 0.0
        lin = C[mi] - D[bi] + (psi[gi] * f if with_psi else 0.0)
        s = sigmoid(al[bi] * lin)
        ds = s * (1 - s)
        rows = [obs, obs, obs[free]]
        cols = [mi, M + bi, a_col[free]]
        vals = [ds * al[bi], -ds * al[bi], (ds * lin)[free]]
        if with_psi:
            rows.append(obs); cols.append(n_base + gi); vals.append(ds * al[bi] * f)
            gb = (ds * al[bi] * psi[gi])[:, None] * G
            rows.append(np.repeat(obs, K)); cols.append(np.tile(n_base + P + np.arange(K), len(obs)))
            vals.append(gb.ravel())
        if ridge_rows == "per_param":
            rows.append(len(obs) + np.arange(n_base)); cols.append(np.arange(n_base))
            vals.append(np.full(n_base, np.sqrt(w)))
            n_rows = len(obs) + n_base
            if with_psi:
                rows.append(n_rows + np.arange(P)); cols.append(n_base + np.arange(P))
                vals.append(np.full(P, np.sqrt(w_psi)))
                n_rows += P
            if penalized:
                rows.append(n_rows + np.arange(K)); cols.append(n_base + P + np.arange(K))
                vals.append(np.full(K, np.sqrt(w_beta)))
                n_rows += K
            J = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n_rows, n)).tocsr()
            return J.toarray() if tr_solver == "exact" else J
        pen = w * np.sum(th[:n_base] ** 2)
        if pen > 0:
            rows.append(np.full(n_base, len(obs))); cols.append(np.arange(n_base))
            vals.append(w * th[:n_base] / np.sqrt(pen))
        n_rows = len(obs) + 1
        if with_psi:
            pp = w_psi * np.sum(psi ** 2)
            if pp > 0:
                rows.append(np.full(P, len(obs) + 1)); cols.append(n_base + np.arange(P))
                vals.append(w_psi * psi / np.sqrt(pp))
            n_rows += 1
        J = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n_rows, n)).tocsr()
        return J.toarray() if tr_solver == "exact" else J

    lo = np.concatenate([np.full(M + B, -10.0), np.full(B - 1, 0.1), np.full(P, -psi_bound), np.full(K, -np.inf)])
    hi = np.concatenate([np.full(M + B, 10.0), np.full(B - 1, 10.0), np.full(P, psi_bound), np.full(K, np.inf)])
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = np.concatenate([rng.normal(0, .1, M), rng.normal(0, .1, B), np.ones(B - 1),
                             rng.normal(0, .01, P), np.zeros(K)])
        if penalized:  # psi = 0, f = 0 is a saddle of the product; start off it
            x0[n_base:] = rng.normal(0, .1, P + K)
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    res = least_squares(resid, x0, jac=jac, bounds=(lo, hi), method="trf", max_nfev=max_nfev,
                        tr_solver=tr_solver, x_scale=x_scale, ftol=tol, xtol=tol, gtol=tol)
    C, D, al, psi, beta = unpack(res.x)
    a, b = _affine(C[mnames.index(ANCHOR_LOW)], C[mnames.index(ANCHOR_HIGH)])
    out = {
        "x": res.x, "success": res.success, "cost": res.cost, "nfev": res.nfev,
        "status": res.status, "message": res.message, "optimality": res.optimality,
        "n_active": int(np.sum(res.active_mask != 0)),
        "sse": float(np.sum(res.fun[:len(y)] ** 2)), "n_obs": len(y), "n_params": n,
        "a": a, "b": b, "index": (model_ids, bench_ids, m_of, b_of, mnames, bnames),
        "models": pd.DataFrame({"model_id": model_ids, "Model": mnames, "C": C, "eci": a + b * C}),
        "benchmarks": pd.DataFrame({"benchmark_id": bench_ids, "benchmark": bnames, "D": D,
                                    "edi": a + b * D, "alpha": al, "slope_eci": al / b}),
        "resid": res.fun[:len(y)], "mi": mi, "bi": bi, "groups": groups, "grp": grp,
    }
    if with_psi:
        out["spline"] = spline
        out["beta"] = beta
        # penalized: report f / f(1) and psi * f(1), which leaves psi * f unchanged
        s1 = float(spline(np.array([1.0]), beta)[0]) if penalized else 1.0
        if penalized:
            out.update(f1=s1, rms_psi=float(np.sqrt(np.mean(psi ** 2))),
                       rms_f=float(np.sqrt(np.mean((G @ beta) ** 2))), kappa=kappa)
            out["models"]["psi_raw"] = psi[grp]
        out["models"]["psi"] = s1 * psi[grp]
        out["models"]["psi_eci"] = b * s1 * psi[grp]
        out["psi_groups"] = pd.DataFrame({"group": groups, "psi": s1 * psi, "psi_eci": b * s1 * psi})
    return out


def f_norm(out, x):
    """f at gaps x, normalized to f(1) = 1 for penalized fits (as fitted for pinned ones)."""
    return out["spline"](np.asarray(x, float), out["beta"]) / out.get("f1", 1.0)


def _affine(c_low, c_high):
    spread = c_high - c_low
    if not spread > 1e-12:
        raise ValueError(f"anchors do not define a scale: low {c_low}, high {c_high}")
    b = (ECI_HIGH - ECI_LOW) / spread
    return ECI_LOW - b * c_low, b


def predict(out, df):
    """Predicted performance for rows of df (models and benchmarks must be in the fit index)."""
    _, _, m_of, b_of, _, _ = out["index"]
    th = out["x"]
    M, B = len(m_of), len(b_of)
    n_base = M + B + (B - 1)
    mi = np.array([m_of[m] for m in df["model_id"]])
    bi = np.array([b_of[b] for b in df["benchmark_id"]])
    bnames = out["index"][5]
    al = np.insert(th[M + B:n_base], bnames.index(ANCHOR_BENCHMARK), 1.0)
    lin = th[:M][mi] - th[M:M + B][bi]
    if "spline" in out:
        psi = th[n_base:n_base + len(out["groups"])][out["grp"]]
        lin = lin + psi[mi] * out["spline"](df["gap"].to_numpy(), out["beta"])
    return sigmoid(al[bi] * lin)
