"""Common-mechanism model: one shared gap effect, tested against rival explanations.

    eta_ij = exp(rho_i) * (alpha_j * (theta_i + gamma_i z_j + c_sel s_ij + psi_i h(gap_ij)) - beta_j)
    psi_i  = m + u_org(i) + v_i
    y_ij  ~ sigmoid(eta_ij), fit by penalized least squares

- h(x) = tau * (1 - exp(-max(x - x0, 0) / tau)): zero before the kink x0, slope 1 just after it,
  saturating at tau (tau = inf gives the hinge). m is the common effect per year of gap just after
  the kink, in location units; the model's ECI scale converts it.
- gamma_i z_j: a per-model slope on centred benchmark date (years). This is the specialization rival:
  a model that is relatively better on older (or newer) benchmarks, whatever the gap.
- rho_i: per-model discrimination (log scale), the second rival.
- s_ij = E[max of n_ij standard normals]: selection from keeping the best of n_ij raw rows.
- u_org, v_i: organization and model deviations of the gap coefficient, ridge-shrunk (random-effect
  style). A single ridge weight shrinks each deviation according to how much data informs it.

Benchmarks enter in slope-intercept form (difficulty delta_j = beta_j / alpha_j) with alpha_j = 1 + e_j
unconstrained. In the slope-difficulty form a nearly flat benchmark (TriviaQA and LAMBADA in some bootstrap
replicates) sends alpha toward 0 and delta toward minus infinity along a curved valley, and Levenberg-
Marquardt crawls for thousands of iterations with a large cost decrease still available.

Gauge. The anchor benchmark has beta = 0, which fixes location. Scale is fixed by the mean benchmark
slope (sum of e_j is 0), and the rho are centred; both are imposed as stiff penalties (weight L_GAUGE
on the squared sum). Pinning one benchmark's slope instead
(Epoch's Winogrande = 1) holds the scale only weakly, and the fit then shrinks the whole scale to dodge
the ridge on gamma: ability sd fell from 0.91 to 0.34 and the ARC-AGI slope rose from 5.8 to 33, which
wrecked held-out predictions. ECI is unaffected by the gauge: it maps through the two anchor models.

Ridges. A small ridge lam_a on e_j (alpha toward 1). theta and beta carry a ridge w_loc. The analysis
uses 1e-6 (effectively none) on the full data. CV folds need 1e-4 (the default here): with less, a model
or benchmark whose training scores all sit at or below chance drifts toward minus or plus infinity and
the fold does not converge. On the full data, 1e-4 pulls strong models toward the anchor and the gap term
compensates, which inflates the gap effect (as Fable's review predicted).

Targets. The primary fits use Epoch's clipped scores (y_fit). With unclipped scores, a benchmark or model
whose (simulated) scores are mostly below 0 has no finite optimum: the fit keeps steepening the curve or
lowering the ability, and bootstrap refits did not converge in 1000 iterations with a cost decrease of
up to 1e-2 still available. On the real data, clipping changes the gap effect by about 3%.

The solver is Levenberg-Marquardt on the normal equations with a sparse Jacobian (Marquardt scaling,
Nielsen damping updates). Every fit carries a convergence certificate (see fit).
"""
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, LinAlgError
from scipy.special import expit
from scipy import integrate, stats

ANCHOR_BENCHMARK = "Winogrande"
ANCHOR_LOW, ECI_LOW = "Claude 3.5 Sonnet", 130.0
ANCHOR_HIGH, ECI_HIGH = "GPT-5", 150.0
L_GAUGE = 100.0
CLIP = 1e-3  # Epoch's fit clips scores to [CLIP, 1 - CLIP]; targets at or below 0 are unattainable for a sigmoid


def load(src="data/benchmark_data.zip", check="out/data_check08.csv"):
    """Epoch's cells plus rows per cell and the unclipped max (from analysis/08_data_check.py), benchmark
    date in decimal years, and the organization map. y is the unclipped score, y_epoch Epoch's score (clipped
    to [0, 1]), y_fit the score Epoch fits (clipped to [CLIP, 1 - CLIP])."""
    import zipfile
    import bmx
    from eci import prepare_benchmark_data
    df = bmx.add_gap(prepare_benchmark_data(source=src))
    chk = pd.read_csv(check)[["Model", "benchmark", "n_rows", "raw_max"]]
    df = df.merge(chk, on=["Model", "benchmark"], how="left", validate="one_to_one")
    assert df["n_rows"].notna().all()
    bd = df["benchmark_release_date"]
    df["bdate"] = bd.dt.year + (bd.dt.dayofyear - 1) / 365.25
    df["y"], df["y_epoch"] = df["raw_max"], df["performance"]
    df["y_fit"] = df["performance"].clip(CLIP, 1 - CLIP)
    meta = pd.read_csv(zipfile.ZipFile(src).open("model_metadata.csv"))
    org = meta.drop_duplicates("model_group").set_index("model_group")["organization"].fillna("unknown").to_dict()
    return df.reset_index(drop=True), org


def expected_max(n):
    """E[max of n iid standard normals]; 0 for n = 1."""
    f = lambda x, k: x * k * stats.norm.pdf(x) * stats.norm.cdf(x) ** (k - 1)
    return 0.0 if n <= 1 else integrate.quad(f, -10, 10, args=(n,))[0]


def h(gap, tau, x0=0.0):
    x = np.maximum(np.asarray(gap, float) - x0, 0.0)
    return x if np.isinf(tau) else tau * (1.0 - np.exp(-x / tau))


class Data:
    """Index arrays and covariates for the cells. w are cell weights (0 drops a cell, as in a CV fold)."""

    def __init__(self, df, y_col="y", org=None, w=None):
        self.df = df
        self.models = list(dict.fromkeys(df["Model"]))
        self.benches = list(dict.fromkeys(df["benchmark"]))
        m_of = {m: i for i, m in enumerate(self.models)}
        b_of = {b: j for j, b in enumerate(self.benches)}
        self.mi = df["Model"].map(m_of).to_numpy()
        self.bi = df["benchmark"].map(b_of).to_numpy()
        self.y = df[y_col].to_numpy(float)
        self.w = np.ones(len(df)) if w is None else np.asarray(w, float)
        self.sw = np.sqrt(self.w)
        self.gap = df["gap"].to_numpy(float)
        bdate = df.groupby("benchmark")["bdate"].first()
        self.z_b = (bdate.reindex(self.benches).to_numpy(float) - float(np.mean(bdate.to_numpy(float))))
        emax = {n: expected_max(n) for n in np.unique(df["n_rows"])}
        self.sel = df["n_rows"].map(emax).to_numpy(float)
        self.M, self.B, self.n = len(self.models), len(self.benches), len(df)
        self.anchor = self.benches.index(ANCHOR_BENCHMARK)
        org = org or {}
        self.orgs = sorted({org.get(m, "unknown") for m in self.models})
        o_of = {o: g for g, o in enumerate(self.orgs)}
        self.oi = np.array([o_of[org.get(m, "unknown")] for m in self.models])


class Spec:
    """Which terms are in the model, the penalties, and the gap shape."""

    def __init__(self, gap=True, gamma=True, rho=True, sel=True, org=False, model_dev=False,
                 tau=1.0, x0=0.0, lam_g=1.0, lam_r=1.0, lam_u=1.0, lam_v=10.0, lam_a=1e-3, w_loc=1e-4):
        self.__dict__.update(locals()); del self.__dict__["self"]

    def replace(self, **kw):
        s = Spec(**{k: v for k, v in self.__dict__.items()}); s.__dict__.update(kw); return s


class Layout:
    def __init__(self, d, spec):
        sizes = [("theta", d.M), ("beta", d.B - 1), ("e", d.B)]
        if spec.gamma: sizes.append(("gamma", d.M))
        if spec.rho: sizes.append(("rho", d.M))
        if spec.sel: sizes.append(("csel", 1))
        if spec.gap:
            sizes.append(("m", 1))
            if spec.org: sizes.append(("u", len(d.orgs)))
            if spec.model_dev: sizes.append(("v", d.M))
        self.sl, k = {}, 0
        for name, n in sizes:
            self.sl[name] = slice(k, k + n); k += n
        self.n = k
        lam = {"theta": spec.w_loc, "beta": spec.w_loc, "e": spec.lam_a, "gamma": spec.lam_g, "rho": spec.lam_r,
               "csel": 0.0, "m": 0.0, "u": spec.lam_u, "v": spec.lam_v}
        self.lam = np.zeros(k)
        for name, s in self.sl.items():
            self.lam[s] = lam[name]
        self.P = np.diag(self.lam)  # penalty matrix: cost = 0.5 * (r'r + x'Px)
        for name in ("e", "rho"):
            if name in self.sl:
                self.P[self.sl[name], self.sl[name]] += L_GAUGE
        free = np.delete(np.arange(d.B), d.anchor)       # benchmark index -> free-parameter position
        self.bfree = np.full(d.B, -1); self.bfree[free] = np.arange(d.B - 1)

    def get(self, x, name, n=0):
        return x[self.sl[name]] if name in self.sl else np.zeros(n)


def _parts(x, d, spec, lay, gap_h):
    th = lay.get(x, "theta")
    be = np.zeros(d.B); be[lay.bfree >= 0] = lay.get(x, "beta")
    al = 1.0 + lay.get(x, "e")
    ga, rh = lay.get(x, "gamma", d.M), lay.get(x, "rho", d.M)
    cs = lay.get(x, "csel", 1)[0] if "csel" in lay.sl else 0.0
    psi = np.zeros(d.M)
    if "m" in lay.sl:
        psi = psi + x[lay.sl["m"]][0] + lay.get(x, "u", len(d.orgs))[d.oi] + lay.get(x, "v", d.M)
    s, ab = np.exp(rh)[d.mi], al[d.bi]
    Lin = th[d.mi] + ga[d.mi] * d.z_b[d.bi] + cs * d.sel + psi[d.mi] * gap_h
    return s, ab, Lin, s * (ab * Lin - be[d.bi])


def predict(x, d, spec, lay):
    return expit(_parts(x, d, spec, lay, h(d.gap, spec.tau, spec.x0))[3])


def _resid_jac(x, d, spec, lay, gap_h, want_jac=True, sw=None):
    """Weighted residuals and their Jacobian; sw=1 gives the Jacobian of the fitted values on every cell."""
    sw = d.sw if sw is None else sw
    s, ab, Lin, eta = _parts(x, d, spec, lay, gap_h)
    p = expit(eta)
    r = sw * (p - d.y)
    if not want_jac:
        return r, None
    dp = sw * p * (1 - p)
    rows, cols, vals = [], [], []
    idx = np.arange(d.n)

    def add(col, val, mask=None):
        if mask is not None:
            rows.append(idx[mask]); cols.append(col[mask]); vals.append(val[mask])
        else:
            rows.append(idx); cols.append(col); vals.append(val)

    dl = dp * s * ab  # d p / d(anything inside the ability bracket)
    add(lay.sl["theta"].start + d.mi, dl)
    fb = lay.bfree[d.bi]; has = fb >= 0
    add(lay.sl["beta"].start + np.where(has, fb, 0), -dp * s, has)
    add(lay.sl["e"].start + d.bi, dp * s * Lin)
    if "gamma" in lay.sl: add(lay.sl["gamma"].start + d.mi, dl * d.z_b[d.bi])
    if "rho" in lay.sl: add(lay.sl["rho"].start + d.mi, dp * eta)
    if "csel" in lay.sl: add(np.full(d.n, lay.sl["csel"].start), dl * d.sel)
    if "m" in lay.sl:
        dh = dl * gap_h
        add(np.full(d.n, lay.sl["m"].start), dh)
        if "u" in lay.sl: add(lay.sl["u"].start + d.oi[d.mi], dh)
        if "v" in lay.sl: add(lay.sl["v"].start + d.mi, dh)
    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(d.n, lay.n))
    return r, J


def init_x(d, lay, seed=None):
    """Cold start: theta and beta (alpha = 1) from logits of row and column means, the rest at zero."""
    x = np.zeros(lay.n)
    keep = d.w > 0
    y = np.clip(d.y, 0.02, 0.98)
    lg = np.where(keep, np.log(y / (1 - y)), np.nan)
    th = pd.Series(lg).groupby(d.mi).mean().reindex(range(d.M)).fillna(0).to_numpy()
    de = -pd.Series(lg - th[d.mi]).groupby(d.bi).mean().reindex(range(d.B)).fillna(0).to_numpy()
    de = de - de[d.anchor]
    x[lay.sl["theta"]] = th - de[d.anchor]
    x[lay.sl["beta"]] = de[lay.bfree >= 0]
    if seed is not None:
        rng = np.random.default_rng(seed)
        x[lay.sl["theta"]] += rng.normal(0, 0.5, d.M)
        x[lay.sl["beta"]] += rng.normal(0, 0.5, d.B - 1)
        e = rng.normal(0, 0.2, d.B)
        x[lay.sl["e"]] += e - e.mean()
    return x


def fit(d, spec, x0=None, max_iter=1000, ftol=1e-13, gtol=1e-11, ptol=1e-7, dtol=1e-9):
    """Penalized least squares by Levenberg-Marquardt. Cost = 0.5 * (sum r^2 + sum lam x^2).

    Certificate: one undamped Gauss-Newton step from the returned point, reported as the largest change
    it would make to a fitted probability on any cell (held-out cells included), to m, to any parameter,
    and to the cost (gn_dec, the decrease it predicts: 0.5 g'H^-1 g). The fit also stops once the first
    two are below ptol, or gn_dec is below dtol: a model or benchmark whose scores all sit near zero can
    drift down a flat valley for thousands of iterations while the cost changes by less than 1e-5.
    Statistics built from costs (the test) are then accurate to about 2 gn_dec / sigma^2."""
    lay = Layout(d, spec)
    ones = np.ones(d.n)
    gap_h = h(d.gap, spec.tau, spec.x0)
    x = init_x(d, lay) if x0 is None else np.asarray(x0, float).copy()
    P = lay.P

    def evaluate(x, want_jac=True):
        r, J = _resid_jac(x, d, spec, lay, gap_h, want_jac)
        return r, J, 0.5 * (r @ r + x @ P @ x)

    r, J, cost = evaluate(x)
    H = (J.T @ J).toarray() + P
    g = J.T @ r + P @ x
    diag_floor = 1e-10 * max(np.max(np.diag(H)), 1.0)
    def certificate(x, H, g):
        try:
            step = -cho_solve(cho_factor(H + 1e-12 * np.eye(lay.n), check_finite=False), g, check_finite=False)
        except LinAlgError:
            return np.full(lay.n, np.inf), np.inf, np.inf
        return (step, float(np.max(np.abs(_resid_jac(x, d, spec, lay, gap_h, sw=ones)[1] @ step))),
                float(-0.5 * g @ step))

    mu, nu, status, small, nfev, acc = 1e-3, 2.0, "max_iter", 0, 1, 0
    for it in range(max_iter):
        if np.max(np.abs(g)) < gtol:
            status = "gtol"; break
        D = np.maximum(np.diag(H), diag_floor)
        try:
            cf = cho_factor(H + mu * np.diag(D), check_finite=False)
        except LinAlgError:
            mu *= nu; nu *= 2
            if mu > 1e16:
                status = "stalled"; break
            continue
        dx = -cho_solve(cf, g, check_finite=False)
        pred = -(g @ dx + 0.5 * dx @ (H @ dx))
        xn = x + dx
        rn, _, cn = evaluate(xn, want_jac=False); nfev += 1
        ratio = (cost - cn) / pred if pred > 0 else -1.0
        if ratio > 0:
            drop = cost - cn
            x, cost = xn, cn
            r, J, _ = evaluate(x)
            H = (J.T @ J).toarray() + P
            g = J.T @ r + P @ x
            mu *= max(1 / 3, 1 - (2 * ratio - 1) ** 3); nu = 2.0
            small = small + 1 if drop <= ftol * max(cost, 1e-300) else 0
            if small >= 2 or np.max(np.abs(dx)) < 1e-12:
                status = "ftol"; break
            acc += 1
            if acc % 10 == 0:
                step, dp_max, dec = certificate(x, H, g)
                if dec < dtol or (dp_max < ptol and ("m" not in lay.sl or abs(step[lay.sl["m"]][0]) < ptol)):
                    status = "gn_small"; break
        else:
            mu *= nu; nu *= 2
            if mu > 1e16:
                status = "stalled"; break
    step, dp_max, dec = certificate(x, H, g)
    out = dict(x=x, cost=cost, sse=float(r @ r), status=status, iters=it + 1, nfev=nfev,
               gmax=float(np.max(np.abs(g))), gn_step=float(np.max(np.abs(step))), gn_dp=dp_max, gn_dec=dec,
               gn_step_theta=float(np.max(np.abs(step[lay.sl["theta"]]))), spec=spec, lay=lay, H=H)
    if "m" in lay.sl:
        out["m"] = float(x[lay.sl["m"]][0]); out["gn_step_m"] = float(abs(step[lay.sl["m"]][0]))
    return out


def eci_scale(out, d):
    """Affine map theta -> ECI through the two anchor models; returns (a, b)."""
    th = out["x"][out["lay"].sl["theta"]]
    lo, hi = th[d.models.index(ANCHOR_LOW)], th[d.models.index(ANCHOR_HIGH)]
    b = (ECI_HIGH - ECI_LOW) / (hi - lo)
    return ECI_LOW - b * lo, b


def edf(out, d):
    """Effective degrees of freedom of the penalized fit: trace of J'J (J'J + Lambda)^-1."""
    lay = out["lay"]
    Hd = out["H"] - lay.P
    return float(np.trace(cho_solve(cho_factor(out["H"] + 1e-12 * np.eye(lay.n)), Hd)))


LAM_FIELD = {"gamma": "lam_g", "rho": "lam_r", "u": "lam_u", "v": "lam_v"}


def tune(d, spec, blocks=("gamma", "rho"), x0=None, max_rounds=40, rtol=1e-3):
    """Treat the named blocks as Gaussian random effects and set each ridge weight to sigma^2 / var(block)
    by the Fellner-Schall fixed point: lam <- sigma^2 (q - lam tr(H^-1)_bb) / |b|^2, with sigma^2 the
    residual variance SSE / (n - edf). Returns (fit, spec with tuned weights, rounds)."""
    n_obs = float((d.w > 0).sum())
    o = None
    for rnd in range(1, max_rounds + 1):
        o = fit(d, spec, x0=x0 if o is None else o["x"])
        lay, x = o["lay"], o["x"]
        Hinv = np.linalg.inv(o["H"])
        sigma2 = o["sse"] / (n_obs - float(np.trace(Hinv @ (o["H"] - lay.P))))
        new, change = {}, 0.0
        for b in blocks:
            if b not in lay.sl:
                continue
            sl, lam = lay.sl[b], getattr(spec, LAM_FIELD[b])
            q, bb = sl.stop - sl.start, float(x[sl] @ x[sl])
            lam_new = sigma2 * max(q - lam * np.trace(Hinv[sl, sl]), 1e-3 * q) / max(bb, 1e-12)
            lam_new = float(np.clip(lam_new, 1e-4, 1e8))
            new[LAM_FIELD[b]] = lam_new
            change = max(change, abs(np.log(lam_new / lam)))
        spec = spec.replace(**new)
        if change < rtol:
            break
    o = fit(d, spec, x0=o["x"])
    o["sigma2"], o["rounds"] = sigma2, rnd
    return o, spec, rnd
