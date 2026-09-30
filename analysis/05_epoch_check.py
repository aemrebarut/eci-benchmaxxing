"""Independent check of Epoch's ECI fit for an optimizer convergence problem.

Question: Epoch's fit_eci_model calls scipy.optimize.least_squares(method="trf")
with an analytical Jacobian returned as a scipy sparse matrix and no tr_solver,
so scipy picks tr_solver="lsmr". Does that stop short of the minimum of
Epoch's own objective, and does it matter on the ECI scale and for the
bootstrap CIs?

Only Epoch's own code and objective are used (the installed `eci` package,
commit e937446). Epoch's code is not modified. Instead the module-level name
eci.fitting.least_squares is swapped for a Recorder that either passes each
call through unchanged (recording the OptimizeResult) or re-solves the SAME
call (same residual function, same args, same bounds, same x0) with
tr_solver="exact" on a dense copy of Epoch's own analytical Jacobian. A
pass-through spy on scipy's internal trf() records which tr_solver scipy
actually used.

Usage, from the repo root:
  .venv/bin/python analysis/05_epoch_check.py central
  .venv/bin/python analysis/05_epoch_check.py boot --data published --solver lsmr --n 100
  .venv/bin/python analysis/05_epoch_check.py boot --data published --solver exact --n 100
  .venv/bin/python analysis/05_epoch_check.py summarize

Datasets:
  zip        prepare_benchmark_data(source="data/benchmark_data.zip"), i.e.
             the raw data in the 2026-09-29 snapshot (newer than the
             published scores).
  published  epoch_capabilities_index/processed_data_for_eci.csv from the
             same zip: Epoch's own fit input behind the published
             eci_scores.csv. Used to separate optimizer from data effects.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize._lsq.common import CL_scaling_vector
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import spearmanr, kendalltau

import eci
import eci.fitting as F
from eci import prepare_benchmark_data, fit_eci_model

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "benchmark_data.zip"
OUT = ROOT / "out" / "epoch_check"
ANCHORS = ("Claude 3.5 Sonnet", "GPT-5")
TIGHT = 1e-12
TOPK = 25

# ---------------------------------------------------------------------------
# Instrumentation (pass-through; changes nothing unless a Recorder asks to)
# ---------------------------------------------------------------------------
_ORIG_LS = F.least_squares
_LS_MOD = sys.modules["scipy.optimize._lsq.least_squares"]
_ORIG_TRF = _LS_MOD.trf
_TRF_LOG = []


def _trf_spy(*a, **k):
    """Pass-through wrapper of scipy's internal trf(); logs tr_solver."""
    _TRF_LOG.append(a[13] if len(a) > 13 else k.get("tr_solver"))
    return _ORIG_TRF(*a, **k)


_LS_MOD.trf = _trf_spy


class _Dense:
    """Epoch's analytical Jacobian, returned as a dense ndarray."""

    def __init__(self, jac):
        self.jac = jac

    def __call__(self, x, *args):
        return self.jac(x, *args).toarray()


class Recorder:
    """Stand-in for eci.fitting.least_squares.

    solver="passthrough": call scipy with Epoch's exact arguments.
    solver="exact": same call, but tr_solver="exact" and a dense Jacobian.
    solver="lsmr": same call, but tr_solver="lsmr" forced (plus tr_options).
    tol: if set, overrides ftol, xtol and gtol.
    x0_first: if set, replaces x0 of the first call only (the central fit).
    """

    def __init__(self, solver="passthrough", tol=None, x0_first=None,
                 tr_options=None, max_nfev=None):
        self.solver = solver
        self.tol = tol
        self.x0_first = None if x0_first is None else np.asarray(x0_first, float)
        self.tr_options = tr_options
        self.max_nfev = max_nfev
        self.calls = []
        self.fun = self.args = self.jac = self.bounds = None

    def __call__(self, fun, x0, *a, **kw):
        if not self.calls and self.x0_first is not None:
            x0 = self.x0_first.copy()
        if self.fun is None:
            self.fun, self.args = fun, kw.get("args", ())
            self.jac, self.bounds = kw.get("jac"), kw.get("bounds")
        kwargs = dict(kw)
        if self.solver == "exact":
            if callable(kwargs.get("jac")):
                kwargs["jac"] = _Dense(kwargs["jac"])
            kwargs["tr_solver"] = "exact"
        elif self.solver == "lsmr":
            kwargs["tr_solver"] = "lsmr"
            if self.tr_options is not None:
                kwargs["tr_options"] = dict(self.tr_options)
        if self.tol is not None:
            kwargs.update(ftol=self.tol, xtol=self.tol, gtol=self.tol)
        if self.max_nfev is not None:
            kwargs["max_nfev"] = self.max_nfev
        n0 = len(_TRF_LOG)
        t0 = time.perf_counter()
        res = _ORIG_LS(fun, x0, *a, **kwargs)
        dt = time.perf_counter() - t0
        self.calls.append(dict(
            call=len(self.calls),
            x0=np.asarray(x0, float).copy(),
            x=res.x.copy(),
            cost=float(res.cost),
            status=int(res.status),
            message=str(res.message),
            success=bool(res.success),
            nfev=int(res.nfev),
            njev=int(res.njev) if res.njev is not None else -1,
            optimality=float(res.optimality),
            n_active=int(np.count_nonzero(res.active_mask)),
            tr_solver=_TRF_LOG[n0] if len(_TRF_LOG) > n0 else None,
            seconds=round(dt, 3),
        ))
        return res


def run_fit(df, rec, **kw):
    F.least_squares = rec
    try:
        return fit_eci_model(df, **kw)
    finally:
        F.least_squares = _ORIG_LS


# ---------------------------------------------------------------------------
# Data and objective helpers
# ---------------------------------------------------------------------------
def load_data(name):
    if name == "zip":
        return prepare_benchmark_data(source=str(ZIP))
    if name == "published":
        with zipfile.ZipFile(ZIP) as z:
            return pd.read_csv(z.open(
                "epoch_capabilities_index/processed_data_for_eci.csv"))
    raise ValueError(name)


def load_published():
    with zipfile.ZipFile(ZIP) as z:
        return pd.read_csv(z.open("epoch_capabilities_index/eci_scores.csv"))


def seed_init(df, seed):
    """Epoch's init scheme (N(0, 0.1) capabilities and difficulties, unit
    discriminabilities) with a different seed; same layout as fit_eci_model."""
    n_models = df["model_id"].nunique()
    n_bench = df["benchmark_id"].nunique()
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 0.1, n_models),
                           rng.normal(0.0, 0.1, n_bench),
                           np.full(n_bench - 1, 1.0)])


def cost_at(ref, x):
    r = ref.fun(x, *ref.args)
    return 0.5 * float(r @ r)


def optimality_at(ref, x):
    """scipy's first-order optimality for bounded trf: ||g * v||_inf with
    g = J^T r from Epoch's analytical Jacobian and v the Coleman-Li scaling."""
    r = ref.fun(x, *ref.args)
    J = ref.jac(x, *ref.args)
    g = np.asarray(J.T @ r).ravel()
    lb, ub = ref.bounds
    v, _ = CL_scaling_vector(x, g, lb, ub)
    return float(np.max(np.abs(g * v)))


def eci_series(eci_df):
    return eci_df.set_index("Model")["eci"]


def compare(a, b, topk=TOPK):
    """Compare two ECI Series (index Model). b is the reference."""
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    free = ~common.isin(ANCHORS)
    d = (a - b)[free].abs()
    rho = spearmanr(a, b).statistic
    top = b.sort_values(ascending=False).index[:topk]
    ta, tb = a[top], b[top]
    swaps = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            s = np.sign(ta.iloc[i] - ta.iloc[j]) * np.sign(tb.iloc[i] - tb.iloc[j])
            if s < 0:
                swaps.append((top[i], top[j]))
    set_a = set(a.sort_values(ascending=False).index[:topk])
    rank_a = a.rank(ascending=False, method="min")
    rank_b = b.rank(ascending=False, method="min")
    return dict(
        n=int(free.sum()),
        max_abs=float(d.max()),
        mean_abs=float(d.mean()),
        median_abs=float(d.median()),
        argmax=str(d.idxmax()),
        spearman=float(rho),
        kendall=float(kendalltau(a, b).statistic),
        max_rank_change=int((rank_a - rank_b).abs().max()),
        n_rank_changed=int(((rank_a - rank_b) != 0).sum()),
        topk_swaps=len(swaps),
        topk_swap_pairs=swaps,
        topk_set_changes=sorted(set(top) ^ set_a),
    )


# ---------------------------------------------------------------------------
# Central fits
# ---------------------------------------------------------------------------
def summarize_call(name, rec, ref, eci_s, ref_eci=None):
    c = rec.calls[0]
    row = dict(run=name, tr_solver=c["tr_solver"], status=c["status"],
               success=c["success"], nfev=c["nfev"], njev=c["njev"],
               cost_reported=c["cost"], cost_epoch_obj=cost_at(ref, c["x"]),
               optimality_reported=c["optimality"],
               optimality_epoch_jac=optimality_at(ref, c["x"]),
               n_active=c["n_active"], seconds=c["seconds"],
               message=c["message"])
    if ref_eci is not None:
        cmp = compare(eci_s, ref_eci)
        row.update({f"vs_ref_{k}": v for k, v in cmp.items()
                    if k not in ("topk_swap_pairs", "topk_set_changes")})
        row["vs_ref_topk_swap_pairs"] = "; ".join(" > ".join(p) for p in cmp["topk_swap_pairs"])
    return row


def central(args):
    OUT.mkdir(parents=True, exist_ok=True)
    datasets = args.data
    meta = dict(python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, scipy=scipy.__version__,
                pandas=pd.__version__, eci=eci.__version__,
                eci_file=F.__file__)
    results = {}
    for dname in datasets:
        print(f"\n=== dataset {dname}", flush=True)
        df = load_data(dname)
        print("rows", len(df), "models", df.model_id.nunique(), "benchmarks",
              df.benchmark_id.nunique(), flush=True)
        runs = {}  # name -> (rec, eci Series)

        def go(name, rec, **kw):
            t0 = time.time()
            eci_df, edi_df, _ = run_fit(df, rec, bootstrap_samples=0, **kw)
            c = rec.calls[0]
            print(f"{name:28s} solver={c['tr_solver']} status={c['status']} "
                  f"nfev={c['nfev']} cost={c['cost']:.12g} opt={c['optimality']:.2e} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            runs[name] = (rec, eci_series(eci_df))
            eci_df.to_csv(OUT / f"central_{dname}_{name}.csv", index=False)
            return rec

        # 1. Epoch unmodified (pass-through recorder only)
        A = go("lsmr_default", Recorder("passthrough"))
        # 2. same objective, exact trust-region solver, dense Jacobian
        go("exact_tight", Recorder("exact", tol=TIGHT))
        go("exact_default_tol", Recorder("exact"))
        go("fd_default", Recorder("passthrough"), use_analytical_jacobian=False)
        go("exact_tight_from_lsmr", Recorder("exact", tol=TIGHT,
                                             x0_first=A.calls[0]["x"]))
        # candidate cheap fixes that keep lsmr
        go("lsmr_tight_tol", Recorder("lsmr", tol=TIGHT))
        go("lsmr_tight_tol_tropts", Recorder(
            "lsmr", tol=TIGHT, tr_options=dict(atol=1e-12, btol=1e-12, maxiter=10000)))
        go("lsmr_tropts_only", Recorder(
            "lsmr", tr_options=dict(atol=1e-10, btol=1e-10, maxiter=10000)))
        # (a) uniqueness: multistart, same N(0, 0.1) scheme, other seeds
        for seed in args.seeds:
            x0 = seed_init(df, seed)
            go(f"exact_tight_seed{seed}", Recorder("exact", tol=TIGHT, x0_first=x0))
            go(f"lsmr_default_seed{seed}", Recorder("passthrough", x0_first=x0))

        ref = A  # Epoch's residuals / Jacobian / bounds closures
        ref_eci = runs["exact_tight"][1]
        rows = [summarize_call(n, r, ref, s, ref_eci) for n, (r, s) in runs.items()]
        tab = pd.DataFrame(rows)
        tab.to_csv(OUT / f"central_{dname}_runs.csv", index=False)
        np.savez(OUT / f"central_{dname}_x.npz",
                 **{n: r.calls[0]["x"] for n, (r, _) in runs.items()})
        eci_all = pd.DataFrame({n: s for n, (_, s) in runs.items()})
        eci_all.to_csv(OUT / f"central_{dname}_eci_all.csv")

        # Jacobian sanity: Epoch's analytical Jacobian vs central differences
        xs = runs["exact_tight"][0].calls[0]["x"]
        Ja = ref.jac(xs, *ref.args).toarray()
        Jn = approx_derivative(lambda z: ref.fun(z, *ref.args), xs, method="3-point")
        jac_err = float(np.max(np.abs(Ja - Jn)))
        # parameter-space distances between solutions
        xa = A.calls[0]["x"]
        dx = float(np.linalg.norm(xa - xs))
        results[dname] = dict(
            rows=len(df), n_models=int(df.model_id.nunique()),
            n_benchmarks=int(df.benchmark_id.nunique()),
            n_params=int(xa.size), jac_max_abs_err_vs_fd=jac_err,
            x_dist_lsmr_exact=dx,
            cost_gap_lsmr_minus_exact=cost_at(ref, xa) - cost_at(ref, xs),
        )
        print(json.dumps(results[dname], indent=1), flush=True)
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(tab[["run", "tr_solver", "status", "nfev", "cost_epoch_obj",
                       "optimality_epoch_jac", "vs_ref_max_abs", "vs_ref_mean_abs",
                       "vs_ref_topk_swaps", "seconds"]].to_string(index=False), flush=True)
    meta["results"] = results
    (OUT / "central_meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def boot(args):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data)
    if args.solver == "lsmr":
        rec = Recorder("passthrough")
    else:
        rec = Recorder("exact", tol=args.tol)
    tag = f"boot_{args.data}_{args.solver}_n{args.n}"
    t0 = time.time()
    eci_df, edi_df, draws = run_fit(df, rec, bootstrap_samples=args.n,
                                    bootstrap_seed=args.seed)
    wall = time.time() - t0
    eci_df.to_csv(OUT / f"{tag}_eci.csv", index=False)
    c0 = rec.calls[0]
    x_hat = c0["x"]
    names = draws["model_names"]
    eci_hat = eci_df.set_index("Model").loc[names, "eci"].to_numpy()
    free = ~np.isin(names, ANCHORS)
    recs = []
    for i, c in enumerate(rec.calls[1:]):
        dE = np.abs(draws["eci"][i] - eci_hat)[free]
        recs.append(dict(draw=i, status=c["status"], success=c["success"],
                         nfev=c["nfev"], njev=c["njev"], cost=c["cost"],
                         optimality=c["optimality"], tr_solver=c["tr_solver"],
                         warm_start_is_xhat=bool(np.array_equal(c["x0"], x_hat)),
                         x_move_l2=float(np.linalg.norm(c["x"] - c["x0"])),
                         eci_move_max=float(dE.max()),
                         eci_move_mean=float(dE.mean()),
                         seconds=c["seconds"]))
    pd.DataFrame(recs).to_csv(OUT / f"{tag}_calls.csv", index=False)
    np.savez(OUT / f"{tag}_draws.npz", eci=draws["eci"], a=draws["a"],
             b=draws["b"], model_names=np.array(names), eci_hat=eci_hat,
             x_hat=x_hat, x_draws=np.array([c["x"] for c in rec.calls[1:]]))
    info = dict(data=args.data, solver=args.solver, n=args.n, seed=args.seed,
                tol=args.tol, wall_seconds=round(wall, 1),
                central=dict((k, c0[k]) for k in ("status", "nfev", "cost",
                                                  "optimality", "tr_solver")),
                all_success=bool(all(c["success"] for c in rec.calls)))
    (OUT / f"{tag}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def md(df, floatfmt=4):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append(f"{v:.{floatfmt}g}" if abs(v) < 1e-3 or abs(v) >= 1e5
                             else f"{v:.{floatfmt}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(args):
    out = []
    pub = load_published().set_index("Model")
    pub_hw = ((pub["eci_ci_high"] - pub["eci_ci_low"]) / 2).drop(list(ANCHORS))
    for dname in ("published", "zip"):
        p = OUT / f"central_{dname}_runs.csv"
        if not p.exists():
            continue
        tab = pd.read_csv(p)
        E = pd.read_csv(OUT / f"central_{dname}_eci_all.csv", index_col=0)
        out.append(f"## Central fits, dataset `{dname}`\n")
        cols = ["run", "tr_solver", "status", "nfev", "njev", "cost_epoch_obj",
                "optimality_epoch_jac", "vs_ref_max_abs", "vs_ref_mean_abs",
                "vs_ref_spearman", "vs_ref_topk_swaps", "seconds"]
        t = tab[cols].copy()
        cmin = t["cost_epoch_obj"].min()
        t.insert(6, "cost_minus_min", t["cost_epoch_obj"] - cmin)
        out.append(md(t, 6))
        out.append("")
        swaps = tab.set_index("run")["vs_ref_topk_swap_pairs"].dropna()
        for r, s in swaps.items():
            out.append(f"- top-{TOPK} order swaps, {r} vs exact_tight: {s}")
        # multistart spread
        ex = E[[c for c in E.columns if c.startswith("exact_tight")]]
        ls = E[[c for c in E.columns if c.startswith("lsmr_default")]]
        free = ~E.index.isin(ANCHORS)
        out.append("")
        out.append(f"- multistart exact_tight ({ex.shape[1]} starts): max over models "
                   f"of (max - min) ECI = {(ex.max(axis=1) - ex.min(axis=1))[free].max():.2e}; "
                   f"cost range = {tab[tab.run.str.startswith('exact_tight')].cost_epoch_obj.agg(lambda s: s.max() - s.min()):.3e}")
        out.append(f"- multistart lsmr_default ({ls.shape[1]} starts): max over models "
                   f"of (max - min) ECI = {(ls.max(axis=1) - ls.min(axis=1))[free].max():.4f}; "
                   f"median = {(ls.max(axis=1) - ls.min(axis=1))[free].median():.4f}; "
                   f"cost range = {tab[tab.run.str.startswith('lsmr_default')].cost_epoch_obj.agg(lambda s: s.max() - s.min()):.3e}")
        # published comparison
        common = E.index.intersection(pub.index)
        pf = common[~common.isin(ANCHORS)]
        rows = []
        for r in ("lsmr_default", "exact_tight", "fd_default"):
            d = (E.loc[pf, r] - pub.loc[pf, "eci"]).abs()
            cmp = compare(E.loc[common, r], pub.loc[common, "eci"])
            rows.append(dict(run=r, n_models=len(pf), max_abs_vs_published=d.max(),
                             mean_abs_vs_published=d.mean(),
                             n_within_0p005=int((d <= 0.0051).sum()),
                             spearman=cmp["spearman"], top25_swaps=cmp["topk_swaps"]))
        out.append("")
        out.append(f"Versus published eci_scores.csv (rounded to 0.01), dataset `{dname}`:\n")
        out.append(md(pd.DataFrame(rows), 4))
        # materiality vs published CI half-widths
        diff = (E.loc[pf, "lsmr_default"] - E.loc[pf, "exact_tight"]).abs()
        ratio = diff / pub_hw.reindex(pf)
        out.append("")
        out.append(f"- |lsmr - exact| / published 90% CI half-width ({dname}): "
                   f"median {ratio.median():.3f}, 90th pct {ratio.quantile(0.9):.3f}, "
                   f"max {ratio.max():.3f} ({ratio.idxmax()}), "
                   f"n > 0.25: {(ratio > 0.25).sum()}, n > 0.5: {(ratio > 0.5).sum()}, "
                   f"n > 1: {(ratio > 1).sum()}")
        top = diff.sort_values(ascending=False).head(8)
        out.append("- largest single-model shifts (lsmr - exact): " + "; ".join(
            f"{m}: {E.loc[m, 'lsmr_default'] - E.loc[m, 'exact_tight']:+.3f} "
            f"(exact {E.loc[m, 'exact_tight']:.2f}, published CI half-width "
            f"{pub_hw.get(m, np.nan):.2f})" for m in top.index))
        ex = E.loc[common, "exact_tight"].sort_values(ascending=False)
        signed = (E.loc[pf, "lsmr_default"] - E.loc[pf, "exact_tight"])
        for k in (TOPK, 100):
            members = [m for m in ex.index[:k] if m not in ANCHORS]
            out.append(f"- top-{k} (by exact ECI) max |lsmr - exact| = "
                       f"{signed.reindex(members).abs().max():.4f}")
        out.append(f"- smallest gap between adjacent top-{TOPK} models (exact) = "
                   f"{(-np.diff(ex.to_numpy()[:TOPK])).min():.4f}")
        out.append(f"- models with |lsmr - exact| > 0.05: {(diff > 0.05).sum()} "
                   f"(exact ECI range {ex.reindex(diff[diff > 0.05].index).min():.1f} to "
                   f"{ex.reindex(diff[diff > 0.05].index).max():.1f}); > 0.01: {(diff > 0.01).sum()}")
        out.append("")
    # bootstrap
    for dname in ("published", "zip"):
        runs = {}
        for s in ("lsmr", "exact"):
            cands = sorted(OUT.glob(f"boot_{dname}_{s}_n*_info.json"))
            if cands:
                tag = cands[-1].name[:-len("_info.json")]
                runs[s] = tag
        if not runs:
            continue
        out.append(f"## Bootstrap, dataset `{dname}`\n")
        widths = {}
        rows = []
        for s, tag in runs.items():
            info = json.loads((OUT / f"{tag}_info.json").read_text())
            e = pd.read_csv(OUT / f"{tag}_eci.csv").set_index("Model")
            w = (e["eci_ci_high"] - e["eci_ci_low"]).dropna()
            widths[s] = w
            calls = pd.read_csv(OUT / f"{tag}_calls.csv")
            rows.append(dict(
                solver=s, n=info["n"], all_success=info["all_success"],
                wall_s=info["wall_seconds"],
                width_median=w.median(), width_q10=w.quantile(0.1),
                width_q90=w.quantile(0.9), width_mean=w.mean(),
                nfev_median=calls.nfev.median(), nfev_min=calls.nfev.min(),
                nfev_max=calls.nfev.max(),
                status_counts=str(calls.status.value_counts().to_dict()),
                eci_move_max_median=calls.eci_move_max.median(),
                eci_move_mean_median=calls.eci_move_mean.median(),
                x_move_median=calls.x_move_l2.median()))
        out.append(md(pd.DataFrame(rows), 4))
        if "lsmr" in widths and "exact" in widths:
            wl, we = widths["lsmr"].align(widths["exact"], join="inner")
            r = wl / we
            out.append("")
            out.append(f"- per-model CI width ratio lsmr/exact: median {r.median():.3f}, "
                       f"q10 {r.quantile(0.1):.3f}, q90 {r.quantile(0.9):.3f}, "
                       f"min {r.min():.3f} ({r.idxmin()}), max {r.max():.3f}; "
                       f"lsmr narrower for {(r < 1).sum()} of {len(r)} models")
            dl = np.load(OUT / f"{runs['lsmr']}_draws.npz", allow_pickle=True)
            de = np.load(OUT / f"{runs['exact']}_draws.npz", allow_pickle=True)
            names = list(dl["model_names"])
            free = ~np.isin(names, ANCHORS)
            dd = np.abs(dl["eci"] - de["eci"])[:, free]
            out.append(f"- per-draw |lsmr refit - exact refit| (same resample): "
                       f"median over draws of max over models {np.median(dd.max(1)):.3f}, "
                       f"max {dd.max():.3f}; median over draws of mean {np.median(dd.mean(1)):.4f}")
            cl = pd.read_csv(OUT / f"{runs['lsmr']}_calls.csv")
            ce = pd.read_csv(OUT / f"{runs['exact']}_calls.csv")
            gap = cl["cost"].to_numpy() - ce["cost"].to_numpy()
            out.append(f"- per-draw cost gap lsmr - exact: > 0 in {(gap > 0).sum()} of "
                       f"{gap.size} draws; min {gap.min():.2e}, median {np.median(gap):.2e}, "
                       f"max {gap.max():.2e}")
            nb = int(round((dl["x_hat"].size - len(names) + 1) / 2))
            for lab, dx in (("lsmr", dl), ("exact", de)):
                xd = dx["x_draws"]
                out.append(f"- {lab}: draws with any parameter at the +-10 box: "
                           f"{(np.abs(xd) > 9.99).any(axis=1).sum()}; with a "
                           f"discriminability at the 0.1 floor: "
                           f"{(xd[:, -(nb - 1):] < 0.1001).any(axis=1).sum()}")
            sd_l = dl["eci"][:, free].std(0)
            sd_e = de["eci"][:, free].std(0)
            out.append(f"- per-model bootstrap SD ratio lsmr/exact: median "
                       f"{np.median(sd_l / sd_e):.3f}, q10 {np.quantile(sd_l / sd_e, 0.1):.3f}, "
                       f"q90 {np.quantile(sd_l / sd_e, 0.9):.3f}")
            if dname == "published":
                pw = (pub["eci_ci_high"] - pub["eci_ci_low"]).dropna()
                out.append(f"- published CI width (500 draws): median {pw.median():.3f}, "
                           f"q10 {pw.quantile(0.1):.3f}, q90 {pw.quantile(0.9):.3f}")
            # materiality against own exact CIs
            ecs = pd.read_csv(OUT / f"{runs['exact']}_eci.csv").set_index("Model")
            els = pd.read_csv(OUT / f"{runs['lsmr']}_eci.csv").set_index("Model")
            hw = ((ecs["eci_ci_high"] - ecs["eci_ci_low"]) / 2).dropna()
            dc = (els["eci"] - ecs["eci"]).abs().reindex(hw.index)
            rr = dc / hw
            out.append(f"- |lsmr - exact| central / exact-bootstrap CI half-width: "
                       f"median {rr.median():.3f}, max {rr.max():.3f} ({rr.idxmax()})")
        out.append("")
    text = "\n".join(out)
    (OUT / "tables.md").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("central")
    c.add_argument("--data", nargs="+", default=["published", "zip"])
    c.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    b = sub.add_parser("boot")
    b.add_argument("--data", default="published")
    b.add_argument("--solver", choices=["lsmr", "exact"], required=True)
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--seed", type=int, default=12345)
    b.add_argument("--tol", type=float, default=TIGHT)
    sub.add_parser("summarize")
    a = ap.parse_args()
    {"central": central, "boot": boot, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
