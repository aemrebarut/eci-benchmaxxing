"""Appendix studies: the objections a referee could raise against the step model (analysis/17_prepost.py and
analysis/19_drift.py, Google merged) and the analyses that answer them. Model code in analysis/robust.py. Each study
runs both structures, Epoch's ("epoch") and with benchmark drift ("drift", mu 0.1), and reports Google's step and
the contrast (1.13) of the report: Google minus the inverse-variance weighted mean of the other organizations.

Usage: python analysis/23_appendix.py <study> [reps]; studies:
  unpen    fits with every ridge removed (Epoch's on C, D and slopes, delta's, kappa's) and gauge rows instead
  lr       likelihood-ratio (F) tests and profile-likelihood intervals for Google's step and the contrast
  boot     pairs cluster bootstrap over models (anchors kept), percentile and bootstrap-t intervals
  wild     wild cluster bootstrap with the null imposed (Rademacher weights by model): p-values and the true size of
           the nominal 5% Wald tests
  wildb    the same with the weights and the clustered SE by benchmark
  twoway   sandwich SEs clustered by model, by benchmark, by cell, and two-way (model and benchmark)
  drop     joint deletion of the 3 or 5 most influential models or benchmarks (needs jack)
  labbench lab-by-benchmark effects for the seven largest labs (a triple difference for the steps)
  filter   adversarially filtered benchmarks: dropped, their filter-model cells dropped or flagged
  jack     delete-one-model and delete-one-benchmark jackknife (CV3 standard errors, influence)
  perm     organization labels permuted among models within release half-years (randomization test, max over labs)
  placebo  placebo thresholds, donuts around the release date and bandwidths
  side     placebo cutoffs within one side of the release date (pre cells only, post cells only)
  source   developer-reported scores removed; Epoch's own runs only
  trends   lab- and model-specific trends in benchmark date (the step must then be a discontinuity)
  quasi    quasi-binomial weights, common and per-benchmark dispersion
  select   the selection term (expected maximum of the raw rows behind a cell)
  shape    for Google, a step at the release date against a smooth quadratic in benchmark date (lab trends in)
  misc     multiplicity (max over labs), effect size in score points
Writes out/app23_<study>.csv (and .md where useful). Run under nice -n 10; at most 4 workers."""
import os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
import robust as R
from robust import GOOG, MAIN, STRUCT
from prepost import org_groups
from eci.fitting import DEFAULT_ANCHOR_MODEL_LOW, DEFAULT_ANCHOR_MODEL_HIGH, DEFAULT_ANCHOR_BENCHMARK

WORKERS = 4
LAM = 1e-3
OUT = "out/app23_{}.csv"


def summary(PP, res, lam, mu, w=None, unpen=False, c=None, tag=None):
    """Google's step, the contrast and their standard errors from a fit."""
    d, Vc, Vm, Vs, b, edf = R.cov_unpen(PP, res.xf, lam, mu, unpen, w)
    g = PP.ident.index(GOOG)
    c = R.contrast_vec(PP.ident, np.diag(Vm)) if c is None else c
    e = np.zeros(PP.K); e[g] = 1
    row = dict(status=res.status, nfev=res.nfev, rss=PP.rss(res.xf), edf=edf, b=b, K=PP.K)
    for name, v in [("g", e), ("c", c)]:
        for vt, V in [("mb", Vm), ("cl", Vc), ("sw", Vs)]:
            est, se, z = R.lin_stat(v, d, V)
            row[f"{name}_est"], row[f"{name}_se_{vt}"], row[f"{name}_z_{vt}"] = est, se, z
    row["others_mean"] = float(-(c * (np.arange(PP.K) != g)) @ d)
    xi = res.x
    idx = PP.blocks(True, mu)
    row["at_bounds"] = int(((xi <= PP.LB[idx] + 1e-7) | (xi >= PP.UB[idx] - 1e-7)).sum())
    if tag:
        row.update(tag)
    return row, d, Vm, Vc, c


def fit_summary(df, struct, unpen=False):
    PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], unpen=unpen)
    row, *_ = summary(PP, res, lam, mu, unpen=unpen)
    row["struct"] = struct
    return row


def pool(fn, tasks, init=None, initargs=()):
    with Pool(WORKERS, initializer=init, initargs=initargs) as p:
        return p.map(fn, tasks, chunksize=1)


# ---- unpen ---------------------------------------------------------------------------------------------------------
def _unpen(task):
    struct, unpen = task
    df = R.load()
    t = time.time()
    row = fit_summary(df, struct, unpen=unpen)
    row.update(setting="no ridge at all" if unpen else "near-unpenalized (headline)", secs=time.time() - t)
    return row


def study_unpen():
    rows = pool(_unpen, [(s, u) for s in STRUCT for u in (False, True)])
    return pd.DataFrame(rows)


# ---- lr ------------------------------------------------------------------------------------------------------------
def _lr(task):
    struct, unpen, target = task
    df = R.load()
    mu = STRUCT[struct]
    PP, base, res, lam, mu2 = R.fit(df, mu=mu, unpen=unpen)
    row, d, Vm, Vc, c = summary(PP, res, lam, mu2, unpen=unpen)
    if target == "google":
        c = np.zeros(PP.K); c[PP.ident.index(GOOG)] = 1
    est, se = row["g_est" if target == "google" else "c_est"], row["g_se_mb" if target == "google" else "c_se_mb"]

    def obj(x):
        con, PP.con = PP.con, None
        idx = PP.blocks(True, mu2)
        old = R.prepost.EPOCH_REG
        if unpen:
            R.prepost.EPOCH_REG = 0.0
        r = PP.resid(x[idx], PP.ones, lam, True, mu2)
        R.prepost.EPOCH_REG = old
        PP.con = con
        return float(r @ r)

    o_hat, rss_hat = obj(res.xf), PP.rss(res.xf)
    n, edf = PP.n, row["edf"]
    s2 = rss_hat / (n - edf)
    up = sorted({v for v in est + se * np.linspace(0, 4.5, 19)} | ({0.0} if est < 0 else set()))
    dn = sorted({v for v in est - se * np.linspace(0, 4.5, 19)} | ({0.0} if est > 0 else set()), reverse=True)
    out = []
    for path in (up, dn):
        x = res.xf
        for v in path:
            PP.set_con(c, v)
            r2 = R.refit(PP, x, lam, mu2, unpen)
            x = r2.xf
            _, b = PP.scale(x)
            achieved = float(b * (c @ x[PP.n_ep:PP.n_ep + PP.K]))
            o = obj(x)
            out.append(dict(struct=struct, unpen=unpen, target=target, v=v, achieved=achieved, obj=o,
                            rss=PP.rss(x), F=(o - o_hat) / s2, status=r2.status, est=est, se_mb=se, s2=s2, n=n, edf=edf))
    PP.set_con(None)
    return out


def study_lr():
    tasks = [(s, u, t) for s in STRUCT for u in (True, False) for t in ("google", "contrast")]
    res = pool(_lr, tasks)
    return pd.DataFrame([r for rr in res for r in rr]).drop_duplicates(["struct", "unpen", "target", "v"])


# ---- boot ----------------------------------------------------------------------------------------------------------
def _init_data():
    global DF
    DF = R.load()


def resample_models(df, rng):
    anchors = [DEFAULT_ANCHOR_MODEL_LOW, DEFAULT_ANCHOR_MODEL_HIGH]
    others = [m for m in dict.fromkeys(df.Model) if m not in anchors]
    draw = rng.choice(len(others), len(others), replace=True)
    by = {m: g for m, g in df.groupby("Model", sort=False)}
    parts = [by[a] for a in anchors]
    for k, i in enumerate(draw):
        p = by[others[i]].copy()
        p["Model"] = f"{others[i]}#{k}"
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    out["z"] = out.gap - out.groupby("benchmark").gap.transform("mean")
    return out


def _boot(task):
    struct, rep = task
    rng = np.random.default_rng(10_000 + rep)
    df = resample_models(DF, rng)
    try:
        PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct])
        if GOOG not in PP.ident:
            return dict(struct=struct, rep=rep, ok=False)
        row, *_ = summary(PP, res, lam, mu)
        row.update(struct=struct, rep=rep, ok=True)
        return row
    except Exception as e:
        return dict(struct=struct, rep=rep, ok=False, err=str(e)[:200])


def study_boot(reps):
    return pd.DataFrame(pool(_boot, [(s, r) for s in STRUCT for r in range(reps)], _init_data))


# ---- wild ----------------------------------------------------------------------------------------------------------
def _init_wild(spec):
    global WILD, DF
    DF = R.load()
    WILD = {}
    for struct, hyp, x_hat, x0, c in spec:
        PP = R.PPX(DF, DF.group, trend=DF.z if STRUCT[struct] is not None else None)
        yhat0 = PP.predict(x0)
        WILD[(struct, hyp)] = (PP, x_hat, yhat0, PP.y.copy() - yhat0, c)


def _wild(task):
    struct, hyp, rep = task[:3]
    by = task[3] if len(task) > 3 else "model"
    PP, x_hat, yhat0, r0, c = WILD[(struct, hyp)]
    PP.cluster = by
    if by == "model":
        rng = np.random.default_rng(20_000 + rep)
        v = rng.choice([-1.0, 1.0], PP.M)[PP.mi]
    else:
        rng = np.random.default_rng(40_000 + rep)
        v = rng.choice([-1.0, 1.0], PP.B)[PP.bi]
    y0 = PP.y
    PP.y = yhat0 + v * r0
    try:
        res = R.refit(PP, x_hat, LAM, STRUCT[struct])
        d, Vc, Vm, Vs, b, edf = PP.deltas(res.xf, LAM, STRUCT[struct])
        est, se_mb, _ = R.lin_stat(c, d, Vm)
        se_cl = float(np.sqrt(c @ Vc @ c))
        return dict(struct=struct, hyp=hyp, rep=rep, by=by, est=est, t_mb=est / se_mb, t_cl=est / se_cl, status=res.status)
    finally:
        PP.y = y0


def study_wild(reps, by="model"):
    """by: the cluster of the Rademacher weights and of the clustered SE, "model" or "bench"."""
    df = R.load()
    spec, obs = [], []
    for struct, mu in STRUCT.items():
        PP, base, res, lam, mu2 = R.fit(df, mu=mu)
        row, d, Vm, Vc, cc = summary(PP, res, lam, mu2)
        if by != "model":
            PP.cluster = by
            d, Vc, Vm, Vs, b, edf = PP.deltas(res.xf, lam, mu2)
        e = np.zeros(PP.K); e[PP.ident.index(GOOG)] = 1
        for hyp, c in [("contrast", cc), ("google", e)]:
            PP.set_con(c, 0.0)
            r0 = R.refit(PP, res.xf, lam, mu2)
            PP.set_con(None)
            est, se_mb, t_mb = R.lin_stat(c, d, Vm)
            obs.append(dict(struct=struct, hyp=hyp, rep=-1, by=by, est=est, t_mb=t_mb, t_cl=est / float(np.sqrt(c @ Vc @ c)),
                            status=r0.status, restricted_rss=PP.rss(r0.xf), rss=PP.rss(res.xf)))
            spec.append((struct, hyp, res.xf, r0.xf, c))
    tasks = [(s, h, r, by) for s in STRUCT for h in ("contrast", "google") for r in range(reps if h == "contrast" else reps // 2)]
    with Pool(WORKERS, initializer=_init_wild, initargs=(spec,)) as p:
        rows = p.map(_wild, tasks, chunksize=4)
    return pd.DataFrame(obs + rows)


def study_wildb(reps):
    return study_wild(reps, by="bench")


def study_twoway():
    """Sandwich SEs clustered by model, by benchmark and by cell (heteroskedasticity-robust), and the two-way
    variance V_model + V_bench - V_cell of Cameron, Gelbach and Miller (2011), from the headline fits."""
    df = R.load()
    rows = []
    for struct, mu in STRUCT.items():
        PP, base, res, lam, mu2 = R.fit(df, mu=mu)
        out = {}
        for by in ("model", "bench", "cell"):
            PP.cluster = by
            d, Vc, Vm, Vs, b, edf = PP.deltas(res.xf, lam, mu2)
            out[by] = Vc
        c = R.contrast_vec(PP.ident, np.diag(Vm))
        e = np.zeros(PP.K); e[PP.ident.index(GOOG)] = 1
        for name, v in [("contrast", c), ("google", e)]:
            est = float(v @ d)
            se = {k: float(np.sqrt(v @ V @ v)) for k, V in out.items()}
            two = float(v @ (out["model"] + out["bench"] - out["cell"]) @ v)
            rows.append(dict(struct=struct, target=name, est=est, se_mb=float(np.sqrt(v @ Vm @ v)),
                             **{f"se_{k}": x for k, x in se.items()}, se_twoway=np.sqrt(max(two, 0)), G_model=PP.M, G_bench=PP.B))
    t = pd.DataFrame(rows)
    for k in ("mb", "model", "bench", "cell", "twoway"):
        t[f"z_{k}"] = t.est / t[f"se_{k}"]
    return t


def _drop(task):
    struct, k, kind = task
    j = pd.read_csv(OUT.format("jack"))
    j = j[(j.struct == struct) & (j.kind == kind) & j.ok].sort_values("c_est")
    items = list(j.item.head(k))
    df = DF[~(DF.Model if kind == "model" else DF.benchmark).isin(items)].copy()
    df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    df = df.reset_index(drop=True)
    anchor = None if DEFAULT_ANCHOR_BENCHMARK in set(df.benchmark) else "GPQA diamond"
    PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], anchor=anchor)
    row, *_ = summary(PP, res, lam, mu)
    row.update(struct=struct, kind=kind, k=k, dropped="; ".join(items))
    return row


def study_drop():
    """Joint deletion of the k models (or benchmarks) whose single deletion lowers the contrast most (from jack)."""
    return pd.DataFrame(pool(_drop, [(s, k, kind) for s in STRUCT for kind in ("model", "bench") for k in (3, 5)], _init_data))


# ---- labbench -----------------------------------------------------------------------------------------------------
def labbench_X(df, min_cells=2):
    """One column per (lab, benchmark) for the seven largest labs: removes any lab-specific benchmark profile, so the
    steps are identified only by a lab's models released before and after a benchmark, on that benchmark."""
    cols, names = [], []
    for g in MAIN:
        for b, n in df[df.group == g].benchmark.value_counts().items():
            if n >= min_cells:
                cols.append(((df.group == g) & (df.benchmark == b)).to_numpy(float)); names.append((g, b))
    return np.column_stack(cols), names


def _labbench(task):
    struct, lam_x = task
    X, names = labbench_X(DF)
    PP, base, res, lam, mu = R.fit(DF, mu=STRUCT[struct], X=X, lam_x=lam_x)
    row, *_ = summary(PP, res, lam, mu)
    row.update(struct=struct, lam_x=lam_x, q=X.shape[1])
    # Google models with cells on both sides of some benchmark's release, among Google's cells on that benchmark
    gd = DF[DF.group == GOOG]
    row["google_bench_both"] = int(gd.groupby("benchmark").gap.agg(lambda s: (s > 0).any() and (s <= 0).any()).sum())
    return row


def study_labbench():
    return pd.DataFrame(pool(_labbench, [(s, lx) for s in STRUCT for lx in (1e-3, 1e-2)], _init_data))


# ---- filter --------------------------------------------------------------------------------------------------------
# Benchmarks whose items were selected against named LLMs, with the filter models' cells in Epoch's data (all are pre
# cells). Sources (checked 2026-10-01): HLE, arXiv 2501.14249 app. B.1 (GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, o1,
# o1-mini, o1-preview); SimpleQA, arXiv 2411.04368 (GPT-4 models, GPT-3.5) and SimpleQA Verified, arXiv 2509.07968
# (adds GPT-4o, Gemini 2.0 Flash, Claude 3.7 Sonnet; no cells for the last two); Aider polyglot,
# aider.chat/2024/12/21/polyglot.html (225 problems solved by at most 3 of 7 models); APEX-Agents, arXiv 2601.14242
# (GPT-5, Claude Opus 4.5, Gemini 3 Pro; no cells); FrontierMath Tier 4 (o4-mini bounty, per Scientific American).
FILTER = {
    "HLE": ["Gemini 1.5 Pro (Sept 2024)", "Claude 3.5 Sonnet (October 2024)", "GPT-4o (Nov 2024)", "o1"],
    "SimpleQA Verified": ["GPT-4o (Aug 2024)", "GPT-4o mini"],
    "Aider polyglot": ["GPT-4o mini", "GPT-4o (Aug 2024)", "o1-mini", "Claude 3.5 Haiku", "Claude 3.5 Sonnet (October 2024)",
                       "GPT-4o (Nov 2024)"],
    "APEX-Agents": [],
    "FrontierMath-Tier-4-2025-07-01-Private": [],
    "FrontierMath-Tier-4-v2-Private": ["o4-mini"],
}
# Model-based selection with unnamed models (hardest items by model scores, judge disagreement) or at task level (BBH).
FILTER_UNNAMED = ["LMCA", "FrontierCode", "FrontierSWE", "DTBench", "MirrorCode", "BBH"]
# Filtered against non-LLM or pre-2020 models (n-gram and RNN LMs, IR solvers, BERT, RoBERTa, Kaggle programs, Stockfish).
FILTER_NONLLM = ["LAMBADA", "ARC AI2", "OpenBookQA", "HellaSwag", "Winogrande", "ANLI", "PIQA", "ARC-AGI-2", "Chess Puzzles"]


def filter_mask(df):
    return np.array([m in FILTER.get(b, ()) for m, b in zip(df.Model, df.benchmark)])


def _filter(task):
    struct, kind = task
    df = DF.copy()
    fm = filter_mask(df)
    X, names = None, []
    if kind == "drop named-filter benchmarks":
        df = df[~df.benchmark.isin(list(FILTER))]
    elif kind == "drop all model-filtered benchmarks":
        df = df[~df.benchmark.isin(list(FILTER) + FILTER_UNNAMED)]
    elif kind == "drop filter cells":
        df = df[~fm]
    elif kind == "drop pre cells on named-filter benchmarks":
        df = df[~(df.benchmark.isin(list(FILTER)) & (df.gap <= 0))]
    elif kind == "drop non-LLM filtered benchmarks":
        df = df[~df.benchmark.isin(FILTER_NONLLM)]
    elif kind == "flag filter cells":
        X, names = fm.astype(float)[:, None], ["all"]
    elif kind == "flag filter cells by lab":
        labs = sorted(df.loc[fm, "group"].unique())
        X = np.column_stack([(fm & (df.group == g).to_numpy()).astype(float) for g in labs]); names = labs
    df = df.reset_index(drop=True)
    if X is None:
        df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    anchor = None if DEFAULT_ANCHOR_BENCHMARK in set(df.benchmark) else "GPQA diamond"
    PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], X=X, anchor=anchor)
    row, *_ = summary(PP, res, lam, mu)
    row.update(struct=struct, kind=kind, cells=len(df), benchmarks=df.benchmark.nunique(), filter_cells=int(fm.sum()))
    if X is not None:
        idx, Vc, Vm, Vs, b, edf = PP.cov2(res.xf, lam, mu)
        pos = list(idx)
        row["flags"] = "; ".join(f"{n}: {b * res.xf[PP.P0 + k]:+.2f} +- {b * np.sqrt(Vm[pos.index(PP.P0 + k)] [pos.index(PP.P0 + k)]):.2f}"
                                 for k, n in enumerate(names))
    return row


def study_filter():
    kinds = ["drop named-filter benchmarks", "drop all model-filtered benchmarks", "drop filter cells", "flag filter cells",
             "flag filter cells by lab", "drop pre cells on named-filter benchmarks", "drop non-LLM filtered benchmarks"]
    return pd.DataFrame(pool(_filter, [(s, k) for s in STRUCT for k in kinds], _init_data))


# ---- jack ----------------------------------------------------------------------------------------------------------
def _jack(task):
    struct, kind, item = task
    if kind == "model":
        df = DF[DF.Model != item].reset_index(drop=True)
        anchor = None
    else:
        df = DF[DF.benchmark != item].reset_index(drop=True)
        anchor = "GPQA diamond" if item == DEFAULT_ANCHOR_BENCHMARK else None
    df = df.copy()
    df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    try:
        PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], anchor=anchor)
        row, *_ = summary(PP, res, lam, mu)
        row.update(struct=struct, kind=kind, item=item, org=DF.loc[DF.Model == item, "group"].iloc[0] if kind == "model" else "", ok=True)
        return row
    except Exception as e:
        return dict(struct=struct, kind=kind, item=item, ok=False, err=str(e)[:200])


def study_jack():
    df = R.load()
    anchors = {DEFAULT_ANCHOR_MODEL_LOW, DEFAULT_ANCHOR_MODEL_HIGH}
    tasks = [(s, "model", m) for s in STRUCT for m in dict.fromkeys(df.Model) if m not in anchors]
    tasks += [(s, "bench", b) for s in STRUCT for b in dict.fromkeys(df.benchmark)]
    return pd.DataFrame(pool(_jack, tasks, _init_data))


# ---- perm ----------------------------------------------------------------------------------------------------------
def model_table(df):
    t = df.groupby("Model", sort=False).agg(org=("org", "first"), date=("date", "max"))
    d = pd.to_datetime(t.date)
    t["stratum"] = d.dt.year.astype(str) + "H" + np.where(d.dt.month <= 6, "1", "2")
    return t


def all_contrasts(PP, d, Vm):
    """Each identified group against the precision-weighted others, model-based z."""
    out = {}
    for lab in PP.ident:
        c = R.contrast_vec(PP.ident, np.diag(Vm), target=lab)
        est, se, z = R.lin_stat(c, d, Vm)
        out[lab] = (est, z)
    return out


def _init_perm():
    global DF, MT, BASE
    DF = R.load()
    MT = model_table(DF)
    BASE = {}
    for struct, mu in STRUCT.items():
        PP, base, res, lam, mu2 = R.fit(DF, mu=mu)
        BASE[struct] = base.xf[:PP.n_ep].copy()


def _perm(task):
    struct, rep = task
    rng = np.random.default_rng(30_000 + rep)
    lab = MT.org.copy()
    if rep >= 0:
        for s, idx in MT.groupby("stratum").groups.items():
            lab.loc[idx] = rng.permutation(lab.loc[idx].to_numpy())
    df = DF.copy()
    df["org"] = df.Model.map(lab)
    df["group"] = org_groups(df, "gmerge")
    mu = STRUCT[struct]
    PP = R.PPX(df, df.group, trend=df.z if mu is not None else None)
    res = PP.fit(PP.ones, LAM, BASE[struct], mu=mu)
    d, Vc, Vm, Vs, b, edf = PP.deltas(res.xf, LAM, mu)
    con = all_contrasts(PP, d, Vm)
    big = [g for g in PP.ident if (df[df.group == g].Model.nunique() >= 5)]
    zs = {g: con[g][1] for g in big}
    row = dict(struct=struct, rep=rep, status=res.status, g_contrast=con[GOOG][0] if GOOG in con else np.nan,
               g_z=con[GOOG][1] if GOOG in con else np.nan, max_abs_z=max(abs(v) for v in zs.values()),
               max_z=max(zs.values()), argmax=max(zs, key=lambda k: abs(zs[k])), n_big=len(big))
    if rep < 0:
        row["all"] = "; ".join(f"{k}: {con[k][0]:+.2f} (z {con[k][1]:+.2f})" for k in PP.ident)
    return row


def study_perm(reps):
    return pd.DataFrame(pool(_perm, [(s, r) for s in STRUCT for r in range(-1, reps)], _init_perm))


# ---- placebo -------------------------------------------------------------------------------------------------------
def _placebo(task):
    struct, kind, val = task
    df = DF.copy()
    if kind == "shift":
        df["gap"] = df.gap - val
    elif kind == "donut":
        df = df[df.gap.abs() >= val].reset_index(drop=True)
    elif kind == "bandwidth":
        df = df[df.gap.abs() <= val].reset_index(drop=True)
        df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    anchor = None if DEFAULT_ANCHOR_BENCHMARK in set(df.benchmark) else df.benchmark.value_counts().index[0]
    try:
        PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], anchor=anchor)
        row, *_ = summary(PP, res, lam, mu)
        row.update(struct=struct, kind=kind, val=val, cells=len(df), anchor=anchor or DEFAULT_ANCHOR_BENCHMARK, ok=True)
        return row
    except Exception as e:
        return dict(struct=struct, kind=kind, val=val, ok=False, err=str(e)[:200])


def study_placebo():
    shifts = [-1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    tasks = [(s, "shift", v) for s in STRUCT for v in shifts]
    tasks += [(s, "donut", v) for s in STRUCT for v in (1 / 12, 1 / 6, 0.25, 0.5)]
    tasks += [(s, "bandwidth", v) for s in STRUCT for v in (1, 1.5, 2, 3, 4)]
    return pd.DataFrame(pool(_placebo, tasks, _init_data))


# ---- source --------------------------------------------------------------------------------------------------------
def _side(task):
    """Placebo cutoffs on one side of the release date only (Imbens and Lemieux 2008, section 7.1): the pre cells
    (benchmark released after the model) with a fake cutoff val < 0 years, or the post cells with val > 0."""
    struct, side, val = task
    df = DF[(DF.gap <= 0) if side == "pre" else (DF.gap > 0)].copy()
    df["gap"] = df.gap - val
    df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    df = df.reset_index(drop=True)
    anchor = None if DEFAULT_ANCHOR_BENCHMARK in set(df.benchmark) else df.benchmark.value_counts().index[0]
    try:
        PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], anchor=anchor)
        row, *_ = summary(PP, res, lam, mu)
        gm = df.loc[df.group == GOOG, "Model"].unique()
        row.update(struct=struct, side=side, val=val, cells=len(df), anchor=anchor or DEFAULT_ANCHOR_BENCHMARK,
                   google_models_both=int(PP.both.reindex(gm).fillna(False).sum()),
                   models_both=int(PP.both.sum()), ok=True)
        return row
    except Exception as e:
        return dict(struct=struct, side=side, val=val, ok=False, err=str(e)[:300])


def study_side():
    tasks = [(s, "pre", v) for s in STRUCT for v in (-0.25, -0.5, -0.75, -1.0)]
    tasks += [(s, "post", v) for s in STRUCT for v in (0.25, 0.5, 0.75, 1.0, 1.5)]
    return pd.DataFrame(pool(_side, tasks, _init_data))


def source_cells():
    """For each cell of the main data: the kind of the row behind its max, and its max over each filtered set."""
    import cm
    df = R.load()
    s = R.rows()
    s = s.rename(columns={"model_group": "Model"})
    top = s.sort_values("clipped", ascending=False).drop_duplicates(["Model", "benchmark"])[["Model", "benchmark", "kind"]]
    df = df.merge(top.rename(columns={"kind": "max_kind"}), on=["Model", "benchmark"], how="left")
    for name, keep in [("indep", ("epoch", "third party")), ("epoch_only", ("epoch",))]:
        m = s[s.kind.isin(keep)].groupby(["Model", "benchmark"]).clipped.max().rename(f"perf_{name}")
        df = df.merge(m.reset_index(), on=["Model", "benchmark"], how="left")
    return df


def _source(task):
    struct, name = task
    df = SRC.copy()
    if name != "all":
        df = df[df[f"perf_{name}"].notna()].copy()
        df["y_fit"] = df[f"perf_{name}"].clip(R_CLIP, 1 - R_CLIP)
        nb = df.groupby("Model").benchmark.nunique()
        df = df[df.Model.isin(nb[nb >= 4].index)].reset_index(drop=True)
        df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
    anchor = None if DEFAULT_ANCHOR_BENCHMARK in set(df.benchmark) else df.benchmark.value_counts().index[0]
    try:
        PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], anchor=anchor)
        row, *_ = summary(PP, res, lam, mu)
        row.update(struct=struct, data=name, cells=len(df), models=df.Model.nunique(), benchmarks=df.benchmark.nunique(),
                   google_models_both=int(PP.both.reindex(df.loc[df.group == GOOG, "Model"].unique()).sum()), ok=True)
        return row
    except Exception as e:
        return dict(struct=struct, data=name, ok=False, err=str(e)[:300])


def _init_source(path):
    global SRC, R_CLIP
    import cm
    SRC = pd.read_pickle(path)
    R_CLIP = cm.CLIP


def study_source():
    df = source_cells()
    path = "out/app23_source_cells.pkl"
    df.to_pickle(path)
    comp = (df.assign(google=df.group == GOOG, post=df.gap > 0)
              .groupby(["google", "post"]).max_kind.value_counts(normalize=True).unstack().round(3))
    print(comp.to_string())
    comp.to_csv("out/app23_source_composition.csv")
    with Pool(WORKERS, initializer=_init_source, initargs=(path,)) as p:
        rows = p.map(_source, [(s, n) for s in STRUCT for n in ("all", "indep", "epoch_only")], chunksize=1)
    return pd.DataFrame(rows)


# ---- trends --------------------------------------------------------------------------------------------------------
def trend_X(df, kind):
    s = df.bdate.to_numpy() - 2024.0
    post = (df.gap > 0).to_numpy(float)
    cols, names = [], []
    if kind in ("lab linear", "lab quadratic", "lab sides"):
        for g in MAIN:
            ind = (df.group == g).to_numpy(float)
            if kind == "lab sides":
                cols += [ind * s * post, ind * s * (1 - post)]; names += [f"{g} post", f"{g} pre"]
            else:
                cols.append(ind * s); names.append(g)
                if kind == "lab quadratic":
                    cols.append(ind * s ** 2); names.append(f"{g} sq")
    elif kind == "model linear":
        sc = s - df.groupby("Model").bdate.transform("mean").to_numpy() + 2024.0
        for m in dict.fromkeys(df.Model):
            ind = (df.Model == m).to_numpy(float)
            if df.loc[df.Model == m, "bdate"].nunique() >= 3:
                cols.append(ind * sc); names.append(m)
    return np.column_stack(cols), names


def _trends(task):
    struct, kind, lam_x = task
    X, names = trend_X(DF, kind)
    PP, base, res, lam, mu = R.fit(DF, mu=STRUCT[struct], X=X, lam_x=lam_x)
    row, *_ = summary(PP, res, lam, mu)
    beta = res.xf[PP.P0:]
    _, b = PP.scale(res.xf)
    row.update(struct=struct, kind=kind, lam_x=lam_x, q=X.shape[1],
               betas="; ".join(f"{n}: {b * v:+.2f}" for n, v in zip(names, beta)) if X.shape[1] <= 14 else "")
    return row


def study_trends():
    tasks = [(s, k, 0.0) for s in STRUCT for k in ("lab linear", "lab quadratic", "lab sides")]
    tasks += [(s, "model linear", lx) for s in STRUCT for lx in (1e-3, 1e-2)]
    return pd.DataFrame(pool(_trends, tasks, _init_data))


# ---- shape --------------------------------------------------------------------------------------------------------
def shape_X(df, google_quad, other_quad=False):
    """Linear trends in benchmark date for the seven largest labs, plus quadratics for Google and/or the others."""
    s = df.bdate.to_numpy() - 2024.0
    cols = []
    for g in MAIN:
        ind = (df.group == g).to_numpy(float)
        cols.append(ind * s)
        if (g == GOOG and google_quad) or (g != GOOG and other_quad):
            cols.append(ind * s ** 2)
    return np.column_stack(cols)


def _shape(task):
    """Google's step free or pinned at 0 (by the constraint row), with or without a Google quadratic in benchmark
    date; every fit has the seven labs' linear trends and the other labs' steps."""
    struct, step, quad, oq = task
    X = shape_X(DF, quad, oq)
    PP, base, res, lam, mu = R.fit(DF, mu=STRUCT[struct], X=X)
    g = PP.ident.index(GOOG)
    if not step:
        e = np.zeros(PP.K); e[g] = 1
        PP.set_con(e, 0.0)
        res = R.refit(PP, res.xf, lam, mu)
        PP.set_con(None)
    _, b = PP.scale(res.xf)
    row = dict(struct=struct, google_step=step, google_quad=quad, other_quad=oq, rss=PP.rss(res.xf), q=X.shape[1],
               status=res.status, b=b, step_eci=b * res.xf[PP.n_ep + g])
    if step:
        d, Vc, Vm, Vs, b2, edf = PP.deltas(res.xf, lam, mu)
        row.update(step_se=float(np.sqrt(Vm[g, g])), edf=edf)
    if quad:
        k = 2 * MAIN.index(GOOG) + 1 if oq else MAIN.index(GOOG) + 1
        row.update(quad_eci=b * res.xf[PP.P0 + k], lin_eci=b * res.xf[PP.P0 + k - 1])
    return row


def study_shape():
    tasks = [(s, st, qd, oq) for s in STRUCT for oq in (False, True) for (st, qd) in
             [(False, False), (True, False), (False, True), (True, True)]]
    return pd.DataFrame(pool(_shape, tasks, _init_data))


# ---- quasi ---------------------------------------------------------------------------------------------------------
def _quasi(task):
    struct, disp = task
    mu = STRUCT[struct]
    PP, base, res, lam, mu2 = R.fit(DF, mu=mu)
    x, w, hist = res.xf, PP.ones, []
    for it in range(12):
        m = np.clip(PP.predict(x), 0.01, 0.99)
        v = m * (1 - m)
        if disp == "per benchmark":
            pr = (PP.y - PP.predict(x)) ** 2 / v
            nj = np.bincount(PP.bi, minlength=PP.B)
            phi = (np.bincount(PP.bi, pr, PP.B) + 5 * pr.mean()) / (nj + 5)
            v = v * phi[PP.bi]
        w = 1 / v
        w = w / w.mean()
        r2 = PP.fit(w, lam, x, mu=mu2)
        g = PP.ident.index(GOOG)
        _, b = PP.scale(r2.xf)
        hist.append(b * r2.xf[PP.n_ep + g])
        if np.abs(r2.xf - x).max() < 1e-7:
            x = r2.xf
            break
        x = r2.xf
    r2.xf = x
    row, *_ = summary(PP, r2, lam, mu2, w=w)
    row.update(struct=struct, dispersion=disp, iters=it + 1, path=" ".join(f"{h:.3f}" for h in hist))
    return row


def study_quasi():
    return pd.DataFrame(pool(_quasi, [(s, d) for s in STRUCT for d in ("common", "per benchmark")], _init_data))


# ---- select --------------------------------------------------------------------------------------------------------
def _select(task):
    struct, kind = task
    import cm
    if kind == "expected max":
        em = {n: cm.expected_max(n) for n in DF.n_rows.unique()}
        X = DF.n_rows.map(em).to_numpy(float)[:, None]
    else:
        X = np.log(DF.n_rows.to_numpy(float))[:, None]
    PP, base, res, lam, mu = R.fit(DF, mu=STRUCT[struct], X=X)
    row, *_ = summary(PP, res, lam, mu)
    _, b = PP.scale(res.xf)
    row.update(struct=struct, kind=kind, beta_eci=b * res.xf[PP.P0])
    return row


def study_select():
    return pd.DataFrame(pool(_select, [(s, k) for s in STRUCT for k in ("expected max", "log rows")], _init_data))


# ---- misc ----------------------------------------------------------------------------------------------------------
def study_misc():
    """Multiplicity by simulation from the fitted covariance, and the effect in score points."""
    df = R.load()
    rows = []
    rng = np.random.default_rng(5)
    for struct, mu in STRUCT.items():
        PP, base, res, lam, mu2 = R.fit(df, mu=mu)
        d, Vc, Vm, Vs, b, edf = PP.deltas(res.xf, lam, mu2)
        C = np.array([R.contrast_vec(PP.ident, np.diag(Vm), target=g) for g in PP.ident])
        Vcc = C @ Vm @ C.T
        se = np.sqrt(np.diag(Vcc))
        z = (C @ d) / se
        g = PP.ident.index(GOOG)
        sims = rng.multivariate_normal(np.zeros(PP.K), Vcc, 200_000, method="eigh") / se
        p_max = float((np.abs(sims).max(1) >= abs(z[g])).mean())
        x0 = res.xf.copy(); x0[PP.n_ep + g] = 0
        cells = (PP.group == GOOG).to_numpy() & (PP.post > 0)
        dy = (PP.predict(res.xf) - PP.predict(x0))[cells]
        rows.append(dict(struct=struct, google_z=z[g], p_single=2 * stats.norm.sf(abs(z[g])), p_maxT=p_max,
                         bonferroni=min(1, PP.K * 2 * stats.norm.sf(abs(z[g]))), K=PP.K,
                         score_pts_mean=100 * dy.mean(), score_pts_median=100 * np.median(dy), score_pts_max=100 * dy.max(),
                         google_post_cells=int(cells.sum())))
    import zipfile
    e = pd.read_csv(zipfile.ZipFile("data/benchmark_data.zip").open("epoch_capabilities_index/eci_scores.csv"))
    half = (e.eci_ci_high - e.eci_ci_low) / 2
    rows.append(dict(struct="eci_ci_halfwidth", score_pts_mean=float(half.mean()), score_pts_median=float(half.median())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    study = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    t0 = time.time()
    fn = globals()[f"study_{study}"]
    out = fn(reps) if study in ("boot", "wild", "wildb", "perm") else fn()
    out.to_csv(OUT.format(study), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(out.head(40).to_string())
    print(f"{study}: {len(out)} rows, {time.time() - t0:.0f} s")
