"""Is Google's step a weakness on agentic benchmarks? The step model (analysis/17_prepost.py and 19_drift.py, Google
merged) with lab-by-agentic terms, using the classification in data/benchmark_types.csv (agentic: yes, no or
borderline; built from Epoch's benchmark pages and the benchmarks' own papers on 2026-10-02).

An agentic indicator is a property of the benchmark, so a term common to all labs is absorbed by D_j; a term for one
lab measures that lab's models on agentic benchmarks relative to the labs without a term. Fits, each without and with
the benchmark drift (mu 0.1), near-unpenalized (lam 1e-3):
  step         the step model alone (reproduces Google's +1.55 and the contrast)
  g_agentic    plus one term for Google's models on agentic benchmarks
  lab_agentic  plus one term for each of the seven largest labs
  g_agentic_x  plus Google's agentic term and its interaction with post: Google's step on agentic benchmarks minus
               its step on the others (the step column itself is then the step on non-agentic benchmarks)
  g_curve      plus the seven labs' linear trends in benchmark date and a Google quadratic (as in the shape study of
               analysis/23_appendix.py), without and with Google's agentic term (g_curve_agentic): does the agentic
               term take over the curve over benchmark age?
  nonagentic   the step model refitted on the non-agentic benchmarks only
and two codings: "yes" (borderline counted as not agentic) and "yes+borderline".
With the argument "borderline": the g_agentic fit under "yes" plus one borderline benchmark at a time, under
"yes+borderline" minus one borderline benchmark at a time, and under "lean" (yes plus the two borderline benchmarks
whose notes lean yes, ProofBench and Surface Evolver Bench). Writes out/p1_agentic_borderline.csv.
Usage: python analysis/31_agentic.py [borderline]. Writes out/p1_agentic.csv. Run under nice -n 10; 4 workers."""
import importlib, os, sys, time
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from multiprocessing import Pool
import robust as R
from robust import GOOG, MAIN, STRUCT

A = importlib.import_module("23_appendix")
WORKERS = 4
CODINGS = {"yes": ("yes",), "yes+borderline": ("yes", "borderline")}
FITS = ["step", "g_agentic", "lab_agentic", "g_agentic_x", "g_curve", "g_curve_agentic", "nonagentic"]


def init():
    global DF, TYPES
    DF = R.load()
    TYPES = pd.read_csv("data/benchmark_types.csv").set_index("benchmark").agentic
    missing = set(DF.benchmark) - set(TYPES.index)
    assert not missing, missing


def columns(df, fit, ag):
    """The extra columns of a fit and their names."""
    g = (df.group == GOOG).to_numpy(float)
    post = (df.gap > 0).to_numpy(float)
    s = df.bdate.to_numpy() - 2024.0
    cols, names = [], []
    if fit in ("g_agentic", "g_agentic_x", "g_curve_agentic"):
        cols.append(g * ag); names.append("google agentic")
    if fit == "g_agentic_x":
        cols.append(g * ag * post); names.append("google agentic x post")
    if fit == "lab_agentic":
        for lab in MAIN:
            cols.append((df.group == lab).to_numpy(float) * ag); names.append(f"{lab} agentic")
    if fit in ("g_curve", "g_curve_agentic"):
        for lab in MAIN:
            cols.append((df.group == lab).to_numpy(float) * s); names.append(f"{lab} linear")
        cols.append(g * s ** 2); names.append("google quadratic")
    return (np.column_stack(cols) if cols else None), names


def agentic_set(coding):
    """The benchmarks counted as agentic: a coding name, or a tuple (base coding, benchmarks added, removed)."""
    if isinstance(coding, tuple):
        base, add, drop = coding
        return (set(TYPES.index[TYPES.isin(CODINGS[base])]) | set(add)) - set(drop)
    return set(TYPES.index[TYPES.isin(CODINGS[coding])])


def run(task):
    struct, coding, fit = task
    df = DF
    ag_all = df.benchmark.isin(agentic_set(coding)).to_numpy(float)
    if fit == "nonagentic":
        df = df[ag_all == 0].reset_index(drop=True)
        df["z"] = df.gap - df.groupby("benchmark").gap.transform("mean")
        ag_all = np.zeros(len(df))
    X, names = columns(df, fit, ag_all)
    t0 = time.time()
    PP, base, res, lam, mu = R.fit(df, mu=STRUCT[struct], X=X)
    row, *_ = A.summary(PP, res, lam, mu)
    row.update(struct=struct, coding=str(coding), fit=fit, cells=len(df), benchmarks=df.benchmark.nunique(),
               agentic_cells=int(ag_all.sum()), google_agentic_cells=int(ag_all[(df.group == GOOG).to_numpy()].sum()),
               secs=round(time.time() - t0, 1))
    if X is not None:
        idx, Vc, Vm, Vs, b, edf = PP.cov2(res.xf, lam, mu)
        pos = {p: k for k, p in enumerate(idx)}
        for k, n in enumerate(names):
            p = pos.get(PP.P0 + k)
            est = b * res.xf[PP.P0 + k]
            se = b * np.sqrt(max(Vm[p, p], Vc[p, p])) if p is not None else np.nan
            row[f"x: {n}"] = f"{est:+.2f} ({se:.2f})"
            if n == "google agentic":
                row.update(g_agentic=est, g_agentic_se=se)
    return row


def main():
    tasks = [(s, c, f) for c in CODINGS for s in STRUCT for f in FITS]
    t0 = time.time()
    with Pool(WORKERS, initializer=init) as p:
        rows = p.map(run, tasks, chunksize=1)
    out = pd.DataFrame(rows)
    out.to_csv("out/p1_agentic.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 40)
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(3)
    cols = ["coding", "struct", "fit", "status", "rss", "g_est", "g_se_mb", "g_z_mb", "c_est", "c_se_mb", "c_z_mb",
            "g_agentic", "g_agentic_se", "cells", "google_agentic_cells", "secs"]
    print(out[cols].to_string(index=False))
    for _, r in out.iterrows():
        xs = {k: v for k, v in r.items() if k.startswith("x: ") and isinstance(v, str)}
        if xs:
            print(r.coding, r.struct, r.fit, "|", "; ".join(f"{k[3:]} {v}" for k, v in xs.items()))
    print(f"done in {time.time() - t0:.0f} s")


def borderline():
    init()
    bl = list(TYPES.index[TYPES == "borderline"])
    codings = [("yes", (), ()), ("yes+borderline", (), ()), ("yes", ("ProofBench", "Surface Evolver Bench"), ())]
    codings += [("yes", (b,), ()) for b in bl] + [("yes+borderline", (), (b,)) for b in bl]
    tasks = [(s, c, "g_agentic") for c in codings for s in STRUCT]
    with Pool(WORKERS, initializer=init) as p:
        out = pd.DataFrame(p.map(run, tasks, chunksize=1))
    out.to_csv("out/p1_agentic_borderline.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
    print(out[["coding", "struct", "g_agentic", "g_agentic_se", "g_est", "c_est", "c_z_mb"]].round(2).to_string(index=False))


if __name__ == "__main__":
    borderline() if sys.argv[1:] == ["borderline"] else main()
