"""Injection test for the curved-trend objection (appendix Section A.10): would giving each lab its own curve in
benchmark date hide a real step at the release date?

Two simulated worlds, each built from a fit to the real data:
  step   the step model of Chapter 1 as fitted (Google's step +1.55 ECI points without drift), no lab curves
  curve  lab quadratics in benchmark date for the seven largest labs, with Google's step pinned at 0
Simulated scores are the world's fitted values plus its residuals with one Rademacher sign per model (as in the wild
bootstrap of analysis/23_appendix.py, which keeps each model's residual pattern but cancels any systematic one). Each
simulated data set is refitted with three designs:
  step      the step model of Chapter 1
  quad      the step model plus lab quadratics in benchmark date (appendix Table A.10, "quadratic, by lab")
  labbench  the step model plus lab-by-benchmark effects for the seven largest labs (the triple difference)
In the step world, quad and labbench show how much of a true step survives the curve; in the curve world, step shows
what a step model makes of a curve. Writes out/app26_inject.csv (rep -1: the real data).

Usage: python analysis/26_inject.py [reps]; run under nice -n 10 (4 workers)."""
import os, sys, time, importlib.util
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
import robust as R
from robust import GOOG, STRUCT

_spec = importlib.util.spec_from_file_location("app23", os.path.join(HERE, "23_appendix.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

WORKERS = 4
LAM = A.LAM
OUT = "out/app26_inject.csv"
DESIGNS = {"step": (None, 0.0), "quad": ("lab quadratic", 0.0), "labbench": ("labbench", 1e-3)}
PLAN = [("step", "quad"), ("step", "labbench"), ("step", "step"), ("curve", "step"), ("curve", "quad")]


def design_X(df, design):
    kind, lam_x = DESIGNS[design]
    if kind is None:
        return None, lam_x
    X = A.labbench_X(df)[0] if kind == "labbench" else A.trend_X(df, kind)[0]
    return X, lam_x


def build(df, struct, design):
    X, lam_x = design_X(df, design)
    mu = STRUCT[struct]
    return R.PPX(df, df.group, X=X, trend=df.z if mu is not None else None, lam_x=lam_x)


def prepare():
    """Real-data fits: the start values of each design, and the truth (fitted values, residuals) of each world."""
    df = R.load()
    starts, truths, rows = {}, {}, []
    for struct, mu in STRUCT.items():
        for design in DESIGNS:
            X, lam_x = design_X(df, design)
            PP, base, res, lam, mu2 = R.fit(df, mu=mu, X=X, lam_x=lam_x)
            starts[(struct, design)] = res.xf
            row, *_ = A.summary(PP, res, lam, mu2)
            rows.append(dict(struct=struct, world="real", design=design, rep=-1, **pick(row)))
            if design == "step":
                truths[(struct, "step")] = (PP.predict(res.xf), PP.y - PP.predict(res.xf))
            if design == "quad":
                e = np.zeros(PP.K); e[PP.ident.index(GOOG)] = 1
                PP.set_con(e, 0.0)
                r0 = R.refit(PP, res.xf, lam, mu2)
                PP.set_con(None)
                yhat = PP.predict(r0.xf)
                truths[(struct, "curve")] = (yhat, PP.y - yhat)
        print(f"prepared {struct}", flush=True)
    return starts, truths, rows


def pick(row):
    return {k: row[k] for k in ("g_est", "g_se_mb", "g_z_mb", "c_est", "c_se_mb", "c_z_mb", "b", "status")}


def _init(starts, truths):
    global DF, PPS, STARTS, TRUTHS
    DF = R.load()
    PPS = {(s, d): build(DF, s, d) for s in STRUCT for d in DESIGNS}
    STARTS, TRUTHS = starts, truths


def _task(task):
    struct, world, design, rep = task
    PP = PPS[(struct, design)]
    yhat, r = TRUTHS[(struct, world)]
    v = np.random.default_rng(50_000 + rep).choice([-1.0, 1.0], PP.M)[PP.mi]
    y0 = PP.y
    PP.y = yhat + v * r
    try:
        res = R.refit(PP, STARTS[(struct, design)], LAM, STRUCT[struct])
        row, *_ = A.summary(PP, res, LAM, STRUCT[struct])
        return dict(struct=struct, world=world, design=design, rep=rep, **pick(row))
    except Exception as ex:
        return dict(struct=struct, world=world, design=design, rep=rep, err=str(ex)[:200])
    finally:
        PP.y = y0


def main(reps):
    t0 = time.time()
    starts, truths, rows = prepare()
    print(f"real fits {time.time() - t0:.0f} s", flush=True)
    n = {("step", "labbench"): reps // 2, ("curve", "quad"): reps // 2}
    tasks = [(s, w, d, r) for s in STRUCT for (w, d) in PLAN for r in range(n.get((w, d), reps))]
    with Pool(WORKERS, initializer=_init, initargs=(starts, truths)) as p:
        rows += p.map(_task, tasks, chunksize=2)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    sim = out[out.rep >= 0]
    real = out[out.rep < 0].set_index(["struct", "design"]).c_est
    summ = sim.groupby(["struct", "world", "design"]).agg(
        n=("c_est", "size"), c_mean=("c_est", "mean"), c_sd=("c_est", "std"), se_mean=("c_se_mb", "mean"),
        power=("c_z_mb", lambda z: (z >= 1.96).mean()))
    summ["p_le_real"] = [(sim[(sim.struct == s) & (sim.world == w) & (sim.design == d)].c_est <= real[(s, d)]).mean()
                         for s, w, d in summ.index]
    pd.set_option("display.width", 200)
    print(out[out.rep < 0].round(3).to_string())
    print(summ.round(3).to_string())
    print(f"inject: {len(out)} rows, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
