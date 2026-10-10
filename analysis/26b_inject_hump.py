"""Injection test, second part: benchmaxxing shaped as a hump rather than a step. analysis/26_inject.py plants a step at
the release date. The other form left open by appendix Section A.10 is tuning to the benchmarks of the moment, which
would lift a lab's scores on benchmarks released shortly before its models and not on older ones. This world takes the
step fit of Chapter 1 and replaces Google's step with a triangle in the gap, zero at 0, peaking at 0.9 years (the peak
of Google's residual profile, appendix Figure A.4) and back to zero at 1.8 years, scaled so that its mean over Google's
post cells equals the fitted step. Simulated data (fitted values plus residuals with one Rademacher sign per model, as
in analysis/26_inject.py) are refitted with the step, quad and labbench designs.

Writes out/app26b_hump.csv, or out/app26b_hump_w<width>.csv for a longer fall after the peak, in years
(default 0.9, so zero again at 1.8 years). Usage: python analysis/26b_inject_hump.py [reps] [width]; run under nice -n 10 (4 workers)."""
import os, sys, time, importlib.util
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
from multiprocessing import Pool
import robust as R
from robust import GOOG, STRUCT

_spec = importlib.util.spec_from_file_location("inject26", os.path.join(HERE, "26_inject.py"))
I = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(I)

OUT = "out/app26b_hump.csv"
PEAK, WIDTH = 0.9, 0.9


def hump_truth(df, struct, x):
    """Fitted values of the hump world and the residuals of the step fit."""
    PP = I.build(df, struct, "step")
    a, lin, _ = PP.eta_parts(x)
    g = PP.ident.index(GOOG)
    goog = PP.gi == g
    d = x[PP.n_ep + g]
    gap = PP.df.gap.to_numpy()
    # rises from 0 at the release to 1 at PEAK years, back to 0 at PEAK + WIDTH; nothing for later benchmarks
    tri = np.where(gap <= PEAK, gap / PEAK, 1 - (gap - PEAK) / WIDTH).clip(0.0)
    post = goog & (PP.post > 0)
    h = d / tri[post].mean() * tri
    lin_h = lin - d * PP.post * goog + h * goog
    yhat = PP.ceiling(x)[PP.bi] * (1 / (1 + np.exp(-a[PP.bi] * lin_h)))
    return (yhat, PP.y - PP.predict(x)), d, d / tri[post].mean()


def _init(starts, truths):
    I._init(starts, truths)


def _task(task):
    return I._task(task)


def main(reps):
    t0 = time.time()
    starts, truths, rows = I.prepare()
    df = R.load()
    for s in STRUCT:
        truths[(s, "hump")], d, peak = hump_truth(df, s, starts[(s, "step")])
        b = rows[[r["struct"] == s and r["design"] == "step" for r in rows].index(True)]["b"]
        print(f"{s}: step {d * b:.3f} ECI points, hump peak {peak * b:.3f} ECI points at {PEAK} years, zero at 0 and {PEAK + WIDTH:g}", flush=True)
    n = {"labbench": reps // 2}
    tasks = [(s, "hump", d, r) for s in STRUCT for d in ("step", "quad", "labbench") for r in range(n.get(d, reps))]
    with Pool(I.WORKERS, initializer=_init, initargs=(starts, truths)) as p:
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
    print(summ.round(3).to_string())
    print(f"hump: {len(out)} rows, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) > 2 and float(sys.argv[2]) != WIDTH:
        WIDTH = float(sys.argv[2])
        OUT = f"out/app26b_hump_w{WIDTH:g}.csv"
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
