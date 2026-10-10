"""Inputs for the main post: the lab steps of Chapter 1 (Epoch's structure, Google merged, near-unpenalized) with 95%
intervals, and how far each model's ECI moves when the steps are added. The move is the upper bound of the post's
first section: what a model's ECI would lose if all of its lab's step were benchmaxxing. A model scored only on
benchmarks that predate it loses the whole step; a model with newer benchmarks too loses less, because those cells
pin its capability. ECI is on Epoch's scale in both fits (Claude 3.5 Sonnet 130, GPT-5 150).

Writes out/app28_labs.csv (lab, step, se, 95% interval, models, mean and range of the move, b), out/app28_models.csv
(model, lab, date, published ECI, ECI without and with the steps, the move, cells before and after the model) and
out/app28_bench.csv (benchmark, slope a_j, difficulty, release date: the inputs of the launch-table adjustment).
Usage: python analysis/28_movement.py"""
import os, sys
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io, zipfile
import numpy as np, pandas as pd
import robust as R

MAX_SE = 2.5  # labs shown in the post: as Figure 1.5 of Chapter 1


def main():
    df = R.load()
    PP, base, res, lam, mu = R.fit(df, mu=None, lam=1e-3)
    d, Vc, Vm, b, edf = PP.analyse(res.xf, lam, mu)
    se = np.sqrt(np.maximum(np.diag(Vc), np.diag(Vm)))
    off0, b0 = PP.scale(base.xf)
    off1, b1 = PP.scale(res.xf)
    eci0 = off0 + b0 * base.xf[:PP.M]
    eci1 = off1 + b1 * res.xf[:PP.M]
    e = pd.read_csv(io.BytesIO(zipfile.ZipFile("data/benchmark_data.zip").read("epoch_capabilities_index/eci_scores.csv")))
    info = df.groupby("Model").agg(lab=("group", "first"), date=("date", "first"),
                                   n_before=("gap", lambda g: int((g > 0).sum())), n_after=("gap", lambda g: int((g <= 0).sum())))
    m = pd.DataFrame(dict(Model=PP.models, eci_no_step=eci0, eci_step=eci1))
    m = m.join(info, on="Model").merge(e[["Model", "eci"]].rename(columns=dict(eci="eci_published")), on="Model", how="left")
    m["move"] = m.eci_step - m.eci_no_step
    m = m.sort_values(["lab", "date"])
    m.to_csv("out/app28_models.csv", index=False)

    labs = pd.DataFrame(dict(lab=PP.ident, step=d, se=se))
    labs["lo"], labs["hi"] = labs.step - 1.96 * labs.se, labs.step + 1.96 * labs.se
    g = m.groupby("lab").move.agg(models="size", move_mean="mean", move_min="min", move_max="max")
    labs = labs.join(g, on="lab").sort_values("step", ascending=False)
    labs["shown"] = labs.se < MAX_SE
    labs["b"] = b  # ECI points per unit of capability in the fit with steps
    labs.to_csv("out/app28_labs.csv", index=False)

    C, D, a, _, _ = PP.unpack(res.xf)
    bd = df.groupby("benchmark").bdate.first()
    pd.DataFrame(dict(benchmark=PP.benches, a=a, D=D, bdate=[bd[x] for x in PP.benches])).to_csv("out/app28_bench.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"b = {b:.4f} ECI points per unit of C (base {b0:.4f}, steps {b1:.4f}); lam {lam}")
    print(labs.round(3).to_string(index=False))
    goog = m[m.lab == R.GOOG]
    print(goog[["Model", "date", "eci_published", "eci_no_step", "eci_step", "move", "n_before", "n_after"]].round(2).to_string(index=False))
    print("published vs refit without steps: max abs diff", float((m.eci_published - m.eci_no_step).abs().max()))


if __name__ == "__main__":
    main()
