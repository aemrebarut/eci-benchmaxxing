"""The launch tables of Gemini 3.7 Flash and Gemini 3.8 Flash with Google's step taken out, for the main post. The
upper bound: if all of Google's step were benchmaxxing, a reported score y on a benchmark released before the model
would have been sigmoid(logit(y) - a_j delta / b) without it, with a_j the benchmark's slope and delta / b the step on
the capability scale, both from the fit of analysis/28_movement.py (Epoch's structure, Google merged). Benchmarks
released after the model are not adjusted. Only rows whose benchmark, version and metric match Epoch's are used; for
these benchmarks Epoch's rescaled score is the raw proportion (chance 0). Epoch's own record of the same model and
benchmark is shown beside Google's number.

Reads data/launch/gemini-3.{7,8}-flash.csv, out/app28_bench.csv, out/app28_labs.csv and
out/app28_models.csv; writes out/app29_launch.csv. Usage: python analysis/29_launch.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import robust as R

LAUNCH = "data/launch"
# model -> (launch-table row, variant or None, Epoch benchmark, which Google table); exact matches only
MATCH = {
    "Gemini 3.7 Flash": [("FrontierCode 1.1 Main", None, "FrontierCode", "launch table"),
                         ("DeepSWE v1.1", None, "DeepSWE", "launch table"),
                         ("Humanity's Last Exam (HLE)", None, "HLE", "Cloud guide")],
    "Gemini 3.8 Flash": [("DeepSWE v1.1", None, "DeepSWE", "launch table"),
                         ("Humanity's Last Exam (HLE)", None, "HLE", "Cloud guide")],
}
FILES = {"Gemini 3.7 Flash": "gemini-3.7-flash.csv", "Gemini 3.8 Flash": "gemini-3.8-flash.csv"}


def logit(p):
    return np.log(p / (1 - p))


def main():
    df = R.load()
    bench = pd.read_csv("out/app28_bench.csv").set_index("benchmark")
    labs = pd.read_csv("out/app28_labs.csv").set_index("lab")
    models = pd.read_csv("out/app28_models.csv").set_index("Model")
    step, PPb = labs.loc[R.GOOG, "step"], labs.loc[R.GOOG, "b"]
    rows = []
    for model, items in MATCH.items():
        t = pd.read_csv(os.path.join(LAUNCH, FILES[model]))
        mdate = pd.Timestamp(models.loc[model, "date"])
        myear = mdate.year + (mdate.dayofyear - 1) / (366 if mdate.is_leap_year else 365)
        for row, variant, bm, src in items:
            r = t[t.benchmark == row]
            if src == "Cloud guide":
                r = r[r.source_url.str.contains("cloud.google", na=False)]
            else:
                r = r[~r.source_url.str.contains("cloud.google", na=False)]
            assert len(r) == 1, (model, row, len(r))
            y = float(r.score.iloc[0]) / 100
            a = bench.loc[bm, "a"]
            before = bench.loc[bm, "bdate"] < myear
            y_adj = 1 / (1 + np.exp(-(logit(y) - a * step / PPb))) if before else y
            epoch = df[(df.Model == model) & (df.benchmark == bm)].y
            rows.append(dict(model=model, launch_row=row, source=src, benchmark=bm, bdate=bench.loc[bm, "bdate"],
                             released_before=before, slope=a, reported=100 * y, without_step=100 * y_adj,
                             change=100 * (y_adj - y), epoch=100 * float(epoch.iloc[0]) if len(epoch) else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv("out/app29_launch.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"Google step {step:.3f} ECI points, b {PPb:.4f}, step on the capability scale {step / PPb:.4f}")
    print(out.round(2).to_string(index=False))
    for m in MATCH:
        r = models.loc[m]
        print(f"{m}: published ECI {r.eci_published:.2f}, refit without steps {r.eci_no_step:.2f}, with steps {r.eci_step:.2f}, "
              f"move {r.move:+.2f}; cells before {r.n_before}, after {r.n_after}")


if __name__ == "__main__":
    main()
