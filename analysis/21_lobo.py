"""Leave one benchmark out for the organization pre/post model without drift (near-unpenalized, lam 1e-3), for any
grouping of prepost.org_groups (analysis/18_bench_resid.py does this for the original organization names only).
When Winogrande (Epoch's slope anchor) is dropped, HellaSwag's slope is pinned instead.
Usage: python analysis/21_lobo.py [org | gmerge | gsplit]. Writes out/lobo21_<mode>.csv (one row per dropped
benchmark, "none" for the full data; delta and model-based SE per group, ECI points)."""
import os, sys
os.environ.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import numpy as np, pandas as pd
from multiprocessing import Pool
import cm
from prepost import PrePost, org_groups

LAM_FE = 1e-3
MODE = sys.argv[1] if len(sys.argv) > 1 else "org"
assert MODE in ("org", "gmerge", "gsplit"), MODE


def load():
    df, ORG = cm.load()
    df["org"] = df.Model.map(ORG).fillna("unknown")
    df["group"] = org_groups(df, MODE)
    return df


def one(bench):
    df = load()
    d = df if bench == "none" else df[df.benchmark != bench]
    pp = PrePost(d, d.group, anchor="HellaSwag" if bench == "Winogrande" else "Winogrande")
    _, f = pp.fit_full(LAM_FE)
    dd, _, Vm, _, _ = pp.analyse(f.xf, LAM_FE)
    se = np.sqrt(np.diag(Vm))
    return dict(dropped=bench, status=f.status, **{f"delta|{g}": v for g, v in zip(pp.ident, dd)},
                **{f"se|{g}": v for g, v in zip(pp.ident, se)})


if __name__ == "__main__":
    benches = ["none"] + sorted(load().benchmark.unique())
    with Pool(4) as pool:
        t = pd.DataFrame(pool.map(one, benches)).set_index("dropped")
    t.round(4).to_csv(f"out/lobo21_{MODE}.csv")
    full, rest = t.loc["none"], t.drop("none")
    print(f"non-positive solver status: {list(rest.index[rest.status <= 0]) or 'none'}")
    for c in [c for c in t.columns if c.startswith("delta|") and "Google" in c]:
        g = c.split("|", 1)[1]
        z = rest[c] / rest[f"se|{g}"]
        print(f"{g}: full {full[c]:+.2f} +- {full[f'se|{g}']:.2f}; without one benchmark {rest[c].min():+.2f} "
              f"({rest[c].idxmin()}) to {rest[c].max():+.2f} ({rest[c].idxmax()}); z from {z.min():.2f} to {z.max():.2f}")
