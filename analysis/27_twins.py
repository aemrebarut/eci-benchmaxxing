"""Same-ECI twins: pairs of models from different labs with nearly the same published ECI, released 6 to 9 months
apart, compared benchmark by benchmark. Two models with the same capability have the same predicted score on every
benchmark, so their score difference should not depend on when the benchmark came out. A lab that tunes to public
benchmarks should beat its twin on benchmarks released before its model and not on those released after it.

Benchmarks the two models share are split by their release date: before both models, between the two releases, and
after both. Scores are Epoch's rescaled scores (0 is chance, 1 is the ceiling), reported in points (x 100).
Google pairs (one Google model) are oriented Google minus twin; control pairs (no Google model) later minus earlier.

Writes out/app27_twins_pairs.csv (one row per pair), out/app27_twins_cells.csv (one row per pair and benchmark),
out/app27_twins_summary.csv (the before-minus-after advantage, Google pairs against control pairs) and
out/app27_twins_pairs.md (every Google pair, for the appendix).
Usage: python analysis/27_twins.py [max_eci_diff=2] [min_months=6] [max_months=9]"""
import io, os, sys, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import robust as R
from robust import GOOG

MONTH = 30.44
GROUPS = ["before both", "between", "after both"]


def models(df):
    e = pd.read_csv(io.BytesIO(zipfile.ZipFile("data/benchmark_data.zip").read("epoch_capabilities_index/eci_scores.csv")))
    m = df.groupby("Model").agg(lab=("group", "first"), date=("date", "first")).reset_index()
    m = m.merge(e[["Model", "eci"]], on="Model").dropna(subset=["eci"])
    m["date"] = pd.to_datetime(m.date)
    return m


def pairs(m, max_eci, lo, hi):
    a, b = m.add_suffix("_a"), m.add_suffix("_b")
    p = a.merge(b, how="cross")
    p = p[(p.Model_a < p.Model_b) & (p.lab_a != p.lab_b)]
    p["months"] = (p.date_a - p.date_b).dt.days / MONTH
    p = p[p.months.abs().between(lo, hi) & ((p.eci_a - p.eci_b).abs() <= max_eci)].copy()
    rows = []
    for r in p.itertuples():
        A, B = (r.Model_a, r.Model_b)
        ga, gb = r.lab_a == GOOG, r.lab_b == GOOG
        if ga or gb:
            kind = "google"
            if gb:
                A, B = B, A
        else:
            kind = "control"
            if r.date_b > r.date_a:
                A, B = B, A
        rows.append((kind, A, B))
    out = pd.DataFrame(rows, columns=["kind", "A", "B"])
    info = m.set_index("Model")
    for s in ("A", "B"):
        out[f"lab_{s}"] = out[s].map(info.lab)
        out[f"date_{s}"] = out[s].map(info.date)
        out[f"eci_{s}"] = out[s].map(info.eci)
    out["A_later"] = out.date_A > out.date_B
    out["months"] = (out.date_A - out.date_B).dt.days / MONTH
    return out


def cells(df, pr):
    by = {k: g.set_index("benchmark") for k, g in df.groupby("Model")}
    rows = []
    for i, r in pr.iterrows():
        a, b = by[r.A], by[r.B]
        common = a.index.intersection(b.index)
        for bm in common:
            pa, pb = a.at[bm, "gap"] > 0, b.at[bm, "gap"] > 0
            grp = "before both" if (pa and pb) else ("after both" if not (pa or pb) else "between")
            rows.append(dict(pair=i, benchmark=bm, bdate=a.at[bm, "bdate"], group=grp, y_A=a.at[bm, "y"], y_B=b.at[bm, "y"],
                             diff=100 * (a.at[bm, "y"] - b.at[bm, "y"])))
    return pd.DataFrame(rows)


def advantage(pr, c, reps=2000, seed=7):
    """Per pair, the focal model's mean advantage over its twin on benchmarks released before the focal model minus
    that on benchmarks released after it (pairs with both kinds only). Focal model: the Google model in Google pairs;
    in control pairs either the later or the earlier model, to match the two orientations of the Google pairs.
    Standard errors from a bootstrap over focal models (pairs that share a focal model are resampled together)."""
    c = c.merge(pr[["kind", "A", "B", "A_later"]], left_on="pair", right_index=True)
    sets = [("Google model released later", c[(c.kind == "google") & c.A_later], False, "A"),
            ("Google model released earlier", c[(c.kind == "google") & ~c.A_later], False, "A"),
            ("control, later model", c[c.kind == "control"], False, "A"),
            ("control, earlier model", c[c.kind == "control"], True, "B")]
    rng = np.random.default_rng(seed)
    rows = []
    for name, d, flip, focal in sets:
        d = d.copy()
        if flip:
            d["diff"] = -d["diff"]
        focal_later = d.A_later if focal == "A" else ~d.A_later
        # before the focal model: before both, plus between when the focal model is the later one
        d["side"] = np.where((d.group == "before both") | ((d.group == "between") & focal_later), "before", "after")
        m = d.pivot_table(index="pair", columns="side", values="diff", aggfunc="mean").dropna()
        v = (m["before"] - m["after"]).to_numpy()
        cl = pr.loc[m.index, focal].to_numpy()
        u = np.unique(cl)
        where = {x: np.flatnonzero(cl == x) for x in u}
        bs = [v[np.concatenate([where[x] for x in rng.choice(u, len(u))])].mean() for _ in range(reps)]
        rows.append(dict(set=name, pairs=len(v), focal_models=len(u), before=m["before"].mean(), after=m["after"].mean(),
                         advantage=v.mean(), se=float(np.std(bs))))
    out = pd.DataFrame(rows)
    for g, k in [("Google model released later", "control, later model"), ("Google model released earlier", "control, earlier model")]:
        a, b = out.set_index("set").loc[g], out.set_index("set").loc[k]
        out.loc[len(out)] = dict(set=f"{g} minus control", pairs=a.pairs, focal_models=a.focal_models, before=np.nan,
                                 after=np.nan, advantage=a.advantage - b.advantage, se=float(np.hypot(a.se, b.se)))
    return out


def pairs_md(pr, c):
    """Every Google pair as a markdown table for the appendix (out/app27_twins_pairs.md): mean difference (Google minus
    twin, score points) and number of benchmarks in each group, and the pair's advantage, the mean difference on
    benchmarks released before the Google model minus that on benchmarks released after it."""
    g = pr[pr.kind == "google"].copy()
    c = c.merge(pr[["A_later"]], left_on="pair", right_index=True)
    c["side"] = np.where((c.group == "before both") | ((c.group == "between") & c.A_later), "before", "after")
    s = c[c.pair.isin(g.index)].pivot_table(index="pair", columns="side", values="diff", aggfunc="mean")
    g["adv"] = (s["before"] - s["after"]).reindex(g.index)
    g = g.sort_values(["date_A", "A", "date_B"])
    lab = {"Z.ai (Zhipu AI)": "Z.ai", "Meta AI": "Meta", "Mistral AI": "Mistral"}

    def cell(r, grp):
        n = 0 if pd.isna(r[f"n_{grp}"]) else int(r[f"n_{grp}"])
        return "" if n == 0 else f"${r[f'mean_{grp}']:+.1f}$ ({n})"

    out = ["| Google model | twin | twin's lab | months | ECI | twin's ECI | before both | between | after both | advantage |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        adv = "" if pd.isna(r.adv) else f"${r.adv:+.1f}$"
        out.append(f"| {r.A} | {r.B} | {lab.get(r.lab_B, r.lab_B)} | ${r.months:+.1f}$ | {r.eci_A:.1f} | {r.eci_B:.1f} | "
                   f"{cell(r, 'before both')} | {cell(r, 'between')} | {cell(r, 'after both')} | {adv} |")
    open("out/app27_twins_pairs.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
    return g


def main(max_eci=2.0, lo=6.0, hi=9.0):
    df = R.load()
    m = models(df)
    pr = pairs(m, max_eci, lo, hi).reset_index(drop=True)
    c = cells(df, pr)
    agg = c.pivot_table(index="pair", columns="group", values="diff", aggfunc=["mean", "size"])
    for g in GROUPS:
        pr[f"mean_{g}"] = agg[("mean", g)] if ("mean", g) in agg else np.nan
        pr[f"n_{g}"] = agg[("size", g)].fillna(0).astype(int) if ("size", g) in agg else 0
    pr.to_csv("out/app27_twins_pairs.csv", index=False)
    c.to_csv("out/app27_twins_cells.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(pr)} pairs: {(pr.kind == 'google').sum()} with a Google model ({(pr.A_later & (pr.kind == 'google')).sum()} Google later), "
          f"{(pr.kind == 'control').sum()} control")
    # Mean over pairs of each pair's mean difference; cells pooled within pairs
    for (kind, later), g in pr.groupby(["kind", "A_later"]):
        line = [f"{kind:8s} A later={later!s:5s} pairs={len(g):3d}"]
        for grp in GROUPS:
            v = g[f"mean_{grp}"].dropna()
            line.append(f"{grp}: {v.mean():+.2f} (pairs {len(v)}, cells {g[f'n_{grp}'].sum()})")
        print("  ".join(line))
    adv = advantage(pr, c)
    adv.to_csv("out/app27_twins_summary.csv", index=False)
    print(adv.round(2).to_string())
    g = pairs_md(pr, c)
    print(f"out/app27_twins_pairs.md: {len(g)} Google pairs, {g.adv.notna().sum()} with benchmarks on both sides, "
          f"advantage > 0 in {(g.adv > 0).sum()}")
    return pr, c


if __name__ == "__main__":
    a = [float(x) for x in sys.argv[1:4]]
    main(*a)
