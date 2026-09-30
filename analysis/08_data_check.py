"""Data checks before the common-mechanism model.

1. Selection: each cell is the max over several raw rows (model versions and sources within a model group).
   If the number of rows grows with the gap, the max alone can create a gap effect.
2. Clipping: are unclipped rescaled scores available, and how many cells are clipped?

Replicates eci.dataloader.prepare_benchmark_data up to its aggregation step and checks that it rebuilds
the same cells."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from eci import dataloader as L
import bmx

SRC = "data/benchmark_data.zip"
df = bmx.add_gap(L.prepare_benchmark_data(source=SRC))

dfs = L.download_benchmark_data(SRC)
meta = L.load_benchmark_metadata(dfs)
meta = meta[meta["in_eci"].fillna(True).astype(bool)]
models = L.load_model_metadata(dfs)[["model_version", "model_group", "date"]]
s = L._load_scores(dfs, meta)
s["performance"] = s["performance"].replace([np.inf, -np.inf], np.nan)
s = s[s["performance"].notna()].rename(columns={"performance": "raw"})
s["clipped"] = s["raw"].clip(0.0, 1.0)
s = s.merge(models, on="model_version", how="inner")
s = s[s["date"] >= pd.Timestamp("2023-01-01")]
sup = meta.dropna(subset=["superseded_by"])
sup = sup[sup["superseded_by"].astype(str) != ""]
for old, new in zip(sup["benchmark"], sup["superseded_by"]):
    g = set(s.loc[s["benchmark"] == new, "model_group"])
    s = s[~((s["benchmark"] == old) & s["model_group"].isin(g))]
keep = s.groupby("model_group")["benchmark"].nunique()
s = s[s["model_group"].isin(keep[keep >= 4].index)]

cells = s.groupby(["model_group", "benchmark"]).agg(
    n_rows=("raw", "size"), n_versions=("model_version", "nunique"),
    raw_max=("raw", "max"), clipped_max=("clipped", "max")).reset_index()
m = df.merge(cells, left_on=["Model", "benchmark"], right_on=["model_group", "benchmark"], how="left")
assert m["n_rows"].notna().all(), "some cells were not rebuilt"
assert np.allclose(m["clipped_max"], m["performance"]), "rebuilt max differs from the loader"

# The row that achieved the max: its version date, versus the cell date (max over versions).
s["vdate"] = pd.to_datetime(s["date"])
top = s.sort_values("clipped", ascending=False).drop_duplicates(["model_group", "benchmark"])
m = m.merge(top[["model_group", "benchmark", "vdate"]], on=["model_group", "benchmark"], how="left")
m["max_from_older_version_days"] = (pd.to_datetime(m["date"]) - m["vdate"]).dt.days

pos = m[m.gap > 0]
lines = ["# Data checks (analysis/08_data_check.py)", "",
         f"Rebuilt all {len(m)} cells from the pre-aggregation rows; the max of clipped rows matches the loader exactly.", "",
         "## 1. Rows per cell and the gap", "",
         f"- Cells with more than one raw row: {int((m.n_rows > 1).sum())} of {len(m)} ({(m.n_rows > 1).mean():.1%}).",
         f"- Rows per cell: median {m.n_rows.median():.0f}, mean {m.n_rows.mean():.2f}, max {m.n_rows.max()}.",
         f"- Spearman correlation of rows per cell with the gap: {spearmanr(m.n_rows, m.gap)[0]:+.3f} (all cells), "
         f"{spearmanr(pos.n_rows, pos.gap)[0]:+.3f} (gap > 0).", ""]
bins = pd.cut(m.gap, [-10, -1, 0, 1, 2, 4, 20])
t = m.groupby(bins, observed=True).agg(cells=("n_rows", "size"), mean_rows=("n_rows", "mean"),
                                        share_multi=("n_rows", lambda x: (x > 1).mean())).reset_index()
lines += ["| gap bin (years) | cells | mean rows | share with >1 row |", "|---|---|---|---|"]
lines += [f"| {r.gap} | {r.cells} | {r.mean_rows:.2f} | {r.share_multi:.1%} |" for r in t.itertuples()]
# Within benchmark: does a later model (larger gap) have more rows on the same benchmark?
w = m.assign(rows_c=m.n_rows - m.groupby("benchmark").n_rows.transform("mean"),
             gap_c=m.gap - m.groupby("benchmark").gap.transform("mean"))
lines += ["", f"- Within-benchmark Spearman (rows vs gap, both centred by benchmark): {spearmanr(w.rows_c, w.gap_c)[0]:+.3f}.",
          f"- Cells whose max came from an older version than the cell date: {int((m.max_from_older_version_days > 0).sum())} "
          f"({(m.max_from_older_version_days > 0).mean():.1%}); median lag among them "
          f"{m.loc[m.max_from_older_version_days > 0, 'max_from_older_version_days'].median():.0f} days.", "",
          "## 2. Clipping and unclipped scores", "",
          "- Unclipped rescaled scores are available: the raw rows carry them before the loader's clip to [0, 1].",
          f"- Cells whose max raw score is below chance (clipped up to 0): {int((m.raw_max < 0).sum())} ({(m.raw_max < 0).mean():.1%}).",
          f"- Cells above the ceiling (clipped down to 1): {int((m.raw_max > 1).sum())}.",
          f"- Cells at or below 0.001 or at or above 0.999 after the fit's clip: "
          f"{int(((m.performance <= 1e-3) | (m.performance >= 1 - 1e-3)).sum())}.",
          f"- Spearman of the sub-chance indicator with the gap: {spearmanr(m.raw_max < 0, m.gap)[0]:+.3f}."]
m[["Model", "benchmark", "gap", "n_rows", "n_versions", "raw_max", "performance", "max_from_older_version_days"]] \
    .to_csv("out/data_check08.csv", index=False)
open("out/data_check08.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
