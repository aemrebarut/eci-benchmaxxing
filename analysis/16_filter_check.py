"""First look at adversarial benchmark construction (tags in data/benchmark_filtering.csv): do the models a
benchmark was filtered against score below their ECI on it, and are residuals just before or after release
different on filtered benchmarks? Residuals from Epoch's fit, as in analysis/14_resid_plots.py. Descriptive
only; the model-based test comes later. Writes out/filter16.md.

r_eci = (logit y - logit fitted) / slope: the ECI shift that would make the fit match the cell (unreliable at the
clip floor, where y = 0.001). pct: share of the benchmark's cells with a lower residual. Filter models are those
named in the tags that are in Epoch's data; where the tag does not give the version (GPT-4o in Aider polyglot),
every version in the data is listed. SEs in the gap-bin table are clustered by model."""
import numpy as np, pandas as pd
from scipy.special import logit
from eci import prepare_benchmark_data, fit_eci_model
from eci.fitting import sigmoid
import cm

FILTER_MODELS = {
    "HLE": ["GPT-4o (Nov 2024)", "Gemini 1.5 Pro (Sept 2024)", "Claude 3.5 Sonnet (October 2024)", "o1"],
    "Aider polyglot": ["Claude 3.5 Sonnet (October 2024)", "Claude 3.5 Haiku", "o1-mini", "GPT-4o (Aug 2024)",
                       "GPT-4o (Nov 2024)", "GPT-4o mini"],
    "SimpleQA Verified": ["GPT-4o (Aug 2024)"],
    "FrontierMath-Tier-4-v2-Private": ["o4-mini"],
}
BINS = [-4, -1, -0.5, 0, 0.5, 1, 2, 4, 9]

df, ORG = cm.load()
eci, edi, _ = fit_eci_model(prepare_benchmark_data(source="data/benchmark_data.zip"), bootstrap_samples=0)
x = df.merge(eci[["model_id", "eci"]], on="model_id", validate="many_to_one") \
      .merge(edi[["benchmark_id", "edi", "discriminability_scaled"]], on="benchmark_id", validate="many_to_one")
x["mu"] = sigmoid(x.discriminability_scaled * (x.eci - x.edi))
x["r"] = x.y_fit - x.mu
x["r_eci"] = (logit(x.y_fit) - logit(x.mu)) / x.discriminability_scaled
tags = pd.read_csv("data/benchmark_filtering.csv").set_index("benchmark")
assert set(x.benchmark) == set(tags.index)
x["filtered"] = x.benchmark.map(tags.filtered).isin(["yes", "partial"])

rows = []
for b, ms in FILTER_MODELS.items():
    g = x[x.benchmark == b]
    for m in ms:
        c = g[g.Model == m]
        assert len(c) == 1, (b, m)
        c = c.iloc[0]
        rows.append(dict(benchmark=b, model=m, gap=c.gap, score=c.y_fit, fitted=c.mu, r=c.r, r_eci=c.r_eci,
                         pct=(g.r < c.r).mean(), sd_r_benchmark=g.r.std()))
t = pd.DataFrame(rows)


def clustered(g):
    e = g.r - g.r.mean()
    return pd.Series(dict(cells=len(g), models=g.Model.nunique(), mean_r=g.r.mean(),
                          se=np.sqrt((e.groupby(g.Model).sum() ** 2).sum()) / len(g)))


x["bin"] = pd.cut(x.gap, BINS, right=False)
tb = x.groupby(["filtered", "bin"], observed=True).apply(clustered, include_groups=False).reset_index()

L = ["# Filtered benchmarks, first look (analysis/16_filter_check.py)", "",
     f"Tags: {tags.filtered.value_counts().to_dict()} (yes and partial count as filtered).", "",
     "## Filter models on the benchmarks built against them", "",
     "| benchmark | model | gap | score | fitted | residual | ECI shift | percentile in benchmark | benchmark sd of residual |",
     "|---|---|---|---|---|---|---|---|---|"]
L += [f"| {r.benchmark} | {r.model} | {r.gap:+.2f} | {r.score:.3f} | {r.fitted:.3f} | {r.r:+.3f} | {r.r_eci:+.1f} | {r.pct:.2f} | {r.sd_r_benchmark:.3f} |"
      for r in t.itertuples()]
L += ["", f"Mean residual {t.r.mean():+.3f}, mean percentile {t.pct.mean():.2f} ({len(t)} cells). "
      f"Excluding HLE (all four at or near the floor): mean residual {t[t.benchmark != 'HLE'].r.mean():+.3f}.", "",
      "## Mean residual by gap bin, filtered vs other benchmarks (SE clustered by model)", "",
      "| gap bin | filtered: cells | mean | SE | other: cells | mean | SE |", "|---|---|---|---|---|---|---|"]
for b in tb.bin.cat.categories:
    f, o = tb[(tb.bin == b) & tb.filtered], tb[(tb.bin == b) & ~tb.filtered]
    cell = lambda s: f"{int(s.cells.iloc[0])} | {s.mean_r.iloc[0]:+.4f} | {s.se.iloc[0]:.4f}" if len(s) else " | | "
    L.append(f"| {b} | {cell(f)} | {cell(o)} |")
open("out/filter16.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
