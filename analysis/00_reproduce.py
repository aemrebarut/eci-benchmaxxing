"""Reproduce Epoch's published ECI fit from the data snapshot (no bootstrap)."""
import os, time, zipfile
import numpy as np, pandas as pd
from eci import prepare_benchmark_data, fit_eci_model

ZIP = "data/benchmark_data.zip"
t0 = time.time()
df = prepare_benchmark_data(source=ZIP)
print("fit input rows", len(df), "models", df.model_id.nunique(), "benchmarks", df.benchmark_id.nunique())
eci_df, edi_df, _ = fit_eci_model(df, bootstrap_samples=0)
print("fit seconds", round(time.time() - t0, 1))

with zipfile.ZipFile(ZIP) as z:
    pub = pd.read_csv(z.open("epoch_capabilities_index/eci_scores.csv"))
    pub_edi = pd.read_csv(z.open("epoch_capabilities_index/edi_scores.csv"))
    proc = pd.read_csv(z.open("epoch_capabilities_index/processed_data_for_eci.csv"))
print("published eci columns", list(pub.columns))
print("published edi columns", list(pub_edi.columns))
print("processed columns", list(proc.columns), "rows", len(proc))
os.makedirs("out", exist_ok=True)
df.to_csv("out/fit_input.csv", index=False)
eci_df.to_csv("out/eci_repro.csv", index=False)
edi_df.to_csv("out/edi_repro.csv", index=False)
print(eci_df.head(15).to_string(index=False))
print(edi_df.to_string(index=False))
