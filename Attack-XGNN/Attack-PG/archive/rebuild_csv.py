import os, glob, torch
import numpy as np
import pandas as pd

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"
CSV_PATH = r"C:\habib dissertation\Projects\ResiliX\results_tables\pg_baseline_reproduction.csv"

NAMES = {
    "syn3": "Tree-Cycle",
    "syn1": "BA-House",
    "syn2": "BA-Community",
    "mutag": "MUTAG",
    "REDDIT-BINARY": "Reddit-Binary",
}

PAPER_GCN_ACC = {
    "syn3": 93.00,
    "syn1": 92.71,
    "syn2": 83.17,
    "mutag": 86.28,
    "REDDIT-BINARY": 80.88,
}

rows = []
for ds in ["syn3", "syn1", "mutag", "REDDIT-BINARY", "syn2"]:
    for att in ["loss", "deduction"]:
        folder = os.path.join(SAVE_ROOT, ds, att)
        files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
        if not files:
            continue
        fools = []
        k_val = 6
        for f in files:
            d = torch.load(f, map_location="cpu")
            fools.append(d["undefended_fool_count"])
            k_val = d["k"]
        ratio = (np.mean(fools) / k_val) * 100.0
        rows.append({
            "Explainer": "PGExplainer",
            "Dataset": ds,
            "Dataset_Name": NAMES[ds],
            "Attack": att,
            "Valid_Cases": len(files),
            "Base_GCN_Acc_Pct": PAPER_GCN_ACC[ds],
            "Misalignment_Ratio_Pct": round(ratio, 2),
        })

df = pd.DataFrame(rows)
df.to_csv(CSV_PATH, index=False)
print("\n=== COMPLETE PGEXPLAINER BASELINE TABLE ===")
print(df.to_string(index=False))
