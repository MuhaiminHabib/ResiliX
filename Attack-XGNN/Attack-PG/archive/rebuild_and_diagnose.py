import os, glob, torch
import numpy as np
import pandas as pd
import networkx as nx
import torch_geometric as ptgeom

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"
CSV_PATH = r"C:\habib dissertation\Projects\ResiliX\results_tables\pg_baseline_reproduction.csv"

NAMES = {"syn3": "Tree-Cycle", "syn1": "BA-House", "syn2": "BA-Community", "mutag": "MUTAG", "REDDIT-BINARY": "Reddit-Binary"}
PAPER_GCN_ACC = {"syn3": 93.00, "syn1": 92.71, "syn2": 83.17, "mutag": 86.28, "REDDIT-BINARY": 80.88}

rows = []
print("=== 1. REBUILDING BASELINE CSV & INSPECTING ATTACK SIGNATURES ===")
for ds in ["syn3", "syn1", "mutag", "REDDIT-BINARY"]:
    for att in ["loss", "deduction"]:
        folder = os.path.join(SAVE_ROOT, ds, att)
        files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
        if not files:
            continue
        fools = []
        n_added_list, n_deleted_list = [], []
        added_touch_indx, total_added = 0, 0
        added_spans, motif_spans = [], []

        for f in files:
            d = torch.load(f, map_location="cpu")
            fools.append(d["undefended_fool_count"])
            k_val = d["k"]
            task = d.get("task", "node" if ds in ["syn1", "syn2", "syn3"] else "graph")
            indx = int(d["indx"])

            cg = d["clean_graphs"][0].numpy() if task == "graph" else d["clean_graphs"].numpy()
            ag = d["attacked_graphs"][0].numpy() if task == "graph" else d["attacked_graphs"].numpy()

            c_set = set((min(int(cg[0, i]), int(cg[1, i])), max(int(cg[0, i]), int(cg[1, i]))) for i in range(cg.shape[1]) if cg[0, i] != cg[1, i])
            a_set = set((min(int(ag[0, i]), int(ag[1, i])), max(int(ag[0, i]), int(ag[1, i]))) for i in range(ag.shape[1]) if ag[0, i] != ag[1, i])
            es_set = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in d["clean_Es_pairs"] if u != v)

            e_add = a_set - c_set
            e_del = c_set - a_set
            n_added_list.append(len(e_add))
            n_deleted_list.append(len(e_del))

            if task == "node":
                subg = ptgeom.utils.k_hop_subgraph(indx, 3, d["attacked_graphs"])[1].numpy()
            else:
                subg = ag

            G = nx.Graph()
            for i in range(subg.shape[1]):
                u, v = int(subg[0, i]), int(subg[1, i])
                if u != v:
                    G.add_edge(u, v)

            for u, v in e_add:
                total_added += 1
                if task == "node" and (u == indx or v == indx):
                    added_touch_indx += 1
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    try:
                        sp = nx.shortest_path_length(G, u, v)
                    except nx.NetworkXNoPath:
                        sp = 10
                    G.add_edge(u, v)
                    added_spans.append(sp)

            for u, v in es_set:
                if G.has_edge(u, v):
                    G.remove_edge(u, v)
                    try:
                        sp = nx.shortest_path_length(G, u, v)
                    except nx.NetworkXNoPath:
                        sp = 10
                    G.add_edge(u, v)
                    motif_spans.append(sp)

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
        avg_add_sp = np.mean(added_spans) if added_spans else 0
        avg_mot_sp = np.mean(motif_spans) if motif_spans else 0
        print(f"[{ds:13s} | {att:9s}] Misalign: {ratio:5.2f}% | Avg +Edges: {np.mean(n_added_list):.2f}, -Edges: {np.mean(n_deleted_list):.2f} | TouchTarget: {added_touch_indx}/{total_added} | Span(Add vs Motif): {avg_add_sp:.1f} vs {avg_mot_sp:.1f}")

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
df.to_csv(CSV_PATH, index=False)
print("\n[SAVED CLEAN CSV]:", CSV_PATH)
