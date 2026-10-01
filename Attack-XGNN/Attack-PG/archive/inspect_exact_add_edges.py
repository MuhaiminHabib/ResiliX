import os, glob, torch
import numpy as np
import networkx as nx
import torch_geometric as ptgeom

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

for ds in ["syn3", "syn1", "mutag"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))[:5]
    print(f"\n==================== {ds} ====================")

    for case_idx, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if ds in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        clean_g = d["clean_graphs"] if task == "node" else d["clean_graphs"][0]
        att_g = d["attacked_graphs"] if task == "node" else d["attacked_graphs"][0]

        cg_np = clean_g.numpy()
        ag_np = att_g.numpy()
        c_set = set((min(int(cg_np[0, i]), int(cg_np[1, i])), max(int(cg_np[0, i]), int(cg_np[1, i]))) for i in range(cg_np.shape[1]) if cg_np[0, i] != cg_np[1, i])
        a_set = set((min(int(ag_np[0, i]), int(ag_np[1, i])), max(int(ag_np[0, i]), int(ag_np[1, i]))) for i in range(ag_np.shape[1]) if ag_np[0, i] != ag_np[1, i])

        e_add = a_set - c_set
        if not e_add:
            print(f"Case {case_idx}: (No edges added by attacker)")
            continue

        G = nx.Graph()
        for i in range(ag_np.shape[1]):
            u, v = int(ag_np[0, i]), int(ag_np[1, i])
            if u != v:
                G.add_edge(u, v)

        print(f"Case {case_idx} (Target indx={indx}):")
        for (u, v) in e_add:
            deg_u = G.degree[u]
            deg_v = G.degree[v]
            nu = set(G.neighbors(u)) - {v}
            nv = set(G.neighbors(v)) - {u}
            jaccard = len(nu & nv) / max(1, len(nu | nv))

            # Span: shortest path if this edge is temporarily removed
            G.remove_edge(u, v)
            try:
                span = nx.shortest_path_length(G, u, v)
            except nx.NetworkXNoPath:
                span = "inf"
            G.add_edge(u, v)

            touches_target = (u == indx or v == indx)
            print(f"   Added Edge ({u}, {v}): deg=({deg_u}, {deg_v}) | Touches Target? {touches_target} | Span={span} | Jaccard={jaccard:.3f}")
