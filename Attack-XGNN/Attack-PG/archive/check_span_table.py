import os, glob, torch
import numpy as np
import networkx as nx
import torch_geometric as ptgeom
from collections import Counter

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def get_edge_span(G, u, v):
    G.remove_edge(u, v)
    try:
        sp = nx.shortest_path_length(G, u, v)
    except nx.NetworkXNoPath:
        sp = -1  # -1 means cut-edge (no alternate path)
    G.add_edge(u, v)
    return sp

for ds in ["syn3", "syn1", "mutag", "REDDIT-BINARY"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))

    es_spans = Counter()
    clean_sub_spans = Counter()
    add_spans = Counter()

    for f in files:
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if ds in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        clean_g = d["clean_graphs"] if task == "node" else d["clean_graphs"][0]
        att_g = d["attacked_graphs"] if task == "node" else d["attacked_graphs"][0]
        clean_Es = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in d["clean_Es_pairs"] if int(u) != int(v))

        cg_np = clean_g.numpy()
        ag_np = att_g.numpy()
        c_set = set((min(int(cg_np[0, i]), int(cg_np[1, i])), max(int(cg_np[0, i]), int(cg_np[1, i]))) for i in range(cg_np.shape[1]) if cg_np[0, i] != cg_np[1, i])
        a_set = set((min(int(ag_np[0, i]), int(ag_np[1, i])), max(int(ag_np[0, i]), int(ag_np[1, i]))) for i in range(ag_np.shape[1]) if ag_np[0, i] != ag_np[1, i])
        e_add = a_set - c_set

        # Clean graph spans inside local subgraph
        G_clean = nx.Graph()
        G_clean.add_edges_from(c_set)
        sub_c = ptgeom.utils.k_hop_subgraph(indx, 3, clean_g)[1].numpy() if task == "node" else cg_np
        sub_c_set = set((min(int(sub_c[0, i]), int(sub_c[1, i])), max(int(sub_c[0, i]), int(sub_c[1, i]))) for i in range(sub_c.shape[1]) if sub_c[0, i] != sub_c[1, i])

        for (u, v) in sub_c_set:
            sp = get_edge_span(G_clean, u, v)
            clean_sub_spans[sp] += 1
            if (u, v) in clean_Es:
                es_spans[sp] += 1

        # Attacked graph spans for added edges
        G_att = nx.Graph()
        G_att.add_edges_from(a_set)
        for (u, v) in e_add:
            sp = get_edge_span(G_att, u, v)
            add_spans[sp] += 1

    print(f"\n==================== {ds} (Span Distribution; -1 = No Alternate Path) ====================")
    print(f"  Real Motif Edges (E_S)  : {dict(sorted(es_spans.items()))}")
    print(f"  All Clean Subgraph Edges: {dict(sorted(clean_sub_spans.items()))}")
    print(f"  Attacker Added (E_add)  : {dict(sorted(add_spans.items()))}")
