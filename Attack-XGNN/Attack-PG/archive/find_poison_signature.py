import os, glob, torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as ptgeom
import networkx as nx
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

for ds in ["syn3", "syn1", "mutag", "REDDIT-BINARY"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
    if ds == "syn1": model = NodeGCN(10, 4)
    elif ds == "syn3": model = NodeGCN(10, 2)
    elif ds == "mutag": model = GraphGCN(14, 2)
    elif ds == "REDDIT-BINARY": model = GraphGCN(11, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    add_cos1, real_cos1 = [], []
    add_cos3, real_cos3 = [], []
    add_emb_shift, real_emb_shift = [], []
    add_deg_prod, real_deg_prod = [], []

    for f in files[:20]:
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if ds in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        feats = d["features"] if task == "node" else d["features"][0]
        clean_g = d["clean_graphs"] if task == "node" else d["clean_graphs"][0]
        att_g = d["attacked_graphs"] if task == "node" else d["attacked_graphs"][0]

        cg_np = clean_g.numpy()
        ag_np = att_g.numpy()
        c_set = set((min(int(cg_np[0, i]), int(cg_np[1, i])), max(int(cg_np[0, i]), int(cg_np[1, i]))) for i in range(cg_np.shape[1]) if cg_np[0, i] != cg_np[1, i])

        if task == "node":
            subg = ptgeom.utils.k_hop_subgraph(indx, 3, att_g)[1].numpy()
        else:
            subg = ag_np

        sub_edges = set((min(int(subg[0, i]), int(subg[1, i])), max(int(subg[0, i]), int(subg[1, i]))) for i in range(subg.shape[1]) if subg[0, i] != subg[1, i])

        with torch.no_grad():
            out1 = F.relu(F.normalize(model.conv1(feats, att_g), p=2, dim=1))
            full_emb = model.embedding(feats, att_g)

        G = nx.Graph()
        G.add_edges_from(sub_edges)

        for (u, v) in sub_edges:
            c1 = F.cosine_similarity(out1[u].unsqueeze(0), out1[v].unsqueeze(0)).item()
            c3 = F.cosine_similarity(full_emb[u].unsqueeze(0), full_emb[v].unsqueeze(0)).item()
            dp = G.degree[u] * G.degree[v]

            # Measure how much removing (u, v) shifts the local embeddings
            mask_e = ~(((att_g[0] == u) & (att_g[1] == v)) | ((att_g[0] == v) & (att_g[1] == u)))
            with torch.no_grad():
                emb_without = model.embedding(feats, att_g[:, mask_e])
                if task == "node":
                    shift = torch.norm(full_emb[indx] - emb_without[indx], p=2).item()
                else:
                    shift = torch.norm(full_emb - emb_without, p=2).item()

            if (u, v) not in c_set:
                add_cos1.append(c1)
                add_cos3.append(c3)
                add_emb_shift.append(shift)
                add_deg_prod.append(dp)
            else:
                real_cos1.append(c1)
                real_cos3.append(c3)
                real_emb_shift.append(shift)
                real_deg_prod.append(dp)

    print(f"\n=== {ds} (Added Poison Edges vs Real Edges) ===")
    if add_cos1:
        print(f"  1. Layer-1 Cosine Sim  | Added: {np.mean(add_cos1):.4f} vs Real: {np.mean(real_cos1):.4f}")
        print(f"  2. Full Emb Cosine Sim | Added: {np.mean(add_cos3):.4f} vs Real: {np.mean(real_cos3):.4f}")
        print(f"  3. Embedding Shift     | Added: {np.mean(add_emb_shift):.4f} vs Real: {np.mean(real_emb_shift):.4f}")
        print(f"  4. Degree Product (u*v)| Added: {np.mean(add_deg_prod):.1f} vs Real: {np.mean(real_deg_prod):.1f}")
    else:
        print("  (Few added edges in first 20 cases)")
