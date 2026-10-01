import os
import glob
import torch
import numpy as np
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

folder = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer\syn3\deduction"
files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))

model = NodeGCN(10, 2)
ckpt = torch.load("./checkpoints/GNN/syn3/best_model", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

total_fools = 0
from_added_edges = 0
from_pulled_in_3hop = 0
from_existing_3hop = 0

for fpath in files:
    data = torch.load(fpath, map_location="cpu")
    indx = int(data["indx"])
    k = data["k"]
    features = data["features"]
    clean_g = data["clean_graphs"]
    att_g = data["attacked_graphs"]
    clean_Es = set((min(u, v), max(u, v)) for u, v in data["clean_Es_pairs"])

    # Clean 3-hop subgraph edges and full clean edges
    clean_subg = ptgeom.utils.k_hop_subgraph(indx, 3, clean_g)[1].numpy()
    clean_subg_set = set((min(int(clean_subg[0, i]), int(clean_subg[1, i])), max(int(clean_subg[0, i]), int(clean_subg[1, i]))) for i in range(clean_subg.shape[1]))
    clean_full_np = clean_g.numpy()
    clean_full_set = set((min(int(clean_full_np[0, i]), int(clean_full_np[1, i])), max(int(clean_full_np[0, i]), int(clean_full_np[1, i]))) for i in range(clean_full_np.shape[1]))

    explainer = PGExplainer(model, att_g, features, "node")
    explainer.explainer_model = torch.nn.Sequential(
        torch.nn.Linear(explainer.expl_embedding, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
    )
    explainer.explainer_model.load_state_dict(data["explainer_state_dict"])
    explainer.explainer_model.eval()

    n_graph, n_expl = explainer.explain(indx)
    n_graph_np = n_graph.detach().numpy()
    n_expl_np = n_expl.detach().numpy()
    bi_n_expl = n_expl_np.copy()
    for i in range(n_expl_np.shape[0]):
        r_pair = [n_graph_np[1, i], n_graph_np[0, i]]
        if n_graph_np[0, i] <= n_graph_np[1, i]:
            n_expl_np[i] += bi_n_expl[index_edge(n_graph_np, r_pair)]
        else:
            n_expl_np[i] = 0

    now_id = np.flip(np.argsort(n_expl_np.reshape(-1))[-k:])
    for idx_e in now_id:
        u, v = int(n_graph_np[0, idx_e]), int(n_graph_np[1, idx_e])
        pair = (min(u, v), max(u, v))
        if pair not in clean_Es:
            total_fools += 1
            if pair not in clean_full_set:
                from_added_edges += 1
            elif pair not in clean_subg_set:
                from_pulled_in_3hop += 1
            else:
                from_existing_3hop += 1

print(f"Total Misaligned Edges across 15 cases: {total_fools}")
print(f"  1. Directly Added Fake Edges (E_add):          {from_added_edges} ({from_added_edges/total_fools*100:.1f}%)")
print(f"  2. Distant Edges Pulled into 3-Hop Subgraph:   {from_pulled_in_3hop} ({from_pulled_in_3hop/total_fools*100:.1f}%)")
print(f"  3. Existing 3-Hop Edges (Embedding Shift):     {from_existing_3hop} ({from_existing_3hop/total_fools*100:.1f}%)")
