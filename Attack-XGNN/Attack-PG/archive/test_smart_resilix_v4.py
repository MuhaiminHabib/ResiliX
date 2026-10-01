import os, glob, torch
import numpy as np
import networkx as nx
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def eval_defense(ds, attack="deduction"):
    folder = os.path.join(SAVE_ROOT, ds, attack)
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
    if ds == "syn1": model = NodeGCN(10, 4)
    elif ds == "syn3": model = NodeGCN(10, 2)
    elif ds == "mutag": model = GraphGCN(14, 2)
    elif ds == "REDDIT-BINARY": model = GraphGCN(11, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    undef_fools = []
    clean_sanity_fools = []
    def_fools = []

    for f in files:
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if ds in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        k = d["k"]
        feats = d["features"]
        clean_g = d["clean_graphs"]
        att_g = d["attacked_graphs"]
        clean_Es = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in d["clean_Es_pairs"])

        explainer = PGExplainer(model, clean_g, feats, task)
        explainer.explainer_model = torch.nn.Sequential(
            torch.nn.Linear(explainer.expl_embedding, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        explainer.explainer_model.load_state_dict(d["explainer_state_dict"])
        explainer.explainer_model.eval()

        undef_fools.append(d["undefended_fool_count"] / k)

        for is_clean, g_tensor in [(True, clean_g), (False, att_g)]:
            explainer.graphs = g_tensor
            subg, expl = explainer.explain(indx if task == "node" else 0)
            subg_np, expl_np = subg.detach().numpy(), expl.detach().numpy()

            num_e = subg_np.shape[1]
            raw_scores = {}
            for i in range(num_e):
                u, v = int(subg_np[0, i]), int(subg_np[1, i])
                if u != v:
                    pair = (min(u, v), max(u, v))
                    raw_scores[pair] = raw_scores.get(pair, 0.0) + float(expl_np[i])

            G = nx.Graph()
            for pair in raw_scores.keys():
                G.add_edge(pair[0], pair[1])

            refined_scores = {}
            for (u, v), s in raw_scores.items():
                cn = len(list(nx.common_neighbors(G, u, v)))
                deg = G.degree[u] + G.degree[v]
                cluster_factor = 1.0 + (cn / (deg + 1e-5))

                if task == "node" and G.has_node(indx):
                    try:
                        dist = min(nx.shortest_path_length(G, indx, u), nx.shortest_path_length(G, indx, v))
                    except:
                        dist = 3
                    dist_factor = 1.0 / (1.0 + 0.6 * dist)
                else:
                    dist_factor = 1.0

                refined_scores[(u, v)] = s * cluster_factor * dist_factor

            top_k_def = sorted(refined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            def_pairs = set(p for p, _ in top_k_def)

            misalign = len(clean_Es - def_pairs) / k
            if is_clean:
                clean_sanity_fools.append(misalign)
            else:
                def_fools.append(misalign)

    print(f"[{ds:13s}] Clean Sanity: {np.mean(clean_sanity_fools)*100:5.2f}% | Undefended: {np.mean(undef_fools)*100:5.2f}% ---> Defended: {np.mean(def_fools)*100:5.2f}% (Drop: -{(np.mean(undef_fools)-np.mean(def_fools))*100:5.2f}%)")

print("=== CONTEXTUAL RELEVANCE WEIGHTING TEST ===")
for ds in ["syn3", "syn1", "mutag", "REDDIT-BINARY"]:
    eval_defense(ds, "deduction")
