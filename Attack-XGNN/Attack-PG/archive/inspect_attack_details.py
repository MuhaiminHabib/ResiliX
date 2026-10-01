import os, glob, torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def analyze_dataset(ds, attack="deduction"):
    folder = os.path.join(SAVE_ROOT, ds, attack)
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
    if ds == "syn1": model = NodeGCN(10, 4)
    elif ds == "syn3": model = NodeGCN(10, 2)
    elif ds == "mutag": model = GraphGCN(14, 2)
    elif ds == "REDDIT-BINARY": model = GraphGCN(11, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    es_clean_scores, es_att_scores = [], []
    fool_clean_scores, fool_att_scores = [], []
    es_deg, fool_deg = [], []
    es_cn, fool_cn = [], []
    es_dist, fool_dist = [], []
    es_gnn_drop, fool_gnn_drop = [], []
    fool_is_added = []

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

        # Clean explanation
        explainer.graphs = clean_g
        c_graph, c_expl = explainer.explain(indx if task == "node" else 0)
        c_graph_np, c_expl_np = c_graph.detach().numpy(), c_expl.detach().numpy()
        c_scores = {}
        for i in range(c_expl_np.shape[0]):
            u, v = int(c_graph_np[0, i]), int(c_graph_np[1, i])
            if u != v:
                pair = (min(u, v), max(u, v))
                c_scores[pair] = c_scores.get(pair, 0.0) + float(c_expl_np[i])

        # Attacked explanation
        explainer.graphs = att_g
        a_graph, a_expl = explainer.explain(indx if task == "node" else 0)
        a_graph_np, a_expl_np = a_graph.detach().numpy(), a_expl.detach().numpy()
        a_scores = {}
        for i in range(a_expl_np.shape[0]):
            u, v = int(a_graph_np[0, i]), int(a_graph_np[1, i])
            if u != v:
                pair = (min(u, v), max(u, v))
                a_scores[pair] = a_scores.get(pair, 0.0) + float(a_expl_np[i])

        # Top-k on attacked graph
        sorted_a = sorted(a_scores.items(), key=lambda x: x[1], reverse=True)
        att_top_k = [p for p, s in sorted_a[:k]]

        # Build NX graph of attacked subgraph
        G = nx.Graph()
        for p in a_scores.keys():
            G.add_edge(p[0], p[1])

        # Clean full edge set
        cg_np = clean_g[0].numpy() if task == "graph" else clean_g.numpy()
        clean_all_edges = set((min(int(cg_np[0, i]), int(cg_np[1, i])), max(int(cg_np[0, i]), int(cg_np[1, i]))) for i in range(cg_np.shape[1]) if cg_np[0, i] != cg_np[1, i])

        for p in clean_Es:
            if p in a_scores:
                es_clean_scores.append(c_scores.get(p, 0.0))
                es_att_scores.append(a_scores[p])
                es_deg.append(G.degree[p[0]] + G.degree[p[1]])
                es_cn.append(len(list(nx.common_neighbors(G, p[0], p[1]))))
                if task == "node" and G.has_node(indx):
                    try:
                        d_u = nx.shortest_path_length(G, indx, p[0])
                        d_v = nx.shortest_path_length(G, indx, p[1])
                        es_dist.append(min(d_u, d_v))
                    except:
                        pass

        for p in att_top_k:
            if p not in clean_Es:
                fool_clean_scores.append(c_scores.get(p, 0.0))
                fool_att_scores.append(a_scores[p])
                fool_is_added.append(1 if p not in clean_all_edges else 0)
                fool_deg.append(G.degree[p[0]] + G.degree[p[1]])
                fool_cn.append(len(list(nx.common_neighbors(G, p[0], p[1]))))
                if task == "node" and G.has_node(indx):
                    try:
                        d_u = nx.shortest_path_length(G, indx, p[0])
                        d_v = nx.shortest_path_length(G, indx, p[1])
                        fool_dist.append(min(d_u, d_v))
                    except:
                        pass

    print(f"\n==================== {ds} ({attack}) ====================")
    print(f"Real Motif Edges (E_S)     | Clean Score: {np.mean(es_clean_scores):.4f} -> Attacked Score: {np.mean(es_att_scores):.4f}")
    print(f"Winning Decoy Edges (Fool) | Clean Score: {np.mean(fool_clean_scores):.4f} -> Attacked Score: {np.mean(fool_att_scores):.4f}")
    print(f"Are Decoy Edges Newly Added? {np.mean(fool_is_added)*100:.1f}% are brand-new edges ({sum(fool_is_added)}/{len(fool_is_added)})")
    print(f"Sum of Endpoint Degrees    | Real E_S: {np.mean(es_deg):.2f} vs Decoy: {np.mean(fool_deg):.2f}")
    print(f"Common Neighbors (Triangles)| Real E_S: {np.mean(es_cn):.2f} vs Decoy: {np.mean(fool_cn):.2f}")
    if es_dist:
        print(f"Hop Distance to Target Node| Real E_S: {np.mean(es_dist):.2f} vs Decoy: {np.mean(fool_dist):.2f}")

for ds in ["syn3", "syn1", "mutag", "REDDIT-BINARY"]:
    analyze_dataset(ds, "deduction")
