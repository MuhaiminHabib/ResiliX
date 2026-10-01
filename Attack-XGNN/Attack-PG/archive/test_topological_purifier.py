import os, glob, torch
import numpy as np
import networkx as nx
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def get_exact_top_k(explainer, model, g_tensor, feats, indx, k, task, edge_weights=None):
    if task == "node":
        subg = ptgeom.utils.k_hop_subgraph(indx, 3, g_tensor)[1]
        embeds = model.embedding(feats, g_tensor, edge_weights=edge_weights).detach()
        input_expl = explainer._create_explainer_input(subg, embeds, indx).unsqueeze(0)
    else:
        subg = g_tensor[0]
        embeds = model.embedding(feats[0], subg, edge_weights=edge_weights).detach()
        input_expl = explainer._create_explainer_input(subg, embeds, 0).unsqueeze(0)

    sampling_weights = explainer.explainer_model(input_expl)
    mask = explainer._sample_graph(sampling_weights, training=False).squeeze()

    n_graph_np = np.array(subg.detach())
    n_expl_np = np.array(mask.detach())
    bi_n_expl = n_expl_np.copy()
    for i in range(n_expl_np.shape[0]):
        r_pair = [n_graph_np[1, i], n_graph_np[0, i]]
        if n_graph_np[0, i] <= n_graph_np[1, i]:
            n_expl_np[i] += bi_n_expl[index_edge(n_graph_np, r_pair)]
        else:
            n_expl_np[i] = 0

    now_id = np.flip(np.argsort(n_expl_np.reshape(-1))[-k:])
    top_pairs = set((int(n_graph_np[0, idx_e]), int(n_graph_np[1, idx_e])) for idx_e in now_id)
    return top_pairs

def purify_graph_edges(g_tensor, task, indx):
    g_in = g_tensor if task == "node" else g_tensor[0]
    g_np = g_in.numpy()

    G = nx.Graph()
    for i in range(g_np.shape[1]):
        u, v = int(g_np[0, i]), int(g_np[1, i])
        if u != v:
            G.add_edge(u, v)

    edges_to_mute = set()
    ew = torch.ones(g_in.size(1), dtype=torch.float32)

    for i in range(g_np.shape[1]):
        u, v = int(g_np[0, i]), int(g_np[1, i])
        if u >= v:
            continue

        nu = set(G.neighbors(u)) - {v}
        nv = set(G.neighbors(v)) - {u}
        shared_neighbors = len(nu & nv)
        deg_u = G.degree[u]
        deg_v = G.degree[v]

        G.remove_edge(u, v)
        try:
            span = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            span = 999
        G.add_edge(u, v)

        # Condition 1 (Graph datasets like MUTAG): Bridge edges with high span (>= 4 or disconnected)
        if task == "graph" and span >= 4:
            edges_to_mute.add((u, v))
            edges_to_mute.add((v, u))

        # Condition 2 (Node datasets like syn3/syn1): Edges directly touching target node indx
        # with zero common neighbors and span >= 3 (pure shortcut bridges)
        elif task == "node" and (u == indx or v == indx) and shared_neighbors == 0 and span >= 3:
            edges_to_mute.add((u, v))
            edges_to_mute.add((v, u))

    for i in range(g_np.shape[1]):
        if (int(g_np[0, i]), int(g_np[1, i])) in edges_to_mute:
            ew[i] = 0.0

    return ew, len(edges_to_mute) // 2

for ds in ["syn3", "syn1", "mutag"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
    if ds == "syn1": model = NodeGCN(10, 4)
    elif ds == "syn3": model = NodeGCN(10, 2)
    elif ds == "mutag": model = GraphGCN(14, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    clean_err, att_undef_err, att_purified_err = [], [], []
    muted_clean_cnt, muted_att_cnt = [], []

    for f in files:
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if ds in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        k = d["k"]
        feats = d["features"]
        clean_g = d["clean_graphs"]
        att_g = d["attacked_graphs"]
        clean_Es = set((int(u), int(v)) for u, v in d["clean_Es_pairs"])

        explainer = PGExplainer(model, clean_g, feats, task)
        explainer.explainer_model = torch.nn.Sequential(
            torch.nn.Linear(explainer.expl_embedding, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        explainer.explainer_model.load_state_dict(d["explainer_state_dict"])
        explainer.explainer_model.eval()

        # 1. Clean Sanity Check (with purifier applied to clean graph)
        clean_ew, c_muted = purify_graph_edges(clean_g, task, indx)
        c_top = get_exact_top_k(explainer, model, clean_g, feats, indx, k, task, edge_weights=clean_ew)
        clean_err.append(len(clean_Es - c_top) / k)
        muted_clean_cnt.append(c_muted)

        # 2. Undefended Attacked Graph
        undef_top = get_exact_top_k(explainer, model, att_g, feats, indx, k, task)
        att_undef_err.append(len(clean_Es - undef_top) / k)

        # 3. Purified Attacked Graph
        att_ew, a_muted = purify_graph_edges(att_g, task, indx)
        purified_top = get_exact_top_k(explainer, model, att_g, feats, indx, k, task, edge_weights=att_ew)
        att_purified_err.append(len(clean_Es - purified_top) / k)
        muted_att_cnt.append(a_muted)

    print(f"\n==================== {ds} ====================")
    print(f"Clean Graph Sanity Error   : {np.mean(clean_err)*100:5.2f}% (Avg Muted on Clean: {np.mean(muted_clean_cnt):.2f} edges)")
    print(f"Undefended Attack Error    : {np.mean(att_undef_err)*100:5.2f}%")
    print(f"Defended (Purified) Error  : {np.mean(att_purified_err)*100:5.2f}% (Avg Muted on Attacked: {np.mean(muted_att_cnt):.2f} edges)")
    drop_pct = (np.mean(att_undef_err) - np.mean(att_purified_err)) * 100.0
    print(f"Error Drop Under Attack    : -{drop_pct:.2f}%")
