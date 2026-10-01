import os, glob, torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def explain_with_scc(explainer, model, g_tensor, feats, indx, k, task, lam=1.0):
    explainer.graphs = g_tensor
    n_graph, n_expl = explainer.explain(indx if task == "node" else 0)
    n_graph_np = np.array(n_graph.detach())
    n_expl_np = np.array(n_expl.detach())

    # 1. Base PGExplainer Symmetrized Scoring (Identical to Paper Baseline)
    bi_n_expl = n_expl_np.copy()
    for i in range(n_expl_np.shape[0]):
        r_pair = [n_graph_np[1, i], n_graph_np[0, i]]
        if n_graph_np[0, i] <= n_graph_np[1, i]:
            n_expl_np[i] += bi_n_expl[index_edge(n_graph_np, r_pair)]
        else:
            n_expl_np[i] = 0

    # 2. Candidate Pool: Top 2*k undirected edges
    valid_indices = [i for i in range(n_expl_np.shape[0]) if n_graph_np[0, i] <= n_graph_np[1, i]]
    cand_sorted = sorted(valid_indices, key=lambda i: n_expl_np[i], reverse=True)[:min(len(valid_indices), 2 * k)]

    # 3. Base GNN Prediction
    g_in = g_tensor if task == "node" else g_tensor[0]
    f_in = feats if task == "node" else feats[0]

    with torch.no_grad():
        base_logits = model(f_in, g_in)
        if task == "node":
            base_logits = base_logits[indx].unsqueeze(0)
        target_class = torch.argmax(base_logits, dim=-1).item()
        base_prob = F.softmax(base_logits, dim=-1)[0, target_class].item()

    g_np = g_in.numpy()
    edge_map = {(int(g_np[0, i]), int(g_np[1, i])): i for i in range(g_np.shape[1])}

    final_scores = n_expl_np.copy()

    # 4. Measure Delta_e (Causal necessity for GNN prediction)
    for idx_e in cand_sorted:
        u, v = int(n_graph_np[0, idx_e]), int(n_graph_np[1, idx_e])
        ew = torch.ones(g_in.size(1), dtype=torch.float32)
        if (u, v) in edge_map: ew[edge_map[(u, v)]] = 0.0
        if (v, u) in edge_map: ew[edge_map[(v, u)]] = 0.0

        with torch.no_grad():
            cf_logits = model(f_in, g_in, edge_weights=ew)
            if task == "node":
                cf_logits = cf_logits[indx].unsqueeze(0)
            cf_prob = F.softmax(cf_logits, dim=-1)[0, target_class].item()

        delta = max(0.0, base_prob - cf_prob)
        final_scores[idx_e] += lam * delta

    now_id = np.flip(np.argsort(final_scores.reshape(-1))[-k:])
    top_pairs = set((int(n_graph_np[0, idx_e]), int(n_graph_np[1, idx_e])) for idx_e in now_id)
    return top_pairs

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

    clean_err, att_undef_err, att_def_err = [], [], []

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

        undef_err = d["undefended_fool_count"] / k
        att_undef_err.append(undef_err)

        # 1. Clean Graph Sanity Check
        c_top = explain_with_scc(explainer, model, clean_g, feats, indx, k, task, lam=1.0)
        clean_err.append(len(clean_Es - c_top) / k)

        # 2. Defended Attacked Graph
        a_top = explain_with_scc(explainer, model, att_g, feats, indx, k, task, lam=1.0)
        att_def_err.append(len(clean_Es - a_top) / k)

    print(f"\n==================== {ds} ====================")
    print(f"Clean Graph Sanity Error   : {np.mean(clean_err)*100:5.2f}%")
    print(f"Undefended Attack Error    : {np.mean(att_undef_err)*100:5.2f}%")
    print(f"Defended (SCC) Error       : {np.mean(att_def_err)*100:5.2f}%")
    drop_pct = (np.mean(att_undef_err) - np.mean(att_def_err)) * 100.0
    print(f"Error Drop Under Attack    : -{drop_pct:.2f}%")
