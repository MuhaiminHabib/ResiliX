import os, glob, torch
import numpy as np
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def get_exact_top_k(explainer, g_tensor, indx, k, task):
    explainer.graphs = g_tensor
    n_graph, n_expl = explainer.explain(indx if task == "node" else 0)
    n_graph_np = np.array(n_graph.detach())
    n_expl_np = np.array(n_expl.detach())

    bi_n_expl = n_expl_np.copy()
    for i in range(n_expl_np.shape[0]):
        r_pair = [n_graph_np[1, i], n_graph_np[0, i]]
        if n_graph_np[0, i] <= n_graph_np[1, i]:
            n_expl_np[i] += bi_n_expl[index_edge(n_graph_np, r_pair)]
        else:
            n_expl_np[i] = 0

    now_id = np.flip(np.argsort(n_expl_np.reshape(-1))[-k:])
    top_pairs = set((int(n_graph_np[0, idx_e]), int(n_graph_np[1, idx_e])) for idx_e in now_id)
    return top_pairs, n_graph_np, n_expl_np

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

    clean_err, att_err, remove_add_err, restore_del_err = [], [], [], []

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

        # 1. Clean sanity check
        c_top, _, _ = get_exact_top_k(explainer, clean_g, indx, k, task)
        clean_err.append(len(clean_Es - c_top) / k)

        # 2. Attacked baseline
        a_top, _, _ = get_exact_top_k(explainer, att_g, indx, k, task)
        att_err.append(len(clean_Es - a_top) / k)

        # Find E_add and E_del
        cg_np = clean_g[0].numpy() if task == "graph" else clean_g.numpy()
        ag_np = att_g[0].numpy() if task == "graph" else att_g.numpy()
        c_set = set((int(cg_np[0, i]), int(cg_np[1, i])) for i in range(cg_np.shape[1]))
        a_set = set((int(ag_np[0, i]), int(ag_np[1, i])) for i in range(ag_np.shape[1]))

        # 3. Graph with E_add removed (keeping only edges that existed in clean_g)
        kept_edges = [ [int(ag_np[0, i]), int(ag_np[1, i])] for i in range(ag_np.shape[1]) if (int(ag_np[0, i]), int(ag_np[1, i])) in c_set ]
        g_no_add = torch.tensor(np.array(kept_edges).T, dtype=torch.long)
        if task == "graph": g_no_add = g_no_add.unsqueeze(0)
        no_add_top, _, _ = get_exact_top_k(explainer, g_no_add, indx, k, task)
        remove_add_err.append(len(clean_Es - no_add_top) / k)

        # 4. Graph with E_del restored (att_g + deleted edges)
        del_edges = list(c_set - a_set)
        if del_edges:
            del_t = torch.tensor(np.array(del_edges).T, dtype=torch.long)
            g_restored = torch.cat([att_g[0] if task == "graph" else att_g, del_t], dim=1)
        else:
            g_restored = att_g[0] if task == "graph" else att_g
        if task == "graph": g_restored = g_restored.unsqueeze(0)
        rest_top, _, _ = get_exact_top_k(explainer, g_restored, indx, k, task)
        restore_del_err.append(len(clean_Es - rest_top) / k)

    print(f"[{ds:13s}] Clean Sanity: {np.mean(clean_err)*100:5.2f}% | Attacked: {np.mean(att_err)*100:5.2f}% | If E_add Removed: {np.mean(remove_add_err)*100:5.2f}% | If E_del Restored: {np.mean(restore_del_err)*100:5.2f}%")
