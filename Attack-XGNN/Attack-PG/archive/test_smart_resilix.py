import os, glob, torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as ptgeom
import networkx as nx
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def get_undirected_pairs(subg_np, scores, k, is_graph=False):
    num_e = subg_np.shape[1]
    edge_dict = {(int(subg_np[0, i]), int(subg_np[1, i])): i for i in range(num_e)}
    bi = np.zeros(num_e, dtype=np.float32)
    for i in range(num_e):
        u, v = int(subg_np[0, i]), int(subg_np[1, i])
        if u <= v:
            rev_i = edge_dict.get((v, u), i)
            bi[i] = (scores[i] + scores[rev_i]) / (2.0 if is_graph else 1.0)
    top_indices = np.flip(np.argsort(bi.reshape(-1))[-k:])
    return set((min(int(subg_np[0, i]), int(subg_np[1, i])), max(int(subg_np[0, i]), int(subg_np[1, i]))) for i in top_indices)

@torch.no_grad()
def adaptive_causal_defense(model, explainer, g_tensor, feats, task, indx, k, alpha=1.2, top_eval_factor=3):
    if task == "node":
        _, subg, _, hard_edge_mask = ptgeom.utils.k_hop_subgraph(indx, 3, g_tensor, relabel_nodes=False)
        target_id = indx
        f_in = feats
        g_in = g_tensor
    else:
        subg = g_tensor[0]
        target_id = 0
        f_in = feats[0]
        g_in = g_tensor[0]

    subg_np = subg.cpu().numpy()
    num_sub_edges = subg_np.shape[1]
    edge_dict_sub = {(int(subg_np[0, i]), int(subg_np[1, i])): i for i in range(num_sub_edges)}

    # 1. Base explainer importance mask
    base_emb = model.embedding(f_in, g_in).detach()
    inp = explainer._create_explainer_input(subg, base_emb, target_id).unsqueeze(0)
    raw_mask = explainer._sample_graph(explainer.explainer_model(inp), training=False).squeeze().cpu().numpy()

    # Symmetrize raw mask
    bi_mask = np.zeros(num_sub_edges, dtype=np.float32)
    for i in range(num_sub_edges):
        u, v = int(subg_np[0, i]), int(subg_np[1, i])
        if u <= v:
            rev_i = edge_dict_sub.get((v, u), i)
            bi_mask[i] = (raw_mask[i] + raw_mask[rev_i]) / 2.0
            bi_mask[rev_i] = bi_mask[i]

    # 2. Identify top candidate pool for causal validation
    cand_count = min(num_sub_edges, k * top_eval_factor)
    candidate_edge_indices = np.argsort(bi_mask)[-cand_count:]

    # Clean target class probability
    orig_out = model(f_in, g_in)
    if task == "node":
        orig_out = orig_out[indx].unsqueeze(0)
    pred_label = torch.argmax(orig_out, dim=-1).item()
    base_prob = F.softmax(orig_out, dim=-1)[0, pred_label].item()

    # Degree map for density-aware sensitivity
    G_sub = nx.Graph()
    for i in range(num_sub_edges):
        u, v = int(subg_np[0, i]), int(subg_np[1, i])
        if u != v:
            G_sub.add_edge(u, v)
    degrees = dict(G_sub.degree())

    # Map subg edges to global graph edge index
    g_np = g_in.cpu().numpy()
    edge_dict_global = {(int(g_np[0, i]), int(g_np[1, i])): i for i in range(g_np.shape[1])}

    causal_weights = np.ones(num_sub_edges, dtype=np.float32)

    # 3. Test marginal GNN prediction necessity on candidate edges
    for idx_e in candidate_edge_indices:
        u, v = int(subg_np[0, idx_e]), int(subg_np[1, idx_e])
        if u > v:
            continue

        deg_uv = min(degrees.get(u, 1), degrees.get(v, 1))

        # Counterfactual: evaluate prediction without edge (u, v)
        ew = torch.ones(g_in.size(1))
        if (u, v) in edge_dict_global:
            ew[edge_dict_global[(u, v)]] = 0.0
        if (v, u) in edge_dict_global:
            ew[edge_dict_global[(v, u)]] = 0.0

        cf_out = model(f_in, g_in, edge_weights=ew)
        if task == "node":
            cf_out = cf_out[indx].unsqueeze(0)
        cf_prob = F.softmax(cf_out, dim=-1)[0, pred_label].item()

        # Marginal drop in confidence
        delta = base_prob - cf_prob

        # Density-weighted causal coefficient
        # Fragile nodes (low degree) with positive delta receive higher priority
        causal_multiplier = 1.0 + alpha * delta * (1.0 + 1.0 / deg_uv)
        causal_multiplier = max(0.1, causal_multiplier)

        rev_idx = edge_dict_sub.get((v, u), idx_e)
        causal_weights[idx_e] = causal_multiplier
        causal_weights[rev_idx] = causal_multiplier

    # 4. Connected component bonus
    refined_scores = bi_mask * causal_weights
    top_core = np.argsort(refined_scores)[-(k + 2):]
    G_core = nx.Graph()
    for idx_e in top_core:
        u, v = int(subg_np[0, idx_e]), int(subg_np[1, idx_e])
        if u != v:
            G_core.add_edge(u, v)

    if G_core.number_of_nodes() > 0:
        ccs = list(nx.connected_components(G_core))
        # Find component containing target node or largest connected core
        target_cc = None
        for cc in ccs:
            if target_id in cc:
                target_cc = cc
                break
        if target_cc is None:
            target_cc = max(ccs, key=len)

        for i in range(num_sub_edges):
            u, v = int(subg_np[0, i]), int(subg_np[1, i])
            if u not in target_cc and v not in target_cc:
                refined_scores[i] *= 0.5

    return subg_np, refined_scores


for ds in ["syn3", "syn1", "mutag"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))[:15]
    if ds == "syn1":
        model = NodeGCN(10, 4)
    elif ds == "syn3":
        model = NodeGCN(10, 2)
    elif ds == "mutag":
        model = GraphGCN(14, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    clean_sanity_fools = []
    undef_fools = []
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

        explainer = PGExplainer(model, att_g, feats, task)
        explainer.explainer_model = torch.nn.Sequential(
            torch.nn.Linear(explainer.expl_embedding, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        explainer.explainer_model.load_state_dict(d["explainer_state_dict"])
        explainer.explainer_model.eval()

        undef_fools.append(d["undefended_fool_count"] / k)

        # 1. Clean Sanity Check: Run defense on clean graph
        c_sub, c_scores = adaptive_causal_defense(model, explainer, clean_g, feats, task, indx, k)
        c_top = get_undirected_pairs(c_sub, c_scores, k, is_graph=(task == "graph"))
        clean_sanity_fools.append(len(clean_Es - c_top) / k)

        # 2. Defended Attacked Graph
        a_sub, a_scores = adaptive_causal_defense(model, explainer, att_g, feats, task, indx, k)
        a_top = get_undirected_pairs(a_sub, a_scores, k, is_graph=(task == "graph"))
        def_fools.append(len(clean_Es - a_top) / k)

    drop_val = (np.mean(undef_fools) - np.mean(def_fools)) * 100.0
    print(f"[{ds:6s}] Clean Sanity: {np.mean(clean_sanity_fools)*100:5.2f}% | Undefended: {np.mean(undef_fools)*100:5.2f}% ---> Defended: {np.mean(def_fools)*100:5.2f}% (Drop: -{drop_val:5.2f}%)")
