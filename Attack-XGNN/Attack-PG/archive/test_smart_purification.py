import os, glob, torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as ptgeom
import networkx as nx
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def get_exact_scores_and_topk(explainer, model, g_tensor, feats, indx, k, task, edge_weights=None):
    if task == "node":
        subg = ptgeom.utils.k_hop_subgraph(indx, 3, g_tensor)[1]
        embeds = model.embedding(feats, g_tensor, edge_weights=edge_weights).detach()
        input_expl = explainer._create_explainer_input(subg, embeds, indx).unsqueeze(0)
        f_in = feats
    else:
        subg = g_tensor[0]
        embeds = model.embedding(feats[0], subg, edge_weights=edge_weights).detach()
        input_expl = explainer._create_explainer_input(subg, embeds, 0).unsqueeze(0)
        f_in = feats[0]

    sampling_weights = explainer.explainer_model(input_expl)
    mask = explainer._sample_graph(sampling_weights, training=False).squeeze()

    # Compute PG internal loss for self-supervised quality check
    with torch.no_grad():
        masked_pred = model(f_in, subg, edge_weights=mask)
        orig_pred = model(f_in, subg)
        if task == "node":
            masked_pred = masked_pred[indx].unsqueeze(0)
            orig_pred = orig_pred[indx]
        pg_loss_val = explainer._loss(masked_pred, torch.argmax(orig_pred).unsqueeze(0), mask, explainer.reg_coefs).item()

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
    return top_pairs, n_graph_np, n_expl_np, pg_loss_val

def get_suspicious_edges(model, g_tensor, feats, indx, k, task, base_top_pairs):
    g_in = g_tensor if task == "node" else g_tensor[0]
    f_in = feats if task == "node" else feats[0]
    g_np = g_in.numpy()

    if task == "node":
        subg = ptgeom.utils.k_hop_subgraph(indx, 3, g_in)[1].numpy()
    else:
        subg = g_np

    with torch.no_grad():
        out1 = F.relu(F.normalize(model.conv1(f_in, g_in), p=2, dim=1))

    G_full = nx.Graph()
    for i in range(g_np.shape[1]):
        u, v = int(g_np[0, i]), int(g_np[1, i])
        if u != v:
            G_full.add_edge(u, v)

    sub_pairs = set((min(int(subg[0, i]), int(subg[1, i])), max(int(subg[0, i]), int(subg[1, i]))) for i in range(subg.shape[1]) if subg[0, i] != subg[1, i])

    candidates = []
    for (u, v) in sub_pairs:
        # Never prune edges that are already in the initial top-k explanation
        if (u, v) in base_top_pairs or (v, u) in base_top_pairs:
            continue
        deg_prod = G_full.degree[u] * G_full.degree[v]
        cos1 = F.cosine_similarity(out1[u].unsqueeze(0), out1[v].unsqueeze(0)).item()
        cn = len(list(nx.common_neighbors(G_full, u, v)))
        touch_bonus = 2.5 if (task == "node" and (u == indx or v == indx)) else 1.0
        # High degree product + high layer-1 similarity + low common neighbors = shortcut bridge
        suspicion = (deg_prod * max(0.0, cos1) * touch_bonus) / (1.0 + cn)
        candidates.append(((u, v), suspicion))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in candidates[:8]]

def make_edge_weights_without(g_tensor, task, muted_pairs):
    g_in = g_tensor if task == "node" else g_tensor[0]
    g_np = g_in.numpy()
    ew = torch.ones(g_in.size(1), dtype=torch.float32)
    muted_set = set()
    for u, v in muted_pairs:
        muted_set.add((u, v))
        muted_set.add((v, u))
    for i in range(g_np.shape[1]):
        if (int(g_np[0, i]), int(g_np[1, i])) in muted_set:
            ew[i] = 0.0
    return ew

for ds in ["syn3", "syn1", "mutag", "REDDIT-BINARY"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))[:20]
    if ds == "syn1": model = NodeGCN(10, 4)
    elif ds == "syn3": model = NodeGCN(10, 2)
    elif ds == "mutag": model = GraphGCN(14, 2)
    elif ds == "REDDIT-BINARY": model = GraphGCN(11, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    undef_err = []
    m1_clean, m1_att = [], []
    m2_clean, m2_att = [], []

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

        base_att_top, _, _, _ = get_exact_scores_and_topk(explainer, model, att_g, feats, indx, k, task)
        undef_err.append(len(clean_Es - base_att_top) / k)

        for is_clean, g_curr in [(True, clean_g), (False, att_g)]:
            base_top, base_subg, base_scores, base_loss = get_exact_scores_and_topk(explainer, model, g_curr, feats, indx, k, task)
            suspects = get_suspicious_edges(model, g_curr, feats, indx, k, task, base_top)

            # Method 1: PG-Loss Guided Single/Pairwise Surgical Mute
            best_top = base_top
            best_loss = base_loss
            kept_muted = []
            for s_pair in suspects[:5]:
                test_muted = kept_muted + [s_pair]
                ew = make_edge_weights_without(g_curr, task, test_muted)
                cand_top, _, _, cand_loss = get_exact_scores_and_topk(explainer, model, g_curr, feats, indx, k, task, edge_weights=ew)
                if cand_loss < best_loss - 1e-4:
                    best_loss = cand_loss
                    best_top = cand_top
                    kept_muted = test_muted

            # Method 2: Multi-View Max-Score Consensus across suspect mutes
            score_dict = {}
            for i in range(base_subg.shape[1]):
                if base_subg[0, i] <= base_subg[1, i]:
                    pair = (int(base_subg[0, i]), int(base_subg[1, i]))
                    score_dict[pair] = [float(base_scores[i])]

            for s_pair in suspects[:6]:
                ew = make_edge_weights_without(g_curr, task, [s_pair])
                _, v_subg, v_scores, _ = get_exact_scores_and_topk(explainer, model, g_curr, feats, indx, k, task, edge_weights=ew)
                for i in range(v_subg.shape[1]):
                    if v_subg[0, i] <= v_subg[1, i]:
                        pair = (int(v_subg[0, i]), int(v_subg[1, i]))
                        if pair in score_dict:
                            score_dict[pair].append(float(v_scores[i]))

            # Use 85th percentile across views so suppressed motif edges recover when their poison edge is muted
            agg_pairs = sorted(score_dict.items(), key=lambda x: np.percentile(x[1], 85), reverse=True)[:k]
            m2_top = set(p for p, _ in agg_pairs)

            if is_clean:
                m1_clean.append(len(clean_Es - best_top) / k)
                m2_clean.append(len(clean_Es - m2_top) / k)
            else:
                m1_att.append(len(clean_Es - best_top) / k)
                m2_att.append(len(clean_Es - m2_top) / k)

    print(f"\n[{ds:13s}] Undefended Attack: {np.mean(undef_err)*100:5.2f}%")
    print(f"  1. PG-Loss Guided Mute | Clean Sanity: {np.mean(m1_clean)*100:5.2f}% | Defended Attack: {np.mean(m1_att)*100:5.2f}%")
    print(f"  2. Multi-View Recovery | Clean Sanity: {np.mean(m2_clean)*100:5.2f}% | Defended Attack: {np.mean(m2_att)*100:5.2f}%")
