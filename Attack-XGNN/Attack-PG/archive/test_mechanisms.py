import os, glob, torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as ptgeom
import networkx as nx
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def get_undirected_top_k(subg_np, raw_mask_np, k):
    num_e = subg_np.shape[1]
    edge_dict = {(int(subg_np[0, i]), int(subg_np[1, i])): i for i in range(num_e)}
    bi_expl = np.zeros(num_e, dtype=np.float32)
    for i in range(num_e):
        u, v = int(subg_np[0, i]), int(subg_np[1, i])
        if u <= v:
            rev_i = edge_dict.get((v, u), i)
            bi_expl[i] = raw_mask_np[i] + raw_mask_np[rev_i]
    top_id = np.flip(np.argsort(bi_expl.reshape(-1))[-k:])
    return set((min(int(subg_np[0, i]), int(subg_np[1, i])), max(int(subg_np[0, i]), int(subg_np[1, i]))) for i in top_id), bi_expl

def eval_strategy(dataset, attack, strategy="baseline"):
    folder = os.path.join(SAVE_ROOT, dataset, attack)
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
    if dataset == "syn1": model = NodeGCN(10, 4)
    elif dataset == "syn3": model = NodeGCN(10, 2)
    elif dataset == "mutag": model = GraphGCN(14, 2)
    elif dataset == "REDDIT-BINARY": model = GraphGCN(11, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{dataset}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    clean_fools, att_fools = [], []

    for f in files:
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if dataset in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        k = d["k"]
        feats = d["features"]
        clean_Es = set((min(int(u), int(v)), max(int(u), int(v))) for u, v in d["clean_Es_pairs"])

        explainer = PGExplainer(model, d["attacked_graphs"], feats, task)
        explainer.explainer_model = torch.nn.Sequential(
            torch.nn.Linear(explainer.expl_embedding, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        explainer.explainer_model.load_state_dict(d["explainer_state_dict"])
        explainer.explainer_model.eval()

        for is_clean, g_tensor in [(True, d["clean_graphs"]), (False, d["attacked_graphs"])]:
            with torch.no_grad():
                if task == "node":
                    subg = ptgeom.utils.k_hop_subgraph(indx, 3, g_tensor)[1]
                    base_emb = model.embedding(feats, g_tensor).detach()
                    target_id = indx
                    f_in = feats
                    g_in = g_tensor
                else:
                    subg = g_tensor[0]
                    base_emb = model.embedding(feats[0], subg).detach()
                    target_id = 0
                    f_in = feats[0]
                    g_in = subg

                subg_np = subg.numpy()
                num_e = subg_np.shape[1]

                if strategy == "baseline":
                    inp = explainer._create_explainer_input(subg, base_emb, target_id).unsqueeze(0)
                    mask = explainer._sample_graph(explainer.explainer_model(inp), training=False).squeeze().numpy()

                elif strategy == "loo_jackknife":
                    # Leave-one-undirected-edge-out jackknife over the local subgraph
                    inp0 = explainer._create_explainer_input(subg, base_emb, target_id).unsqueeze(0)
                    mask0 = explainer._sample_graph(explainer.explainer_model(inp0), training=False).squeeze().numpy()
                    # Pick top-40 candidate edges to test leave-one-out on
                    _, bi0 = get_undirected_top_k(subg_np, mask0, k)
                    cand_e = np.argsort(bi0)[-min(num_e, 35):]
                    edge_dict_full = {}
                    g_np = g_in.numpy()
                    for idx_g in range(g_np.shape[1]):
                        edge_dict_full[(int(g_np[0, idx_g]), int(g_np[1, idx_g]))] = idx_g

                    masks_list = [mask0]
                    for ce in cand_e:
                        u, v = int(subg_np[0, ce]), int(subg_np[1, ce])
                        if u >= v: continue
                        ew = torch.ones(g_in.size(1))
                        if (u, v) in edge_dict_full: ew[edge_dict_full[(u, v)]] = 0.0
                        if (v, u) in edge_dict_full: ew[edge_dict_full[(v, u)]] = 0.0
                        emb_loo = model.embedding(f_in, g_in, edge_weights=ew).detach()
                        inp_loo = explainer._create_explainer_input(subg, emb_loo, target_id).unsqueeze(0)
                        m_loo = explainer._sample_graph(explainer.explainer_model(inp_loo), training=False).squeeze().numpy()
                        masks_list.append(m_loo)
                    # Use min/quantile or mean+std penalty
                    stack_m = np.stack(masks_list, axis=0)
                    mask = np.median(stack_m, axis=0) - 0.5 * np.std(stack_m, axis=0)

                elif strategy == "pred_fidelity_rerank":
                    # Score top-3k candidate edges by actual GNN prediction fidelity when kept!
                    inp0 = explainer._create_explainer_input(subg, base_emb, target_id).unsqueeze(0)
                    mask0 = explainer._sample_graph(explainer.explainer_model(inp0), training=False).squeeze().numpy()
                    _, bi0 = get_undirected_top_k(subg_np, mask0, k)
                    cand_e = np.flip(np.argsort(bi0)[-min(num_e, k * 3):])

                    orig_logits = model(f_in, g_in)
                    if task == "node": orig_logits = orig_logits[indx].unsqueeze(0)
                    orig_class = torch.argmax(orig_logits, dim=-1)

                    edge_dict_sub = {(int(subg_np[0, i]), int(subg_np[1, i])): i for i in range(num_e)}
                    mask = mask0.copy()
                    for ce in cand_e:
                        u, v = int(subg_np[0, ce]), int(subg_np[1, ce])
                        if u > v: continue
                        rev_ce = edge_dict_sub.get((v, u), ce)
                        # Test counterfactual drop of this candidate edge on mask0
                        test_w = torch.tensor(mask0, dtype=torch.float32)
                        test_w[ce] = 0.0
                        test_w[rev_ce] = 0.0
                        pred_drop = model(f_in, subg, edge_weights=test_w)
                        if task == "node": pred_drop = pred_drop[indx].unsqueeze(0)
                        cce_drop = F.cross_entropy(pred_drop, orig_class).item()
                        # Higher cce_drop when removed = truly essential for prediction!
                        mask[ce] += 0.25 * cce_drop
                        mask[rev_ce] += 0.25 * cce_drop

                top_pairs, _ = get_undirected_top_k(subg_np, mask, k)
                fool_ratio = len(clean_Es - top_pairs) / k
                if is_clean: clean_fools.append(fool_ratio)
                else: att_fools.append(fool_ratio)

    print(f"  {strategy:22s} | Clean Sanity: {np.mean(clean_fools)*100:5.2f}% | Attacked Misalign: {np.mean(att_fools)*100:5.2f}%")

for ds in ["syn3", "syn1", "mutag"]:
    print(f"\n--- {ds} (deduction) ---")
    for strat in ["baseline", "loo_jackknife", "pred_fidelity_rerank"]:
        eval_strategy(ds, "deduction", strat)
