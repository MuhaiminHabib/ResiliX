import os, glob, torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as ptgeom
import networkx as nx
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

for ds in ["syn3", "syn1", "mutag"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))[:10]
    if ds == "syn1": model = NodeGCN(10, 4)
    elif ds == "syn3": model = NodeGCN(10, 2)
    elif ds == "mutag": model = GraphGCN(14, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    e_add_in_suspects = 0
    total_cases_with_add = 0
    loss_diff_when_e_add_muted = []

    for f in files:
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if ds in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        feats = d["features"]
        clean_g = d["clean_graphs"]
        att_g = d["attacked_graphs"]

        cg_np = clean_g[0].numpy() if task == "graph" else clean_g.numpy()
        ag_np = att_g[0].numpy() if task == "graph" else att_g.numpy()
        c_set = set((min(int(cg_np[0, i]), int(cg_np[1, i])), max(int(cg_np[0, i]), int(cg_np[1, i]))) for i in range(cg_np.shape[1]) if cg_np[0, i] != cg_np[1, i])
        a_set = set((min(int(ag_np[0, i]), int(ag_np[1, i])), max(int(ag_np[0, i]), int(ag_np[1, i]))) for i in range(ag_np.shape[1]) if ag_np[0, i] != ag_np[1, i])
        e_add = a_set - c_set

        if not e_add:
            continue
        total_cases_with_add += 1

        explainer = PGExplainer(model, clean_g, feats, task)
        explainer.explainer_model = torch.nn.Sequential(
            torch.nn.Linear(explainer.expl_embedding, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        explainer.explainer_model.load_state_dict(d["explainer_state_dict"])
        explainer.explainer_model.eval()

        # Compute base PG loss on att_g
        g_in = att_g if task == "node" else att_g[0]
        f_in = feats if task == "node" else feats[0]
        subg = ptgeom.utils.k_hop_subgraph(indx, 3, g_in)[1] if task == "node" else g_in

        emb0 = model.embedding(f_in, g_in).detach()
        inp0 = explainer._create_explainer_input(subg, emb0, indx if task == "node" else 0).unsqueeze(0)
        mask0 = explainer._sample_graph(explainer.explainer_model(inp0), training=False).squeeze()
        with torch.no_grad():
            mp0 = model(f_in, subg, edge_weights=mask0)
            op0 = model(f_in, subg)
            if task == "node":
                mp0 = mp0[indx].unsqueeze(0)
                op0 = op0[indx]
            loss0 = explainer._loss(mp0, torch.argmax(op0).unsqueeze(0), mask0, explainer.reg_coefs).item()

        # Mute true E_add and check PG loss
        ew = torch.ones(g_in.size(1))
        for i in range(g_in.size(1)):
            pair = (min(int(g_in[0, i]), int(g_in[1, i])), max(int(g_in[0, i]), int(g_in[1, i])))
            if pair in e_add:
                ew[i] = 0.0

        emb_clean_add = model.embedding(f_in, g_in, edge_weights=ew).detach()
        inp_ca = explainer._create_explainer_input(subg, emb_clean_add, indx if task == "node" else 0).unsqueeze(0)
        mask_ca = explainer._sample_graph(explainer.explainer_model(inp_ca), training=False).squeeze()
        with torch.no_grad():
            mp_ca = model(f_in, subg, edge_weights=mask_ca)
            op_ca = model(f_in, subg)
            if task == "node":
                mp_ca = mp_ca[indx].unsqueeze(0)
                op_ca = op_ca[indx]
            loss_ca = explainer._loss(mp_ca, torch.argmax(op_ca).unsqueeze(0), mask_ca, explainer.reg_coefs).item()

        loss_diff = loss_ca - loss0
        loss_diff_when_e_add_muted.append(loss_diff)

    print(f"\n[{ds:6s}] Cases with Added Edges: {total_cases_with_add}/10")
    print(f"  When True E_add is Muted: Average Loss Change = {np.mean(loss_diff_when_e_add_muted):+.6f}")
    print(f"  (Does PG-Loss decrease? {'YES (good signal)' if np.mean(loss_diff_when_e_add_muted) < 0 else 'NO (PG-loss does not decrease!)'})")
