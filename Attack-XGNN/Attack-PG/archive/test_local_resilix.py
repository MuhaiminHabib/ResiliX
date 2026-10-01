import os, glob, torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"

def get_top_k_pairs(subg_np, mask_np, k, is_graph=False):
    num_e = subg_np.shape[1]
    edge_dict = {(int(subg_np[0, i]), int(subg_np[1, i])): i for i in range(num_e)}
    bi = np.zeros(num_e, dtype=np.float32)
    for i in range(num_e):
        u, v = int(subg_np[0, i]), int(subg_np[1, i])
        if u <= v:
            rev_i = edge_dict.get((v, u), i)
            bi[i] = (mask_np[i] + mask_np[rev_i]) / (2.0 if is_graph else 1.0)
    top_id = np.flip(np.argsort(bi.reshape(-1))[-k:])
    return set((min(int(subg_np[0, i]), int(subg_np[1, i])), max(int(subg_np[0, i]), int(subg_np[1, i]))) for i in top_id)

def train_local_resilix(model, graphs, features, task, indx, epochs=50, lr=0.003, drop_rate=0.15, emb_noise=0.04):
    explainer = PGExplainer(model, graphs, features, task, epochs=epochs, lr=lr)
    explainer.explainer_model = nn.Sequential(
        nn.Linear(explainer.expl_embedding, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = Adam(explainer.explainer_model.parameters(), lr=lr)
    temp_schedule = lambda e: 5.0 * ((2.0 / 5.0) ** (e / epochs))

    if task == "node":
        _, subg, _, hard_edge_mask = ptgeom.utils.k_hop_subgraph(indx, 3, graphs, relabel_nodes=False)
        f_in = features
        g_in = graphs
        n_id = indx
        num_local_edges = subg.size(1)
    else:
        subg = graphs[0]
        f_in = features[0]
        g_in = graphs[0]
        n_id = 0
        hard_edge_mask = slice(None)
        num_local_edges = subg.size(1)

    with torch.no_grad():
        orig_pred = model(f_in, g_in)
        if task == "node": orig_pred = orig_pred[indx]
        target_label = torch.argmax(orig_pred).unsqueeze(0)

    for e in range(epochs):
        optimizer.zero_grad()
        t = temp_schedule(e)

        # Apply edge jitter directly to the local subgraph edges
        full_edge_weights = torch.ones(g_in.size(1))
        local_drop = (torch.rand(num_local_edges) > drop_rate).float()
        local_weights = local_drop + (1.0 - local_drop) * 0.15

        if task == "node":
            full_edge_weights[hard_edge_mask] = local_weights
        else:
            full_edge_weights = local_weights

        noisy_embeds = model.embedding(f_in, g_in, edge_weights=full_edge_weights).detach()
        noisy_embeds = noisy_embeds + torch.randn_like(noisy_embeds) * emb_noise

        input_expl = explainer._create_explainer_input(subg, noisy_embeds, n_id).unsqueeze(0)
        sampling_weights = explainer.explainer_model(input_expl)
        mask = explainer._sample_graph(sampling_weights, t, bias=0.0, training=True).squeeze()

        masked_pred = model(f_in, subg, edge_weights=mask)
        if task == "node": masked_pred = masked_pred[indx].unsqueeze(0)

        loss = explainer._loss(masked_pred, target_label, mask, explainer.reg_coefs)
        loss.backward()
        optimizer.step()

    return explainer

@torch.no_grad()
def explain_local_resilix(model, explainer, g_tensor, features, task, indx, k_views=15, drop_rate=0.12):
    if task == "node":
        _, subg, _, hard_edge_mask = ptgeom.utils.k_hop_subgraph(indx, 3, g_tensor, relabel_nodes=False)
        f_in = features
        g_in = g_tensor
        n_id = indx
        num_local_edges = subg.size(1)
    else:
        subg = g_tensor[0]
        f_in = features[0]
        g_in = g_tensor[0]
        n_id = 0
        hard_edge_mask = slice(None)
        num_local_edges = subg.size(1)

    masks = []
    for v in range(k_views):
        if v == 0:
            emb = model.embedding(f_in, g_in).detach()
        else:
            full_ew = torch.ones(g_in.size(1))
            local_drop = (torch.rand(num_local_edges) > drop_rate).float()
            local_ew = local_drop + (1.0 - local_drop) * 0.2
            if task == "node":
                full_ew[hard_edge_mask] = local_ew
            else:
                full_ew = local_ew
            emb = model.embedding(f_in, g_in, edge_weights=full_ew).detach()

        inp = explainer._create_explainer_input(subg, emb, n_id).unsqueeze(0)
        m = explainer._sample_graph(explainer.explainer_model(inp), training=False).squeeze().numpy()
        masks.append(m)

    avg_mask = np.median(np.stack(masks, axis=0), axis=0)
    return subg.numpy(), avg_mask

for ds in ["syn3", "syn1", "mutag"]:
    folder = os.path.join(SAVE_ROOT, ds, "deduction")
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))[:10]
    if ds == "syn1": model = NodeGCN(10, 4)
    elif ds == "syn3": model = NodeGCN(10, 2)
    elif ds == "mutag": model = GraphGCN(14, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{ds}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    undef_fools, def_fools = [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        task = d.get("task", "node" if ds in ["syn1", "syn3"] else "graph")
        indx = int(d["indx"])
        k = d["k"]
        feats = d["features"]
        clean_g = d["clean_graphs"]
        att_g = d["attacked_graphs"]

        undef_fools.append(d["undefended_fool_count"] / k)

        rob_expl = train_local_resilix(model, clean_g, feats, task, indx)
        c_sub, c_mask = explain_local_resilix(model, rob_expl, clean_g, feats, task, indx)
        a_sub, a_mask = explain_local_resilix(model, rob_expl, att_g, feats, task, indx)

        c_top = get_top_k_pairs(c_sub, c_mask, k, is_graph=(task=="graph"))
        a_top = get_top_k_pairs(a_sub, a_mask, k, is_graph=(task=="graph"))

        fool_def = len(c_top - a_top) / k
        def_fools.append(fool_def)

    drop_val = (np.mean(undef_fools) - np.mean(def_fools)) * 100.0
    print(f"[{ds:6s}] Undefended Attack: {np.mean(undef_fools)*100:5.2f}%  --->  ResiliX Defended: {np.mean(def_fools)*100:5.2f}% (Drop: -{drop_val:5.2f}%)")
