import os, glob, torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch_geometric as ptgeom
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.utils.graph import index_edge

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

def train_resilix_explainer(model, graphs, features, task, indx, epochs=50, lr=0.003, drop_rate=0.08, emb_noise=0.03):
    """
    ResiliX Robust Explainer Training + Multi-View Consensus Inference:
    Trains the PGExplainer MLP with stochastic edge-weight jitter and embedding smoothing
    so that its explanations are invariant to small adversarial edge additions/deletions.
    """
    explainer = PGExplainer(model, graphs, features, task, epochs=epochs, lr=lr)
    explainer.explainer_model = nn.Sequential(
        nn.Linear(explainer.expl_embedding, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    optimizer = Adam(explainer.explainer_model.parameters(), lr=lr)
    temp_schedule = lambda e: 5.0 * ((2.0 / 5.0) ** (e / epochs))

    if task == "node":
        subg = ptgeom.utils.k_hop_subgraph(indx, 3, graphs)[1]
        f_in = features
        g_in = graphs
        n_id = indx
    else:
        subg = graphs[0]
        f_in = features[0]
        g_in = graphs[0]
        n_id = 0

    with torch.no_grad():
        orig_pred = model(f_in, g_in)
        if task == "node":
            orig_pred = orig_pred[indx]
        target_label = torch.argmax(orig_pred).unsqueeze(0)
        base_embeds = model.embedding(f_in, g_in).detach()

    for e in range(epochs):
        optimizer.zero_grad()
        t = temp_schedule(e)

        # Stochastic edge-weight jitter during GNN embedding extraction
        ew_jitter = torch.ones(g_in.size(1))
        drop_mask = (torch.rand(g_in.size(1)) > drop_rate).float()
        ew_jitter = ew_jitter * drop_mask + (1.0 - drop_mask) * 0.2
        noisy_embeds = model.embedding(f_in, g_in, edge_weights=ew_jitter).detach()
        noisy_embeds = noisy_embeds + torch.randn_like(noisy_embeds) * emb_noise

        input_expl = explainer._create_explainer_input(subg, noisy_embeds, n_id).unsqueeze(0)
        sampling_weights = explainer.explainer_model(input_expl)
        mask = explainer._sample_graph(sampling_weights, t, bias=0.0, training=True).squeeze()

        masked_pred = model(f_in, subg, edge_weights=mask)
        if task == "node":
            masked_pred = masked_pred[indx].unsqueeze(0)

        loss = explainer._loss(masked_pred, target_label, mask, explainer.reg_coefs)
        loss.backward()
        optimizer.step()

    return explainer

@torch.no_grad()
def explain_with_resilix(model, explainer, g_tensor, features, task, indx, k_views=15, drop_rate=0.08):
    if task == "node":
        subg = ptgeom.utils.k_hop_subgraph(indx, 3, g_tensor)[1]
        f_in = features
        g_in = g_tensor
        n_id = indx
    else:
        subg = g_tensor[0]
        f_in = features[0]
        g_in = g_tensor[0]
        n_id = 0

    # Multi-view smoothed consensus at inference time
    masks = []
    for v in range(k_views):
        if v == 0:
            emb = model.embedding(f_in, g_in).detach()
        else:
            ew = (torch.rand(g_in.size(1)) > drop_rate).float()
            ew = ew + (1.0 - ew) * 0.25
            emb = model.embedding(f_in, g_in, edge_weights=ew).detach()
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

        # Train ResiliX-protected explainer on clean_g, then test on clean_g vs att_g!
        rob_expl = train_resilix_explainer(model, clean_g, feats, task, indx)
        c_sub, c_mask = explain_with_resilix(model, rob_expl, clean_g, feats, task, indx)
        a_sub, a_mask = explain_with_resilix(model, rob_expl, att_g, feats, task, indx)

        c_top = get_top_k_pairs(c_sub, c_mask, k, is_graph=(task=="graph"))
        a_top = get_top_k_pairs(a_sub, a_mask, k, is_graph=(task=="graph"))

        fool_def = len(c_top - a_top) / k
        def_fools.append(fool_def)

    print(f"[{ds:6s}] Undefended Attack Misalign: {np.mean(undef_fools)*100:5.2f}%  --->  ResiliX Defended Misalign: {np.mean(def_fools)*100:5.2f}% (Drop: -{(np.mean(undef_fools)-np.mean(def_fools))*100:5.2f}%)")
