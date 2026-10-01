import time
import numpy as np
import torch
import torch.nn as nn

import explainer_main
import models
import utils.io_utils as io_utils
from explainer import explain as exp_mod
from eval_resilix_gnnex import (
    compute_motif_auc_and_topk,
    run_deduction_attack_subgraph,
    resilix_c1_purify_subgraph,
    run_gnnexplainer_single,
    resilix_c3_consensus_rerank,
)


def extract_node_embeddings(model, adj_t, x_t):
    model.eval()
    with torch.no_grad():
        h = model.conv_first(x_t, adj_t)
        h = h[0] if isinstance(h, tuple) else h
        h = model.act(h)
        for i in range(model.num_layers - 2):
            h = model.conv_block[i](h, adj_t)
            h = h[0] if isinstance(h, tuple) else h
            h = model.act(h)
        z = model.conv_last(h, adj_t)
        z = z[0] if isinstance(z, tuple) else z
    return z[0]


def run_gsat_explainer(model, sub_adj, sub_feat, pred_label, node_idx_new, epochs=60, r_prior=0.5, beta_kl=0.05, smoothing=False):
    """
    GSAT (Graph Stochastic Attention - Miao et al., ICML 2022):
    Optimizes stochastic attention alpha_uv via the Graph Information Bottleneck:
    L = L_pred(Y, G_S) + beta * KL(Bernoulli(p_uv) || Bernoulli(r))
    """
    target_cls = pred_label[node_idx_new]
    adj_t = torch.tensor(sub_adj, dtype=torch.float).unsqueeze(0)
    x_t = torch.tensor(sub_feat, dtype=torch.float).unsqueeze(0)

    z = extract_node_embeddings(model, adj_t, x_t)
    n_sub, dim = z.shape
    u_idx, v_idx = np.where(np.triu(sub_adj, k=1) > 0)
    if len(u_idx) == 0:
        return np.zeros_like(sub_adj)

    z_u = z[u_idx]
    z_v = z[v_idx]
    z_q = z[node_idx_new].unsqueeze(0).expand(len(u_idx), -1)
    edge_feat = torch.cat([z_u, z_v, z_q, z_u * z_v], dim=-1).detach()

    att_net = nn.Sequential(
        nn.Linear(dim * 4, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(att_net.parameters(), lr=0.05)
    idx_u = torch.tensor(u_idx, dtype=torch.long)
    idx_v = torch.tensor(v_idx, dtype=torch.long)

    for ep in range(epochs):
        optimizer.zero_grad()
        logits = att_net(edge_feat).squeeze(-1)
        p_uv = torch.sigmoid(logits)

        temp = 0.4 if smoothing else 0.7
        noise_scale = 0.25 if smoothing else 0.10
        u_rand = torch.clamp(torch.rand_like(logits), 1e-4, 1.0 - 1e-4)
        gumbel = torch.log(u_rand) - torch.log(1.0 - u_rand)
        alpha = torch.sigmoid((logits + noise_scale * gumbel) / temp)

        masked_adj = torch.zeros((n_sub, n_sub), dtype=torch.float)
        masked_adj[idx_u, idx_v] = alpha
        masked_adj[idx_v, idx_u] = alpha

        ypred, _ = model(x_t, masked_adj.unsqueeze(0))
        probs = nn.Softmax(dim=0)(ypred[0, node_idx_new, :])
        pred_loss = -torch.log(probs[target_cls] + 1e-8)

        p_clamped = torch.clamp(p_uv, 1e-4, 1.0 - 1e-4)
        kl_loss = torch.mean(
            p_clamped * torch.log(p_clamped / r_prior)
            + (1.0 - p_clamped) * torch.log((1.0 - p_clamped) / (1.0 - r_prior))
        )

        loss = pred_loss + beta_kl * kl_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_p = torch.sigmoid(att_net(edge_feat).squeeze(-1)).cpu().numpy()
    out_mask = np.zeros_like(sub_adj, dtype=np.float32)
    out_mask[u_idx, v_idx] = final_p
    out_mask[v_idx, u_idx] = final_p
    return out_mask


def run_pgexplainer_parametric(model, sub_adj, sub_feat, pred_label, node_idx_new, epochs=60, smoothing=False):
    target_cls = pred_label[node_idx_new]
    adj_t = torch.tensor(sub_adj, dtype=torch.float).unsqueeze(0)
    x_t = torch.tensor(sub_feat, dtype=torch.float).unsqueeze(0)

    z = extract_node_embeddings(model, adj_t, x_t)
    n_sub, dim = z.shape
    u_idx, v_idx = np.where(np.triu(sub_adj, k=1) > 0)
    if len(u_idx) == 0:
        return np.zeros_like(sub_adj)

    z_u = z[u_idx]
    z_v = z[v_idx]
    z_q = z[node_idx_new].unsqueeze(0).expand(len(u_idx), -1)
    edge_feat = torch.cat([z_u, z_v, z_q, torch.abs(z_u - z_v)], dim=-1).detach()

    mlp = nn.Sequential(
        nn.Linear(dim * 4, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.05)
    idx_u = torch.tensor(u_idx, dtype=torch.long)
    idx_v = torch.tensor(v_idx, dtype=torch.long)

    for ep in range(epochs):
        optimizer.zero_grad()
        logits = mlp(edge_feat).squeeze(-1)
        if smoothing:
            u_rand = torch.clamp(torch.rand_like(logits), 1e-4, 1.0 - 1e-4)
            gumbel = torch.log(u_rand) - torch.log(1.0 - u_rand)
            weights = torch.sigmoid((logits + 0.15 * gumbel) / 0.5)
        else:
            weights = torch.sigmoid(logits)

        masked_adj = torch.zeros((n_sub, n_sub), dtype=torch.float)
        masked_adj[idx_u, idx_v] = weights
        masked_adj[idx_v, idx_u] = weights

        ypred, _ = model(x_t, masked_adj.unsqueeze(0))
        probs = nn.Softmax(dim=0)(ypred[0, node_idx_new, :])
        pred_loss = -torch.log(probs[target_cls] + 1e-8)
        size_loss = 0.005 * torch.sum(weights)
        ent = -weights * torch.log(weights + 1e-8) - (1.0 - weights) * torch.log(1.0 - weights + 1e-8)
        ent_loss = 0.1 * torch.mean(ent)
        loss = pred_loss + size_loss + ent_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_w = torch.sigmoid(mlp(edge_feat).squeeze(-1)).cpu().numpy()
    out_mask = np.zeros_like(sub_adj, dtype=np.float32)
    out_mask[u_idx, v_idx] = final_w
    out_mask[v_idx, u_idx] = final_w
    return out_mask


def run_grad_saliency(model, sub_adj, sub_feat, pred_label, node_idx_new, smoothing=False):
    n_samples = 5 if smoothing else 1
    accum_grad = np.zeros_like(sub_adj, dtype=np.float32)
    target_cls = pred_label[node_idx_new]

    for s in range(n_samples):
        adj_in = sub_adj.copy().astype(np.float32)
        if smoothing:
            noise = np.random.normal(0.0, 0.03, size=adj_in.shape).astype(np.float32)
            noise = 0.5 * (noise + noise.T)
            adj_in = np.clip(adj_in + noise * adj_in, 0.0, 1.0)

        adj_t = torch.tensor(adj_in, dtype=torch.float).unsqueeze(0).requires_grad_(True)
        x_t = torch.tensor(sub_feat, dtype=torch.float).unsqueeze(0)
        model.eval()
        ypred, _ = model(x_t, adj_t)
        probs = nn.Softmax(dim=0)(ypred[0, node_idx_new, :])
        loss = -torch.log(probs[target_cls] + 1e-8)
        loss.backward()

        grad_adj = np.abs(adj_t.grad[0].detach().cpu().numpy())
        grad_sym = 0.5 * (grad_adj + grad_adj.T)
        accum_grad += grad_sym

    return (accum_grad / float(n_samples)) * sub_adj


def run_integrated_gradients(model, sub_adj, sub_feat, pred_label, node_idx_new, steps=10, smoothing=False):
    target_cls = pred_label[node_idx_new]
    x_t = torch.tensor(sub_feat, dtype=torch.float).unsqueeze(0)
    ig_accum = np.zeros_like(sub_adj, dtype=np.float32)

    for alpha in np.linspace(0.1, 1.0, steps):
        scaled_adj = (alpha * sub_adj).astype(np.float32)
        if smoothing:
            noise = np.random.normal(0.0, 0.02, size=scaled_adj.shape).astype(np.float32)
            noise = 0.5 * (noise + noise.T)
            scaled_adj = np.clip(scaled_adj + noise * sub_adj, 0.0, 1.0)

        adj_t = torch.tensor(scaled_adj, dtype=torch.float).unsqueeze(0).requires_grad_(True)
        model.eval()
        ypred, _ = model(x_t, adj_t)
        probs = nn.Softmax(dim=0)(ypred[0, node_idx_new, :])
        score = probs[target_cls]
        score.backward()

        grad = adj_t.grad[0].detach().cpu().numpy()
        ig_accum += 0.5 * (grad + grad.T)

    return np.abs(ig_accum / float(steps)) * sub_adj


def evaluate_explainer_fn(explain_fn, base_explainer, cg_dict, target_nodes, prog_args):
    results = {
        "Clean": {"auc": [], "topk": []},
        "Attacked": {"auc": [], "topk": []},
        "ResiliX_C1": {"auc": [], "topk": []},
        "ResiliX_C1_C2": {"auc": [], "topk": []},
        "ResiliX_Full": {"auc": [], "topk": []},
    }
    for idx in target_nodes:
        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = base_explainer.extract_neighborhood(idx, 0)
        pred_label = np.argmax(cg_dict["pred"][0][neighbors], axis=1)

        c_mask = explain_fn(sub_adj, sub_feat, sub_label, pred_label, node_idx_new, False)
        c_auc, c_topk, c_sym = compute_motif_auc_and_topk(c_mask, node_idx_new)
        results["Clean"]["auc"].append(c_auc)
        results["Clean"]["topk"].append(c_topk)

        atk_adj = run_deduction_attack_subgraph(
            base_explainer.model, sub_adj, sub_feat, sub_label, pred_label, node_idx_new, c_sym, prog_args, p_budget=5
        )
        a_mask = explain_fn(atk_adj, sub_feat, sub_label, pred_label, node_idx_new, False)
        a_auc, a_topk, _ = compute_motif_auc_and_topk(a_mask, node_idx_new)
        results["Attacked"]["auc"].append(a_auc)
        results["Attacked"]["topk"].append(a_topk)

        pur_adj = resilix_c1_purify_subgraph(atk_adj, neighbors, node_idx_new)
        c1_mask = explain_fn(pur_adj, sub_feat, sub_label, pred_label, node_idx_new, False)
        c1_auc, c1_topk, _ = compute_motif_auc_and_topk(c1_mask, node_idx_new)
        results["ResiliX_C1"]["auc"].append(c1_auc)
        results["ResiliX_C1"]["topk"].append(c1_topk)

        c12_mask = explain_fn(pur_adj, sub_feat, sub_label, pred_label, node_idx_new, True)
        c12_auc, c12_topk, _ = compute_motif_auc_and_topk(c12_mask, node_idx_new)
        results["ResiliX_C1_C2"]["auc"].append(c12_auc)
        results["ResiliX_C1_C2"]["topk"].append(c12_topk)

        full_mask = resilix_c3_consensus_rerank(c12_mask, pur_adj, neighbors)
        f_auc, f_topk, _ = compute_motif_auc_and_topk(full_mask, node_idx_new)
        results["ResiliX_Full"]["auc"].append(f_auc)
        results["ResiliX_Full"]["topk"].append(f_topk)

    return results


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    prog_args = explainer_main.arg_parse()
    prog_args.dataset = "syn1"
    prog_args.gpu = False
    prog_args.num_epochs = 100
    prog_args.lr = 0.1
    prog_args.opt = "adam"
    prog_args.opt_scheduler = "none"
    prog_args.mask_act = "sigmoid"

    ckpt = io_utils.load_ckpt(prog_args)
    cg_dict = ckpt["cg"]
    input_dim = cg_dict["feat"].shape[2]
    num_classes = cg_dict["pred"].shape[2]

    model = models.GcnEncoderNode(
        input_dim=input_dim,
        hidden_dim=prog_args.hidden_dim,
        embedding_dim=prog_args.output_dim,
        label_dim=num_classes,
        num_layers=prog_args.num_gc_layers,
        bn=prog_args.bn,
        args=prog_args,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    base_explainer = exp_mod.Explainer(
        model=model,
        adj=cg_dict["adj"],
        feat=cg_dict["feat"],
        label=cg_dict["label"],
        pred=cg_dict["pred"],
        train_idx=cg_dict["train_idx"],
        args=prog_args,
        writer=None,
        print_training=False,
        graph_mode=False,
        graph_idx=0,
    )

    target_nodes = list(range(2000, 2075, 5))

    explainers = {
        "GNNExplainer": lambda adj, feat, lbl, plbl, nidx, sm: run_gnnexplainer_single(
            model, adj, feat, lbl, plbl, nidx, prog_args, in_loop_smoothing=sm
        ),
        "PGExplainer": lambda adj, feat, lbl, plbl, nidx, sm: run_pgexplainer_parametric(
            model, adj, feat, plbl, nidx, epochs=60, smoothing=sm
        ),
        "GSAT": lambda adj, feat, lbl, plbl, nidx, sm: run_gsat_explainer(
            model, adj, feat, plbl, nidx, epochs=60, smoothing=sm
        ),
        "Grad (Saliency)": lambda adj, feat, lbl, plbl, nidx, sm: run_grad_saliency(
            model, adj, feat, plbl, nidx, smoothing=sm
        ),
        "IntegratedGrad": lambda adj, feat, lbl, plbl, nidx, sm: run_integrated_gradients(
            model, adj, feat, plbl, nidx, steps=10, smoothing=sm
        ),
    }

    print("\n" + "=" * 95)
    print(f"5-EXPLAINER RESILIX BENCHMARK ON SYN1 (N={len(target_nodes)} House Motifs, Budget=5)")
    print("=" * 95)
    print(f"{'Explainer':<18} | {'Clean AUC (Top6)':<18} | {'Attacked AUC (Top6)':<20} | {'ResiliX Full AUC (Top6)':<24}")
    print("-" * 95)

    all_summary = {}
    for exp_name, exp_fn in explainers.items():
        res = evaluate_explainer_fn(exp_fn, base_explainer, cg_dict, target_nodes, prog_args)
        all_summary[exp_name] = res
        c_auc, c_tk = np.mean(res["Clean"]["auc"]), np.mean(res["Clean"]["topk"]) * 100
        a_auc, a_tk = np.mean(res["Attacked"]["auc"]), np.mean(res["Attacked"]["topk"]) * 100
        f_auc, f_tk = np.mean(res["ResiliX_Full"]["auc"]), np.mean(res["ResiliX_Full"]["topk"]) * 100
        print(f"{exp_name:<18} | {c_auc:.4f} ({c_tk:5.1f}%)   | {a_auc:.4f} ({a_tk:5.1f}%)     | {f_auc:.4f} ({f_tk:5.1f}%)")
    print("=" * 95)

    print("\nDETAILED ABLATION BREAKDOWN BY EXPLAINER (Mean ROC-AUC / Top-6 Precision):")
    for exp_name, res in all_summary.items():
        print(f"\n--- {exp_name} ---")
        for stage in ["Clean", "Attacked", "ResiliX_C1", "ResiliX_C1_C2", "ResiliX_Full"]:
            m_auc = np.mean(res[stage]["auc"])
            m_tk = np.mean(res[stage]["topk"]) * 100
            print(f"  {stage:<16}: AUC = {m_auc:.4f} | Top-6 Precision = {m_tk:.2f}%")

if __name__ == "__main__":
    main()
