import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import explainer_main
import models
import utils.io_utils as io_utils
from explainer import explain as exp_mod
from eval_resilix_gnnex import (
    run_deduction_attack_subgraph,
    resilix_c1_purify_subgraph,
    run_gnnexplainer_single,
    resilix_c3_consensus_rerank,
)
from eval_multi_explainers import (
    run_pgexplainer_parametric,
    run_gsat_explainer,
)


def compute_syn2_metrics(masked_adj, neighbors, start_idx, top_k=6):
    """
    Evaluates motif ROC-AUC and top-k precision for syn2 community motifs.
    """
    sym_mask = 0.5 * (masked_adj + masked_adj.T)
    np.fill_diagonal(sym_mask, 0.0)

    n_sub = sym_mask.shape[0]
    gt_matrix = np.zeros_like(sym_mask)
    for u in range(n_sub):
        for v in range(u + 1, n_sub):
            gu, gv = neighbors[u], neighbors[v]
            if abs(int(gu) - int(gv)) <= 4:
                gt_matrix[u, v] = 1.0
                gt_matrix[v, u] = 1.0

    triu_idx = np.triu_indices(n_sub, k=1)
    y_true = gt_matrix[triu_idx]
    y_score = sym_mask[triu_idx]

    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_score)

    flat_scores = sym_mask[triu_idx]
    flat_gt = gt_matrix[triu_idx]
    top_indices = np.argsort(flat_scores)[-top_k:]
    top_k_hits = np.sum(flat_gt[top_indices]) / float(top_k) if top_k > 0 else 0.0

    return auc, top_k_hits, sym_mask


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    prog_args = explainer_main.arg_parse()
    prog_args.dataset = "syn2"
    prog_args.gpu = False
    prog_args.num_epochs = 100
    prog_args.lr = 0.1
    prog_args.opt = "adam"
    prog_args.opt_scheduler = "none"
    prog_args.mask_act = "sigmoid"

    try:
        ckpt = io_utils.load_ckpt(prog_args)
        print("Loaded existing syn2 checkpoint successfully.")
    except Exception:
        print("Training syn2 base model for checkpoint generation...")
        import train
        train.train(prog_args)
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

    target_nodes = list(range(2500, 2550, 5))

    explainers = {
        "GNNExplainer": lambda adj, feat, lbl, plbl, nidx, sm: run_gnnexplainer_single(
            model, adj, feat, lbl, plbl, nidx, prog_args, in_loop_smoothing=sm
        ),
        "PGExplainer": lambda adj, feat, lbl, plbl, nidx, sm: run_pgexplainer_parametric(
            model, adj, feat, plbl, nidx, epochs=50, smoothing=sm
        ),
        "GSAT": lambda adj, feat, lbl, plbl, nidx, sm: run_gsat_explainer(
            model, adj, feat, plbl, nidx, epochs=50, smoothing=sm
        ),
    }

    print("\n" + "=" * 90)
    print(f"CROSS-DOMAIN RESILIX EVALUATION ON SYN2 (BA-Community, N={len(target_nodes)} Motifs, Budget=5)")
    print("=" * 90)
    print(f"{'Explainer':<16} | {'Clean AUC':<12} | {'Attacked AUC':<14} | {'ResiliX Full AUC':<16}")
    print("-" * 90)

    for exp_name, exp_fn in explainers.items():
        clean_aucs, atk_aucs, res_aucs = [], [], []
        for idx in target_nodes:
            try:
                node_idx_new, sub_adj, sub_feat, sub_label, neighbors = base_explainer.extract_neighborhood(idx, 0)
                pred_label = np.argmax(cg_dict["pred"][0][neighbors], axis=1)

                c_mask = exp_fn(sub_adj, sub_feat, sub_label, pred_label, node_idx_new, False)
                c_auc, _, c_sym = compute_syn2_metrics(c_mask, neighbors, node_idx_new)
                clean_aucs.append(c_auc)

                atk_adj = run_deduction_attack_subgraph(
                    model, sub_adj, sub_feat, sub_label, pred_label, node_idx_new, c_sym, prog_args, p_budget=5
                )
                a_mask = exp_fn(atk_adj, sub_feat, sub_label, pred_label, node_idx_new, False)
                a_auc, _, _ = compute_syn2_metrics(a_mask, neighbors, node_idx_new)
                atk_aucs.append(a_auc)

                pur_adj = resilix_c1_purify_subgraph(atk_adj, neighbors, node_idx_new)
                c12_mask = exp_fn(pur_adj, sub_feat, sub_label, pred_label, node_idx_new, True)
                full_mask = resilix_c3_consensus_rerank(c12_mask, pur_adj, neighbors)
                f_auc, _, _ = compute_syn2_metrics(full_mask, neighbors, node_idx_new)
                res_aucs.append(f_auc)
            except Exception:
                continue

        if len(clean_aucs) > 0:
            print(f"{exp_name:<16} | {np.mean(clean_aucs):.4f}       | {np.mean(atk_aucs):.4f}         | {np.mean(res_aucs):.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
