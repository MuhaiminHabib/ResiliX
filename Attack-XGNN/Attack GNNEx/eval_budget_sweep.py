import time
import numpy as np
import torch

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
from eval_multi_explainers import (
    run_pgexplainer_parametric,
    run_gsat_explainer,
    run_grad_saliency,
    run_integrated_gradients,
)


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

    # 10 motif anchor nodes for a clean, reliable budget sweep
    target_nodes = list(range(2000, 2050, 5))
    budgets = [1, 2, 3, 4, 5]

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

    print("\n" + "=" * 105)
    print(f"PERTURBATION BUDGET SWEEP (p_budget = 1 to 5) ON SYN1 (N={len(target_nodes)} Motifs)")
    print("=" * 105)
    header = f"{'Explainer':<16} |" + "".join([f" B={b} (Atk / Res)  |" for b in budgets])
    print(header)
    print("-" * 105)

    for exp_name, exp_fn in explainers.items():
        # Precompute clean explanations once per node to avoid redundant overhead
        clean_cache = {}
        for idx in target_nodes:
            node_idx_new, sub_adj, sub_feat, sub_label, neighbors = base_explainer.extract_neighborhood(idx, 0)
            pred_label = np.argmax(cg_dict["pred"][0][neighbors], axis=1)
            c_mask = exp_fn(sub_adj, sub_feat, sub_label, pred_label, node_idx_new, False)
            _, _, c_sym = compute_motif_auc_and_topk(c_mask, node_idx_new)
            clean_cache[idx] = (node_idx_new, sub_adj, sub_feat, sub_label, neighbors, pred_label, c_sym)

        row_str = f"{exp_name:<16} |"
        for b in budgets:
            atk_aucs = []
            res_aucs = []
            for idx in target_nodes:
                node_idx_new, sub_adj, sub_feat, sub_label, neighbors, pred_label, c_sym = clean_cache[idx]

                # Perturb with budget b
                atk_adj = run_deduction_attack_subgraph(
                    model, sub_adj, sub_feat, sub_label, pred_label, node_idx_new, c_sym, prog_args, p_budget=b
                )
                a_mask = exp_fn(atk_adj, sub_feat, sub_label, pred_label, node_idx_new, False)
                a_auc, _, _ = compute_motif_auc_and_topk(a_mask, node_idx_new)
                atk_aucs.append(a_auc)

                # ResiliX Defense
                pur_adj = resilix_c1_purify_subgraph(atk_adj, neighbors, node_idx_new)
                c12_mask = exp_fn(pur_adj, sub_feat, sub_label, pred_label, node_idx_new, True)
                full_mask = resilix_c3_consensus_rerank(c12_mask, pur_adj, neighbors)
                f_auc, _, _ = compute_motif_auc_and_topk(full_mask, node_idx_new)
                res_aucs.append(f_auc)

            mean_atk = np.mean(atk_aucs)
            mean_res = np.mean(res_aucs)
            row_str += f" {mean_atk:.2f} / {mean_res:.2f}     |"

        print(row_str)

    print("=" * 105)

if __name__ == "__main__":
    main()
