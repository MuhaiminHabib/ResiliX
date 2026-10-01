# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import torch
import torch_geometric as ptgeom

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.utils.graph import index_edge
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.explainers.PGAttacker import Deduction_PGAttacker, Loss_PGAttacker
from ratio_test import likelyhood

DATASET_CONFIGS = {
    "syn1": {"task": "node", "k": 6,  "xi": 5,  "name": "BA-House"},
    "syn2": {"task": "node", "k": 28, "xi": 10, "name": "BA-Community"},
    "syn3": {"task": "node", "k": 6,  "xi": 2,  "name": "Tree-Cycle"},
    "mutag": {"task": "graph", "k": 5, "xi": 2, "name": "MUTAG"},
    "REDDIT-BINARY": {"task": "graph", "k": 10, "xi": 2, "name": "Reddit-Binary"},
}

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"


def run_node_repro(dataset, attack, max_cases=None):
    cfg = DATASET_CONFIGS[dataset]
    k, xi = cfg["k"], cfg["xi"]
    task = "node"

    if dataset == "syn1":
        model = NodeGCN(10, 4)
        test_indices = list(range(300, 700, 5))
    elif dataset == "syn2":
        model = NodeGCN(10, 8)
        test_indices = list(range(400, 560, 1))
    elif dataset == "syn3":
        model = NodeGCN(10, 2)
        test_indices = list(range(0, 320, 2))

    if max_cases is not None:
        test_indices = test_indices[:max_cases]

    graphs, features, labels, _, _, _ = load_dataset(dataset)
    graphs = torch.tensor(graphs)
    features = torch.tensor(features)
    labels = torch.tensor(labels)

    path = f"./checkpoints/GNN/{dataset}/best_model"
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Check clean GCN accuracy first
    with torch.no_grad():
        preds = torch.argmax(model(features, graphs), dim=-1)
        acc = (preds == labels).float().mean().item() * 100.0
    print(f"============================================================")
    print(f"Dataset: {dataset} ({cfg['name']}) | Base GCN Accuracy: {acc:.2f}%")
    print(f"Attack: {attack} | Cases: {len(test_indices)} | k={k}, xi={xi}")
    print(f"============================================================")

    beta, gamma = 0.7, 0.7
    fools = []
    save_dir = os.path.join(SAVE_ROOT, dataset, attack)
    os.makedirs(save_dir, exist_ok=True)

    for case_num, indx in enumerate(test_indices):
        t0 = time.time()
        maxfool = 0
        explainer = PGExplainer(model, graphs, features, task)
        explainer.prepare([indx])

        graph, expl = explainer.explain(indx)
        if len(expl) < k * 2:
            continue

        node = []
        maps = np.zeros((len(features)), dtype=np.int32)
        maps[graph[0, 0]] = 1
        node_index = np.sort(graph[0])
        node.append(node_index[0])
        for i in range(1, len(graph[0])):
            if node_index[i] != node_index[i - 1]:
                node.append(node_index[i])
                maps[node_index[i]] = len(node)

        adj = np.zeros((len(node), len(node)))
        mask = np.zeros((len(node), len(node)))
        degree = np.zeros((len(node)))

        for i in range(len(graph[0])):
            adj[maps[graph[0, i]] - 1][maps[graph[1, i]] - 1] = 1
            degree[maps[graph[0, i]] - 1] += 1
            mask[maps[graph[0, i]] - 1][maps[graph[1, i]] - 1] = expl[i]

        bi_expl = np.zeros(expl.shape)
        for i in range(len(graph[0])):
            if graph[0, i] <= graph[1, i]:
                bi_expl[i] = mask[maps[graph[0, i]] - 1][maps[graph[1, i]] - 1] + mask[maps[graph[1, i]] - 1][maps[graph[0, i]] - 1]
            else:
                bi_expl[i] = 0
        expl = torch.tensor(bi_expl)

        o_id = np.flip(np.argsort(np.array(expl.detach()).reshape(-1))[-k:])
        clean_Es_pairs = [(int(graph[0, j]), int(graph[1, j])) for j in o_id]

        o_bid = []
        for j in range(len(o_id)):
            o_bid.append(index_edge(graph, [graph[0, o_id[j]], graph[1, o_id[j]]]))
            o_bid.append(index_edge(graph, [graph[1, o_id[j]], graph[0, o_id[j]]]))

        best_new_graphs = graphs.clone()

        if attack in ["deduction", "loss"]:
            r_graphs = [[], []]
            if len(node) >= 1500:
                for i in range(len(node)):
                    if node[i] != indx:
                        r_graphs[0].extend([node[i], indx])
                        r_graphs[1].extend([indx, node[i]])
            else:
                for i in range(len(node)):
                    for j in range(i + 1, len(node)):
                        if adj[i][j] == 0 or node[i] == indx or node[j] == indx:
                            r_graphs[0].extend([node[i], node[j]])
                            r_graphs[1].extend([node[j], node[i]])

            for i in range(len(o_id)):
                r_graphs[0].extend([graph[0, o_id[i]], graph[1, o_id[i]]])
                r_graphs[1].extend([graph[1, o_id[i]], graph[0, o_id[i]]])

            r_graphs = torch.tensor(np.array(r_graphs))

            o_fliter = np.ones(expl.shape)
            o_bias = np.zeros(expl.shape)
            o_bid = []
            for j in range(len(o_id)):
                i1 = index_edge(graph, [graph[0, o_id[j]], graph[1, o_id[j]]])
                o_fliter[i1] = 0
                o_bias[i1] = 1
                o_bid.append(i1)
                i2 = index_edge(graph, [graph[1, o_id[j]], graph[0, o_id[j]]])
                o_fliter[i2] = 0
                o_bias[i2] = 1
                o_bid.append(i2)

            if attack == "deduction":
                attacker = Deduction_PGAttacker(model, graphs, features, task, beta=beta)
            else:
                attacker = Loss_PGAttacker(model, graphs, features, task, gamma=gamma)

            h_graph, hot_mask = attacker.learn_deletion([indx], o_fliter, o_bias)
            h_graph = np.array(h_graph.detach())
            hot_mask = np.array(hot_mask.detach())

            add_mask = np.zeros((len(node), len(node)))
            for i in range(hot_mask.shape[0]):
                add_mask[maps[h_graph[0, i]] - 1][maps[h_graph[1, i]] - 1] = hot_mask[i]
            for i in range(hot_mask.shape[0]):
                if h_graph[0, i] <= h_graph[1, i]:
                    hot_mask[i] = add_mask[maps[h_graph[0, i]] - 1][maps[h_graph[1, i]] - 1] + add_mask[maps[h_graph[1, i]] - 1][maps[h_graph[0, i]] - 1]
                else:
                    hot_mask[i] = 0
            hot_id = np.flip(np.argsort(hot_mask))

            r_graph = ptgeom.utils.k_hop_subgraph([indx], 3, r_graphs)[1]
            if attack == "deduction":
                attacker = Deduction_PGAttacker(model, r_graphs, features, task, beta=beta)
            else:
                attacker = Loss_PGAttacker(model, r_graphs, features, task, gamma=gamma)

            r_o_fliter = np.ones(r_graph.shape[1])
            r_o_bias = np.zeros(r_graph.shape[1])
            r_o_bid = []
            for j in range(len(o_id)):
                i1 = index_edge(r_graph, [graph[0, o_id[j]], graph[1, o_id[j]]])
                r_o_fliter[i1] = 0
                r_o_bias[i1] = 1
                r_o_bid.append(i1)
                i2 = index_edge(r_graph, [graph[1, o_id[j]], graph[0, o_id[j]]])
                r_o_fliter[i2] = 0
                r_o_bias[i2] = 1
                r_o_bid.append(i2)

            c_graph, cold_mask = attacker.learn_addition([indx], r_o_fliter, r_o_bias)
            c_graph = np.array(c_graph.detach())
            cold_mask = np.array(cold_mask.detach())
            for i in range(cold_mask.shape[0]):
                if i % 2 == 0:
                    cold_mask[i] = cold_mask[i] + cold_mask[i + 1]
                    if c_graph[0, i] > c_graph[1, i]:
                        cold_mask[i + 1] = cold_mask[i]
                        cold_mask[i] = 0
                    else:
                        cold_mask[i + 1] = 0
            cold_id = np.flip(np.argsort(cold_mask.reshape(-1)))

            for t in range(xi + 1):
                deletion_rec = np.ones(expl.shape)
                addition_rec = np.ones(r_graph.shape[1])
                new_degree = degree.copy()
                cnt2 = 0
                hots = []
                for i in range(len(hot_id)):
                    if cnt2 >= t * 2:
                        break
                    if np.isin([hot_id[i]], o_bid) or deletion_rec[hot_id[i]] == 0:
                        continue
                    a = hot_id[i]
                    pair = np.array(graph.T[a])
                    npair = np.array([pair[1], pair[0]])
                    b = index_edge(graph, npair)
                    deletion_rec[a] = 0
                    deletion_rec[b] = 0
                    n1 = np.where(np.array(node, dtype=np.int32) == int(pair[0]))[0][0]
                    n2 = np.where(np.array(node, dtype=np.int32) == int(pair[1]))[0][0]
                    new_degree[n1] -= 1
                    new_degree[n2] -= 1
                    hots.extend([index_edge(graphs, pair), index_edge(graphs, npair)])
                    cnt2 += 2
                new_graphs = torch.tensor(np.delete(graphs.permute(1, 0).numpy(), hots, axis=0))

                for i in range(len(cold_id)):
                    if cnt2 >= xi * 2:
                        break
                    if np.isin([cold_id[i]], r_o_bid) or addition_rec[cold_id[i]] == 0:
                        continue
                    a = cold_id[i]
                    pair = r_graph.T[a].clone()
                    n1 = np.where(np.array(node, dtype=np.int32) == int(pair[0]))[0][0]
                    n2 = np.where(np.array(node, dtype=np.int32) == int(pair[1]))[0][0]
                    if adj[n1, n2] == 1 or adj[n2, n1] == 1:
                        continue
                    new_graphs = torch.cat((new_graphs, torch.tensor([[pair[0], pair[1]], [pair[1], pair[0]]])))
                    rev_pair = [int(pair[1]), int(pair[0])]
                    b = index_edge(r_graph, rev_pair)
                    new_degree[n1] += 1
                    new_degree[n2] += 1
                    addition_rec[a] = 0
                    addition_rec[b] = 0
                    cnt2 += 2
                new_graphs = new_graphs.permute(1, 0)

                ls = likelyhood(degree, new_degree, 5)
                predict_before = np.argmax(np.array(model(features, graphs).detach()[indx]))
                predict_after = np.argmax(np.array(model(features, new_graphs).detach()[indx]))
                if ls > 0.000157 or predict_before != predict_after:
                    continue

                explainer.graphs = new_graphs
                n_graph, n_expl = explainer.explain(indx)
                n_graph_np = np.array(n_graph.detach())
                n_expl_np = np.array(n_expl.detach())

                bi_n_expl = n_expl_np.copy()
                for i in range(n_expl_np.shape[0]):
                    r_pair = [n_graph_np[1, i], n_graph_np[0, i]]
                    if n_graph_np[0, i] <= n_graph_np[1, i]:
                        n_expl_np[i] += bi_n_expl[index_edge(n_graph_np, r_pair)]
                    else:
                        n_expl_np[i] = 0

                n_o_id = []
                for i in range(n_expl_np.shape[0]):
                    for j in range(len(o_id)):
                        if n_graph_np[0, i] == graph[0, o_id[j]] and n_graph_np[1, i] == graph[1, o_id[j]]:
                            n_o_id.append(i)

                now_id = np.flip(np.argsort(n_expl_np.reshape(-1))[-k:])
                fool = 0
                for edge_idx in now_id:
                    if not np.isin([edge_idx], n_o_id):
                        fool += 1

                if fool >= maxfool:
                    maxfool = fool
                    best_new_graphs = new_graphs.clone()

            explainer.graphs = graphs
            fools.append(maxfool)

        torch.save({
            "dataset": dataset,
            "attack": attack,
            "indx": indx,
            "k": k,
            "xi": xi,
            "clean_graphs": graphs,
            "attacked_graphs": best_new_graphs,
            "features": features,
            "clean_Es_pairs": clean_Es_pairs,
            "undefended_fool_count": maxfool,
            "explainer_state_dict": explainer.explainer_model.state_dict(),
        }, os.path.join(save_dir, f"case_{indx}.pt"))

        misalign_pct = (maxfool / k) * 100.0
        running_pct = (np.mean(fools) / k) * 100.0
        print(f"  [{case_num+1}/{len(test_indices)}] Node {indx} | Misaligned: {maxfool}/{k} ({misalign_pct:.1f}%) | Running Avg: {running_pct:.2f}% | Time: {time.time()-t0:.1f}s")

    final_ratio = (np.mean(fools) / k) * 100.0 if fools else 0.0
    print(f"\n>>> FINISHED {dataset} ({attack}): Avg Misaligned = {np.mean(fools):.2f}/{k} | Misalignment Ratio = {final_ratio:.2f}% <<<\n")
    return final_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="syn3", choices=["syn1", "syn2", "syn3"])
    parser.add_argument("--attack", type=str, default="deduction", choices=["deduction", "loss"])
    parser.add_argument("--max_cases", type=int, default=20, help="Number of cases to run (use None or 160 for full)")
    args = parser.parse_args()

    run_node_repro(args.dataset, args.attack, args.max_cases)
