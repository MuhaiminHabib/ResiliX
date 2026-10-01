# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch_geometric as ptgeom

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset
from ExplanationEvaluation.models.GNN_paper import NodeGCN, GraphGCN
from ExplanationEvaluation.utils.graph import index_edge
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer
from ExplanationEvaluation.explainers.PGAttacker import Deduction_PGAttacker, Loss_PGAttacker
from ratio_test import likelyhood

DATASET_CONFIGS = {
    "syn1":          {"task": "node",  "k": 6,  "xi": 5,  "name": "BA-House"},
    "syn3":          {"task": "node",  "k": 6,  "xi": 2,  "name": "Tree-Cycle"},
    "mutag":         {"task": "graph", "k": 5,  "xi": 2,  "name": "MUTAG"},
    "REDDIT-BINARY": {"task": "graph", "k": 10, "xi": 2,  "name": "Reddit-Binary"},
    "syn2":          {"task": "node",  "k": 28, "xi": 10, "name": "BA-Community"},
}

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"
CSV_PATH = r"C:\habib dissertation\Projects\ResiliX\results_tables\pg_baseline_reproduction.csv"


def get_label_scalar(lbl):
    if lbl.numel() > 1:
        return int(torch.argmax(lbl).item())
    return int(lbl.item())


def run_node_dataset(dataset, attack, max_cases=40):
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

    ckpt = torch.load(f"./checkpoints/GNN/{dataset}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        preds = torch.argmax(model(features, graphs), dim=-1)
        lbl_idx = torch.argmax(labels, dim=-1) if labels.dim() > 1 else labels
        gcn_acc = (preds == lbl_idx).float().mean().item() * 100.0

    print(f"\n============================================================")
    print(f"[NODE] {dataset} ({cfg['name']}) | GCN Acc: {gcn_acc:.2f}% | Attack: {attack} | Cases: {len(test_indices)}")
    print(f"============================================================")

    beta, gamma = 0.7, 0.7
    fools, times = [], []
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

        attacker = Deduction_PGAttacker(model, graphs, features, task, beta=beta) if attack == "deduction" else Loss_PGAttacker(model, graphs, features, task, gamma=gamma)
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
        attacker = Deduction_PGAttacker(model, r_graphs, features, task, beta=beta) if attack == "deduction" else Loss_PGAttacker(model, r_graphs, features, task, gamma=gamma)

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

        # Exact author behavior: new_degree = degree (in-place reference across t)
        for t in range(xi + 1):
            deletion_rec = np.ones(expl.shape)
            addition_rec = np.ones(r_graph.shape[1])
            new_degree = degree
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
            fool = sum(1 for edge_idx in now_id if not np.isin([edge_idx], n_o_id))

            if fool >= maxfool:
                maxfool = fool
                best_new_graphs = new_graphs.clone()

        explainer.graphs = graphs
        fools.append(maxfool)
        elapsed = time.time() - t0
        times.append(elapsed)

        torch.save({
            "task": "node",
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

        print(f"  [{case_num+1}/{len(test_indices)}] Node {indx} | Misaligned: {maxfool}/{k} | Running Avg: {(np.mean(fools)/k)*100:.2f}% | {elapsed:.1f}s")

    final_ratio = (np.mean(fools) / k) * 100.0 if fools else 0.0
    return gcn_acc, final_ratio, np.mean(times) if times else 0.0, len(fools)


def run_graph_dataset(dataset, attack, max_cases=40):
    cfg = DATASET_CONFIGS[dataset]
    k, xi = cfg["k"], cfg["xi"]
    task = "graph"

    model = GraphGCN(14, 2) if dataset == "mutag" else GraphGCN(11, 2)
    o_graphs, o_features, labels, _, _, _ = load_dataset(dataset)
    labels = torch.tensor(labels)

    ckpt = torch.load(f"./checkpoints/GNN/{dataset}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    correct = 0
    total_eval = min(len(o_graphs), 100)
    with torch.no_grad():
        for i in range(total_eval):
            g_t = torch.tensor(np.array(o_graphs[i]))
            f_t = torch.tensor(np.array([o_features[i]]))
            pred = int(torch.argmax(model(f_t, g_t)).item())
            if pred == get_label_scalar(labels[i]):
                correct += 1
    gcn_acc = (correct / total_eval) * 100.0

    num_cases = min(40, max_cases) if max_cases else 40
    print(f"\n============================================================")
    print(f"[GRAPH] {dataset} ({cfg['name']}) | GCN Acc: {gcn_acc:.2f}% | Attack: {attack} | Cases: {num_cases}")
    print(f"============================================================")

    beta, gamma = 0.7, 0.7
    fools, times = [], []
    save_dir = os.path.join(SAVE_ROOT, dataset, attack)
    os.makedirs(save_dir, exist_ok=True)

    for indx in range(num_cases):
        t0 = time.time()
        maxfool = 0
        graphs = torch.tensor(np.array([o_graphs[indx]]))
        features = torch.tensor(np.array([o_features[indx]]))
        N = features.shape[1]

        explainer = PGExplainer(model, graphs, features, task, epochs=50)
        explainer.prepare([0])

        graph, expl = explainer.explain(0)
        if len(expl) < 4 * k:
            continue

        adj = np.zeros((N, N))
        mask = np.zeros((N, N))
        degree = np.zeros((N))

        for i in range(len(graph[0])):
            k1, k2 = int(graph[0, i]), int(graph[1, i])
            adj[k1][k2] = 1
            degree[k1] += 1
            mask[k1][k2] = expl[i]

        bi_expl = np.zeros(expl.shape)
        for i in range(len(graph[0])):
            k1, k2 = int(graph[0, i]), int(graph[1, i])
            if k1 <= k2:
                bi_expl[i] = (mask[k1][k2] + mask[k2][k1]) / 2.0
            else:
                bi_expl[i] = 0.0
        expl = torch.tensor(bi_expl)

        o_id = np.flip(np.argsort(np.array(expl.detach()).reshape(-1))[-k:])
        clean_Es_pairs = [(int(graph[0, j]), int(graph[1, j])) for j in o_id]

        o_bid = []
        for j in range(len(o_id)):
            o_bid.append(index_edge(graph, [graph[0, o_id[j]], graph[1, o_id[j]]]))
            if graph[1, o_id[j]] != graph[0, o_id[j]]:
                o_bid.append(index_edge(graph, [graph[1, o_id[j]], graph[0, o_id[j]]]))

        r_graphs = [[], []]
        for i in range(N):
            for j in range(i, N):
                if adj[i][j] == 0:
                    r_graphs[0].extend([i, j])
                    r_graphs[1].extend([j, i])
        for i in range(len(o_bid)):
            r_graphs[0].append(int(graph[0, o_bid[i]]))
            r_graphs[1].append(int(graph[1, o_bid[i]]))
        r_graphs = torch.tensor(np.array([r_graphs]))

        o_fliter = np.ones(expl.shape)
        o_bias = np.zeros(expl.shape)
        o_fliter[o_bid] = 0
        o_bias[o_bid] = 1

        attacker = Deduction_PGAttacker(model, graphs, features, task, beta=beta, N=4, r_epochs=10) if attack == "deduction" else Loss_PGAttacker(model, graphs, features, task, gamma=gamma, r_epochs=10)
        h_graph, hot_mask = attacker.learn_deletion([0], o_fliter, o_bias)
        h_graph = np.array(h_graph.detach())
        hot_mask = np.array(hot_mask.detach())

        add_mask = np.zeros((N, N))
        for i in range(hot_mask.shape[0]):
            add_mask[h_graph[0, i]][h_graph[1, i]] = hot_mask[i]
        for i in range(hot_mask.shape[0]):
            if h_graph[0, i] <= h_graph[1, i]:
                hot_mask[i] = add_mask[h_graph[0, i]][h_graph[1, i]] + add_mask[h_graph[1, i]][h_graph[0, i]]
            else:
                hot_mask[i] = 0
        hot_id = np.flip(np.argsort(hot_mask.reshape(-1)))

        r_graph = r_graphs[0].clone().detach()
        attacker = Deduction_PGAttacker(model, r_graphs, features, task, beta=beta, N=4, r_epochs=2) if attack == "deduction" else Loss_PGAttacker(model, r_graphs, features, task, gamma=gamma, r_epochs=10)

        r_o_fliter = np.ones(r_graph.shape[1])
        r_o_bias = np.zeros(r_graph.shape[1])
        r_o_bid = []
        for i in range(r_graph.shape[1]):
            for j in range(len(o_bid)):
                if r_graph[0, i] == graph[0, o_bid[j]] and r_graph[1, i] == graph[1, o_bid[j]]:
                    r_o_fliter[i] = 0
                    r_o_bias[i] = 1
                    r_o_bid.append(i)

        c_graph, cold_mask = attacker.learn_addition([0], r_o_fliter, r_o_bias)
        c_graph = np.array(c_graph.detach())
        cold_mask = np.array(cold_mask.detach())

        del_mask = np.zeros((N, N))
        for i in range(cold_mask.shape[0]):
            del_mask[c_graph[0, i]][c_graph[1, i]] = cold_mask[i]
        for i in range(cold_mask.shape[0]):
            if c_graph[0, i] <= c_graph[1, i]:
                cold_mask[i] = del_mask[c_graph[0, i]][c_graph[1, i]] + del_mask[c_graph[1, i]][c_graph[0, i]]
            else:
                cold_mask[i] = 0
        cold_id = np.flip(np.argsort(cold_mask.reshape(-1)))

        best_new_graphs = graphs.clone()

        for t in range(xi + 1):
            rec_d = np.ones(expl.shape)
            rec_a = np.ones(r_graph.shape[1])
            new_d = degree
            cnt2 = 0
            hots = []
            for i in range(len(hot_id)):
                a = hot_id[i]
                pair = np.array(graph.T[a])
                npair = np.array([pair[1], pair[0]])
                b = index_edge(graph, pair)
                if cnt2 >= t * 2:
                    break
                if np.isin([hot_id[i]], o_bid) or rec_d[hot_id[i]] == 0:
                    continue
                rec_d[a] = 0
                rec_d[b] = 0
                n1, n2 = int(pair[0]), int(pair[1])
                new_d[n1] -= 1
                if n1 != n2:
                    new_d[n2] -= 1
                a1 = index_edge(graphs[0], pair)
                b1 = index_edge(graphs[0], npair)
                hots.append(a1)
                if a1 != b1:
                    hots.append(b1)
                cnt2 += 2
            new_graphs = torch.tensor(np.delete(graphs[0].permute(1, 0).numpy(), hots, axis=0))

            for i in range(len(cold_id)):
                if cnt2 >= 2 * xi:
                    break
                if np.isin([cold_id[i]], r_o_bid) or rec_a[cold_id[i]] == 0:
                    continue
                a = cold_id[i]
                pair = r_graph.T[a].clone()
                new_graphs = torch.cat((new_graphs, torch.tensor([[pair[0], pair[1]]])))
                if pair[0] != pair[1]:
                    new_graphs = torch.cat((new_graphs, torch.tensor([[pair[1], pair[0]]])))
                rev_pair = [int(pair[1]), int(pair[0])]
                b = index_edge(r_graph, rev_pair)
                rec_a[a] = 0
                rec_a[b] = 0
                n1, n2 = int(pair[0]), int(pair[1])
                new_d[n1] += 1
                if n1 != n2:
                    new_d[n2] += 1
                cnt2 += 2

            new_graphs = torch.tensor(np.array([new_graphs.permute(1, 0).numpy()]))
            ls = likelyhood(degree, new_d)
            predict_before = np.argmax(np.array(model(features, graphs[0]).detach()))
            predict_after = np.argmax(np.array(model(features, new_graphs[0]).detach()))
            if ls > 0.000157 or predict_before != predict_after:
                continue

            explainer.graphs = new_graphs
            n_graph, n_expl = explainer.explain(0)
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
            for i in range(len(n_expl_np)):
                for j in range(len(o_id)):
                    if n_graph_np[0, i] == graph[0, o_id[j]] and n_graph_np[1, i] == graph[1, o_id[j]]:
                        n_o_id.append(i)

            now_id = np.flip(np.argsort(n_expl_np.reshape(-1))[-k:])
            fool = sum(1 for edge_idx in now_id if not np.isin([edge_idx], n_o_id))
            if fool >= maxfool:
                maxfool = fool
                best_new_graphs = new_graphs.clone()

        explainer.graphs = graphs
        fools.append(maxfool)
        elapsed = time.time() - t0
        times.append(elapsed)

        torch.save({
            "task": "graph",
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

        print(f"  [{indx+1}/{num_cases}] Graph {indx} | Misaligned: {maxfool}/{k} | Running Avg: {(np.mean(fools)/k)*100:.2f}% | {elapsed:.1f}s")

    final_ratio = (np.mean(fools) / k) * 100.0 if fools else 0.0
    return gcn_acc, final_ratio, np.mean(times) if times else 0.0, len(fools)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["syn1", "mutag", "REDDIT-BINARY"])
    parser.add_argument("--attacks", nargs="+", default=["loss", "deduction"])
    parser.add_argument("--max_cases", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        existing_df = existing_df[existing_df["Dataset"] == "syn3"]
        rows = existing_df.to_dict("records")
    else:
        rows = []

    for ds in args.datasets:
        for att in args.attacks:
            task = DATASET_CONFIGS[ds]["task"]
            if task == "node":
                acc, ratio, avg_t, n_valid = run_node_dataset(ds, att, args.max_cases)
            else:
                acc, ratio, avg_t, n_valid = run_graph_dataset(ds, att, args.max_cases)

            rows.append({
                "Explainer": "PGExplainer",
                "Dataset": ds,
                "Dataset_Name": DATASET_CONFIGS[ds]["name"],
                "Attack": att,
                "Valid_Cases": n_valid,
                "Base_GCN_Acc_Pct": round(acc, 2),
                "Misalignment_Ratio_Pct": round(ratio, 2),
                "Avg_Sec_Per_Case": round(avg_t, 2),
            })
            df = pd.DataFrame(rows)
            df.to_csv(CSV_PATH, index=False)
            print(f"\n[SAVED PROGRESS TO CSV]: {CSV_PATH}")
            print(df.to_string(index=False))
