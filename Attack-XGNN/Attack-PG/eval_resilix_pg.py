# -*- coding: utf-8 -*-
import os
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as ptgeom
import networkx as nx

from ExplanationEvaluation.models.GNN_paper import NodeGCN
from ExplanationEvaluation.explainers.PGExplainer import PGExplainer

SAVE_ROOT = r"C:\habib dissertation\Projects\ResiliX\saved_attacks\PGExplainer"


class ResiliXDefenseV2:
    """
    ResiliX V2:
      1. Cycle-Basis & 2-Core Topology Filter (protects cycles of ANY length: 3-cycles, 6-cycles, houses, rings)
      2. Message-Passing Level Stochastic Smoothing (passes smoothed edge_weights into GCN embedding layer)
      3. Connected-Component Anchor Pruning (penalizes disconnected decoy islands)
    """
    def __init__(self, k_views=15, feat_noise=0.02, bridge_attenuation=0.35, cohesion_bonus=1.35):
        self.k_views = k_views
        self.feat_noise = feat_noise
        self.bridge_attenuation = bridge_attenuation
        self.cohesion_bonus = cohesion_bonus

    @torch.no_grad()
    def compute_topology_trust(self, subg_edges: torch.Tensor, target_node: int):
        """
        Identifies edges that belong to cycles/biconnected motifs vs. loose bridge/decoy edges
        and computes distance from target_node.
        """
        edges_np = subg_edges.cpu().numpy().T
        num_edges = len(edges_np)
        G = nx.Graph()
        for u, v in edges_np:
            if u != v:
                G.add_edge(int(u), int(v))

        # 1. Find all edges participating in ANY cycle (triangles, 6-cycles, house motifs)
        cycle_edges = set()
        try:
            for cyc in nx.cycle_basis(G):
                L = len(cyc)
                for i in range(L):
                    u, v = cyc[i], cyc[(i + 1) % L]
                    cycle_edges.add((min(u, v), max(u, v)))
        except Exception:
            pass

        # 2. Shortest-path hop distance from target_node
        try:
            hop_dist = nx.single_source_shortest_path_length(G, int(target_node))
        except Exception:
            hop_dist = {int(target_node): 0}

        is_cycle_edge = np.zeros(num_edges, dtype=np.float32)
        prox_score = np.zeros(num_edges, dtype=np.float32)

        for idx, (u, v) in enumerate(edges_np):
            pair = (min(int(u), int(v)), max(int(u), int(v)))
            if pair in cycle_edges:
                is_cycle_edge[idx] = 1.0
            d_u = hop_dist.get(int(u), 4)
            d_v = hop_dist.get(int(v), 4)
            min_d = min(d_u, d_v)
            # Edges closer to target_node (1-2 hops) are more trustworthy than far 3-hop peripheral edges
            prox_score[idx] = 1.0 / (1.0 + 0.25 * min_d)

        return torch.tensor(is_cycle_edge), torch.tensor(prox_score)

    @torch.no_grad()
    def explain_defended_node(self, model, explainer, features, full_graphs, target_node, k, mode="full"):
        target_node = int(target_node)
        subg_nodes, subg, _, hard_edge_mask = ptgeom.utils.k_hop_subgraph(
            target_node, 3, full_graphs, relabel_nodes=False
        )
        subg_np = subg.cpu().numpy()
        num_sub_edges = subg_np.shape[1]

        # Map each directed edge in subg to its reverse directed edge
        edge_dict = {(int(subg_np[0, i]), int(subg_np[1, i])): i for i in range(num_sub_edges)}
        rev_map = np.array([edge_dict.get((int(subg_np[1, i]), int(subg_np[0, i])), i) for i in range(num_sub_edges)])

        is_cycle_edge, prox_score = self.compute_topology_trust(subg, target_node)

        # Base edge trust weight on the local subgraph:
        # Cycle edges get high keep weight (~0.95-1.0); non-cycle bridge edges get attenuated
        base_keep = torch.where(
            is_cycle_edge > 0.5,
            torch.full_like(is_cycle_edge, 0.98),
            torch.full_like(is_cycle_edge, self.bridge_attenuation)
        )

        accum_bi_mask = np.zeros(num_sub_edges, dtype=np.float32)
        views = 1 if mode == "cohesion_only" else self.k_views

        # Pre-create full-graph edge weights tensor
        num_full_edges = full_graphs.size(1)

        for v_idx in range(views):
            if mode == "cohesion_only":
                embeds = model.embedding(features, full_graphs).detach()
            else:
                # Sample stochastic edge weights on the 3-hop subgraph
                # Cycle edges stay near 1.0; non-cycle bridge edges fluctuate in [0.1, 0.8]
                rand_u = torch.rand(num_sub_edges)
                sym_rand = 0.5 * (rand_u + rand_u[rev_map])
                sub_weights = torch.where(
                    is_cycle_edge > 0.5,
                    0.90 + 0.10 * sym_rand,
                    self.bridge_attenuation * (0.3 + 1.4 * sym_rand)
                )

                full_edge_weights = torch.ones(num_full_edges, dtype=torch.float32)
                full_edge_weights[hard_edge_mask] = sub_weights

                noisy_feats = features + torch.randn_like(features) * self.feat_noise
                # CRITICAL FIX: Run GCN message passing WITH smoothed edge weights!
                embeds = model.embedding(noisy_feats, full_graphs, edge_weights=full_edge_weights).detach()

            input_expl = explainer._create_explainer_input(subg, embeds, target_node).unsqueeze(0)
            sampling_weights = explainer.explainer_model(input_expl)
            raw_mask = explainer._sample_graph(sampling_weights, training=False).squeeze().detach().cpu().numpy()

            bi_mask = np.zeros(num_sub_edges, dtype=np.float32)
            for i in range(num_sub_edges):
                if subg_np[0, i] <= subg_np[1, i]:
                    bi_mask[i] = raw_mask[i] + raw_mask[rev_map[i]]
                else:
                    bi_mask[i] = 0.0

            accum_bi_mask += bi_mask / views

        consensus_score = accum_bi_mask.copy()

        # Stage 1 & 3 Post-Consensus Structural + Connectedness Refinement
        if mode in ["full", "cohesion_only"]:
            # Boost edges that belong to a cycle and are proximate to target_node
            cycle_np = is_cycle_edge.numpy()
            prox_np = prox_score.numpy()
            for i in range(num_sub_edges):
                if subg_np[0, i] <= subg_np[1, i]:
                    if cycle_np[i] > 0.5:
                        consensus_score[i] *= self.cohesion_bonus
                    consensus_score[i] *= (0.85 + 0.15 * prox_np[i])

            # Enforce connected motif around highest-scoring core edges
            cand_k = min(num_sub_edges, k * 3)
            cand_indices = np.argsort(consensus_score)[-cand_k:]

            # Build graph of top-(k+2) candidate edges
            G_cand = nx.Graph()
            for idx in np.argsort(consensus_score)[-(k + 2):]:
                u, v = int(subg_np[0, idx]), int(subg_np[1, idx])
                if u <= v and consensus_score[idx] > 0:
                    G_cand.add_edge(u, v, score=float(consensus_score[idx]), idx=idx)

            if G_cand.number_of_nodes() > 0:
                # Pick connected component with highest total edge score
                best_cc = max(
                    nx.connected_components(G_cand),
                    key=lambda cc: sum(d["score"] for _, _, d in G_cand.subgraph(cc).edges(data=True))
                )
                for idx in cand_indices:
                    u, v = int(subg_np[0, idx]), int(subg_np[1, idx])
                    if u not in best_cc and v not in best_cc:
                        consensus_score[idx] *= 0.65

        top_k_indices = np.flip(np.argsort(consensus_score)[-k:])
        defended_pairs = set()
        for idx in top_k_indices:
            u, v = int(subg_np[0, idx]), int(subg_np[1, idx])
            defended_pairs.add((min(u, v), max(u, v)))

        return defended_pairs


def evaluate_saved_folder(dataset="syn3", attack="deduction"):
    folder = os.path.join(SAVE_ROOT, dataset, attack)
    files = sorted(glob.glob(os.path.join(folder, "case_*.pt")))
    if not files:
        print(f"No saved files found in {folder}")
        return

    if dataset == "syn1":
        model = NodeGCN(10, 4)
    elif dataset == "syn2":
        model = NodeGCN(10, 8)
    elif dataset == "syn3":
        model = NodeGCN(10, 2)

    ckpt = torch.load(f"./checkpoints/GNN/{dataset}/best_model", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    defense = ResiliXDefenseV2(k_views=15, feat_noise=0.02, bridge_attenuation=0.35, cohesion_bonus=1.35)

    undef_fools, def_fools, clean_def_fools = [], [], []
    t0 = time.time()

    for fpath in files:
        data = torch.load(fpath, map_location="cpu")
        indx = data["indx"]
        k = data["k"]
        features = data["features"]
        clean_graphs = data["clean_graphs"]
        attacked_graphs = data["attacked_graphs"]
        clean_Es = set((min(u, v), max(u, v)) for u, v in data["clean_Es_pairs"])

        explainer = PGExplainer(model, attacked_graphs, features, "node")
        explainer.explainer_model = torch.nn.Sequential(
            torch.nn.Linear(explainer.expl_embedding, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        explainer.explainer_model.load_state_dict(data["explainer_state_dict"])
        explainer.explainer_model.eval()

        undef_fools.append(data["undefended_fool_count"] / k)

        def_pairs = defense.explain_defended_node(model, explainer, features, attacked_graphs, indx, k, mode="full")
        def_fools.append(len(clean_Es - def_pairs) / k)

        clean_def_pairs = defense.explain_defended_node(model, explainer, features, clean_graphs, indx, k, mode="full")
        clean_def_fools.append(len(clean_Es - clean_def_pairs) / k)

    elapsed = time.time() - t0
    print(f"\n==================== RESILIX V2 DEFENSE RESULTS ({dataset} | {attack}) ====================")
    print(f"Evaluated {len(files)} cases in {elapsed:.2f} seconds ({elapsed/len(files)*1000:.1f} ms/case)")
    print(f"  1. Clean Graph + ResiliX Misalignment (Sanity Check): {np.mean(clean_def_fools)*100:.2f}% (Target: <10%)")
    print(f"  2. Attacked Graph (Undefended Baseline):              {np.mean(undef_fools)*100:.2f}%")
    print(f"  3. Attacked Graph + ResiliX Defense:                  {np.mean(def_fools)*100:.2f}%")
    print(f"  >>> ABSOLUTE IMPROVEMENT:                             -{(np.mean(undef_fools)-np.mean(def_fools))*100:.2f} percentage points! <<<")
    print(f"===========================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="syn3")
    parser.add_argument("--attack", type=str, default="deduction")
    args = parser.parse_args()
    evaluate_saved_folder(args.dataset, args.attack)
