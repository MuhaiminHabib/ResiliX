# ResiliX: Robust Defense Framework Against Adversarial Attacks on Graph Neural Network (GNN) Explainers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%28%2B%29-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%25-Geometric-orange.svg)](https://pyg.org/)
[![Status: Dissertation Polish](https://img.shields.io/badge/Status-Dissertation%20Ready-success.svg)]()

**ResiliX** is a rigorous, multi-layered defense framework designed to protect Graph Neural Network (GNN) explainers against adversarial structural perturbations and feature-spoofing attacks. By safeguarding explanation fidelity, ResiliX ensures trustworthy AI audits in high-stakes domains such as **Anti-Money Laundering (AML)** and financial crime detection.

---

## 🏛️ System Architecture

ResiliX employs a robust **3-Layer Defense Pipeline**:

1. **C1 — Topological Purification:** Automatically filters out anomalous edge connections, high-degree transaction shortcuts, and adversarial injection links.
2. **C2 — In-Loop Stochastic Smoothing:** Injects controlled Gaussian noise into feature attribution passes to stabilize gradients and prevent gradient masking.
3. **C3 — Subgraph Consensus Reranking:** Enforces symmetry and evaluates multi-hop structural cluster consensus to guarantee reliable feature attribution.

---

## 📊 Summary of Empirical Results

ResiliX has been extensively benchmarked across controlled synthetic motifs and real-world compliance networks:

| Benchmark Dataset | Graph Type | Clean Explainer AUC | Attacked Explainer AUC (B=5) | ResiliX Full AUC (Ours) |
| :--- | :--- | :---: | :---: | :---: |
| **Syn1 (BA-Shapes)** | Synthetic Motif | 0.9994 | 0.4164 | **0.9990** |
| **Syn2 (BA-Community)** | Community Motif | 0.9850 | 0.4520 | **0.8850** |
| **Syn3 (Tree-Grids)** | Hierarchical Tree | 0.5181 | 0.5179 | **0.7375** |
| **Elliptic Bitcoin** | Crypto AML Transactions | 0.9482 | 0.5094 | **0.8975** |
| **IBM AMLSim (HI-Small)** | Account Transfers | 0.9607 | 0.5459 | **0.9216** |

---

## ⚡ Computational Performance & Latency

Evaluated on transaction sub-networks (N = 15 nodes):
* **Unprotected Baseline Explainer:** 0.01 ms per subgraph
* **ResiliX Full 3-Layer Defense (C1+C2+C3):** 0.03 ms per subgraph
* **Total Added Overhead:** **+0.03 ms** (Well below the 50 ms industry-viable threshold for real-time compliance monitoring).

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
\\\ash
git clone https://github.com/MuhaiminHabib/ResiliX.git
cd ResiliX
\\\

### 2. Set Up Virtual Environment & Dependencies
\\\ash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
\\\

### 3. Run Evaluations & Generate Figures
\\\ash
cd "Attack-XGNN\Attack GNNEx"
python generate_dissertation_plots.py
python eval_elliptic_resilix.py
python eval_ibm_aml_resilix.py
python eval_latency.py
\\\

---

## 📄 Citation

If you use ResiliX in your research or academic dissertation, please cite:
\\\ibtex
@phdthesis{habib2026resilix,
  title={ResiliX: Robust Defense Framework Against Adversarial Attacks on Graph Neural Network Explainers},
  author={Habib, Muhaimin},
  year={2026},
  school={Dissertation Project}
}
\\\

---

## 📜 License
Distributed under the MIT License. See \LICENSE\ for more information.
