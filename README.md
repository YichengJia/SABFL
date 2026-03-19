# Staleness-Aware Asynchronous Federated Learning for Heterogeneous Robotic Systems

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Research-yellow)](#performance-highlights)

A unified research codebase for **asynchronous federated learning (FL)** under:
- client heterogeneity (non-IID, dynamic participation),
- unstable networks (stale updates),
- communication constraints (compressed updates).

The core method is an **Improved Async Protocol** with:
1. staleness-aware aggregation,
2. adaptive buffering,
3. pluggable compression.
4. a lightweight **LLM-style Transformer backbone** (replacing MLP/CNN-only assumptions).

---

## Table of Contents
- [1. Highlights](#1-highlights)
- [2. Problem Setting](#2-problem-setting)
- [3. Core Method](#3-core-method)
- [4. Repository Structure](#4-repository-structure)
- [5. Installation](#5-installation)
- [6. Quick Start](#6-quick-start)
- [7. Journal-Oriented Evaluation](#7-journal-oriented-evaluation)
- [8. Reproducibility Notes](#8-reproducibility-notes)
- [9. Citation](#9-citation)

---

## 1. Highlights

- **Asynchronous FL protocols in one framework**: FedAvg, FedAsync, FedBuff, SCAFFOLD, Improved Async.
- **Configurable staleness decay** for improved async: `linear`, `quadratic`, `exp`.
- **Compression options**: Top-K, SignSGD, QSGD.
- **Budget-aware evaluation**:
  - communication budget (`comm_budget_mb`),
  - latency budget (`latency_budget_sec`),
  - tri-objective score (Accuracy–Communication–Latency),
  - feasibility flag (`within_budget`).

---

## 2. Problem Setting

We target cross-device FL with:
- heterogeneous clients,
- asynchronous update arrivals,
- stale updates (server receives updates based on older model versions),
- communication-constrained uplink.

The code supports reporting results under explicit constrained objectives:
- maximize accuracy under communication/latency budgets,
- or rank by a tri-objective composite score.

---

## 3. Core Method

### Improved Async (main algorithm)
- staleness-aware update weighting,
- adaptive buffer trigger based on network health,
- optional scale-aware parameter defaults (`auto_scale_params`),
- configurable staleness decay (`staleness_mode`, `staleness_floor`),
- observed-system adaptive staleness cap (`quantile` of recent staleness history + hard bounds),
- momentum smoothing + gradient clipping on server update.

### Minimal theory (paper-ready framing)
- Staleness-aware weighting:  
  `w_i = |D_i| * alpha(tau_i) * q_i`, where `alpha(tau)` is linear/quadratic/exp decay.
- Adaptive buffering:  
  server chooses aggregation threshold from `[min_buffer_size, max_buffer_size]` based on estimated network health.
- Bounded scale-aware parameterization:  
  `min_buffer_size = clip(ceil(sqrt(n)), 2, 64)`,  
  `max_buffer_size = clip(2 * min_buffer_size, min_buffer_size + 1, 256)`.
- System-observed staleness bound:  
  `effective_max_staleness = clip(Q_q(staleness_history), min_bound, max_bound)` (with health-aware relaxation).
- Compression tradeoff:  
  compare `(Intent-F1, BLEU)` vs communication cost for Top-K / SignSGD / QSGD.
- Pareto analysis:  
  each run is evaluated in the 3D objective space `(accuracy, communication, latency)` and marked as Pareto-optimal/non-optimal.

### Model backbone
- `SimpleNN` is a compatibility entry point with configurable backbones.
- Default backbone: `resnet18` (reproducible and widely used).
- Additional supported backbones:
  - `mobilenet_v3_small` (lighter/faster option for smaller budgets),
  - `tiny_transformer` (legacy patch-transformer path for ablations/continuity).
- Supports both vector input (synthetic baseline) and image input (e.g., TartanAir/CIFAR).

### TartanAir task protocol (updated)
- `generate_federated_data(..., dataset_name="tartanair", label_mode="turn_intent")` now supports:
  - `pose_turn` / `turn_intent`: derive turn-intent labels from trajectory pose files when available,
  - automatic fallback to temporal motion proxy labels when pose files are unavailable in the local release.
- This separates protocol robustness benchmarking from a pure brightness-threshold labeling setup.

### Compression
- Top-K sparsification,
- SignSGD 1-bit sign packing with magnitude scaling,
- QSGD quantization.

---

## 4. Repository Structure
```text
.
├── federated_protocol_framework.py      # Protocol implementations + factory
├── compression_strategies.py            # TopK / SignSGD / QSGD
├── metrics.py                           # F1, BLEU, tri-objective, budget checks
├── unified_protocol_comparison.py       # Main protocol comparison runner
├── intelligent_parameter_tuning.py      # Grid-style tuner (tri-objective ranking)
├── joint_protocol_topk_study.py         # Protocol × Top-K study
├── optimize_improved_async.py           # Improved-async optimization helper
├── optimized_protocol_config.py         # Scenario configs + scale sweep helpers
├── paper_profiles.py                    # Paper-oriented strict profiles + notes
├── generate_reproduction_report.py      # Auto-generate supplementary reproducibility report
└── readme.md
```

---

## 5. Installation

```
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy scikit-learn matplotlib
```
---

## 6. Quick Start

### 6.1 Main comparison
```
python unified_protocol_comparison.py
```
By default, this script now runs on `tartanair-test-mono-release/mono` if available.

### 6.2 Parameter tuning
```
python intelligent_parameter_tuning.py
```

### 6.3 Protocol × Top-K analysis
```
python joint_protocol_topk_study.py
```

### 6.4 Export strict profile + reproducibility report
```
python generate_reproduction_report.py
```
Outputs:
- `paper_profile_suite.json`
- `reproducibility_report.md`

### 6.5 Run ablation study (ImprovedAsync)
```
python ablation_runner.py
```
Outputs:
- `ablation_results.csv`
- `ablation_results.json`

### 6.6 Build ablation figures + LaTeX table
```
python build_ablation_artifacts.py
```
Outputs:
- `figures/ablation_tri_objective.png`
- `figures/ablation_pareto_accuracy_comm.png`
- `tables/ablation_table.tex`

### 6.7 External validity benchmark (CIFAR photometric non-IID)
```
python external_validity_runner.py --num_clients 20 --samples_per_client 250 --train_size 20000 --test_size 3000 --rounds 30 --local_epochs 2 --local_lr 0.001 --seeds 3 --fairness_modes equal_rounds,equal_updates --topk_fraction 0.03 --dominant_ratio 0.7 --gamma_min 0.75 --gamma_max 1.30 --brightness_min 0.70 --brightness_max 1.30
```
Outputs:
- `results/external_validity_results.csv`
- `results/external_validity_results.json`
- `results/external_validity_summary.csv`
- `results/external_validity_summary.json`
- `results/external_validity_traces.json` (convergence checkpoints)

Optional convergence controls:
- `--track_interval_updates` (default: 20)
- `--acc_thresholds` (default: `0.20,0.25`)

### 6.8 Build external-validity figures + LaTeX table
```
python external_validity_artifacts.py
```
Outputs:
- `figures/external_balanced_score.png`
- `figures/external_accuracy_mean.png`
- `figures/external_convergence_traces.png`
- `tables/external_validity_summary.tex`

### 6.9 Robust large-scale benchmark (anti-cherry-picking)
This benchmark evaluates a profile bank per protocol and reports:
- mean-over-profiles
- best-over-profiles (per seed, then averaged)

```
python robust_external_benchmark.py --num_clients_list 20,200,1000 --samples_per_client 250 --train_size 20000 --test_size 3000 --rounds 30 --seeds 3 --fairness_modes equal_rounds,equal_updates --topk_fractions 0.01,0.03,0.05
```

Outputs:
- `results/robust_external_raw.csv`
- `results/robust_external_raw.json`
- `results/robust_external_summary.csv`
- `results/robust_external_summary.json`

HPC script:
- `run_robust_external_benchmark.sbatch`

---

## 7. Journal-Oriented Evaluation
In experiment_config (main comparison runner), set:
- comm_budget_mb
- latency_budget_sec
- include_autoscale_variants (optional)

The runner outputs:  
- final_accuracy
- total_data_transmitted_mb
- elapsed_sec
- tri_objective_score
- within_budget

Improved async ablation knobs
- staleness_mode: linear | quadratic | exp
- staleness_floor: lower bound of staleness decay
- auto_scale_params: scale-aware defaults for large/small client counts

Scale-sweep config helper
Use generate_scale_sweep_configs(...) in optimized_protocol_config.py for experiments like:
n = [10, 50, 200, 500, 1000, 10000].

---

## 8. Reproducibility Notes
- Set random seeds for Python/NumPy/PyTorch.
- Keep training budgets comparable across protocols.
- Report both raw metrics and tri-objective score.
- Include budget feasibility (within_budget) in summary tables.

## 9. Citation
If you use this code in research, please cite your thesis/paper version and include commit hash/version for reproducibility.
