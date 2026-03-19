"""
robust_external_benchmark.py
Large-scale external benchmark with protocol profile banks to reduce
"single-parameter cherry-picking" concerns.
"""

import argparse
import csv
import json
import math
import hashlib
import sys
import platform
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from external_validity_runner import (
    _load_cifar10_subset,
    _build_non_iid_clients,
    ClientPhotometricDataset,
    run_once,
    _estimate_model_param_count,
)
from unified_protocol_comparison import set_seed


def _runtime_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"
    return {
        "timestamp_unix": float(time.time()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "git_commit": git_commit,
        "baseline_fidelity_notes": {
            "scaffold": "Canonical SCAFFOLD: client-side gradient correction (c, c_i) and server control-variate aggregation."
        },
        "args": vars(args),
    }


def _save_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _mean_std_ci(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if len(arr) <= 1:
        return mean, std, 0.0
    ci95 = float(1.96 * std / math.sqrt(len(arr)))
    return mean, std, ci95


def _sign_test_pvalue_two_sided(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value (ties excluded)."""
    n = int(wins + losses)
    if n <= 0:
        return 1.0
    k = int(min(wins, losses))
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def _build_best_seed_rows(
    raw_rows: List[Dict[str, Any]],
    n_clients: int,
    mode: str,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    For each protocol_family and seed, pick the profile with best balanced score.
    This supports fair paired seed-wise significance tests.
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for proto in ("fedavg", "fedasync", "fedbuff", "scaffold", "improved_async"):
        rows = [
            r for r in raw_rows
            if r["num_clients"] == n_clients
            and r["fairness_mode"] == mode
            and r["protocol_family"] == proto
        ]
        if not rows:
            continue
        by_seed: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_seed.setdefault(int(r["seed"]), []).append(r)
        out[proto] = {}
        for seed, rs in by_seed.items():
            out[proto][seed] = max(rs, key=lambda x: float(x["score_balanced"]))
    return out


def _apply_holm_correction(rows: List[Dict[str, Any]]) -> None:
    """
    In-place Holm correction by (num_clients, fairness_mode, metric) family.
    """
    grouped: Dict[Tuple[int, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (int(r["num_clients"]), str(r["fairness_mode"]), str(r["metric"]))
        grouped.setdefault(key, []).append(r)

    for _, group in grouped.items():
        ordered = sorted(group, key=lambda x: float(x["p_value_sign"]))
        m = len(ordered)
        prev = 0.0
        for i, row in enumerate(ordered):
            raw_p = float(row["p_value_sign"])
            adj = min(1.0, (m - i) * raw_p)
            adj = max(prev, adj)
            row["p_value_holm"] = float(adj)
            prev = adj


def _build_pairwise_rows(
    raw_rows: List[Dict[str, Any]],
    n_list: List[int],
    fairness_modes: List[str],
) -> List[Dict[str, Any]]:
    """
    Paired seed-wise tests: improved_async (best profile per seed) vs each baseline family.
    """
    rows: List[Dict[str, Any]] = []
    baselines = ("fedavg", "fedasync", "fedbuff", "scaffold")

    metric_specs = (
        # transformed diff should be positive when improved_async is better
        ("score_balanced", lambda imp, base: float(imp) - float(base)),
        ("accuracy", lambda imp, base: float(imp) - float(base)),
        ("communication_mb", lambda imp, base: float(base) - float(imp)),
    )

    for n_clients in n_list:
        for mode in fairness_modes:
            best = _build_best_seed_rows(raw_rows, n_clients=n_clients, mode=mode)
            if "improved_async" not in best:
                continue
            imp = best["improved_async"]
            for base_proto in baselines:
                if base_proto not in best:
                    continue
                base = best[base_proto]
                shared_seeds = sorted(set(imp.keys()) & set(base.keys()))
                if not shared_seeds:
                    continue

                for metric_name, diff_fn in metric_specs:
                    diffs = []
                    imp_vals = []
                    base_vals = []
                    for s in shared_seeds:
                        iv = float(imp[s][metric_name])
                        bv = float(base[s][metric_name])
                        imp_vals.append(iv)
                        base_vals.append(bv)
                        diffs.append(diff_fn(iv, bv))

                    d_arr = np.array(diffs, dtype=np.float64)
                    mean_diff, std_diff, ci95 = _mean_std_ci(diffs)
                    wins = int(np.sum(d_arr > 1e-12))
                    losses = int(np.sum(d_arr < -1e-12))
                    ties = int(len(d_arr) - wins - losses)
                    p_sign = _sign_test_pvalue_two_sided(wins=wins, losses=losses)
                    dz = float(mean_diff / (std_diff + 1e-12))

                    imp_mu, _, _ = _mean_std_ci(imp_vals)
                    base_mu, _, _ = _mean_std_ci(base_vals)
                    rows.append({
                        "num_clients": int(n_clients),
                        "fairness_mode": mode,
                        "metric": metric_name,
                        "improved_proto": "improved_async(best-profile-per-seed)",
                        "baseline_proto": base_proto,
                        "paired_seeds": int(len(shared_seeds)),
                        "improved_mean": float(imp_mu),
                        "baseline_mean": float(base_mu),
                        "diff_mean_positive_is_better_for_improved": float(mean_diff),
                        "diff_std": float(std_diff),
                        "diff_ci95": float(ci95),
                        "cohen_dz": float(dz),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "p_value_sign": float(p_sign),
                        "p_value_holm": None,
                    })

    _apply_holm_correction(rows)
    rows.sort(
        key=lambda x: (
            int(x["num_clients"]),
            str(x["fairness_mode"]),
            str(x["metric"]),
            float(x["p_value_holm"]) if x["p_value_holm"] is not None else 1.0,
        )
    )
    return rows


def _build_summary_rows(
    raw_rows: List[Dict[str, Any]],
    n_list: List[int],
    fairness_modes: List[str],
) -> List[Dict[str, Any]]:
    """Aggregate by (n, mode, protocol_family): mean and best over profiles, then mean over seeds."""
    summary: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for n_clients in n_list:
        for mode in fairness_modes:
            for proto in ("fedavg", "fedasync", "fedbuff", "scaffold", "improved_async"):
                vals = [
                    r for r in raw_rows
                    if r["num_clients"] == n_clients
                    and r["fairness_mode"] == mode
                    and r["protocol_family"] == proto
                ]
                if not vals:
                    continue
                acc_mean, acc_std, acc_ci95 = _mean_std_ci([float(r["accuracy"]) for r in vals])
                bal_mean, bal_std, bal_ci95 = _mean_std_ci([float(r["score_balanced"]) for r in vals])
                comm_mean, comm_std, comm_ci95 = _mean_std_ci([float(r["communication_mb"]) for r in vals])

                seed_vals = {}
                for r in vals:
                    seed_vals.setdefault(r["seed"], []).append(r)
                best_seed_acc = []
                best_seed_bal = []
                for _, rs in seed_vals.items():
                    best = max(rs, key=lambda x: x["score_balanced"])
                    best_seed_acc.append(float(best["accuracy"]))
                    best_seed_bal.append(float(best["score_balanced"]))
                best_acc_mu, best_acc_std, best_acc_ci95 = _mean_std_ci(best_seed_acc)
                best_bal_mu, best_bal_std, best_bal_ci95 = _mean_std_ci(best_seed_bal)

                first_seed = next(iter(seed_vals.keys()))
                profile_count = len([r for r in vals if r["seed"] == first_seed])
                summary[(n_clients, mode, proto)] = {
                    "num_clients": n_clients,
                    "fairness_mode": mode,
                    "protocol_family": proto,
                    "mean_over_profiles_accuracy": acc_mean,
                    "mean_over_profiles_accuracy_std": acc_std,
                    "mean_over_profiles_accuracy_ci95": acc_ci95,
                    "mean_over_profiles_balanced_score": bal_mean,
                    "mean_over_profiles_balanced_score_std": bal_std,
                    "mean_over_profiles_balanced_score_ci95": bal_ci95,
                    "mean_over_profiles_comm_mb": comm_mean,
                    "mean_over_profiles_comm_mb_std": comm_std,
                    "mean_over_profiles_comm_mb_ci95": comm_ci95,
                    "best_over_profiles_accuracy": best_acc_mu,
                    "best_over_profiles_accuracy_std": best_acc_std,
                    "best_over_profiles_accuracy_ci95": best_acc_ci95,
                    "best_over_profiles_balanced_score": best_bal_mu,
                    "best_over_profiles_balanced_score_std": best_bal_std,
                    "best_over_profiles_balanced_score_ci95": best_bal_ci95,
                    "seeds": int(len(seed_vals)),
                    "profiles": int(profile_count),
                }

    summary_rows = list(summary.values())
    summary_rows.sort(
        key=lambda x: (x["num_clients"], x["fairness_mode"], -x["best_over_profiles_balanced_score"])
    )
    return summary_rows


def _write_checkpoints(
    raw_rows: List[Dict[str, Any]],
    n_list: List[int],
    fairness_modes: List[str],
) -> None:
    """Persist intermediate outputs so long runs survive timeout/preemption."""
    Path("results").mkdir(exist_ok=True)
    summary_rows = _build_summary_rows(raw_rows, n_list=n_list, fairness_modes=fairness_modes)
    pairwise_rows = _build_pairwise_rows(raw_rows, n_list=n_list, fairness_modes=fairness_modes)
    _save_csv("results/robust_external_raw.csv", raw_rows)
    _save_csv("results/robust_external_summary.csv", summary_rows)
    _save_csv("results/robust_external_pairwise_tests.csv", pairwise_rows)
    with open("results/robust_external_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_rows, f, indent=2)
    with open("results/robust_external_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open("results/robust_external_pairwise_tests.json", "w", encoding="utf-8") as f:
        json.dump(pairwise_rows, f, indent=2)


def _profile_bank(num_clients: int, model_cfg: Dict[str, Any], topk_fracs: List[float]) -> Dict[str, List[Dict[str, Any]]]:
    n_params = _estimate_model_param_count(model_cfg)
    k_vals = [max(200, int(n_params * float(fr))) for fr in topk_fracs]
    # Keep improved async buffering conservative at large scale to avoid
    # over-buffering and excessive staleness accumulation.
    base_min_buffer = int(np.clip(np.sqrt(max(1, num_clients)) / 3.0, 3, 5))
    base_max_buffer = int(np.clip(base_min_buffer * 2, 6, 10))
    k_common = k_vals[min(1, len(k_vals) - 1)]
    return {
        "fedavg": [
            {"strict_reproduction": True, "participation_rate": 0.5, "use_timeout": False, "max_round_time": 10.0},
            {"strict_reproduction": True, "participation_rate": 0.6, "use_timeout": False, "max_round_time": 10.0},
            {"strict_reproduction": True, "participation_rate": 0.7, "use_timeout": False, "max_round_time": 10.0, "compression": "topk", "k": k_common},
        ],
        "fedasync": [
            {"max_staleness": 8, "learning_rate": 0.6, "staleness_mode": "linear", "staleness_floor": 0.1, "server_tick_sec": 0.02},
            {"max_staleness": 10, "learning_rate": 0.8, "staleness_mode": "linear", "staleness_floor": 0.1, "server_tick_sec": 0.02},
            {"max_staleness": 15, "learning_rate": 1.0, "staleness_mode": "linear", "staleness_floor": 0.1, "server_tick_sec": 0.02, "compression": "topk", "k": k_common},
        ],
        "fedbuff": [
            {"buffer_size": 3, "max_staleness": 10, "server_lr": 0.2, "gradient_clip": 5.0},
            {"buffer_size": 5, "max_staleness": 15, "server_lr": 0.2, "gradient_clip": 5.0},
            {"buffer_size": 8, "max_staleness": 20, "server_lr": 0.2, "gradient_clip": 5.0, "compression": "topk", "k": k_common},
        ],
        "scaffold": [
            {"strict_reproduction": True, "participation_rate": 0.5, "use_timeout": False, "learning_rate": 0.6, "max_round_time": 10.0},
            {"strict_reproduction": True, "participation_rate": 0.6, "use_timeout": False, "learning_rate": 0.8, "max_round_time": 10.0},
            {"strict_reproduction": True, "participation_rate": 0.7, "use_timeout": False, "learning_rate": 1.0, "max_round_time": 10.0, "compression": "topk", "k": k_common},
        ],
        "improved_async": [
            {
                "max_staleness": 15,
                "min_buffer_size": base_min_buffer,
                "max_buffer_size": base_max_buffer,
                "momentum": 0.6,
                "adaptive_weighting": True,
                "auto_scale_params": False,
                "compression": None,
                "staleness_quantile": 0.9,
                "server_lr": 0.4,
                "gradient_clip": 5.0,
                "include_staleness_in_quality": True,
            },
            {
                "max_staleness": 15,
                "min_buffer_size": base_min_buffer,
                "max_buffer_size": base_max_buffer,
                "momentum": 0.5,
                "adaptive_weighting": True,
                "auto_scale_params": False,
                "compression": "qsgd",
                "num_bits": 8,
                "staleness_quantile": 0.9,
                "server_lr": 0.35,
                "gradient_clip": 5.0,
                "include_staleness_in_quality": True,
            },
            {
                "max_staleness": 18,
                "min_buffer_size": max(3, base_min_buffer - 1),
                "max_buffer_size": max(7, base_max_buffer - 1),
                "momentum": 0.5,
                "adaptive_weighting": True,
                "auto_scale_params": False,
                "compression": "topk",
                "k": k_common,
                "staleness_quantile": 0.9,
                "server_lr": 0.35,
                "gradient_clip": 5.0,
                "include_staleness_in_quality": True,
            },
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Robust large-scale protocol benchmark.")
    p.add_argument("--num_clients_list", type=str, default="20,200,1000")
    p.add_argument("--samples_per_client", type=int, default=250)
    p.add_argument("--train_size", type=int, default=20000)
    p.add_argument("--test_size", type=int, default=3000)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--local_lr", type=float, default=0.001)
    p.add_argument("--participation_rate", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fairness_modes", type=str, default="equal_rounds,equal_updates,equal_accepted_updates")
    p.add_argument("--track_interval_updates", type=int, default=20)
    p.add_argument("--acc_thresholds", type=str, default="0.20,0.25")
    p.add_argument("--topk_fractions", type=str, default="0.01,0.03,0.05")
    p.add_argument("--dominant_ratio", type=float, default=0.7)
    p.add_argument("--gamma_min", type=float, default=0.75)
    p.add_argument("--gamma_max", type=float, default=1.30)
    p.add_argument("--brightness_min", type=float, default=0.70)
    p.add_argument("--brightness_max", type=float, default=1.30)
    p.add_argument("--comm_budget_mb", type=float, default=300.0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--backbone", type=str, default="resnet18", help="Model backbone (e.g. resnet18)")
    p.add_argument("--checkpoint_every_mode", action="store_true", help="Save intermediate outputs after each mode")
    args = p.parse_args()
    print(f"[device] requested={args.device}", flush=True)
    Path("results").mkdir(exist_ok=True)
    with open("results/robust_external_metadata.json", "w", encoding="utf-8") as f:
        json.dump(_runtime_metadata(args), f, indent=2)

    n_list = [int(x.strip()) for x in args.num_clients_list.split(",") if x.strip()]
    fairness_modes = [x.strip() for x in args.fairness_modes.split(",") if x.strip()]
    topk_fracs = [float(x.strip()) for x in args.topk_fractions.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.acc_thresholds.split(",") if x.strip()]

    model_cfg = {
        "input_dim": 12,
        "hidden_dim": 96,
        "output_dim": 10,
        "image_size": 32,
        "image_channels": 3,
        "backbone": getattr(args, "backbone", "resnet18"),
    }

    raw_rows: List[Dict[str, Any]] = []

    try:
        for n_clients in n_list:
            print(f"\n========== Scale n={n_clients} ==========", flush=True)
            profile_bank = _profile_bank(n_clients, model_cfg=model_cfg, topk_fracs=topk_fracs)
            target_updates = int(args.rounds * max(1, int(n_clients * args.participation_rate)))

            for seed_offset in range(args.seeds):
                seed = int(args.seed + seed_offset)
                set_seed(seed)
                x_train, y_train, x_test, y_test = _load_cifar10_subset(args.train_size, args.test_size)
                client_datasets = _build_non_iid_clients(
                    x_train,
                    y_train,
                    n_clients,
                    args.samples_per_client,
                    dominant_ratio=args.dominant_ratio,
                    gamma_min=args.gamma_min,
                    gamma_max=args.gamma_max,
                    brightness_min=args.brightness_min,
                    brightness_max=args.brightness_max,
                    augment=True,
                    normalize=True,
                )
                test_ds = ClientPhotometricDataset(
                    x_test, y_test, gamma=1.0, brightness=1.0, train_mode=False, augment=False, normalize=True
                )

                for mode in fairness_modes:
                    print(f"\n=== n={n_clients}, seed={seed}, mode={mode} ===", flush=True)
                    n_active = max(1, int(n_clients * args.participation_rate))
                    # Deterministic per-(n, seed, fairness mode) client-activation RNG.
                    # This creates paired randomness across all protocol/profile runs.
                    schedule_base = (
                        f"{int(seed)}|{int(n_clients)}|{str(mode)}|{int(n_active)}|"
                        f"{int(args.rounds)}|{int(target_updates)}"
                    )
                    schedule_hash = hashlib.sha256(schedule_base.encode("utf-8")).hexdigest()
                    active_selection_seed = int(schedule_hash[:8], 16)
                    for proto, cfgs in profile_bank.items():
                        for idx, cfg in enumerate(cfgs):
                            key = f"{proto}_p{idx}"
                            row = run_once(
                                protocol_key=key,
                                protocol_name=proto,
                                protocol_cfg=cfg,
                                client_datasets=client_datasets,
                                test_dataset=test_ds,
                                model_cfg=model_cfg,
                                rounds=args.rounds,
                                fairness_mode=mode,
                                target_updates=target_updates,
                                local_epochs=args.local_epochs,
                                local_lr=args.local_lr,
                                participation_rate=args.participation_rate,
                                comm_budget_mb=args.comm_budget_mb,
                                track_interval_updates=args.track_interval_updates,
                                acc_thresholds=thresholds,
                                device=args.device,
                                active_selection_seed=active_selection_seed,
                                schedule_hash=schedule_hash,
                            )
                            row["num_clients"] = int(n_clients)
                            row["seed"] = int(seed)
                            row["protocol_family"] = proto
                            row["profile_idx"] = int(idx)
                            row.pop("trace", None)  # raw trace can be huge; keep per-run compact here
                            raw_rows.append(row)
                            print(
                                f"{key:22s} acc={row['accuracy']:.4f} bal={row['score_balanced']:.4f} comm={row['communication_mb']:.2f}MB",
                                flush=True,
                            )

                    if args.checkpoint_every_mode:
                        _write_checkpoints(raw_rows, n_list=n_list, fairness_modes=fairness_modes)
                        print(
                            f"[checkpoint] saved partial results after n={n_clients}, seed={seed}, mode={mode}",
                            flush=True,
                        )
    finally:
        # Final checkpoint always written, even if interrupted/failed.
        _write_checkpoints(raw_rows, n_list=n_list, fairness_modes=fairness_modes)

    print("\nSaved:")
    print("- results/robust_external_raw.csv")
    print("- results/robust_external_raw.json")
    print("- results/robust_external_summary.csv")
    print("- results/robust_external_summary.json")
    print("- results/robust_external_pairwise_tests.csv")
    print("- results/robust_external_pairwise_tests.json")


if __name__ == "__main__":
    main()
