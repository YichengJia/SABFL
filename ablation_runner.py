"""
ablation_runner.py
Run component ablations for ImprovedAsync and export results.
"""

import csv
import json
import time
from typing import Dict, Any, List
import argparse

import numpy as np
import torch

from federated_protocol_framework import create_protocol, ClientUpdate
from metrics import tri_objective_score, pareto_front_mask
from unified_protocol_comparison import (
    SimpleNN,
    generate_federated_data,
    train_client,
    evaluate_model,
    evaluate_with_intent_and_explanation,
    set_seed,
)


def run_single_experiment(
    protocol_name: str,
    protocol_cfg: Dict[str, Any],
    client_datasets,
    test_dataset,
    model_cfg: Dict[str, Any],
    rounds: int,
    local_epochs: int,
    local_lr: float,
    participation_rate: float,
    comm_budget_mb: float,
) -> Dict[str, Any]:
    protocol = create_protocol(protocol_name, num_clients=len(client_datasets), **protocol_cfg)
    initial_model = SimpleNN(**model_cfg)
    protocol.set_global_model(initial_model.state_dict())

    start = time.time()
    for _ in range(rounds):
        num_clients = len(client_datasets)
        n_active = max(1, int(num_clients * participation_rate))
        active = np.random.choice(num_clients, size=n_active, replace=False)

        for cid in active:
            global_state, pulled_version = protocol.get_global_model_with_version()
            if global_state is None:
                continue

            local_model = SimpleNN(**model_cfg)
            local_model.load_state_dict(global_state, strict=False)

            scaffold_c_global, scaffold_c_client = None, None
            if hasattr(protocol, "get_scaffold_controls"):
                scaffold_c_global, scaffold_c_client = protocol.get_scaffold_controls(f"client_{cid}")
            updated_state, local_loss, data_size, local_steps = train_client(
                local_model,
                client_datasets[cid],
                epochs=local_epochs,
                lr=local_lr,
                scaffold_c_global=scaffold_c_global,
                scaffold_c_client=scaffold_c_client,
            )

            update_dict = {}
            for pname, pnew in updated_state.items():
                if pname in global_state and "num_batches_tracked" not in pname:
                    update_dict[pname] = pnew.float() - global_state[pname].float()

            use_scaffold = hasattr(protocol, "get_scaffold_controls")
            update = ClientUpdate(
                client_id=f"client_{cid}",
                update_data=update_dict,
                model_version=int(pulled_version),
                local_loss=float(local_loss),
                data_size=int(data_size),
                timestamp=time.time(),
                local_steps=int(local_steps) if use_scaffold else None,
                local_lr=float(local_lr) if use_scaffold else None,
            )
            protocol.receive_update(update)

    elapsed = float(time.time() - start)
    final_state = protocol.get_global_model()
    eval_model = SimpleNN(**model_cfg)
    if final_state:
        eval_model.load_state_dict(final_state, strict=False)

    acc, loss = evaluate_model(eval_model, test_dataset)
    intent_f1, bleu, _, _ = evaluate_with_intent_and_explanation(
        eval_model, test_dataset, device="auto", num_classes=model_cfg["output_dim"]
    )
    metrics = protocol.metrics.get_summary()
    comm_mb = float(metrics.get("total_data_transmitted_mb", 0.0))

    score = tri_objective_score(
        accuracy=float(acc),
        communication_mb=comm_mb,
        latency_sec=elapsed,
        comm_budget_mb=comm_budget_mb,
        latency_budget_sec=max(elapsed, 1.0),
    )

    out = {
        "accuracy": float(acc),
        "loss": float(loss),
        "intent_f1": float(intent_f1),
        "bleu": float(bleu),
        "communication_mb": comm_mb,
        "elapsed_sec": elapsed,
        "aggregations": int(metrics.get("aggregations_performed", 0)),
        "tri_objective": float(score),
    }
    protocol.shutdown()
    return out


def build_ablation_suite(num_clients: int) -> Dict[str, Dict[str, Any]]:
    base = {
        "max_staleness": 15,
        "min_buffer_size": 3,
        "max_buffer_size": 8,
        "adaptive_weighting": True,
        "momentum": 0.85,
        "compression": "topk",
        "k": 100,
        "auto_scale_params": True,
        "staleness_mode": "quadratic",
        "staleness_floor": 0.2,
        "staleness_quantile": 0.9,
        "enable_staleness_adaptation": True,
        "enable_health_relaxation": True,
        "server_lr": 0.2,
        "gradient_clip": 2.0,
        # scale-sensitive defaults (overridden by auto_scale if enabled)
        "participation_rate": float(np.clip(2.0 / np.sqrt(max(num_clients, 1)), 0.1, 1.0)),
    }
    return {
        "A0_full": dict(base),
        "A1_no_auto_scale": dict(base, auto_scale_params=False, min_buffer_size=3, max_buffer_size=8),
        "A2_no_staleness_adapt": dict(base, enable_staleness_adaptation=False, enable_health_relaxation=False),
        "A3_no_adaptive_weighting": dict(base, adaptive_weighting=False),
        "A4_no_momentum": dict(base, momentum=0.0),
        "A5_no_compression": dict(base, compression=None),
        "A6_linear_staleness": dict(base, staleness_mode="linear"),
        "A7_fixed_buffer_small": dict(base, auto_scale_params=False, min_buffer_size=2, max_buffer_size=2),
    }


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run ImprovedAsync ablations.")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--samples_per_client", type=int, default=150)
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--participation_rate", type=float, default=0.5)
    parser.add_argument("--dataset_name", type=str, default="tartanair")
    parser.add_argument("--tartanair_root", type=str, default="tartanair-test-mono-release/mono")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = {
        "num_clients": int(args.num_clients),
        "samples_per_client": int(args.samples_per_client),
        "input_dim": 12,
        "hidden_dim": 64,
        "num_classes": int(args.num_classes),
        "heterogeneity": 0.5,
        "dataset_name": str(args.dataset_name),
        "tartanair_root": str(args.tartanair_root),
        "image_size": int(args.image_size),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "local_lr": 0.01,
        "participation_rate": float(args.participation_rate),
        "comm_budget_mb": 60.0,
    }

    client_datasets, test_dataset = generate_federated_data(
        num_clients=cfg["num_clients"],
        samples_per_client=cfg["samples_per_client"],
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        heterogeneity=cfg["heterogeneity"],
        dataset_name=cfg["dataset_name"],
        tartanair_root=cfg["tartanair_root"],
        image_size=cfg["image_size"],
    )
    model_cfg = {
        "input_dim": cfg["input_dim"],
        "hidden_dim": cfg["hidden_dim"],
        "output_dim": cfg["num_classes"],
        "image_size": cfg["image_size"],
        "image_channels": 1,
        "backbone": "resnet18",
    }

    suite = build_ablation_suite(num_clients=cfg["num_clients"])
    rows = []
    detailed = {}

    print("\n" + "=" * 72)
    print("ABLATION: IMPROVED ASYNC COMPONENT STUDY")
    print("=" * 72)

    for name, p_cfg in suite.items():
        print(f"\nRunning {name} ...")
        result = run_single_experiment(
            protocol_name="improved_async",
            protocol_cfg=p_cfg,
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            model_cfg=model_cfg,
            rounds=cfg["rounds"],
            local_epochs=cfg["local_epochs"],
            local_lr=cfg["local_lr"],
            participation_rate=cfg["participation_rate"],
            comm_budget_mb=cfg["comm_budget_mb"],
        )
        detailed[name] = {"config": p_cfg, "metrics": result}
        row = {"ablation": name}
        row.update(result)
        rows.append(row)
        print(
            f"  Acc={result['accuracy']:.4f} F1={result['intent_f1']:.4f} "
            f"Comm={result['communication_mb']:.2f}MB Time={result['elapsed_sec']:.1f}s "
            f"Score={result['tri_objective']:.4f}"
        )

    # Pareto marking across ablation runs
    acc = [r["accuracy"] for r in rows]
    comm = [r["communication_mb"] for r in rows]
    lat = [r["elapsed_sec"] for r in rows]
    front = pareto_front_mask(acc, comm, lat)
    for r, f in zip(rows, front):
        r["pareto_optimal"] = bool(f)
        detailed[r["ablation"]]["metrics"]["pareto_optimal"] = bool(f)

    rows = sorted(rows, key=lambda x: x["tri_objective"], reverse=True)
    write_csv("ablation_results.csv", rows)
    with open("ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    print("\nTop-3 by tri-objective:")
    for i, r in enumerate(rows[:3], 1):
        print(f"{i}. {r['ablation']}  score={r['tri_objective']:.4f}  pareto={r['pareto_optimal']}")

    print("\nSaved:")
    print("- ablation_results.csv")
    print("- ablation_results.json")


if __name__ == "__main__":
    main()
