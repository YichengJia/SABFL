import time
import json
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from federated_protocol_framework import create_protocol, ClientUpdate, build_scaffold_control_payload
from unified_protocol_comparison import SimpleNN, generate_federated_data, train_client, set_seed
from unified_protocol_comparison import evaluate_with_intent_and_explanation
from metrics import tri_objective_score, pareto_front_mask

PROTOCOLS = ["fedavg", "fedasync", "fedbuff", "scaffold", "improved_async"]
TOPK_LIST = [None, 1, 10, 50, 100, 200, 500, 1000]
NUM_CLIENTS = 20
ROUNDS = 100
LOCAL_EPOCHS = 1
INPUT_DIM = 16
NUM_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "tartanair"
TARTANAIR_ROOT = "tartanair-test-mono-release/mono"
IMAGE_SIZE = 32

W_INTENT = 0.6
W_BLEU   = 0.2
W_COMM   = 0.2

def normalized_score(intent_f1, bleu, comm_mb, comm_ref):
    # keep BLEU influence through pseudo-accuracy blend for backwards compatibility
    pseudo_acc = 0.5 * intent_f1 + 0.5 * bleu
    return tri_objective_score(
        accuracy=pseudo_acc,
        communication_mb=comm_mb,
        latency_sec=1.0,
        comm_budget_mb=max(comm_ref, 1e-9),
        latency_budget_sec=1.0,
        w_acc=W_INTENT + W_BLEU,
        w_comm=W_COMM,
        w_lat=0.0,
    )

def main():
    set_seed(42)
    print("\n" + "="*70)
    print("JOINT STUDY: Protocols x Top-K (Intent-F1 & Explanation-BLEU)")
    print("="*70)

    # Data
    client_datasets, test_dataset = generate_federated_data(
        num_clients=NUM_CLIENTS,
        samples_per_client=500,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        heterogeneity=0.5,
        dataset_name=DATASET_NAME,
        tartanair_root=TARTANAIR_ROOT,
        image_size=IMAGE_SIZE,
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    results = defaultdict(dict)

    for proto in PROTOCOLS:
        # Protocol-specific normalization: run k=None once to get comm_ref
        cfg_base = {
            "participation_rate": 0.5,
            "buffer_size": 5,
            "max_staleness": 10,
            "compression": None,
            "k": 0,
            "num_bits": 8,
            "server_lr": 0.2,
            "gradient_clip": 1.0,
            "adaptive_weighting": True,
            "min_buffer_size": 2,
            "max_buffer_size": 6,
            "momentum": 0.8,
        }
        protocol = create_protocol(proto, NUM_CLIENTS, **cfg_base)

        # Init model
        global_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES).to(DEVICE)
        protocol.set_global_model(global_model.state_dict())

        # Warm-up run to estimate comm_ref_mb
        start = time.time()
        for r in range(5):  # short warm-up rounds
            num_active = max(1, int(NUM_CLIENTS * cfg_base["participation_rate"]))
            active = np.random.choice(NUM_CLIENTS, size=num_active, replace=False)
            for cid in active:
                ds = client_datasets[cid]
                local_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES)
                state, pulled_version = protocol.get_global_model_with_version()
                if state:
                    protocol.account_model_downlink(state)
                    local_model.load_state_dict(state, strict=False)

                scaffold_c_global = None
                scaffold_c_client = None
                if proto == "scaffold":
                    scaffold_c_global, scaffold_c_client = protocol.get_scaffold_controls(str(cid))
                updated_state, local_loss, n_samples, local_steps = train_client(
                    local_model,
                    ds,
                    epochs=LOCAL_EPOCHS,
                    lr=0.01,
                    device=DEVICE,
                    scaffold_c_global=scaffold_c_global,
                    scaffold_c_client=scaffold_c_client,
                )

                update_dict = {}
                if state:
                    for name, p_new in updated_state.items():
                        if name in state and "num_batches_tracked" not in name:
                            update_dict[name] = (p_new.float() - state[name].float())
                else:
                    for name, p_new in updated_state.items():
                        if "num_batches_tracked" not in name:
                            update_dict[name] = p_new.float()
                update_transport = protocol.compress_for_transport(
                    update_dict,
                    client_id=str(cid),
                )
                scaffold_payload = None
                if proto == "scaffold" and scaffold_c_global is not None and scaffold_c_client is not None:
                    scaffold_payload = build_scaffold_control_payload(
                        update_delta=update_dict,
                        c_global=scaffold_c_global,
                        c_client=scaffold_c_client,
                        local_steps=int(local_steps),
                        local_lr=0.01,
                    )

                update = ClientUpdate(
                    client_id=str(cid),
                    update_data=update_transport,
                    model_version=int(pulled_version),
                    local_loss=float(local_loss),
                    data_size=int(n_samples),
                    timestamp=time.time(),
                    local_steps=int(local_steps) if proto == "scaffold" else None,
                    local_lr=0.01 if proto == "scaffold" else None,
                    scaffold_control_payload=scaffold_payload,
                )
                protocol.receive_update(update)
        comm_ref_mb = float(protocol.metrics.metrics.get("total_data_transmitted_mb", 1.0))
        # Reset protocol for actual run
        protocol.shutdown()
        del protocol

        # Now run for all k with this protocol
        for k in TOPK_LIST:
            cfg = dict(cfg_base)
            cfg["compression"] = "topk" if k is not None else None
            cfg["k"] = 0 if k is None else int(k)

            protocol = create_protocol(proto, NUM_CLIENTS, **cfg)
            global_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES).to(DEVICE)
            protocol.set_global_model(global_model.state_dict())

            print(f"\nRunning {proto} with k={k}")
            start = time.time()

            # Optional budgets (uncomment to enforce)
            # COMM_BUDGET_MB = 40.0
            # AGG_BUDGET = 1000

            for r in range(ROUNDS):
                num_active = max(1, int(NUM_CLIENTS * cfg["participation_rate"]))
                active = np.random.choice(NUM_CLIENTS, size=num_active, replace=False)

                for cid in active:
                    ds = client_datasets[cid]
                    local_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES)
                    state, pulled_version = protocol.get_global_model_with_version()
                    if state:
                        protocol.account_model_downlink(state)
                        local_model.load_state_dict(state, strict=False)

                    scaffold_c_global = None
                    scaffold_c_client = None
                    if proto == "scaffold":
                        scaffold_c_global, scaffold_c_client = protocol.get_scaffold_controls(str(cid))
                    updated_state, local_loss, n_samples, local_steps = train_client(
                        local_model,
                        ds,
                        epochs=LOCAL_EPOCHS,
                        lr=0.01,
                        device=DEVICE,
                        scaffold_c_global=scaffold_c_global,
                        scaffold_c_client=scaffold_c_client,
                    )

                    update_dict = {}
                    if state:
                        for name, p_new in updated_state.items():
                            if name in state and "num_batches_tracked" not in name:
                                update_dict[name] = (p_new.float() - state[name].float())
                    else:
                        for name, p_new in updated_state.items():
                            if "num_batches_tracked" not in name:
                                update_dict[name] = p_new.float()
                    update_transport = protocol.compress_for_transport(
                        update_dict,
                        client_id=str(cid),
                    )
                    scaffold_payload = None
                    if proto == "scaffold" and scaffold_c_global is not None and scaffold_c_client is not None:
                        scaffold_payload = build_scaffold_control_payload(
                            update_delta=update_dict,
                            c_global=scaffold_c_global,
                            c_client=scaffold_c_client,
                            local_steps=int(local_steps),
                            local_lr=0.01,
                        )

                    update = ClientUpdate(
                        client_id=str(cid),
                        update_data=update_transport,
                        model_version=int(pulled_version),
                        local_loss=float(local_loss),
                        data_size=int(n_samples),
                        timestamp=time.time(),
                        local_steps=int(local_steps) if proto == "scaffold" else None,
                        local_lr=0.01 if proto == "scaffold" else None,
                        scaffold_control_payload=scaffold_payload,
                    )
                    protocol.receive_update(update)

                # Early-stop by budgets (optional)
                # if protocol.metrics.metrics.get("total_data_transmitted_mb", 0.0) >= COMM_BUDGET_MB:
                #     break
                # if protocol.metrics.metrics.get("aggregations_performed", 0) >= AGG_BUDGET:
                #     break

            # Evaluate
            eval_model = SimpleNN(input_dim=INPUT_DIM, hidden_dim=64, output_dim=NUM_CLASSES).to(DEVICE)
            state = protocol.get_global_model()
            if state:
                eval_model.load_state_dict(state, strict=False)

            intent_f1, expl_bleu, _, _ = evaluate_with_intent_and_explanation(
                eval_model, test_loader, DEVICE, id2label={i: f"intent_{i}" for i in range(NUM_CLASSES)}, num_classes=NUM_CLASSES
            )
            comm_mb = float(protocol.metrics.metrics.get("total_data_transmitted_mb", 0.0))
            aggs = int(protocol.metrics.metrics.get("aggregations_performed", 0))
            score = normalized_score(intent_f1, expl_bleu, comm_mb, comm_ref_mb)
            elapsed = time.time() - start

            results[proto][str(k)] = {
                "intent_f1": round(intent_f1, 4),
                "explanation_bleu": round(expl_bleu, 4),
                "communication_mb": round(comm_mb, 2),
                "aggregations": aggs,
                "elapsed_sec": round(elapsed, 1),
                "score": round(score, 4),
            }
            print(f"  Intent-F1={intent_f1:.4f}  BLEU={expl_bleu:.4f}  Comm={comm_mb:.2f}MB  Agg={aggs}  Score={score:.4f}")

            protocol.shutdown()

    print("\n" + "="*70)
    print("SUMMARY (best per protocol)")
    print("="*70)
    for proto in PROTOCOLS:
        ks = list(results[proto].keys())
        accs = [results[proto][k]["intent_f1"] for k in ks]
        comms = [results[proto][k]["communication_mb"] for k in ks]
        lats = [results[proto][k]["elapsed_sec"] for k in ks]
        front = pareto_front_mask(accs, comms, lats)
        for k, is_front in zip(ks, front):
            results[proto][k]["pareto_optimal"] = bool(is_front)

        best_k, best = None, None
        for k, v in results[proto].items():
            if best is None or v["score"] > best["score"]:
                best_k, best = k, v
        print(f"{proto:12s}  best_k={best_k:>4}  score={best['score']:.4f}  "
              f"Intent-F1={best['intent_f1']:.4f}  BLEU={best['explanation_bleu']:.4f}  "
              f"Comm={best['communication_mb']:.4f}MB  Agg={best['aggregations']}")

    with open("joint_protocol_topk_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: joint_protocol_topk_results.json")


if __name__ == "__main__":
    main()
