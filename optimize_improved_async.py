"""
optimize_improved_async.py
Find optimal parameters for improved async protocol
"""

import torch
import numpy as np
import time
from federated_protocol_framework import create_protocol, ClientUpdate
from unified_protocol_comparison import SimpleNN, generate_federated_data, train_client, evaluate_model, \
    evaluate_with_intent_and_explanation
import json
from metrics import tri_objective_score

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
DATASET_NAME = "tartanair"
TARTANAIR_ROOT = "tartanair-test-mono-release/mono"
IMAGE_SIZE = 32


def test_configuration(config, client_datasets, test_dataset, model_config, num_rounds=10):
    """
    Test a specific (valid) configuration of Improved Async protocol.
    Supported compression keys:
      - compression: None | 'topk' | 'signsgd' | 'qsgd'
      - k: int (only for 'topk')
      - num_bits: int (only for 'qsgd')
    Other keys: max_staleness, min_buffer_size, max_buffer_size, momentum, adaptive_weighting, etc.
    """
    protocol = create_protocol('improved_async', num_clients=len(client_datasets), **config)
    start_time = time.time()

    # Set initial model
    initial_model = SimpleNN(**model_config)
    protocol.set_global_model(initial_model.state_dict())

    # Simulate training
    for round_num in range(num_rounds):
        for client_id in range(len(client_datasets)):
            global_state, pulled_version = protocol.get_global_model_with_version()
            if global_state is None:
                continue

            # Local training
            local_model = SimpleNN(**model_config)
            local_model.load_state_dict(global_state)

            scaffold_c_global, scaffold_c_client = None, None
            if hasattr(protocol, "get_scaffold_controls"):
                scaffold_c_global, scaffold_c_client = protocol.get_scaffold_controls(f"client_{client_id}")
            lr = 0.01
            updated_state, loss, data_size, local_steps = train_client(
                local_model, client_datasets[client_id],
                epochs=2, lr=lr,
                scaffold_c_global=scaffold_c_global,
                scaffold_c_client=scaffold_c_client,
            )

            # Calculate update (delta = local - global)
            update_dict = {}
            for name, param in updated_state.items():
                if name in global_state and "num_batches_tracked" not in name:
                    update_dict[name] = param.float() - global_state[name].float()

            # Submit update
            use_scaffold = hasattr(protocol, "get_scaffold_controls")
            update = ClientUpdate(
                client_id=f"client_{client_id}",
                update_data=update_dict,
                model_version=int(pulled_version),
                local_loss=float(loss),
                data_size=int(data_size),
                timestamp=0.0,
                local_steps=int(local_steps) if use_scaffold else None,
                local_lr=float(lr) if use_scaffold else None,
            )
            protocol.receive_update(update)

    # Final evaluation
    final_model_state = protocol.get_global_model()
    if final_model_state:
        eval_model = SimpleNN(**model_config)
        eval_model.load_state_dict(final_model_state)

        # Keep legacy metrics
        accuracy, loss = evaluate_model(eval_model, test_dataset)
        # Add intent/bleu for consistency with other runners
        intent_f1, expl_bleu, _, _ = evaluate_with_intent_and_explanation(
            eval_model, test_dataset, device="cpu", id2label={i: f"intent_{i}" for i in range(model_config['output_dim'])},
            num_classes=model_config['output_dim']
        )

        metrics = protocol.metrics.get_summary()
        elapsed_sec = float(time.time() - start_time)
        protocol.shutdown()

        return {
            'accuracy': accuracy,
            'loss': loss,
            'intent_f1': intent_f1,
            'bleu': expl_bleu,
            'aggregations': metrics['aggregations_performed'],
            'communication_mb': metrics['total_data_transmitted_mb'],
            'tri_objective': tri_objective_score(
                accuracy=accuracy,
                communication_mb=metrics['total_data_transmitted_mb'],
                latency_sec=elapsed_sec,
                comm_budget_mb=60.0,
                latency_budget_sec=max(elapsed_sec, 1.0),
            )
        }

    protocol.shutdown()
    return None


def find_optimal_parameters():
    """Find optimal parameters for improved async protocol"""
    print("=" * 70)
    print("OPTIMIZING IMPROVED ASYNC PROTOCOL PARAMETERS")
    print("=" * 70)

    # Generate test data
    print("\nGenerating test data...")
    client_datasets, test_dataset = generate_federated_data(
        num_clients=5,
        samples_per_client=50,
        input_dim=10,
        num_classes=3,
        heterogeneity=0.5,
        dataset_name=DATASET_NAME,
        tartanair_root=TARTANAIR_ROOT,
        image_size=IMAGE_SIZE,
    )

    model_config = {'input_dim': 10, 'hidden_dim': 32, 'output_dim': 3}

    # Valid configuration space (no 'compression_ratio' anymore)
    configurations = [
        # Baseline (no compression)
        {
            'name': 'Baseline (No Compression)',
            'config': {
                'compression': None,
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Top-K (light)
        {
            'name': 'TopK k=10',
            'config': {
                'compression': 'topk',
                'k': 10,
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Top-K (medium)
        {
            'name': 'TopK k=50',
            'config': {
                'compression': 'topk',
                'k': 50,
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Top-K (stronger)
        {
            'name': 'TopK k=100',
            'config': {
                'compression': 'topk',
                'k': 100,
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # SignSGD
        {
            'name': 'SignSGD',
            'config': {
                'compression': 'signsgd',
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # QSGD (8-bit)
        {
            'name': 'QSGD 8-bit',
            'config': {
                'compression': 'qsgd',
                'num_bits': 8,
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Larger buffer
        {
            'name': 'Larger Buffer',
            'config': {
                'compression': 'topk',
                'k': 50,
                'max_staleness': 15,
                'min_buffer_size': 3,
                'max_buffer_size': 6,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Frequent aggregation
        {
            'name': 'Frequent Aggregation',
            'config': {
                'compression': 'topk',
                'k': 50,
                'max_staleness': 10,
                'min_buffer_size': 1,
                'max_buffer_size': 2,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Lower momentum
        {
            'name': 'Lower Momentum',
            'config': {
                'compression': 'topk',
                'k': 50,
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 4,
                'momentum': 0.7,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Balanced
        {
            'name': 'Balanced',
            'config': {
                'compression': 'topk',
                'k': 50,
                'max_staleness': 15,
                'min_buffer_size': 2,
                'max_buffer_size': 5,
                'momentum': 0.85,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        # Accuracy Focus (no compression)
        {
            'name': 'Accuracy Focus (No Compression)',
            'config': {
                'compression': None,
                'max_staleness': 8,
                'min_buffer_size': 1,
                'max_buffer_size': 3,
                'momentum': 0.95,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        }
    ]

    results = []
    best_config = None
    best_score = -1.0

    print("\nTesting configurations...")
    print("-" * 70)

    for cfg in configurations:
        print(f"\nTesting: {cfg['name']}")
        print(f"  Config: {cfg['config']}")

        result = test_configuration(
            cfg['config'],
            client_datasets,
            test_dataset,
            model_config,
            num_rounds=15
        )

        if result:
            # Composite score: prioritize intent-F1, then comm efficiency, then accuracy
            comm_eff = 1.0 / (1.0 + result['communication_mb'])
            score = 0.6 * result['intent_f1'] + 0.2 * comm_eff + 0.2 * result['accuracy']

            results.append({
                'name': cfg['name'],
                'config': cfg['config'],
                'accuracy': result['accuracy'],
                'loss': result['loss'],
                'intent_f1': result['intent_f1'],
                'bleu': result['bleu'],
                'communication_mb': result['communication_mb'],
                'aggregations': result['aggregations'],
                'score': score
            })

            print(f"  Results:")
            print(f"    Acc: {result['accuracy']:.4f} | F1: {result['intent_f1']:.4f} | BLEU: {result['bleu']:.4f}")
            print(f"    Comm: {result['communication_mb']:.3f} MB | Agg: {result['aggregations']}")
            print(f"    Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_config = cfg['config']
                print("    *** NEW BEST ***")

    results.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    print("\nTop 3 Configurations:")
    for i, r in enumerate(results[:3], 1):
        print(f"\n{i}. {r['name']}")
        print(f"   Score: {r['score']:.4f}")
        print(f"   Acc: {r['accuracy']:.4f} | F1: {r['intent_f1']:.4f} | BLEU: {r['bleu']:.4f}")
        print(f"   Comm: {r['communication_mb']:.3f} MB, Agg: {r['aggregations']}")

    if best_config:
        print("\n" + "-" * 70)
        print("BEST CONFIGURATION:")
        print("-" * 70)
        for key, value in best_config.items():
            print(f"  {key}: {value}")

        with open('optimal_improved_async_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)

        print("\n✓ Best configuration saved to 'optimal_improved_async_config.json'")

    return best_config, results


def compare_with_baseline():
    """Compare optimized config with baseline protocols"""
    print("\n" + "=" * 70)
    print("COMPARING OPTIMIZED CONFIG WITH BASELINE PROTOCOLS")
    print("=" * 70)

    # Load optimal config
    try:
        with open('optimal_improved_async_config.json', 'r') as f:
            optimal_config = json.load(f)
    except:
        print("No optimal config found. Running optimization first...")
        optimal_config, _ = find_optimal_parameters()

    # Generate test data
    client_datasets, test_dataset = generate_federated_data(
        num_clients=5,
        samples_per_client=50,
        input_dim=10,
        num_classes=3,
        heterogeneity=0.5,
        dataset_name=DATASET_NAME,
        tartanair_root=TARTANAIR_ROOT,
        image_size=IMAGE_SIZE,
    )

    model_config = {
        'input_dim': 10,
        'hidden_dim': 32,
        'output_dim': 3,
        'image_size': IMAGE_SIZE,
        'image_channels': 1,
        'backbone': 'resnet18',
    }

    protocols = {
        'SCAFFOLD': {
            'protocol': 'scaffold',
            'config': {'participation_rate': 0.5},
        },
        'FedAsync': {
            'protocol': 'fedasync',
            'config': {'max_staleness': 10}
        },
        'FedBuff': {
            'protocol': 'fedbuff',
            'config': {'buffer_size': 3, 'max_staleness': 10}
        },
        'Improved (Original-like)': {
            'protocol': 'improved_async',
            'config': {
                'compression': 'topk',
                'k': 50,
                'max_staleness': 10,
                'min_buffer_size': 2,
                'max_buffer_size': 3,
                'momentum': 0.9,
                'adaptive_weighting': True,
                'staleness_mode': 'quadratic'
            }
        },
        'Improved (Optimized)': {
            'protocol': 'improved_async',
            'config': optimal_config
        }
    }

    print("\nRunning comparison...")
    print("-" * 70)

    results = {}

    for name, proto_info in protocols.items():
        print(f"\nTesting {name}...")

        protocol = create_protocol(
            proto_info['protocol'],
            num_clients=len(client_datasets),
            **proto_info['config']
        )

        # Set initial model
        initial_model = SimpleNN(**model_config)
        protocol.set_global_model(initial_model.state_dict())

        # Train
        for round_num in range(15):
            for client_id in range(len(client_datasets)):
                global_state, pulled_version = protocol.get_global_model_with_version()
                if global_state is None:
                    continue

                local_model = SimpleNN(**model_config)
                local_model.load_state_dict(global_state)

                scaffold_c_global, scaffold_c_client = None, None
                if hasattr(protocol, 'get_scaffold_controls'):
                    scaffold_c_global, scaffold_c_client = protocol.get_scaffold_controls(f"client_{client_id}")
                lr = 0.01
                updated_state, loss, data_size, local_steps = train_client(
                    local_model, client_datasets[client_id],
                    epochs=2, lr=lr,
                    scaffold_c_global=scaffold_c_global,
                    scaffold_c_client=scaffold_c_client,
                )

                update_dict = {}
                for param_name, param in updated_state.items():
                    if param_name in global_state and 'num_batches_tracked' not in param_name:
                        update_dict[param_name] = param.float() - global_state[param_name].float()

                use_scaffold = hasattr(protocol, 'get_scaffold_controls')
                update = ClientUpdate(
                    client_id=f"client_{client_id}",
                    update_data=update_dict,
                    model_version=int(pulled_version),
                    local_loss=float(loss),
                    data_size=int(data_size),
                    timestamp=0.0,
                    local_steps=int(local_steps) if use_scaffold else None,
                    local_lr=float(lr) if use_scaffold else None,
                )
                protocol.receive_update(update)

        # Evaluate
        final_model_state = protocol.get_global_model()
        if final_model_state:
            eval_model = SimpleNN(**model_config)
            eval_model.load_state_dict(final_model_state)
            accuracy, loss = evaluate_model(eval_model, test_dataset)

            metrics = protocol.metrics.get_summary()

            results[name] = {
                'accuracy': accuracy,
                'loss': loss,
                'communication_mb': metrics['total_data_transmitted_mb'],
                'aggregations': metrics['aggregations_performed']
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Communication: {metrics['total_data_transmitted_mb']:.3f} MB")

        protocol.shutdown()

    # Print comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Protocol':<20} {'Accuracy':<12} {'Comm (MB)':<12} {'Efficiency':<12}")
    print("-" * 56)

    for name, m in results.items():
        efficiency = m['accuracy'] / (1.0 + m['communication_mb'])
        print(f"{name:<20} {m['accuracy']:<12.4f} {m['communication_mb']:<12.3f} {efficiency:<12.4f}")


def main():
    """Main optimization pipeline"""
    print("\nIMPROVED ASYNC PROTOCOL OPTIMIZATION\n")

    # Step 1: Find optimal parameters
    best_config, all_results = find_optimal_parameters()

    # Step 2: Compare with baselines
    compare_with_baseline()

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print("\nUse the configuration in 'optimal_improved_async_config.json'")
    print("for best performance in your experiments.")


if __name__ == "__main__":
    main()
