"""
optimized_protocol_config.py
Better configuration for improved async protocol based on analysis
"""

# Base scenario settings (without compression)
BASE_SCENARIOS = {
    'high_accuracy': {
        'max_staleness': 10,
        'min_buffer_size': 2,
        'max_buffer_size': 4,
        'momentum': 0.95,
        'adaptive_weighting': True,
    },
    'balanced': {
        'max_staleness': 15,
        'min_buffer_size': 3,
        'max_buffer_size': 6,
        'momentum': 0.85,
        'adaptive_weighting': True,
    },
    'low_communication': {
        'max_staleness': 20,
        'min_buffer_size': 5,
        'max_buffer_size': 10,
        'momentum': 0.8,
        'adaptive_weighting': True,
    }
}

# Compression options
COMPRESSION_OPTIONS = [
    {'compression': None},
    {'compression': 'topk', 'k': 100},
    {'compression': 'signsgd'},
    {'compression': 'qsgd', 'num_bits': 8},
]


def generate_all_configs():
    """Generate all scenario x compression combinations"""
    configs = {}
    for scenario_name, base_config in BASE_SCENARIOS.items():
        for comp in COMPRESSION_OPTIONS:
            # Build new config by merging base and compression
            config = base_config.copy()
            config.update(comp)
            key_name = scenario_name
            if comp['compression'] is not None:
                key_name += f"_{comp['compression']}"
            else:
                key_name += "_no_compression"
            configs[key_name] = config
    return configs


def get_improved_config(scenario='balanced', compression=None):
    """Get improved configuration for a scenario, with optional compression override"""
    if scenario not in BASE_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    config = BASE_SCENARIOS[scenario].copy()

    if compression is not None:
        # Find compression option
        for comp in COMPRESSION_OPTIONS:
            if comp['compression'] == compression:
                config.update(comp)
                break
    return config


def get_scaled_improved_config(
    num_clients: int,
    scenario: str = 'balanced',
    compression=None,
    auto_scale_params: bool = True,
):
    """
    Build improved-async config with optional scale-aware parameterization.

    Notes:
    - When auto_scale_params=True, protocol internals may override
      max_staleness/min_buffer_size/max_buffer_size based on num_clients.
    - Scenario/compression values remain useful defaults for momentum,
      adaptive weighting, and compression strategy selection.
    """
    config = get_improved_config(scenario=scenario, compression=compression)
    config['auto_scale_params'] = bool(auto_scale_params)
    return config


def get_layered_improved_config(
    num_clients: int,
    scenario: str = 'balanced',
    compression=None,
    staleness_quantile: float = 0.9,
):
    """
    Three-layer parameterization:
      1) scale-related: buffer sizes and participation
      2) system-related: quantile-based effective max staleness
      3) stability-related: momentum / server_lr / gradient_clip
    """
    cfg = get_scaled_improved_config(
        num_clients=num_clients,
        scenario=scenario,
        compression=compression,
        auto_scale_params=True,
    )
    cfg.update({
        # system-observed adaptation
        'staleness_quantile': float(staleness_quantile),
        'staleness_history_size': 512,
        'min_effective_staleness': 5,
        'max_effective_staleness': 128,
        # stability layer (kept in narrow, weakly changing ranges)
        'server_lr': cfg.get('server_lr', 0.2),
        'gradient_clip': cfg.get('gradient_clip', 2.0),
        'momentum': cfg.get('momentum', 0.85),
    })
    return cfg


def generate_scale_sweep_configs(
    client_sizes,
    scenario: str = 'balanced',
    compression=None,
    staleness_mode: str = 'quadratic',
):
    """Build configs for cross-scale sweeps (n=10..10000 etc.)."""
    out = {}
    for n in client_sizes:
        cfg = get_scaled_improved_config(
            num_clients=int(n),
            scenario=scenario,
            compression=compression,
            auto_scale_params=True,
        )
        cfg['staleness_mode'] = staleness_mode
        out[f"n{int(n)}_{scenario}_{compression or 'none'}_{staleness_mode}"] = cfg
    return out


def quick_test_improved_config():
    """Quick test of generated configs"""
    from federated_protocol_framework import create_protocol
    from unified_protocol_comparison import (
        SimpleNN, generate_federated_data,
        train_client, evaluate_model, ClientUpdate
    )
    import torch
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("TESTING IMPROVED CONFIGURATIONS")
    print("=" * 70)

    # Generate test data
    client_datasets, test_dataset = generate_federated_data(
        num_clients=50,
        samples_per_client=500,
        input_dim=12,
        num_classes=3,
        heterogeneity=0.5,
        dataset_name='tartanair',
        tartanair_root='tartanair-test-mono-release/mono',
        image_size=32,
    )

    model_config = {
        'input_dim': 8,
        'hidden_dim': 16,
        'output_dim': 2,
        'image_size': 32,
        'image_channels': 1,
        'backbone': 'resnet18',
    }

    all_configs = generate_all_configs()

    for name, config in all_configs.items():
        print(f"\nTesting config: {name}")

        protocol = create_protocol('improved_async', num_clients=3, **config)
        initial_model = SimpleNN(**model_config)
        protocol.set_global_model(initial_model.state_dict())

        # Train briefly
        for round_num in range(5):
            for client_id in range(3):
                global_state, pulled_version = protocol.get_global_model_with_version()
                if global_state is None:
                    continue
                local_model = SimpleNN(**model_config)
                local_model.load_state_dict(global_state)

                updated_state, loss, data_size, _ = train_client(
                    local_model, client_datasets[client_id], epochs=1, lr=0.01
                )

                update_dict = {}
                for param_name, param in updated_state.items():
                    if param_name in global_state and 'num_batches_tracked' not in param_name:
                        param_update = param.clone().float()
                        global_param = global_state[param_name].clone().float()
                        update_dict[param_name] = param_update - global_param

                update = ClientUpdate(
                    client_id=f"client_{client_id}",
                    update_data=update_dict,
                    model_version=int(pulled_version),
                    local_loss=loss,
                    data_size=data_size,
                    timestamp=0.0
                )
                protocol.receive_update(update)

        # Evaluate
        final_model_state = protocol.get_global_model()
        if final_model_state:
            eval_model = SimpleNN(**model_config)
            eval_model.load_state_dict(final_model_state)
            accuracy, loss = evaluate_model(eval_model, test_dataset)

            metrics = protocol.metrics.get_summary()
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Communication: {metrics['total_data_transmitted_mb']:.3f} MB")
            print(f"  Aggregations: {metrics['aggregations_performed']}")

        protocol.shutdown()


if __name__ == "__main__":
    quick_test_improved_config()
