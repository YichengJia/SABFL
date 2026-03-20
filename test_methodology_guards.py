"""
Lightweight methodology guard tests.

Run with:
  python -m unittest test_methodology_guards.py
"""

import unittest
import torch
from torch.utils.data import TensorDataset

from federated_protocol_framework import create_protocol
from external_validity_runner import run_once


class MethodologyGuardsTest(unittest.TestCase):
    def test_model_downlink_accounting_increments(self):
        protocol = create_protocol("fedavg", num_clients=2, strict_reproduction=True)
        model_state = {"w": torch.zeros(16, dtype=torch.float32)}
        protocol.set_global_model(model_state)
        pulled, _ = protocol.get_global_model_with_version()
        protocol.account_model_downlink(pulled)
        summary = protocol.metrics.get_summary()
        self.assertGreater(float(summary.get("model_downlink_mb", 0.0)), 0.0)
        protocol.shutdown()

    def test_run_once_emits_schedule_and_downlink_fields(self):
        x = torch.rand(80, 12)
        y = torch.randint(0, 3, (80,))
        client_datasets = [TensorDataset(x[i * 20:(i + 1) * 20], y[i * 20:(i + 1) * 20]) for i in range(4)]
        test_dataset = TensorDataset(torch.rand(40, 12), torch.randint(0, 3, (40,)))
        model_cfg = {
            "input_dim": 12,
            "hidden_dim": 16,
            "output_dim": 3,
            "image_size": 32,
            "image_channels": 1,
            "backbone": "resnet18",
        }
        schedule = [[0, 1], [2, 3], [1, 2], [0, 3]]
        row = run_once(
            protocol_key="fedavg_guard",
            protocol_name="fedavg",
            protocol_cfg={"strict_reproduction": True},
            client_datasets=client_datasets,
            test_dataset=test_dataset,
            model_cfg=model_cfg,
            rounds=1,
            fairness_mode="equal_updates",
            target_updates=4,
            local_epochs=1,
            local_lr=0.01,
            participation_rate=0.5,
            comm_budget_mb=20.0,
            track_interval_updates=2,
            acc_thresholds=[0.2],
            device="cpu",
            active_selection_seed=123,
            schedule_hash="guard_hash",
            precomputed_active_schedule=schedule,
        )
        self.assertTrue(bool(row.get("schedule_hash_actual", "")))
        self.assertGreater(float(row.get("model_downlink_mb", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
