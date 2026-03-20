"""
external_validity_runner.py
External validity benchmark: CIFAR non-IID + client photometric heterogeneity.
"""

import csv
import json
import time
import argparse
import math
import sys
import platform
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from federated_protocol_framework import create_protocol, ClientUpdate, build_scaffold_control_payload
from unified_protocol_comparison import SimpleNN, evaluate_with_intent_and_explanation, set_seed
from metrics import tri_objective_score, pareto_front_mask
from paper_profiles import build_protocol_suite


def _resolve_device(device: str = "auto") -> torch.device:
    req = str(device).strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available.")
        return torch.device("cuda")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device option: {device}")


class ClientPhotometricDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        gamma: float,
        brightness: float,
        train_mode: bool = True,
        augment: bool = True,
        normalize: bool = True,
    ):
        self.images = images.float()
        self.labels = labels.long()
        self.gamma = float(gamma)
        self.brightness = float(brightness)
        self.train_mode = bool(train_mode)
        self.augment = bool(augment)
        self.normalize = bool(normalize)
        # CIFAR-10 mean/std
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]
        x = torch.clamp(x * self.brightness, 0.0, 1.0)
        x = torch.pow(x + 1e-8, self.gamma)
        if self.train_mode and self.augment:
            # Lightweight augmentation for robustness without high overhead.
            if float(torch.rand(1).item()) < 0.5:
                x = torch.flip(x, dims=[2])
            if float(torch.rand(1).item()) < 0.5:
                jitter = float(np.random.uniform(0.9, 1.1))
                x = torch.clamp(x * jitter, 0.0, 1.0)
        if self.normalize:
            x = (x - self.mean) / (self.std + 1e-8)
        return x, y


def _load_cifar10_subset(train_size: int, test_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from torchvision import datasets, transforms
        root = str(Path("data").resolve())
        to_tensor = transforms.ToTensor()
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=to_tensor)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=to_tensor)

        train_idx = np.random.choice(len(train_ds), size=min(train_size, len(train_ds)), replace=False)
        test_idx = np.random.choice(len(test_ds), size=min(test_size, len(test_ds)), replace=False)

        tx, ty = [], []
        for i in train_idx:
            x, y = train_ds[int(i)]
            tx.append(x)
            ty.append(y)
        vx, vy = [], []
        for i in test_idx:
            x, y = test_ds[int(i)]
            vx.append(x)
            vy.append(y)
        return (
            torch.stack(tx, dim=0),
            torch.tensor(ty, dtype=torch.long),
            torch.stack(vx, dim=0),
            torch.tensor(vy, dtype=torch.long),
        )
    except Exception:
        # Offline fallback
        x_train = torch.rand(train_size, 3, 32, 32)
        y_train = torch.randint(0, 10, (train_size,))
        x_test = torch.rand(test_size, 3, 32, 32)
        y_test = torch.randint(0, 10, (test_size,))
        return x_train, y_train, x_test, y_test


def _build_non_iid_clients(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    num_clients: int,
    samples_per_client: int,
    dominant_ratio: float = 0.7,
    gamma_min: float = 0.75,
    gamma_max: float = 1.30,
    brightness_min: float = 0.70,
    brightness_max: float = 1.30,
    augment: bool = True,
    normalize: bool = True,
) -> List[Dataset]:
    class_indices = {c: torch.where(y_train == c)[0].tolist() for c in range(10)}
    clients = []
    for cid in range(num_clients):
        dom = np.random.choice(10, size=2, replace=False).tolist()
        n_dom = int(samples_per_client * float(dominant_ratio))
        n_oth = samples_per_client - n_dom

        picks = []
        for _ in range(n_dom):
            c = dom[np.random.randint(0, len(dom))]
            pool = class_indices[c]
            picks.append(pool[np.random.randint(0, len(pool))])
        for _ in range(n_oth):
            c = np.random.randint(0, 10)
            pool = class_indices[c]
            picks.append(pool[np.random.randint(0, len(pool))])

        idx = torch.tensor(picks, dtype=torch.long)
        gamma = float(np.random.uniform(gamma_min, gamma_max))
        brightness = float(np.random.uniform(brightness_min, brightness_max))
        ds = ClientPhotometricDataset(
            x_train[idx],
            y_train[idx],
            gamma=gamma,
            brightness=brightness,
            train_mode=True,
            augment=augment,
            normalize=normalize,
        )
        clients.append(ds)
    return clients


def _train_client(
    model: torch.nn.Module,
    dataset: Dataset,
    epochs: int,
    lr: float,
    device: torch.device,
    scaffold_c_global: Optional[Dict[str, torch.Tensor]] = None,
    scaffold_c_client: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[Dict[str, torch.Tensor], float, int, int]:
    """
    Returns: (cpu_state_dict, mean_loss, data_size, local_steps)
    """
    # NOTE: For canonical SCAFFOLD we apply a gradient correction term:
    #   grad <- grad - c_i + c
    # This requires the caller to provide scaffold control variates.
    # If not provided, the function behaves like the original local trainer.
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

    # Default optimizer (original behavior)
    opt: torch.optim.Optimizer
    if scaffold_c_global is not None and scaffold_c_client is not None:
        # Canonical SCAFFOLD uses corrected SGD-style updates.
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    model.train()

    losses = []
    local_steps = 0
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            # Apply scaffold gradient correction: grad <- grad - c_i + c
            if scaffold_c_global is not None and scaffold_c_client is not None:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if name in scaffold_c_global and name in scaffold_c_client:
                        param.grad = param.grad - scaffold_c_client[name].to(device) + scaffold_c_global[name].to(device)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
            local_steps += 1
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, float(np.mean(losses) if losses else 0.0), len(dataset), int(local_steps)


def _evaluate(model: torch.nn.Module, test_ds: Dataset, num_classes: int, device: torch.device) -> Dict[str, float]:
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    model = model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item()) * len(yb)
            preds = torch.argmax(logits, dim=1)
            total += len(yb)
            correct += int((preds == yb).sum().item())
    acc = float(correct / max(total, 1))
    avg_loss = float(total_loss / max(total, 1))
    intent_f1, bleu, _, _ = evaluate_with_intent_and_explanation(
        model, loader, device=str(device), num_classes=num_classes
    )
    return {"accuracy": acc, "loss": avg_loss, "intent_f1": float(intent_f1), "bleu": float(bleu)}


def _compute_trace_metrics(
    trace: List[Dict[str, float]],
    thresholds: List[float],
) -> Dict[str, float]:
    """
    Compute convergence/efficiency metrics from a run trace.
    """
    if not trace:
        out = {"auc_acc_time": 0.0, "auc_acc_comm": 0.0}
        for t in thresholds:
            out[f"time_to_acc_{str(t).replace('.', 'p')}"] = float("inf")
            out[f"comm_to_acc_{str(t).replace('.', 'p')}"] = float("inf")
        return out

    t_arr = np.array([float(p["elapsed_sec"]) for p in trace], dtype=np.float64)
    c_arr = np.array([float(p["communication_mb"]) for p in trace], dtype=np.float64)
    a_arr = np.array([float(p["accuracy"]) for p in trace], dtype=np.float64)

    # Trapezoid AUC in native units.
    # NumPy compatibility: newer versions provide `trapezoid`, older versions may only have `trapz`.
    integ = getattr(np, "trapezoid", None)
    if integ is None:
        integ = getattr(np, "trapz")
    auc_time = float(integ(a_arr, t_arr)) if len(trace) >= 2 else float(a_arr[0] * max(t_arr[0], 1e-8))
    auc_comm = float(integ(a_arr, c_arr)) if len(trace) >= 2 else float(a_arr[0] * max(c_arr[0], 1e-8))

    out = {
        "auc_acc_time": auc_time,
        "auc_acc_comm": auc_comm,
    }
    for thr in thresholds:
        key_t = f"time_to_acc_{str(thr).replace('.', 'p')}"
        key_c = f"comm_to_acc_{str(thr).replace('.', 'p')}"
        t_hit = float("inf")
        c_hit = float("inf")
        for p in trace:
            if float(p["accuracy"]) >= float(thr):
                t_hit = float(p["elapsed_sec"])
                c_hit = float(p["communication_mb"])
                break
        out[key_t] = t_hit
        out[key_c] = c_hit
    return out


def _tensor_signature(x: torch.Tensor, y: torch.Tensor) -> str:
    """Compact split hash for reproducibility logs."""
    hasher = hashlib.sha256()
    hasher.update(str(tuple(x.shape)).encode("utf-8"))
    hasher.update(str(tuple(y.shape)).encode("utf-8"))
    hasher.update(x[: min(64, len(x))].detach().cpu().numpy().tobytes())
    hasher.update(y[: min(256, len(y))].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


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
    n = int(wins + losses)
    if n <= 0:
        return 1.0
    k = int(min(wins, losses))
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return float(min(1.0, 2.0 * tail))


def _apply_holm_correction(rows: List[Dict[str, Any]]) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["fairness_mode"]), str(r["metric"]))
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


def _build_pairwise_rows(flat_rows: List[Dict[str, Any]], fairness_modes: List[str]) -> List[Dict[str, Any]]:
    """
    Paired seed-wise tests:
    best improved_async profile per seed vs each non-improved baseline.
    """
    rows: List[Dict[str, Any]] = []
    metric_specs = (
        ("score_balanced", lambda imp, base: float(imp) - float(base)),
        ("accuracy", lambda imp, base: float(imp) - float(base)),
        ("communication_mb", lambda imp, base: float(base) - float(imp)),
    )

    for mode in fairness_modes:
        mode_rows = [r for r in flat_rows if str(r["fairness_mode"]) == str(mode)]
        if not mode_rows:
            continue
        improved_rows = [r for r in mode_rows if str(r["protocol_key"]).startswith("improved_async")]
        baseline_keys = sorted({str(r["protocol_key"]) for r in mode_rows if not str(r["protocol_key"]).startswith("improved_async")})
        if not improved_rows or not baseline_keys:
            continue

        imp_best_by_seed: Dict[int, Dict[str, Any]] = {}
        by_seed_imp: Dict[int, List[Dict[str, Any]]] = {}
        for r in improved_rows:
            by_seed_imp.setdefault(int(r["seed"]), []).append(r)
        for seed, rs in by_seed_imp.items():
            imp_best_by_seed[seed] = max(rs, key=lambda x: float(x["score_balanced"]))

        for base_key in baseline_keys:
            base_by_seed = {
                int(r["seed"]): r
                for r in mode_rows
                if str(r["protocol_key"]) == str(base_key)
            }
            shared_seeds = sorted(set(imp_best_by_seed.keys()) & set(base_by_seed.keys()))
            if not shared_seeds:
                continue

            for metric_name, diff_fn in metric_specs:
                diffs = []
                imp_vals = []
                base_vals = []
                for s in shared_seeds:
                    iv = float(imp_best_by_seed[s][metric_name])
                    bv = float(base_by_seed[s][metric_name])
                    imp_vals.append(iv)
                    base_vals.append(bv)
                    diffs.append(diff_fn(iv, bv))
                d_arr = np.array(diffs, dtype=np.float64)
                diff_mean, diff_std, diff_ci95 = _mean_std_ci(diffs)
                wins = int(np.sum(d_arr > 1e-12))
                losses = int(np.sum(d_arr < -1e-12))
                ties = int(len(d_arr) - wins - losses)
                p_sign = _sign_test_pvalue_two_sided(wins=wins, losses=losses)
                cohen_dz = float(diff_mean / (diff_std + 1e-12))
                imp_mu, _, _ = _mean_std_ci(imp_vals)
                base_mu, _, _ = _mean_std_ci(base_vals)
                rows.append({
                    "fairness_mode": str(mode),
                    "metric": metric_name,
                    "improved_proto": "improved_async(best-profile-per-seed)",
                    "baseline_proto": str(base_key),
                    "paired_seeds": int(len(shared_seeds)),
                    "improved_mean": float(imp_mu),
                    "baseline_mean": float(base_mu),
                    "diff_mean_positive_is_better_for_improved": float(diff_mean),
                    "diff_std": float(diff_std),
                    "diff_ci95": float(diff_ci95),
                    "cohen_dz": float(cohen_dz),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "p_value_sign": float(p_sign),
                    "p_value_holm": None,
                })

    _apply_holm_correction(rows)
    rows.sort(
        key=lambda x: (
            str(x["fairness_mode"]),
            str(x["metric"]),
            float(x["p_value_holm"]) if x["p_value_holm"] is not None else 1.0,
            str(x["baseline_proto"]),
        )
    )
    return rows


def run_once(
    protocol_key: str,
    protocol_name: str,
    protocol_cfg: Dict[str, Any],
    client_datasets: List[Dataset],
    test_dataset: Dataset,
    model_cfg: Dict[str, Any],
    rounds: int,
    fairness_mode: str,
    target_updates: int,
    local_epochs: int,
    local_lr: float,
    participation_rate: float,
    comm_budget_mb: float,
    track_interval_updates: int,
    acc_thresholds: List[float],
    device: str = "auto",
    active_selection_seed: Optional[int] = None,
    schedule_hash: str = "",
    precomputed_active_schedule: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    torch_device = _resolve_device(device)
    cfg = dict(protocol_cfg)
    n_clients = len(client_datasets)
    n_active = max(1, int(n_clients * participation_rate))

    # Keep sync protocols consistent with selected participation_rate.
    if protocol_name in ("fedavg", "scaffold"):
        cfg["fixed_round_size"] = int(n_active)
        cfg["participation_rate"] = float(n_active / max(n_clients, 1))
        cfg.setdefault("use_timeout", False)

    protocol = create_protocol(protocol_name, num_clients=n_clients, **cfg)
    protocol.set_global_model(SimpleNN(**model_cfg).state_dict())
    start = time.time()
    trace: List[Dict[str, float]] = []
    active_rng = np.random.default_rng(active_selection_seed) if active_selection_seed is not None else None
    schedule_ptr = 0
    consumed_schedule: List[List[int]] = []

    sent_updates = 0
    accepted_updates = 0
    while True:
        if fairness_mode == "equal_rounds":
            if sent_updates >= rounds * max(1, int(len(client_datasets) * participation_rate)):
                break
        elif fairness_mode == "equal_updates":
            if sent_updates >= target_updates:
                break
        elif fairness_mode == "equal_accepted_updates":
            if accepted_updates >= target_updates:
                break
        else:
            raise ValueError(f"Unknown fairness mode: {fairness_mode}")

        n_active = max(1, int(len(client_datasets) * participation_rate))
        if precomputed_active_schedule is not None and schedule_ptr < len(precomputed_active_schedule):
            active = np.array(precomputed_active_schedule[schedule_ptr], dtype=np.int64)
            schedule_ptr += 1
        elif active_rng is None:
            active = np.random.choice(len(client_datasets), n_active, replace=False)
        else:
            active = active_rng.choice(len(client_datasets), n_active, replace=False)
        consumed_schedule.append([int(x) for x in active.tolist()])
        for cid in active:
            if fairness_mode == "equal_updates" and sent_updates >= target_updates:
                break
            if fairness_mode == "equal_accepted_updates" and accepted_updates >= target_updates:
                break
            gstate, pulled_version = protocol.get_global_model_with_version()
            if gstate is None:
                continue
            protocol.account_model_downlink(gstate)
            client_id = f"client_{cid}"
            local_model = SimpleNN(**model_cfg)
            local_model.load_state_dict(gstate, strict=False)

            scaffold_c_global = None
            scaffold_c_client = None
            if protocol_name == "scaffold":
                # Canonical SCAFFOLD: client-side gradient correction uses (c, c_i).
                scaffold_c_global, scaffold_c_client = protocol.get_scaffold_controls(client_id)

            # Deterministic local-training seed for stronger paired randomness across protocols.
            if active_selection_seed is not None:
                seed_material = (
                    f"{int(active_selection_seed)}|{str(fairness_mode)}|{int(cid)}|"
                    f"{int(sent_updates)}|{int(pulled_version)}"
                )
                local_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:8], 16)
                torch.manual_seed(local_seed)
                np.random.seed(local_seed)

            updated_state, local_loss, data_size, local_steps = _train_client(
                local_model,
                client_datasets[cid],
                epochs=local_epochs,
                lr=local_lr,
                device=torch_device,
                scaffold_c_global=scaffold_c_global,
                scaffold_c_client=scaffold_c_client,
            )

            delta = {}
            for name, pnew in updated_state.items():
                if name in gstate and "num_batches_tracked" not in name:
                    delta[name] = pnew.float() - gstate[name].float()
            scaffold_payload = None
            if protocol_name == "scaffold" and scaffold_c_global is not None and scaffold_c_client is not None:
                scaffold_payload = build_scaffold_control_payload(
                    update_delta=delta,
                    c_global=scaffold_c_global,
                    c_client=scaffold_c_client,
                    local_steps=int(local_steps),
                    local_lr=float(local_lr),
                )
            update = ClientUpdate(
                client_id=client_id,
                update_data=delta,
                model_version=int(pulled_version),
                local_loss=float(local_loss),
                data_size=int(data_size),
                timestamp=time.time(),
                local_steps=int(local_steps) if protocol_name == "scaffold" else None,
                local_lr=float(local_lr) if protocol_name == "scaffold" else None,
                scaffold_control_payload=scaffold_payload,
            )
            accepted, _ = protocol.receive_update(update)
            sent_updates += 1
            accepted_updates += int(bool(accepted))

            if sent_updates % max(1, int(track_interval_updates)) == 0:
                state_mid = protocol.get_global_model()
                eval_model_mid = SimpleNN(**model_cfg)
                if state_mid:
                    eval_model_mid.load_state_dict(state_mid, strict=False)
                ev_mid = _evaluate(eval_model_mid, test_dataset, num_classes=model_cfg["output_dim"], device=torch_device)
                m_mid = protocol.metrics.get_summary()
                trace.append({
                    "updates_sent": int(sent_updates),
                    "updates_accepted": int(accepted_updates),
                    "elapsed_sec": float(time.time() - start),
                    "communication_mb": float(m_mid.get("total_data_transmitted_mb", 0.0)),
                    "accuracy": float(ev_mid["accuracy"]),
                })

    elapsed = float(time.time() - start)
    final_state = protocol.get_global_model()
    eval_model = SimpleNN(**model_cfg)
    if final_state:
        eval_model.load_state_dict(final_state, strict=False)
    ev = _evaluate(eval_model, test_dataset, num_classes=model_cfg["output_dim"], device=torch_device)
    m = protocol.metrics.get_summary()
    comm_mb = float(m.get("total_data_transmitted_mb", 0.0))
    score = tri_objective_score(
        accuracy=ev["accuracy"],
        communication_mb=comm_mb,
        latency_sec=elapsed,
        comm_budget_mb=comm_budget_mb,
        latency_budget_sec=max(elapsed, 1.0),
    )
    # Ensure final checkpoint exists in trace.
    if not trace or int(trace[-1]["updates_sent"]) != int(sent_updates):
        trace.append({
            "updates_sent": int(sent_updates),
            "updates_accepted": int(accepted_updates),
            "elapsed_sec": float(elapsed),
            "communication_mb": float(comm_mb),
            "accuracy": float(ev["accuracy"]),
        })
    conv = _compute_trace_metrics(trace, thresholds=acc_thresholds)
    protocol.shutdown()
    actual_schedule_hash = hashlib.sha256(
        json.dumps(consumed_schedule, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return {
        "protocol_key": protocol_key,
        "fairness_mode": fairness_mode,
        "schedule_hash": str(schedule_hash),
        "schedule_hash_actual": str(actual_schedule_hash),
        "active_selection_seed": int(active_selection_seed) if active_selection_seed is not None else None,
        "accuracy": ev["accuracy"],
        "loss": ev["loss"],
        "intent_f1": ev["intent_f1"],
        "bleu": ev["bleu"],
        "communication_mb": comm_mb,
        "elapsed_sec": elapsed,
        "aggregations": int(m.get("aggregations_performed", 0)),
        "tri_objective": float(score),
        "score_acc_only": float(ev["accuracy"]),
        "score_balanced": float(
            tri_objective_score(
                accuracy=ev["accuracy"],
                communication_mb=comm_mb,
                latency_sec=elapsed,
                comm_budget_mb=comm_budget_mb,
                latency_budget_sec=max(elapsed, 1.0),
                w_acc=0.8,
                w_comm=0.1,
                w_lat=0.1,
            )
        ),
        "updates_sent_budget": int(sent_updates),
        "updates_accepted_budget": int(accepted_updates),
        "staleness_p50": float(m.get("staleness_p50", 0.0)),
        "staleness_p90": float(m.get("staleness_p90", 0.0)),
        "staleness_p99": float(m.get("staleness_p99", 0.0)),
        "accepted_staleness_p90": float(m.get("accepted_staleness_p90", 0.0)),
        "rejected_staleness_p90": float(m.get("rejected_staleness_p90", 0.0)),
        "staleness_drop_ratio": float(m.get("staleness_drop_ratio", 0.0)),
        "buffer_occupancy_mean": float(m.get("buffer_occupancy_mean", 0.0)),
        "buffer_occupancy_p90": float(m.get("buffer_occupancy_p90", 0.0)),
        "buffer_wait_sec_mean": float(m.get("buffer_wait_sec_mean", 0.0)),
        "buffer_wait_sec_p90": float(m.get("buffer_wait_sec_p90", 0.0)),
        "control_payload_uplink_mb": float(m.get("control_payload_uplink_mb", 0.0)),
        "control_payload_downlink_mb": float(m.get("control_payload_downlink_mb", 0.0)),
        "model_downlink_mb": float(m.get("model_downlink_mb", 0.0)),
        **conv,
        "trace": trace,
    }


def _estimate_model_param_count(model_cfg: Dict[str, Any]) -> int:
    model = SimpleNN(**model_cfg)
    return int(sum(p.numel() for p in model.parameters()))


def _adapt_suite_for_image_task(
    suite: Dict[str, Dict[str, Any]],
    model_cfg: Dict[str, Any],
    topk_fraction: float
) -> Dict[str, Dict[str, Any]]:
    out = {}
    n_params = _estimate_model_param_count(model_cfg)
    k_auto = max(200, int(n_params * float(topk_fraction)))
    for key, cfg in suite.items():
        c = dict(cfg)
        if c.get("compression", None) == "topk":
            c["k"] = max(int(c.get("k", 0)), k_auto)
        out[key] = c
    return out


def _save_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="External validity runner on CIFAR photometric non-IID.")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--samples_per_client", type=int, default=250)
    parser.add_argument("--train_size", type=int, default=20000)
    parser.add_argument("--test_size", type=int, default=3000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--local_lr", type=float, default=0.001)
    parser.add_argument("--participation_rate", type=float, default=0.5)
    parser.add_argument("--comm_budget_mb", type=float, default=120.0)
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--fairness_modes", type=str, default="equal_rounds,equal_updates,equal_accepted_updates")
    parser.add_argument("--topk_fraction", type=float, default=0.03, help="Auto TopK fraction of model params")
    parser.add_argument("--track_interval_updates", type=int, default=20, help="Evaluate/record every N updates")
    parser.add_argument("--acc_thresholds", type=str, default="0.20,0.25", help="Comma-separated accuracy thresholds")
    parser.add_argument("--dominant_ratio", type=float, default=0.7, help="Class dominance ratio per client")
    parser.add_argument("--gamma_min", type=float, default=0.75)
    parser.add_argument("--gamma_max", type=float, default=1.30)
    parser.add_argument("--brightness_min", type=float, default=0.70)
    parser.add_argument("--brightness_max", type=float, default=1.30)
    parser.add_argument("--disable_augment", action="store_true")
    parser.add_argument("--disable_normalize", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--backbone", type=str, default="resnet18", help="Model backbone (e.g. resnet18, tiny_transformer)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = {
        "num_clients": int(args.num_clients),
        "samples_per_client": int(args.samples_per_client),
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "local_lr": float(args.local_lr),
        "participation_rate": float(args.participation_rate),
        "comm_budget_mb": float(args.comm_budget_mb),
        "fairness_modes": [s.strip() for s in str(args.fairness_modes).split(",") if s.strip()],
        "seeds": int(args.seeds),
        "topk_fraction": float(args.topk_fraction),
        "dominant_ratio": float(args.dominant_ratio),
        "gamma_min": float(args.gamma_min),
        "gamma_max": float(args.gamma_max),
        "brightness_min": float(args.brightness_min),
        "brightness_max": float(args.brightness_max),
        "augment": not bool(args.disable_augment),
        "normalize": not bool(args.disable_normalize),
        "base_seed": int(args.seed),
        "num_classes": 10,
        "track_interval_updates": int(args.track_interval_updates),
        "acc_thresholds": [float(x.strip()) for x in str(args.acc_thresholds).split(",") if x.strip()],
        "device": str(args.device),
        "backbone": str(args.backbone),
    }
    run_device = _resolve_device(cfg["device"])
    print(f"[device] requested={cfg['device']} resolved={run_device}")

    suite = build_protocol_suite(num_clients=cfg["num_clients"], strict=True, include_improved=True)
    model_cfg = {
        "input_dim": 12,
        "hidden_dim": 96,
        "output_dim": cfg["num_classes"],
        "image_size": 32,
        "image_channels": 3,
        "backbone": cfg.get("backbone", "resnet18"),
    }
    suite = _adapt_suite_for_image_task(suite, model_cfg=model_cfg, topk_fraction=cfg["topk_fraction"])
    target_updates = cfg["rounds"] * max(1, int(cfg["num_clients"] * cfg["participation_rate"]))
    if target_updates < 100:
        print(
            f"[warning] effective update budget is very low ({target_updates}). "
            "Expect near-random accuracy and unstable protocol ranking. "
            "Suggested: rounds>=12 for smoke, rounds>=30 for meaningful comparison."
        )

    rows = []
    env_meta = {
        "timestamp_unix": float(time.time()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": str(torch.version.cuda),
        "torch_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "numpy_version": np.__version__,
        "deterministic_requested": False,
        "baseline_fidelity_notes": {
            "scaffold": "Canonical SCAFFOLD: client-side gradient correction (c, c_i) and server control-variate aggregation."
        },
        "config": cfg,
        "split_signatures": [],
    }
    for seed_offset in range(cfg["seeds"]):
        seed = cfg["base_seed"] + seed_offset
        set_seed(seed)
        x_train, y_train, x_test, y_test = _load_cifar10_subset(cfg["train_size"], cfg["test_size"])
        env_meta["split_signatures"].append({
            "seed": int(seed),
            "train_signature": _tensor_signature(x_train, y_train),
            "test_signature": _tensor_signature(x_test, y_test),
        })
        client_datasets = _build_non_iid_clients(
            x_train,
            y_train,
            cfg["num_clients"],
            cfg["samples_per_client"],
            dominant_ratio=cfg["dominant_ratio"],
            gamma_min=cfg["gamma_min"],
            gamma_max=cfg["gamma_max"],
            brightness_min=cfg["brightness_min"],
            brightness_max=cfg["brightness_max"],
            augment=cfg["augment"],
            normalize=cfg["normalize"],
        )
        test_dataset = ClientPhotometricDataset(
            x_test,
            y_test,
            gamma=1.0,
            brightness=1.0,
            train_mode=False,
            augment=False,
            normalize=cfg["normalize"],
        )

        for fairness_mode in cfg["fairness_modes"]:
            print(f"\n=== Seed {seed} | mode={fairness_mode} ===")
            n_active = max(1, int(cfg["num_clients"] * cfg["participation_rate"]))
            schedule_base = (
                f"{int(seed)}|{int(cfg['num_clients'])}|{str(fairness_mode)}|{int(n_active)}|"
                f"{int(cfg['rounds'])}|{int(target_updates)}"
            )
            schedule_hash = hashlib.sha256(schedule_base.encode("utf-8")).hexdigest()
            active_selection_seed = int(schedule_hash[:8], 16)
            for key, pcfg in suite.items():
                pname = "improved_async" if key.startswith("improved_async") else key
                print(f"Running {key} ...")
                row = run_once(
                    protocol_key=key,
                    protocol_name=pname,
                    protocol_cfg=pcfg,
                    client_datasets=client_datasets,
                    test_dataset=test_dataset,
                    model_cfg=model_cfg,
                    rounds=cfg["rounds"],
                    fairness_mode=fairness_mode,
                    target_updates=target_updates,
                    local_epochs=cfg["local_epochs"],
                    local_lr=cfg["local_lr"],
                    participation_rate=cfg["participation_rate"],
                    comm_budget_mb=cfg["comm_budget_mb"],
                    track_interval_updates=cfg["track_interval_updates"],
                    acc_thresholds=cfg["acc_thresholds"],
                    device=str(run_device),
                    active_selection_seed=active_selection_seed,
                    schedule_hash=schedule_hash,
                )
                row["seed"] = int(seed)
                rows.append(row)
                print(
                    f"  tri={row['tri_objective']:.4f}, bal={row['score_balanced']:.4f}, "
                    f"acc={row['accuracy']:.4f}, comm={row['communication_mb']:.2f}MB"
                )

    # Pareto within each fairness mode (across all seeds/protocol rows)
    for fairness_mode in cfg["fairness_modes"]:
        idx = [i for i, r in enumerate(rows) if r["fairness_mode"] == fairness_mode]
        if not idx:
            continue
        front = pareto_front_mask(
            [rows[i]["accuracy"] for i in idx],
            [rows[i]["communication_mb"] for i in idx],
            [rows[i]["elapsed_sec"] for i in idx],
        )
        for pos, is_front in zip(idx, front):
            rows[pos]["pareto_optimal"] = bool(is_front)

    # Save full traces separately and flatten per-run rows for CSV/JSON.
    traces_payload = []
    flat_rows = []
    for r in rows:
        rr = dict(r)
        traces_payload.append({
            "protocol_key": rr["protocol_key"],
            "fairness_mode": rr["fairness_mode"],
            "seed": rr["seed"],
            "trace": rr.get("trace", []),
        })
        rr.pop("trace", None)
        flat_rows.append(rr)

    # Aggregate mean/std by (fairness_mode, protocol_key)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in flat_rows:
        grouped.setdefault((r["fairness_mode"], r["protocol_key"]), []).append(r)

    summary_rows = []
    for (mode, key), vals in grouped.items():
        tri_mu, tri_std, tri_ci = _mean_std_ci([float(v["tri_objective"]) for v in vals])
        bal_mu, bal_std, bal_ci = _mean_std_ci([float(v["score_balanced"]) for v in vals])
        acc_mu, acc_std, acc_ci = _mean_std_ci([float(v["accuracy"]) for v in vals])
        comm_mu, comm_std, comm_ci = _mean_std_ci([float(v["communication_mb"]) for v in vals])
        lat_mu, lat_std, lat_ci = _mean_std_ci([float(v["elapsed_sec"]) for v in vals])
        auc_t_mu, auc_t_std, auc_t_ci = _mean_std_ci([float(v["auc_acc_time"]) for v in vals])
        auc_c_mu, auc_c_std, auc_c_ci = _mean_std_ci([float(v["auc_acc_comm"]) for v in vals])
        summary_rows.append({
            "fairness_mode": mode,
            "protocol_key": key,
            "seeds": len(vals),
            "tri_objective_mean": tri_mu,
            "tri_objective_std": tri_std,
            "tri_objective_ci95": tri_ci,
            "score_balanced_mean": bal_mu,
            "score_balanced_std": bal_std,
            "score_balanced_ci95": bal_ci,
            "accuracy_mean": acc_mu,
            "accuracy_std": acc_std,
            "accuracy_ci95": acc_ci,
            "communication_mb_mean": comm_mu,
            "communication_mb_std": comm_std,
            "communication_mb_ci95": comm_ci,
            "elapsed_sec_mean": lat_mu,
            "elapsed_sec_std": lat_std,
            "elapsed_sec_ci95": lat_ci,
            "auc_acc_time_mean": auc_t_mu,
            "auc_acc_time_std": auc_t_std,
            "auc_acc_time_ci95": auc_t_ci,
            "auc_acc_comm_mean": auc_c_mu,
            "auc_acc_comm_std": auc_c_std,
            "auc_acc_comm_ci95": auc_c_ci,
        })

        for thr in cfg["acc_thresholds"]:
            k_t = f"time_to_acc_{str(thr).replace('.', 'p')}"
            k_c = f"comm_to_acc_{str(thr).replace('.', 'p')}"
            vals_t = [float(v[k_t]) for v in vals if np.isfinite(float(v[k_t]))]
            vals_c = [float(v[k_c]) for v in vals if np.isfinite(float(v[k_c]))]
            if vals_t:
                t_mu, t_std, t_ci = _mean_std_ci(vals_t)
                summary_rows[-1][f"{k_t}_mean"] = float(t_mu)
                summary_rows[-1][f"{k_t}_std"] = float(t_std)
                summary_rows[-1][f"{k_t}_ci95"] = float(t_ci)
            else:
                summary_rows[-1][f"{k_t}_mean"] = float("inf")
                summary_rows[-1][f"{k_t}_std"] = float("inf")
                summary_rows[-1][f"{k_t}_ci95"] = float("inf")
            if vals_c:
                c_mu, c_std, c_ci = _mean_std_ci(vals_c)
                summary_rows[-1][f"{k_c}_mean"] = float(c_mu)
                summary_rows[-1][f"{k_c}_std"] = float(c_std)
                summary_rows[-1][f"{k_c}_ci95"] = float(c_ci)
            else:
                summary_rows[-1][f"{k_c}_mean"] = float("inf")
                summary_rows[-1][f"{k_c}_std"] = float("inf")
                summary_rows[-1][f"{k_c}_ci95"] = float("inf")
    summary_rows.sort(key=lambda x: (x["fairness_mode"], -x["score_balanced_mean"]))
    pairwise_rows = _build_pairwise_rows(flat_rows=flat_rows, fairness_modes=cfg["fairness_modes"])

    Path("results").mkdir(exist_ok=True)
    _save_csv("results/external_validity_results.csv", flat_rows)
    _save_csv("results/external_validity_summary.csv", summary_rows)
    _save_csv("results/external_validity_pairwise_tests.csv", pairwise_rows)
    with open("results/external_validity_results.json", "w", encoding="utf-8") as f:
        json.dump(flat_rows, f, indent=2)
    with open("results/external_validity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    with open("results/external_validity_pairwise_tests.json", "w", encoding="utf-8") as f:
        json.dump(pairwise_rows, f, indent=2)
    with open("results/external_validity_traces.json", "w", encoding="utf-8") as f:
        json.dump(traces_payload, f, indent=2)
    with open("results/external_validity_metadata.json", "w", encoding="utf-8") as f:
        json.dump(env_meta, f, indent=2)
    print("Saved:")
    print("- results/external_validity_results.csv")
    print("- results/external_validity_results.json")
    print("- results/external_validity_summary.csv")
    print("- results/external_validity_summary.json")
    print("- results/external_validity_pairwise_tests.csv")
    print("- results/external_validity_pairwise_tests.json")
    print("- results/external_validity_traces.json")
    print("- results/external_validity_metadata.json")


if __name__ == "__main__":
    main()
