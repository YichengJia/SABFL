"""
unified_protocol_comparison.py
Unified comparison of multiple federated learning protocols
"""

import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple, Optional
import copy
from pathlib import Path
from PIL import Image
import math

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['axes.unicode_minus'] = False

from federated_protocol_framework import (
    create_protocol, ClientUpdate, FederatedProtocol, build_scaffold_control_payload
)
from optimized_protocol_config import generate_all_configs
from paper_profiles import build_protocol_suite
from metrics import tri_objective_score, within_budgets

import os, random, numpy as np, torch
import json


def resolve_device(device: str = "auto") -> torch.device:
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


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Make cuDNN deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# -----------------------------
# LLM-Style Model Definition
# -----------------------------
def _choose_num_heads(hidden_dim: int) -> int:
    for n in (8, 4, 2, 1):
        if hidden_dim % n == 0:
            return n
    return 1


class SimpleNN(nn.Module):
    """
    Default backbone: ResNet-18 (widely used, reproducible).
    Keep the legacy class name (`SimpleNN`) for compatibility with existing scripts.

    Backbones:
    - resnet18: torchvision ResNet-18 for image inputs
    - mobilenet_v3_small: lightweight torchvision MobileNetV3-Small for image inputs
    - tiny_transformer: legacy patch-transformer backbone (kept for ablations / continuity)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        image_size: int = 32,
        patch_size: int = 4,
        image_channels: int = 1,
        backbone: str = "resnet18",
    ):
        super(SimpleNN, self).__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.image_channels = int(image_channels)
        self.backbone = str(backbone).strip().lower()

        # Vector path (reproducible MLP; used for synthetic / non-image tasks).
        self.vector_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        # Image path
        self.image_model: Optional[nn.Module] = None
        if self.backbone == "tiny_transformer":
            patch_dim = self.patch_size * self.patch_size * self.image_channels
            self.patch_proj = nn.Linear(patch_dim, self.hidden_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
            num_patches = (self.image_size // self.patch_size) ** 2
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=_choose_num_heads(self.hidden_dim),
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.norm = nn.LayerNorm(self.hidden_dim)
            self.classifier = nn.Linear(self.hidden_dim, self.output_dim)
        else:
            try:
                from torchvision import models
            except Exception as e:
                raise RuntimeError("torchvision is required for image backbones") from e

            if self.backbone == "resnet18":
                m = models.resnet18(weights=None)
                if self.image_channels != 3:
                    m.conv1 = nn.Conv2d(self.image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                m.fc = nn.Linear(m.fc.in_features, self.output_dim)
                self.image_model = m
            elif self.backbone == "mobilenet_v3_small":
                m = models.mobilenet_v3_small(weights=None)
                if self.image_channels != 3:
                    stem = m.features[0][0]
                    m.features[0][0] = nn.Conv2d(
                        self.image_channels,
                        stem.out_channels,
                        kernel_size=stem.kernel_size,
                        stride=stem.stride,
                        padding=stem.padding,
                        bias=False,
                    )
                m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, self.output_dim)
                self.image_model = m
            else:
                raise ValueError(f"Unknown backbone: {self.backbone}")

    def _vector_tokens(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_vector_tokens only used in tiny_transformer mode")

    def _image_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # only used in tiny_transformer mode
        # Keep channel information for RGB datasets (e.g., CIFAR).
        if x.size(1) != self.image_channels:
            if x.size(1) == 1 and self.image_channels == 3:
                x = x.repeat(1, 3, 1, 1)
            elif x.size(1) > self.image_channels:
                x = x[:, :self.image_channels, :, :]
            else:
                pad = self.image_channels - x.size(1)
                x = torch.cat([x, x[:, :1, :, :].repeat(1, pad, 1, 1)], dim=1)
        x = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        bsz = x.size(0)
        patches = patches.contiguous().view(
            bsz, self.image_channels, -1, self.patch_size * self.patch_size
        ).permute(0, 2, 1, 3).reshape(bsz, -1, self.image_channels * self.patch_size * self.patch_size)
        return self.patch_proj(patches)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()

        if x.dim() == 2:
            return self.vector_mlp(x)
        if x.dim() == 4:
            if self.backbone == "tiny_transformer":
                tokens = self._image_tokens(x)
                cls = self.cls_token.expand(tokens.size(0), -1, -1)
                token_stream = torch.cat([cls, tokens], dim=1)
                token_stream = token_stream + self.pos_embed[:, :token_stream.size(1), :]
                encoded = self.encoder(token_stream)
                pooled = self.norm(encoded[:, 0, :])
                return self.classifier(pooled)
            return self.image_model(x)  # type: ignore[operator]

        x = x.view(x.size(0), -1)
        return self.vector_mlp(x)

    def explain(self, xb: torch.Tensor, preds: torch.Tensor) -> List[str]:
        with torch.no_grad():
            probs = torch.softmax(self.forward(xb), dim=1)
            conf = torch.gather(probs, 1, preds.view(-1, 1)).squeeze(1)
        outputs = []
        for p, c in zip(preds.detach().cpu().tolist(), conf.detach().cpu().tolist()):
            outputs.append(
                f"pred intent_{int(p)} based on transformer context aggregation, confidence={float(c):.2f}"
            )
        return outputs


# -----------------------------
# Data Functions
# -----------------------------
def _load_numeric_pose_rows(pose_path: Path) -> List[np.ndarray]:
    rows: List[np.ndarray] = []
    for raw in pose_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        s = s.replace(",", " ")
        parts = s.split()
        try:
            vals = np.array([float(x) for x in parts], dtype=np.float32)
        except ValueError:
            continue
        if vals.size >= 3:
            rows.append(vals)
    return rows


def _extract_positions_from_rows(rows: List[np.ndarray]) -> Optional[np.ndarray]:
    if not rows:
        return None
    positions = []
    for vals in rows:
        if vals.size == 8:
            # common format: t tx ty tz qx qy qz qw
            p = vals[1:4]
        elif vals.size in (7, 9):
            # common format: tx ty tz qx qy qz qw (+ optional extra)
            p = vals[0:3]
        else:
            p = vals[0:3]
        positions.append(p.astype(np.float32))
    out = np.stack(positions, axis=0)
    return out if len(out) >= 3 else None


def _find_scene_positions(scene_dir: Path) -> Optional[np.ndarray]:
    candidates = [
        "pose_left.txt",
        "pose.txt",
        "groundtruth.txt",
        "trajectory.txt",
        "poses.txt",
    ]
    check_dirs = [scene_dir, scene_dir.parent, scene_dir.parent.parent]
    for cdir in check_dirs:
        if not cdir.exists():
            continue
        for fname in candidates:
            p = cdir / fname
            if p.exists():
                rows = _load_numeric_pose_rows(p)
                pos = _extract_positions_from_rows(rows)
                if pos is not None:
                    return pos
    return None


def _turn_labels_from_positions(num_frames: int, positions: np.ndarray, num_classes: int) -> np.ndarray:
    if num_classes < 3:
        return np.zeros(num_frames, dtype=np.int64)
    k = min(num_frames, len(positions))
    headings = []
    for i in range(max(1, k - 1)):
        p0 = positions[i]
        p1 = positions[min(i + 1, k - 1)]
        d = p1 - p0
        headings.append(float(np.arctan2(d[1], d[0] + 1e-8)))
    headings = np.array(headings if headings else [0.0], dtype=np.float32)
    dtheta = np.diff(headings, prepend=headings[0])
    thr = max(np.std(dtheta) * 0.5, 0.03)
    labels = np.ones(k, dtype=np.int64)  # 0:left, 1:straight, 2:right
    labels[dtheta < -thr] = 0
    labels[dtheta > thr] = 2
    if k < num_frames:
        pad = np.full(num_frames - k, labels[-1], dtype=np.int64)
        labels = np.concatenate([labels, pad], axis=0)
    return labels[:num_frames]


def _turn_label_from_motion_proxy(prev_img: np.ndarray, cur_img: np.ndarray, threshold: float = 0.01) -> int:
    # Use x-axis center-of-mass shift as turn proxy in absence of explicit poses.
    h, w = prev_img.shape
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    p = prev_img.mean(axis=0)
    c = cur_img.mean(axis=0)
    p_com = float((p * xs).sum() / (p.sum() + 1e-8))
    c_com = float((c * xs).sum() / (c.sum() + 1e-8))
    delta = c_com - p_com
    if delta < -threshold:
        return 0
    if delta > threshold:
        return 2
    return 1


def _build_tartanair_labels(
    scene: Path,
    frame_paths: List[Path],
    num_classes: int,
    image_size: int,
    label_mode: str,
) -> List[int]:
    mode = str(label_mode).lower()
    if num_classes < 3:
        return [0] * len(frame_paths)

    if mode in ("pose_turn", "turn_intent"):
        positions = _find_scene_positions(scene)
        if positions is not None:
            return _turn_labels_from_positions(len(frame_paths), positions, num_classes).tolist()

        # Fallback: temporal motion proxy from frame-to-frame center shift.
        labels = [1]
        prev = np.array(Image.open(frame_paths[0]).convert("L").resize((image_size, image_size)), dtype=np.float32) / 255.0
        for p in frame_paths[1:]:
            cur = np.array(Image.open(p).convert("L").resize((image_size, image_size)), dtype=np.float32) / 255.0
            labels.append(_turn_label_from_motion_proxy(prev, cur))
            prev = cur
        return labels

    # Legacy fallback: scene-id proxy label.
    scene_id = int("".join(ch for ch in scene.name if ch.isdigit()) or "0")
    lbl = scene_id % max(1, num_classes)
    return [lbl] * len(frame_paths)


def generate_federated_data(num_clients: int, samples_per_client: int,
                           input_dim: int, num_classes: int,
                           heterogeneity: float = 0.5,
                           dataset_name: str = "synthetic",
                           tartanair_root: str = "tartanair-test-mono-release/mono",
                           image_size: int = 32,
                           label_mode: str = "turn_intent") -> Tuple:
    if str(dataset_name).lower() == "tartanair":
        root = Path(tartanair_root)
        scene_dirs = sorted([p for p in root.iterdir() if p.is_dir()]) if root.exists() else []
        if not scene_dirs:
            raise ValueError(f"TartanAir directory not found or empty: {tartanair_root}")

        total_train = int(num_clients * samples_per_client)
        total_needed = total_train + max(total_train // 5, 100)
        per_scene = max(16, total_needed // max(len(scene_dirs), 1))

        selected_paths: List[Path] = []
        selected_labels = []
        for scene in scene_dirs:
            files = sorted(scene.glob("*.png"))
            if not files:
                continue
            take = min(len(files), per_scene)
            choice = np.random.choice(len(files), size=take, replace=False)
            picked = [files[int(ci)] for ci in choice]
            picked.sort()
            labels = _build_tartanair_labels(
                scene=scene,
                frame_paths=picked,
                num_classes=num_classes,
                image_size=image_size,
                label_mode=label_mode,
            )
            for p, lbl in zip(picked, labels):
                selected_paths.append(p)
                selected_labels.append(int(lbl))

        if len(selected_paths) < num_clients:
            raise ValueError("Insufficient Tartanair samples for federated partitioning.")

        images = []
        for p in selected_paths:
            img = Image.open(p).convert("L").resize((image_size, image_size))
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
        X = torch.tensor(np.stack(images), dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(selected_labels, dtype=torch.long)

        perm = torch.randperm(len(X))
        X = X[perm]
        y = y[perm]

        test_size = min(len(X) // 5, max(100, len(X) // 10))
        test_X = X[:test_size]
        test_y = y[:test_size]
        train_X = X[test_size:]
        train_y = y[test_size:]

        train_indices = torch.randperm(len(train_X))
        splits = torch.tensor_split(train_indices, num_clients)
        client_datasets = []
        for s in splits:
            if len(s) == 0:
                ridx = torch.randint(0, len(train_X), (1,))
                s = ridx
            client_datasets.append(TensorDataset(train_X[s], train_y[s]))

        global_test_dataset = TensorDataset(test_X, test_y)
        return client_datasets, global_test_dataset

    total_samples = num_clients * samples_per_client

    X, y = make_classification(
        n_samples=total_samples,
        n_features=input_dim,
        n_informative=input_dim // 2,
        n_redundant=0,
        n_classes=num_classes,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=42
    )

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    X = (X - mean) / std

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    client_datasets = []
    indices = torch.randperm(total_samples)
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_indices = indices[start_idx:end_idx]
        client_X = X[client_indices]
        client_y = y[client_indices]
        client_datasets.append(TensorDataset(client_X, client_y))

    test_size = min(len(X) // 5, 500)
    test_indices = np.random.choice(len(X), test_size, replace=False)
    test_X = X[test_indices]
    test_y = y[test_indices]
    global_test_dataset = TensorDataset(test_X, test_y)

    return client_datasets, global_test_dataset


def train_client(
    model: nn.Module,
    dataset: TensorDataset,
    epochs: int = 3,
    lr: float = 0.01,
    device: str = "auto",
    scaffold_c_global: Optional[Dict[str, torch.Tensor]] = None,
    scaffold_c_client: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple:
    if len(dataset) == 0:
        return model.state_dict(), float('inf'), 0, 0

    torch_device = resolve_device(device)
    model = model.to(torch_device)
    model.train()
    if scaffold_c_global is not None and scaffold_c_client is not None:
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    total_loss = 0.0
    local_steps = 0
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            local_steps += 1
            batch_X = batch_X.to(torch_device, non_blocking=True)
            batch_y = batch_y.to(torch_device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            if scaffold_c_global is not None and scaffold_c_client is not None:
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if name in scaffold_c_global and name in scaffold_c_client:
                        param.grad = (
                            param.grad
                            - scaffold_c_client[name].to(torch_device)
                            + scaffold_c_global[name].to(torch_device)
                        )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        total_loss += epoch_loss / len(dataloader)

    avg_loss = total_loss / epochs
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return state, avg_loss, len(dataset), local_steps


def evaluate_model(model: nn.Module, test_dataset: TensorDataset, device: str = "auto") -> Tuple[float, float]:
    if len(test_dataset) == 0:
        return 0.0, float('inf')

    torch_device = resolve_device(device)
    model = model.to(torch_device)
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(torch_device, non_blocking=True)
            y = y.to(torch_device, non_blocking=True)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else float('inf')
    return accuracy, avg_loss


# -----------------------------
# Comparison Logic
# -----------------------------
def compare_protocols(protocols_config: Dict, experiment_config: Dict) -> Dict:
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING PROTOCOL COMPARISON")
    print("=" * 70)

    run_device = resolve_device(experiment_config.get("device", "auto"))
    client_datasets, test_dataset = generate_federated_data(
        num_clients=experiment_config['num_clients'],
        samples_per_client=experiment_config['samples_per_client'],
        input_dim=experiment_config['input_dim'],
        num_classes=experiment_config['num_classes'],
        heterogeneity=experiment_config['heterogeneity'],
        dataset_name=experiment_config.get('dataset_name', 'synthetic'),
        tartanair_root=experiment_config.get('tartanair_root', 'tartanair-test-mono-release/mono'),
        image_size=experiment_config.get('image_size', 32),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=max(32, len(test_dataset) // 10),
        shuffle=False
    )

    model_config = {
        'input_dim': experiment_config['input_dim'],
        'hidden_dim': experiment_config['hidden_dim'],
        'output_dim': experiment_config['num_classes'],
        'image_size': int(experiment_config.get('image_size', 32)),
        'image_channels': int(experiment_config.get('image_channels', 1)),
        'backbone': str(experiment_config.get('backbone', 'resnet18')),
    }

    comm_budget_mb = float(experiment_config.get('comm_budget_mb', 60.0))
    latency_budget_sec = float(experiment_config.get('latency_budget_sec', experiment_config['duration']))

    results = {}
    for protocol_name, protocol_params in protocols_config.items():
        base_name = protocol_name.split("_")[0]  # e.g. improved_async
        protocol = create_protocol(
            base_name,
            num_clients=experiment_config['num_clients'],
            **protocol_params
        )
        protocol.set_global_model(SimpleNN(**model_config).state_dict())

        # Run simple loop
        start_time = time.time()
        while time.time() - start_time < experiment_config['duration']:
            for client_id in range(experiment_config['num_clients']):
                global_state, pulled_version = protocol.get_global_model_with_version()
                if global_state is None:
                    continue
                protocol.account_model_downlink(global_state)
                local_model = SimpleNN(**model_config)
                local_model.load_state_dict(global_state)

                scaffold_c_global = None
                scaffold_c_client = None
                if base_name == "scaffold":
                    scaffold_c_global, scaffold_c_client = protocol.get_scaffold_controls(f"client_{client_id}")

                updated_state, loss, data_size, local_steps = train_client(
                    local_model,
                    client_datasets[client_id],
                    epochs=1,
                    lr=0.01,
                    device=str(run_device),
                    scaffold_c_global=scaffold_c_global,
                    scaffold_c_client=scaffold_c_client,
                )

                update_dict = {}
                for param_name, param in updated_state.items():
                    if param_name in global_state and 'num_batches_tracked' not in param_name:
                        param_update = param.clone().float()
                        global_param = global_state[param_name].clone().float()
                        update_dict[param_name] = param_update - global_param
                update_transport = protocol.compress_for_transport(
                    update_dict,
                    client_id=f"client_{client_id}",
                )
                scaffold_payload = None
                if base_name == "scaffold" and scaffold_c_global is not None and scaffold_c_client is not None:
                    scaffold_payload = build_scaffold_control_payload(
                        update_delta=update_dict,
                        c_global=scaffold_c_global,
                        c_client=scaffold_c_client,
                        local_steps=int(local_steps),
                        local_lr=0.01,
                    )

                update = ClientUpdate(
                    client_id=f"client_{client_id}",
                    update_data=update_transport,
                    model_version=int(pulled_version),
                    local_loss=loss,
                    data_size=data_size,
                    timestamp=time.time(),
                    local_steps=int(local_steps) if base_name == "scaffold" else None,
                    local_lr=0.01 if base_name == "scaffold" else None,
                    scaffold_control_payload=scaffold_payload,
                )
                protocol.receive_update(update)

        # Evaluate
        final_model_state = protocol.get_global_model()
        eval_model = SimpleNN(**model_config)
        eval_model.load_state_dict(final_model_state)
        accuracy, loss = evaluate_model(eval_model, test_dataset, device=str(run_device))

        # Intent-F1 & Explanation-BLEU (synthetic explanation baseline)
        intent_f1, expl_bleu, _, _ = evaluate_with_intent_and_explanation(
            eval_model,
            test_dataset,
            device=str(run_device),
            id2label={i: f"intent_{i}" for i in range(experiment_config['num_classes'])}
        )

        elapsed_sec = time.time() - start_time
        metrics = protocol.metrics.get_summary()
        metrics['final_accuracy'] = accuracy
        metrics['intent_f1'] = round(intent_f1, 4)
        metrics['explanation_bleu'] = round(expl_bleu, 4)
        metrics['elapsed_sec'] = float(elapsed_sec)
        metrics['tri_objective_score'] = tri_objective_score(
            accuracy=float(accuracy),
            communication_mb=float(metrics.get('total_data_transmitted_mb', 0.0)),
            latency_sec=float(elapsed_sec),
            comm_budget_mb=comm_budget_mb,
            latency_budget_sec=latency_budget_sec,
        )
        metrics['within_budget'] = within_budgets(
            communication_mb=float(metrics.get('total_data_transmitted_mb', 0.0)),
            latency_sec=float(elapsed_sec),
            comm_budget_mb=comm_budget_mb,
            latency_budget_sec=latency_budget_sec,
        )
        results[protocol_name] = metrics

        protocol.shutdown()
        print(f"\nResults for {protocol_name}:")
        print(f"  Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, "
              f"Intent-F1: {intent_f1:.4f}, BLEU: {expl_bleu:.4f}")

    return results


# -----------------------------
# Main
# -----------------------------
def default_baseline_configs() -> Dict[str, Dict]:
    """Centralized baseline config registry for fair comparisons."""
    # Keep function for backward compatibility while delegating to profile builder.
    return build_protocol_suite(num_clients=50, strict=True, include_improved=False)


def main():
    """Main comparison experiment"""

    experiment_config = {
        'num_clients': 50,
        'samples_per_client': 500,
        'input_dim': 12,
        'hidden_dim': 32,
        'num_classes': 3,
        'heterogeneity': 0.5,
        'duration': 360,
        'comm_budget_mb': 60.0,
        'latency_budget_sec': 360.0,
        'dataset_name': 'tartanair',
        'tartanair_root': 'tartanair-test-mono-release/mono',
        'image_size': 32,
        'strict_baseline_profiles': True,
        'include_improved_profiles': True,
        'include_legacy_grid': False,
        'include_autoscale_variants': False,
        'device': 'auto',
    }

    # Paper-oriented suite (recommended) or legacy expanded grid.
    protocols_config = build_protocol_suite(
        num_clients=experiment_config['num_clients'],
        strict=experiment_config.get('strict_baseline_profiles', True),
        include_improved=experiment_config.get('include_improved_profiles', True),
    )
    improved_async_configs = {}
    if experiment_config.get('include_legacy_grid', False):
        improved_async_configs = generate_all_configs()
        for name, cfg in improved_async_configs.items():
            protocols_config[f"improved_async_{name}"] = cfg

    if experiment_config.get('include_autoscale_variants', False):
        # Optional scale-aware variants for robustness checks across different client counts.
        for name, cfg in list(improved_async_configs.items()):
            scaled_cfg = dict(cfg)
            scaled_cfg['auto_scale_params'] = True
            protocols_config[f"improved_async_{name}_autoscale"] = scaled_cfg

    with open("active_protocol_suite.json", "w", encoding="utf-8") as f:
        json.dump(protocols_config, f, indent=2, default=str)


    results = compare_protocols(protocols_config, experiment_config)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for proto, metrics in results.items():
        print(f"{proto}: Final Acc={metrics['final_accuracy']:.4f}, "
              f"Comm={metrics['total_data_transmitted_mb']:.2f}MB, "
              f"Aggregations={metrics['aggregations_performed']}")


from metrics import macro_f1, corpus_bleu
from torch.utils.data import DataLoader

def evaluate_with_intent_and_explanation(
    model,
    data_or_loader,
    device: str = "cpu",
    id2label=None,
    num_classes: int = None
):
    """
    Evaluate intent macro-F1 and explanation BLEU.

    - If the model provides `explain(xb, preds)` -> List[str], we use that.
    - Otherwise we synthesize minimal explanations as single-token labels ("intent_k"),
      and compute BLEU-1, making BLEU correlate with label agreement rather than template wording.
    """
    model.eval()
    model.to(device)

    loader = data_or_loader if isinstance(data_or_loader, DataLoader) \
             else DataLoader(data_or_loader, batch_size=512, shuffle=False)

    all_preds, all_golds = [], []
    refs: list[str] = []
    hyps: list[str] = []

    def _lab(idx: int) -> str:
        if id2label is not None and idx in id2label:
            return str(id2label[idx])
        return f"intent_{idx}"

    has_explain = hasattr(model, "explain") and callable(getattr(model, "explain"))

    with torch.no_grad():
        for xb, yb in loader:
            if not isinstance(xb, torch.Tensor):
                xb = torch.tensor(xb)
            xb = xb.to(device).float()
            yb = yb.to(device)

            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_golds.extend(yb.detach().cpu().tolist())

            # references / hypotheses for BLEU
            batch_refs = [_lab(int(g)) for g in yb.detach().cpu().tolist()]

            if has_explain:
                try:
                    batch_hyps = model.explain(xb, preds)  # List[str]
                except Exception:
                    # fallback to single-token label
                    batch_hyps = [_lab(int(p)) for p in preds.detach().cpu().tolist()]
            else:
                batch_hyps = [_lab(int(p)) for p in preds.detach().cpu().tolist()]

            refs.extend(batch_refs)
            hyps.extend(batch_hyps)

    intent_f1 = macro_f1(all_preds, all_golds, num_classes=num_classes)
    # Use BLEU-1 to avoid higher-order n-gram sparsity without a real explainer
    explanation_bleu = corpus_bleu(refs, hyps, max_n=1, smooth=True, lowercase=True)
    return intent_f1, explanation_bleu, all_preds, all_golds


if __name__ == "__main__":
    main()
