"""
federated_protocol_framework.py
Unified framework for federated learning protocol comparison
"""

import copy
import time
import threading
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod  # Abstract Base Classes for interface definition
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging
from compression_strategies import TopKCompressor, SignSGDCompressor, QSGDCompressor

logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ClientUpdate:
    """Unified client update structure"""
    client_id: str
    update_data: Dict[str, torch.Tensor]
    model_version: int
    local_loss: float
    data_size: int
    timestamp: float
    staleness: Optional[float] = None
    # For SCAFFOLD canonical control variate updates
    local_steps: Optional[int] = None
    local_lr: Optional[float] = None
    # Explicit SCAFFOLD uplink payload: delta_c (or equivalent control payload).
    scaffold_control_payload: Optional[Dict[str, torch.Tensor]] = None


def build_scaffold_control_payload(
    update_delta: Dict[str, torch.Tensor],
    c_global: Dict[str, torch.Tensor],
    c_client: Dict[str, torch.Tensor],
    local_steps: int,
    local_lr: float,
) -> Dict[str, torch.Tensor]:
    """
    Build explicit SCAFFOLD client->server control payload (delta_c_i).

    Canonical relation:
      c_i_new = c_i_old - c - delta/(K*eta), where delta = (w_i - w)
      delta_c_i = c_i_new - c_i_old
    """
    k_steps = max(1, int(local_steps))
    eta = float(local_lr)
    denom = max(float(k_steps) * eta, 1e-12)

    payload: Dict[str, torch.Tensor] = {}
    for name, delta in update_delta.items():
        delta_f = delta.float()
        c_old = c_client[name].float() if name in c_client else torch.zeros_like(delta_f, dtype=torch.float32)
        c_g = c_global[name].float() if name in c_global else torch.zeros_like(delta_f, dtype=torch.float32)
        c_new = c_old - c_g - (delta_f / denom)
        payload[name] = (c_new - c_old).detach().cpu()
    return payload


class ProtocolMetrics:
    """Unified metrics collector for all protocols"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {
            # Communication metrics
            'total_data_transmitted_mb': 0.0,
            'total_updates_sent': 0,
            'total_updates_accepted': 0,
            'total_updates_rejected': 0,

            # Performance metrics
            'aggregations_performed': 0,
            'final_accuracy': 0.0,
            'max_accuracy': 0.0,
            'convergence_time': float('inf'),

            # Efficiency metrics
            'average_round_time': 0.0,
            'idle_time_percentage': 0.0,
            'throughput_updates_per_second': 0.0,

            # Quality metrics
            'average_staleness': 0.0,
            'high_quality_updates': 0,
            'staleness_samples_all': [],
            'staleness_samples_accepted': [],
            'staleness_samples_rejected': [],
            'staleness_drop_ratio': 0.0,
            'staleness_p50': 0.0,
            'staleness_p90': 0.0,
            'staleness_p99': 0.0,
            'accepted_staleness_p90': 0.0,
            'rejected_staleness_p90': 0.0,
            'buffer_occupancy_samples': [],
            'buffer_wait_sec_samples': [],
            'buffer_occupancy_mean': 0.0,
            'buffer_occupancy_p90': 0.0,
            'buffer_wait_sec_mean': 0.0,
            'buffer_wait_sec_p90': 0.0,
            'control_payload_uplink_mb': 0.0,
            'control_payload_downlink_mb': 0.0,

            # Time series data
            'accuracy_history': [],
            'loss_history': [],
            'timestamps': []
        }

    def record_staleness(self, staleness: float, accepted: bool):
        s = float(max(0.0, staleness))
        self.metrics['staleness_samples_all'].append(s)
        if accepted:
            self.metrics['staleness_samples_accepted'].append(s)
        else:
            self.metrics['staleness_samples_rejected'].append(s)

    def record_buffer_stats(self, occupancy: int, wait_time_sec: Optional[float] = None):
        self.metrics['buffer_occupancy_samples'].append(float(max(0, occupancy)))
        if wait_time_sec is not None:
            self.metrics['buffer_wait_sec_samples'].append(float(max(0.0, wait_time_sec)))

    def update_communication(self, data_size_bytes: float, accepted: bool = True):
        """
        Update communication metrics.
        The input must be BYTES (from calculate_update_size). We convert to MB here.
        """
        mb = float(data_size_bytes) / (1024.0 * 1024.0)
        self.metrics['total_data_transmitted_mb'] += mb
        self.metrics['total_updates_sent'] += 1
        if accepted:
            self.metrics['total_updates_accepted'] += 1
        else:
            self.metrics['total_updates_rejected'] += 1

    def add_overhead_communication(self, data_size_bytes: float, bucket: str = "control_payload_uplink_mb"):
        """
        Add payload overhead in BYTES without touching update acceptance counters.
        Used for protocol-specific extra payloads (e.g., SCAFFOLD controls).
        """
        mb = float(data_size_bytes) / (1024.0 * 1024.0)
        self.metrics['total_data_transmitted_mb'] += mb
        if bucket in self.metrics:
            self.metrics[bucket] += mb

    def update_performance(self, accuracy: float, loss: float, timestamp: float):
        """Update performance metrics"""
        self.metrics['accuracy_history'].append(accuracy)
        self.metrics['loss_history'].append(loss)
        self.metrics['timestamps'].append(timestamp)

        if accuracy > self.metrics['max_accuracy']:
            self.metrics['max_accuracy'] = accuracy

        # Check convergence (accuracy stable for last 5 measurements)
        if len(self.metrics['accuracy_history']) >= 5:
            recent_std = np.std(self.metrics['accuracy_history'][-5:])
            if recent_std < 0.01 and self.metrics['convergence_time'] == float('inf'):
                self.metrics['convergence_time'] = timestamp

    def finalize(self):
        """Calculate final metrics"""
        if self.metrics['accuracy_history']:
            self.metrics['final_accuracy'] = self.metrics['accuracy_history'][-1]

        if self.metrics['timestamps'] and self.metrics['total_updates_accepted'] > 0:
            total_time = self.metrics['timestamps'][-1]
            self.metrics['throughput_updates_per_second'] = (
                self.metrics['total_updates_accepted'] / total_time
            )

        total_seen = int(self.metrics['total_updates_sent'])
        rejected = int(self.metrics['total_updates_rejected'])
        self.metrics['staleness_drop_ratio'] = (float(rejected) / float(total_seen)) if total_seen > 0 else 0.0

        all_s = self.metrics.get('staleness_samples_all', [])
        acc_s = self.metrics.get('staleness_samples_accepted', [])
        rej_s = self.metrics.get('staleness_samples_rejected', [])
        if all_s:
            arr = np.array(all_s, dtype=np.float32)
            self.metrics['staleness_p50'] = float(np.quantile(arr, 0.50))
            self.metrics['staleness_p90'] = float(np.quantile(arr, 0.90))
            self.metrics['staleness_p99'] = float(np.quantile(arr, 0.99))
        if acc_s:
            self.metrics['accepted_staleness_p90'] = float(np.quantile(np.array(acc_s, dtype=np.float32), 0.90))
        if rej_s:
            self.metrics['rejected_staleness_p90'] = float(np.quantile(np.array(rej_s, dtype=np.float32), 0.90))

        occ = self.metrics.get('buffer_occupancy_samples', [])
        if occ:
            occ_arr = np.array(occ, dtype=np.float32)
            self.metrics['buffer_occupancy_mean'] = float(np.mean(occ_arr))
            self.metrics['buffer_occupancy_p90'] = float(np.quantile(occ_arr, 0.90))
        waits = self.metrics.get('buffer_wait_sec_samples', [])
        if waits:
            w_arr = np.array(waits, dtype=np.float32)
            self.metrics['buffer_wait_sec_mean'] = float(np.mean(w_arr))
            self.metrics['buffer_wait_sec_p90'] = float(np.quantile(w_arr, 0.90))

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        self.finalize()
        return self.metrics.copy()


class FederatedProtocol(ABC):
    """Base class for all federated learning protocols"""

    def __init__(self, num_clients: int, **kwargs):
        self.num_clients = num_clients
        self.global_model = None
        self.model_version = 0
        self.metrics = ProtocolMetrics()
        self.running = True
        self._lock = threading.RLock()

        # ---- Common defaults to avoid attribute errors across protocols ----
        self.compression = kwargs.get('compression', None)
        self.server_lr = kwargs.get('server_lr', 0.2)  # generic default
        self.gradient_clip = kwargs.get('gradient_clip', 2.0)
        self.compressor = None  # will be set by protocols that support compression
        self.round_participation_rate = kwargs.get('participation_rate', 0.5)


        # Protocol-specific parameters
        self.configure(**kwargs)

    @abstractmethod
    def configure(self, **kwargs):
        """Configure protocol-specific parameters"""
        pass

    @abstractmethod
    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        """
        Receive update from client
        Returns: (accepted, new_model_version)
        """
        pass

    @abstractmethod
    def aggregate_updates(self):
        """Perform aggregation of updates"""
        pass

    def set_global_model(self, model_state: Dict[str, torch.Tensor]):
        """Set initial global model"""
        with self._lock:
            self.global_model = copy.deepcopy(model_state)
            self.model_version = 0

    def get_global_model(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current global model"""
        with self._lock:
            if self.global_model is not None:
                return copy.deepcopy(self.global_model)
            return None

    def get_global_model_with_version(self) -> Tuple[Optional[Dict[str, torch.Tensor]], int]:
        """Get a consistent (state, version) snapshot for pull-based staleness accounting."""
        with self._lock:
            state = copy.deepcopy(self.global_model) if self.global_model is not None else None
            return state, int(self.model_version)

    def calculate_update_size(self, update_data: Dict[str, Any]) -> int:
        """
        Return size in bytes of the update_data in its transported (compressed) form.
        Handles:
          - TopK: ((indices, values), shape)
          - SignSGD: (sign_bits_like_np, magnitude_float32)
          - QSGD: (qvals_like_np, norm_float32, scale_float32)
          - Uncompressed: torch.Tensor
        """
        comp = getattr(self, "compression", None)
        total = 0

        for _, v in update_data.items():
            # compressed tuple ((...), shape)
            if isinstance(v, tuple) and len(v) == 2:
                compressed, shape = v
                if comp == "topk" and isinstance(compressed, tuple) and len(compressed) == 2:
                    indices, values = compressed
                    total += getattr(indices, "nbytes", 0) + getattr(values, "nbytes", 0)
                elif comp == "signsgd" and isinstance(compressed, tuple) and len(compressed) == 2:
                    signs, meta = compressed
                    # meta can be (num_elements, magnitude), count two float32-equivalent scalars
                    total += getattr(signs, "nbytes", 0) + 8
                elif comp == "qsgd" and isinstance(compressed, tuple) and len(compressed) == 3:
                    qvals, norm, scale = compressed
                    total += getattr(qvals, "nbytes", 0) + 8  # two float32
                else:
                    # Unknown compressed structure; ignore or log
                    pass
            else:
                # Uncompressed tensor fallback
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
        return total

    def calculate_tensor_dict_size(self, tensor_data: Dict[str, Any]) -> int:
        """Return dense tensor payload size in bytes."""
        total = 0
        for _, v in tensor_data.items():
            if isinstance(v, torch.Tensor):
                total += int(v.numel() * v.element_size())
            elif hasattr(v, "nbytes"):
                total += int(v.nbytes)
        return int(total)

    def shutdown(self):
        """Shutdown protocol"""
        self.running = False
        logger.info(f"{self.__class__.__name__} shutdown")


class SyncFedAvg(FederatedProtocol):
    """Traditional synchronous FedAvg implementation"""

    def configure(self, **kwargs):
        self.strict_reproduction = kwargs.get('strict_reproduction', False)
        self.round_participation_rate = kwargs.get(
            'participation_rate',
            1.0 if self.strict_reproduction else 0.5
        )
        self.max_round_time = kwargs.get('max_round_time', 30.0)
        self.use_timeout = kwargs.get('use_timeout', not self.strict_reproduction)
        self.fixed_round_size = max(1, kwargs.get(
            'fixed_round_size',
            int(self.num_clients * self.round_participation_rate)
        ))
        self.current_round = 0
        self.round_buffer = []
        self.round_start_time = time.time()

        self.compression = kwargs.get('compression', None)

        # Initialize compressor
        if self.compression == "topk":
            self.compressor = TopKCompressor(kwargs.get('k', 100))
        elif self.compression == "signsgd":
            self.compressor = SignSGDCompressor()
        elif self.compression == "qsgd":
            self.compressor = QSGDCompressor(kwargs.get('num_bits', 8))
        else:
            self.compressor = None

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            # Optional compression before storing
            if hasattr(self, "compressor") and self.compressor is not None:
                compressed_update = {}
                for name, delta in update.update_data.items():
                    comp, shape = self.compressor.compress(delta)
                    compressed_update[name] = (comp, shape)
                update.update_data = compressed_update

            # Check if update is for current round
            if update.model_version != self.current_round:
                self.metrics.record_staleness(abs(float(self.current_round - update.model_version)), accepted=False)
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data),
                    accepted=False
                )
                return False, self.current_round

            # Add to round buffer
            self.round_buffer.append(update)

            # Record metrics (count compressed size if enabled)
            update_size = self.calculate_update_size(update.update_data)
            self.metrics.record_staleness(0.0, accepted=True)
            self.metrics.update_communication(update_size, accepted=True)

            # Check if we should aggregate
            if self.strict_reproduction:
                should_aggregate = len(self.round_buffer) >= self.fixed_round_size
            else:
                timeout_reached = (time.time() - self.round_start_time) >= self.max_round_time
                min_clients = max(2, int(self.num_clients * self.round_participation_rate))
                should_aggregate = len(self.round_buffer) >= min_clients or (
                    self.use_timeout and timeout_reached and len(self.round_buffer) >= 2
                )

            if should_aggregate:
                self.aggregate_updates()
                return True, self.current_round + 1

            return True, self.current_round

    def aggregate_updates(self):
        """Perform FedAvg aggregation with optional decompression"""
        if not self.round_buffer:
            return

        # Decompress if needed
        decompressed_buffer = []
        for update in self.round_buffer:
            if hasattr(self, "compressor") and self.compressor is not None:
                decompressed_update = {}
                for name, maybe_compressed in update.update_data.items():
                    # Only decompress if value is a (comp, shape) two-tuple
                    if isinstance(maybe_compressed, tuple) and len(maybe_compressed) == 2:
                        comp, shape = maybe_compressed
                        decompressed_update[name] = self.compressor.decompress(comp, shape)
                    else:
                        # Already a tensor (or plain value) — keep as is
                        decompressed_update[name] = maybe_compressed
                update.update_data = decompressed_update
            decompressed_buffer.append(update)

        # Calculate weighted average
        total_data_size = sum(u.data_size for u in decompressed_buffer)
        aggregated = {}

        for update in decompressed_buffer:
            weight = update.data_size / total_data_size
            for name, param in update.update_data.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(param, dtype=torch.float32)
                if param.dtype != torch.float32:
                    param = param.float()
                aggregated[name] = aggregated[name] + (param * weight)

        # Apply to global model
        if self.global_model is None:
            self.global_model = {}
            for name, param in aggregated.items():
                self.global_model[name] = param.clone()
        else:
            for name, param in aggregated.items():
                if name in self.global_model:
                    if self.global_model[name].dtype != torch.float32:
                        self.global_model[name] = self.global_model[name].float()
                    self.global_model[name] = self.global_model[name] + param
                else:
                    self.global_model[name] = param.clone()

        # Update metrics and prepare for next round
        self.metrics.metrics['aggregations_performed'] += 1
        round_time = time.time() - self.round_start_time
        self.metrics.metrics['average_round_time'] = (
                0.9 * self.metrics.metrics['average_round_time'] + 0.1 * round_time
        )

        num_clients = len(self.round_buffer)

        self.current_round += 1
        self.model_version = self.current_round
        self.round_buffer.clear()
        self.round_start_time = time.time()

        logger.info(f"FedAvg Round {self.current_round} completed with {num_clients} clients")


class AsyncFedAvg(FederatedProtocol):
    """Basic asynchronous FedAvg (FedAsync)"""

    def configure(self, **kwargs):
        self.max_staleness = kwargs.get('max_staleness', 10)
        self.learning_rate = kwargs.get('learning_rate', 1.0)
        self.server_tick_sec = float(kwargs.get('server_tick_sec', 0.02))
        self.staleness_mode = kwargs.get('staleness_mode', 'linear')
        self.staleness_floor = float(kwargs.get('staleness_floor', 0.1))
        self.update_queue = deque()
        self.aggregation_thread = threading.Thread(target=self._continuous_aggregation, daemon=True)
        self.aggregation_thread.start()

        self.compression = kwargs.get('compression', None)

        # Initialize compressor
        if self.compression == "topk":
            self.compressor = TopKCompressor(kwargs.get('k', 100))
        elif self.compression == "signsgd":
            self.compressor = SignSGDCompressor()
        elif self.compression == "qsgd":
            self.compressor = QSGDCompressor(kwargs.get('num_bits', 8))
        else:
            self.compressor = None

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            # Optional compression before storing
            if hasattr(self, "compressor") and self.compressor is not None:
                compressed_update = {}
                for name, delta in update.update_data.items():
                    comp, shape = self.compressor.compress(delta)
                    compressed_update[name] = (comp, shape)
                update.update_data = compressed_update

            current_version = self.model_version
            staleness = current_version - update.model_version

            # Reject if too stale
            if staleness > self.max_staleness:
                self.metrics.record_staleness(float(staleness), accepted=False)
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data),
                    accepted=False
                )
                return False, current_version

            # Count accepted traffic ONCE (compressed form), then apply
            update_size = self.calculate_update_size(update.update_data)
            self.metrics.record_staleness(float(staleness), accepted=True)
            self.metrics.update_communication(update_size, accepted=True)

            update.staleness = staleness
            self.update_queue.append(update)

            return True, current_version

    def _alpha_staleness(self, staleness: float) -> float:
        tau = float(max(0.0, staleness))
        denom = max(float(self.max_staleness), 1e-8)
        base = max(float(self.staleness_floor), 1.0 - tau / denom)
        if self.staleness_mode == 'exp':
            return max(float(self.staleness_floor), float(np.exp(-tau / denom)))
        # default linear
        return base

    def _apply_update(self, update: "ClientUpdate"):
        """
        Apply a single update to the global model.
        Decompress if compression is enabled, then apply a staleness-aware LR.
        """
        # Decompress if needed
        update_data = update.update_data
        if hasattr(self, "compressor") and self.compressor is not None:
            decompressed = {}
            for name, maybe_compressed in update_data.items():
                if isinstance(maybe_compressed, tuple) and len(maybe_compressed) == 2:
                    comp, shape = maybe_compressed
                    decompressed[name] = self.compressor.decompress(comp, shape)
                else:
                    decompressed[name] = maybe_compressed
            update_data = decompressed

        staleness_factor = self._alpha_staleness(update.staleness)
        effective_lr = float(getattr(self, "learning_rate", 0.1)) * staleness_factor

        # Initialize or update
        if self.global_model is None:
            self.global_model = {}
            for name, param in update_data.items():
                if isinstance(param, torch.Tensor) and param.dtype != torch.float32:
                    param = param.float()
                self.global_model[name] = param * effective_lr
        else:
            for name, param in update_data.items():
                if isinstance(param, torch.Tensor) and param.dtype != torch.float32:
                    param = param.float()
                if name in self.global_model:
                    if self.global_model[name].dtype != torch.float32:
                        self.global_model[name] = self.global_model[name].float()
                    self.global_model[name] = self.global_model[name] + (param * effective_lr)
                else:
                    self.global_model[name] = param * effective_lr

        self.model_version += 1
        self.metrics.metrics["aggregations_performed"] += 1

    def _continuous_aggregation(self):
        """Server-side scheduler for queued asynchronous updates."""
        while self.running:
            time.sleep(self.server_tick_sec)
            with self._lock:
                if self.update_queue:
                    next_update = self.update_queue.popleft()
                    self._apply_update(next_update)

    def aggregate_updates(self):
        """No batch aggregation in basic FedAsync"""
        pass


class FedBuff(FederatedProtocol):
    """FedBuff - Buffered asynchronous aggregation"""

    def configure(self, **kwargs):
        self.buffer_size = kwargs.get('buffer_size', 5)
        self.max_staleness = kwargs.get('max_staleness', 15)
        self.update_buffer = deque()
        self.aggregation_thread = threading.Thread(target=self._buffer_aggregation, daemon=True)
        self.aggregation_thread.start()

        self.compression = kwargs.get('compression', None)

        # Initialize compressor
        if self.compression == "topk":
            self.compressor = TopKCompressor(kwargs.get('k', 100))
        elif self.compression == "signsgd":
            self.compressor = SignSGDCompressor()
        elif self.compression == "qsgd":
            self.compressor = QSGDCompressor(kwargs.get('num_bits', 8))
        else:
            self.compressor = None

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            # Optional compression before storing
            if hasattr(self, "compressor") and self.compressor is not None:
                compressed_update = {}
                for name, delta in update.update_data.items():
                    comp, shape = self.compressor.compress(delta)
                    compressed_update[name] = (comp, shape)
                update.update_data = compressed_update

            current_version = self.model_version
            staleness = current_version - update.model_version

            if staleness > self.max_staleness:
                self.metrics.record_staleness(float(staleness), accepted=False)
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data),
                    accepted=False
                )
                return False, current_version

            update.staleness = staleness
            self.update_buffer.append(update)

            update_size = self.calculate_update_size(update.update_data)
            self.metrics.record_staleness(float(staleness), accepted=True)
            self.metrics.update_communication(update_size, accepted=True)

            return True, current_version

    def _buffer_aggregation(self):
        """Aggregate when buffer is full"""
        while self.running:
            time.sleep(0.1)

            with self._lock:
                if len(self.update_buffer) >= self.buffer_size:
                    self.aggregate_updates()

    def aggregate_updates(self):
        """Aggregate buffered updates with server learning rate and clipping"""
        if not self.update_buffer:
            return
        self.metrics.record_buffer_stats(len(self.update_buffer))

        # Decompress if needed
        decompressed_buffer = []
        for update in self.update_buffer:
            if hasattr(self, "compressor") and self.compressor is not None:
                decompressed_update = {}
                for name, maybe_compressed in update.update_data.items():
                    # Only decompress if value is a (comp, shape) two-tuple
                    if isinstance(maybe_compressed, tuple) and len(maybe_compressed) == 2:
                        comp, shape = maybe_compressed
                        decompressed_update[name] = self.compressor.decompress(comp, shape)
                    else:
                        # Already a tensor (or plain value) — keep as is
                        decompressed_update[name] = maybe_compressed
                update.update_data = decompressed_update
            decompressed_buffer.append(update)

        updates_to_aggregate = list(self.update_buffer)
        self.update_buffer.clear()

        # Weighted aggregation of DELTAS
        aggregated_delta = {}
        total_weight = 0

        for update in updates_to_aggregate:
            staleness_weight = 1.0 / (1.0 + update.staleness)
            weight = staleness_weight * update.data_size
            total_weight += weight

            for name, delta in update.update_data.items():
                if name not in aggregated_delta:
                    aggregated_delta[name] = torch.zeros_like(delta, dtype=torch.float32)
                if delta.dtype != torch.float32:
                    delta = delta.float()
                aggregated_delta[name] = aggregated_delta[name] + (delta * weight)

        # Apply aggregated delta to global model
        if total_weight > 0:
            server_lr = getattr(self, "server_lr", 0.1)

            # Normalize and clip
            for name in aggregated_delta:
                aggregated_delta[name] = aggregated_delta[name] / total_weight
                aggregated_delta[name] = torch.clamp(aggregated_delta[name], -5.0, 5.0)

            if self.global_model is None:
                self.global_model = {}
                for name in aggregated_delta:
                    self.global_model[name] = server_lr * aggregated_delta[name]
            else:
                for name in aggregated_delta:
                    if name in self.global_model:
                        self.global_model[name] = self.global_model[name] + server_lr * aggregated_delta[name]
                    else:
                        self.global_model[name] = server_lr * aggregated_delta[name]

        self.model_version += 1
        self.metrics.metrics['aggregations_performed'] += 1
        logger.info(f"FedBuff aggregated {len(updates_to_aggregate)} updates")

class ImprovedAsyncProtocol(FederatedProtocol):
    """Your improved asynchronous protocol with optimizations"""

    @staticmethod
    def _clip_int(value: float, lower: int, upper: int) -> int:
        return max(lower, min(int(round(value)), upper))

    def _apply_auto_scaling(self, kwargs: Dict[str, Any]):
        """
        Scale key protocol parameters with the number of clients.
        This keeps behavior stable from tiny deployments to large simulations.
        """
        if not kwargs.get('auto_scale_params', False):
            return

        n = max(1, int(self.num_clients))

        # Buffer sizes scale sub-linearly with client count, then are safely clipped.
        base_min = self._clip_int(np.sqrt(n), 2, 64)
        base_max = self._clip_int(2 * base_min, base_min + 1, 256)

        # Staleness budget grows slowly with population size.
        base_staleness = self._clip_int(4.0 + 2.5 * np.log2(n), 5, 128)

        self.min_buffer_size = kwargs.get('min_buffer_size', base_min)
        self.max_buffer_size = kwargs.get('max_buffer_size', base_max)
        self.max_staleness = kwargs.get('max_staleness', base_staleness)
        base_participation = float(np.clip(2.0 / np.sqrt(n), 0.1, 1.0))
        self.round_participation_rate = float(kwargs.get('participation_rate', base_participation))

        # Ensure valid order after overrides.
        if self.max_buffer_size <= self.min_buffer_size:
            self.max_buffer_size = self.min_buffer_size + 1

        logger.info(
            "ImprovedAsync auto-scaled params: n=%s, min_buffer=%s, max_buffer=%s, max_staleness=%s",
            n,
            self.min_buffer_size,
            self.max_buffer_size,
            self.max_staleness,
        )

    def configure(self, **kwargs):
        # Core parameters
        self.max_staleness = kwargs.get('max_staleness', 20)
        self.min_buffer_size = kwargs.get('min_buffer_size', 3)
        self.max_buffer_size = kwargs.get('max_buffer_size', 8)

        # Optional scale-aware override for cross-scenario robustness.
        self._apply_auto_scaling(kwargs)

        # Adaptive features
        self.adaptive_weighting = kwargs.get('adaptive_weighting', True)
        self.momentum = kwargs.get('momentum', 0.9)

        # Journal-friendly staleness weighting controls
        self.staleness_mode = kwargs.get('staleness_mode', 'quadratic')
        self.staleness_floor = float(kwargs.get('staleness_floor', 0.2))
        self.staleness_quantile = float(kwargs.get('staleness_quantile', 0.9))
        self.enable_staleness_adaptation = bool(kwargs.get('enable_staleness_adaptation', True))
        self.enable_health_relaxation = bool(kwargs.get('enable_health_relaxation', True))
        self.include_staleness_in_quality = bool(kwargs.get('include_staleness_in_quality', True))
        self.staleness_history = deque(maxlen=int(kwargs.get('staleness_history_size', 512)))
        self.min_effective_staleness = int(kwargs.get('min_effective_staleness', 5))
        self.max_effective_staleness = int(
            kwargs.get('max_effective_staleness', max(self.max_staleness * 2, 32))
        )

        # Compression selection
        self.compression = kwargs.get('compression', None)
        if self.compression == "topk":
            self.compressor = TopKCompressor(kwargs.get('k', 100))
        elif self.compression == "signsgd":
            self.compressor = SignSGDCompressor()
        elif self.compression == "qsgd":
            self.compressor = QSGDCompressor(kwargs.get('num_bits', 8))
        else:
            self.compressor = None

        # ---- MUST set these BEFORE any background thread starts ----
        self.server_lr = kwargs.get(
            'server_lr',
            0.1 if self.compression == "signsgd" else 0.2
        )
        self.gradient_clip = kwargs.get(
            'gradient_clip',
            1.0 if self.compression == "signsgd" else 2.0
        )
        # -----------------------------------------------------------

        # Internal state
        self.update_buffer = deque()
        self.momentum_buffer = {}
        self.client_contribution_scores = defaultdict(lambda: 1.0)
        self.network_health = 0.5

        # Start aggregation thread (after all attributes are set)
        self.aggregation_thread = threading.Thread(target=self._smart_aggregation, daemon=True)
        self.aggregation_thread.start()

    def _compute_effective_max_staleness(self) -> int:
        if not self.enable_staleness_adaptation:
            return max(self.min_effective_staleness, min(int(self.max_staleness), self.max_effective_staleness))

        if len(self.staleness_history) >= 8:
            q = float(np.quantile(np.array(self.staleness_history, dtype=np.float32), self.staleness_quantile))
            base = max(float(self.max_staleness), q)
        else:
            base = float(self.max_staleness)

        # Relax threshold under poor network health while keeping hard safety bounds.
        if self.enable_health_relaxation:
            relax = 1.15 if self.network_health < 0.3 else (1.05 if self.network_health < 0.6 else 1.0)
        else:
            relax = 1.0
        effective = int(round(base * relax))
        return max(self.min_effective_staleness, min(effective, self.max_effective_staleness))

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        """
        Receive a client update:
          - Optionally compress deltas (server-side).
          - Count communication exactly once in compressed form.
          - Drop overly stale updates using an adaptive threshold.
          - Push accepted updates to the buffer without double counting.
        """
        with self._lock:
            # Optional compression before storing
            if getattr(self, "compressor", None) is not None:
                compressed_update = {}
                for name, delta in update.update_data.items():
                    if isinstance(self.compressor, TopKCompressor):
                        comp, shape = self.compressor.compress(delta, key=f"{update.client_id}:{name}")
                    else:
                        comp, shape = self.compressor.compress(delta)
                    compressed_update[name] = (comp, shape)
                update.update_data = compressed_update

            current_version = self.model_version
            # Non-negative staleness guard
            staleness = max(0, current_version - update.model_version)
            self.staleness_history.append(float(staleness))

            # Adaptive staleness threshold from observed quantile + bounded control.
            effective_max_staleness = self._compute_effective_max_staleness()

            # Compute transported size in compressed form (robust)
            try:
                transported_bytes = self.calculate_update_size(update.update_data)
            except Exception:
                transported_bytes = 0

            # Drop too-stale updates: count as rejected traffic once
            if staleness > effective_max_staleness:
                self.metrics.record_staleness(float(staleness), accepted=False)
                self.metrics.update_communication(transported_bytes, accepted=False)
                # Update network health given staleness observation
                self._update_network_health(staleness)
                return False, current_version

            # Accept: count as accepted traffic once (do not double-count later)
            self.metrics.record_staleness(float(staleness), accepted=True)
            self.metrics.update_communication(transported_bytes, accepted=True)

            # Store metadata and enqueue
            update.staleness = staleness
            self.update_buffer.append(update)

            # Update network health
            self._update_network_health(staleness)

            return True, current_version

    def _update_network_health(self, staleness: float):
        """Update network health estimation"""
        staleness_score = max(0, 1.0 - staleness / self.max_staleness)
        self.network_health = 0.9 * self.network_health + 0.1 * staleness_score

    def _smart_aggregation(self):
        """Smart aggregation with adaptive buffer size"""
        while self.running:
            time.sleep(0.05)

            with self._lock:
                buffer_size = len(self.update_buffer)

                # Adaptive buffer threshold
                if self.network_health > 0.7:
                    threshold = self.min_buffer_size
                elif self.network_health > 0.4:
                    threshold = (self.min_buffer_size + self.max_buffer_size) // 2
                else:
                    threshold = self.max_buffer_size

                # Aggregate if buffer reaches threshold or timeout
                should_aggregate = False

                if buffer_size >= threshold:
                    should_aggregate = True
                elif buffer_size >= self.min_buffer_size and self.update_buffer:
                    # Check oldest update age
                    oldest_age = time.time() - self.update_buffer[0].timestamp
                    if oldest_age > 2.0:  # 2 second timeout
                        should_aggregate = True

                if should_aggregate:
                    self.aggregate_updates()

    def aggregate_updates(self):
        """Intelligent aggregation with quality-based weighting + server learning rate + clipping"""
        if not self.update_buffer:
            return
        self.metrics.record_buffer_stats(len(self.update_buffer))

        # Decompress if needed
        decompressed_buffer = []
        for update in self.update_buffer:
            if hasattr(self, "compressor") and self.compressor is not None:
                decompressed_update = {}
                for name, maybe_compressed in update.update_data.items():
                    # Only decompress if value is a (comp, shape) two-tuple
                    if isinstance(maybe_compressed, tuple) and len(maybe_compressed) == 2:
                        comp, shape = maybe_compressed
                        decompressed_update[name] = self.compressor.decompress(comp, shape)
                    else:
                        # Already a tensor (or plain value) — keep as is
                        decompressed_update[name] = maybe_compressed
                update.update_data = decompressed_update
            decompressed_buffer.append(update)

        # NOTE: we intentionally skip immediate averaging over the full decompressed
        # buffer and instead perform quality-aware selection below.
        # Select best updates (quality-based selection)
        num_to_aggregate = min(len(self.update_buffer), self.max_buffer_size)

        # Sort by quality score with indices
        scored_updates = []
        for i, update in enumerate(self.update_buffer):
            quality_score = self._calculate_quality_score(update)
            scored_updates.append((quality_score, i, update))

        scored_updates.sort(key=lambda x: x[0], reverse=True)

        # Get selected updates and their indices
        selected_indices = set()
        selected_updates = []
        for score, idx, update in scored_updates[:num_to_aggregate]:
            selected_indices.add(idx)
            selected_updates.append(update)

        # Keep remaining updates (not selected)
        remaining_updates = [update for i, update in enumerate(self.update_buffer)
                             if i not in selected_indices]
        self.update_buffer.clear()
        self.update_buffer.extend(remaining_updates)

        # Weighted aggregation of DELTAS
        aggregated_delta = {}
        total_weight = 0

        for update in selected_updates:
            weight = self._calculate_weight(update)
            total_weight += weight

            for name, delta in update.update_data.items():
                if name not in aggregated_delta:
                    aggregated_delta[name] = torch.zeros_like(delta, dtype=torch.float32)
                if delta.dtype != torch.float32:
                    delta = delta.float()
                aggregated_delta[name] = aggregated_delta[name] + (delta * weight)

        if total_weight > 0:
            server_lr = getattr(self, "server_lr", 0.1)

            # Normalize
            for name in aggregated_delta:
                aggregated_delta[name] = aggregated_delta[name] / total_weight

            # Momentum smoothing
            if self.momentum > 0:
                for name in aggregated_delta:
                    if name in self.momentum_buffer:
                        self.momentum_buffer[name] = (
                                self.momentum * self.momentum_buffer[name] +
                                (1 - self.momentum) * aggregated_delta[name]
                        )
                        aggregated_delta[name] = self.momentum_buffer[name]
                    else:
                        self.momentum_buffer[name] = aggregated_delta[name]

            # Clip to prevent gradient explosion
            for name in aggregated_delta:
                aggregated_delta[name] = torch.clamp(aggregated_delta[name], -self.gradient_clip, self.gradient_clip)

            # Apply delta to global model
            if self.global_model is None:
                self.global_model = {}
                for name in aggregated_delta:
                    self.global_model[name] = server_lr * aggregated_delta[name]
            else:
                for name in aggregated_delta:
                    if name in self.global_model:
                        self.global_model[name] = self.global_model[name] + server_lr * aggregated_delta[name]
                    else:
                        self.global_model[name] = server_lr * aggregated_delta[name]

        self.model_version += 1
        self.metrics.metrics['aggregations_performed'] += 1

        # Update metrics
        avg_staleness = np.mean([u.staleness for u in selected_updates])
        self.metrics.metrics['average_staleness'] = (
                0.9 * self.metrics.metrics['average_staleness'] + 0.1 * avg_staleness
        )
        high_quality_count = sum(1 for u in selected_updates
                                 if self._calculate_quality_score(u) > 0.7)
        self.metrics.metrics['high_quality_updates'] += high_quality_count

        # Update per-client reliability with a stable EMA so adaptive weighting
        # uses learned behavior instead of an almost constant prior.
        for u in selected_updates:
            loss_score = 1.0 / (1.0 + max(float(u.local_loss), 0.0))
            stale_score = self._alpha_staleness(getattr(u, "staleness", 0.0))
            observed_quality = 0.5 * loss_score + 0.5 * stale_score
            prev = float(self.client_contribution_scores.get(u.client_id, 1.0))
            self.client_contribution_scores[u.client_id] = float(np.clip(0.9 * prev + 0.1 * observed_quality, 0.05, 1.5))

        # Buffer wait-time observability for staleness-aware analysis.
        if selected_updates:
            now = time.time()
            waits = [max(0.0, now - float(u.timestamp)) for u in selected_updates]
            self.metrics.record_buffer_stats(len(selected_updates), wait_time_sec=float(np.mean(waits)))

    def _alpha_staleness(self, staleness: float) -> float:
        """Staleness decay alpha(tau), configurable for paper ablations."""
        tau = float(max(0.0, staleness))
        denom = max(float(self.max_staleness), 1e-8)
        base = max(float(self.staleness_floor), 1.0 - tau / denom)

        if self.staleness_mode == 'linear':
            return base
        if self.staleness_mode == 'exp':
            return max(float(self.staleness_floor), float(np.exp(-tau / denom)))
        # default: quadratic decay
        return base ** 2

    def _calculate_quality_score(self, update: ClientUpdate) -> float:
        """Calculate update quality score"""
        # Staleness-aware quality component
        staleness_score = self._alpha_staleness(update.staleness)

        # Loss improvement score
        loss_score = 1.0 / (1.0 + update.local_loss) if update.local_loss > 0 else 0.5

        # Client reliability score
        client_score = self.client_contribution_scores.get(update.client_id, 0.5)

        # Combined score
        if not self.include_staleness_in_quality:
            return 0.5 * loss_score + 0.5 * client_score
        return 0.4 * staleness_score + 0.3 * loss_score + 0.3 * client_score

    def _calculate_weight(self, update: ClientUpdate) -> float:
        """
        Calculate aggregation weight for update.

        Formal shape:
          w_i = data_size_i * quality_i
        Staleness is included once through the quality score when enabled.
        """
        if not self.adaptive_weighting:
            return update.data_size

        quality_score = self._calculate_quality_score(update)
        return update.data_size * quality_score

class Scaffold(FederatedProtocol):
    """SCAFFOLD protocol: variance reduction with control variates"""

    def set_global_model(self, model_state: Dict[str, torch.Tensor]):
        """Initialize global control variate shapes from the initial model."""
        super().set_global_model(model_state)
        # Initialize server control variate c with zeros (same parameter shapes).
        if self.global_model is None:
            self.c_global = {}
            return
        self.c_global = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.global_model.items()}

    def get_scaffold_controls(
        self,
        client_id: str,
        account_communication: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return (c, c_i) for a given client. Initialized lazily."""
        # Ensure c_global exists.
        if not getattr(self, "c_global", None) and self.global_model is not None:
            self.c_global = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.global_model.items()}

        if client_id not in self.client_controls or not self.client_controls[client_id]:
            self.client_controls[client_id] = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.c_global.items()}
        else:
            # Backfill any missing keys (e.g., after architecture changes).
            for k, v in self.c_global.items():
                if k not in self.client_controls[client_id]:
                    self.client_controls[client_id][k] = torch.zeros_like(v, dtype=torch.float32)

        if account_communication and getattr(self, "account_control_downlink", True):
            # Account explicit downlink of control variates c and c_i.
            downlink_bytes = (
                self.calculate_tensor_dict_size(self.c_global)
                + self.calculate_tensor_dict_size(self.client_controls[client_id])
            )
            self.metrics.add_overhead_communication(
                downlink_bytes,
                bucket="control_payload_downlink_mb",
            )
        return self.c_global, self.client_controls[client_id]

    def configure(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1.0)
        self.max_round_time = kwargs.get('max_round_time', 30.0)
        self.strict_reproduction = kwargs.get('strict_reproduction', False)

        # NEW: participation rate for sync rounds (default 50%)
        self.round_participation_rate = kwargs.get(
            'participation_rate',
            1.0 if self.strict_reproduction else 0.5
        )
        self.use_timeout = kwargs.get('use_timeout', not self.strict_reproduction)
        self.fixed_round_size = max(1, kwargs.get(
            'fixed_round_size',
            int(self.num_clients * self.round_participation_rate)
        ))

        # Optional compression support (align with other protocols)
        self.compression = kwargs.get('compression', None)
        if self.compression == "topk":
            self.compressor = TopKCompressor(kwargs.get('k', 100))
        elif self.compression == "signsgd":
            self.compressor = SignSGDCompressor()
        elif self.compression == "qsgd":
            self.compressor = QSGDCompressor(kwargs.get('num_bits', 8))
        else:
            self.compressor = None
        # Communication accounting switches for canonical SCAFFOLD control payloads.
        self.account_control_uplink = bool(kwargs.get("account_control_uplink", True))
        self.account_control_downlink = bool(kwargs.get("account_control_downlink", True))

        # Global control variate (same shape as model)
        self.c_global = {}
        self.client_controls = defaultdict(dict)

        # Round state
        self.current_round = 0
        self.round_buffer = []
        self.round_start_time = time.time()

    def receive_update(self, update: ClientUpdate) -> Tuple[bool, int]:
        with self._lock:
            control_payload_bytes = (
                self.calculate_tensor_dict_size(update.scaffold_control_payload or {})
                if getattr(self, "account_control_uplink", True)
                else 0
            )
            if update.model_version != self.current_round:
                self.metrics.update_communication(
                    self.calculate_update_size(update.update_data),
                    accepted=False
                )
                if control_payload_bytes > 0:
                    self.metrics.add_overhead_communication(
                        control_payload_bytes,
                        bucket="control_payload_uplink_mb",
                    )
                return False, self.current_round

            # Optional compression before storing
            if hasattr(self, "compressor") and self.compressor is not None:
                compressed_update = {}
                for name, delta in update.update_data.items():
                    comp, shape = self.compressor.compress(delta)
                    compressed_update[name] = (comp, shape)
                update.update_data = compressed_update

            # Buffer the update
            self.round_buffer.append(update)

            # Record communication
            update_size = self.calculate_update_size(update.update_data)
            self.metrics.update_communication(update_size, accepted=True)
            if control_payload_bytes > 0:
                self.metrics.add_overhead_communication(
                    control_payload_bytes,
                    bucket="control_payload_uplink_mb",
                )

            # Aggregate when enough clients joined or timeout
            if self.strict_reproduction:
                should_aggregate = len(self.round_buffer) >= self.fixed_round_size
            else:
                min_clients = max(2, int(self.num_clients * self.round_participation_rate))
                timeout_reached = (time.time() - self.round_start_time) >= self.max_round_time
                should_aggregate = len(self.round_buffer) >= min_clients or (
                    self.use_timeout and timeout_reached and len(self.round_buffer) >= 2
                )

            if should_aggregate:
                self.aggregate_updates()
                return True, self.current_round + 1

            return True, self.current_round

    def aggregate_updates(self):
        """Canonical SCAFFOLD: update global model + control variates."""
        if not self.round_buffer:
            return

        # Decompress if needed
        decompressed_buffer = []
        for update in self.round_buffer:
            if hasattr(self, "compressor") and self.compressor is not None:
                decompressed_update = {}
                for name, maybe_compressed in update.update_data.items():
                    # Only decompress if value is a (comp, shape) two-tuple
                    if isinstance(maybe_compressed, tuple) and len(maybe_compressed) == 2:
                        comp, shape = maybe_compressed
                        decompressed_update[name] = self.compressor.decompress(comp, shape)
                    else:
                        # Already a tensor (or plain value) — keep as is
                        decompressed_update[name] = maybe_compressed
                update.update_data = decompressed_update
            decompressed_buffer.append(update)

        total_data_size = sum(u.data_size for u in decompressed_buffer)
        aggregated = {}

        # FedAvg aggregation (deltas)
        for update in decompressed_buffer:
            weight = update.data_size / total_data_size
            for name, delta in update.update_data.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(delta, dtype=torch.float32)
                aggregated[name] += weight * delta.float()

        # Apply aggregated updates to global model
        if self.global_model is None:
            self.global_model = {}
            for name, param in aggregated.items():
                self.global_model[name] = param.clone()
        else:
            for name, param in aggregated.items():
                if name in self.global_model:
                    self.global_model[name] += self.learning_rate * param
                else:
                    self.global_model[name] = self.learning_rate * param

        # Canonical control variate updates.
        # Client update uses: c_i <- c_i - c + (1/(K*eta)) * (w - w_i)
        # Here (w_i - w) == delta, so (w - w_i) == -delta.
        m = max(1, len(decompressed_buffer))
        c_global_old = {k: v.clone() for k, v in self.c_global.items()}
        control_delta_sum: Dict[str, torch.Tensor] = {
            k: torch.zeros_like(v, dtype=torch.float32) for k, v in c_global_old.items()
        }

        for update in decompressed_buffer:
            client_id = update.client_id
            c_old, c_i_old = self.get_scaffold_controls(client_id, account_communication=False)
            # Use server control c as of the start of the round.
            c_start = c_global_old

            K = int(update.local_steps) if update.local_steps is not None else 1
            eta = float(update.local_lr) if update.local_lr is not None else 1.0
            denom = max(float(K) * float(eta), 1e-12)

            # Ensure client control has all keys.
            for name, delta in update.update_data.items():
                if name not in c_i_old:
                    c_i_old[name] = torch.zeros_like(delta, dtype=torch.float32)
                if name not in c_start:
                    c_start[name] = torch.zeros_like(delta, dtype=torch.float32)
                    c_global_old[name] = c_start[name].clone()
                    control_delta_sum[name] = torch.zeros_like(delta, dtype=torch.float32)

                payload = update.scaffold_control_payload or {}
                if name in payload:
                    delta_c = payload[name].float()
                    c_i_new = c_i_old[name] + delta_c
                else:
                    # Backward-compatible fallback when explicit control payload is absent.
                    delta_f = delta.float()
                    c_i_new = c_i_old[name] - c_start[name] - (delta_f / denom)

                control_delta_sum[name] += (c_i_new - c_i_old[name])
                c_i_old[name] = c_i_new

            self.client_controls[client_id] = c_i_old

        # Server update: c <- c + (1/m) * sum_i (c_i_new - c_i_old)
        for name, c_old_v in c_global_old.items():
            self.c_global[name] = c_old_v + (control_delta_sum[name] / float(m))

        self.metrics.metrics['aggregations_performed'] += 1
        self.current_round += 1
        self.model_version = self.current_round
        self.round_buffer.clear()
        self.round_start_time = time.time()

        logger.info(f"SCAFFOLD Round {self.current_round} completed with {len(self.client_controls)} clients")

# Factory function to create protocols
def create_protocol(protocol_name: str, num_clients: int, **kwargs) -> FederatedProtocol:
    """Factory function to create protocol instances with backward-compatible aliasing"""
    name = protocol_name.lower().strip()

    # Backward-compatible aliasing:
    # - Accept plain "improved"
    # - Accept any variant starting with "improved" (e.g., "improved_async_topk")
    if name.startswith('improved'):
        name = 'improved_async'

    protocols = {
        'fedavg': SyncFedAvg,
        'fedasync': AsyncFedAvg,
        'fedbuff': FedBuff,
        'improved_async': ImprovedAsyncProtocol,
        'scaffold': Scaffold,
    }

    if name not in protocols:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    return protocols[name](num_clients, **kwargs)
