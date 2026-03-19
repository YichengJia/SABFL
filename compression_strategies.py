"""
compression_strategies.py
Implements gradient compression strategies for federated learning.
"""

import torch
import numpy as np
from typing import Tuple, Dict

class Compressor:
    """Base class for compressors"""
    def compress(self, tensor: torch.Tensor):
        raise NotImplementedError

    def decompress(self, compressed, shape):
        raise NotImplementedError


class TopKCompressor:
    """
    Top-K compressor that returns a compact representation:
    ((indices:int32 ndarray, values:float32 ndarray), original_shape)
    Decompress restores a dense tensor with zeros for non-topk positions.
    """

    def __init__(self, k: int):
        self.k = int(k)
        # Error-feedback memory to recover information discarded by Top-K.
        self._residual_memory: Dict[str, torch.Tensor] = {}
        self.compression_stats = {
            "total_elements": 0,
            "compressed_elements": 0,
            "theoretical_ratio": 0.0
        }
        self.debug = False  # set True to enable diagnostic prints

    def compress(
        self,
        tensor: torch.Tensor,
        key: str = "__global__",
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], torch.Size]:
        """Return ((indices_np, values_np), original_shape)."""
        # Safe flatten (reshape is robust to non-contiguous)
        tensor_flat = tensor.reshape(-1)
        memory_key = str(key)
        residual = self._residual_memory.get(memory_key)
        if residual is not None and residual.numel() == tensor_flat.numel():
            tensor_flat = tensor_flat + residual.to(tensor_flat.device, dtype=tensor_flat.dtype)
        num_elements = tensor_flat.numel()

        effective_k = min(max(self.k, 0), num_elements)
        if num_elements == 0 or effective_k == 0:
            self._residual_memory[memory_key] = tensor_flat.detach().clone()
            return (np.array([], dtype=np.int32), np.array([], dtype=np.float32)), tensor.shape

        if effective_k > num_elements * 0.3 and self.debug:
            print(f"[TopK] k={effective_k} is {effective_k/num_elements:.1%} of tensor, compression may be inefficient")

        # Select by magnitude
        values_topk, indices_topk = torch.topk(tensor_flat.abs(), effective_k, sorted=False)
        actual_values = tensor_flat[indices_topk]

        indices_np = indices_topk.detach().cpu().numpy().astype(np.int32)
        values_np = actual_values.detach().cpu().numpy().astype(np.float32)

        # Store dropped information as residual for the next compression of same key.
        reconstructed = torch.zeros_like(tensor_flat)
        reconstructed[indices_topk] = actual_values
        self._residual_memory[memory_key] = (tensor_flat - reconstructed).detach().cpu()

        # Update stats
        self.compression_stats["total_elements"] += num_elements
        self.compression_stats["compressed_elements"] += effective_k
        self.compression_stats["theoretical_ratio"] = (num_elements / effective_k) if effective_k > 0 else 0.0

        if self.debug:
            ratio = (num_elements * 4) / float(indices_np.nbytes + values_np.nbytes + 1e-8)
            print(f"[TopK] original={num_elements*4:.0f}B, compressed={indices_np.nbytes+values_np.nbytes:.0f}B, ratio={ratio:.2f}x")

        return (indices_np, values_np), tensor.shape

    def decompress(self, compressed: Tuple[np.ndarray, np.ndarray], shape: torch.Size, device: str = "cpu") -> torch.Tensor:
        """Restore dense tensor with zeros elsewhere."""
        indices_np, values_np = compressed
        num_elements = int(np.prod(shape))
        out = torch.zeros(num_elements, dtype=torch.float32, device=device)
        if indices_np.size > 0:
            idx = torch.from_numpy(indices_np).long().to(device)
            val = torch.from_numpy(values_np).float().to(device)
            out[idx] = val
        return out.view(shape)

    def get_stats(self) -> Dict:
        return dict(self.compression_stats)


class SignSGDCompressor(Compressor):
    """True 1-bit sign compression with magnitude scaling."""
    def __init__(self, scale: str = "median"):
        # scale ∈ {"mean","median"}; median is more robust
        self.scale = scale

    def compress(self, tensor: torch.Tensor):
        # Flatten and compute scale
        flat = tensor.reshape(-1).detach()  # use reshape() for non-contiguous tensors
        if flat.numel() == 0:
            packed = np.frombuffer(b"", dtype=np.uint8)
            meta = (flat.numel(), 0.0)
            return (packed, meta), tensor.shape

        if self.scale == "median":
            mag = flat.abs().median().item()
        else:
            mag = flat.abs().mean().item()
        # Avoid zero scaling
        if mag == 0.0:
            mag = 1e-8

        # Pack signs into bits: 1 -> 1, -1/0 -> 0
        bits = (flat >= 0).to(torch.uint8).cpu().numpy()  # 0/1 per element
        packed = np.packbits(bits)  # uint8 array, length = ceil(N/8)

        # meta carries (num_elements, magnitude)
        meta = (int(flat.numel()), float(mag))
        return (packed, meta), tensor.shape

    def decompress(self, compressed, shape):
        packed, meta = compressed
        n, mag = meta
        if n == 0:
            return torch.zeros(shape, dtype=torch.float32)

        # Unpack bits back to 0/1 then map to {-1,+1}
        bits = np.unpackbits(packed)[:n]
        signs = (bits * 2 - 1).astype(np.int8)  # 0->-1, 1->+1
        out = torch.from_numpy(signs).to(torch.float32) * float(mag)
        return out.view(shape)


class QSGDCompressor(Compressor):
    """Quantized SGD"""
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits

    def compress(self, tensor: torch.Tensor):
        tensor_flat = tensor.reshape(-1)
        norm = torch.norm(tensor_flat)

        if norm.item() == 0:
            return (np.zeros_like(tensor_flat.cpu().numpy()), 0.0, 1.0), tensor.shape

        q_levels = 2 ** self.num_bits
        scale = norm / q_levels
        scaled = torch.abs(tensor_flat) / scale
        probs = scaled - torch.floor(scaled)
        quantized = torch.floor(scaled) + torch.bernoulli(probs)
        signs = torch.sign(tensor_flat)

        values = (quantized * signs).to(torch.float32).cpu().numpy()
        return (values, norm.item(), scale.item()), tensor.shape

    def decompress(self, compressed, shape):
        values, norm, scale = compressed
        tensor_flat = torch.tensor(values, dtype=torch.float32) * scale
        return tensor_flat.view(shape)
