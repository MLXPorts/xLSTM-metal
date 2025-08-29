import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class MemoryCube(nn.Module):
    """
    Lightweight content-addressable memory with cosine top‑K retrieval.

    - Keys/values are stored in fixed-capacity tensors (ring buffer).
    - Query uses normalized cosine similarity + softmax over top‑K hits.
    - Returns (value, confidence, indices, scores) for gating/audits.

    Fused keys: optionally blend dense keys with sparse "spike/comb" keys.
    Confidence: concentration of attention over top‑K (Gini-like proxy).
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        capacity: int = 1024,
        topk: int = 8,
        fuse_weight: float = 0.3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        self.topk = topk
        self.fuse_weight = fuse_weight

        dev = device or torch.device("cpu")
        self.register_buffer("keys", torch.zeros(capacity, key_dim, device=dev))
        self.register_buffer("values", torch.zeros(capacity, value_dim, device=dev))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long, device=dev))
        self.register_buffer("ptr", torch.zeros((), dtype=torch.long, device=dev))

        # Small encoder to turn spikes into the key space (optional)
        self.spike_encoder = nn.Linear(key_dim, key_dim)

    @torch.no_grad()
    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.filled.zero_()
        self.ptr.zero_()

    def _fuse_key(self, dense_key: torch.Tensor, spike_key: Optional[torch.Tensor]) -> torch.Tensor:
        if spike_key is None:
            return dense_key
        # Project spike pattern into key space and blend
        enc = self.spike_encoder(spike_key)
        return _l2_normalize((1.0 - self.fuse_weight) * dense_key + self.fuse_weight * enc)

    @torch.no_grad()
    def update(self, key: torch.Tensor, value: torch.Tensor, spike_key: Optional[torch.Tensor] = None) -> None:
        """
        Insert a single (key, value) pair using a ring buffer policy.
        key: (Dk) or (B,Dk) — batches are enqueued row-wise
        value: (Dv) or (B,Dv)
        spike_key: optional sparse vector in key space
        """
        if key.dim() == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            if spike_key is not None and spike_key.dim() == 1:
                spike_key = spike_key.unsqueeze(0)

        key = _l2_normalize(key)
        if spike_key is not None:
            key = self._fuse_key(key, spike_key)

        B = key.size(0)
        for i in range(B):
            idx = int(self.ptr.item())
            self.keys[idx] = key[i]
            self.values[idx] = value[i]
            self.ptr.copy_(((self.ptr + 1) % self.capacity))
            self.filled.copy_(torch.clamp(self.filled + 1, max=self.capacity))

    def query(
        self,
        q: torch.Tensor,
        spike_key: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
        temperature: float = 0.1,
    ) -> Tuple[torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """
        Query memory with a key.
        q: (Dk) or (B,Dk)
        returns: (value, confidence, indices, scores)
        - value: (Dv) or (B,Dv)
        - confidence: float (batch-mean concentration metric)
        - indices: top‑K indices used
        - scores: attention weights over top‑K
        """
        k = topk or self.topk
        if self.filled.item() == 0:
            # No content yet — return zeros
            v = torch.zeros(q.shape[:-1] + (self.value_dim,), device=q.device, dtype=q.dtype)
            return v, 0.0, torch.empty(0, dtype=torch.long, device=q.device), torch.empty(0, device=q.device)

        if q.dim() == 1:
            q = q.unsqueeze(0)
            if spike_key is not None and spike_key.dim() == 1:
                spike_key = spike_key.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        qn = _l2_normalize(q)
        if spike_key is not None:
            qn = self._fuse_key(qn, spike_key)

        # Use only the filled prefix of memory
        N = int(self.filled.item())
        keys = _l2_normalize(self.keys[:N])  # (N,Dk)
        sims = torch.matmul(qn, keys.t())  # (B,N)
        top_vals, top_idx = torch.topk(sims, k=min(k, N), dim=-1)
        attn = (top_vals / max(temperature, 1e-4)).softmax(dim=-1)  # (B,K)
        vals = self.values[:N][top_idx]  # (B,K,Dv)
        out = torch.einsum("bk,bkd->bd", attn, vals)

        # Confidence: 1 - normalized entropy over top‑K
        p = torch.clamp(attn, min=1e-8)
        H = -(p * p.log()).sum(dim=-1)
        H_max = torch.log(torch.tensor(p.size(-1), device=p.device, dtype=p.dtype))
        conf = (1.0 - (H / (H_max + 1e-8))).mean().item()

        if squeeze:
            return out.squeeze(0), float(conf), top_idx.squeeze(0), attn.squeeze(0)
        return out, float(conf), top_idx, attn

    @staticmethod
    def spike_comb(x: torch.Tensor, bins: int = 32, threshold: float = 0.0) -> torch.Tensor:
        """Simple spike/comb encoder: binarize features into a sparse comb pattern."""
        # Center, then binarize by sign with optional threshold
        xz = x - x.mean(dim=-1, keepdim=True)
        if threshold > 0:
            mask = xz.abs() > threshold * xz.abs().amax(dim=-1, keepdim=True)
            spikes = torch.where(mask, (xz > 0).float(), torch.zeros_like(xz))
        else:
            spikes = (xz > 0).float()
        # Downsample to target bins via average pooling then threshold
        if spikes.size(-1) > bins:
            stride = spikes.size(-1) // bins
            pooled = F.avg_pool1d(spikes.unsqueeze(1), kernel_size=stride, stride=stride)
            spikes = (pooled.squeeze(1) > 0.5).float()
        return spikes

    def audit_snapshot(self) -> Dict[str, torch.Tensor]:
        """Return a non‑grad snapshot for external logging."""
        with torch.no_grad():
            N = int(self.filled.item())
            return {
                "keys": self.keys[:N].detach().clone().cpu(),
                "values": self.values[:N].detach().clone().cpu(),
            }

