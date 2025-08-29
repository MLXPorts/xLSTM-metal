import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryCube(nn.Module):
    """Per-block associative memory for residual predictions.

    Stores (keys, values) with cosine similarity retrieval. Values typically encode residuals Δy.
    """

    def __init__(self, d_key: int, d_val: int, max_items: int = 65536, topk: int = 8, device: str = "cpu"):
        super().__init__()
        self.register_buffer("keys", torch.empty(0, d_key, device=device))
        self.register_buffer("vals", torch.empty(0, d_val, device=device))
        self.max_items = int(max_items)
        self.topk = int(topk)

    @torch.no_grad()
    def query(self, q: torch.Tensor):
        """Query keys with q [Q, d_key]; return (pred [Q, d_val], conf [Q])."""
        if self.keys.numel() == 0:
            d_val = self.vals.size(-1) if self.vals.numel() else q.size(-1)
            return torch.zeros(q.size(0), d_val, device=q.device), torch.zeros(q.size(0), device=q.device)
        k = F.normalize(self.keys, dim=-1)
        qn = F.normalize(q, dim=-1)
        sims = qn @ k.T
        topv, topi = sims.topk(min(self.topk, sims.size(1)), dim=-1)
        weights = topv.softmax(dim=-1)
        gathered = self.vals.index_select(0, topi.reshape(-1)).view(q.size(0), -1, self.vals.size(-1))
        pred = torch.einsum("qb,qbd->qd", weights, gathered)
        conf = topv.mean(dim=-1).clamp(0, 1)
        return pred, conf

    @torch.no_grad()
    def update(self, k_new: torch.Tensor, v_new: torch.Tensor):
        keys = torch.cat([self.keys, k_new], dim=0)
        vals = torch.cat([self.vals, v_new], dim=0)
        if keys.size(0) > self.max_items:
            keys, vals = keys[-self.max_items:], vals[-self.max_items:]
        self.keys = keys
        self.vals = vals

