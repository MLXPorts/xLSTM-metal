import torch


def energy(t: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """Compute an energy-like metric ||t||^2 over the last dim, then reduce.

    For (B,L,D): returns scalar per batch if reduce==mean; otherwise returns (B,L).
    """
    if t is None:
        return torch.tensor(0.0, device="cpu")
    e = (t.float() ** 2).sum(dim=-1)
    if reduce == "mean":
        return e.mean()
    return e

