import torch


def energy(t: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    if t is None:
        return torch.tensor(0.0)
    e = (t.float() ** 2).sum(dim=-1)
    return e.mean() if reduce == "mean" else e

