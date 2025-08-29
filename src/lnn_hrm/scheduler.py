import torch


def z5_slots(times: torch.Tensor) -> torch.Tensor:
    """Return Z5 envelope slot indices in {0,1,2,3,4} for given time steps.

    times: (B, L) float or int tensor of monotonically increasing steps.
    """
    return (times.to(torch.long) % 5).to(torch.int64)


def boundary_commit_mask(times: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask (B, L) that is True on boundary commit steps.

    We define slot 4 → 0 rollover as the control carry. For simple sequences, we
    mark steps whose slot equals 4 as the commit boundary.
    """
    slots = z5_slots(times)
    return slots.eq(4)

