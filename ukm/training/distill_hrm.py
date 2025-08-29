import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KDSchedule:
    alpha_i: float = 0.8
    alpha_f: float = 0.5
    delta_alpha: float = 0.05
    T_i: float = 2.0
    T_f: float = 1.0
    delta_T: float = 0.05

    def step(self, k: int, epoch: int) -> Tuple[float, float]:
        # Logarithmic decay in step, clamped per epoch
        alpha_k = self.alpha_f + (self.alpha_i - self.alpha_f) / (1.0 + math.log(k + 1))
        T_k = self.T_f + (self.T_i - self.T_f) / (1.0 + math.log(k + 1))
        alpha_k = max(self.alpha_f, alpha_k - epoch * self.delta_alpha)
        T_k = max(self.T_f, T_k - epoch * self.delta_T)
        return alpha_k, T_k


class ProjectionHead(nn.Module):
    """Projects teacher hidden width to student width before Frobenius matching."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillTrainer(nn.Module):
    """
    Generic KD trainer to distill a Transformer teacher into an HRM/xLSTM student.
    Student must return logits (B,L,V). Teacher provides logits and hidden states.
    """

    def __init__(
        self,
        student: nn.Module,
        vocab_size: int,
        teacher_hidden_dim: Optional[int] = None,
        student_hidden_dim: Optional[int] = None,
        schedule: Optional[KDSchedule] = None,
        beta: float = 0.1,
    ) -> None:
        super().__init__()
        self.student = student
        self.vocab_size = vocab_size
        self.schedule = schedule or KDSchedule()
        self.beta = beta
        self.k = 0
        self.epoch = 0
        self.proj: Optional[ProjectionHead] = None
        if teacher_hidden_dim is not None and student_hidden_dim is not None:
            self.proj = ProjectionHead(teacher_hidden_dim, student_hidden_dim)

    def kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
        # KL(student || teacher) with temperature T
        p = F.log_softmax(student_logits / T, dim=-1)
        q = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(p, q, reduction='batchmean')

    def frobenius_loss(self, student_hidden: torch.Tensor, teacher_hidden: torch.Tensor) -> torch.Tensor:
        # teacher_hidden and student_hidden: (B,L,D)
        if self.proj is not None:
            teacher_hidden = self.proj(teacher_hidden)
        diff = student_hidden - teacher_hidden
        denom = math.sqrt(max(1, diff.size(-1)))
        return (diff.pow(2).sum() / denom) / max(1, diff.numel())

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_hidden: Optional[torch.Tensor] = None,
        student_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.student.train()
        alpha, T = self.schedule.step(self.k, self.epoch)
        beta = self.beta
        assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0 and (1 - alpha - beta) >= 0.0

        outputs = self.student(input_ids)
        if isinstance(outputs, dict):
            student_logits = outputs['logits']
            student_hidden_out = outputs.get('hidden', None)
        else:
            student_logits = outputs
            student_hidden_out = student_hidden

        # Cross-entropy on labels
        ce = F.cross_entropy(student_logits.view(-1, self.vocab_size), labels.view(-1))
        # KD term
        kl = self.kd_loss(student_logits, teacher_logits, T)
        # Hidden Frobenius
        frob = torch.tensor(0.0, device=student_logits.device)
        if teacher_hidden is not None and student_hidden_out is not None:
            frob = self.frobenius_loss(student_hidden_out, teacher_hidden)

        loss = (1 - alpha - beta) * ce + alpha * (T * T) * kl + beta * frob
        loss.backward()

        self.k += 1
        return {
            'loss': float(loss.detach().item()),
            'ce': float(ce.detach().item()),
            'kl': float(kl.detach().item()),
            'frob': float(frob.detach().item()),
            'alpha': alpha,
            'T': T,
        }

