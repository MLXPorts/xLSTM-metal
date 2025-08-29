import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from ..telemetry.logger import TelemetryLogger
from ..telemetry.trace import trace_hash


class PonderTrainer:
    """Minimal trainer wrapper that adds a ponder loss using ACT telemetry.

    Expects model(x, times=...) -> (y, telem) where telem contains:
      - act_prob_mean or act_open_rate
    Loss: L = CE(y, labels) + λ · ponder_metric
    """

    def __init__(self, model: nn.Module, vocab_size: int, lambda_ponder: float = 0.01, logger: Optional[TelemetryLogger] = None, seed: Optional[int] = None):
        self.model = model
        self.vocab_size = vocab_size
        self.lambda_ponder = float(lambda_ponder)
        self.opt = None
        self.logger = logger
        self.seed = seed

    def set_optimizer(self, opt: torch.optim.Optimizer):
        self.opt = opt

    def step(self, input_ids: torch.Tensor, labels: torch.Tensor, step: int = 0) -> Dict[str, float]:
        assert self.opt is not None, "Call set_optimizer before training."
        self.model.train()
        self.opt.zero_grad()
        B, L = labels.shape
        # use token positions as times
        times = torch.arange(L, device=labels.device).unsqueeze(0).expand(B, -1)
        outputs, telem = self.model(input_ids.float(), times=times)
        logits = outputs  # assume model outputs logits or features mapped to vocab elsewhere
        if logits.size(-1) != self.vocab_size:
            # project to vocab for demo purposes
            proj = nn.Linear(logits.size(-1), self.vocab_size, device=logits.device)
            logits = proj(logits)
        ce = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        ponder_metric = telem.get("act_prob_mean", telem.get("act_open_rate", 0.0))
        ponder = torch.as_tensor(ponder_metric, device=labels.device, dtype=ce.dtype)
        loss = ce + self.lambda_ponder * ponder
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        metrics = {
            "loss": float(loss.detach().item()),
            "ce": float(ce.detach().item()),
            "ponder": float(ponder.detach().item()),
        }
        if self.logger is not None:
            # include telemetry + trace hash
            fields = {**telem, **metrics}
            fields['trace_hash'] = trace_hash(input_ids, times=times, seed=self.seed)
            self.logger.log(step=step, **fields)
        return metrics
