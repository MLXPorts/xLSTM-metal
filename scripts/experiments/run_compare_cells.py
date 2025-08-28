#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch

from mlstm_kernels.torch.experiments.reversible_cell_experiment import (
    SLSTMHead,
    CfCHead,
    ReversibleTauAccumulator,
    CfCTorch,
    TrainConfig,
    train_one,
    quick_eval,
    device_auto,
)


def main():
    device = device_auto()
    outdir = Path("runs/experiments")
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig()

    results = {}
    # Baseline sLSTM head
    s_model = SLSTMHead(hidden=cfg.hidden)
    s_hist = train_one(s_model, cfg, device)
    s_eval = quick_eval(s_model, device)
    results["sLSTM"] = {**s_hist, **s_eval}

    # CfC head (sigmoid)
    c_model = CfCHead(hidden=cfg.hidden, act_c="sigmoid", act_g="sigmoid")
    c_hist = train_one(c_model, cfg, device)
    c_eval = quick_eval(c_model, device)
    results["CfC"] = {**c_hist, **c_eval}

    # CfC head (LeCun tanh)
    c2_model = CfCHead(hidden=cfg.hidden, act_c="lecun_tanh", act_g="lecun_tanh")
    c2_hist = train_one(c2_model, cfg, device)
    c2_eval = quick_eval(c2_model, device)
    results["CfC_lecun_tanh"] = {**c2_hist, **c2_eval}

    # Reversible τ-accumulator
    r_model = ReversibleTauAccumulator(hidden=cfg.hidden, env=cfg.env)
    r_hist = train_one(r_model, cfg, device)
    r_eval = quick_eval(r_model, device)
    results["Reversible"] = {**r_hist, **r_eval}

    # CfC true (default mode, lecun_tanh backbone)
    cfc_true = CfCTorch(input_size=1, units=cfg.hidden, mode="default", backbone_units=64, backbone_layers=1, backbone_dropout=0.0, activation="lecun_tanh")
    # Wrap to produce 2‑logit readout via a tiny head for fairness
    class CfCTrueWrapper(torch.nn.Module):
        def __init__(self, core):
            super().__init__()
            self.core = core
            self.head = torch.nn.Linear(cfg.hidden, 2)
        def forward(self, x):
            hseq = self.core(x.unsqueeze(-1))  # (B,S,H)
            return self.head(hseq)
    cfcw = CfCTrueWrapper(cfc_true)
    cfcw.to(device)
    t_hist = train_one(cfcw, cfg, device)
    t_eval = quick_eval(cfcw, device)
    results["CfC_true"] = {**t_hist, **t_eval}

    out = outdir / "compare_cells_report.json"
    out.write_text(json.dumps({"device": str(device), "cfg": cfg.__dict__, "results": results}, indent=2))
    print("Report:", out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
