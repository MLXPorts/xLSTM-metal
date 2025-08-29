import os
import json
import shutil
from pathlib import Path
import torch


def test_trace_hash_determinism():
    from src.lnn_hrm.telemetry.trace import trace_hash
    x = torch.randn(2, 4, 3)
    times = torch.arange(4).unsqueeze(0).expand(2, -1)
    h1 = trace_hash(x, times=times, seed=123)
    h2 = trace_hash(x, times=times, seed=123)
    h3 = trace_hash(x, times=times, seed=124)
    assert h1 == h2 and h1 != h3


def test_ponder_trainer_writes_logs(tmp_path: Path):
    from xlstm_official_full.xlstm_block_stack import xLSTMBlockStackConfig
    from xlstm_official_full.blocks.slstm.block import sLSTMBlockConfig
    from src.lnn_hrm.xlstm_hrm import HRMXLSTM
    from src.lnn_hrm.training.ponder_trainer import PonderTrainer
    from src.lnn_hrm.telemetry.logger import TelemetryLogger
    import torch.nn as nn

    if not torch.backends.mps.is_available():
        import pytest
        pytest.skip("MPS required in this repo configuration")
    dev = torch.device('mps')

    # Tiny model
    V = 32; D = 16; B = 1; L = 8
    sl = sLSTMBlockConfig(); sl.slstm.embedding_dim = D; sl.slstm.dropout = 0.0
    cfg = xLSTMBlockStackConfig(num_blocks=1, embedding_dim=D, dropout=0.0, slstm_block=sl, slstm_at="all")
    model = HRMXLSTM(cfg).to(dev)
    logger = TelemetryLogger(out_dir=str(tmp_path), csv_name='log.csv', jsonl_name='log.jsonl')
    trainer = PonderTrainer(model, vocab_size=V, lambda_ponder=0.01, logger=logger, seed=7)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer.set_optimizer(opt)
    # Data
    emb = nn.Embedding(V, D).to(dev)
    input_ids = torch.randint(0, V, (B, L), device=dev)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    x = emb(input_ids)
    # One step
    _ = trainer.step(x, labels, step=0)
    logger.close()
    # Check files exist and have content
    csv_path = tmp_path / 'log.csv'
    jsonl_path = tmp_path / 'log.jsonl'
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert jsonl_path.exists() and jsonl_path.stat().st_size > 0
    # Basic JSONL sanity: last line parses as JSON and has fields
    with jsonl_path.open('r') as f:
        last = None
        for line in f:
            last = line
    rec = json.loads(last)
    for k in ["alpha_mean", "act_prob_mean", "energy_pre_gate", "trace_hash", "step", "ts"]:
        assert k in rec
