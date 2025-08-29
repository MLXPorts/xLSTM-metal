import torch
import torch.nn as nn
from pathlib import Path

from xlstm_official_full.xlstm_block_stack import xLSTMBlockStackConfig
from xlstm_official_full.blocks.slstm.block import sLSTMBlockConfig
from src.lnn_hrm.xlstm_hrm import HRMXLSTM
from src.lnn_hrm.training.ponder_trainer import PonderTrainer
from src.lnn_hrm.telemetry.logger import TelemetryLogger
from src.lnn_hrm.preflight import assert_mps, assert_ray


def make_batch(B=2, L=16, V=64, device='cpu'):
    input_ids = torch.randint(0, V, (B, L), device=device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    D = 32
    emb = nn.Embedding(V, D, device=device)
    x = emb(input_ids)
    return x, labels


def run_level(level: float, steps: int = 8):
    assert_mps(); assert_ray()
    dev = torch.device('mps')
    V = 64; D = 32
    slcfg = sLSTMBlockConfig(); slcfg.slstm.embedding_dim = D; slcfg.slstm.dropout = 0.0
    cfg = xLSTMBlockStackConfig(num_blocks=2, embedding_dim=D, dropout=0.0, slstm_block=slcfg, slstm_at="all")
    model = HRMXLSTM(cfg, k_5ht=0.6).to(dev)
    out_dir = Path(f'runs/5ht_sweep/level_{int(level*100):02d}')
    logger = TelemetryLogger(out_dir=str(out_dir), csv_name='run.csv', jsonl_name='run.jsonl')
    trainer = PonderTrainer(model, vocab_size=V, lambda_ponder=0.01, logger=logger, seed=777)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer.set_optimizer(opt)
    for step in range(steps):
        x, y = make_batch(device=dev)
        mod = torch.full((x.size(0), x.size(1)), fill_value=level, device=dev)
        metrics = trainer.step(x, y, step=step, mod_5ht=mod)
        print(f"5-HT={level:.2f} step {step:02d} loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} ponder={metrics['ponder']:.4f}")
    logger.close()


def main():
    for lvl in [0.0, 0.5, 1.0]:
        run_level(lvl, steps=8)


if __name__ == '__main__':
    main()

