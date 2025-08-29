import torch
import torch.nn as nn
from xlstm_official_full.xlstm_block_stack import xLSTMBlockStackConfig
from xlstm_official_full.blocks.slstm.block import sLSTMBlockConfig
from src.lnn_hrm.xlstm_hrm import HRMXLSTM
from src.lnn_hrm.training.ponder_trainer import PonderTrainer
from src.lnn_hrm.telemetry.logger import TelemetryLogger
from src.lnn_hrm.preflight import assert_mps, assert_ray


def make_batch(B=2, L=16, V=64, device='cpu'):
    input_ids = torch.randint(0, V, (B, L), device=device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    # simple embedding to features D
    D = 32
    emb = nn.Embedding(V, D, device=device)
    x = emb(input_ids)
    return x, labels


def main():
    # Preflight: MPS + Ray required (no native fallback)
    assert_mps(); assert_ray()
    dev = torch.device('mps')
    V = 64; D = 32
    slcfg = sLSTMBlockConfig(); slcfg.slstm.embedding_dim = D; slcfg.slstm.dropout = 0.0
    cfg = xLSTMBlockStackConfig(num_blocks=2, embedding_dim=D, dropout=0.0, slstm_block=slcfg, slstm_at="all")
    model = HRMXLSTM(cfg).to(dev)
    logger = TelemetryLogger(out_dir='runs/telem_demo', csv_name='demo.csv', jsonl_name='demo.jsonl')
    trainer = PonderTrainer(model, vocab_size=V, lambda_ponder=0.01, logger=logger, seed=123)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer.set_optimizer(opt)
    for step in range(5):
        x, y = make_batch(device=dev)
        metrics = trainer.step(x, y, step=step)
        print(f"step {step:02d} loss={metrics['loss']:.4f} ce={metrics['ce']:.4f} ponder={metrics['ponder']:.4f}")
    logger.close()


if __name__ == '__main__':
    main()
