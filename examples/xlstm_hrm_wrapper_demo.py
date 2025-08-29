import torch
from xlstm_official_full.xlstm_block_stack import xLSTMBlockStackConfig
from xlstm_official_full.blocks.slstm.block import sLSTMBlockConfig
from src.lnn_hrm.xlstm_hrm import HRMXLSTM


def main():
    B, L, D = 2, 16, 32
    slcfg = sLSTMBlockConfig()
    # set required dimensions
    slcfg.slstm.embedding_dim = D
    slcfg.slstm.dropout = 0.0
    cfg = xLSTMBlockStackConfig(
        num_blocks=2,
        embedding_dim=D,
        dropout=0.0,
        slstm_block=slcfg,
        slstm_at="all",  # simplest: only sLSTM blocks
    )
    model = HRMXLSTM(cfg)
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(dev)
    x = torch.randn(B, L, D, device=dev)
    with torch.no_grad():
        y, telem = model(x)
    print('y', tuple(y.shape))
    print('telem', telem)


if __name__ == '__main__':
    main()
