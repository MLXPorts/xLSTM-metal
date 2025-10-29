import mlx.core as mx
from src.mlx_impl.xlstm_mlx import create_xlstm_model

if __name__ == "__main__":
    mx.random.seed(0)
    model = create_xlstm_model(vocab_size=256, num_layers=2, signature=(1,1), inp_dim=128, head_dim=32, head_num=4, dropout=0.0)
    B,T=2,8
    tokens = mx.random.randint(0, 256, (B,T))
    logits = model(tokens)
    print('logits shape:', logits.shape)
    print('dtype:', logits.dtype)
    print('mean/std:', float(mx.mean(logits)), float(mx.std(logits)))
