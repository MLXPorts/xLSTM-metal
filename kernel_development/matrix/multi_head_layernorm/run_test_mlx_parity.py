import mlx.core as mx
from src.mlx_impl.xlstm_mlx import create_xlstm_model
from tools.mlx_runtime import configure_model, reset_runtime_config


def _decode_argmax(model, tokens, new_tokens=16):
    logits, state = model(tokens, return_hidden=True)
    last = logits[:, -1, :]
    out_ids = tokens.tolist()[0]
    for _ in range(new_tokens):
        nxt = int(mx.argmax(last[0]))
        out_ids.append(nxt)
        step = mx.array([[nxt]], dtype=mx.int32)
        logits, state = model(step, hidden_states=state, return_hidden=True)
        last = logits[:, -1, :]
    return out_ids


if __name__ == "__main__":
    mx.random.seed(0)
    model = create_xlstm_model(vocab_size=1024, num_layers=4, signature=(1, 1), inp_dim=512, head_dim=64, head_num=8,
                               dropout=0.0)
    tokens = mx.random.randint(0, 1024, (1, 64))
    reset_runtime_config();
    configure_model(fast_head=False)
    seq_off = _decode_argmax(model, tokens)
    reset_runtime_config();
    configure_model(fast_head=True)
    seq_on = _decode_argmax(model, tokens)
    print('parity:', seq_on == seq_off)
