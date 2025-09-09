
import mlx.core as mx
from implementations.mlx.xlstm_mlx import create_xlstm_model


def test_batch_decode_shapes():
    """Tests the shapes of the outputs of the batch decoding function."""
    mx.random.seed(0)
    model = create_xlstm_model(vocab_size=2048, num_layers=3, signature=(1,1), inp_dim=256, head_dim=64, head_num=4, dropout=0.0)
    B, T = 3, 50
    tokens = mx.random.randint(0, 2048, (B, T))
    logits, state = model(tokens, return_hidden=True)
    assert logits.shape == (B, T, 2048)
    assert isinstance(state, list) and len(state) == len(model.blocks)
    # One decode step
    step_in = mx.random.randint(0, 2048, (B, 1))
    logits2, state2 = model(step_in, hidden_states=state, return_hidden=True)
    assert logits2.shape == (B, 1, 2048)
    assert len(state2) == len(model.blocks)

