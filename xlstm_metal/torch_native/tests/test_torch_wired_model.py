"""Basic tests for torch-native WiredxLSTM backend (sanity).
Run with: pytest -q xlstm_metal/torch_native/tests/test_torch_wired_model.py
"""
from pathlib import Path
import torch

from xlstm_metal.torch_native.models.wired_xlstm import WiredxLSTM


def _get_model_dir():
    p = Path("xlstm_7b_model")
    if not p.exists():
        # Allow test skip gracefully
        return None
    return p


def test_instantiate_without_weights():
    model_dir = _get_model_dir()
    if model_dir is None:
        import pytest
        pytest.skip("Model directory not present")
    model = WiredxLSTM.from_pretrained(model_dir, load_weights=False)
    assert len(model.blocks) == model.wiring.structure['num_blocks']
    assert model.embedding is not None
    assert model.lm_head is not None


def test_forward_shapes():
    model_dir = _get_model_dir()
    if model_dir is None:
        import pytest
        pytest.skip("Model directory not present")
    model = WiredxLSTM.from_pretrained(model_dir, load_weights=False)
    B, S = 2, 5
    input_ids = torch.randint(low=0, high=model.config['vocab_size'], size=(B, S))
    logits = model(input_ids)
    assert logits.shape == (B, S, model.config['vocab_size'])


def test_weight_tying_flag():
    model_dir = _get_model_dir()
    if model_dir is None:
        import pytest
        pytest.skip("Model directory not present")
    model = WiredxLSTM.from_pretrained(model_dir, load_weights=False)
    if model.tie_word_embeddings and model.embedding is not None and model.lm_head is not None:
        # Same storage if tied
        assert model.lm_head.weight is model.embedding.weight


