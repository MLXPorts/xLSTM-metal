"""Pytest configuration and shared fixtures for xLSTM-Metal tests."""

import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def model_cache_dir(project_root_path):
    """Return the model cache directory path."""
    return project_root_path / "model_cache"


@pytest.fixture
def sample_prompt():
    """Provide a sample prompt for testing."""
    return "The capital of France is"


@pytest.fixture
def test_device():
    """Determine which device to use for tests."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@pytest.fixture(scope="session")
def mlx_available():
    """Check if MLX is available."""
    try:
        import mlx.core as mx
        return True
    except ImportError:
        return False


@pytest.fixture(scope="session")
def pytorch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # Skip MLX tests if MLX is not available
    skip_mlx = pytest.mark.skip(reason="MLX not available")
    skip_pytorch = pytest.mark.skip(reason="PyTorch not available")

    try:
        import mlx.core as mx
        mlx_available = True
    except ImportError:
        mlx_available = False

    try:
        import torch
        pytorch_available = True
    except ImportError:
        pytorch_available = False

    for item in items:
        if not mlx_available and "mlx" in item.keywords:
            item.add_marker(skip_mlx)
        if not pytorch_available and "pytorch" in item.keywords:
            item.add_marker(skip_pytorch)
