# MLX xLSTM Testing Guide

Complete test suite for validating xLSTM-Metal correctness and performance.

## Quick Start

Run the complete test suite:

```bash
python run_pytest.py
```

This runs all tests with proper configuration and environment setup.

## Test Categories

### Kernel Parity Tests

Validate low-level MLX kernels against reference implementations:

```bash
# Run specific kernel tests
PYTHONPATH=. pytest -v tests/test_mlx_gemm.py
PYTHONPATH=. pytest -v tests/test_mlx_qr_kernels.py
PYTHONPATH=. pytest -v tests/test_mlx_svd_kernels.py
PYTHONPATH=. pytest -v tests/test_mlx_ivf_kernels.py
```

**Acceptance Criteria:** Maximum absolute difference ≤ 1e-4 for float32 comparisons.

### xLSTM Inference Parity

Validate end-to-end xLSTM implementation:

```bash
# Inference parity tests
PYTHONPATH=. pytest -v tests/test_xlstm_mlx_inference_parity.py

# Batch decoding tests
PYTHONPATH=. pytest -v tests/test_xlstm_mlx_batch_decode.py

# Core xLSTM tests
PYTHONPATH=. pytest -v tests/test_xlstm.py
```

**What's Tested:**

- Token generation determinism (with fixed seeds)
- Batch prefill/decode shape correctness
- State management across blocks
- Argmax decode consistency
- Fast head ON vs OFF parity

### Integration Tests

Test the complete system including model loading and generation:

```bash
# Test model loading and inference
python generate.py --model ./test_model --info

# Test with small sequence
python generate.py --model ./test_model --prompt "test" --max-tokens 10
```

## Test Organization

```
tests/
├── test_xlstm.py                       # Core xLSTM functionality
├── test_xlstm_mlx_inference_parity.py  # End-to-end parity
├── test_xlstm_mlx_batch_decode.py      # Batch processing
├── test_mlx_gemm.py                    # GEMM kernels
├── test_mlx_qr_kernels.py              # QR decomposition
├── test_mlx_svd_kernels.py             # SVD kernels
├── test_mlx_ivf_kernels.py             # IVF kernels
└── run_tests.py                        # Test runner utility
```

## Best Practices

### Determinism

Always use fixed seeds for reproducible tests:

```python
import mlx.core as mx

# Set seed at test start
mx.random.seed(42)
```

### Numerical Precision

For large vocabulary tests (32k-50k tokens):

- Allow tiny numeric jitter in logits (±1e-4)
- Use argmax decode to compare token sequences
- Token-level equality is the primary criterion

### Test Data

Keep test cases small and fast:

- Use small sequence lengths (≤ 64 tokens)
- Use reduced vocab sizes when possible
- Mock large models with minimal configurations

## Writing New Tests

### Test Structure

```python
import mlx.core as mx
import pytest
from xlstm_metal import xLSTMRunner

def test_example():
    """Test description."""
    # Setup
    mx.random.seed(42)
    
    # Test logic
    runner = xLSTMRunner("model_path")
    output = runner.generate([1, 2, 3], max_tokens=10)
    
    # Assertions
    assert len(output) == 13  # 3 prompt + 10 generated
    assert all(isinstance(t, int) for t in output)
```

### Naming Conventions

- Test files: `test_<component>.py`
- Test functions: `test_<feature>_<scenario>()`
- Fixtures: `@pytest.fixture` for shared setup

### Common Assertions

```python
# Shape validation
assert output.shape == expected_shape

# Numerical tolerance
assert mx.abs(output - expected).max() < 1e-4

# Token sequence equality
assert output_tokens == expected_tokens

# State correctness
assert state is not None
assert len(state) == num_blocks
```

## Continuous Integration

Tests run automatically on:

- Pull requests to main branch
- Commits to feature branches
- Release tags

## Debugging Failed Tests

### Verbose Output

```bash
# Show detailed test output
pytest -vv tests/test_xlstm.py

# Show print statements
pytest -s tests/test_xlstm.py

# Stop on first failure
pytest -x tests/
```

### Interactive Debugging

```python
# Add breakpoint in test
def test_example():
    output = generate_text()
    breakpoint()  # Drops into debugger
    assert output == expected
```

### Isolate Failures

```bash
# Run single test
pytest tests/test_xlstm.py::test_specific_function

# Run tests matching pattern
pytest -k "test_inference" tests/
```

## Performance Testing

For performance benchmarks (not run by default):

```bash
# Run with performance markers
pytest -m performance tests/

# Generate timing reports
pytest --durations=10 tests/
```

## Coverage Reports

Generate test coverage reports:

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage
pytest --cov=xlstm_metal --cov-report=html tests/

# View report
open htmlcov/index.html
```

## Notes

- Keep seeds fixed (`mx.random.seed`) for deterministic argmax sequences
- For large vocab tests, numeric jitter in logits is acceptable
- Parity tests compare tokens, not raw logits
- All tests should pass on Apple Silicon M1/M2/M3/M4

## Related Documentation

- [MLX Guide](mlx_guide.md) - Implementation patterns
- [XLSTM Architecture](XLSTM_MLX_ARCHITECTURE.md) - System design
- [MLX Numerics Guide](MLX_NUMERICS_AND_DTYPE_GUIDE.md) - Precision handling

