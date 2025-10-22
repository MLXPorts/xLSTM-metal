#!/usr/bin/env python
"""
Export ChunkwiseMLSTM to CoreML for Apple Neural Engine (ANE) deployment.

TorchScript → CoreML export pathway:
1. Trace model to TorchScript using torch.jit.trace
2. Convert to CoreML using coremltools
3. Deploy to ANE for 10x faster inference and 14x lower memory

Requirements:
    pip install coremltools

References:
- Apple ANE Transformers: https://github.com/apple/ml-ane-transformers
- CoreML Tools: https://apple.github.io/coremltools/docs-guides/
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from pathlib import Path

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("Warning: coremltools not installed. Install with: pip install coremltools")


def export_chunkwise_to_coreml(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str,
    model_name: str = "ChunkwiseMLSTM",
    compute_units: str = "ALL",  # "ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"
    minimum_deployment_target: str = "iOS15",  # ANE requires iOS15+
) -> Optional[Path]:
    """
    Export ChunkwiseMLSTM to CoreML for ANE deployment.

    Args:
        model: ChunkwiseMLSTM instance
        example_input: Example input tensor [B, S, D]
        output_path: Path to save .mlpackage file
        model_name: Model name for CoreML
        compute_units: Target compute units ("CPU_AND_NE" for ANE)
        minimum_deployment_target: iOS version (iOS15+ for ANE)

    Returns:
        Path to exported model if successful, None otherwise

    Example:
        >>> from xlstm_solace_torch.mad.blocks.mlstm_torch import ChunkwiseMLSTM
        >>> model = ChunkwiseMLSTM(dim=512, num_heads=4)
        >>> model.eval()
        >>> example = torch.randn(1, 64, 512)
        >>> export_chunkwise_to_coreml(
        ...     model, example, "mlstm_ane.mlpackage",
        ...     compute_units="CPU_AND_NE"
        ... )
    """
    if not COREML_AVAILABLE:
        print("ERROR: coremltools not available")
        return None

    print(f"Exporting {model_name} to CoreML...")
    model.eval()

    # Step 1: Trace to TorchScript
    print("Step 1: Tracing to TorchScript...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        print("✓ TorchScript tracing successful")
    except Exception as e:
        print(f"ERROR: TorchScript tracing failed: {e}")
        return None

    # Step 2: Convert to CoreML
    print("Step 2: Converting to CoreML...")
    try:
        # Map compute units
        compute_units_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,  # ANE (Neural Engine)
        }

        # Map deployment targets
        target_map = {
            "iOS15": ct.target.iOS15,
            "iOS16": ct.target.iOS16,
            "iOS17": ct.target.iOS17,
        }

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(
                name="input",
                shape=example_input.shape,
                dtype=float,
            )],
            outputs=[ct.TensorType(name="output")],
            compute_units=compute_units_map.get(compute_units, ct.ComputeUnit.ALL),
            minimum_deployment_target=target_map.get(minimum_deployment_target, ct.target.iOS15),
            convert_to="mlprogram",  # Use ML Program (required for ANE)
        )

        # Add metadata
        mlmodel.short_description = f"{model_name} for Apple Neural Engine"
        mlmodel.author = "xlstm-solace-torch"
        mlmodel.license = "NXAI Community License"
        mlmodel.version = "1.0"

        print("✓ CoreML conversion successful")

    except Exception as e:
        print(f"ERROR: CoreML conversion failed: {e}")
        return None

    # Step 3: Save model
    print(f"Step 3: Saving to {output_path}...")
    try:
        output_path = Path(output_path)
        mlmodel.save(str(output_path))
        print(f"✓ Model saved to {output_path}")
        print(f"\nModel size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

        # Print ANE compatibility info
        if compute_units == "CPU_AND_NE":
            print("\n🚀 Model configured for Apple Neural Engine (ANE)")
            print("   - Requires A14/M1 or newer chip")
            print("   - Up to 10x faster inference")
            print("   - Up to 14x lower peak memory")
            print("   - FP16 precision only")

        return output_path

    except Exception as e:
        print(f"ERROR: Failed to save model: {e}")
        return None


def benchmark_coreml_model(
    mlmodel_path: str,
    example_input: torch.Tensor,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark CoreML model performance.

    Args:
        mlmodel_path: Path to .mlpackage file
        example_input: Example input tensor
        num_iterations: Number of iterations for benchmarking

    Returns:
        Dictionary with timing statistics
    """
    if not COREML_AVAILABLE:
        print("ERROR: coremltools not available")
        return {}

    import time
    import coremltools as ct

    # Load model
    model = ct.models.MLModel(mlmodel_path)

    # Convert input
    input_dict = {"input": example_input.numpy()}

    # Warmup
    for _ in range(10):
        model.predict(input_dict)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        model.predict(input_dict)
        times.append(time.perf_counter() - start)

    import numpy as np
    times = np.array(times)

    return {
        "mean_ms": float(times.mean() * 1000),
        "std_ms": float(times.std() * 1000),
        "min_ms": float(times.min() * 1000),
        "max_ms": float(times.max() * 1000),
        "median_ms": float(np.median(times) * 1000),
    }


if __name__ == "__main__":
    print("CoreML Export for ChunkwiseMLSTM")
    print("=" * 50)

    # Example usage
    from xlstm_solace_torch.mad.blocks.mlstm_torch import ChunkwiseMLSTM

    # Create model
    model = ChunkwiseMLSTM(
        dim=256,
        num_heads=4,
        chunk_size=32,
        use_compile=False,  # Disable torch.compile for export
    )
    model.eval()

    # Example input
    example = torch.randn(1, 64, 256)

    # Export for ANE
    output_path = export_chunkwise_to_coreml(
        model=model,
        example_input=example,
        output_path="chunkwise_mlstm_ane.mlpackage",
        compute_units="CPU_AND_NE",  # Enable ANE
        minimum_deployment_target="iOS15",
    )

    if output_path:
        print("\n✓ Export successful!")
        print(f"Deploy to iOS/macOS using: {output_path}")

        # Benchmark
        print("\nBenchmarking...")
        stats = benchmark_coreml_model(str(output_path), example)
        if stats:
            print(f"  Mean: {stats['mean_ms']:.2f} ms")
            print(f"  Std:  {stats['std_ms']:.2f} ms")
            print(f"  Min:  {stats['min_ms']:.2f} ms")
            print(f"  Max:  {stats['max_ms']:.2f} ms")
