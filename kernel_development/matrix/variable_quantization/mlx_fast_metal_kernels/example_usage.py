"""Example usage and benchmark for variable quantization kernel."""

import mlx.core as mx
import numpy as np
import time
from variable_quantization import VariableQuantizationMLXKernel, quantize


def example_basic_usage():
    """Demonstrate basic usage of the quantization kernel."""
    # Create some test data in [-1, 1] range
    x = mx.array(np.linspace(-1, 1, 20), dtype=mx.float32)

    print("Original values:")
    print(x)

    # Quantize to 4 bits
    y_4bit = quantize(x, bits=4)
    print("\n4-bit quantized:")
    print(y_4bit)

    # Quantize to 8 bits (higher precision)
    y_8bit = quantize(x, bits=8)
    print("\n8-bit quantized:")
    print(y_8bit)


def example_early_compilation():
    """Demonstrate early compilation pattern."""
    # Create kernel and compile immediately
    kernel = VariableQuantizationMLXKernel()
    kernel.compile()

    # Now use it multiple times without recompilation
    for bits in [2, 4, 8]:
        x = mx.array([0.5, 0.25, 0.125], dtype=mx.float32)
        y = kernel(x, bits=bits)
        print(f"{bits}-bit: {y.tolist()}")


def benchmark():
    """Benchmark quantization performance."""
    kernel = VariableQuantizationMLXKernel()
    kernel.compile()

    sizes = [1000, 10000, 100000, 1000000]

    print("\nBenchmark Results:")
    print(f"{'Size':<12} {'Time (ms)':<12} {'Throughput (M elements/s)':<30}")
    print("-" * 60)

    for size in sizes:
        # Create random data
        x = mx.random.uniform(-1, 1, shape=(size,), dtype=mx.float32)

        # Warmup
        _ = kernel(x, bits=4)
        mx.eval(_)

        # Benchmark
        num_runs = 100
        start = time.time()
        for _ in range(num_runs):
            y = kernel(x, bits=4)
            mx.eval(y)
        end = time.time()

        avg_time_ms = (end - start) / num_runs * 1000
        throughput = size / (avg_time_ms / 1000) / 1e6

        print(f"{size:<12} {avg_time_ms:<12.4f} {throughput:<30.2f}")


def compare_bit_depths():
    """Compare quantization at different bit depths."""
    x = mx.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=mx.float32)

    print("\nComparison of Bit Depths:")
    print(f"{'Original':<12} ", end="")
    for val in x.tolist():
        print(f"{val:>6.3f} ", end="")
    print()

    for bits in [2, 3, 4, 6, 8]:
        y = quantize(x, bits=bits)
        print(f"{bits}-bit{'':<7} ", end="")
        for val in y.tolist():
            print(f"{val:>6.3f} ", end="")
        print()


if __name__ == "__main__":
    print("=== Basic Usage ===")
    example_basic_usage()

    print("\n=== Early Compilation Pattern ===")
    example_early_compilation()

    print("\n=== Bit Depth Comparison ===")
    compare_bit_depths()

    print("\n=== Performance Benchmark ===")
    benchmark()

