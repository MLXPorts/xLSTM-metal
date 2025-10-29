"""Benchmark to compare standard vs optimized quantization kernels."""

import mlx.core as mx
import numpy as np
import time
from variable_quantization import VariableQuantizationMLXKernel


def benchmark_kernel(kernel, size, bits, warmup=10, iterations=100):
    """Benchmark a quantization kernel."""
    x = mx.random.uniform(-1, 1, shape=(size,), dtype=mx.float32)

    # Warmup
    for _ in range(warmup):
        y = kernel(x, bits)
        mx.eval(y)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        y = kernel(x, bits)
        mx.eval(y)
    end = time.time()

    avg_time_ms = (end - start) / iterations * 1000
    throughput = size / (avg_time_ms / 1000) / 1e6

    return avg_time_ms, throughput


def main():
    print("Variable Quantization Kernel Performance Comparison")
    print("=" * 70)

    # Create both kernel variants
    kernel_standard = VariableQuantizationMLXKernel(use_block_kernel=False)
    kernel_standard.compile()

    kernel_block = VariableQuantizationMLXKernel(use_block_kernel=True)
    kernel_block.compile()

    sizes = [1000, 10000, 100000, 1000000, 10000000]
    bits = 4

    print(f"\nBenchmarking with {bits}-bit quantization\n")
    print(f"{'Size':<12} {'Kernel':<15} {'Time (ms)':<12} {'Throughput (M/s)':<20} {'Speedup':<10}")
    print("-" * 70)

    for size in sizes:
        # Benchmark standard kernel
        time_std, throughput_std = benchmark_kernel(kernel_standard, size, bits)

        # Benchmark block kernel
        time_block, throughput_block = benchmark_kernel(kernel_block, size, bits)

        speedup = time_std / time_block

        print(f"{size:<12} {'Standard':<15} {time_std:<12.4f} {throughput_std:<20.2f}")
        print(f"{'':<12} {'Block (opt)':<15} {time_block:<12.4f} {throughput_block:<20.2f} {speedup:.2f}x")
        print()

    # Test different bit depths on large array
    print("\nBit Depth Performance (1M elements)")
    print(f"{'Bits':<10} {'Time (ms)':<12} {'Throughput (M/s)':<20}")
    print("-" * 45)

    size = 1000000
    for bits in [2, 4, 6, 8, 12, 16]:
        time_ms, throughput = benchmark_kernel(kernel_block, size, bits)
        print(f"{bits:<10} {time_ms:<12.4f} {throughput:<20.2f}")

    # Memory bandwidth test
    print("\nMemory Bandwidth Analysis")
    print("-" * 45)
    size = 10000000
    time_ms, throughput = benchmark_kernel(kernel_block, size, 4)

    # Each element: 1 read (4 bytes) + 1 write (4 bytes) = 8 bytes
    bytes_per_element = 8
    total_bytes = size * bytes_per_element
    bandwidth_gbs = (total_bytes / (time_ms / 1000)) / 1e9

    print(f"Array size: {size:,} elements")
    print(f"Total memory: {total_bytes / 1e6:.1f} MB")
    print(f"Time: {time_ms:.2f} ms")
    print(f"Bandwidth: {bandwidth_gbs:.2f} GB/s")
    print(f"Throughput: {throughput:.2f} M elements/s")


if __name__ == "__main__":
    main()

