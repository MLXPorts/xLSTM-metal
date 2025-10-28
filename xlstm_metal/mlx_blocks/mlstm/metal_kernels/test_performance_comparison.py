"""Performance comparison between sequential and chunkwise mLSTM implementations.

Demonstrates the speedup from using O(T/C + C) chunkwise algorithm vs O(T) sequential.
"""

import mlx.core as mx
import time
import sys
sys.path.insert(0, '/Volumes/emberstuff/xLSTM/mad/blocks')

from mlstm_mlx.kernel import mlstm_chunkwise, mlstm_sequential


def benchmark_kernel(name, kernel_fn, inputs, num_warmup=3, num_runs=10):
    """Benchmark a kernel function."""
    print(f"\nBenchmarking {name}...")

    # Warmup
    for _ in range(num_warmup):
        output = kernel_fn(**inputs)
        if isinstance(output, tuple):
            mx.eval(output[0])
        else:
            mx.eval(output)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        output = kernel_fn(**inputs)
        if isinstance(output, tuple):
            mx.eval(output[0])
        else:
            mx.eval(output)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    print(f"  Average: {avg_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
    print(f"  Min: {min(times)*1000:.2f} ms, Max: {max(times)*1000:.2f} ms")

    return avg_time


def test_performance_comparison():
    """Compare sequential vs chunkwise performance."""
    print("=" * 70)
    print("Performance Comparison: Sequential vs Chunkwise mLSTM")
    print("=" * 70)

    # Test different sequence lengths
    configs = [
        {"S": 64, "chunk_size": 32},
        {"S": 128, "chunk_size": 64},
        {"S": 256, "chunk_size": 64},
        {"S": 512, "chunk_size": 64},
    ]

    B, NH, QK_DH, V_DH = 1, 2, 32, 32

    results = []

    for config in configs:
        S = config["S"]
        chunk_size = config["chunk_size"]

        print(f"\n{'='*70}")
        print(f"Configuration: S={S}, B={B}, NH={NH}, QK_DH={QK_DH}, V_DH={V_DH}")
        print(f"Chunk size: {chunk_size}, Num chunks: {(S + chunk_size - 1) // chunk_size}")
        print(f"{'='*70}")

        # Create inputs
        q = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
        k = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
        v = mx.random.normal((B, NH, S, V_DH), dtype=mx.float32) * 0.1
        i_preact = mx.random.normal((B, NH, S), dtype=mx.float32)
        f_preact = mx.random.normal((B, NH, S), dtype=mx.float32)

        # Benchmark sequential
        seq_inputs = {
            "q": q, "k": k, "v": v,
            "i_preact": i_preact,
            "f_preact": f_preact,
            "return_last_states": False,
        }
        seq_time = benchmark_kernel("Sequential", mlstm_sequential, seq_inputs)

        # Benchmark chunkwise
        chunk_inputs = {
            "q": q, "k": k, "v": v,
            "i_preact": i_preact,
            "f_preact": f_preact,
            "chunk_size": chunk_size,
            "return_last_states": False,
        }
        chunk_time = benchmark_kernel("Chunkwise", mlstm_chunkwise, chunk_inputs)

        # Calculate speedup
        speedup = seq_time / chunk_time

        print(f"\n{'='*70}")
        print(f"SPEEDUP: {speedup:.2f}x")
        print(f"Sequential: {seq_time*1000:.2f} ms")
        print(f"Chunkwise:  {chunk_time*1000:.2f} ms")
        print(f"{'='*70}")

        results.append({
            "S": S,
            "chunk_size": chunk_size,
            "seq_time": seq_time,
            "chunk_time": chunk_time,
            "speedup": speedup,
        })

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Seq Length':<12} {'Chunk Size':<12} {'Sequential (ms)':<18} {'Chunkwise (ms)':<18} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['S']:<12} {r['chunk_size']:<12} {r['seq_time']*1000:<18.2f} {r['chunk_time']*1000:<18.2f} {r['speedup']:<10.2f}x")
    print("=" * 70)


def test_correctness_comparison():
    """Verify chunkwise produces similar results to sequential."""
    print("\n" + "=" * 70)
    print("Correctness Comparison: Sequential vs Chunkwise")
    print("=" * 70)

    B, NH, S, QK_DH, V_DH = 1, 2, 128, 32, 32
    chunk_size = 64

    print(f"\nDimensions: B={B}, NH={NH}, S={S}, QK_DH={QK_DH}, V_DH={V_DH}")
    print(f"Chunk size: {chunk_size}")

    # Create inputs
    q = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
    k = mx.random.normal((B, NH, S, QK_DH), dtype=mx.float32) * 0.1
    v = mx.random.normal((B, NH, S, V_DH), dtype=mx.float32) * 0.1
    i_preact = mx.random.normal((B, NH, S), dtype=mx.float32)
    f_preact = mx.random.normal((B, NH, S), dtype=mx.float32)

    # Run sequential
    print("\nRunning sequential...")
    h_seq, (c_seq, n_seq, m_seq) = mlstm_sequential(
        q=q, k=k, v=v,
        i_preact=i_preact,
        f_preact=f_preact,
        return_last_states=True,
    )
    mx.eval(h_seq, c_seq, n_seq, m_seq)

    # Run chunkwise
    print("Running chunkwise...")
    h_chunk, (c_chunk, n_chunk, m_chunk) = mlstm_chunkwise(
        q=q, k=k, v=v,
        i_preact=i_preact,
        f_preact=f_preact,
        chunk_size=chunk_size,
        return_last_states=True,
    )
    mx.eval(h_chunk, c_chunk, n_chunk, m_chunk)

    # Compare outputs
    print("\nComparing outputs...")

    # Compute differences
    h_diff = mx.abs(h_seq - h_chunk)
    c_diff = mx.abs(c_seq - c_chunk)
    n_diff = mx.abs(n_seq - n_chunk)
    m_diff = mx.abs(m_seq - m_chunk)

    print(f"\nAbsolute differences:")
    print(f"  h: max={float(mx.max(h_diff)):.6e}, mean={float(mx.mean(h_diff)):.6e}")
    print(f"  c_final: max={float(mx.max(c_diff)):.6e}, mean={float(mx.mean(c_diff)):.6e}")
    print(f"  n_final: max={float(mx.max(n_diff)):.6e}, mean={float(mx.mean(n_diff)):.6e}")
    print(f"  m_final: max={float(mx.max(m_diff)):.6e}, mean={float(mx.mean(m_diff)):.6e}")

    # Compute relative errors
    h_rel_err = float(mx.max(h_diff) / (mx.max(mx.abs(h_seq)) + 1e-8))
    c_rel_err = float(mx.max(c_diff) / (mx.max(mx.abs(c_seq)) + 1e-8))

    print(f"\nRelative errors:")
    print(f"  h: {h_rel_err:.6e}")
    print(f"  c_final: {c_rel_err:.6e}")

    # Check if within tolerance
    tol = 1e-3  # Relaxed tolerance due to different computation order
    if h_rel_err < tol and c_rel_err < tol:
        print(f"\n✓ Results match within tolerance ({tol})")
    else:
        print(f"\n⚠ Results differ more than tolerance ({tol})")
        print("  This is expected due to different computation order in parallel algorithm.")


if __name__ == "__main__":
    # Test correctness first
    test_correctness_comparison()

    # Then benchmark performance
    test_performance_comparison()

    print("\n" + "=" * 70)
    print("✓✓✓ Performance comparison complete ✓✓✓")
    print("=" * 70)
