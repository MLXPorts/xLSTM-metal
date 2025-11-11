"""
Performance benchmark comparing MLX and PyTorch xLSTM implementations
"""

import time
import numpy as np
import sys
from tabulate import tabulate


def benchmark_mlx(configs):
    """Benchmark MLX implementation"""
    import mlx.core as mx
    from xlstm_mlx import create_xlstm_model

    results = []

    for config_name, config in configs.items():
        print(f"\nBenchmarking MLX - {config_name}...")

        model = create_xlstm_model(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            signature=config['signature'],
            inp_dim=config['inp_dim'],
            head_dim=config['head_dim'],
            head_num=config['head_num'],
            dropout=0.0
        )

        batch_size = config['batch_size']
        seq_len = config['seq_len']

        # Warm-up
        tokens = mx.random.randint(0, config['vocab_size'], (batch_size, seq_len))
        _ = model(tokens)
        mx.eval(tokens)

        # Benchmark
        times = []
        for _ in range(10):
            tokens = mx.random.randint(0, config['vocab_size'], (batch_size, seq_len))
            start = time.time()
            logits = model(tokens)
            mx.eval(logits)  # Force evaluation
            times.append(time.time() - start)

        avg_time = np.mean(times[2:])  # Skip first two for stability
        std_time = np.std(times[2:])
        throughput = (batch_size * seq_len) / avg_time

        results.append({
            'Config': config_name,
            'Framework': 'MLX',
            'Batch Size': batch_size,
            'Seq Length': seq_len,
            'Avg Time (s)': f"{avg_time:.4f}",
            'Std Time (s)': f"{std_time:.4f}",
            'Throughput (tok/s)': f"{throughput:.0f}"
        })

    return results


def benchmark_pytorch(configs):
    """Benchmark PyTorch implementation"""
    import torch
    from xlstm_pytorch import create_xlstm_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    for config_name, config in configs.items():
        print(f"\nBenchmarking PyTorch ({device}) - {config_name}...")

        model = create_xlstm_model(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            signature=config['signature'],
            inp_dim=config['inp_dim'],
            head_dim=config['head_dim'],
            head_num=config['head_num'],
            dropout=0.0,
            device=device
        )
        model.eval()

        batch_size = config['batch_size']
        seq_len = config['seq_len']

        # Warm-up
        with torch.no_grad():
            tokens = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
            _ = model(tokens)
            if device == 'cuda':
                torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(10):
                tokens = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
                start = time.time()
                logits = model(tokens)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        avg_time = np.mean(times[2:])  # Skip first two for stability
        std_time = np.std(times[2:])
        throughput = (batch_size * seq_len) / avg_time

        results.append({
            'Config': config_name,
            'Framework': f'PyTorch ({device})',
            'Batch Size': batch_size,
            'Seq Length': seq_len,
            'Avg Time (s)': f"{avg_time:.4f}",
            'Std Time (s)': f"{std_time:.4f}",
            'Throughput (tok/s)': f"{throughput:.0f}"
        })

    return results


def memory_benchmark():
    """Benchmark memory usage"""
    print("\n" + "=" * 60)
    print("Memory Usage Benchmark")
    print("=" * 60)

    results = []

    # MLX memory test
    try:
        import mlx.core as mx
        from xlstm_mlx import create_xlstm_model

        model = create_xlstm_model(
            vocab_size=10000,
            num_layers=12,
            signature=(7, 1),
            inp_dim=512,
            head_dim=64,
            head_num=8
        )

        # Count parameters
        def count_mlx_params(params):
            """

            :param params:
            :return:
            """
            count = 0
            for p in params.values():
                if isinstance(p, mx.array):
                    count += p.size
                elif isinstance(p, dict):
                    count += count_mlx_params(p)
            return count

        param_count = count_mlx_params(model.parameters())
        memory_mb = (param_count * 4) / (1024 * 1024)  # Assuming float32

        results.append({
            'Framework': 'MLX',
            'Parameters': f"{param_count:,}",
            'Memory (MB)': f"{memory_mb:.2f}"
        })
    except:
        pass

    # PyTorch memory test
    try:
        import torch
        from xlstm_pytorch import create_xlstm_model

        model = create_xlstm_model(
            vocab_size=10000,
            num_layers=12,
            signature=(7, 1),
            inp_dim=512,
            head_dim=64,
            head_num=8,
            device='cpu'
        )

        param_count = sum(p.numel() for p in model.parameters())
        memory_mb = (param_count * 4) / (1024 * 1024)  # Assuming float32

        results.append({
            'Framework': 'PyTorch',
            'Parameters': f"{param_count:,}",
            'Memory (MB)': f"{memory_mb:.2f}"
        })
    except:
        pass

    if results:
        print(tabulate(results, headers='keys', tablefmt='grid'))


def main():
    """Run benchmarks"""
    print("=" * 60)
    print("xLSTM Performance Benchmark")
    print("=" * 60)

    # Define test configurations
    configs = {
        'Small': {
            'vocab_size': 1000,
            'num_layers': 2,
            'signature': (1, 1),
            'inp_dim': 128,
            'head_dim': 16,
            'head_num': 8,
            'batch_size': 4,
            'seq_len': 32
        },
        'Medium': {
            'vocab_size': 5000,
            'num_layers': 4,
            'signature': (3, 1),
            'inp_dim': 256,
            'head_dim': 32,
            'head_num': 8,
            'batch_size': 8,
            'seq_len': 64
        },
        'Large': {
            'vocab_size': 10000,
            'num_layers': 8,
            'signature': (6, 2),
            'inp_dim': 512,
            'head_dim': 64,
            'head_num': 8,
            'batch_size': 4,
            'seq_len': 128
        }
    }

    all_results = []

    # Benchmark MLX
    print("\n" + "=" * 60)
    print("MLX Benchmarks")
    print("=" * 60)
    try:
        mlx_results = benchmark_mlx(configs)
        all_results.extend(mlx_results)
    except Exception as e:
        print(f"MLX benchmark failed: {e}")

    # Benchmark PyTorch
    print("\n" + "=" * 60)
    print("PyTorch Benchmarks")
    print("=" * 60)
    try:
        pytorch_results = benchmark_pytorch(configs)
        all_results.extend(pytorch_results)
    except Exception as e:
        print(f"PyTorch benchmark failed: {e}")

    # Display results
    if all_results:
        print("\n" + "=" * 60)
        print("Performance Results")
        print("=" * 60)
        print(tabulate(all_results, headers='keys', tablefmt='grid'))

    # Memory benchmark
    memory_benchmark()

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    # Check if tabulate is installed
    try:
        from tabulate import tabulate
    except ImportError:
        print("Installing tabulate for better output formatting...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
        from tabulate import tabulate

    sys.exit(main())
