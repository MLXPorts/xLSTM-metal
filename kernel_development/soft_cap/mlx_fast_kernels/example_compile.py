"""
Example: Using .compile() for early kernel compilation

This demonstrates the proper pattern for pre-compiling Metal kernels
at module level to avoid recompilation overhead.
"""
import mlx.core as mx
from softcap import SoftCapMLXFastKernel

# Pattern 1: Compile at module level (recommended for production)
# This compiles the kernel once when the module is imported
global_softcap = SoftCapMLXFastKernel()
global_softcap.compile()  # ← This is the key improvement!


def process_batch(data, cap_value=5.0):
    """Process a batch using the pre-compiled kernel."""
    return global_softcap(data, cap_value)


# Pattern 2: Lazy compilation (compile on first use)
lazy_softcap = SoftCapMLXFastKernel()


# No .compile() call - will compile on first __call__

def example_usage():
    """Demonstrate both patterns."""
    print("Example 1: Pre-compiled kernel (fast)")
    data = mx.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=mx.float32)

    # Using pre-compiled kernel - no compilation overhead
    result = global_softcap(data, 5.0)
    print(f"Result: {result.tolist()}")

    print("\nExample 2: Lazy compilation (compile on first call)")
    # First call will trigger compilation
    result2 = lazy_softcap(data, 5.0)
    print(f"Result: {result2.tolist()}")

    print("\nExample 3: Manual compilation check")
    custom_kernel = SoftCapMLXFastKernel()
    print(f"Before compile: kernel is None = {custom_kernel.kernel is None}")

    compiled = custom_kernel.compile()
    print(f"After compile: kernel is None = {custom_kernel.kernel is None}")
    print(f"Compiled kernel object: {compiled}")

    # Calling compile() again returns the same cached kernel
    compiled2 = custom_kernel.compile()
    print(f"Same kernel? {compiled is compiled2}")


if __name__ == "__main__":
    example_usage()
