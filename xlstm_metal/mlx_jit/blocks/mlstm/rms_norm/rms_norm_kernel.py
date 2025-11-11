import mlx.core as mx


@mx.custom_function
def metal_rms_norm_impl(hidden_states, weight, eps):
    """RMSNorm implementation using Metal kernel"""
    source = """
        uint tid = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        
        uint hidden_size = hidden_states_shape[1];
        uint batch_size = hidden_states_shape[0];
        
        if (batch >= batch_size) return;
        
        // Single-threaded per batch for simplicity and correctness
        if (tid == 0) {
            uint base_idx = batch * hidden_size;
            
            // Compute mean square
            float mean_square = 0.0f;
            for (uint i = 0; i < hidden_size; i++) {
                float val = hidden_states[base_idx + i];
                mean_square += val * val;
            }
            mean_square /= float(hidden_size);
            
            // Compute rsqrt(variance + eps)
            float rsqrt_var = rsqrt(mean_square + eps[0]);
            
            // Normalize and scale
            for (uint i = 0; i < hidden_size; i++) {
                uint idx = base_idx + i;
                float val = hidden_states[idx];
                output[idx] = weight[i] * val * rsqrt_var;
            }
        }
    """

    kernel = mx.fast.metal_kernel(
        name="rmsnorm_kernel",
        input_names=["hidden_states", "weight", "eps"],
        output_names=["output"],
        source=source,
        header="""
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;
        """
    )

    return kernel(
        inputs=[hidden_states, weight, eps],
        output_shapes=[hidden_states.shape],
        output_dtypes=[hidden_states.dtype],
        grid=(1, hidden_states.shape[0], 1),
        threadgroup=(1, 1, 1)
    )[0]


class MetalRMSNorm(nn.Module):
    """RMSNorm using Metal kernels"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        eps_array = mx.array([self.variance_epsilon], dtype=hidden_states.dtype)
        return metal_rms_norm_impl(hidden_states, self.weight, eps_array)
