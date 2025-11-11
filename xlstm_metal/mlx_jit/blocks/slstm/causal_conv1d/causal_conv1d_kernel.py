import mlx.core as mx
import mlx.nn as nn
from typing import Optional


@mx.custom_function
def metal_causal_conv1d(x: mx.array, weight: mx.array, bias: Optional[mx.array] = None) -> mx.array:
    """Causal 1D convolution with Metal kernel"""

    source = """
    uint batch = thread_position_in_grid.x;
    uint channel = thread_position_in_grid.y; 
    uint time = thread_position_in_grid.z;
    
    uint batch_size = x_shape[0];
    uint seq_len = x_shape[1];
    uint in_channels = x_shape[2];
    uint out_channels = weight_shape[0];
    uint kernel_size = weight_shape[2];
    
    if (batch >= batch_size || channel >= out_channels || time >= seq_len) return;
    
    float sum = 0.0f;
    
    // Causal convolution: only look at current and past
    for (uint k = 0; k < kernel_size; k++) {
        int t_input = int(time) - int(k);
        if (t_input >= 0) {
            for (uint in_ch = 0; in_ch < in_channels; in_ch++) {
                uint x_idx = batch * seq_len * in_channels + t_input * in_channels + in_ch;
                uint w_idx = channel * in_channels * kernel_size + in_ch * kernel_size + k;
                sum += x[x_idx] * weight[w_idx];
            }
        }
    }
    
    // Add bias if present
    if (*bias_size > 0) {
        sum += bias[channel];
    }
    
    uint out_idx = batch * seq_len * out_channels + time * out_channels + channel;
    output[out_idx] = sum;
    """

    kernel = mx.fast.metal_kernel(
        name="causal_conv1d_kernel",
        input_names=["x", "weight"] + (["bias"] if bias is not None else []),
        output_names=["output"],
        source=source,
        header="""
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;
        """
    )

    batch_size, seq_len, in_channels = x.shape
    out_channels = weight.shape[0]
    output_shape = (batch_size, seq_len, out_channels)

    inputs = [x, weight]
    if bias is not None:
        inputs.append(bias)
    else:
        inputs.append(mx.array([], dtype=x.dtype))  # Empty bias

    return kernel(
        inputs=inputs,
        output_shapes=[output_shape],
        output_dtypes=[x.dtype],
        grid=(batch_size, out_channels, seq_len),
        threadgroup=(min(batch_size, 8), min(out_channels, 8), min(seq_len, 8))
    )[0]


class CausalConv1dLayer(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        # Conv1d weights: (out_channels, in_channels, kernel_size)
        self.weight = mx.random.normal((channels, channels, kernel_size)) / (channels * kernel_size) ** 0.5
        self.bias = mx.zeros((channels,))

    def __call__(self, x):
        return metal_causal_conv1d(x, self.weight, self.bias)
