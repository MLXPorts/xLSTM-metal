"""
xLSTM Metal Kernel Implementation

Proper Metal kernel implementations for xLSTM operations using MLX fast.metal_kernel API.
Based on the comprehensive Metal guide from ember-ml backend.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class xLSTMConfig:
    vocab_size: int = 50257
    num_layers: int = 6
    d_model: int = 512
    signature: Tuple[int, ...] = (1, 1)
    head_dim: int = 32
    head_num: int = 4
    mlstm_proj_factor: float = 2.0
    slstm_proj_factor: float = 1.333
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    dropout: float = 0.1
    causal_conv_kernel: int = 4


@mx.custom_function
def metal_soft_cap(x: mx.array, cap_value: float) -> mx.array:
    """Soft capping using Metal kernel"""
    source = """
    uint elem = thread_position_in_grid.x;
    
    // Calculate total size dynamically based on shape
    uint total_size = 1;
    for (uint i = 0; i < 16; i++) {  // Max 16 dimensions
        if (inp0_shape[i] == 0) break;
        total_size *= inp0_shape[i];
    }
    
    if (elem >= total_size) return;
    
    float val = inp0[elem];
    float cap = cap_val[0];
    out0[elem] = cap * tanh(val / cap);
    """

    kernel = mx.fast.metal_kernel(
        name="soft_cap_kernel",
        input_names=["inp0", "cap_val"],
        output_names=["out0"],
        source=source,
        header="""
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;
        """
    )

    cap_array = mx.array([cap_value], dtype=x.dtype)

    return kernel(
        inputs=[x, cap_array],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(min(x.size, 256), 1, 1)
    )[0]


@mx.custom_function
def metal_mlstm_step(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        i_gate: mx.array,
        f_gate: mx.array,
        o_gate: mx.array,
        hidden_state: mx.array
) -> Tuple[mx.array, mx.array]:
    """Single mLSTM step with Metal kernel"""

    source = """
    uint tid = thread_position_in_grid.x;
    uint head = thread_position_in_grid.y;
    uint batch = thread_position_in_grid.z;
    
    uint num_heads = q_shape[1];
    uint head_dim = q_shape[2];
    
    if (head >= num_heads || tid >= head_dim || batch >= q_shape[0]) return;
    
    uint q_idx = batch * num_heads * head_dim + head * head_dim + tid;
    uint k_idx = batch * num_heads * head_dim + head * head_dim + tid;
    uint v_idx = batch * num_heads * head_dim + head * head_dim + tid;
    uint gate_idx = batch * num_heads + head;
    
    float q_val = q[q_idx];
    float k_val = k[k_idx];
    float v_val = v[v_idx];
    float i_val = i_gate[gate_idx];
    float f_val = f_gate[gate_idx];
    float o_val = o_gate[gate_idx];
    
    // Update matrix memory: H = f * H + i * (k ⊗ v)
    uint h_base = batch * num_heads * head_dim * head_dim + head * head_dim * head_dim;
    
    // Compute outer product k ⊗ v and update memory
    for (uint j = 0; j < head_dim; j++) {
        uint h_idx = h_base + tid * head_dim + j;
        float kv_outer = k_val * v[batch * num_heads * head_dim + head * head_dim + j];
        hidden_state[h_idx] = f_val * hidden_state[h_idx] + i_val * kv_outer;
    }
    
    // Compute output: h = H * q
    float h_sum = 0.0f;
    for (uint j = 0; j < head_dim; j++) {
        uint h_idx = h_base + j * head_dim + tid;
        h_sum += hidden_state[h_idx] * q_val;
    }
    
    // Apply output gate
    uint out_idx = batch * num_heads * head_dim + head * head_dim + tid;
    output[out_idx] = o_val * h_sum;
    """

    kernel = mx.fast.metal_kernel(
        name="mlstm_step_kernel",
        input_names=["q", "k", "v", "i_gate", "f_gate", "o_gate", "hidden_state"],
        output_names=["output", "updated_hidden"],
        source=source,
        header="""
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;
        """
    )

    batch_size, num_heads, head_dim = q.shape
    output_shape = (batch_size, num_heads, head_dim)
    hidden_shape = hidden_state.shape

    results = kernel(
        inputs=[q, k, v, i_gate, f_gate, o_gate, hidden_state],
        output_shapes=[output_shape, hidden_shape],
        output_dtypes=[q.dtype, hidden_state.dtype],
        grid=(head_dim, num_heads, batch_size),
        threadgroup=(min(head_dim, 16), min(num_heads, 16), min(batch_size, 4))
    )

    return results[0], results[1]


@mx.custom_function
def metal_slstm_step(
        x: mx.array,
        i_gate: mx.array,
        f_gate: mx.array,
        o_gate: mx.array,
        hidden_state: mx.array
) -> Tuple[mx.array, mx.array]:
    """Single sLSTM step with Metal kernel"""

    source = """
    uint tid = thread_position_in_grid.x;
    uint batch = thread_position_in_grid.y;
    
    uint proj_dim = x_shape[1];
    
    if (tid >= proj_dim || batch >= x_shape[0]) return;
    
    uint idx = batch * proj_dim + tid;
    
    float x_val = x[idx];
    float i_val = i_gate[idx];
    float f_val = f_gate[idx];
    float o_val = o_gate[idx];
    float h_val = hidden_state[idx];
    
    // Update scalar memory: h = f * h + i * tanh(x)
    float new_h = f_val * h_val + i_val * tanh(x_val);
    updated_hidden[idx] = new_h;
    
    // Output: y = o * tanh(h)
    output[idx] = o_val * tanh(new_h);
    """

    kernel = mx.fast.metal_kernel(
        name="slstm_step_kernel",
        input_names=["x", "i_gate", "f_gate", "o_gate", "hidden_state"],
        output_names=["output", "updated_hidden"],
        source=source,
        header="""
        #include <metal_stdlib>
        #include <metal_math>
        using namespace metal;
        """
    )

    batch_size, proj_dim = x.shape

    results = kernel(
        inputs=[x, i_gate, f_gate, o_gate, hidden_state],
        output_shapes=[x.shape, hidden_state.shape],
        output_dtypes=[x.dtype, hidden_state.dtype],
        grid=(proj_dim, batch_size, 1),
        threadgroup=(min(proj_dim, 256), min(batch_size, 4), 1)
    )

    return results[0], results[1]


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


class MetalxLSTMBlock(nn.Module):
    """xLSTM block using Metal kernels"""

    def __init__(self, config: xLSTMConfig, layer_idx: int, is_mlstm: bool = True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_mlstm = is_mlstm

        if is_mlstm:
            # mLSTM configuration
            self.head_dim = config.head_dim
            self.num_heads = config.head_num

            self.q_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=False)

            self.i_proj = nn.Linear(config.d_model, self.num_heads, bias=False)
            self.f_proj = nn.Linear(config.d_model, self.num_heads, bias=False)
            self.o_proj = nn.Linear(config.d_model, self.num_heads, bias=False)

            self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)
        else:
            # sLSTM configuration
            proj_dim = int(config.d_model * config.slstm_proj_factor)

            self.conv = self._create_causal_conv(config.d_model, config.causal_conv_kernel)

            self.i_proj = nn.Linear(config.d_model, proj_dim, bias=False)
            self.f_proj = nn.Linear(config.d_model, proj_dim, bias=False)
            self.o_proj = nn.Linear(config.d_model, proj_dim, bias=False)
            self.c_proj = nn.Linear(config.d_model, proj_dim, bias=False)

            self.out_proj = nn.Linear(proj_dim, config.d_model, bias=False)

        self.layer_norm = MetalRMSNorm(config.d_model)

    def _create_causal_conv(self, channels: int, kernel_size: int):
        """Create causal convolution layer with proper weights"""

        class CausalConv1dLayer(nn.Module):
            def __init__(self, channels, kernel_size):
                super().__init__()
                # Conv1d weights: (out_channels, in_channels, kernel_size)
                self.weight = mx.random.normal((channels, channels, kernel_size)) / (channels * kernel_size) ** 0.5
                self.bias = mx.zeros((channels,))

            def __call__(self, x):
                return metal_causal_conv1d(x, self.weight, self.bias)

        return CausalConv1dLayer(channels, kernel_size)

    def __call__(self, x: mx.array, hidden_state: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        batch_size, seq_len, d_model = x.shape
        residual = x
        x = self.layer_norm(x)

        if self.is_mlstm:
            return self._mlstm_forward(x, hidden_state, residual)
        else:
            return self._slstm_forward(x, hidden_state, residual)

    def _mlstm_forward(self, x: mx.array, hidden_state: Optional[mx.array], residual: mx.array) -> Tuple[
        mx.array, mx.array]:
        batch_size, seq_len, d_model = x.shape

        # Projections
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Gates with soft capping
        i_gate = mx.sigmoid(metal_soft_cap(self.i_proj(x), self.config.gate_soft_cap))
        f_gate = mx.sigmoid(metal_soft_cap(self.f_proj(x), self.config.gate_soft_cap))
        o_gate = mx.sigmoid(metal_soft_cap(self.o_proj(x), self.config.gate_soft_cap))

        # Initialize hidden state
        if hidden_state is None:
            hidden_state = mx.zeros(
                (batch_size, self.num_heads, self.head_dim, self.head_dim),
                dtype=x.dtype
            )

        outputs = []
        for t in range(seq_len):
            q_t = q[:, t]  # [batch_size, num_heads, head_dim]
            k_t = k[:, t]
            v_t = v[:, t]
            i_t = i_gate[:, t]  # [batch_size, num_heads]
            f_t = f_gate[:, t]
            o_t = o_gate[:, t]

            # Use Metal kernel for step computation
            h_t, hidden_state = metal_mlstm_step(q_t, k_t, v_t, i_t, f_t, o_t, hidden_state)
            outputs.append(h_t)

        output = mx.stack(outputs, axis=1)  # [batch_size, seq_len, num_heads, head_dim]
        output = output.reshape(batch_size, seq_len, -1)

        return residual + self.out_proj(output), hidden_state

    def _slstm_forward(self, x: mx.array, hidden_state: Optional[mx.array], residual: mx.array) -> Tuple[
        mx.array, mx.array]:
        batch_size, seq_len, d_model = x.shape

        # Causal convolution using Metal kernel
        x = self.conv(x)

        # Gates and projections
        i_gate = mx.sigmoid(metal_soft_cap(self.i_proj(x), self.config.gate_soft_cap))
        f_gate = mx.sigmoid(metal_soft_cap(self.f_proj(x), self.config.gate_soft_cap))
        o_gate = mx.sigmoid(metal_soft_cap(self.o_proj(x), self.config.gate_soft_cap))
        c_input = self.c_proj(x)

        proj_dim = i_gate.shape[-1]

        # Initialize hidden state
        if hidden_state is None:
            hidden_state = mx.zeros((batch_size, proj_dim), dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            c_t = c_input[:, t]
            i_t = i_gate[:, t]
            f_t = f_gate[:, t]
            o_t = o_gate[:, t]

            # Use Metal kernel for step computation
            h_t, hidden_state = metal_slstm_step(c_t, i_t, f_t, o_t, hidden_state)
            outputs.append(h_t)

        output = mx.stack(outputs, axis=1)

        return residual + self.out_proj(output), hidden_state


class MetalxLSTMModel(nn.Module):
    """Complete xLSTM model using Metal kernels"""

    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Layers
        self.blocks = []
        for i in range(config.num_layers):
            is_mlstm = i >= len(config.signature) or config.signature[i] == 0
            block = MetalxLSTMBlock(config, i, is_mlstm=is_mlstm)
            self.blocks.append(block)

        # Output head
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def init_hidden(self, batch_size: int) -> List[mx.array]:
        """Initialize hidden states for all blocks"""
        hidden_states = []
        for block in self.blocks:
            if block.is_mlstm:
                hidden_states.append(mx.zeros((batch_size, block.num_heads, block.head_dim, block.head_dim)))
            else:
                proj_dim = int(self.config.d_model * self.config.slstm_proj_factor)
                hidden_states.append(mx.zeros((batch_size, proj_dim)))
        return hidden_states

    def __call__(self, tokens: mx.array, hidden_states: Optional[List] = None) -> Tuple[mx.array, List]:
        x = self.embedding(tokens)

        if hidden_states is None:
            hidden_states = self.init_hidden(tokens.shape[0])

        for i, block in enumerate(self.blocks):
            x, hidden_states[i] = block(x, hidden_states[i])

        logits = self.head(x)

        # Apply soft capping to output logits
        if self.config.output_logit_soft_cap > 0:
            logits = metal_soft_cap(logits, self.config.output_logit_soft_cap)

        return logits, hidden_states


def create_metal_xlstm_model(
        vocab_size: int = 50257,
        num_layers: int = 6,
        d_model: int = 512,
        signature: Tuple[int, ...] = (1, 1),
        head_dim: int = 32,
        head_num: int = 4
) -> MetalxLSTMModel:
    """Create xLSTM model with Metal kernel optimizations"""

    config = xLSTMConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        signature=signature,
        head_dim=head_dim,
        head_num=head_num
    )

    return MetalxLSTMModel(config)


# Example usage and testing
if __name__ == "__main__":
    import time

    print("Creating xLSTM model with Metal kernels...")

    # Create model
    model = create_metal_xlstm_model(vocab_size=1000, num_layers=4, d_model=256, signature=(1, 0, 1, 0), head_num=8)

    # Test generation
    batch_size = 1
    seq_len = 32
    prompt = mx.random.randint(0, 1000, (batch_size, seq_len))

    print(f"Prompt shape: {prompt.shape}")

    # Forward pass
    start_time = time.time()
    logits, hidden_states = model(prompt)
    forward_time = time.time() - start_time

    print(f"Forward pass completed in {forward_time:.3f}s")
    print(f"Output shape: {logits.shape}")

    # Test soft capping
    test_tensor = mx.random.normal((100,)) * 10
    capped = metal_soft_cap(test_tensor, 5.0)
    print(f"Soft capping: max uncapped = {mx.max(test_tensor):.2f}, max capped = {mx.max(capped):.2f}")

    print("Metal xLSTM implementation complete!")
