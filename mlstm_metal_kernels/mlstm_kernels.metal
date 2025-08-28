
#include <metal_stdlib>
using namespace metal;

kernel void soft_cap_kernel(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& cap_value [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= size) return;
    float val = input[id];
    output[id] = cap_value * tanh(val / cap_value);
}

kernel void mlstm_step_kernel(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device float* v [[buffer(2)]],
    device float* i_gate [[buffer(3)]],
    device float* f_gate [[buffer(4)]],
    device float* o_gate [[buffer(5)]],
    device float* hidden_state [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    uint3 id [[thread_position_in_grid]]
) {
    uint batch = id.z;
    uint head = id.y;  
    uint dim = id.x;
    
    if (batch >= batch_size || head >= num_heads || dim >= head_dim) return;
    
    uint idx = batch * num_heads * head_dim + head * head_dim + dim;
    uint gate_idx = batch * num_heads + head;
    
    float q_val = q[idx];
    float k_val = k[idx];
    float v_val = v[idx];
    float i_val = i_gate[gate_idx];
    float f_val = f_gate[gate_idx];
    float o_val = o_gate[gate_idx];
    
    // Update matrix memory: H = f * H + i * (k ⊗ v)
    uint h_base = batch * num_heads * head_dim * head_dim + head * head_dim * head_dim;
    
    // Compute outer product k ⊗ v and update memory
    for (uint j = 0; j < head_dim; j++) {
        uint h_idx = h_base + dim * head_dim + j;
        float kv_outer = k_val * v[batch * num_heads * head_dim + head * head_dim + j];
        hidden_state[h_idx] = f_val * hidden_state[h_idx] + i_val * kv_outer;
    }
    
    // Compute output: h = H * q
    float h_sum = 0.0f;
    for (uint j = 0; j < head_dim; j++) {
        uint h_idx = h_base + j * head_dim + dim;
        h_sum += hidden_state[h_idx] * q_val;
    }
    
    // Apply output gate
    output[idx] = o_val * h_sum;
}
