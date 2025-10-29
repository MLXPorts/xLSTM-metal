#include <metal_stdlib>
using namespace metal;
inline float logsigmoid(float x) {
    // Stable computation of log(sigmoid(x)) = -softplus(-x)
    if (x >= 0.0f) {
        // exp(-x) <= 1, safe
        return -log(1.0f + exp(-x));
    } else {
        // exp(x) <= 1, safe
        return x - log(1.0f + exp(x));
    }
}

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

// Simple memcpy to validate pointer mapping
kernel void memcpy_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= size) return;
    output[id] = input[id];
}

kernel void mlstm_step_full_kernel(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* i_pre [[buffer(3)]],
    device const float* f_pre [[buffer(4)]],
    device float* C [[buffer(5)]],
    device float* N [[buffer(6)]],
    device float* M [[buffer(7)]],
    device float* H [[buffer(8)]],
    constant uint& B [[buffer(9)]],
    constant uint& NH [[buffer(10)]],
    constant uint& DHQK [[buffer(11)]],
    constant uint& DHHV [[buffer(12)]],
    constant float& eps [[buffer(13)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint b = tid.z;
    uint h = tid.y;
    uint dv = tid.x;
    if (b >= B || h >= NH || dv >= DHHV) return;

    uint bh = b * NH + h;
    uint idx_q_base = bh * DHQK;
    uint idx_v_base = bh * DHHV;
    uint idx_C_base = bh * DHQK * DHHV;

    // Gating per (b,h)
    float m_old = M[bh];
    float i_p = i_pre[bh];
    float f_p = f_pre[bh];
    float f_log = logsigmoid(f_p);
    float m_new = max(f_log + m_old, i_p);
    float i_act = exp(i_p - m_new);
    float f_act = exp(f_log + m_old - m_new);

    // q scaling by 1/sqrt(DHQK)
    float inv_sqrt_DHQK = rsqrt((float)DHQK);

    float h_num = 0.0f;
    float qn_dot = 0.0f; // q_scaled · N_new

    for (uint j = 0; j < DHQK; ++j) {
        float qj = q[idx_q_base + j];
        float qj_scaled = qj * inv_sqrt_DHQK;
        float kj = k[idx_q_base + j];

        // C update at (j,dv)
        uint c_idx = idx_C_base + j * DHHV + dv;
        float c_old = C[c_idx];
        float c_new = f_act * c_old + i_act * kj * v[idx_v_base + dv];
        C[c_idx] = c_new;
        h_num += c_new * qj_scaled;

        // N update at (j)
        float n_old = N[idx_q_base + j];
        float n_new = f_act * n_old + i_act * kj;
        if (dv == 0) {
            N[idx_q_base + j] = n_new;
        }
        qn_dot += qj_scaled * n_new;
    }

    if (dv == 0) {
        M[bh] = m_new;
    }

    // denom = max(|q·N|, exp(-M)) + eps
    float denom = max(fabs(qn_dot), exp(-m_new)) + eps;
    H[idx_v_base + dv] = h_num / denom;
}
