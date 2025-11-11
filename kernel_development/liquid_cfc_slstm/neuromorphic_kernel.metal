//
//  NeuromorphicKernel.metal
//  MetalCoroutinesTest
//
//  Created by Sydney Renee on 2/23/25.
//

#include <metal_stdlib>
using namespace metal;

// Threadgroup size - adjust as needed
#define TILE_SIZE 16

// Structure for scalar parameters
struct KernelParams {
    int N;
    float dt;
    int num_steps;
    float alpha;
    float target_sum;
    float neural_clock;
    uint step_num;
    float eta;
    bool use_hebbian;
    float decay_rate;
};

kernel void liquid_cfc_xlstm_kernel(
    uint2 gid [[thread_position_in_grid]],  // 2D Global Thread ID
    constant KernelParams &params [[buffer(0)]],  // Pass parameters as a struct
    device atomic_float* W_recurrent [[buffer(1)]],      // NOW DEVICE ATOMIC, for Hebbian updates
    constant float* W_i [[buffer(2)]],          // [N]
    constant float* U_i [[buffer(3)]],          // [N]
    constant float* b_i [[buffer(4)]],          // [N]
    constant float* W_f [[buffer(5)]],          // [N]
    constant float* U_f [[buffer(6)]],          // [N]
    constant float* b_f [[buffer(7)]],          // [N]
    constant float* W_o [[buffer(8)]],         // [N]
    constant float* U_o [[buffer(9)]],         // [N]
    constant float* b_o [[buffer(10)]],         // [N]
    constant float* W_g [[buffer(11)]],         // [N]
    constant float* U_g [[buffer(12)]],         // [N]
    constant float* b_g [[buffer(13)]],         // [N]
    constant float* lambda [[buffer(14)]],      // [N]
    constant int* gate_mask [[buffer(15)]],     // [N]
    constant int* lambda_mask [[buffer(16)]],   // [N]
    device float* h_liquid_read [[buffer(17)]], // [N]
    device float* h_liquid_write [[buffer(18)]],// [N]
    device float* c_t [[buffer(19)]],           // [N]
    device float* n_t [[buffer(20)]],           // [N]
    device char* logBuffer [[buffer(21)]],      // Log buffer
    uint2 lid [[thread_position_in_threadgroup]] // Local ID, 2D
    )
     {
        uint i = gid.y * TILE_SIZE + lid.y;  // Global linear index, from 2D gid
        if (i >= uint(params.N)) return;

        // Double buffering for h_liquid
        device float* h_liquid_current = (params.step_num % 2u == 0u) ? h_liquid_read : h_liquid_write;
        device float* h_liquid_next = (params.step_num % 2u == 0u) ? h_liquid_write : h_liquid_read;

        // Threadgroup memory for tiling
        threadgroup float W_tile[TILE_SIZE][TILE_SIZE];
        threadgroup float h_tile[TILE_SIZE];

        float x_t = 0.0f;

        // Tiled matrix multiplication (W_recurrent * h_liquid_current)
        uint numTiles = (uint(params.N) + TILE_SIZE - 1u) / TILE_SIZE;
        for (uint tile = 0; tile < numTiles; tile++) {
            // Load tile of W_recurrent into threadgroup memory
            uint row = gid.y * TILE_SIZE + lid.y;
            uint col = tile * TILE_SIZE + lid.x;
            if (row < uint(params.N) && col < uint(params.N)) {
                W_tile[lid.y][lid.x] = atomic_load_explicit(&W_recurrent[row * uint(params.N) + col], memory_order_relaxed); // Atomic Load
            } else {
                W_tile[lid.y][lid.x] = 0.0f;
            }

            // Load tile of h_liquid_current into threadgroup memory
            uint h_index = tile * TILE_SIZE + lid.x;
            if (h_index < uint(params.N)) {
                h_tile[lid.x] = h_liquid_current[h_index];
            } else {
                h_tile[lid.x] = 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Compute partial dot product within the tile
            for (uint k = 0; k < TILE_SIZE; k++) {
                if (row < uint(params.N) && (tile * TILE_SIZE + k) < uint(params.N)) {
                    x_t += W_tile[lid.y][k] * h_tile[k];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // xLSTM gates
        float i_t, f_t, o_t;
        if (gate_mask[i] == 0) {
            i_t = 1.0f;
            f_t = 1.0f;
            o_t = 1.0f;
        } else {
            float input_i = W_i[i] * x_t + U_i[i] * h_liquid_current[i] + b_i[i] - n_t[i];
            float input_f = W_f[i] * x_t + U_f[i] * h_liquid_current[i] + b_f[i] - n_t[i];
            float input_o = W_o[i] * x_t + U_o[i] * h_liquid_current[i] + b_o[i] - n_t[i];
            i_t = exp(input_i);
            f_t = exp(input_f);
            o_t = exp(input_o);
        }

        // Candidate cell update (sigmoid)
        float g_t = 1.0f / (1.0f + exp(-(W_g[i] * x_t + U_g[i] * h_liquid_current[i] + b_g[i])));

        // Update cell state
        float c_new = f_t * c_t[i] + i_t * g_t;

        // Compute feed-forward value for CfC
        float feed_forward = o_t * (1.0f / (1.0f + exp(-c_new)));

        // Sparsity control via lambda_mask
        float effective_lambda = (lambda_mask[i] == 0) ? 0.0f : lambda[i];

        // Update hidden state using CfC formula
        float h_old = h_liquid_current[i];
        float denom = 1.0f + params.neural_clock * effective_lambda;
        float h_new = (h_old + params.neural_clock * feed_forward) / denom;

        // Update normalizer (only when gating is active)
        if (gate_mask[i] == 1) {
            float sum_gates = i_t + f_t + o_t;
            float n_new = n_t[i] + params.alpha * (sum_gates - params.target_sum);
            n_t[i] = n_new;
        }

        // Optional Hebbian Update
        if (params.use_hebbian) {
            for (uint j = 0; j < uint(params.N); j++) {
                float delta_w = params.eta * h_liquid_next[j] * h_new * i_t;  // Gated by input, use next state
                float w_recurrent_value = atomic_load_explicit(&W_recurrent[j * uint(params.N) + i], memory_order_relaxed); // Get value
                delta_w -= params.decay_rate * w_recurrent_value; // Weight decay and correct index
                atomic_fetch_add_explicit((device atomic_float*)&W_recurrent[j * uint(params.N) + i], delta_w, memory_order_relaxed); // Atomic update
            }
        }

        // Write new states
        h_liquid_next[i] = h_new;
        c_t[i] = c_new;

        // Log neuron state for debugging
        if (i == 0) {
            // Format the log message
            constant char* logMessage = "Neuron 0 state: ";
            int logMessageLength = 0;
            while (logMessage[logMessageLength] != '\0') {
                logMessageLength++;
            }

            // Write the log message to the log buffer
            for (int k = 0; k < logMessageLength; ++k) {
                logBuffer[k] = logMessage[k];
            }

            // Convert h_new to string and write to log buffer
            int index = logMessageLength;
            float value = h_new;
            int intPart = int(value);
            float fracPart = value - float(intPart);
            int fracInt = int(fracPart * 1000000); // 6 decimal places

            // Write integer part
            if (intPart == 0) {
                logBuffer[index++] = '0';
            } else {
                if (intPart < 0) {
                    logBuffer[index++] = '-';
                    intPart = -intPart;
                }
                char intStr[10];
                int intLen = 0;
                while (intPart > 0) {
                    intStr[intLen++] = '0' + (intPart % 10);
                    intPart /= 10;
                }
                for (int j = intLen - 1; j >= 0; --j) {
                    logBuffer[index++] = intStr[j];
                }
            }

            // Write decimal point
            logBuffer[index++] = '.';

            // Write fractional part
            char fracStr[7];
            int fracLen = 0;
            while (fracInt > 0) {
                fracStr[fracLen++] = '0' + (fracInt % 10);
                fracInt /= 10;
            }
            for (int j = 6 - fracLen; j > 0; --j) {
                logBuffer[index++] = '0';
            }
            for (int j = fracLen - 1; j >= 0; --j) {
                logBuffer[index++] = fracStr[j];
            }

            // Null-terminate the string
            logBuffer[index] = '\0';
        }
    }

