#!/usr/bin/env python
"""
Stream-chained split FFT convolution kernels for 80-core GPU saturation.

Split the unified kernel at threadgroup_barrier boundaries into 4 phases:
1. FFT(k) - once per channel  
2. FFT(u) - per (batch, channel)
3. Complex multiply - per (batch, channel)
4. IFFT + bias - per (batch, channel)

Each phase gets its own compiled kernel. Orchestration uses MLX stream chaining.
"""

import mlx.core as mx
import mlx.nn as nn
from .typed import u32

# Global kernel cache - compile once, reuse forever
_KERNELS = {}

# Copy header and helpers from unified kernel
_HEADER = """#include <metal_stdlib>
using namespace metal;

constant float PI_F = 3.14159265358979323846f;
#define MAX_TW 2048

// Double-double helpers (copied from unified kernel)
struct dd_t { float hi; float lo; };
inline dd_t quick_two_sum(float a, float b) { float s=a+b; return dd_t{s, b-(s-a)}; }
inline dd_t two_sum(float a, float b) { float s=a+b; float v=s-a; return dd_t{s, (a-(s-v))+(b-v)}; }
inline dd_t two_prod(float a, float b) { float p=a*b; return dd_t{p, fma(a,b,-p)}; }
inline dd_t dd_add(dd_t a, dd_t b) { dd_t s=two_sum(a.hi,b.hi); dd_t t=two_sum(a.lo,b.lo); s.lo+=t.hi; s=quick_two_sum(s.hi,s.lo); s.lo+=t.lo; return quick_two_sum(s.hi,s.lo); }
inline dd_t dd_sub(dd_t a, dd_t b) { return dd_add(a, dd_t{-b.hi,-b.lo}); }
inline dd_t dd_mul(dd_t a, dd_t b) { dd_t p=two_prod(a.hi,b.hi); p.lo+=a.hi*b.lo+a.lo*b.hi; return quick_two_sum(p.hi,p.lo); }

inline float2 dd2_quick_two_sum(float a, float b) { float s=a+b; return float2(s, b-(s-a)); }
inline float2 dd2_two_sum(float a, float b) { float s=a+b; float v=s-a; return float2(s, (a-(s-v))+(b-v)); }
inline float2 dd2_two_prod(float a, float b) { float p=a*b; return float2(p, fma(a,b,-p)); }
inline float2 dd2_add(float2 a, float2 b) { float2 s=dd2_two_sum(a.x,b.x); float2 t=dd2_two_sum(a.y,b.y); s.y+=t.x; s=dd2_quick_two_sum(s.x,s.y); s.y+=t.y; return dd2_quick_two_sum(s.x,s.y); }
inline float2 dd2_sub(float2 a, float2 b) { return dd2_add(a, float2(-b.x,-b.y)); }
inline float2 dd2_mul(float2 a, float2 b) { float2 p=dd2_two_prod(a.x,b.x); p.y+=a.x*b.y+a.y*b.x; return dd2_quick_two_sum(p.x,p.y); }
inline float dd2_to_float(float2 a) { return a.x+a.y; }

inline void cdd2_mul(float2 ar, float2 ai, float2 br, float2 bi, thread float2& rr, thread float2& ri) {
    float2 ac=dd2_mul(ar,br), bd=dd2_mul(ai,bi);
    rr=dd2_sub(ac,bd);
    float2 ad=dd2_mul(ar,bi), bc=dd2_mul(ai,br);
    ri=dd2_add(ad,bc);
}

// FFT helper (copied from unified)
inline void fft_inplace_global_table(device float* re, device float* im, threadgroup const float* Twr, threadgroup const float* Twi, uint N, uint tid, uint tpg, bool inverse, bool comp_bfly) {
    uint logn=0; for(uint n=N; n>1; n>>=1) ++logn;
    // Bit reversal
    for(uint i=tid; i<N; i+=tpg) {
        uint j=0; for(uint k=0, m=i; k<logn; ++k, m>>=1) j=(j<<1)|(m&1);
        if(j>i) { float tr=re[i], ti=im[i]; re[i]=re[j]; im[i]=im[j]; re[j]=tr; im[j]=ti; }
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    // Early stages
    uint s_begin=1;
    if(logn>=2) {
        for(uint p=tid; p<(N>>1); p+=tpg) {
            uint i1=p<<1, i2=i1+1;
            float ur=re[i1], ui=im[i1], vr=re[i2], vi=im[i2];
            re[i1]=ur+vr; im[i1]=ui+vi; re[i2]=ur-vr; im[i2]=ui-vi;
        }
        threadgroup_barrier(mem_flags::mem_device);
        
        float sw1=inverse?1.0f:-1.0f;
        for(uint p=tid; p<(N>>2); p+=tpg) {
            uint b=p<<2;
            { uint i1=b, i2=b+2; float ur=re[i1], ui=im[i1], vr=re[i2], vi=im[i2];
              re[i1]=ur+vr; im[i1]=ui+vi; re[i2]=ur-vr; im[i2]=ui-vi; }
            { uint i1=b+1, i2=b+3; float ur=re[i1], ui=im[i1], vr=re[i2], vi=im[i2];
              float tr=-sw1*vi, ti=sw1*vr;
              re[i1]=ur+tr; im[i1]=ui+ti; re[i2]=ur-tr; im[i2]=ui-ti; }
        }
        threadgroup_barrier(mem_flags::mem_device);
        s_begin=3;
    }
    
    // Twiddle stages
    for(uint s=s_begin; s<=logn; ++s) {
        uint m=1<<s, halfm=m>>1, stride=N/m, blocks=N/m;
        for(uint j=tid; j<halfm; j+=tpg) {
            uint tw_idx=j*stride;
            float cw=Twr[tw_idx], sw=Twi[tw_idx];
            if(inverse) sw=-sw;
            for(uint g=0; g<blocks; ++g) {
                uint i1=g*m+j, i2=i1+halfm;
                float r2=re[i2], i2_v=im[i2];
                float tr=cw*r2-sw*i2_v, ti=cw*i2_v+sw*r2;
                float ur=re[i1], ui=im[i1];
                re[i1]=ur+tr; im[i1]=ui+ti; re[i2]=ur-tr; im[i2]=ui-ti;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    
    if(inverse) {
        float invN=1.0f/float(N);
        for(uint i=tid; i<N; i+=tpg) { re[i]*=invN; im[i]*=invN; }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"""

# Phase 1: FFT(k) - processes ALL channels in one launch
_SRC_FFT_K = r"""
    uint B=params[0], C=params[1], L=params[2], N=params[3];
    uint gtid=thread_position_in_grid.x, tpg=threads_per_threadgroup.x;
    uint c=gtid/tpg, tid=gtid-c*tpg;
    if(c>=C) return;
    
    uint k_base=c*L, f_base=c*N;
    
    threadgroup float Twr[MAX_TW], Twi[MAX_TW];
    uint halfN=N>>1;
    for(uint i=tid; i<halfN && i<MAX_TW; i+=tpg) {
        float ang=-2.0f*PI_F*(float(i)/float(N));
        float cw; float sw=precise::sincos(ang, cw);
        Twr[i]=cw; Twi[i]=sw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) { Twr[0]=1.0f; Twi[0]=0.0f; if((N&3)==0) { uint q=N>>2; if(q<MAX_TW) { Twr[q]=0.0f; Twi[q]=-1.0f; }}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for(uint i=tid; i<N; i+=tpg) {
        Kr_out[f_base+i]=(i<L)?k_time[k_base+i]:0.0f;
        Ki_out[f_base+i]=0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_inplace_global_table(&Kr_out[f_base], &Ki_out[f_base], Twr, Twi, N, tid, tpg, false, true);
    if(tid==0) { Ki_out[f_base]=0.0f; if((N&1)==0) Ki_out[f_base+(N>>1)]=0.0f; }
"""

# Phase 2: FFT(u) - ONE (b,c) pair per launch
_SRC_FFT_U = r"""
    uint B=params[0], C=params[1], L=params[2], N=params[3], b=params[4], c=params[5];
    uint tid=thread_position_in_threadgroup.x, tpg=threads_per_threadgroup.x;
    
    uint u_base=(b*C+c)*L, f_base=(b*C+c)*N;
    
    threadgroup float Twr[MAX_TW], Twi[MAX_TW];
    uint halfN=N>>1;
    for(uint i=tid; i<halfN && i<MAX_TW; i+=tpg) {
        float ang=-2.0f*PI_F*(float(i)/float(N));
        float cw; float sw=precise::sincos(ang, cw);
        Twr[i]=cw; Twi[i]=sw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) { Twr[0]=1.0f; Twi[0]=0.0f; if((N&3)==0) { uint q=N>>2; if(q<MAX_TW) { Twr[q]=0.0f; Twi[q]=-1.0f; }}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for(uint i=tid; i<N; i+=tpg) {
        Ur_out[f_base+i]=(i<L)?u[u_base+i]:0.0f;
        Ui_out[f_base+i]=0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_inplace_global_table(&Ur_out[f_base], &Ui_out[f_base], Twr, Twi, N, tid, tpg, false, true);
    if(tid==0) { Ui_out[f_base]=0.0f; if((N&1)==0) Ui_out[f_base+(N>>1)]=0.0f; }
"""

# Phase 3: Complex multiply - ONE (b,c) pair per launch
_SRC_CMUL = r"""
    uint B=params[0], C=params[1], N=params[2], b=params[3], c=params[4];
    uint tid=thread_position_in_threadgroup.x, tpg=threads_per_threadgroup.x;
    
    uint u_f=(b*C+c)*N, k_f=c*N;
    
    for(uint i=tid; i<N; i+=tpg) {
        float ur=Ur_in[u_f+i], ui=Ui_in[u_f+i];
        float kr=Kr_in[k_f+i], ki=Ki_in[k_f+i];
        float2 ar=float2(ur,0.0f), ai=float2(ui,0.0f);
        float2 br=float2(kr,0.0f), bi=float2(ki,0.0f);
        thread float2 rr, ri;
        cdd2_mul(ar,ai,br,bi,rr,ri);
        Ur_out[u_f+i]=dd2_to_float(rr);
        Ui_out[u_f+i]=dd2_to_float(ri);
    }
"""

# Phase 4: IFFT + bias - ONE (b,c) pair per launch
_SRC_IFFT = r"""
    uint B=params[0], C=params[1], L=params[2], N=params[3], b=params[4], c=params[5];
    uint tid=thread_position_in_threadgroup.x, tpg=threads_per_threadgroup.x;
    
    uint u_base=(b*C+c)*L, f_base=(b*C+c)*N;
    
    threadgroup float Twr[MAX_TW], Twi[MAX_TW];
    uint halfN=N>>1;
    for(uint i=tid; i<halfN && i<MAX_TW; i+=tpg) {
        float ang=-2.0f*PI_F*(float(i)/float(N));
        float cw; float sw=precise::sincos(ang, cw);
        Twr[i]=cw; Twi[i]=sw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) { Twr[0]=1.0f; Twi[0]=0.0f; if((N&3)==0) { uint q=N>>2; if(q<MAX_TW) { Twr[q]=0.0f; Twi[q]=-1.0f; }}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Copy to output buffer for in-place IFFT
    for(uint i=tid; i<N; i+=tpg) {
        Ur_work[f_base+i]=Ur_in[f_base+i];
        Ui_work[f_base+i]=Ui_in[f_base+i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    
    fft_inplace_global_table(&Ur_work[f_base], &Ui_work[f_base], Twr, Twi, N, tid, tpg, true, true);
    
    float bias=D[c];
    for(uint i=tid; i<L; i+=tpg) {
        y_out[u_base+i]=Ur_work[f_base+i]+u[u_base+i]*bias;
    }
"""

def _get_kernel(name):
    if name not in _KERNELS:
        if name == 'fft_k':
            _KERNELS[name] = mx.fast.metal_kernel(name="fft_k", input_names=["params","k_time"], output_names=["Kr_out","Ki_out"], header=_HEADER, source=_SRC_FFT_K)
        elif name == 'fft_u':
            _KERNELS[name] = mx.fast.metal_kernel(name="fft_u", input_names=["params","u"], output_names=["Ur_out","Ui_out"], header=_HEADER, source=_SRC_FFT_U)
        elif name == 'cmul':
            _KERNELS[name] = mx.fast.metal_kernel(name="cmul", input_names=["params","Ur_in","Ui_in","Kr_in","Ki_in"], output_names=["Ur_out","Ui_out"], header=_HEADER, source=_SRC_CMUL)
        elif name == 'ifft':
            _KERNELS[name] = mx.fast.metal_kernel(name="ifft", input_names=["params","Ur_in","Ui_in","u","D"], output_names=["Ur_work","Ui_work","y_out"], header=_HEADER, source=_SRC_IFFT)
    return _KERNELS[name]


class MetalFFTConvStreamed(nn.Module):
    def __init__(self, num_streams=8):
        super().__init__()
        self.num_streams = num_streams
        self.streams = [mx.new_stream(mx.default_device()) for _ in range(num_streams)]
    
    def __call__(self, u, k, D):
        B, C, L = u.shape
        N = 2 * L
        
        u, k = u.astype(mx.float32), k.astype(mx.float32)
        D = D.reshape(-1).astype(mx.float32) if D.ndim == 3 else D.astype(mx.float32)
        
        # Allocate workspace
        Kr = mx.zeros((C, N), dtype=mx.float32)
        Ki = mx.zeros((C, N), dtype=mx.float32)
        Ur = mx.zeros((B, C, N), dtype=mx.float32)
        Ui = mx.zeros((B, C, N), dtype=mx.float32)
        y = mx.zeros((B, C, L), dtype=mx.float32)
        
        one, tpg = u32(1), u32(min(256, N))
        
        # Phase 1: FFT(k) all channels on stream 0
        with mx.stream(self.streams[0]):
            params = mx.array([B, C, L, N], dtype=mx.uint32)
            Kr, Ki = _get_kernel('fft_k')(
                inputs=[params, k.reshape(-1)],
                output_shapes=[(C*N,), (C*N,)],
                output_dtypes=[mx.float32, mx.float32],
                grid=(mx.multiply(u32(C), tpg), one, one),
                threadgroup=(tpg, one, one)
            )
            Kr, Ki = Kr.reshape(C, N), Ki.reshape(C, N)
        
        # Phases 2-4: per (b,c) with stream chaining
        y_slices = []
        
        for b in range(B):
            for c in range(C):
                s = self.streams[(b*C+c) % self.num_streams]
                with mx.stream(s):
                    p = mx.array([B, C, L, N, b, c], dtype=mx.uint32)
                    g, tg = (tpg, one, one), (tpg, one, one)
                    
                    # Phase 2: FFT(u)
                    Ur_bc, Ui_bc = _get_kernel('fft_u')(
                        inputs=[p, u.reshape(-1)],
                        output_shapes=[(B*C*N,), (B*C*N,)],
                        output_dtypes=[mx.float32, mx.float32],
                        grid=g, threadgroup=tg
                    )
                    
                    # Phase 3: cmul
                    p2 = mx.array([B, C, N, b, c], dtype=mx.uint32)
                    Ur_bc, Ui_bc = _get_kernel('cmul')(
                        inputs=[p2, Ur_bc, Ui_bc, Kr.reshape(-1), Ki.reshape(-1)],
                        output_shapes=[(B*C*N,), (B*C*N,)],
                        output_dtypes=[mx.float32, mx.float32],
                        grid=g, threadgroup=tg
                    )
                    
                    # Phase 4: IFFT
                    _, _, y_full = _get_kernel('ifft')(
                        inputs=[p, Ur_bc, Ui_bc, u.reshape(-1), D],
                        output_shapes=[(B*C*N,), (B*C*N,), (B*C*L,)],
                        output_dtypes=[mx.float32, mx.float32, mx.float32],
                        grid=g, threadgroup=tg
                    )
                    
                    # Extract just this (b,c) slice
                    start = (b * C + c) * L
                    y_slice = y_full[start:start+L]
                    y_slices.append(y_slice)
        
        mx.synchronize()
        return mx.stack(y_slices).reshape(B, C, L)
