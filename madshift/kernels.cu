// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Matthew Priester
//
// kernels.cu: the trellis search of madshift/trellis.py (search()) as CUDA kernels, with the SAME floating-point operations in
// the SAME order, so the chosen states and the dequantized values are bit-identical to the PyTorch reference.
//
// Per window of Bdec committed columns (window Wwin):
//   Viterbi over columns [c0, c0 + T) of the error-fed matrix Wc  ->  nd committed states per row
//   for c in committed columns:  q = code(s) * sig;  e = (Wc[:, c] - q) / Hu[c, c];  Wc[:, c+1:] -= e * Hu[c, c+1:]
// The last line touches every later column for every column: O(R C^2) memory traffic, the dominant cost for wide matrices.
// Here every element still receives the updates fl(w - fl(e_c * Hu[c, j])) for c = 0, 1, ... in increasing order, but they are
// applied lazily, many columns at a time (vit_apply: a register-tiled kernel with the sequential accumulation order), and only
// the columns the next window reads are kept current. Total memory traffic O(R C (C / PB + const)).
//
// vit2: one CUDA block per row; the 1MAD code value computed in registers (no table reads), the row's branch values v, v^2 and
// -2v kept in registers for the whole window, and the min over predecessors split over all threads for k >= 5. Ties keep the
// lowest index, as torch's min / argmin.
//
// Build: build_kernels.bat (Windows) or build_kernels.sh (Linux). Loaded by madshift/kernels.py (ctypes).
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

__device__ __forceinline__ float mad_val(unsigned s) {      // 1MAD: bytesum(34038481 s + 76625530 mod 2^32) - 510
    unsigned x = 34038481u * s + 76625530u;
    int v = (int)(x & 255u) + (int)((x >> 8) & 255u) + (int)((x >> 16) & 255u) + (int)(x >> 24) - 510;
    return (float)v;
}

__device__ __forceinline__ float code_val(const float* __restrict__ cbp, unsigned s) {
    return cbp ? cbp[s] : mad_val(s);
}

// mode bit 0: addcmul fused (fma) or two roundings; bit 1: cost update fused or two roundings (mode 2 = the roundings of torch.addcmul / torch.add on CUDA)
__device__ __forceinline__ float inc_of(float vsq, float a2, float x, int mode) {
    return (mode & 1) ? __fmaf_rn(a2, x, vsq) : __fadd_rn(vsq, __fmul_rn(a2, x));
}
__device__ __forceinline__ float upd_of(float m, float inc, float p, int mode) {
    return (mode & 2) ? __fmaf_rn(p, inc, m) : __fadd_rn(m, __fmul_rn(p, inc));
}

template <int NT>
__global__ void __launch_bounds__(NT, 1)
vit2(const float* __restrict__ X, int ldx, const float* __restrict__ sig, const float* __restrict__ cbp,
     const float* __restrict__ piv, const long long* __restrict__ state, int has_state, int k, int T, int nd,
     uint8_t* __restrict__ bp, long long* __restrict__ out, int ldo, int mode) {
    constexpr int L = 14, S = 1 << L, NS = S / NT;
    extern __shared__ float sm[];
    __shared__ float rv[32];
    __shared__ int ri[32];
    const int A = 1 << k, J = S >> k, tid = threadIdx.x;
    float* cur = sm;                         // S path costs
    float* mj = sm + S;                      // J best-predecessor costs (k <= 4) or NT partial mins (k >= 5)
    uint8_t* pa = (uint8_t*)(mj + (J >= NT ? J : NT));   // NT partial argmins (k >= 5)
    float* mj2 = (float*)(pa + NT);          // J combined mins (k >= 5)
    const int r = blockIdx.x;
    const float* x = X + (size_t)r * ldx;
    const float sg = sig[r];
    uint8_t* bpr = bp + (size_t)r * T * J;
    const int keepj = has_state ? (int)(state[r] & (J - 1)) : -1;
    float vsq[NS], a2[NS];
    {
        const float x0 = x[0], p0 = piv[0];
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int s = tid + i * NT;
            const float v = __fmul_rn(code_val(cbp, (unsigned)s), sg);
            vsq[i] = __fmul_rn(v, v);
            a2[i] = __fmul_rn(-2.0f, v);
            const float c = __fmul_rn(inc_of(vsq[i], a2[i], x0, mode), p0);
            cur[s] = (keepj < 0 || (s >> k) == keepj) ? c : INFINITY;
        }
    }
    for (int t = 1; t < T; ++t) {
        __syncthreads();
        const float xt = x[t], pt = piv[t];
        const float* mjr;
        if (J >= NT) {                                       // k <= 4: each thread owns whole predecessor sets
            for (int j = tid; j < J; j += NT) {
                float m = cur[j];
                int am = 0;
                for (int a = 1; a < A; ++a) {
                    const float c = cur[a * J + j];
                    if (c < m) { m = c; am = a; }
                }
                bpr[(size_t)t * J + j] = (uint8_t)am;
                mj[j] = m;
            }
            mjr = mj;
        } else {                                             // k >= 5: P = NT / J threads per j, A / P predecessors each
            const int P = NT / J, AP = A / P, j = tid & (J - 1), p = tid / J, a0 = p * AP;
            float m = cur[a0 * J + j];
            int am = a0;
            for (int a = a0 + 1; a < a0 + AP; ++a) {
                const float c = cur[a * J + j];
                if (c < m) { m = c; am = a; }
            }
            mj[tid] = m;
            pa[tid] = (uint8_t)am;
            __syncthreads();
            if (tid < J) {
                float m2 = mj[tid];
                int am2 = pa[tid];
                for (int q = 1; q < P; ++q) {
                    const float c = mj[q * J + tid];
                    if (c < m2) { m2 = c; am2 = pa[q * J + tid]; }
                }
                bpr[(size_t)t * J + tid] = (uint8_t)am2;
                mj2[tid] = m2;
            }
            mjr = mj2;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int s = tid + i * NT;
            cur[s] = upd_of(mjr[s >> k], inc_of(vsq[i], a2[i], xt, mode), pt, mode);
        }
    }
    __syncthreads();
    float bv = INFINITY;
    int bi = 0x7fffffff;
    for (int s = tid; s < S; s += NT) {
        const float c = cur[s];
        if (c < bv || (c == bv && s < bi)) { bv = c; bi = s; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffff, bv, o);
        const int oi = __shfl_down_sync(0xffffffff, bi, o);
        if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const int lane = tid & 31, warp = tid >> 5, nw = NT >> 5;
    if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < nw; ++w)
            if (rv[w] < bv || (rv[w] == bv && ri[w] < bi)) { bv = rv[w]; bi = ri[w]; }
        if (bi == 0x7fffffff) bi = 0;
        int path[64];
        int s = bi;
        path[T - 1] = s;
        for (int t = T - 1; t > 0; --t) {
            const int j = s >> k;
            s = ((int)bpr[(size_t)t * J + j] << (L - k)) | j;
            path[t - 1] = s;
        }
        for (int i = 0; i < nd; ++i) out[(size_t)r * ldo + i] = path[i];
    }
}

// vit3: vit2 specialized at compile time on k and the rounding mode (no runtime branches, unrolled predecessor loops), the
// window's x and piv in shared memory, and for k >= 5 the back-pointers in shared memory (traceback without L2 round trips).
// Same operations in the same order as vit2, so the same decisions.
template <int NT, int K, int MODE, bool BPS>
__global__ void __launch_bounds__(NT, 1)
vit3(const float* __restrict__ X, int ldx, const float* __restrict__ sig, const float* __restrict__ cbp,
     const float* __restrict__ piv, const long long* __restrict__ state, int has_state, int T, int nd,
     uint8_t* __restrict__ bp, long long* __restrict__ out, int ldo) {
    constexpr int L = 14, S = 1 << L, NS = S / NT, A = 1 << K, J = S >> K;
    constexpr bool SPLIT = J < NT;
    constexpr int P = SPLIT ? NT / J : 1;
    [[maybe_unused]] constexpr int AP = A / P;
    extern __shared__ float sm[];
    __shared__ float xs[64], ps[64];
    __shared__ float rv[32];
    __shared__ int ri[32];
    float* cur = sm;
    float* pm = sm + S;                                       // SPLIT: NT partial mins; else J mins
    uint8_t* pa = (uint8_t*)(pm + (SPLIT ? NT : J));          // SPLIT: NT partial argmins
    float* mj2 = (float*)(pa + (SPLIT ? NT : 0));             // SPLIT: J mins
    uint8_t* bps = (uint8_t*)(mj2 + (SPLIT ? J : 0));         // BPS: T * J back-pointers
    const int tid = threadIdx.x, r = blockIdx.x;
    const float* x = X + (size_t)r * ldx;
    if (tid < T) { xs[tid] = x[tid]; ps[tid] = piv[tid]; }
    const float sg = sig[r];
    uint8_t* bpr = BPS ? bps : bp + (size_t)r * T * J;
    const int keepj = has_state ? (int)(state[r] & (J - 1)) : -1;
    __syncthreads();
    float vsq[NS], a2[NS];
    {
        const float x0 = xs[0], p0 = ps[0];
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int s = tid + i * NT;
            const float v = __fmul_rn(code_val(cbp, (unsigned)s), sg);
            vsq[i] = __fmul_rn(v, v);
            a2[i] = __fmul_rn(-2.0f, v);
            const float c = __fmul_rn(inc_of(vsq[i], a2[i], x0, MODE), p0);
            cur[s] = (keepj < 0 || (s >> K) == keepj) ? c : INFINITY;
        }
    }
    for (int t = 1; t < T; ++t) {
        __syncthreads();
        const float xt = xs[t], pt = ps[t];
        const float* mjr;
        if constexpr (!SPLIT) {
#pragma unroll
            for (int q = 0; q < J / NT; ++q) {
                const int j = tid + q * NT;
                float m = cur[j];
                int am = 0;
#pragma unroll
                for (int a = 1; a < A; ++a) {
                    const float c = cur[a * J + j];
                    if (c < m) { m = c; am = a; }
                }
                bpr[(size_t)t * J + j] = (uint8_t)am;
                pm[j] = m;
            }
            mjr = pm;
        } else {
            const int j = tid & (J - 1), p = tid / J, a0 = p * AP;
            float m = cur[a0 * J + j];
            int am = a0;
#pragma unroll
            for (int a = 1; a < AP; ++a) {
                const float c = cur[(a0 + a) * J + j];
                if (c < m) { m = c; am = a0 + a; }
            }
            pm[tid] = m;
            pa[tid] = (uint8_t)am;
            __syncthreads();
            if (tid < J) {
                float m2 = pm[tid];
                int am2 = pa[tid];
#pragma unroll
                for (int q = 1; q < P; ++q) {
                    const float c = pm[q * J + tid];
                    if (c < m2) { m2 = c; am2 = pa[q * J + tid]; }
                }
                bpr[(size_t)t * J + tid] = (uint8_t)am2;
                mj2[tid] = m2;
            }
            mjr = mj2;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < NS; ++i) {
            const int s = tid + i * NT;
            cur[s] = upd_of(mjr[s >> K], inc_of(vsq[i], a2[i], xt, MODE), pt, MODE);
        }
    }
    __syncthreads();
    float bv = INFINITY;
    int bi = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < NS; ++i) {
        const int s = tid + i * NT;
        const float c = cur[s];
        if (c < bv || (c == bv && s < bi)) { bv = c; bi = s; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffff, bv, o);
        const int oi = __shfl_down_sync(0xffffffff, bi, o);
        if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const int lane = tid & 31, warp = tid >> 5, nw = NT >> 5;
    if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < nw; ++w)
            if (rv[w] < bv || (rv[w] == bv && ri[w] < bi)) { bv = rv[w]; bi = ri[w]; }
        if (bi == 0x7fffffff) bi = 0;
        int path[64];
        int s = bi;
        path[T - 1] = s;
        for (int t = T - 1; t > 0; --t) {
            const int j = s >> K;
            s = ((int)bpr[(size_t)t * J + j] << (L - K)) | j;
            path[t - 1] = s;
        }
        for (int i = 0; i < nd; ++i) out[(size_t)r * ldo + i] = path[i];
    }
}

// vit4: the path costs live in registers. Thread t owns a fixed set of 16 states s = a * J + j (all predecessors a of its j's
// for k <= 4; for k >= 5 a contiguous range of a for one j, P = 1024 / J threads per j). The best-predecessor min of a j is then
// a register scan (no shared-memory reads), and the update of an owned state reads one value, mj[s >> k], from shared memory
// (double-buffered, so one barrier per step for k <= 4). Same comparisons in the same order (lowest index on ties), same
// floating-point operations, so the same decisions as vit2 / vit3.
template <int K, bool BPS>
__global__ void __launch_bounds__(1024, 1)
vit4(const float* __restrict__ X, int ldx, const float* __restrict__ sig, const float* __restrict__ cbp,
     const float* __restrict__ piv, const long long* __restrict__ state, int has_state, int T, int nd,
     uint8_t* __restrict__ bp, long long* __restrict__ out, int ldo) {
    constexpr int NT = 1024, L = 14, S = 1 << L, NS = S / NT, A = 1 << K, J = S >> K;
    constexpr bool SPLIT = J < NT;
    constexpr int NJ = SPLIT ? 1 : J / NT, NA = SPLIT ? A / (NT / J) : A, P = SPLIT ? NT / J : 1;
    static_assert(NJ * NA == NS, "16 states per thread");
    extern __shared__ float sm[];
    __shared__ float xs[64], ps[64];
    __shared__ float rv[32];
    __shared__ int ri[32];
    float* mjb = sm;                                          // 2 x J best-predecessor costs (double buffer)
    float* pmb = sm + 2 * J;                                  // SPLIT: 2 x NT partial mins
    uint8_t* pab = (uint8_t*)(pmb + (SPLIT ? 2 * NT : 0));    // SPLIT: 2 x NT partial argmins
    uint8_t* bps = pab + (SPLIT ? 2 * NT : 0);                // BPS: T x J back-pointers
    const int tid = threadIdx.x, r = blockIdx.x;
    const float* x = X + (size_t)r * ldx;
    if (tid < T) { xs[tid] = x[tid]; ps[tid] = piv[tid]; }
    const float sg = sig[r];
    uint8_t* bpr = BPS ? bps : bp + (size_t)r * T * J;
    const int keepj = has_state ? (int)(state[r] & (J - 1)) : -1;
    const int jbase = SPLIT ? (tid & (J - 1)) : tid, abase = SPLIT ? (tid / J) * NA : 0;
    float c[NS], vsq[NS], a2[NS];
    __syncthreads();
    {
        const float x0 = xs[0], p0 = ps[0];
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj)
#pragma unroll
            for (int aa = 0; aa < NA; ++aa) {
                const int i = jj * NA + aa, s = ((abase + aa) << (L - K)) | (jbase + jj * NT);
                const float v = __fmul_rn(code_val(cbp, (unsigned)s), sg);
                vsq[i] = __fmul_rn(v, v);
                a2[i] = __fmul_rn(-2.0f, v);
                const float cc = __fmul_rn(inc_of(vsq[i], a2[i], x0, 2), p0);
                c[i] = (keepj < 0 || (s >> K) == keepj) ? cc : INFINITY;
            }
    }
    for (int t = 1; t < T; ++t) {
        float* mj = mjb + (t & 1) * J;
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            // pairwise tree; the right operand wins only if strictly smaller: the leftmost minimum, as the sequential scan
            float tv[NA]; int ti[NA];
#pragma unroll
            for (int aa = 0; aa < NA; ++aa) { tv[aa] = c[jj * NA + aa]; ti[aa] = aa; }
#pragma unroll
            for (int w = 1; w < NA; w <<= 1)
#pragma unroll
                for (int aa = 0; aa + w < NA; aa += 2 * w)
                    if (tv[aa + w] < tv[aa]) { tv[aa] = tv[aa + w]; ti[aa] = ti[aa + w]; }
            const float m = tv[0];
            const int am = abase + ti[0];
            if constexpr (!SPLIT) {
                const int j = jbase + jj * NT;
                mj[j] = m;
                bpr[(size_t)t * J + j] = (uint8_t)am;
            } else {
                pmb[(t & 1) * NT + tid] = m;
                pab[(t & 1) * NT + tid] = (uint8_t)am;
            }
        }
        __syncthreads();
        if constexpr (SPLIT) {
            if (tid < J) {
                const float* pm = pmb + (t & 1) * NT;
                const uint8_t* pa = pab + (t & 1) * NT;
                float m2 = pm[tid];
                int am2 = pa[tid];
#pragma unroll
                for (int q = 1; q < P; ++q) {
                    const float v = pm[q * J + tid];
                    if (v < m2) { m2 = v; am2 = pa[q * J + tid]; }
                }
                mj[tid] = m2;
                bpr[(size_t)t * J + tid] = (uint8_t)am2;
            }
            __syncthreads();
        }
        const float xt = xs[t], pt = ps[t];
#pragma unroll
        for (int jj = 0; jj < NJ; ++jj)
#pragma unroll
            for (int aa = 0; aa < NA; ++aa) {
                const int i = jj * NA + aa, s = ((abase + aa) << (L - K)) | (jbase + jj * NT);
                c[i] = upd_of(mj[s >> K], inc_of(vsq[i], a2[i], xt, 2), pt, 2);
            }
    }
    float bv = INFINITY;
    int bi = 0x7fffffff;
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj)
#pragma unroll
        for (int aa = 0; aa < NA; ++aa) {
            const int i = jj * NA + aa, s = ((abase + aa) << (L - K)) | (jbase + jj * NT);
            if (c[i] < bv || (c[i] == bv && s < bi)) { bv = c[i]; bi = s; }
        }
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffff, bv, o);
        const int oi = __shfl_down_sync(0xffffffff, bi, o);
        if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const int lane = tid & 31, warp = tid >> 5, nw = NT >> 5;
    if (lane == 0) { rv[warp] = bv; ri[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < nw; ++w)
            if (rv[w] < bv || (rv[w] == bv && ri[w] < bi)) { bv = rv[w]; bi = ri[w]; }
        if (bi == 0x7fffffff) bi = 0;
        int path[64];
        int s = bi;
        path[T - 1] = s;
        for (int t = T - 1; t > 0; --t) {
            const int j = s >> K;
            s = ((int)bpr[(size_t)t * J + j] << (L - K)) | j;
            path[t - 1] = s;
        }
        for (int i = 0; i < nd; ++i) out[(size_t)r * ldo + i] = path[i];
    }
}

template <int K, bool BPS>
static int launch4(const float* X, int ldx, const float* sig, const float* cbp, const float* piv, const long long* state,
                   int has_state, int R, int T, int nd, uint8_t* bp, long long* out, int ldo, cudaStream_t st) {
    constexpr int NT = 1024, S = 1 << 14, J = S >> K;
    constexpr bool SPLIT = J < NT;
    const size_t smem = (size_t)2 * J * 4 + (SPLIT ? (size_t)2 * NT * 5 : 0) + (BPS ? (size_t)T * J : 0);
    cudaError_t e = cudaFuncSetAttribute(vit4<K, BPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return (int)e;
    vit4<K, BPS><<<R, NT, smem, st>>>(X, ldx, sig, cbp, piv, state, has_state, T, nd, bp, out, ldo);
    return (int)cudaGetLastError();
}

EXPORT int msc_vit4(const float* X, int ldx, const float* sig, const float* cbp, const float* piv, const long long* state,
                    int has_state, int R, int k, int T, int nd, uint8_t* bp, long long* out, int ldo, int mode, void* stream) {
    if (mode != 2 || T > 64 || T < 1 || nd > T) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    const bool b = T <= 32;                    // back-pointers in shared memory when T * J fits (k >= 3)
    switch (k) {
        case 2: return launch4<2, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 3: return b ? launch4<3, true>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s)
                         : launch4<3, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 4: return b ? launch4<4, true>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s)
                         : launch4<4, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 5: return b ? launch4<5, true>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s)
                         : launch4<5, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 6: return b ? launch4<6, true>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s)
                         : launch4<6, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
    }
    return -1;
}

template <int K, bool BPS>
static int launch3(const float* X, int ldx, const float* sig, const float* cbp, const float* piv, const long long* state,
                   int has_state, int R, int T, int nd, uint8_t* bp, long long* out, int ldo, cudaStream_t st) {
    constexpr int NT = 1024, S = 1 << 14, J = S >> K;
    constexpr bool SPLIT = J < NT;
    const size_t smem = (size_t)S * 4 + (SPLIT ? (size_t)NT * 4 + NT + (size_t)J * 4 : (size_t)J * 4) + (BPS ? (size_t)T * J : 0);
    cudaError_t e = cudaFuncSetAttribute(vit3<NT, K, 2, BPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return (int)e;
    vit3<NT, K, 2, BPS><<<R, NT, smem, st>>>(X, ldx, sig, cbp, piv, state, has_state, T, nd, bp, out, ldo);
    return (int)cudaGetLastError();
}

EXPORT int msc_vit3(const float* X, int ldx, const float* sig, const float* cbp, const float* piv, const long long* state,
                    int has_state, int R, int k, int T, int nd, uint8_t* bp, long long* out, int ldo, int mode, void* stream) {
    if (mode != 2 || T > 64 || T < 1 || nd > T) return -2;      // other rounding modes: msc_vit2
    cudaStream_t s = (cudaStream_t)stream;
    const bool bps = T <= 32;                                  // k >= 5: T * J <= 16 KB fits next to the 64 KB of costs
    switch (k) {
        case 1: return launch3<1, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 2: return launch3<2, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 3: return launch3<3, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 4: return launch3<4, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 5: return bps && false ? launch3<5, true>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s)
                           : launch3<5, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 6: return bps ? launch3<6, true>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s)
                           : launch3<6, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
        case 7: return bps ? launch3<7, true>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s)
                           : launch3<7, false>(X, ldx, sig, cbp, piv, state, has_state, R, T, nd, bp, out, ldo, s);
    }
    return -1;
}

// Commit nd columns of a window: q = code(s) * sig, e_c with the in-window updates of the earlier committed columns, in order.
// E is column-major (C x R): E[c * R + r]. Columns c0.. of W must hold all updates from columns < c0.
__global__ void vit_commit(const float* __restrict__ W, int ldw, float* __restrict__ E, int R, const float* __restrict__ Hu, int ldh,
                           const long long* __restrict__ dec, int nd, int c0, const float* __restrict__ sig,
                           const float* __restrict__ cbp, float* __restrict__ Q, int ldq, int16_t* __restrict__ states,
                           int lds, long long* __restrict__ state_out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    const float sg = sig[r];
    float e[64];
    for (int i = 0; i < nd; ++i) {
        const long long s = dec[(size_t)r * nd + i];
        const float q = __fmul_rn(code_val(cbp, (unsigned)s), sg);
        float w = W[(size_t)r * ldw + c0 + i];
        for (int m = 0; m < i; ++m) w = __fsub_rn(w, __fmul_rn(e[m], Hu[(size_t)(c0 + m) * ldh + c0 + i]));
        e[i] = __fdiv_rn(__fsub_rn(w, q), Hu[(size_t)(c0 + i) * ldh + c0 + i]);
        Q[(size_t)r * ldq + c0 + i] = q;
        states[(size_t)r * lds + c0 + i] = (int16_t)s;
        E[(size_t)(c0 + i) * R + r] = e[i];
    }
    state_out[r] = dec[(size_t)r * nd + nd - 1];
}

// W[:, j0:j1] -= E[c_a:c_b] (x) Hu[c_a:c_b, j0:j1], applied as fl(w - fl(e * h)) for c = c_a, c_a + 1, ... in order.
// 64 x 64 output tile per block, 256 threads with 4 x 4 elements each, E/Hu staged 16 columns at a time.
__global__ void __launch_bounds__(256)
vit_apply(float* __restrict__ W, int ldw, int R, int j0, int j1, const float* __restrict__ E, const float* __restrict__ Hu,
          int ldh, int c_a, int c_b) {
    __shared__ __align__(16) float Es[16][64];
    __shared__ __align__(16) float Hs[16][64];
    const int tr = threadIdx.x >> 4, tc = threadIdx.x & 15;
    const int rb = blockIdx.y * 64, jb = j0 + blockIdx.x * 64;
    float w[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int rr = rb + tr * 4 + i, jj = jb + tc * 4 + q;
            w[i][q] = (rr < R && jj < j1) ? W[(size_t)rr * ldw + jj] : 0.0f;
        }
    for (int cc = c_a; cc < c_b; cc += 16) {
        const int nc = min(16, c_b - cc);
        for (int idx = threadIdx.x; idx < 16 * 64; idx += 256) {
            const int c = idx >> 6, xx = idx & 63;
            Es[c][xx] = (c < nc && rb + xx < R) ? E[(size_t)(cc + c) * R + rb + xx] : 0.0f;
            Hs[c][xx] = (c < nc && jb + xx < j1) ? Hu[(size_t)(cc + c) * ldh + jb + xx] : 0.0f;
        }
        __syncthreads();
        for (int c = 0; c < nc; ++c) {
            const float4 ev = *reinterpret_cast<const float4*>(&Es[c][tr * 4]);
            const float4 hv = *reinterpret_cast<const float4*>(&Hs[c][tc * 4]);
            const float ea[4] = {ev.x, ev.y, ev.z, ev.w}, ha[4] = {hv.x, hv.y, hv.z, hv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int q = 0; q < 4; ++q) w[i][q] = __fsub_rn(w[i][q], __fmul_rn(ea[i], ha[q]));
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int rr = rb + tr * 4 + i, jj = jb + tc * 4 + q;
            if (rr < R && jj < j1) W[(size_t)rr * ldw + jj] = w[i][q];
        }
}

EXPORT int msc_vit2(const float* X, int ldx, const float* sig, const float* cbp, const float* piv, const long long* state,
                    int has_state, int R, int k, int T, int nd, uint8_t* bp, long long* out, int ldo, int mode, void* stream) {
    constexpr int NT = 1024, S = 1 << 14;
    if (T > 64 || T < 1 || k < 1 || k > 7 || nd > T) return -1;
    const int J = S >> k;
    const size_t smem = (J >= NT) ? (size_t)(S + J) * sizeof(float)
                                  : (size_t)(S + NT) * sizeof(float) + NT + (size_t)J * sizeof(float);
    cudaError_t e = cudaFuncSetAttribute(vit2<NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return (int)e;
    vit2<NT><<<R, NT, smem, (cudaStream_t)stream>>>(X, ldx, sig, cbp, piv, state, has_state, k, T, nd, bp, out, ldo, mode);
    return (int)cudaGetLastError();
}

EXPORT int msc_commit(const float* W, int ldw, float* E, int R, const float* Hu, int ldh, const long long* dec, int nd, int c0,
                      const float* sig, const float* cbp, float* Q, int ldq, int16_t* states, int lds, long long* state_out,
                      void* stream) {
    if (nd > 64 || nd < 1) return -1;
    vit_commit<<<(R + 127) / 128, 128, 0, (cudaStream_t)stream>>>(W, ldw, E, R, Hu, ldh, dec, nd, c0, sig, cbp, Q, ldq, states,
                                                                  lds, state_out);
    return (int)cudaGetLastError();
}

EXPORT int msc_apply(float* W, int ldw, int R, int j0, int j1, const float* E, const float* Hu, int ldh, int c_a, int c_b,
                     void* stream) {
    if (j1 <= j0 || c_b <= c_a || R <= 0) return 0;
    dim3 grid((j1 - j0 + 63) / 64, (R + 63) / 64);
    vit_apply<<<grid, 256, 0, (cudaStream_t)stream>>>(W, ldw, R, j0, j1, E, Hu, ldh, c_a, c_b);
    return (int)cudaGetLastError();
}
