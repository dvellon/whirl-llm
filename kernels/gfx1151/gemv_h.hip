// WHIRL device code (gfx1151 / RDNA 3.5): precise-mode decode / verify matmul (gemvh*).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included after gemm_small.hip by kernels/gfx1151/whirl_kernels_gfx1151.hip.
//
// gemvh_<type> (n_tok <= 16) / gemvh2_<type> (n_tok <= 32): y[t][r] (+)= sum_k W[r][k] x16[t][k]
// Same ABI and numerics as the gfx1201 kernel (kernels/gemv_h.hip) and gemms11_impl: weights are
// dec_group_h<T> values (16 consecutive k per WMMA step, row lane % 16), the f16 activations are the
// other operand (WMMA16(b = tokens, a = weights)), one WMMA per 16-k step in increasing k from zero,
// tokens past n_tok are never stored -> bitwise == gemms_c* == gemm_c*.
//
// Two implementations:
//  * "direct" (q4_k q5_k q6_k q8_0 iq4_xs mxfp4): one wave = 16 weight rows, no LDS and no barriers. Every
//    lane loads its row's 16-byte pieces straight from global memory (lanes l and l + 16 read the same bytes),
//    a ring of superblocks (GhW<T>::DP) stays in registers so the next loads are in flight while the current
//    one is decoded, and each lane decodes the full 16-k fragment itself (no cross-lane join). The decoders
//    use the same arithmetic as dec_group_h<T> (identical bits), with the byte -> half spreading done by
//    v_perm_b32. The generic 2-aligned 16-byte loads rely on unaligned global access (as the int8 Q6_K GEMV does).
//  * "staged" (f16 q3_k iq4_nl iq3_s): the original design, the raw stages go through LDS (below).

// ---- direct-load implementation ----
__device__ __forceinline__ gs_u32x4 gh_ld16a2(const uint8_t *p) {
    gs_u32x4 v;
    __builtin_memcpy(&v, __builtin_assume_aligned(p, 2), 16);
    return v;
}
__device__ __forceinline__ uint32_t gh_ld16u(const uint8_t *p) { return *(const uint16_t *)p; }
__device__ __forceinline__ uint32_t gh_ld32a4(const uint8_t *p) { return *(const uint32_t *)p; }
__device__ __forceinline__ half16 gh_join(half8 a, half8 b) { return __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); }
// scale / min of K-quant sub-block s from the 12 packed scale bytes (3 dwords)
__device__ __forceinline__ void gh_scale_min(const uint32_t *scw, int s, int &sc, int &m) {
    auto by = [&](int idx) { return (scw[idx >> 2] >> (8 * (idx & 3))) & 0xFFu; };
    if (s < 4) {
        sc = by(s) & 63;
        m = by(s + 4) & 63;
    } else {
        sc = (by(s + 4) & 0xF) | ((by(s - 4) >> 6) << 4);
        m = (by(s + 4) >> 4) | ((by(s) >> 6) << 4);
    }
}
__device__ __forceinline__ float gh_h2f(uint32_t u) { return __half2float(__ushort_as_half((uint16_t)u)); }
// cheaper table select than iq4_lut4/mx_lut4: the bit-3 blend mask comes from a v_perm of a constant instead of a multiply
__device__ __forceinline__ uint32_t gh_lut4(uint32_t nib4, uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
    const uint32_t sel = nib4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = __builtin_amdgcn_perm(0u, 0x0000FF00u, (nib4 >> 3) & 0x01010101u);
    return (hi & m) | (lo & ~m);
}
// bytes8_to_h8 with the byte -> (1024 + byte) half spreading done by one v_perm per half2
__device__ __forceinline__ half8 gh_b8h(uint32_t u0, uint32_t u1, _Float16 bias, _Float16 scale, _Float16 add) {
    const uint32_t m = 0x64646464u;
    const uint32_t pw[4] = {__builtin_amdgcn_perm(m, u0, 0x04010400u), __builtin_amdgcn_perm(m, u0, 0x04030402u), __builtin_amdgcn_perm(m, u1, 0x04010400u),
                            __builtin_amdgcn_perm(m, u1, 0x04030402u)};
    const half2_t b2 = {(_Float16)1024.f + bias, (_Float16)1024.f + bias};
    const half2_t s2 = {scale, scale}, a2 = {add, add};
    half8 out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const half2_t v = (__builtin_bit_cast(half2_t, pw[i]) - b2) * s2 + a2;
        out[2 * i] = v[0];
        out[2 * i + 1] = v[1];
    }
    return out;
}
// exact (byte - bias) as halves: perm-spread (1024 + byte), one packed subtract
__device__ __forceinline__ void gh_x4(uint32_t u, int bias, _Float16 *o) {
    const uint32_t m = 0x64646464u;
    const half2_t b2 = {(_Float16)(1024 + bias), (_Float16)(1024 + bias)};
    const half2_t v0 = __builtin_bit_cast(half2_t, __builtin_amdgcn_perm(m, u, 0x04010400u)) - b2;
    const half2_t v1 = __builtin_bit_cast(half2_t, __builtin_amdgcn_perm(m, u, 0x04030402u)) - b2;
    o[0] = v0[0]; o[1] = v0[1]; o[2] = v1[0]; o[3] = v1[1];
}
template <int T> struct GhW;
// ---- Q4_K / Q5_K (256-k superblock, 16 steps) ----

template <> struct GhW<QT_Q4_K> {
    static constexpr int UK = 16, UB = 144, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8]; uint32_t dm; uint32_t sc[3]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
        r.dm = gh_ld32a4(b);
        r.sc[0] = gh_ld32a4(b + 4); r.sc[1] = gh_ld32a4(b + 8); r.sc[2] = gh_ld32a4(b + 12);
#pragma unroll
        for (int i = 0; i < 8; ++i) r.q[i] = *(const gs_u32x4 *)(b + 16 + 16 * i);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int s = kk >> 1, c = kk & 1, hi = s & 1, j64 = s >> 1;
        const float d = gh_h2f(r.dm), dmin = gh_h2f(r.dm >> 16);
        int sc, m;
        gh_scale_min(r.sc, s, sc, m);
        const gs_u32x4 q = r.q[j64 * 2 + c];
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = hi ? (q[i] >> 4) & 0x0F0F0F0F : q[i] & 0x0F0F0F0F;
        const _Float16 hs = (_Float16)(d * sc), ha = (_Float16)(-dmin * m);
        return gh_join(gh_b8h(w[0], w[1], (_Float16)0.f, hs, ha), gh_b8h(w[2], w[3], (_Float16)0.f, hs, ha));
    }
};
template <> struct GhW<QT_Q5_K> {
    static constexpr int UK = 16, UB = 176, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8]; gs_u32x4 qh[2]; uint32_t dm; uint32_t sc[3]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
        r.dm = gh_ld32a4(b);
        r.sc[0] = gh_ld32a4(b + 4); r.sc[1] = gh_ld32a4(b + 8); r.sc[2] = gh_ld32a4(b + 12);
        r.qh[0] = *(const gs_u32x4 *)(b + 16); r.qh[1] = *(const gs_u32x4 *)(b + 32);
#pragma unroll
        for (int i = 0; i < 8; ++i) r.q[i] = *(const gs_u32x4 *)(b + 48 + 16 * i);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int s = kk >> 1, c = kk & 1, hi = s & 1, j64 = s >> 1;
        const float d = gh_h2f(r.dm), dmin = gh_h2f(r.dm >> 16);
        int sc, m;
        gh_scale_min(r.sc, s, sc, m);
        const gs_u32x4 q = r.q[j64 * 2 + c], h = r.qh[c];
        uint32_t w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = (hi ? (q[i] >> 4) & 0x0F0F0F0F : q[i] & 0x0F0F0F0F) | (((h[i] >> s) & 0x01010101) << 4);
        const _Float16 hs = (_Float16)(d * sc), ha = (_Float16)(-dmin * m);
        return gh_join(gh_b8h(w[0], w[1], (_Float16)0.f, hs, ha), gh_b8h(w[2], w[3], (_Float16)0.f, hs, ha));
    }
};
// ---- Q6_K (210-byte superblock, 2-byte aligned) ----
template <> struct GhW<QT_Q6_K> {
    static constexpr int UK = 16, UB = 210, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 ql[8]; gs_u32x4 qh[4]; gs_u32x4 sc; uint32_t d; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
#pragma unroll
        for (int i = 0; i < 8; ++i) r.ql[i] = gh_ld16a2(b + 16 * i);
#pragma unroll
        for (int i = 0; i < 4; ++i) r.qh[i] = gh_ld16a2(b + 128 + 16 * i);
        r.sc = gh_ld16a2(b + 192);
        r.d = gh_ld16u(b + 208);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int n = kk >> 3, quarter = (kk >> 1) & 3, c = kk & 1;
        const float d = gh_h2f(r.d);
        const int sidx = 8 * n + c + 2 * quarter;
        const float ds = d * (float)(int8_t)((r.sc[sidx >> 2] >> (8 * (sidx & 3))) & 0xFF);
        const gs_u32x4 ql = r.ql[4 * n + 2 * (quarter & 1) + c], qh = r.qh[2 * n + c];
        const int lsh = (quarter >> 1) ? 4 : 0, hsh = 2 * quarter;
        half16 out;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const uint32_t w = ((ql[wi] >> lsh) & 0x0F0F0F0Fu) | (((qh[wi] >> hsh) & 0x03030303u) << 4);
            _Float16 x[4];
            gh_x4(w, 32, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) out[4 * wi + i] = (_Float16)(ds * (float)x[i]);
        }
        return out;
    }
};
// ---- Q8_0 (34-byte blocks; unit = 4 blocks = 8 steps) ----
template <> struct GhW<QT_Q8_0> {
    static constexpr int UK = 4, UB = 17 * 4, DP = 3, SBE = 2;
    struct Regs { gs_u32x4 q[4]; uint32_t d[4 / 2]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
#pragma unroll
        for (int i = 0; i < 4 / 2; ++i) {
            r.d[i] = gh_ld16u(b + 34 * i);
            r.q[2 * i] = gh_ld16a2(b + 34 * i + 2);
            r.q[2 * i + 1] = gh_ld16a2(b + 34 * i + 18);
        }
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const float d = gh_h2f(r.d[kk >> 1]);
        const gs_u32x4 q = r.q[kk];
        half16 out;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            _Float16 x[4];
            gh_x4(q[wi] ^ 0x80808080u, 128, x);
#pragma unroll
            for (int i = 0; i < 4; ++i) out[4 * wi + i] = (_Float16)(d * (float)x[i]);
        }
        return out;
    }
};
// ---- MXFP4 / IQ4_XS (136-byte superblock, 8-aligned) ----
template <> struct GhW<QT_MXFP4> {
    static constexpr int UK = 16, UB = 136, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8]; uint32_t sc[2]; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
        r.sc[0] = gh_ld32a4(b); r.sc[1] = gh_ld32a4(b + 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) r.q[i] = gh_ld16a2(b + 8 + 16 * i);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int ib = kk >> 1, c = kk & 1;
#if 0
        return __builtin_bit_cast(half16, (uint32_t __attribute__((ext_vector_type(8)))){r.q[ib][0], r.q[ib][1], r.q[ib][2], r.q[ib][3], r.sc[0], r.sc[1], r.q[ib][0], r.q[ib][1]});
#endif
        const float dl = e8m0_half((r.sc[ib >> 2] >> (8 * (ib & 3))) & 0xFF);
        const gs_u32x4 q = r.q[ib];
        uint32_t u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = gh_lut4(c ? (q[i] >> 4) & 0x0F0F0F0F : q[i] & 0x0F0F0F0F, 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu) ^ 0x80808080u;
        return gh_join(gh_b8h(u[0], u[1], (_Float16)128.f, (_Float16)dl, (_Float16)0.f), gh_b8h(u[2], u[3], (_Float16)128.f, (_Float16)dl, (_Float16)0.f));
    }
};
template <> struct GhW<QT_IQ4_XS> {
    static constexpr int UK = 16, UB = 136, DP = 2, SBE = 4;
    struct Regs { gs_u32x4 q[8]; uint32_t hdr, sl; };
    static __device__ __forceinline__ void load(Regs &r, const uint8_t *rowp, int u) {
        const uint8_t *b = rowp + (size_t)u * UB;
        r.hdr = gh_ld32a4(b); r.sl = gh_ld32a4(b + 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) r.q[i] = gh_ld16a2(b + 8 + 16 * i);
    }
    static __device__ __forceinline__ half16 frag(const Regs &r, int kk) {
        const int ib = kk >> 1, c = kk & 1;
        const float d = gh_h2f(r.hdr);
        const int ls = ((r.sl >> (8 * (ib >> 1)) >> (4 * (ib & 1))) & 0xF) | ((((r.hdr >> 16) >> (2 * ib)) & 3) << 4);
        const gs_u32x4 q = r.q[ib];
        uint32_t u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = gh_lut4(c ? (q[i] >> 4) & 0x0F0F0F0F : q[i] & 0x0F0F0F0F, 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u) ^ 0x80808080u;
        const _Float16 hs = (_Float16)(d * (ls - 32));
        return gh_join(gh_b8h(u[0], u[1], (_Float16)128.f, hs, (_Float16)0.f), gh_b8h(u[2], u[3], (_Float16)128.f, hs, (_Float16)0.f));
    }
};


template <int T, int TN>
__device__ __forceinline__ void gemvh_direct_impl(const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X, float *__restrict__ Y,
                                           int ncols, int nrows, int n_tok, int accumulate) {
    using D = GhW<T>;
    using Regs = typename D::Regs;
    constexpr int DEPTH = D::DP, SB_EVERY = D::SBE;
    const int lane = threadIdx.x % WARP;
    const int m0 = blockIdx.x * 16;
    if (m0 >= nrows) return;
    const int nsb = ncols / (16 * D::UK);
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    const uint8_t *rowp = W + (size_t)row * row_bytes;
    float8 acc[TN];
    const _Float16 *xr[TN];
    bool xv[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xv[j] = t < n_tok;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    Regs buf[DEPTH];
#pragma unroll
    for (int u = 0; u < DEPTH - 1; ++u)
        if (u < nsb) D::load(buf[u], rowp, u);
    for (int sb0 = 0; sb0 < nsb; sb0 += DEPTH) {
#pragma unroll
        for (int u = 0; u < DEPTH; ++u) {
            const int sb = sb0 + u;
            if (sb < nsb) {
                if (sb + DEPTH - 1 < nsb) D::load(buf[(u + DEPTH - 1) % DEPTH], rowp, sb + DEPTH - 1);
#pragma unroll
                for (int kk = 0; kk < D::UK; ++kk) {
                    if (kk % SB_EVERY == 0) __builtin_amdgcn_sched_barrier(0);
                    const half16 a = D::frag(buf[u], kk);
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        // (tokens past n_tok read a valid row; their output columns are never stored, columns are independent)
                        const half16 b = ld_frag16(xr[j] + sb * (16 * D::UK) + kk * 16);
                        acc[j] = WMMA16(b, a, acc[j]);
                    }
                }
            }
        }
    }
    const int orow = m0 + rl;
    if (orow >= nrows) return;
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = j * 16 + CROW(lane, v);
            if (t < n_tok) {
                float *dst = Y + (size_t)t * nrows + orow;
                *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
            }
        }
}

#define GHD_TYPE(SUF, T)                                                                                                                           extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols,                                                                   int nrows, int n_tok, int accumulate) {                                               gemvh_direct_impl<T, 1>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                              }                                                                                                                                              extern "C" __global__ void __launch_bounds__(32) gemvh2_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y, int ncols,                                                                   int nrows, int n_tok, int accumulate) {                                              gemvh_direct_impl<T, 2>(W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);                                                              }
GHD_TYPE(q8_0, QT_Q8_0)
GHD_TYPE(q4_k, QT_Q4_K)
GHD_TYPE(q5_k, QT_Q5_K)
GHD_TYPE(q6_k, QT_Q6_K)
GHD_TYPE(iq4_xs, QT_IQ4_XS)
GHD_TYPE(mxfp4, QT_MXFP4)

// ---- staged (LDS) implementation ----
template <int T, int TN, int S>
__device__ __forceinline__ void gemvh11_impl(uint8_t *__restrict__ Ws, const uint8_t *__restrict__ W, uint64_t row_bytes, const _Float16 *__restrict__ X,
                                             float *__restrict__ Y, int ncols, int nrows, int n_tok, int accumulate) {
    constexpr int SB = S * gs_sbb<T>();
    constexpr int CH = (SB + 15) / 16 + 1;  // 16-byte chunks per row stage (covers the misaligned start)
    constexpr int SLOT = CH * 16;
    constexpr int WCH = 16 * CH;
    constexpr int WPER = (WCH + WARP - 1) / WARP;
    const int lane = threadIdx.x % WARP;
    const int m0 = blockIdx.x * 16;
    if (m0 >= nrows) return;
    const int nst = ncols / (256 * S);
    const int rl = lane % 16;
    const int row = min(m0 + rl, nrows - 1);
    float8 acc[TN];
    const _Float16 *xr[TN];
    bool xv[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        acc[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
        const int t = j * 16 + rl;
        xv[j] = t < n_tok;
        xr[j] = X + (size_t)min(t, n_tok - 1) * ncols;
    }
    gs_u32x4 rw[WPER];
#pragma unroll
    for (int e = 0; e < WPER; ++e) rw[e] = (gs_u32x4){0, 0, 0, 0};
    // (macro, not a lambda: a by-reference capture of rw pushes it to scratch)
#define GEMVH11_FETCH(js_)                                                                                   \
    do {                                                                                                     \
        _Pragma("unroll") for (int e = 0; e < WPER; ++e) {                                                   \
            const int idx = lane + e * WARP;                                                                 \
            if (WCH % WARP == 0 || idx < WCH) {                                                              \
                const size_t st = (size_t)min(m0 + idx / CH, nrows - 1) * row_bytes + (size_t)(js_) * SB;                                             \
                const size_t a = (st & ~(size_t)15) + (idx % CH) * 16;                                       \
                /* no 16-byte chunk wholly past the stage (the matrix may end there) */                      \
                if (a < st + SB) rw[e] = *(const gs_u32x4 *)(W + a);                                         \
            }                                                                                                \
        }                                                                                                    \
    } while (0)
    GEMVH11_FETCH(0);
    for (int js = 0; js < nst; ++js) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < WPER; ++e) {
            const int idx = lane + e * WARP;
            if (WCH % WARP == 0 || idx < WCH) *(gs_u32x4 *)&Ws[(idx / CH) * SLOT + (idx % CH) * 16] = rw[e];
        }
        __syncthreads();
        if (js + 1 < nst) GEMVH11_FETCH(js + 1);
        const uint8_t *wp = Ws + rl * SLOT + (int)(((size_t)row * row_bytes + (size_t)js * SB) & 15);
        // (partial unroll: a full unroll lets the scheduler hoist every step's decode and X loads and spill)
#pragma unroll 2
        for (int kk = 0; kk < 16 * S; ++kk) {
            const half16 a = join_halves(dec_group_h<T>(wp, 2 * kk + lane / 16), lane);
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                half16 b = ld_frag16(xr[j] + js * 256 * S + kk * 16);
                if (!xv[j]) b = (half16){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                acc[j] = WMMA16(b, a, acc[j]);
            }
        }
    }
#undef GEMVH11_FETCH
    const int orow = m0 + rl;
    if (orow >= nrows) return;
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int t = j * 16 + CROW(lane, v);
            if (t < n_tok) {
                float *dst = Y + (size_t)t * nrows + orow;
                *dst = accumulate ? *dst + acc[j][v] : acc[j][v];
            }
        }
}

// S = 2 (two 256-k stages per LDS round) when ncols % 512 == 0 and the type's stage is small; one LDS buffer
// per kernel, sized for the S used.
template <int T> constexpr int gv11_s2() { return gs_sbb<T>() <= 288 ? 2 : 1; }
#define GEMVH11_BODY(T, TN)                                                                                      constexpr int S2 = gv11_s2<T>();                                                                             __shared__ __attribute__((aligned(16))) uint8_t Ws[16 * 16 * ((S2 * gs_sbb<T>() + 15) / 16 + 1)];            if (S2 == 2 && ncols % 512 == 0) gemvh11_impl<T, TN, S2>(Ws, W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);     else gemvh11_impl<T, TN, 1>(Ws, W, row_bytes, X, Y, ncols, nrows, n_tok, accumulate);
#define GEMVH11_TYPE(SUF, T)                                                                                                        extern "C" __global__ void __launch_bounds__(32) gemvh_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,                                                                    int ncols, int nrows, int n_tok, int accumulate) {                   GEMVH11_BODY(T, 1)                                                                                                          }                                                                                                                               extern "C" __global__ void __launch_bounds__(32) gemvh2_##SUF(const uint8_t *W, uint64_t row_bytes, const _Float16 *X, float *Y,                                                                     int ncols, int nrows, int n_tok, int accumulate) {                  GEMVH11_BODY(T, 2)                                                                                                          }

GEMVH11_TYPE(f16, QT_F16)
GEMVH11_TYPE(q3_k, QT_Q3_K)
GEMVH11_TYPE(iq4_nl, QT_IQ4_NL)
GEMVH11_TYPE(iq3_s, QT_IQ3_S)
