// WHIRL device code (gfx1151 / RDNA 3.5): attn_ws4, 4-wave pipelined split-K decode attention.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included by kernels/gfx1151/whirl_kernels_gfx1151.hip after attn.hip
// (KvArgs, kv_dq8 / kv_dq4, AwGroups, aw_qk_chain / aw_softmax_step / aw_pv).
// Design: audit/DES-ATTN-8S.md (sections 3.4-3.7, T1 F4 / F7, T0). The LDS-only barrier idea
// (s_barrier fenced on the local address space only, so in-flight global loads are not drained
// and the vector L0 is not invalidated) comes from asf0/strix-llama.cpp fattn-rdna35.cu
// (MIT, Copyright (c) 2023-2026 The ggml authors); the code here is written from scratch.
//
// attn_ws4<F, PD> (F: 0 f16, 1 int8 (q8 / q8h), 2 int4 (q4); PD: chunks prefetched in
// registers). Same arguments, grid (kv head, split, group), 128 threads and partial layout as
// attn_wsplit1[_q8|_q4], and bitwise the same partials for groups of one query (the gfx1151
// host's only kind, caps.attn_group1). Per 32-key chunk:
//   LB0  Ks / Vt of the chunk ready
//   QK   wave kt (0, 1): aw_qk_chain over its 16 keys from Ks -> Sc (f32, C-fragment order)
//   LB1  every wave: aw_softmax_step on both tiles (identical bits in all waves), aw_pv on its
//        own 64 dims (dt 4 w .. 4 w + 3) from Vt
//   LB2  stage the next chunk's raw K / V bytes (registers, issued PD chunks ahead) into
//        Ks / Vt (in-register dequant: kv_dq8 / kv_dq4, the same rounding as kv_ld16 / kv_ld8),
//        then issue the raw loads of chunk + 1 + PD into the freed registers
// All K / V bytes are read once per block (keys >= b1 of the last chunk: K reads row b1 - 1
// again, the clamp of attn_wsplit_impl; V is zero there), loaded by all 128 threads.
// The split rule uses the constant ws4::kNs (64), not gridDim.y: the splits depend on n_pos
// only (the same ranges as attn_wsplit1 whenever n_pos <= 64 * gridDim.y, i.e. always).
// ---------------------------------------------------------------------------
#include "attn_ws4_index.h"

__device__ __forceinline__ void ws4_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// scales as f16 vectors (constant lane indices after unrolling; a _Float16 array was promoted
// to LDS by the compiler)
typedef _Float16 ws4_h4 __attribute__((ext_vector_type(4)));
template <int F> struct Ws4Raw;
template <> struct Ws4Raw<0> {  // f16: K 8 x 8 values, V 8 keys x 8 values
    uint4 k[8];
    uint4 v[8];
};
template <> struct Ws4Raw<1> {  // int8: K 4 x 16 values + scale, V 8 keys x 8 values + scale
    uint4 k[4];
    uint2 v[8];
    ws4_h4 ks;
    half8 vs;
};
template <> struct Ws4Raw<2> {  // int4: K 4 x 16 nibbles + scale, V 8 keys x 8 nibbles + scale
    uint2 k[4];
    uint32_t v[8];
    ws4_h4 ks;
    half8 vs;
};

// issue the raw loads of chunk k0 (keys < b1) of kv head g for thread tv
template <int F>
__device__ __forceinline__ void ws4_load(Ws4Raw<F> &r, const KvArgs &kv, int tb, int k0, int b1, int g, int n_kv, int tv) {
    const int dl = kv_delta(kv, tb, k0);  // 32-key chunks never straddle a page
#pragma unroll
    for (int p = 0; p < ws4::k_pieces(F); ++p) {
        const int key = min(k0 + ws4::k_key(F, tv, p), b1 - 1);  // clamp (attn_wsplit_impl krow)
        const size_t e = ((size_t)(key + dl) * n_kv + g) * AW_D + ws4::k_d0(F, tv);
        if constexpr (F == 0) {
            r.k[p] = *(const uint4 *)((const _Float16 *)kv.k + e);
        } else if constexpr (F == 1) {
            r.k[p] = *(const uint4 *)((const int8_t *)kv.k + e);
            r.ks[p] = kv.ks[e >> 5];
        } else {
            r.k[p] = *(const uint2 *)((const uint8_t *)kv.k + (e >> 1));
            r.ks[p] = kv.ks[e >> 5];
        }
    }
    const int kb = ws4::v_kb(tv), d8 = ws4::v_d8(tv);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int key = k0 + kb * 8 + i;
        const size_t e = ((size_t)(key + dl) * n_kv + g) * AW_D + d8;
        if (key < b1) {
            if constexpr (F == 0) {
                r.v[i] = *(const uint4 *)((const _Float16 *)kv.v + e);
            } else if constexpr (F == 1) {
                r.v[i] = *(const uint2 *)((const int8_t *)kv.v + e);
                r.vs[i] = kv.vs[e >> 5];
            } else {
                r.v[i] = *(const uint32_t *)((const uint8_t *)kv.v + (e >> 1));
                r.vs[i] = kv.vs[e >> 5];
            }
        } else {
            if constexpr (F == 0) {
                r.v[i] = make_uint4(0, 0, 0, 0);
            } else if constexpr (F == 1) {
                r.v[i] = make_uint2(0, 0);
                r.vs[i] = (_Float16)0.f;
            } else {
                r.v[i] = 0;
                r.vs[i] = (_Float16)0.f;
            }
        }
    }
}

// dequantize / move the raw chunk k0 into Ks [32][kQs] and Vt (attn_wsplit_impl key permutation)
template <int F>
__device__ __forceinline__ void ws4_stage(const Ws4Raw<F> &r, _Float16 (*Ks)[ws4::kQs], _Float16 *Vt, int k0, int b1, int tv) {
#pragma clang fp contract(off)
#pragma unroll
    for (int p = 0; p < ws4::k_pieces(F); ++p) {
        _Float16 *dst = &Ks[ws4::k_key(F, tv, p)][ws4::k_d0(F, tv)];
        if constexpr (F == 0) {
            *(half8 *)dst = __builtin_bit_cast(half8, r.k[p]);
        } else if constexpr (F == 1) {
            *(half8 *)dst = kv_dq8(__builtin_bit_cast(kvq8_t, make_uint2(r.k[p].x, r.k[p].y)), r.ks[p]);
            *(half8 *)(dst + 8) = kv_dq8(__builtin_bit_cast(kvq8_t, make_uint2(r.k[p].z, r.k[p].w)), r.ks[p]);
        } else {
            *(half8 *)dst = kv_dq4(r.k[p].x, r.ks[p]);
            *(half8 *)(dst + 8) = kv_dq4(r.k[p].y, r.ks[p]);
        }
    }
    const int kb = ws4::v_kb(tv), d8 = ws4::v_d8(tv);
    half8 rows[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (F == 0) {
            rows[i] = __builtin_bit_cast(half8, r.v[i]);
        } else {
            half8 x;
            if constexpr (F == 1)
                x = kv_dq8(__builtin_bit_cast(kvq8_t, r.v[i]), r.vs[i]);
            else
                x = kv_dq4(r.v[i], r.vs[i]);
            const bool valid = k0 + kb * 8 + i < b1;  // keys >= b1: +0 as attn_wsplit_impl
#pragma unroll
            for (int e2 = 0; e2 < 8; ++e2) rows[i][e2] = valid ? x[e2] : (_Float16)0.f;
        }
    }
    typedef _Float16 half4_t __attribute__((ext_vector_type(4)));
    const int tb = ws4::vt_tb(kb);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        half4_t ev, od;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ev[i] = rows[2 * i][c];
            od[i] = rows[2 * i + 1][c];
        }
        *(half4_t *)&Vt[ws4::vt_addr(d8 + c, tb)] = ev;
        *(half4_t *)&Vt[ws4::vt_addr(d8 + c, tb + 8)] = od;
    }
}

template <int F, int PD>
__device__ __forceinline__ void attn_ws4_impl(const float *q, const KvArgs &kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                                              int q_stride, const int *pos, float scale, const int *skip, const AwGroups &groups) {
#pragma clang fp contract(off)
    if (skip != nullptr && *skip) return;
    __shared__ __align__(16) _Float16 Qs[16][ws4::kQs];
    __shared__ __align__(16) _Float16 Ks[AW_K][ws4::kQs];
    __shared__ __align__(16) _Float16 Vt[ws4::kVtSize];
    __shared__ __align__(16) float Sc[ws4::kScSize];
    const int tv = threadIdx.x, lane = tv % WARP, wid = tv / WARP;
    const int g = blockIdx.x, split = blockIdx.y, n_split = gridDim.y, z = blockIdx.z;
    const int grp = n_head / n_kv;
    const int first = groups.first[z], count = groups.count[z];
    const int ncol = count * grp;  // <= 16 (one query per group on gfx1151)
    const int tb = kv_tb(kv, first);
    const int n_pos = pos[first + count - 1] + 1;
    const ws4::Range rg = ws4::block_range(n_pos, split, ws4::kNs, kv.win);
    const int b0 = rg.b0, b1 = rg.b1;
    const size_t ml_q = (size_t)n_split * n_head * 2, acc_q = (size_t)n_split * n_head * AW_D;
    const int col = lane % 16;
    const bool cvalid = col < ncol;
    const int colc = cvalid ? col : 0;
    const int t = first + colc / grp, h = g * grp + colc % grp;
    const int my_pos = cvalid ? pos[t] : -1;
    if (b0 >= n_pos) {
        if (wid == 0 && cvalid && lane < 16) {
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = -INFINITY;
            part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = 0.f;
        }
        return;
    }
    const int nch = (b1 - b0 + AW_K - 1) / AW_K;
    Ws4Raw<F> raw[PD];
    ws4_load<F>(raw[0], kv, tb, b0, b1, g, n_kv, tv);
    // Q^T (f16) of the group's columns, as attn_wsplit_impl
    for (int i = tv; i < 16 * (AW_D / 8); i += 128) {
        const int c = i / (AW_D / 8), d8 = (i % (AW_D / 8)) * 8;
        half8 v8;
        if (c < ncol) {
            const float *qr = q + ((size_t)(first + c / grp) * n_head + g * grp + c % grp) * q_stride + d8;
            const float4 a = *(const float4 *)qr, b = *(const float4 *)(qr + 4);
            v8 = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                         (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) v8[e] = (_Float16)0.f;
        }
        *(half8 *)&Qs[c][d8] = v8;
    }
    ws4_stage<F>(raw[0], Ks, Vt, b0, b1, tv);
#pragma unroll
    for (int j = 1; j <= PD; ++j)
        if (j < nch) ws4_load<F>(raw[j % PD], kv, tb, b0 + j * AW_K, b1, g, n_kv, tv);
    float8 o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = (float8){0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    for (int i0 = 0; i0 < nch; i0 += PD) {
#pragma unroll
        for (int s = 0; s < PD; ++s) {
            const int c = i0 + s;
            if (c < nch) {
                const int k0 = b0 + c * AW_K;
                ws4_lds_barrier();  // LB0: Ks / Vt (and Qs) of chunk c written
                if (wid < 2) {
                    const int kt = wid;
                    const float8 sc = aw_qk_chain((float8){0, 0, 0, 0, 0, 0, 0, 0}, &Qs[lane % 16][0],
                                                  [&](int st) { return ld_frag16(&Ks[kt * 16 + lane % 16][16 * st]); });
                    *(float4 *)&Sc[ws4::sc_addr(kt, lane, 0)] = (float4){sc[0], sc[1], sc[2], sc[3]};
                    *(float4 *)&Sc[ws4::sc_addr(kt, lane, 4)] = (float4){sc[4], sc[5], sc[6], sc[7]};
                }
                ws4_lds_barrier();  // LB1: Sc written, Ks free
                float8 sc[AW_K / 16];
#pragma unroll
                for (int kt = 0; kt < AW_K / 16; ++kt) {
                    const float4 a = *(const float4 *)&Sc[ws4::sc_addr(kt, lane, 0)];
                    const float4 b = *(const float4 *)&Sc[ws4::sc_addr(kt, lane, 4)];
                    sc[kt] = (float8){a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                }
                half8 pt[AW_K / 16];
                const float alpha = aw_softmax_step(sc, pt, k0, lane, my_pos, b1, scale, m, l);
                aw_pv<4>(o, alpha, pt, lane,
                         [&](int j, int kt) { return ld_frag16(&Vt[ws4::vt_addr((4 * wid + j) * 16 + lane % 16, kt * 16)]); });
                if (c + 1 < nch) {
                    ws4_lds_barrier();  // LB2: Vt, Sc read
                    ws4_stage<F>(raw[(s + 1) % PD], Ks, Vt, k0 + AW_K, b1, tv);
                    if (c + 1 + PD < nch) ws4_load<F>(raw[(s + 1) % PD], kv, tb, k0 + (1 + PD) * AW_K, b1, g, n_kv, tv);
                }
            }
        }
    }
    if (!cvalid) return;
    if (wid == 0 && lane < 16) {
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 0] = m;
        part_ml[t * ml_q + ((size_t)split * n_head + h) * 2 + 1] = l;
    }
    float *pa = part_acc + t * acc_q + ((size_t)split * n_head + h) * AW_D;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) pa[(4 * wid + j) * 16 + CROW(lane, v)] = o[j][v];
}

#define WS4_ENTRY(name, F, PD)                                                                                                        \
    extern "C" __global__ void __launch_bounds__(128)                                                                                 \
        name(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv, int q_stride, const int *pos, float scale, \
             const int *skip, AwGroups groups) {                                                                                      \
        attn_ws4_impl<F, PD>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);                            \
    }
WS4_ENTRY(attn_ws4, 0, 1)
WS4_ENTRY(attn_ws4_q8, 1, 1)
WS4_ENTRY(attn_ws4_q4, 2, 2)
// prefetch depth 1 for q4 (kernel development / bench; not used by the host). Depth 2 for f16 /
// int8 does not fit: 256 VGPRs with 76 / 94 spilled (ROCm 7.2), so they stay at depth 1.
WS4_ENTRY(attn_ws4_q4_pd1, 2, 1)
#undef WS4_ENTRY
