// KLIB-1 tier-0 micro-benchmark (developer tool, not shipped). SPDX-License-Identifier: Apache-2.0
// Times WHIRL's own gfx1151 kernels the way test-backend-ops perf times llama.cpp's (back-to-back launches on the
// same buffers, mean us per call), on the Qwen3.8-27B shapes, from the engine's code object:
//   gemv:  gemvq_q3_k (1 token, grouped GvArgs ABI) and gemvq_nt4_q3_k (4 tokens), 17408x5120 and 5120x17408,
//          random Q3_K superblocks (valid f16 d), int8 activations; GB/s of weight bytes; output bit stability
//   gdn:   gdn_chunk_prep + gdn_chunk_scan (the gfx1151 prefill recurrence, C = 64) and gdn_seq_128 for n tokens,
//          16 k heads / 48 v heads, dk = dv = 128; NMSE of chunk vs seq; output bit stability
// Build: hipcc -O3 --offload-arch=gfx1151 -std=c++17 ours_bench.hip -o ours_bench.exe
// Run:   ours_bench.exe --hsaco <whirl_kernels.gfx1151.hsaco> [--iters 200] [--n 2048]
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                                   \
    do {                                                                                                        \
        hipError_t e_ = (x);                                                                                    \
        if (e_ != hipSuccess) {                                                                                 \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); \
            std::exit(1);                                                                                       \
        }                                                                                                       \
    } while (0)

struct GvArgs {  // = kernels/gfx1151/gemv_q.hip
    const uint8_t *W0, *W1, *W2;
    float *y0, *y1, *y2;
    uint64_t rb0, rb1, rb2;
    int n0, n1, n2;
    int b1, b2;
};

static std::mt19937 rng(77);

template <class F>
static double time_us(F f, int iters) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 5; ++i) f();
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a, 0));
    for (int i = 0; i < iters; ++i) f();
    CK(hipEventRecord(b, 0));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return ms * 1000.0 / iters;
}

int main(int argc, char **argv) {
    std::string hsaco;
    int iters = 200, ngdn = 2048;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hsaco" && i + 1 < argc) hsaco = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--n" && i + 1 < argc) ngdn = std::atoi(argv[++i]);
    }
    int ndev = 0, dev = -1;
    CK(hipGetDeviceCount(&ndev));
    for (int d = 0; d < ndev; ++d) {
        hipDeviceProp_t p;
        CK(hipGetDeviceProperties(&p, d));
        if (std::strncmp(p.gcnArchName, "gfx1151", 7) == 0) dev = d;
    }
    if (dev < 0) { std::fprintf(stderr, "no gfx1151\n"); return 1; }
    CK(hipSetDevice(dev));
    hipModule_t mod;
    CK(hipModuleLoad(&mod, hsaco.c_str()));
    auto fn = [&](const char *nm) {
        hipFunction_t f;
        CK(hipModuleGetFunction(&f, mod, nm));
        return f;
    };
    // ---------------- Q3_K GEMV ----------------
    std::printf("## gemv q3_k (ours), us per call over %d back-to-back calls, same buffers\n", iters);
    std::printf("| kernel | rows x cols | tokens | us | weight GB/s | %% of 256 GB/s | bits stable |\n|---|---|---:|---:|---:|---:|---|\n");
    hipFunction_t g1 = fn("gemvq_q3_k"), g4 = fn("gemvq_nt4_q3_k");
    for (auto [nrows, ncols] : {std::pair{17408, 5120}, std::pair{5120, 17408}}) {
        const uint64_t rb = (uint64_t)ncols / 256 * 110;
        std::vector<uint8_t> w(rb * nrows);
        for (auto &x : w) x = (uint8_t)(rng() & 0xFF);
        for (int r = 0; r < nrows; ++r)
            for (int b = 0; b < ncols / 256; ++b) {
                const uint16_t d = 0x1400 + (uint16_t)(rng() % 0x400);  // f16 ~ 1e-3 .. 2e-3
                std::memcpy(&w[r * rb + b * 110 + 108], &d, 2);
            }
        std::vector<int8_t> xq((size_t)4 * ncols);
        for (auto &x : xq) x = (int8_t)((int)(rng() % 255) - 127);
        std::vector<float> xd((size_t)4 * ncols / 32);
        for (auto &x : xd) x = 0.01f + 0.001f * (float)(rng() % 100);
        uint8_t *dw;
        int8_t *dxq;
        float *dxd, *dy;
        CK(hipMalloc(&dw, w.size()));
        CK(hipMalloc(&dxq, xq.size()));
        CK(hipMalloc(&dxd, xd.size() * 4));
        CK(hipMalloc(&dy, (size_t)4 * nrows * 4));
        CK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dxq, xq.data(), xq.size(), hipMemcpyHostToDevice));
        CK(hipMemcpy(dxd, xd.data(), xd.size() * 4, hipMemcpyHostToDevice));
        const unsigned nb = (nrows + 7) / 8;
        int nc = ncols, nr = nrows, acc = 0;
        int *skip = nullptr;
        auto l1 = [&]() {
            GvArgs g{dw, nullptr, nullptr, dy, nullptr, nullptr, rb, 0, 0, nr, 0, 0, (int)nb, (int)nb};
            void *args[] = {&g, &dxq, &dxd, &nc, &acc, &skip};
            CK(hipModuleLaunchKernel(g1, nb, 1, 1, 256, 1, 1, 0, 0, args, nullptr));
        };
        auto l4 = [&]() {
            uint64_t rbv = rb;
            void *args[] = {&dw, &rbv, &dxq, &dxd, &dy, &nc, &nr, &acc, &skip};
            CK(hipModuleLaunchKernel(g4, nb, 1, 1, 256, 1, 1, 0, 0, args, nullptr));
        };
        for (int t : {1, 4}) {
            auto f = [&]() { t == 1 ? l1() : l4(); };
            f();
            CK(hipDeviceSynchronize());
            std::vector<float> y1((size_t)t * nrows), y2(y1.size());
            CK(hipMemcpy(y1.data(), dy, y1.size() * 4, hipMemcpyDeviceToHost));
            const double us = time_us(f, iters);
            CK(hipMemcpy(y2.data(), dy, y2.size() * 4, hipMemcpyDeviceToHost));
            const double gbs = (double)w.size() / us / 1e3;
            std::printf("| %s | %d x %d | %d | %.1f | %.0f | %.0f%% | %s |\n", t == 1 ? "gemvq_q3_k" : "gemvq_nt4_q3_k", nrows, ncols, t, us, gbs,
                        gbs / 2.56, std::memcmp(y1.data(), y2.data(), y1.size() * 4) == 0 ? "yes" : "NO");
        }
        CK(hipFree(dw)); CK(hipFree(dxq)); CK(hipFree(dxd)); CK(hipFree(dy));
    }
    // ---------------- GDN prefill recurrence ----------------
    {
        const int n = ngdn, nk = 16, nv = 48, dk = 128, dv = 128, ch = 2 * nk * dk + nv * dv;
        const int n_chunks = (n + 63) / 64;
        const float scale = 1.0f / std::sqrt((float)dk);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> ud(0.f, 1.f);
        std::vector<float> qkv((size_t)n * ch), gl((size_t)n * nv), beta((size_t)n * nv), st((size_t)nv * dk * dv);
        for (auto &x : qkv) x = nd(rng);
        for (int t = 0; t < n; ++t)  // q, k rows L2-normalized (gdn_conv_l2 does this in the engine)
            for (int h = 0; h < 2 * nk; ++h) {
                float *r = &qkv[(size_t)t * ch + h * dk];
                double s = 0;
                for (int d = 0; d < dk; ++d) s += (double)r[d] * r[d];
                const float inv = (float)(1.0 / std::sqrt(s + 1e-6));
                for (int d = 0; d < dk; ++d) r[d] *= inv;
            }
        for (auto &x : gl) x = -0.5f * ud(rng) - 1e-3f;  // log decay
        for (auto &x : beta) x = ud(rng);
        for (auto &x : st) x = 0.05f * nd(rng);
        float *dqkv, *dgl, *dbeta, *dst, *dst0, *dout, *dw, *du, *dm, *dg;
        CK(hipMalloc(&dqkv, qkv.size() * 4));
        CK(hipMalloc(&dgl, gl.size() * 4));
        CK(hipMalloc(&dbeta, beta.size() * 4));
        CK(hipMalloc(&dst, st.size() * 4));
        CK(hipMalloc(&dst0, st.size() * 4));
        CK(hipMalloc(&dout, (size_t)n * nv * dv * 4));
        CK(hipMalloc(&dw, (size_t)n_chunks * nv * 64 * 128 * 4));
        CK(hipMalloc(&du, (size_t)n_chunks * nv * 64 * 128 * 4));
        CK(hipMalloc(&dm, (size_t)n_chunks * nv * 64 * 64 * 4));
        CK(hipMalloc(&dg, (size_t)n_chunks * nv * 64 * 4));
        CK(hipMemcpy(dqkv, qkv.data(), qkv.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dgl, gl.data(), gl.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dbeta, beta.data(), beta.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dst0, st.data(), st.size() * 4, hipMemcpyHostToDevice));
        hipFunction_t fp = fn("gdn_chunk_prep"), fs = fn("gdn_chunk_scan"), fq = fn("gdn_seq_128");
        int in = n, ink = nk, inv = nv, ich = ch, idv = dv, zero = 0;
        float sc = scale;
        uint64_t nosnap = 0;
        auto chunk = [&]() {
            void *a1[] = {&dqkv, &dgl, &dbeta, &dw, &du, &dm, &dg, &in, &ink, &inv, &ich};
            CK(hipModuleLaunchKernel(fp, n_chunks, nv, 1, 256, 1, 1, 0, 0, a1, nullptr));
            void *a2[] = {&dqkv, &dw, &du, &dm, &dg, &dst, &dout, &in, &ink, &inv, &ich, &sc};
            CK(hipModuleLaunchKernel(fs, nv, dv / 32, 1, 256, 1, 1, 0, 0, a2, nullptr));
        };
        auto seq = [&]() {
            void *a[] = {&dqkv, &dgl, &dbeta, &dst, &dout, &in, &ink, &inv, &idv, &ich, &sc, &nosnap, &zero};
            CK(hipModuleLaunchKernel(fq, nv, dv / 32, 1, 128, 1, 1, 0, 0, a, nullptr));
        };
        auto run_once = [&](auto f, std::vector<float> &o) {
            CK(hipMemcpy(dst, dst0, st.size() * 4, hipMemcpyDeviceToDevice));
            f();
            CK(hipDeviceSynchronize());
            o.resize((size_t)n * nv * dv);
            CK(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
        };
        std::vector<float> oc1, oc2, os1, os2;
        run_once(chunk, oc1);
        run_once(chunk, oc2);
        run_once(seq, os1);
        run_once(seq, os2);
        double e = 0, r2 = 0;
        for (size_t i = 0; i < oc1.size(); ++i) {
            e += (double)(oc1[i] - os1[i]) * (oc1[i] - os1[i]);
            r2 += (double)os1[i] * os1[i];
        }
        const int it = std::max(5, iters / 20);
        // state evolves across timed calls (as in test-backend-ops, which also reruns on the same state)
        const double us_c = time_us(chunk, it), us_s = time_us(seq, it);
        std::printf("\n## gdn prefill recurrence (ours), n %d tokens, 16/48 heads, dk = dv = 128, us per layer over %d calls\n", n, it);
        std::printf("| path | us | bits stable | NMSE vs gdn_seq_128 |\n|---|---:|---|---:|\n");
        std::printf("| gdn_chunk_prep + gdn_chunk_scan | %.0f | %s | %.2e |\n", us_c,
                    std::memcmp(oc1.data(), oc2.data(), oc1.size() * 4) == 0 ? "yes" : "NO", e / r2);
        std::printf("| gdn_seq_128 | %.0f | %s | - |\n", us_s, std::memcmp(os1.data(), os2.data(), os1.size() * 4) == 0 ? "yes" : "NO");
    }
    return 0;
}
