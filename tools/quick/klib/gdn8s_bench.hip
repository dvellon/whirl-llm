// GDN-8S tier-0 micro-benchmark (developer tool, not shipped). SPDX-License-Identifier: Apache-2.0
// From the engine's gfx1151 code object, on the Qwen3.8-27B DeltaNet shapes (16 k heads, 48 v heads, 128x128,
// conv rows q | k | v with L2-normalized q / k):
//   * gdn_sx_swap_probe: the half-wave swap the WMMA fragment shuffle uses == __shfl_xor(x, 16) (per-op check)
//   * gdn_sx_prep + gdn_sx_scan (strix-llama.cpp chunked f16 WMMA, GDN-8S) vs gdn_seq_128 (f32 reference):
//     NMSE of the output and of the final state, max |err| / rms, run-to-run bit stability
//   * gdn_chunk_prep + gdn_chunk_scan (current gfx1151 default) for comparison; timing (us per layer)
//   * two chained calls (state carried) to check the in-place state handoff
// Build: hipcc -O3 --offload-arch=gfx1151 -std=c++17 gdn8s_bench.hip -o gdn8s_bench.exe
// Run:   gdn8s_bench.exe --hsaco <whirl_kernels.gfx1151.hsaco> [--iters 40] [--n 2048 --n 300 ...]
#include <hip/hip_runtime.h>

#include <algorithm>
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

static std::mt19937 rng(77);

template <class F>
static double time_us(F f, int iters) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    for (int i = 0; i < 3; ++i) f();
    CK(hipDeviceSynchronize());
    CK(hipEventRecord(a, 0));
    for (int i = 0; i < iters; ++i) f();
    CK(hipEventRecord(b, 0));
    CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    return ms * 1000.0 / iters;
}

struct Err {
    double nmse, maxrel;
};
static Err cmp(const std::vector<float> &g, const std::vector<float> &r) {
    double e = 0, r2 = 0, mx = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        const double d = (double)g[i] - r[i];
        e += d * d;
        r2 += (double)r[i] * r[i];
        mx = std::max(mx, std::fabs(d));
    }
    return {e / r2, mx / std::sqrt(r2 / g.size())};
}

int main(int argc, char **argv) {
    std::string hsaco, sxco, strix_scan;  // --sx: code object for gdn_sx_*; --strix-scan NAME: scan with strix's original ABI
    int iters = 40;
    std::vector<int> ns;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hsaco" && i + 1 < argc) hsaco = argv[++i];
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--n" && i + 1 < argc) ns.push_back(std::atoi(argv[++i]));
        else if (a == "--sx" && i + 1 < argc) sxco = argv[++i];
        else if (a == "--strix-scan" && i + 1 < argc) strix_scan = argv[++i];
    }
    if (ns.empty()) ns = {2048};
    int ndev = 0, dev = -1;
    CK(hipGetDeviceCount(&ndev));
    for (int d = 0; d < ndev; ++d) {
        hipDeviceProp_t p;
        CK(hipGetDeviceProperties(&p, d));
        if (std::strncmp(p.gcnArchName, "gfx1151", 7) == 0) dev = d;
    }
    if (dev < 0) {
        std::fprintf(stderr, "no gfx1151\n");
        return 1;
    }
    CK(hipSetDevice(dev));
    hipModule_t mod;
    CK(hipModuleLoad(&mod, hsaco.c_str()));
    hipModule_t smod = mod;
    if (!sxco.empty()) CK(hipModuleLoad(&smod, sxco.c_str()));
    auto fn = [&](const char *nm) {
        hipFunction_t f;
        CK(hipModuleGetFunction(&f, mod, nm));
        return f;
    };
    auto sfn = [&](const char *nm) {
        hipFunction_t f;
        CK(hipModuleGetFunction(&f, smod, nm));
        return f;
    };
    // ---- per-op check of the half-wave swap
    {
        std::vector<float> in(32), out(64);
        for (int l = 0; l < 32; ++l) in[l] = 1000.f + l * 1.25f;
        float *din, *dout;
        CK(hipMalloc(&din, 128));
        CK(hipMalloc(&dout, 256));
        CK(hipMemcpy(din, in.data(), 128, hipMemcpyHostToDevice));
        void *a[] = {&din, &dout};
        CK(hipModuleLaunchKernel(fn("gdn_sx_swap_probe"), 1, 1, 1, 32, 1, 1, 0, 0, a, nullptr));
        CK(hipMemcpy(out.data(), dout, 256, hipMemcpyDeviceToHost));
        int bad = 0;
        for (int l = 0; l < 32; ++l) bad += out[l] != out[32 + l] || out[l] != in[l ^ 16];
        std::printf("swap probe (gdn_sx_swap_probe vs __shfl_xor 16 vs host lane^16): %s (%d bad lanes)\n", bad ? "FAIL" : "ok", bad);
        CK(hipFree(din));
        CK(hipFree(dout));
    }
    std::printf("\n| n | path | us/layer | out NMSE vs seq | out max|err|/rms | state NMSE vs seq | bits stable | 2-call chain out NMSE |\n");
    std::printf("|---:|---|---:|---:|---:|---:|---|---:|\n");
    for (int n : ns) {
        const int nk = 16, nv = 48, dk = 128, dv = 128, ch = 2 * nk * dk + nv * dv;
        const int n64 = (n + 63) / 64, n32 = (n + 31) / 32;
        const float scale = 1.0f / std::sqrt((float)dk);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> ud(0.f, 1.f);
        std::vector<float> qkv((size_t)n * ch), gl((size_t)n * nv), beta((size_t)n * nv), st((size_t)nv * dk * dv);
        for (auto &x : qkv) x = nd(rng);
        for (int t = 0; t < n; ++t)
            for (int h = 0; h < 2 * nk; ++h) {
                float *r = &qkv[(size_t)t * ch + h * dk];
                double s = 0;
                for (int d = 0; d < dk; ++d) s += (double)r[d] * r[d];
                const float inv = (float)(1.0 / std::sqrt(s + 1e-6));
                for (int d = 0; d < dk; ++d) r[d] *= inv;
            }
        for (auto &x : gl) x = -0.5f * ud(rng) - 1e-3f;
        for (auto &x : beta) x = ud(rng);
        for (auto &x : st) x = 0.05f * nd(rng);
        float *dqkv, *dgl, *dbeta, *dst, *dst0, *dout, *dw, *du, *dm, *dg;
        void *tb, *pb, *gb, *qh, *kh, *vh;
        CK(hipMalloc(&dqkv, qkv.size() * 4));
        CK(hipMalloc(&dgl, gl.size() * 4));
        CK(hipMalloc(&dbeta, beta.size() * 4));
        CK(hipMalloc(&dst, st.size() * 4));
        CK(hipMalloc(&dst0, st.size() * 4));
        CK(hipMalloc(&dout, (size_t)n * nv * dv * 4));
        CK(hipMalloc(&dw, (size_t)n64 * nv * 64 * 128 * 4));
        CK(hipMalloc(&du, (size_t)n64 * nv * 64 * 128 * 4));
        CK(hipMalloc(&dm, (size_t)n64 * nv * 64 * 64 * 4));
        CK(hipMalloc(&dg, (size_t)n64 * nv * 64 * 4));
        CK(hipMalloc(&tb, (size_t)nv * n32 * 32 * 32 * 2));
        CK(hipMalloc(&pb, (size_t)nv * n32 * 32 * 32 * 2));
        CK(hipMalloc(&gb, (size_t)nv * n32 * 32 * 8));
        CK(hipMalloc(&qh, (size_t)nk * n * 128 * 2));
        CK(hipMalloc(&kh, (size_t)nk * n * 128 * 2));
        CK(hipMalloc(&vh, (size_t)nv * n * 128 * 2));
        CK(hipMemcpy(dqkv, qkv.data(), qkv.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dgl, gl.data(), gl.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dbeta, beta.data(), beta.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dst0, st.data(), st.size() * 4, hipMemcpyHostToDevice));
        hipFunction_t fp = fn("gdn_chunk_prep"), fs = fn("gdn_chunk_scan"), fq = fn("gdn_seq_128");
        hipFunction_t xp = sfn("gdn_sx_prep"), xs = sfn(strix_scan.empty() ? "gdn_sx_scan" : strix_scan.c_str());
        void *nul = nullptr;
        float feps = 0.f;
        bool fsilu = false;
        int in = n, ink = nk, inv = nv, ich = ch, idv = dv, zero = 0, i32n = n32;
        float sc = scale;
        uint64_t nosnap = 0;
        auto chunk = [&]() {
            void *a1[] = {&dqkv, &dgl, &dbeta, &dw, &du, &dm, &dg, &in, &ink, &inv, &ich};
            CK(hipModuleLaunchKernel(fp, n64, nv, 1, 256, 1, 1, 0, 0, a1, nullptr));
            void *a2[] = {&dqkv, &dw, &du, &dm, &dg, &dst, &dout, &in, &ink, &inv, &ich, &sc};
            CK(hipModuleLaunchKernel(fs, nv, dv / 32, 1, 256, 1, 1, 0, 0, a2, nullptr));
        };
        auto seq = [&]() {
            void *a[] = {&dqkv, &dgl, &dbeta, &dst, &dout, &in, &ink, &inv, &idv, &ich, &sc, &nosnap, &zero};
            CK(hipModuleLaunchKernel(fq, nv, dv / 32, 1, 128, 1, 1, 0, 0, a, nullptr));
        };
        auto strix = [&]() {
            void *a1[] = {&dqkv, &dgl, &dbeta, &tb, &pb, &gb, &qh, &kh, &vh, &inv, &ink, &in, &i32n, &ich, &sc};
            CK(hipModuleLaunchKernel(xp, nk, n32, 1, 256, 1, 1, 0, 0, a1, nullptr));
            void *a2[] = {&qh, &kh, &vh, &dst, &dout, &tb, &pb, &gb, &inv, &ink, &in, &i32n, &sc};
            void *a3[] = {&qh, &kh, &vh, &dst, &dout, &dst, &tb, &pb, &gb, &inv, &ink, &in, &i32n, &sc, &nul, &nul, &nul, &nul, &feps, &fsilu};
            CK(hipModuleLaunchKernel(xs, nv, 1, 1, 256, 1, 1, 0, 0, a3, nullptr));
        };
        auto run_once = [&](auto f, std::vector<float> &o, std::vector<float> &s, int calls) {
            CK(hipMemcpy(dst, dst0, st.size() * 4, hipMemcpyDeviceToDevice));
            for (int c = 0; c < calls; ++c) f();
            CK(hipDeviceSynchronize());
            o.resize((size_t)n * nv * dv);
            s.resize(st.size());
            CK(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
            CK(hipMemcpy(s.data(), dst, s.size() * 4, hipMemcpyDeviceToHost));
        };
        std::vector<float> os, ss, os2, ss2, o1, s1, o2, s2, oc, scs, oc2, sc2;
        run_once(seq, os, ss, 1);
        run_once(seq, os2, ss2, 2);  // second call continues from the state of the first
        const int it = std::max(5, iters);
        struct P {
            const char *name;
            int kind;
        };
        for (P p : {P{"gdn_chunk_prep+scan (default, f32)", 0}, P{"gdn_sx_prep+scan (strix f16 WMMA)", 1}}) {
            auto f = [&]() { p.kind == 0 ? chunk() : strix(); };
            run_once(f, o1, s1, 1);
            run_once(f, o2, s2, 1);
            const bool stable = std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0 && std::memcmp(s1.data(), s2.data(), s1.size() * 4) == 0;
            std::vector<float> oc_, sc_;
            run_once(f, oc_, sc_, 2);
            const Err eo = cmp(o1, os), es = cmp(s1, ss), e2 = cmp(oc_, os2);
            const double us = time_us(f, it);
            std::printf("| %d | %s | %.0f | %.2e | %.2e | %.2e | %s | %.2e |\n", n, p.name, us, eo.nmse, eo.maxrel, es.nmse, stable ? "yes" : "NO",
                        e2.nmse);
        }
        {
            auto sp = [&]() {
                void *a1[] = {&dqkv, &dgl, &dbeta, &tb, &pb, &gb, &qh, &kh, &vh, &inv, &ink, &in, &i32n, &ich, &sc};
                CK(hipModuleLaunchKernel(xp, nk, n32, 1, 256, 1, 1, 0, 0, a1, nullptr));
            };
            auto ss_ = [&]() {
                void *a2[] = {&qh, &kh, &vh, &dst, &dout, &tb, &pb, &gb, &inv, &ink, &in, &i32n, &sc};
                void *a3[] = {&qh, &kh, &vh, &dst, &dout, &dst, &tb, &pb, &gb, &inv, &ink, &in, &i32n, &sc, &nul, &nul, &nul, &nul, &feps, &fsilu};
                CK(hipModuleLaunchKernel(xs, nv, 1, 1, 256, 1, 1, 0, 0, a3, nullptr));
            };
            std::printf("| %d | (split) gdn_sx_prep alone | %.0f | | | | | |\n", n, time_us(sp, it));
            std::printf("| %d | (split) gdn_sx_scan alone | %.0f | | | | | |\n", n, time_us(ss_, it));
        }
        {
            const double us = time_us(seq, std::max(3, it / 4));
            std::printf("| %d | gdn_seq_128 (reference) | %.0f | - | - | - | - | - |\n", n, us);
        }
        for (void *p : {(void *)dqkv, (void *)dgl, (void *)dbeta, (void *)dst, (void *)dst0, (void *)dout, (void *)dw, (void *)du, (void *)dm,
                        (void *)dg, tb, pb, gb, qh, kh, vh})
            CK(hipFree(p));
    }
    return 0;
}
