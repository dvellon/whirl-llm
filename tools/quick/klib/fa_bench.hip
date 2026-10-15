// KLIB-1 tier-0 micro-benchmark (developer tool, not shipped). SPDX-License-Identifier: Apache-2.0
// Prefill attention on the 8060S (gfx1151), Qwen3.8-27B shape (24 / 4 heads, head dim 256, causal):
//   ours  = WHIRL attn_prefill_wmma (f16 K/V) loaded from the engine's code object (--hsaco)
//   strix = flash_attn_rdna35<256, 2, false> from asf0/strix-llama.cpp (MIT, fa_rdna35_kernel.h), with its
//           mask pre-pass (mask_classify) and a per-tile KV_max, launched the way launch_fattn_rdna35 does
// Same data for both: Q f32 [tok][head][512] (q, then the output gate = +30 so WHIRL's sigmoid gate is 1.0f),
// K / V f16 [row][kv head][256] contiguous (identity page table), out f32 [tok][head][256].
// Prints ms (median), TFLOPS (4*heads*hd*n*(depth + n/2)), NMSE / max abs vs a double CPU reference on sampled
// (token, head) pairs, ours-vs-strix difference, and run-to-run bit identity of each kernel.
// Build: hipcc -O3 --offload-arch=gfx1151 -std=c++17 -DKLIB_SWAP=4 fa_bench.hip -o fa_bench.exe
//   (KLIB_SWAP=4: swap_halves via ds_swizzle; the original permlanex16 form gives wrong results with the
//    HIP SDK 7.2 clang 21 on Windows, see audit/KLIB-1.md)
// Run:   fa_bench.exe --hsaco <whirl_kernels.gfx1151.hsaco> [--n 2048] [--depths 0,30720,62464,94208,126976] [--reps 3]
#include "fa_rdna35_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                   \
    do {                                                                                        \
        hipError_t e_ = (x);                                                                    \
        if (e_ != hipSuccess) {                                                                 \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

struct KvArgs {  // = kernels/gfx1151/attn.hip
    void *k, *v;
    _Float16 *ks, *vs;
    const int *ptab, *kvbase;
    int tab0, win;
};

static float h2f(_Float16 h) { return (float)h; }

int main(int argc, char **argv) {
    std::string hsaco;
    int n = 2048, reps = 3;
    std::vector<int> depths = {0, 30720, 62464, 94208, 126976};
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hsaco" && i + 1 < argc) hsaco = argv[++i];
        else if (a == "--n" && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--depths" && i + 1 < argc) {
            depths.clear();
            std::string s = argv[++i];
            for (size_t p = 0; p < s.size();) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                depths.push_back(std::atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
    }
    int ndev = 0, dev = -1;
    CK(hipGetDeviceCount(&ndev));
    for (int d = 0; d < ndev; ++d) {
        hipDeviceProp_t p;
        CK(hipGetDeviceProperties(&p, d));
        if (std::strncmp(p.gcnArchName, "gfx1151", 7) == 0) dev = d;
    }
    if (dev < 0) { std::fprintf(stderr, "no gfx1151 device\n"); return 1; }
    CK(hipSetDevice(dev));
    hipModule_t mod;
    hipFunction_t fours;
    CK(hipModuleLoad(&mod, hsaco.c_str()));
    CK(hipModuleGetFunction(&fours, mod, "attn_prefill_wmma"));

    const int H = 24, HKV = 4, D = 256, QS = 2 * D, GQA = H / HKV;
    const float scale = 1.0f / 16.0f, log2e = 1.44269504088896340736f;
    constexpr int NC2 = 2, NC1 = 64 / NC2;
    using namespace fattn_rdna35;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::printf("device %d gfx1151, n %d, heads %d/%d, hd %d, reps %d\n", dev, n, H, HKV, D, reps);
    std::printf("| depth | ours ms | strix ms | ours TF/s | strix TF/s | strix/ours speed | ours NMSE | strix NMSE | ours maxabs | strix maxabs | ours-vs-strix maxabs | ours bits stable | strix bits stable | classify ms |\n");
    std::printf("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|\n");
    for (const int depth : depths) {
        const int nkeys = depth + n;
        const int rows = (nkeys + 255) / 256 * 256;  // whole pages (also a multiple of 64 for strix)
        // data
        std::vector<float> hq((size_t)n * H * QS);
        for (int t = 0; t < n; ++t)
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < QS; ++d) hq[((size_t)t * H + h) * QS + d] = d < D ? 2.0f * nd(rng) : 30.0f;
        std::vector<_Float16> hk((size_t)rows * HKV * D), hv(hk.size());
        for (size_t i = 0; i < hk.size(); ++i) {
            hk[i] = (_Float16)nd(rng);
            hv[i] = (_Float16)nd(rng);
        }
        if (std::getenv("FA_VDIAG")) {  // V row k = k, Q = 0: out[t] = mean(0..depth+t) = (depth+t)/2
            for (int r = 0; r < rows; ++r)
                for (int d = 0; d < HKV * D; ++d) hv[(size_t)r * HKV * D + d] = (_Float16)(float)(r % 2048);
            for (int t = 0; t < n; ++t)
                for (int h = 0; h < H; ++h)
                    for (int d = 0; d < D; ++d) hq[((size_t)t * H + h) * QS + d] = 0.f;
        }
        // a few "attention sink" style keys with large dot products so softmax is not flat
        for (int r = 0; r < rows; r += 997)
            for (int d = 0; d < HKV * D; ++d) hk[(size_t)r * HKV * D + d] = (_Float16)(3.0f * h2f(hk[(size_t)r * HKV * D + d]));
        float *dq, *dout_o, *dout_s;
        _Float16 *dk, *dv;
        int *dptab, *dpos;
        CK(hipMalloc(&dq, hq.size() * 4));
        CK(hipMalloc(&dk, hk.size() * 2));
        CK(hipMalloc(&dv, hv.size() * 2));
        CK(hipMalloc(&dout_o, (size_t)n * H * D * 4));
        CK(hipMalloc(&dout_s, (size_t)n * H * D * 4));
        CK(hipMemcpy(dq, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
        CK(hipMemcpy(dv, hv.data(), hv.size() * 2, hipMemcpyHostToDevice));
        std::vector<int> ptab(rows / 256);
        for (size_t i = 0; i < ptab.size(); ++i) ptab[i] = (int)i;
        CK(hipMalloc(&dptab, ptab.size() * 4));
        CK(hipMemcpy(dptab, ptab.data(), ptab.size() * 4, hipMemcpyHostToDevice));
        CK(hipMalloc(&dpos, 4));
        CK(hipMemcpy(dpos, &depth, 4, hipMemcpyHostToDevice));
        // strix: causal mask [n][mstride] f16 (0 / -inf), classes, KV_max per 32-token tile
        const int ne11 = rows, mstride = (ne11 + 1023) / 1024 * 1024;
        std::vector<uint16_t> hm((size_t)n * mstride);
        for (int t = 0; t < n; ++t)
            for (int k = 0; k < mstride; ++k) hm[(size_t)t * mstride + k] = (k <= depth + t) ? 0x0000 : 0xFC00;
        half *dmask;
        CK(hipMalloc(&dmask, hm.size() * 2));
        CK(hipMemcpy(dmask, hm.data(), hm.size() * 2, hipMemcpyHostToDevice));
        std::vector<uint16_t>().swap(hm);
        const int ntiles = (n + NC1 - 1) / NC1, n_cls = ne11 / CLS_KEYS, n_bytes = (n_cls + 3) / 4;
        uint8_t *dcls;
        CK(hipMalloc(&dcls, (size_t)ntiles * n_bytes));
        std::vector<int> kvmax(ntiles);
        for (int j = 0; j < ntiles; ++j) kvmax[j] = depth + std::min((j + 1) * NC1, n);
        int *dkvmax;
        CK(hipMalloc(&dkvmax, ntiles * 4));
        CK(hipMemcpy(dkvmax, kvmax.data(), ntiles * 4, hipMemcpyHostToDevice));

        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        auto classify = [&]() {
            const int n_blk = (ne11 + CLS_BLOCK_KEYS - 1) / CLS_BLOCK_KEYS;
            hipLaunchKernelGGL(mask_classify<NC1>, dim3(ntiles, n_blk, 1), dim3(CLS_WAVES * 32), 0, 0, (const half *)dmask, dcls, n,
                               (int64_t)mstride, (int64_t)0, n_cls, n_bytes);
        };
        auto run_strix = [&]() {
            hipLaunchKernelGGL((flash_attn_rdna35<256, NC2, false>), dim3(ntiles * (H / NC2), 1, 1), dim3(cfg<256>::NTHR), 0, 0,
                               (const char *)dq, (const char *)dk, (const char *)dv, (const char *)dmask, std::getenv("FA_NOCLS") ? (const uint8_t *)nullptr : (const uint8_t *)dcls,
                               (const int *)dkvmax, dout_s, (float *)nullptr, (float2 *)nullptr, (const float *)nullptr, scale * log2e,
                               0.0f, n, H, ne11, GQA, (int64_t)H * QS * 4, (int64_t)QS * 4, (int64_t)0, (int64_t)HKV * D * 2,
                               (int64_t)D * 2, (int64_t)0, (int64_t)HKV * D * 2, (int64_t)D * 2, (int64_t)0, n, 1,
                               (int64_t)mstride * 2, (int64_t)0);
        };
        KvArgs kv{dk, dv, nullptr, nullptr, dptab, nullptr, 0, 0};
        auto run_ours = [&]() {
            int nh = H, nkv = HKV, qs = QS, nt = n, h0 = 0;
            float sc = scale;
            void *args[] = {&dq, &kv, &dout_o, &nh, &nkv, &qs, &dpos, &nt, &sc, &h0};
            CK(hipModuleLaunchKernel(fours, (n + 127) / 128, H, 1, 256, 1, 1, 0, 0, args, nullptr));
        };
        auto timeit = [&](auto f) {
            CK(hipEventRecord(e0, 0));
            f();
            CK(hipEventRecord(e1, 0));
            CK(hipEventSynchronize(e1));
            float ms = 0;
            CK(hipEventElapsedTime(&ms, e0, e1));
            return ms;
        };
        const float cls_ms = timeit(classify);
        CK(hipGetLastError());
        // warm-up (clocks ramp: the first cold launches ran ~2x slower), first outputs
        for (int w = 0; w < 3; ++w) {
            run_ours();
            run_strix();
        }
        CK(hipDeviceSynchronize());
        CK(hipGetLastError());
        std::vector<float> o1((size_t)n * H * D), s1(o1.size()), o2(o1.size()), s2(o1.size());
        CK(hipMemcpy(o1.data(), dout_o, o1.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(s1.data(), dout_s, s1.size() * 4, hipMemcpyDeviceToHost));
        if (std::getenv("FA_DEBUG"))
            for (int t : {0, 1, 40}) for (int h : {0, 1, 2}) std::printf("dbg t%d h%d ours %.4f %.4f strix %.4f %.4f v0 %.4f %.4f\n", t, h, o1[((size_t)t*H+h)*D], o1[((size_t)t*H+h)*D+1], s1[((size_t)t*H+h)*D], s1[((size_t)t*H+h)*D+1], h2f(hv[(size_t)(h/GQA)*D]), h2f(hv[(size_t)(h/GQA)*D+1]));
        if (std::getenv("FA_DEBUG")) {
            for (int d = 0; d < 32; ++d) {
                const float sv = s1[d];
                int hit = -1;
                for (int e = 0; e < HKV * D * 4 && e < (int)hv.size(); ++e)
                    if (std::fabs(h2f(hv[e]) - sv) < 2e-3f) { hit = e; break; }
                std::printf("d%d strix %.4f -> v elem %d (row %d kvh %d dim %d)\n", d, sv, hit, hit / (HKV * D), (hit / D) % HKV, hit % D);
            }
        }
        std::vector<float> to, ts;
        for (int r = 0; r < reps; ++r) {  // ABAB, order flipped every rep
            if (r % 2 == 0) {
                to.push_back(timeit(run_ours));
                ts.push_back(timeit(run_strix));
            } else {
                ts.push_back(timeit(run_strix));
                to.push_back(timeit(run_ours));
            }
        }
        CK(hipMemcpy(o2.data(), dout_o, o2.size() * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(s2.data(), dout_s, s2.size() * 4, hipMemcpyDeviceToHost));
        const bool o_stable = std::memcmp(o1.data(), o2.data(), o1.size() * 4) == 0;
        const bool s_stable = std::memcmp(s1.data(), s2.data(), s1.size() * 4) == 0;
        std::sort(to.begin(), to.end());
        std::sort(ts.begin(), ts.end());
        const double mo = to[to.size() / 2], ms = ts[ts.size() / 2];
        const double flops = 4.0 * H * D * (double)n * ((double)depth + n / 2.0);
        // CPU reference on sampled (token, head)
        double eo = 0, es = 0, ref2 = 0, mxo = 0, mxs = 0;
        std::vector<double> sc(nkeys), acc(D);
        const int samples[] = {0, 1, 31, 32, 63, 500, 1023, 1024, 1500, 2000, n - 2, n - 1};
        for (int t : samples) {
            if (t < 0 || t >= n) continue;
            for (int h : {0, 7, 23}) {
                const int kvh = h / GQA;
                const float *qr = &hq[((size_t)t * H + h) * QS];
                const int nk = depth + t + 1;
                double mx = -1e300;
                for (int k = 0; k < nk; ++k) {
                    const _Float16 *kr = &hk[((size_t)k * HKV + kvh) * D];
                    double s = 0;
                    for (int d = 0; d < D; ++d) s += (double)qr[d] * (double)h2f(kr[d]);
                    sc[k] = s * scale;
                    mx = std::max(mx, sc[k]);
                }
                double l = 0;
                std::fill(acc.begin(), acc.end(), 0.0);
                for (int k = 0; k < nk; ++k) {
                    const double p = std::exp(sc[k] - mx);
                    l += p;
                    const _Float16 *vr = &hv[((size_t)k * HKV + kvh) * D];
                    for (int d = 0; d < D; ++d) acc[d] += p * (double)h2f(vr[d]);
                }
                for (int d = 0; d < D; ++d) {
                    const double r = acc[d] / l;
                    const double a = o1[((size_t)t * H + h) * D + d], b = s1[((size_t)t * H + h) * D + d];
                    eo += (a - r) * (a - r);
                    es += (b - r) * (b - r);
                    ref2 += r * r;
                    mxo = std::max(mxo, std::fabs(a - r));
                    mxs = std::max(mxs, std::fabs(b - r));
                }
            }
        }
        double dos = 0;
        for (size_t i = 0; i < o1.size(); ++i) dos = std::max(dos, (double)std::fabs(o1[i] - s1[i]));
        std::printf("| %d | %.2f | %.2f | %.1f | %.1f | %.3fx | %.2e | %.2e | %.2e | %.2e | %.2e | %s | %s | %.2f |\n", depth, mo, ms,
                    flops / mo / 1e9, flops / ms / 1e9, mo / ms, eo / ref2, es / ref2, mxo, mxs, dos, o_stable ? "yes" : "NO",
                    s_stable ? "yes" : "NO", cls_ms);
        std::fflush(stdout);
        CK(hipFree(dq)); CK(hipFree(dk)); CK(hipFree(dv)); CK(hipFree(dout_o)); CK(hipFree(dout_s));
        CK(hipFree(dptab)); CK(hipFree(dpos)); CK(hipFree(dmask)); CK(hipFree(dcls)); CK(hipFree(dkvmax));
        CK(hipEventDestroy(e0)); CK(hipEventDestroy(e1));
    }
    return 0;
}
