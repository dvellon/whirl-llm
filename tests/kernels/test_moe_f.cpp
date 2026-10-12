// whirl-kernel-test "moef": precise-mode MoE decode / verify experts (kernels/moe_f.hip,
// FIX-K97 / FR-K97 P2-2), every type with a moe_gu_f_<T> / moe_down_f_<T> kernel:
//   * moe_gu_f_<T> (gate rows [0, ff) from W1, up rows from W2) and moe_down_f_<T>
//     (K experts weighted + shared expert + residual) vs an independent CPU reference
//     (weights decoded exactly by ref::dequantRow, f32 activations, double accumulation);
//   * boundary shapes: C with fewer 8-value groups than lanes, ff / E rows not a multiple of
//     the 8 rows per block, one row; the shared expert absent (ysh = null);
//   * the stated precise guarantee: a pair's result depends only on (token, expert) - the
//     rows of an n-token launch (n = 1..16, prefix and offset windows) are bitwise the same
//     rows of the 32-token launch;
//   * the code's "as gemv_<T>_1" claim (one wave per row, groups g = lane, lane + 32, ...,
//     warp_sum): moe_gu_f_<T> rows == gemv_<T>_1 on the expert's rows, and moe_down_f_<T>
//     with K = 1, w = 1, no shared expert, x = 0 == gemv_<T>_1, bitwise (not MXFP4: its
//     whole-block kernel has its own order).
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <type_traits>

#include "cpu_ref.h"

namespace kt {

namespace {

Result inv(const std::string& name, std::size_t n, std::size_t bad) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    return r;
}

// K distinct experts per token, positive weights summing to 1
void routing(Ctx& c, int n, int R, int K, std::vector<int>& ids, std::vector<float>& w) {
    ids.assign(static_cast<std::size_t>(n) * K, 0);
    w.assign(ids.size(), 0.f);
    std::uniform_int_distribution<int> ur(0, R - 1);
    std::uniform_real_distribution<float> uw(0.05f, 1.f);
    for (int t = 0; t < n; ++t) {
        float s = 0;
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = ur(c.rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup = dup || ids[static_cast<std::size_t>(t) * K + j] == e;
            } while (dup);
            ids[static_cast<std::size_t>(t) * K + k] = e;
            w[static_cast<std::size_t>(t) * K + k] = uw(c.rng);
            s += w[static_cast<std::size_t>(t) * K + k];
        }
        for (int k = 0; k < K; ++k) w[static_cast<std::size_t>(t) * K + k] /= s;
    }
}

int minCols(QType t) {
    const int bv = wk::blockInfo(t).values;
    // fewer 8-value groups than the 32 lanes (idle lanes) where the block allows it
    return bv <= 8 ? 40 : bv <= 32 ? 96 : 256;
}

constexpr int kR = 12, kK = 8, kN = 32;

// gate + up: (C, ff) shape
void guCase(Ctx& c, QType t, int C, int ff) {
    const int ti = static_cast<int>(t);
    const std::string sfx = wk::typeSuffix(t);
    const std::string tag = " C=" + std::to_string(C) + " ff=" + std::to_string(ff);
    const HostMat W1 = randomMat(c, t, kR * ff, C), W2 = randomMat(c, t, kR * ff, C);
    Buf d1(W1.data), d2(W2.data);
    std::vector<int> ids;
    std::vector<float> w;
    routing(c, kN, kR, kK, ids, w);
    const std::vector<float> h = c.randn(static_cast<std::size_t>(kN) * C);
    auto run = [&](int t0, int n) {
        std::vector<int> id(ids.begin() + static_cast<std::ptrdiff_t>(t0) * kK, ids.begin() + static_cast<std::ptrdiff_t>(t0 + n) * kK);
        std::vector<float> hh(h.begin() + static_cast<std::ptrdiff_t>(t0) * C, h.begin() + static_cast<std::ptrdiff_t>(t0 + n) * C);
        Buf did(id), dh(hh), y1(static_cast<std::size_t>(n) * kK * ff * 4), y2(static_cast<std::size_t>(n) * kK * ff * 4);
        y1.fill(0xff);
        y2.fill(0xff);
        hip::launch(c.k.moe_gu_f[ti], {cdiv(2 * ff, 8), static_cast<unsigned>(n * kK), 1}, {256, 1, 1}, 0, c.s, d1.p(), d2.p(), W1.row_bytes, ff,
                    2 * ff, dh.p(), did.p(), y1.p(), y2.p(), C, kK);
        c.sync();
        std::vector<float> o = y1.down<float>(static_cast<std::size_t>(n) * kK * ff), u = y2.down<float>(o.size());
        o.insert(o.end(), u.begin(), u.end());  // [gate pairs][ff] then [up pairs][ff]
        return o;
    };
    const std::vector<float> full = run(0, kN);
    const std::size_t half = static_cast<std::size_t>(kN) * kK * ff;
    // CPU reference (every pair, every row)
    {
        const std::vector<float> w1 = ref::dequantRows(W1, 0, W1.nrows), w2 = ref::dequantRows(W2, 0, W2.nrows);
        std::vector<float> got;
        std::vector<double> rf, sc;
        for (int p = 0; p < kN * kK; ++p) {
            const int tok = p / kK, e = ids[static_cast<std::size_t>(p)];
            for (int m = 0; m < 2; ++m)
                for (int r = 0; r < ff; ++r) {
                    const float* wr = (m ? w2 : w1).data() + (static_cast<std::size_t>(e) * ff + r) * C;
                    double y = 0, a = 0;
                    for (int i = 0; i < C; ++i) {
                        const double pr = static_cast<double>(wr[i]) * h[static_cast<std::size_t>(tok) * C + i];
                        y += pr;
                        a += std::fabs(pr);
                    }
                    got.push_back(full[(m ? half : 0) + static_cast<std::size_t>(p) * ff + r]);
                    rf.push_back(y);
                    sc.push_back(a);
                }
        }
        c.rep.add(cmpTol("moe_gu_f_" + sfx + " vs CPU" + tag, got, rf, sc, 1e-5, 1e-7));
    }
    // batch invariance: n = 1..16 prefix, plus offset windows, == the 32-token rows
    {
        std::size_t bad = 0, cnt = 0;
        auto cmpWin = [&](int t0, int n) {
            const std::vector<float> o = run(t0, n);
            const std::size_t hn = static_cast<std::size_t>(n) * kK * ff;
            for (int m = 0; m < 2; ++m) {
                bad += std::memcmp(o.data() + (m ? hn : 0), full.data() + (m ? half : 0) + static_cast<std::size_t>(t0) * kK * ff, hn * 4) != 0;
                ++cnt;
            }
        };
        for (int n = 1; n <= 16; ++n) cmpWin(0, n);
        cmpWin(9, 7);
        cmpWin(16, 16);
        cmpWin(31, 1);
        c.rep.add(inv("moe_gu_f_" + sfx + ": n = 1..16 / offset windows == 32-token rows" + tag, cnt, bad));
    }
    // == gemv_<T>_1 on the expert's rows (the code's stated reduction order)
    if (t != QType::mxfp4 && c.k.gemv1[ti]) {
        std::size_t bad = 0, cnt = 0;
        Buf dh(h), y(static_cast<std::size_t>(ff) * 4);
        for (int p = 0; p < kN * kK; p += 13) {
            const int tok = p / kK, e = ids[static_cast<std::size_t>(p)];
            for (int m = 0; m < 2; ++m) {
                const DevPtr Wp = (m ? d2.p() : d1.p()) + static_cast<std::uint64_t>(e) * ff * W1.row_bytes;
                hip::launch(c.k.gemv1[ti], {cdiv(ff, 8), 1, 1}, {256, 1, 1}, 0, c.s, Wp, W1.row_bytes,
                            dh.p() + static_cast<std::uint64_t>(tok) * C * 4, y.p(), C, ff, C, ff, 0);
                c.sync();
                const auto g = y.down<float>(static_cast<std::size_t>(ff));
                bad += std::memcmp(g.data(), full.data() + (m ? half : 0) + static_cast<std::size_t>(p) * ff, static_cast<std::size_t>(ff) * 4) != 0;
                ++cnt;
            }
        }
        c.rep.add(inv("moe_gu_f_" + sfx + " == gemv_" + sfx + "_1 per expert row" + tag, cnt, bad));
    }
}

// down + combine: (E rows, F cols)
void downCase(Ctx& c, QType t, int E, int F, bool shared) {
    const int ti = static_cast<int>(t);
    const std::string sfx = wk::typeSuffix(t);
    const std::string tag = " E=" + std::to_string(E) + " F=" + std::to_string(F) + (shared ? "" : " no shared");
    const HostMat W = randomMat(c, t, kR * E, F);
    Buf dW(W.data);
    std::vector<int> ids;
    std::vector<float> w;
    routing(c, kN, kR, kK, ids, w);
    const std::vector<float> act = c.randn(static_cast<std::size_t>(kN) * kK * F), x0 = c.randn(static_cast<std::size_t>(kN) * E),
                             ysh = c.randn(static_cast<std::size_t>(kN) * E), sg = c.randu(kN, 0.f, 1.f);
    auto run = [&](int t0, int n) {
        auto sl = [&](const auto& v, std::size_t per) {
            using V = std::decay_t<decltype(v)>;
            return V(v.begin() + static_cast<std::ptrdiff_t>(t0 * per), v.begin() + static_cast<std::ptrdiff_t>((t0 + n) * per));
        };
        Buf did(sl(ids, kK)), dw(sl(w, kK)), da(sl(act, static_cast<std::size_t>(kK) * F)), dx(sl(x0, E)), dys(sl(ysh, E)), dsg(sl(sg, 1));
        hip::launch(c.k.moe_down_f[ti], {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dW.p(), W.row_bytes, E, da.p(), did.p(),
                    dw.p(), dsg.p(), shared ? dys.p() : DevPtr{0}, dx.p(), F, kK);
        c.sync();
        return dx.down<float>(static_cast<std::size_t>(n) * E);
    };
    const std::vector<float> full = run(0, kN);
    {
        const std::vector<float> wf = ref::dequantRows(W, 0, W.nrows);
        std::vector<float> got;
        std::vector<double> rf, sc;
        for (int tok = 0; tok < kN; ++tok)
            for (int r = 0; r < E; ++r) {
                double acc = 0, a = 0;
                for (int k = 0; k < kK; ++k) {
                    const int p = tok * kK + k;
                    const float* wr = wf.data() + (static_cast<std::size_t>(ids[static_cast<std::size_t>(p)]) * E + r) * F;
                    double s = 0, sa = 0;
                    for (int i = 0; i < F; ++i) {
                        const double pr = static_cast<double>(wr[i]) * act[static_cast<std::size_t>(p) * F + i];
                        s += pr;
                        sa += std::fabs(pr);
                    }
                    acc += static_cast<double>(w[static_cast<std::size_t>(p)]) * s;
                    a += static_cast<double>(w[static_cast<std::size_t>(p)]) * sa;
                }
                const std::size_t o = static_cast<std::size_t>(tok) * E + r;
                const double sh = shared ? static_cast<double>(sg[static_cast<std::size_t>(tok)]) * ysh[o] : 0.0;
                got.push_back(full[o]);
                rf.push_back(x0[o] + acc + sh);
                sc.push_back(std::fabs(x0[o]) + a + std::fabs(sh));
            }
        c.rep.add(cmpTol("moe_down_f_" + sfx + " (+ shared, residual) vs CPU" + tag, got, rf, sc, 1e-5, 1e-7));
    }
    {
        std::size_t bad = 0, cnt = 0;
        auto cmpWin = [&](int t0, int n) {
            const std::vector<float> o = run(t0, n);
            bad += std::memcmp(o.data(), full.data() + static_cast<std::size_t>(t0) * E, o.size() * 4) != 0;
            ++cnt;
        };
        for (int n = 1; n <= 16; ++n) cmpWin(0, n);
        cmpWin(9, 7);
        cmpWin(16, 16);
        cmpWin(31, 1);
        c.rep.add(inv("moe_down_f_" + sfx + ": n = 1..16 / offset windows == 32-token rows" + tag, cnt, bad));
    }
    // K = 1, w = 1, no shared expert, x = 0: == gemv_<T>_1 on the expert's rows
    if (t != QType::mxfp4 && c.k.gemv1[ti] && shared) {
        std::size_t bad = 0, cnt = 0;
        Buf da(act), y(static_cast<std::size_t>(E) * 4), x(static_cast<std::size_t>(E) * 4);
        const std::vector<float> one{1.f};
        Buf d1(one);
        for (int p = 0; p < kN * kK; p += 29) {
            const int e = ids[static_cast<std::size_t>(p)];
            const DevPtr ap = da.p() + static_cast<std::uint64_t>(p) * F * 4;
            Buf did(std::vector<int>{e});
            x.zero();
            hip::launch(c.k.moe_down_f[ti], {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p(), W.row_bytes, E, ap, did.p(), d1.p(), d1.p(), DevPtr{0},
                        x.p(), F, 1);
            hip::launch(c.k.gemv1[ti], {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p() + static_cast<std::uint64_t>(e) * E * W.row_bytes, W.row_bytes,
                        ap, y.p(), F, E, F, E, 0);
            c.sync();
            const auto gx = x.down<float>(static_cast<std::size_t>(E)), gy = y.down<float>(static_cast<std::size_t>(E));
            for (int r = 0; r < E; ++r) {
                // 0 + v: only an exact -0 sum would differ in bits; compare values
                bad += gx[static_cast<std::size_t>(r)] != gy[static_cast<std::size_t>(r)];
                ++cnt;
            }
        }
        c.rep.add(inv("moe_down_f_" + sfx + " (K = 1, w = 1) == gemv_" + sfx + "_1" + tag, cnt, bad));
    }
}

}  // namespace

void testMoeF(Ctx& c) {
    c.rep.family = "moef";
    int ran = 0;
    for (const QType t : wk::kAllTypes) {
        const int ti = static_cast<int>(t);
        const std::string sfx = wk::typeSuffix(t);
        if (!c.k.moe_gu_f[ti] || !c.k.moe_down_f[ti]) {
            c.rep.skip("moef", std::string("moe_gu_f_") + sfx, "no precise MoE expert kernel for this type");
            continue;
        }
        ++ran;
        const int cmin = minCols(t);
        guCase(c, t, cmin, 7);
        guCase(c, t, 768, 100);
        if (!c.quick) guCase(c, t, 2048, 64);
        downCase(c, t, 7, cmin, true);
        downCase(c, t, 100, 768, true);
        downCase(c, t, 1, 256, false);
        if (!c.quick) downCase(c, t, 300, 512, true);
    }
    if (ran == 0) {
        Result r;
        r.name = "moe_gu_f / moe_down_f: no kernel bound";
        r.pass = false;
        c.rep.add(r);
    }
}

}  // namespace kt
