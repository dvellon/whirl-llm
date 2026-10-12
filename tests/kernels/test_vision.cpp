// whirl-kernel-test family "vision": the vision encoder kernels
// (kernels/vision/vision.hip, their own code object per architecture) vs C++
// CPU references (VIS-8S: gfx1151 port; the same checks run on gfx1201).
// SPDX-License-Identifier: Apache-2.0
//
// Inputs are f16-exact where the kernel reads f16, so the references see the
// same operands; tolerances cover f32 accumulation order, the f16 rounding of
// outputs / softmax probabilities and the device transcendental functions.

#include <algorithm>
#include <cmath>

#include "kt.h"
#include "whirl/vision.h"

namespace kt {

namespace {

constexpr int VD = 72, VDP = 80, VH = 16, VE = 1152;

std::vector<std::uint16_t> toH(const std::vector<float>& v) {
    std::vector<std::uint16_t> h(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]);
    return h;
}
std::vector<float> fromH(const std::vector<std::uint16_t>& h) {
    std::vector<float> v(h.size());
    for (std::size_t i = 0; i < h.size(); ++i) v[i] = h2f(h[i]);
    return v;
}
// random values rounded to f16 (as f32)
std::vector<float> randh(Ctx& c, std::size_t n, float sigma) {
    std::vector<float> v = c.randn(n, sigma);
    for (auto& x : v) x = roundh(x);
    return v;
}
void tokXY(int t, int pw, int& px, int& py) {
    const int blk = t >> 2, q = t & 3, bw = pw >> 1;
    px = 2 * (blk % bw) + (q & 1);
    py = 2 * (blk / bw) + (q >> 1);
}

void testGemmV(Ctx& c, const hip::Module& m) {
    const hip::Function f = m.getFunction("vis_gemm");
    struct S {
        int ncols, nrows, n;
    };
    // patch (K 1536), qkv (3456 rows), ff tails (nrows % 128 != 0, % 8 != 0), token tails
    const S shapes[] = {{1536, 1152, 300}, {1152, 3456, 256}, {4320, 1152, 77}, {64, 200, 513}, {32, 37, 19}, {4608, 260, 33}};
    for (const S& s : shapes) {
        const std::vector<float> w = randh(c, static_cast<std::size_t>(s.nrows) * s.ncols, 0.05f);
        const std::vector<float> x = randh(c, static_cast<std::size_t>(s.n) * s.ncols, 1.0f);
        const std::vector<float> y0 = c.randn(static_cast<std::size_t>(s.n) * s.nrows);
        Buf dw(toH(w)), dx(toH(x));
        for (int acc = 0; acc < 2; ++acc) {
            Buf dy(y0);
            hip::launch(f, {static_cast<unsigned>((s.n + 255) / 256), static_cast<unsigned>((s.nrows + 127) / 128), 1}, {256, 1, 1}, 0, c.s,
                        dw.p(), dx.p(), dy.p(), s.ncols, s.nrows, s.n, acc);
            c.sync();
            const auto got = dy.down<float>(y0.size());
            std::vector<double> ref(y0.size()), sc(y0.size());
            for (int t = 0; t < s.n; ++t)
                for (int r = 0; r < s.nrows; ++r) {
                    double a = 0, b = 0;
                    const float* wr = w.data() + static_cast<std::size_t>(r) * s.ncols;
                    const float* xr = x.data() + static_cast<std::size_t>(t) * s.ncols;
                    for (int k = 0; k < s.ncols; ++k) {
                        a += static_cast<double>(wr[k]) * xr[k];
                        b += std::fabs(static_cast<double>(wr[k]) * xr[k]);
                    }
                    const std::size_t i = static_cast<std::size_t>(t) * s.nrows + r;
                    ref[i] = acc ? a + y0[i] : a;
                    sc[i] = acc ? b + std::fabs(y0[i]) : b;
                }
            c.rep.add(cmpTol("vis_gemm K=" + std::to_string(s.ncols) + " rows=" + std::to_string(s.nrows) + " n=" + std::to_string(s.n) +
                                 (acc ? " (accumulate)" : ""),
                             got, ref, sc, 2e-6, 1e-6));
        }
    }
}

void testAttnV(Ctx& c, const hip::Module& m) {
    const hip::Function f = m.getFunction("vis_attn2");
    for (int N : {64, 100, 256, 1000, 1500}) {
        const int Np = (N + 127) / 128 * 128;
        // Q, K: [VH][Np][VDP], Vt: [VH][VDP][Np]; pads (dims >= 72, tokens >= N) zero
        std::vector<float> q(static_cast<std::size_t>(VH) * Np * VDP, 0.f), k(q.size(), 0.f), vt(q.size(), 0.f);
        const std::vector<float> rq = randh(c, q.size(), 1.0f), rk = randh(c, q.size(), 1.0f), rv = randh(c, q.size(), 1.0f);
        for (int h = 0; h < VH; ++h)
            for (int t = 0; t < N; ++t)
                for (int d = 0; d < VD; ++d) {
                    const std::size_t i = (static_cast<std::size_t>(h) * Np + t) * VDP + d;
                    q[i] = rq[i];
                    k[i] = rk[i];
                    vt[(static_cast<std::size_t>(h) * VDP + d) * Np + t] = rv[i];
                }
        Buf dq(toH(q)), dk(toH(k)), dv(toH(vt));
        std::vector<std::uint16_t> o0(static_cast<std::size_t>(N) * VE, 0x7e00);  // NaN fill: every output written
        Buf dout(o0);
        const double scale = 1.0 / std::sqrt(static_cast<double>(VD));
        const float scale_log2 = static_cast<float>(scale * 1.4426950408889634);
        hip::launch(f, {static_cast<unsigned>(Np / 128), static_cast<unsigned>(VH), 1}, {128, 1, 1}, 0, c.s, dq.p(), dk.p(), dv.p(), dout.p(), N,
                    Np, scale_log2);
        c.sync();
        const std::vector<float> got = fromH(dout.down<std::uint16_t>(o0.size()));
        std::vector<double> ref(got.size()), sc(got.size());
        std::vector<double> s(static_cast<std::size_t>(N));
        for (int h = 0; h < VH; ++h)
            for (int i = 0; i < N; ++i) {
                const float* qi = q.data() + (static_cast<std::size_t>(h) * Np + i) * VDP;
                double mx = -1e300;
                for (int j = 0; j < N; ++j) {
                    const float* kj = k.data() + (static_cast<std::size_t>(h) * Np + j) * VDP;
                    double a = 0;
                    for (int d = 0; d < VD; ++d) a += static_cast<double>(qi[d]) * kj[d];
                    s[static_cast<std::size_t>(j)] = a * scale;
                    mx = std::max(mx, s[static_cast<std::size_t>(j)]);
                }
                double L = 0;
                for (int j = 0; j < N; ++j) {
                    s[static_cast<std::size_t>(j)] = std::exp(s[static_cast<std::size_t>(j)] - mx);
                    L += s[static_cast<std::size_t>(j)];
                }
                for (int d = 0; d < VD; ++d) {
                    double a = 0, b = 0;
                    for (int j = 0; j < N; ++j) {
                        const double vv = vt[(static_cast<std::size_t>(h) * VDP + d) * Np + j];
                        a += s[static_cast<std::size_t>(j)] * vv;
                        b += s[static_cast<std::size_t>(j)] * std::fabs(vv);
                    }
                    const std::size_t o = static_cast<std::size_t>(i) * VE + static_cast<std::size_t>(h) * VD + d;
                    ref[o] = a / L;
                    sc[o] = b / L + std::fabs(a / L);
                }
            }
        // f16 probabilities (2^-11 relative) and f16 output: 1.5e-3 of sum p|v| / L + |o|
        c.rep.add(cmpTol("vis_attn2 N=" + std::to_string(N) + " (16 heads, d 72)", got, ref, sc, 1.5e-3, 1e-4));
    }
}

void testQkvPrep(Ctx& c, const hip::Module& m) {
    const hip::Function f = m.getFunction("vis_qkv_prep2");
    const int pw = 12, ph = 10, N = pw * ph;  // 120 tokens: a partial 64-token tile
    const int Np = (N + 127) / 128 * 128;
    const float theta_scale = static_cast<float>(std::pow(10000.0, -2.0 / 36.0));
    const std::vector<float> T = c.randn(static_cast<std::size_t>(N) * 3 * VE), bias = c.randn(3 * VE, 0.1f);
    Buf dT(T), db(bias);
    // pre-filled with garbage: the kernel must write every pad
    const std::size_t qn = static_cast<std::size_t>(VH) * Np * VDP;
    std::vector<std::uint16_t> junk(qn, 0x3c00);
    Buf dq(junk), dk(junk), dv(junk);
    hip::launch(f, {static_cast<unsigned>(Np / 64), static_cast<unsigned>(VH), 1}, {256, 1, 1}, 0, c.s, dT.p(), db.p(), dq.p(), dk.p(), dv.p(), N, Np, pw,
                theta_scale);
    c.sync();
    const std::vector<float> gq = fromH(dq.down<std::uint16_t>(qn)), gk = fromH(dk.down<std::uint16_t>(qn)), gv = fromH(dv.down<std::uint16_t>(qn));
    std::vector<double> rq(qn, 0.0), rk(qn, 0.0), rv(qn, 0.0), sq(qn, 0.0), sk(qn, 0.0), sv(qn, 0.0);
    for (int t = 0; t < N; ++t) {
        int px, py;
        tokXY(t, pw, px, py);
        for (int h = 0; h < VH; ++h) {
            const float* tr = T.data() + static_cast<std::size_t>(t) * 3 * VE + h * VD;
            const float* br = bias.data() + h * VD;
            const std::size_t o = (static_cast<std::size_t>(h) * Np + t) * VDP;
            for (int j = 0; j < 36; ++j) {
                const double p = j < 18 ? py : px;
                const double th = p * std::pow(static_cast<double>(theta_scale), j < 18 ? j : j - 18);
                const double cs = std::cos(th), sn = std::sin(th);
                for (int w = 0; w < 2; ++w) {
                    const double x0 = static_cast<double>(tr[w * VE + j]) + br[w * VE + j];
                    const double x1 = static_cast<double>(tr[w * VE + j + 36]) + br[w * VE + j + 36];
                    std::vector<double>& r = w ? rk : rq;
                    std::vector<double>& sc = w ? sk : sq;
                    r[o + j] = x0 * cs - x1 * sn;
                    r[o + j + 36] = x0 * sn + x1 * cs;
                    sc[o + j] = sc[o + j + 36] = std::fabs(x0) + std::fabs(x1);
                }
            }
            for (int d = 0; d < VD; ++d) {
                const double x = static_cast<double>(tr[2 * VE + d]) + br[2 * VE + d];
                rv[(static_cast<std::size_t>(h) * VDP + d) * Np + t] = x;
                sv[(static_cast<std::size_t>(h) * VDP + d) * Np + t] = std::fabs(x);
            }
        }
    }
    // f16 outputs: 2^-11 relative + device sin / cos / pow; pads and tokens >= N exactly 0
    c.rep.add(cmpTol("vis_qkv_prep2 q (2D RoPE, pads)", gq, rq, sq, 1e-3, 1e-5));
    c.rep.add(cmpTol("vis_qkv_prep2 k (2D RoPE, pads)", gk, rk, sk, 1e-3, 1e-5));
    c.rep.add(cmpTol("vis_qkv_prep2 v^T (pads)", gv, rv, sv, 1e-3, 1e-5));
}

// LayerNorm reference over one f32 row
void lnRef(const double* x, const float* g, const float* b, double eps, double* out) {
    double mean = 0;
    for (int i = 0; i < VE; ++i) mean += x[i];
    mean /= VE;
    double var = 0;
    for (int i = 0; i < VE; ++i) var += (x[i] - mean) * (x[i] - mean);
    const double r = 1.0 / std::sqrt(var / VE + eps);
    for (int i = 0; i < VE; ++i) out[i] = (x[i] - mean) * r * g[i] + b[i];
}

void testElementwise(Ctx& c, const hip::Module& m) {
    const float eps = 1e-6f;
    // vis_patchify: exact (same float expression, one f16 rounding)
    {
        const hip::Function f = m.getFunction("vis_patchify");
        const int pw = 6, ph = 4, W = pw * 16, H = ph * 16, N = pw * ph;
        std::vector<std::uint8_t> img(static_cast<std::size_t>(W) * H * 3);
        for (auto& p : img) p = static_cast<std::uint8_t>(c.randi(1, 0, 255)[0]);
        const float mean[3] = {0.5f, 0.45f, 0.4f}, sd[3] = {0.5f, 0.27f, 0.26f};
        Buf di(img), dp(static_cast<std::size_t>(N) * 1536 * 2);
        hip::launch(f, {static_cast<unsigned>(N), 1, 1}, {256, 1, 1}, 0, c.s, di.p(), dp.p(), W, pw, mean[0], mean[1], mean[2], sd[0], sd[1], sd[2]);
        c.sync();
        const auto got = dp.down<std::uint16_t>(static_cast<std::size_t>(N) * 1536);
        std::vector<std::uint16_t> ref(got.size());
        for (int t = 0; t < N; ++t) {
            int px, py;
            tokXY(t, pw, px, py);
            for (int i = 0; i < 768; ++i) {
                const int ch = i / 256, ky = (i / 16) % 16, kx = i % 16;
                const float v = static_cast<float>(img[(static_cast<std::size_t>(py * 16 + ky) * W + px * 16 + kx) * 3 + ch]) / 255.0f;
                const std::uint16_t hv = f2h((v - mean[ch]) / sd[ch]);
                ref[static_cast<std::size_t>(t) * 1536 + i] = ref[static_cast<std::size_t>(t) * 1536 + i + 768] = hv;
            }
        }
        c.rep.add(cmpExact("vis_patchify (normalize, 2x2 merge order, both temporal halves)", got, ref));
    }
    // vis_pos_ln: interpolated position embedding (grid != n_side) and the identity case
    {
        const hip::Function f = m.getFunction("vis_pos_ln");
        const int n_side = 8;
        const std::vector<float> pos = c.randn(static_cast<std::size_t>(n_side) * n_side * VE), pb = c.randn(VE, 0.1f);
        const std::vector<float> g = c.randu(VE, 0.5f, 1.5f), b = c.randn(VE, 0.1f);
        Buf dpos(pos), dpb(pb), dg(g), dbb(b);
        for (int k = 0; k < 2; ++k) {
            const int pw = k ? 8 : 6, ph = k ? 8 : 10, N = pw * ph;
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(N) * VE);
            Buf dx(x0), dh(static_cast<std::size_t>(N) * VE * 2);
            hip::launch(f, {static_cast<unsigned>(N), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dpos.p(), dpb.p(), pw, ph, n_side, dg.p(), dbb.p(), dh.p(),
                        eps);
            c.sync();
            const auto gx = dx.down<float>(x0.size());
            const auto gh = fromH(dh.down<std::uint16_t>(x0.size()));
            std::vector<double> rx(x0.size()), rh(x0.size()), sx(x0.size());
            const double sf0 = (pw - 1.0) / (n_side - 1), sf1 = (ph - 1.0) / (n_side - 1);
            for (int t = 0; t < N; ++t) {
                int px, py;
                tokXY(t, pw, px, py);
                const double ys = py / sf1, xs = px / sf0;
                int y0 = static_cast<int>(std::floor(ys)), x0i = static_cast<int>(std::floor(xs));
                int y1 = std::clamp(y0 + 1, 0, n_side - 1), x1 = std::clamp(x0i + 1, 0, n_side - 1);
                y0 = std::clamp(y0, 0, n_side - 1);
                x0i = std::clamp(x0i, 0, n_side - 1);
                const double fy = std::clamp(ys - y0, 0.0, 1.0), fx = std::clamp(xs - x0i, 0.0, 1.0);
                const bool same = pw == n_side && ph == n_side;
                double* xr = rx.data() + static_cast<std::size_t>(t) * VE;
                for (int i = 0; i < VE; ++i) {
                    auto P = [&](int yy, int xx) { return static_cast<double>(pos[(static_cast<std::size_t>(yy) * n_side + xx) * VE + i]); };
                    const double pe = same ? P(py, px)
                                           : P(y0, x0i) * (1 - fx) * (1 - fy) + P(y0, x1) * fx * (1 - fy) + P(y1, x0i) * (1 - fx) * fy +
                                                 P(y1, x1) * fx * fy;
                    xr[i] = x0[static_cast<std::size_t>(t) * VE + i] + pb[static_cast<std::size_t>(i)] + pe;
                    sx[static_cast<std::size_t>(t) * VE + i] = std::fabs(x0[static_cast<std::size_t>(t) * VE + i]) + std::fabs(pb[static_cast<std::size_t>(i)]) + 4.0;
                }
                lnRef(xr, g.data(), b.data(), eps, rh.data() + static_cast<std::size_t>(t) * VE);
            }
            const std::string tag = k ? " (grid == n_side)" : " (bilinear 6x10 from 8x8)";
            c.rep.add(cmpTol("vis_pos_ln X" + tag, gx, rx, sx, 1e-6, 1e-6));
            c.rep.add(cmpTolRel("vis_pos_ln LN -> f16" + tag, gh, rh, 1.5e-3, 2e-3));
        }
    }
    // vis_resid_ln: residual + bias, then LN (and the g == null form: residual only)
    {
        const hip::Function f = m.getFunction("vis_resid_ln");
        const int N = 37;
        const std::vector<float> x0 = c.randn(static_cast<std::size_t>(N) * VE), T = c.randn(x0.size()), bias = c.randn(VE, 0.1f);
        const std::vector<float> g = c.randu(VE, 0.5f, 1.5f), b = c.randn(VE, 0.1f);
        Buf dT(T), dbias(bias), dg(g), dbb(b);
        for (int k = 0; k < 2; ++k) {
            Buf dx(x0), dh(x0.size() * 2);
            hip::launch(f, {static_cast<unsigned>(N), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dT.p(), dbias.p(), k ? DevPtr(0) : dg.p(), dbb.p(), dh.p(), eps);
            c.sync();
            const auto gx = dx.down<float>(x0.size());
            std::vector<float> rxf(x0.size());
            std::vector<double> rh(x0.size()), rxd(x0.size());
            for (std::size_t i = 0; i < x0.size(); ++i) {
                rxf[i] = x0[i] + (T[i] + bias[i % VE]);  // same float expression
                rxd[i] = rxf[i];
            }
            c.rep.add(cmpExact(std::string("vis_resid_ln X") + (k ? " (no LN)" : ""), gx, rxf));
            if (!k) {
                for (int t = 0; t < N; ++t) lnRef(rxd.data() + static_cast<std::size_t>(t) * VE, g.data(), b.data(), eps, rh.data() + static_cast<std::size_t>(t) * VE);
                c.rep.add(cmpTolRel("vis_resid_ln LN -> f16", fromH(dh.down<std::uint16_t>(x0.size())), rh, 1.5e-3, 2e-3));
            }
        }
    }
    // vis_gelu (tanh form) with K padding columns, vis_bias
    {
        const hip::Function f = m.getFunction("vis_gelu");
        const int N = 19, n_in = 300, n_out = 320;
        const std::vector<float> T = c.randn(static_cast<std::size_t>(N) * n_in, 2.0f), bias = c.randn(n_in, 0.2f);
        Buf dT(T), db(bias), dm(std::vector<std::uint16_t>(static_cast<std::size_t>(N) * n_out, 0x3c00));
        hip::launch(f, {static_cast<unsigned>(N), 1, 1}, {256, 1, 1}, 0, c.s, dT.p(), db.p(), dm.p(), n_in, n_out);
        c.sync();
        const auto got = fromH(dm.down<std::uint16_t>(static_cast<std::size_t>(N) * n_out));
        std::vector<double> ref(got.size(), 0.0);
        for (int t = 0; t < N; ++t)
            for (int i = 0; i < n_in; ++i) {
                const double x = static_cast<double>(T[static_cast<std::size_t>(t) * n_in + i]) + bias[static_cast<std::size_t>(i)];
                ref[static_cast<std::size_t>(t) * n_out + i] = 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
            }
        c.rep.add(cmpTolRel("vis_gelu (tanh) -> f16, zero K padding", got, ref, 1.5e-3, 1e-4));
        const hip::Function fb = m.getFunction("vis_bias");
        const int cols = 5120, rows = 3, n = cols * rows;
        const std::vector<float> Y = c.randn(static_cast<std::size_t>(n)), bb = c.randn(cols);
        Buf dy(Y), dbb(bb), dout(static_cast<std::size_t>(n) * 4);
        hip::launch(fb, {static_cast<unsigned>((n + 255) / 256), 1, 1}, {256, 1, 1}, 0, c.s, dy.p(), dbb.p(), dout.p(), cols, n);
        c.sync();
        std::vector<float> rb(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) rb[static_cast<std::size_t>(i)] = Y[static_cast<std::size_t>(i)] + bb[static_cast<std::size_t>(i % cols)];
        c.rep.add(cmpExact("vis_bias", dout.down<float>(static_cast<std::size_t>(n)), rb));
    }
}

}  // namespace

void testVision(Ctx& c) {
    const auto img = whirl::vision::kernelImage(c.mod.arch());
    if (img.empty()) {
        c.rep.skip("vision", "vision kernels", std::string("no vision code object for ") + hip::archName(c.mod.arch()));
        return;
    }
    const hip::Module m = hip::Module::loadData(img.data(), c.mod.arch());
    c.rep.family = "vision";
    testElementwise(c, m);
    testQkvPrep(c, m);
    testGemmV(c, m);
    testAttnV(c, m);
}

}  // namespace kt
