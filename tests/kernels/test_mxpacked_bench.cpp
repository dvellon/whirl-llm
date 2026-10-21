// Timing-only benchmark for the gfx1151 packed MXFP4 NT9 kernel at every
// matrix shape in Swift-1.5 Qwen3.8-27B.  It is opt-in because event timing
// and hundreds of MiB of cache-thrashing copies are inappropriate in tests.
// SPDX-License-Identifier: Apache-2.0

#include "kt.h"

#include <algorithm>
#include <cstdio>

namespace kt {

namespace {

double timeLaunches(Ctx& c, const auto& launch, int copies) {
    auto timed = [&](int iters) {
        hip::Event a = hip::eventCreate(true), b = hip::eventCreate(true);
        hip::eventRecord(a, c.s);
        for (int i = 0; i < iters; ++i) launch(i);
        hip::eventRecord(b, c.s);
        hip::eventSync(b);
        const double us = 1000.0 * hip::eventElapsedMs(a, b) / iters;
        hip::eventDestroy(a);
        hip::eventDestroy(b);
        return us;
    };
    for (int i = 0; i < copies; ++i) launch(i);
    c.sync();
    const int base = std::max(copies * 2, 20);
    double us = timed(base);
    const int win = static_cast<int>(std::min(20000.0, std::max<double>(base, 150000.0 / us)));
    timed(std::max(base, win * 2 / 3));
    us = 1e30;
    for (int rep = 0; rep < 3; ++rep) us = std::min(us, timed(win));
    return us;
}

}  // namespace

void benchMxPacked(Ctx& c) {
    c.rep.family = "mxpackedbench";
    if (!c.k.pack_w16_mxfp4 || !c.k.gemvq_nt9r2w16p_mxfp4 || !c.k.gemvq_nt9r2w16p_mxfp4_g) {
        c.rep.skip("mxpackedbench", "kernels", "packed MXFP4 NT9 kernels unavailable");
        return;
    }
    struct Shape { int nrows, ncols, count; const char* what; };
    const Shape shapes[] = {
        {17408, 5120, 128, "gate/up"}, {5120, 17408, 64, "down"}, {10240, 5120, 48, "qkv"},
        {5120, 6144, 64, "attn/ssm out"}, {6144, 5120, 48, "attn gate"},
        {12288, 5120, 16, "attn q"}, {1024, 5120, 32, "attn k/v"},
    };
    constexpr int nt = 9;
    std::printf("   packed NT9 production shapes (cold-copy weight bandwidth)\n");
    for (const Shape& sh : shapes) {
        const HostMat m = randomMat(c, QType::mxfp4, sh.nrows, sh.ncols);
        const std::size_t packed_bytes = static_cast<std::size_t>(cdiv(sh.nrows, 16) * 16) * m.row_bytes;
        const int copies = std::max<int>(2, std::min<int>(48, static_cast<int>((384u << 20) / packed_bytes)));
        Buf canonical(m.data);
        std::vector<Buf> ws;
        ws.reserve(static_cast<std::size_t>(copies));
        for (int i = 0; i < copies; ++i) {
            ws.emplace_back(packed_bytes);
            hip::launch(c.k.pack_w16_mxfp4,
                        {cdiv(sh.nrows, 16) * static_cast<unsigned>(sh.ncols / 256), 1, 1}, {256, 1, 1}, 0, c.s,
                        canonical.p(), ws.back().p(), m.row_bytes, sh.nrows);
        }
        const std::vector<float> hx = c.randn(static_cast<std::size_t>(nt) * sh.ncols);
        Buf x(hx), xq(static_cast<std::size_t>(nt) * sh.ncols),
            xd(static_cast<std::size_t>(nt) * sh.ncols / 32 * 4),
            y(static_cast<std::size_t>(nt) * sh.nrows * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(nt) * sh.ncols, 256), 1, 1}, {256, 1, 1}, 0,
                    c.s, x.p(), xq.p(), xd.p(), nt * sh.ncols);
        c.sync();
        auto launch = [&](int i) {
            hip::launch(c.k.gemvq_nt9r2w16p_mxfp4, {cdiv(sh.nrows, 16), 1, 1}, {512, 1, 1}, 0, c.s,
                        ws[static_cast<std::size_t>(i % copies)].p(), m.row_bytes, xq.p(), xd.p(), y.p(),
                        sh.ncols, sh.nrows, 0, DevPtr{0});
        };
        const double us = timeLaunches(c, launch, copies);
        const double gbs = static_cast<double>(packed_bytes) / (us * 1e3);
        std::printf("   %5d x %-5d %-13s count %3d: %8.1f us  %6.1f GB/s  (%d copies)\n",
                    sh.nrows, sh.ncols, sh.what, sh.count, us, gbs, copies);
        std::fflush(stdout);
    }

    // Gate and up are dispatched together in production.  Time the grouped
    // entry with two independently packed matrices to expose launch/segment loss.
    {
        constexpr int rows = 17408, cols = 5120;
        const HostMat m = randomMat(c, QType::mxfp4, rows, cols);
        const std::size_t bytes = static_cast<std::size_t>(rows) * m.row_bytes;
        const int pairs = std::max<int>(2, std::min<int>(24, static_cast<int>((384u << 20) / (2 * bytes))));
        Buf canonical(m.data);
        std::vector<Buf> ws;
        ws.reserve(static_cast<std::size_t>(pairs * 2));
        for (int i = 0; i < pairs * 2; ++i) {
            ws.emplace_back(bytes);
            hip::launch(c.k.pack_w16_mxfp4, {cdiv(rows, 16) * static_cast<unsigned>(cols / 256), 1, 1}, {256, 1, 1}, 0,
                        c.s, canonical.p(), ws.back().p(), m.row_bytes, rows);
        }
        const std::vector<float> hx = c.randn(static_cast<std::size_t>(nt) * cols);
        Buf x(hx), xq(static_cast<std::size_t>(nt) * cols), xd(static_cast<std::size_t>(nt) * cols / 32 * 4),
            y0(static_cast<std::size_t>(nt) * rows * 4), y1(static_cast<std::size_t>(nt) * rows * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(nt) * cols, 256), 1, 1}, {256, 1, 1}, 0,
                    c.s, x.p(), xq.p(), xd.p(), nt * cols);
        c.sync();
        auto launch = [&](int i) {
            wk::GvArgs g{};
            g.w[0] = ws[static_cast<std::size_t>((i % pairs) * 2)].p();
            g.w[1] = ws[static_cast<std::size_t>((i % pairs) * 2 + 1)].p();
            g.y[0] = y0.p(); g.y[1] = y1.p();
            g.rb[0] = g.rb[1] = m.row_bytes;
            g.n[0] = g.n[1] = rows;
            g.b1 = static_cast<int>(cdiv(rows, 16));
            g.b2 = g.b1 * 2;
            hip::launch(c.k.gemvq_nt9r2w16p_mxfp4_g, {static_cast<unsigned>(g.b2), 1, 1}, {512, 1, 1}, 0, c.s,
                        g, xq.p(), xd.p(), cols, 0, DevPtr{0});
        };
        const double us = timeLaunches(c, launch, pairs);
        std::printf("   grouped gate+up 2 x %d x %d: %8.1f us  %6.1f GB/s  (%d pairs)\n",
                    rows, cols, us, static_cast<double>(2 * bytes) / (us * 1e3), pairs);
    }
    Result ok;
    ok.name = "packed NT9 production-shape timing completed";
    ok.pass = true;
    ok.n = 1;
    c.rep.add(ok);
}

}  // namespace kt
