// whirl-kernel-test families "ws4" (bitwise) and "ws4bench" (timing, on request): gfx1151 attn_ws4
// (4-wave pipelined split-K decode attention, kernels/gfx1151/attn_ws4.hip) against attn_wsplit1.
// SPDX-License-Identifier: Apache-2.0
// ws4: for f16 / q8 / q4 KV (synthetic pool, shuffled pages, two sequences with their own page
// tables through kvbase), one query per group (the gfx1151 host's only kind): the partials
// (m, l, acc of every split) of attn_ws4[_q8|_q4] are bitwise
// those of attn_wsplit1[_q8|_q4] for contexts 1 .. 256k incl. chunk / page / split boundaries
// and random positions, with the MTP draft window, for a smaller grid (split rule independent of
// gridDim.y), and over 200 repeated launches with changing batches.
// ws4bench: kernel time and KV GB/s at 32k / 64k / 96k / 128k (WHIRL_KT_BENCH_L) for
// attn_wsplit1 / attn_ws4 / attn_dq4 per format, plus attn_combine.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "gfx1151/attn_ws4_index.h"
#include "kt.h"

namespace kt {
namespace {

constexpr int kH = 24, kNkv = 4, kD = 256, kQStr = 2 * kD, kNs = 64;
const float kSc = 1.0f / 16.0f;

enum class F { f16, q8, q4 };
const char* fname(F f) { return f == F::f16 ? "f16" : f == F::q8 ? "q8" : "q4"; }
const char* fsuf(F f) { return f == F::f16 ? "" : f == F::q8 ? "_q8" : "_q4"; }

struct XorShift {
    std::uint64_t s;
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

// synthetic paged pool of `pages` pages; ptab: two sequences (tables [0, pages), [pages, 2 pages)),
// each a different shuffle of the pool
struct Pool {
    F f;
    int pages;
    Buf k, v, ks, vs, ptab, kvbase;
    Pool(Ctx& c, F fmt, int np) : f(fmt), pages(np) {
        const std::size_t rows = static_cast<std::size_t>(np) * wk::kKvPage, nel = rows * kNkv * kD;
        XorShift x{c.rng() | 1};
        if (f == F::f16) {
            // f16 values: random sign, exponent 2^-6 .. 2^1, random mantissa (no inf / nan)
            std::vector<std::uint16_t> h(nel);
            for (int which = 0; which < 2; ++which) {
                for (std::size_t i = 0; i < nel; i += 4) {
                    const std::uint64_t r = x.next();
                    for (int j = 0; j < 4; ++j) {
                        const std::uint32_t b = static_cast<std::uint32_t>(r >> (16 * j));
                        const std::uint32_t e = 9 + (b >> 10) % 8;
                        h[i + j] = static_cast<std::uint16_t>((b & 0x8000u) | (e << 10) | (b & 0x3ffu));
                    }
                }
                (which ? v : k) = Buf(h);
            }
        } else {
            const std::size_t nb = f == F::q8 ? nel : nel / 2;
            std::vector<std::uint8_t> b(nb);
            for (int which = 0; which < 2; ++which) {
                for (std::size_t i = 0; i < nb; i += 8) {
                    std::uint64_t r = x.next();
                    for (int j = 0; j < 8 && i + j < nb; ++j) {
                        std::uint8_t y = static_cast<std::uint8_t>(r >> (8 * j));
                        if (f == F::q8 && y == 0x80) y = 0x81;  // int8 -127 .. 127 as kv_st32 stores
                        b[i + j] = y;
                    }
                }
                (which ? v : k) = Buf(b);
            }
            // scales 2^-9 .. 2^-6 (f16, random mantissa)
            std::vector<std::uint16_t> s(nel / 32);
            for (int which = 0; which < 2; ++which) {
                for (auto& y : s) {
                    const std::uint32_t r = static_cast<std::uint32_t>(x.next());
                    y = static_cast<std::uint16_t>(((6 + r % 4) << 10) | ((r >> 8) & 0x3ffu));
                }
                (which ? vs : ks) = Buf(s);
            }
        }
        std::vector<int> pt(static_cast<std::size_t>(2 * np));
        std::iota(pt.begin(), pt.begin() + np, 0);
        std::iota(pt.begin() + np, pt.end(), 0);
        std::shuffle(pt.begin(), pt.begin() + np, c.rng);
        std::shuffle(pt.begin() + np, pt.end(), c.rng);
        ptab = Buf(pt);
    }
    wk::KvArgs args(DevPtr kvb, int win = 0) const {
        wk::KvArgs a;
        a.k = k.p();
        a.v = v.p();
        a.ks = f == F::f16 ? 0 : ks.p();
        a.vs = f == F::f16 ? 0 : vs.p();
        a.ptab = ptab.p();
        a.kvbase = kvb;
        a.tab0 = 0;
        a.win = win;
        return a;
    }
};

struct Partials {
    std::vector<float> ml, acc;
};

// launch fn over rows (one per group): positions pos[r], page table offset tbs[r]
Partials run(Ctx& c, hip::Function fn, const Pool& p, const Buf& dq, const std::vector<int>& pos, const std::vector<int>& tbs, int ns, int win) {
    const int n = static_cast<int>(pos.size());
    Buf dpos(pos), dkvb(tbs);
    Buf ml(static_cast<std::size_t>(n) * ns * kH * 2 * 4), acc(static_cast<std::size_t>(n) * ns * kH * kD * 4);
    ml.fill(0x7f);  // unwritten entries: NaN pattern (compared only where written)
    acc.fill(0x7f);
    wk::AwGroups g{};
    for (int r = 0; r < n; ++r) {
        g.first[r] = r;
        g.count[r] = 1;
    }
    hip::launch(fn, {kNkv, static_cast<unsigned>(ns), static_cast<unsigned>(n)}, {128, 1, 1}, 0, c.s, dq.p(), p.args(dkvb.p(), win), ml.p(),
                acc.p(), kH, kNkv, kQStr, dpos.p(), kSc, DevPtr{0}, g);
    c.sync();
    return {ml.down<float>(static_cast<std::size_t>(n) * ns * kH * 2), acc.down<float>(static_cast<std::size_t>(n) * ns * kH * kD)};
}

// bitwise compare per (row, split, head): ml both; acc where the split carries keys (m > -inf).
// nsa / nsb: grid splits of a / b (layouts), compared over the first min(nsa, nsb) splits; splits
// beyond in the larger grid must be empty
void compare(Ctx& c, const std::string& name, const Partials& a, int nsa, const Partials& b, int nsb, int n) {
    std::size_t bad = 0, cnt = 0;
    const int nsm = std::max(nsa, nsb);
    for (int r = 0; r < n; ++r)
        for (int s = 0; s < nsm; ++s)
            for (int h = 0; h < kH; ++h) {
                const bool ina = s < nsa, inb = s < nsb;
                const float* ma = ina ? &a.ml[((static_cast<std::size_t>(r) * nsa + s) * kH + h) * 2] : nullptr;
                const float* mb = inb ? &b.ml[((static_cast<std::size_t>(r) * nsb + s) * kH + h) * 2] : nullptr;
                if (!ina || !inb) {
                    const float* m1 = ina ? ma : mb;
                    cnt += 1;
                    bad += !(std::isinf(m1[0]) && m1[0] < 0);
                    continue;
                }
                cnt += 2;
                const bool d = std::memcmp(ma, mb, 8) != 0;
                bad += d;
                if (!d && !(std::isinf(ma[0]) && ma[0] < 0)) {
                    const float* aa = &a.acc[((static_cast<std::size_t>(r) * nsa + s) * kH + h) * kD];
                    const float* ab = &b.acc[((static_cast<std::size_t>(r) * nsb + s) * kH + h) * kD];
                    cnt += kD;
                    for (int i = 0; i < kD; ++i) bad += std::memcmp(aa + i, ab + i, 4) != 0;
                }
            }
    Result res;
    res.name = name;
    res.kind = Kind::invariant;
    res.n = cnt;
    res.mismatches = bad;
    res.pass = bad == 0 && cnt > 0;
    c.rep.add(res);
}

}  // namespace

// host checks of the shared index header (the properties the T1 emulator verified): every
// (key, dim) of a chunk staged exactly once for K (per format) and V, Vt / Sc addresses distinct
// and in range, Vt rows 16-byte aligned, and the split ranges tiling [0, n_pos) (with and without
// the draft window) for 64 splits
void ws4IndexChecks(Ctx& c) {
    std::size_t bad = 0, n = 0;
    for (int F = 0; F < 3; ++F) {
        std::vector<int> hit(ws4::kKey * ws4::kD, 0);
        for (int tv = 0; tv < 128; ++tv)
            for (int p = 0; p < ws4::k_pieces(F); ++p)
                for (int i = 0; i < ws4::k_len(F); ++i) hit[static_cast<std::size_t>(ws4::k_key(F, tv, p) * ws4::kD + ws4::k_d0(F, tv) + i)] += 1;
        for (int h : hit) bad += h != 1;
        n += hit.size();
    }
    {
        std::vector<int> hit(ws4::kKey * ws4::kD, 0), vt(ws4::kVtSize, 0);
        for (int tv = 0; tv < 128; ++tv)
            for (int r = 0; r < 8; ++r)
                for (int i = 0; i < 8; ++i) {
                    const int key = ws4::v_kb(tv) * 8 + r, d = ws4::v_d8(tv) + i;
                    hit[static_cast<std::size_t>(key * ws4::kD + d)] += 1;
                    const int a = ws4::vt_addr(d, ws4::vt_col(ws4::v_kb(tv), r));
                    if (a < 0 || a >= ws4::kVtSize) bad += 1;
                    else vt[static_cast<std::size_t>(a)] += 1;
                }
        for (int h : hit) bad += h != 1;
        for (int d = 0; d < ws4::kD; ++d) bad += (ws4::vt_addr(d, 0) % 8) != 0;
        int used = 0;
        for (int h : vt) {
            bad += h > 1;
            used += h;
        }
        bad += used != ws4::kKey * ws4::kD;
        n += hit.size() + ws4::kD + 1;
    }
    {
        std::vector<int> sc(ws4::kScSize, 0);
        for (int kt = 0; kt < 2; ++kt)
            for (int lane = 0; lane < 32; ++lane)
                for (int v = 0; v < 8; ++v) sc[static_cast<std::size_t>(ws4::sc_addr(kt, lane, v))] += 1;
        for (int h : sc) bad += h != 1;
        n += sc.size();
    }
    for (int win : {0, (1024 / 64) | (2 << 16), (512 / 64)}) {
        for (int np : {1, 31, 64, 65, 256, 257, 2047, 2048, 4095, 4097, 9000, 65613, 131072, 262144}) {
            std::vector<int> cov(static_cast<std::size_t>(np), 0);
            const ws4::Range s0 = ws4::block_range(np, 0, ws4::kNs, win);
            const bool active = win != 0 && s0.b0 == 0 && s0.b1 == ws4::kSink && np > ws4::kSink && ws4::block_range(np, 1, ws4::kNs, win).b0 > ws4::kSink;
            for (int s = 0; s < ws4::kNs; ++s) {
                const ws4::Range r = ws4::block_range(np, s, ws4::kNs, win);
                bad += (r.b0 % 32) != 0;
                for (int k = r.b0; k < r.b1; ++k) cov[static_cast<std::size_t>(k)] += 1;
                for (int k = r.b0; k < r.b1; k += 32) bad += (k / wk::kKvPage) != (std::min(k + 31, r.b1 - 1) / wk::kKvPage);  // no page straddle
            }
            const int w0 = active ? ws4::block_range(np, 1, ws4::kNs, win).b0 : 0;
            for (int k = 0; k < np; ++k) bad += cov[static_cast<std::size_t>(k)] != ((active && k >= ws4::kSink && k < w0) ? 0 : 1);
            n += static_cast<std::size_t>(np);
        }
    }
    Result r;
    r.name = "attn_ws4 index header: staging coverage, Vt / Sc addresses, split tiling";
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    c.rep.add(r);
}

void testAttnWs4(Ctx& c) {
    c.rep.family = "ws4";
    ws4IndexChecks(c);
    if (!c.fnOpt("attn_ws4")) {
        c.rep.skip("ws4", "attn_ws4", "kernel not in this code object");
        return;
    }
    const int maxpos = c.quick ? 70000 : 262144;
    const int pages = (maxpos + wk::kKvPage - 1) / wk::kKvPage;
    std::vector<int> ctxs = {1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 4095, 4096, 4097, 8191, 32767, 32768, 32769, 65613};
    if (!c.quick)
        for (int x : {98304, 131071, 131072, 131073, 196613, 262143, 262144}) ctxs.push_back(x);
    for (int i = 0; i < 24; ++i) ctxs.push_back(1 + static_cast<int>(c.rng() % static_cast<std::uint64_t>(maxpos)));
    const std::vector<float> q = c.randn(static_cast<std::size_t>(8) * kH * kQStr);
    Buf dq(q);
    // WHIRL_KT_WS4_BASE: a code object built before attn_wsplit_impl was split into the shared
    // aw_* device functions; its attn_wsplit1* must give the same bits as this build's
    std::vector<char> base_img;
    hip::Module base_mod;
    bool have_base = false;
    if (const char* e = std::getenv("WHIRL_KT_WS4_BASE")) {
        if (std::FILE* fp = std::fopen(e, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            base_img.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(base_img.data(), 1, base_img.size(), fp) != base_img.size()) base_img.clear();
            std::fclose(fp);
        }
        if (!base_img.empty()) {
            base_mod = hip::Module::loadData(base_img.data(), c.mod.arch());
            have_base = true;
        }
    }
    for (F f : {F::f16, F::q8, F::q4}) {
        const std::string tag = std::string(" [kv ") + fname(f) + "]";
        const auto fold = c.fnOpt(std::string("attn_wsplit1") + fsuf(f));
        if (!fold) {
            c.rep.skip("ws4", std::string("attn_ws4") + fsuf(f), "attn_wsplit1 reference not in this code object");
            continue;
        }
        Pool p(c, f, pages);
        std::vector<std::pair<std::string, hip::Function>> news;
        {
            const std::string nm = std::string("attn_ws4") + fsuf(f);
            if (auto fn = c.fnOpt(nm)) news.push_back({nm, fn});
        }
        if (news.empty()) continue;
        if (have_base)
            if (auto fb = base_mod.getFunctionOpt((std::string("attn_wsplit1") + fsuf(f)).c_str()))
                news.push_back({std::string("base attn_wsplit1") + fsuf(f), fb});
        // contexts in batches of 8 rows (one per group), alternating the two sequences' page tables
        for (std::size_t i = 0; i < ctxs.size(); i += 8) {
            std::vector<int> pos, tbs;
            std::string lst;
            for (std::size_t j = i; j < std::min(ctxs.size(), i + 8); ++j) {
                pos.push_back(ctxs[j] - 1);
                tbs.push_back((j % 2) ? pages : 0);
                lst += (lst.empty() ? "" : ",") + std::to_string(ctxs[j]);
            }
            const Partials ref = run(c, fold, p, dq, pos, tbs, kNs, 0);
            for (const auto& [nm, fn] : news) {
                const Partials got = run(c, fn, p, dq, pos, tbs, kNs, 0);
                compare(c, nm + " == attn_wsplit1" + fsuf(f) + " (ctx " + lst + ")" + tag, got, kNs, ref, kNs, static_cast<int>(pos.size()));
            }
        }
        const auto& [nm0, fn0] = news[0];
        // MTP draft window (W 1024, threshold 2048; and a window reaching the sink)
        {
            const std::vector<int> pos = {2047, 4099, 9000, 65612}, tbs = {0, pages, 0, pages};
            for (int win : {(1024 / 64) | (2 << 16), (512 / 64) | (0 << 16), (131072 / 64) | (1 << 16)}) {
                const Partials ref = run(c, fold, p, dq, pos, tbs, kNs, win);
                const Partials got = run(c, fn0, p, dq, pos, tbs, kNs, win);
                compare(c, nm0 + " == attn_wsplit1" + fsuf(f) + " (draft window arg " + std::to_string(win) + ")" + tag, got, kNs, ref, kNs, 4);
            }
        }
        // split rule independent of the grid (gridDim.y 16 vs 64, contexts <= 1024)
        {
            const std::vector<int> pos = {0, 63, 64, 500, 1023}, tbs = {0, pages, 0, pages, 0};
            const Partials a = run(c, fn0, p, dq, pos, tbs, 16, 0);
            const Partials b = run(c, fn0, p, dq, pos, tbs, kNs, 0);
            compare(c, nm0 + " grid 16 splits == grid 64 splits (ctx <= 1024)" + tag, a, 16, b, kNs, 5);
        }
        // 200 launches with changing batches (1 .. 8 rows): each == its first run
        {
            std::vector<std::vector<int>> batches;
            std::vector<Partials> firsts;
            for (int b = 0; b < 8; ++b) {
                std::vector<int> pos;
                for (int r = 0; r <= b; ++r) pos.push_back(static_cast<int>(c.rng() % static_cast<std::uint64_t>(std::min(maxpos, 140000))));
                batches.push_back(pos);
            }
            std::size_t bad = 0, cnt = 0;
            for (int it = 0; it < 200; ++it) {
                const int b = it < 8 ? it : static_cast<int>(c.rng() % 8);
                const std::vector<int>& pos = batches[static_cast<std::size_t>(b)];
                const std::vector<int> tbs(pos.size(), 0);
                const Partials got = run(c, fn0, p, dq, pos, tbs, kNs, 0);
                if (it < 8) {
                    firsts.push_back(got);
                    continue;
                }
                const Partials& f0 = firsts[static_cast<std::size_t>(b)];
                cnt += got.ml.size() + got.acc.size();
                bad += std::memcmp(got.ml.data(), f0.ml.data(), got.ml.size() * 4) != 0;
                for (std::size_t i = 0; i < got.ml.size(); i += 2) {
                    if (std::isinf(got.ml[i]) && got.ml[i] < 0) continue;
                    bad += std::memcmp(&got.acc[i / 2 * kD], &f0.acc[i / 2 * kD], kD * 4) != 0;
                }
            }
            Result res;
            res.name = nm0 + " 192 repeated launches (batches of 1..8 rows) == first" + tag;
            res.kind = Kind::invariant;
            res.n = cnt;
            res.mismatches = bad;
            res.pass = bad == 0;
            c.rep.add(res);
        }
    }
}

void benchAttnWs4(Ctx& c) {
    c.rep.family = "ws4bench";
    std::vector<int> lens = {32768, 65536, 98304, 131072};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_L")) lens = {std::atoi(e)};
    const int iters = std::getenv("WHIRL_KT_BENCH_ITERS") ? std::atoi(std::getenv("WHIRL_KT_BENCH_ITERS")) : 20;
    const std::vector<float> q = c.randn(static_cast<std::size_t>(kH) * kQStr);
    Buf dq(q);
    const int maxl = *std::max_element(lens.begin(), lens.end());
    // WHIRL_KT_BENCH_HSACO: attn_ws4* from this code object instead (kernel development)
    std::vector<char> image;
    hip::Module xmod;
    bool have_x = false;
    if (const char* e = std::getenv("WHIRL_KT_BENCH_HSACO")) {
        if (std::FILE* fp = std::fopen(e, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            image.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(image.data(), 1, image.size(), fp) != image.size()) image.clear();
            std::fclose(fp);
        }
        if (!image.empty()) {
            xmod = hip::Module::loadData(image.data(), c.mod.arch());
            have_x = true;
            std::printf("   attn_ws4* from %s\n", e);
        }
    }
    std::vector<F> fmts = {F::q8, F::f16, F::q4};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_FMT")) {
        const std::string fs = e;
        fmts = {fs == "f16" ? F::f16 : fs == "q4" ? F::q4 : F::q8};
    }
    for (F f : fmts) {
        Pool p(c, f, (maxl + wk::kKvPage - 1) / wk::kKvPage);
        const double row_bytes = f == F::f16 ? 2.0 * kD * 2 : f == F::q8 ? 2.0 * (kD + kD / 32 * 2) : 2.0 * (kD / 2 + kD / 32 * 2);
        std::vector<std::pair<std::string, unsigned>> names = {{std::string("attn_wsplit1") + fsuf(f), 128},
                                                               {std::string("attn_ws4") + fsuf(f), 128}};
        if (const char* e = std::getenv("WHIRL_KT_BENCH_KERNELS")) names.push_back({e, 128});
        if (f == F::q4) names.push_back({"attn_dq4", 256});
        for (const int L : lens) {
            Buf dpos(std::vector<int>{L - 1}), dkvb(std::vector<int>{0});
            Buf ml(static_cast<std::size_t>(kNs) * kH * 2 * 4), acc(static_cast<std::size_t>(kNs) * kH * kD * 4), out(kH * kD * 4);
            wk::AwGroups g{};
            g.first[0] = 0;
            g.count[0] = 1;
            for (const auto& [nm, threads] : names) {
                const auto fn = have_x && nm.rfind("attn_ws4", 0) == 0 ? xmod.getFunctionOpt(nm.c_str()) : c.fnOpt(nm);
                if (!fn) {
                    std::printf("   %s: not in this code object\n", nm.c_str());
                    continue;
                }
                auto launch = [&]() {
                    hip::launch(fn, {kNkv, kNs, 1}, {threads, 1, 1}, 0, c.s, dq.p(), p.args(dkvb.p()), ml.p(), acc.p(), kH, kNkv, kQStr, dpos.p(), kSc,
                                DevPtr{0}, g);
                };
                for (int i = 0; i < 3; ++i) launch();
                c.sync();
                hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                hip::eventRecord(e0, c.s);
                for (int i = 0; i < iters; ++i) launch();
                hip::eventRecord(e1, c.s);
                hip::eventSync(e1);
                const double us = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
                hip::eventDestroy(e0);
                hip::eventDestroy(e1);
                const double bytes = static_cast<double>(L) * kNkv * row_bytes;
                std::printf("   ws4bench %-4s L %6d %-20s %8.1f us  %6.1f GB/s\n", fname(f), L, nm.c_str(), us, bytes / us / 1e3);
                std::fflush(stdout);
            }
            if (f == F::q8 && L == lens[0]) {  // attn_combine (24 heads, 64 splits), for the fusion estimate
                auto launch = [&]() {
                    hip::launch(c.k.attn_combine, {kH, 1, 1}, {256, 1, 1}, 0, c.s, ml.p(), acc.p(), dq.p(), out.p(), kH, kD, kQStr, kNs);
                };
                for (int i = 0; i < 3; ++i) launch();
                c.sync();
                hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                hip::eventRecord(e0, c.s);
                for (int i = 0; i < iters; ++i) launch();
                hip::eventRecord(e1, c.s);
                hip::eventSync(e1);
                std::printf("   ws4bench attn_combine (24 heads, 64 splits): %8.1f us per launch (back to back)\n",
                            1000.0 * hip::eventElapsedMs(e0, e1) / iters);
                hip::eventDestroy(e0);
                hip::eventDestroy(e1);
            }
        }
    }
}

}  // namespace kt
