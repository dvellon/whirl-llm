// whirl-kernel-test: the Radeon 8060S grouped expert GEMMs (optional kernels, gfx1151):
//   * moe_tiles bn=64 == moe_route bn=64 tiles (same expert counts -> same list);
//   * gemm_moegu_<T> (gate + up + SwiGLU, 64-token tiles) == gemm_moe_<T> gate,
//     gemm_moe_<T> up, moe_act_f16 (f16 bits, every row);
//   * gemm_moe32r_<T> (down, 32-token tiles, row block fastest) == gemm_moe32_<T>;
//   at 200 and 1500 tokens with a skewed routing (one hot expert: tiles of 1..64
//   tokens, several per expert), so every padded-fragment skip case is hit.
// SPDX-License-Identifier: Apache-2.0

#include "cpu_ref.h"

namespace kt {

namespace {
unsigned cdivu(std::size_t a, unsigned b) { return static_cast<unsigned>((a + b - 1) / b); }
}  // namespace

void moeGuChecks(Ctx& c, const HostMat& gate, const HostMat& up, const HostMat& down, int R, int K) {
    const int tg = static_cast<int>(gate.type), td = static_cast<int>(down.type);
    const std::string sg = wk::typeSuffix(gate.type), sd = wk::typeSuffix(down.type);
    if (!c.k.moe_tiles || !c.k.gemm_moegu[tg] || !c.k.gemm_moe32r[td] || gate.type != up.type) {
        c.rep.skip(c.rep.family, "gemm_moegu", "8060S grouped expert GEMMs not in this code object");
        return;
    }
    const int E = gate.ncols, F = down.ncols;
    Buf dgate(gate.data), dup(up.data), ddown(down.data);
    for (const int n : {200, 1500}) {
        const int pairs = n * K;
        std::vector<int> ids(static_cast<std::size_t>(pairs));
        for (int p = 0; p < pairs; ++p)
            ids[static_cast<std::size_t>(p)] = p % 7 == 0 ? 3 : static_cast<int>(c.rng() % static_cast<unsigned>(R));
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        Buf dids(ids), dh(h), perm(static_cast<std::size_t>(pairs) * 4), inv(static_cast<std::size_t>(pairs) * 4);
        const int mt32 = (pairs + 31) / 32 + R, mt64 = (pairs + 63) / 64 + R;
        Buf t32(static_cast<std::size_t>(mt32) * 16), n32(4), t64(static_cast<std::size_t>(mt64) * 16), n64(4), t64r(static_cast<std::size_t>(mt64) * 16),
            n64r(4), p2(static_cast<std::size_t>(pairs) * 4), i2(static_cast<std::size_t>(pairs) * 4);
        hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, 32, perm.p(), inv.p(), t32.p(), n32.p());
        hip::launch(c.k.moe_tiles, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, 64, t64.p(), n64.p());
        hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, 64, p2.p(), i2.p(), t64r.p(), n64r.p());
        Buf xg(static_cast<std::size_t>(pairs) * E * 2);
        hip::launch(c.k.moe_gather_f16, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), xg.p(), E, K);
        c.sync();
        const std::string tag = " (" + std::to_string(n) + " tokens)";
        {
            const int a = n64.down<int>(1)[0], b = n64r.down<int>(1)[0];
            const auto ta = t64.down<int>(static_cast<std::size_t>(a) * 4), tb = t64r.down<int>(static_cast<std::size_t>(b) * 4);
            Result r;
            r.name = "moe_tiles bn=64 == moe_route bn=64 tiles" + tag;
            r.n = static_cast<std::size_t>(b);
            r.mismatches = a != b ? 1 : static_cast<std::size_t>(ta != tb);
            r.pass = r.mismatches == 0;
            c.rep.add(r);
        }
        // reference: gemm_moe gate, gemm_moe up (64-token tiles), moe_act_f16
        Buf yg(static_cast<std::size_t>(pairs) * F * 4), yu(static_cast<std::size_t>(pairs) * F * 4), aref(static_cast<std::size_t>(pairs) * F * 2),
            anew(static_cast<std::size_t>(pairs) * F * 2);
        anew.fill(0xff);
        hip::launch(c.k.gemm_moe[tg], {static_cast<unsigned>(mt64), cdivu(F, wk::kMoeBm), 1}, {256, 1, 1}, 0, c.s, dgate.p(), gate.row_bytes, F, xg.p(),
                    yg.p(), E, t64.p(), n64.p());
        hip::launch(c.k.gemm_moe[tg], {static_cast<unsigned>(mt64), cdivu(F, wk::kMoeBm), 1}, {256, 1, 1}, 0, c.s, dup.p(), up.row_bytes, F, xg.p(),
                    yu.p(), E, t64.p(), n64.p());
        hip::launch(c.k.moe_act_f16, {cdivu(static_cast<std::size_t>(pairs) * F, 256), 1, 1}, {256, 1, 1}, 0, c.s, yg.p(), yu.p(), aref.p(), pairs * F);
        hip::launch(c.k.gemm_moegu[tg], {static_cast<unsigned>(mt64) * cdivu(F, 64), 1, 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), gate.row_bytes, F,
                    xg.p(), anew.p(), E, t64.p(), n64.p());
        c.sync();
        const auto ar = aref.down<std::uint16_t>(static_cast<std::size_t>(pairs) * F), an = anew.down<std::uint16_t>(static_cast<std::size_t>(pairs) * F);
        c.rep.add(cmpExact("gemm_moegu_" + sg + " == gemm_moe_ gate, up + moe_act_f16" + tag, an, ar, Kind::invariant));
        // down: gemm_moe32r == gemm_moe32 on the same activations / tiles
        Buf yd0(static_cast<std::size_t>(pairs) * E * 4), yd1(static_cast<std::size_t>(pairs) * E * 4);
        yd1.fill(0xff);
        hip::launch(c.k.gemm_moe32[td], {static_cast<unsigned>(mt32), cdivu(E, wk::kMoeBm), 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E, aref.p(),
                    yd0.p(), F, t32.p(), n32.p());
        hip::launch(c.k.gemm_moe32r[td], {static_cast<unsigned>(mt32) * cdivu(E, wk::kMoeBm), 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E,
                    aref.p(), yd1.p(), F, t32.p(), n32.p());
        c.sync();
        c.rep.add(cmpExact("gemm_moe32r_" + sd + " == gemm_moe32_" + sd + tag, yd1.down<float>(static_cast<std::size_t>(pairs) * E),
                           yd0.down<float>(static_cast<std::size_t>(pairs) * E), Kind::invariant));
    }
}

// FIX-K97 (FR-K97 P2-3): the gfx1201 precise MoE prefill GEMMs. kernels/moe_mxfp4.hip says every value
// gemmr_moe* stores is bitwise the old path's:
//   gemmr_moegu[32]_mxfp4 == gemm_moe[32]_mxfp4 gate, gemm_moe[32]_mxfp4 up, moe_act_f16 (f16 bits)
//   gemmr_moedn[32]_mxfp4 == gemm_moe[32]_mxfp4 (down, f32)
// checked on every routed position (outputs pre-filled with NaN bytes, so a missed store fails), token
// tiles 64 and 32, skewed routing (tiles of 1..bn tokens, empty 16-token fragments), a 3-token batch, and
// synthetic shapes with rows_e % 8 != 0 and partial row blocks (gate/up rows 196, down rows 300).
void gemmrChecks(Ctx& c, const HostMat& gate, const HostMat& up, const HostMat& down, int R, int K) {
    const int tg = static_cast<int>(QType::mxfp4);
    if (!c.k.gemmr_moegu || !c.k.gemmr_moegu32 || !c.k.gemmr_moedn || !c.k.gemmr_moedn32 || !c.k.gemm_moe[tg] || !c.k.gemm_moe32[tg]) {
        c.rep.skip(c.rep.family, "gemmr_moe*", "gfx1201 precise MoE prefill GEMMs not in this code object");
        return;
    }
    auto one = [&](const HostMat& g, const HostMat& u, const HostMat& d, int nR, int n, const std::string& what) {
        const int E = g.ncols, F = g.nrows / nR, Ed = d.nrows / nR, Fd = d.ncols;
        const int pairs = n * K;
        std::vector<int> ids(static_cast<std::size_t>(pairs));
        for (int p = 0; p < pairs; ++p) ids[static_cast<std::size_t>(p)] = p % 7 == 0 ? 3 % nR : static_cast<int>(c.rng() % static_cast<unsigned>(nR));
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        Buf dg(g.data), du(u.data), dd(d.data), dids(ids), dh(h);
        for (const int bn : {64, 32}) {
            const std::string tag = " (" + what + ", " + std::to_string(n) + " tokens, tile " + std::to_string(bn) + ")";
            const int mt = (pairs + bn - 1) / bn + nR;
            Buf perm(static_cast<std::size_t>(pairs) * 4), inv(static_cast<std::size_t>(pairs) * 4), tiles(static_cast<std::size_t>(mt) * 16), nt(4);
            hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, nR, bn, perm.p(), inv.p(), tiles.p(), nt.p());
            Buf xg(static_cast<std::size_t>(pairs) * E * 2);
            hip::launch(c.k.moe_gather_f16, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), xg.p(), E, K);
            const auto gm = bn == 32 ? c.k.gemm_moe32[tg] : c.k.gemm_moe[tg];
            Buf yg(static_cast<std::size_t>(pairs) * F * 4), yu(static_cast<std::size_t>(pairs) * F * 4), aref(static_cast<std::size_t>(pairs) * F * 2),
                anew(static_cast<std::size_t>(pairs) * F * 2);
            anew.fill(0xff);
            hip::launch(gm, {static_cast<unsigned>(mt), cdivu(F, wk::kMoeBm), 1}, {256, 1, 1}, 0, c.s, dg.p(), g.row_bytes, F, xg.p(), yg.p(), E, tiles.p(),
                        nt.p());
            hip::launch(gm, {static_cast<unsigned>(mt), cdivu(F, wk::kMoeBm), 1}, {256, 1, 1}, 0, c.s, du.p(), u.row_bytes, F, xg.p(), yu.p(), E, tiles.p(),
                        nt.p());
            hip::launch(c.k.moe_act_f16, {cdivu(static_cast<std::size_t>(pairs) * F, 256), 1, 1}, {256, 1, 1}, 0, c.s, yg.p(), yu.p(), aref.p(), pairs * F);
            hip::launch(bn == 32 ? c.k.gemmr_moegu32 : c.k.gemmr_moegu, {static_cast<unsigned>(mt) * cdivu(F, 64), 1, 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(),
                        g.row_bytes, F, xg.p(), anew.p(), E, tiles.p(), nt.p());
            c.sync();
            const auto ar = aref.down<std::uint16_t>(static_cast<std::size_t>(pairs) * F);
            c.rep.add(cmpExact(std::string("gemmr_moegu") + (bn == 32 ? "32" : "") + "_mxfp4 == gemm_moe gate, up + moe_act_f16" + tag,
                               anew.down<std::uint16_t>(ar.size()), ar, Kind::invariant));
            // down: the routed activations of the gate/up step when the shapes chain, else random f16 rows
            Buf arnd;
            DevPtr ain = aref.p();
            if (Fd != F) {
                const std::vector<float> a = c.randn(static_cast<std::size_t>(pairs) * Fd);
                std::vector<std::uint16_t> a16(a.size());
                for (std::size_t i = 0; i < a.size(); ++i) a16[i] = f2h(a[i]);
                arnd = Buf(a16);
                ain = arnd.p();
            }
            Buf yd0(static_cast<std::size_t>(pairs) * Ed * 4), yd1(static_cast<std::size_t>(pairs) * Ed * 4);
            yd0.fill(0xff);
            yd1.fill(0xff);
            hip::launch(bn == 32 ? c.k.gemm_moe32[tg] : c.k.gemm_moe[tg], {static_cast<unsigned>(mt), cdivu(Ed, wk::kMoeBm), 1}, {256, 1, 1}, 0, c.s, dd.p(),
                        d.row_bytes, Ed, ain, yd0.p(), Fd, tiles.p(), nt.p());
            hip::launch(bn == 32 ? c.k.gemmr_moedn32 : c.k.gemmr_moedn, {static_cast<unsigned>(mt) * cdivu(Ed, 128), 1, 1}, {256, 1, 1}, 0, c.s, dd.p(),
                        DevPtr{0}, d.row_bytes, Ed, ain, yd1.p(), Fd, tiles.p(), nt.p());
            c.sync();
            c.rep.add(cmpExact(std::string("gemmr_moedn") + (bn == 32 ? "32" : "") + "_mxfp4 == gemm_moe" + (bn == 32 ? "32" : "") + "_mxfp4 (down)" + tag,
                               yd1.down<float>(static_cast<std::size_t>(pairs) * Ed), yd0.down<float>(static_cast<std::size_t>(pairs) * Ed),
                               Kind::invariant));
        }
    };
    const std::string src = gate.name.rfind("random", 0) == 0 ? "synthetic" : "model";
    for (const int n : {3, 200, 1500}) one(gate, up, down, R, n, src + " E=" + std::to_string(gate.ncols) + " F=" + std::to_string(down.ncols));
    // odd shapes: gate/up rows_e 196 (% 8 = 4, partial 64-row block), down rows_e 300 (% 8 = 4, partial 128-row block)
    const int nR = 16;
    const HostMat g2 = randomMat(c, QType::mxfp4, nR * 196, 512), u2 = randomMat(c, QType::mxfp4, nR * 196, 512),
                  d2 = randomMat(c, QType::mxfp4, nR * 300, 256);
    for (const int n : {3, 200}) one(g2, u2, d2, nR, n, "synthetic rows 196 / 300");
}

}  // namespace kt
