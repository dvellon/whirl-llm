// whirl-kernel-test: runs the WHIRL HIP kernels on a GPU and compares them
// with C++ CPU references and with each other (bitwise invariances).
// SPDX-License-Identifier: Apache-2.0
//
// usage: whirl-kernel-test [--device SPEC] [--only fam,fam] [--quick] [-v] [--require-all]
//                          [--q4 GGUF] [--mx GGUF] [--moe GGUF] [--moemx GGUF]
// Exit: 0 pass, 1 a failing check (or, with --require-all, a kernel missing that the arch requires),
// 2 usage error: unknown --only family, a given model path that does not exist, 0 checks run, or
// --require-all on an arch without a required-kernel list (FIX-TEST-A, FR-TEST P3-3 / P3-4).
// families: quant gemv gemm attn gdn moe moemx misc

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>

#include "cpu_ref.h"
#include "whirl/common.h"

namespace {

// default model paths come from the environment (unset: synthetic data is used)
std::string envOr(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

int usage() {
    std::printf(
        "usage: whirl-kernel-test [--device SPEC] [--only quant,gemv,gemm,attn,attndq4,gdn,moe,moemx,moef,misc,dflash,gemvh] [--quick] [-v]\n"
        "                         [--q4 GGUF] [--mx GGUF] [--moe GGUF] [--moemx GGUF]\n"
        "  --require-all  a kernel missing from the code object fails unless the arch lacks it by design\n"
        "  --device  index, gfx arch, name substring, r9700 or 8060s (default: WHIRL_DEVICE, else the\n"
        "            first R9700, else the first GPU this build has kernels for)\n"
        "  models default to WHIRL_TEST_Q4 (else WHIRL_TEST_GGUF) / WHIRL_TEST_MX / WHIRL_TEST_MOE /\n"
        "  WHIRL_TEST_MOEMX; an unset model or '-' means synthetic data is used\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string device = envOr("WHIRL_DEVICE");
    std::set<std::string> only;
    bool require_all = false;
    kt::Ctx c;
    c.models = {envOr("WHIRL_TEST_Q4"), envOr("WHIRL_TEST_MX"), envOr("WHIRL_TEST_MOE"), envOr("WHIRL_TEST_MOEMX")};
    if (c.models.q4.empty()) c.models.q4 = envOr("WHIRL_TEST_GGUF");
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        try {
            if (a == "--device") device = next();
            else if (a == "--only") {
                std::stringstream ss(next());
                std::string f;
                while (std::getline(ss, f, ',')) only.insert(f);
            } else if (a == "--quick") c.quick = true;
            else if (a == "--require-all") require_all = true;
            else if (a == "-v" || a == "--verbose") c.rep.verbose = true;
            else if (a == "--q4") c.models.q4 = next();
            else if (a == "--mx") c.models.mx = next();
            else if (a == "--moe") c.models.moe = next();
            else if (a == "--moemx") c.models.moemx = next();
            else return usage();
        } catch (const std::exception& e) {
            std::printf("%s\n", e.what());
            return usage();
        }
    }
    for (std::string* p : {&c.models.q4, &c.models.mx, &c.models.moe, &c.models.moemx})
        if (*p == "-") p->clear();
    // a model path that was given but does not exist is an error, not "synthetic data" (FR-TEST P3-4)
    for (const std::string* p : {&c.models.q4, &c.models.mx, &c.models.moe, &c.models.moemx}) {
        std::error_code ec;
        if (!p->empty() && !std::filesystem::exists(std::filesystem::path(whirl::widen(*p)), ec)) {
            std::printf("whirl-kernel-test: model file not found: %s\n", p->c_str());
            return 2;
        }
    }
    const char* fam_names[] = {"quant", "gemv", "gemm", "attn", "attndq4", "gdn", "moe", "moemx", "moef", "misc", "attnbench",
                               "gemvh", "gemvhbench", "dflash"};
    for (const std::string& f : only) {
        bool known = false;
        for (const char* n : fam_names) known = known || f == n;
        if (!known) {
            std::printf("whirl-kernel-test: --only: no family named \"%s\"\n", f.c_str());
            return usage();
        }
    }
    std::string arch;

    try {
        namespace hip = whirl::hip;
        auto have = [](const hip::DeviceInfo& d) {
            for (const hip::EmbeddedObject& o : hip::embeddedObjects())
                if (d.gcn_arch.rfind(o.arch, 0) == 0) return true;
            return false;
        };
        int dev = -1;
        if (!device.empty()) {
            dev = hip::findDevice(device == "8060s" ? "gfx1151" : device == "r9700" ? "gfx1201" : device);
            if (dev < 0) throw std::runtime_error("no device matches '" + device + "'");
        } else {
            for (const hip::DeviceInfo& d : hip::listDevices())
                if (dev < 0 && d.gcn_arch.rfind("gfx1201", 0) == 0 && have(d)) dev = d.index;
            for (const hip::DeviceInfo& d : hip::listDevices())
                if (dev < 0 && have(d)) dev = d.index;
            if (dev < 0) dev = 0;
        }
        hip::setDevice(dev);
        const auto info = hip::describeDevice(dev);
        std::printf("device %d: %s (%s)\n", dev, info.name.c_str(), info.gcn_arch.c_str());
        arch = info.gcn_arch;
        if (require_all && !kt::requiredKernelAbsentOk(arch, "").has_value()) {
            std::printf("whirl-kernel-test: --require-all: no required-kernel list for %s\n", arch.c_str());
            return 2;
        }
        c.mod = hip::Module::loadEmbedded();
        c.k = whirl::kernels::KernelTable::load(c.mod, whirl::kernels::KvFormat::f16);
        // gfx1151 stores packed int8 scale words; the CPU references follow the code object
        kt::ref::setXdSum(c.k.caps.xd_sum);
        const auto& cp = c.k.caps;
        std::printf("caps: fp8_gemm %d, kv_q8v %d, kv_q8h %d, gemvw %d, gdn_replay %d, mrope %d, xd_sum %d, attn_group1 %d, draft_window %d\n",
                    cp.fp8_gemm, cp.kv_q8v, cp.kv_q8h, cp.gemvw, cp.gdn_replay, cp.mrope, cp.xd_sum, cp.attn_group1, cp.draft_window);
        c.s = whirl::hip::streamCreate(false);  // blocking: ordered with the null-stream copies / memsets
    } catch (const std::exception& e) {
        std::printf("setup failed: %s\n", e.what());
        return 1;
    }

    struct Fam {
        const char* name;
        void (*run)(kt::Ctx&);
    };
    const Fam fams[] = {{"quant", kt::testQuant}, {"gemv", kt::testGemv}, {"gemm", kt::testGemm},
                        {"attn", kt::testAttn},   {"attndq4", kt::testAttnDq4},   {"gdn", kt::testGdn},   {"moe", kt::testMoe},
                        {"moemx", kt::testMoeMx}, {"moef", kt::testMoeF}, {"misc", kt::testMisc},   {"attnbench", kt::benchAttn},
                        {"gemvh", kt::testGemvh}, {"gemvhbench", kt::benchGemvh}, {"dflash", kt::testDflash}};
    for (const Fam& f : fams) {
        if (!only.empty() && !only.count(f.name)) continue;
        if (only.empty() && (std::string(f.name) == "attnbench" || std::string(f.name) == "gemvhbench")) continue;  // timing only, on request
        const auto t0 = std::chrono::steady_clock::now();
        std::printf("== %s\n", f.name);
        std::fflush(stdout);
        try {
            f.run(c);
        } catch (const std::exception& e) {
            kt::Result r;
            r.family = f.name;
            r.name = std::string("exception: ") + e.what();
            r.pass = false;
            c.rep.add(r);
        }
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("   (%s: %.1f s)\n", f.name, s);
    }
    int fails = c.rep.summary();
    whirl::hip::streamDestroy(c.s);
    if (c.rep.checks() == 0) {
        std::printf("whirl-kernel-test: 0 checks ran: refusing to report success\n");
        return 2;
    }
    // FR-TEST P3-4: a renamed or unbuilt kernel used to turn its family green as a skip
    int missing = 0;
    for (const auto& s : c.rep.skips()) {
        if (!kt::skipIsMissingKernel(s.why)) continue;
        const bool ok = kt::requiredKernelAbsentOk(arch, s.key()).value_or(false);
        if (require_all && !ok) {
            std::printf("MISSING (required on %s): %s: %s\n", arch.c_str(), s.key().c_str(), s.why.c_str());
            ++missing;
        }
    }
    if (require_all) std::printf("--require-all: %d required kernel(s) missing\n", missing);
    fails += missing;
    return fails ? 1 : 0;
}
