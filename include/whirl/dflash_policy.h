// SPDX-License-Identifier: Apache-2.0
// DFlash2 speculative decoding (`--draft dflash`): the host-side policy shared by whirl chat /
// bench (src/cli/main.cpp), whirl-server (src/server/server_main.cpp, engine.cpp) and the drafter
// forward (src/model/dflash.cpp). Header-only and GPU-free so tests/unit covers it.
//
// Design (DFLASH-1, owner-approved 2026-10-05, v0.2.0):
//   * `--draft mtp|dflash` picks the draft source; default mtp. dflash needs the drafter GGUF in
//     WHIRL_DRAFT_MODEL that fits the target (DFlashConfig::validate, headers only) and the dflash_*
//     kernels in the GPU's code object; otherwise a warning and the MTP drafts (choose()).
//   * dflash mode does not load the MTP block (no weights, no MTP KV layer): the KV pool is sized
//     after the drafter's weights, capture and rings, and the startup log prints its capacity.
//   * n-gram drafts stay on in both modes; per cycle the cost model picks n-gram or DFlash2.
//   * DFlash2 drafts are literal drafts (like n-gram): the target verifies them with the same
//     bitwise verify as plain decoding, and acceptance is EXACT in every numerics mode
//     (precise / balance / fast). relaxacc (fast) relaxes MTP drafts only (relax_accept.h), so with
//     --draft dflash the output equals plain greedy decoding of the same numerics mode; for
//     sampling the reply is a function of (target logits, seed) only. relaxApplies() is the rule.
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::dflash {

enum class Source { mtp, dflash };

// "mtp" / "dflash"; anything else: nullopt (the caller reports the usage error)
inline std::optional<Source> parseSource(std::string_view s) {
    if (s == "mtp") return Source::mtp;
    if (s == "dflash") return Source::dflash;
    return std::nullopt;
}

inline const char* sourceName(Source s) { return s == Source::dflash ? "dflash" : "mtp"; }

// The draft source actually used, and the warning to print when a dflash request falls back.
struct Choice {
    Source source = Source::mtp;
    std::string warning;  // empty: nothing to report
};

// requested: --draft; draft_model: WHIRL_DRAFT_MODEL; incompatible: dflashIncompatible() of the
// two headers ("" = fits; ignored without a draft model); kernels_ok: the code object has every
// dflash_* kernel (false only for an old / foreign WHIRL_CODE_OBJECT once both kernel sets carry
// them; pass true before the module is loaded and check again after).
inline Choice choose(Source requested, const std::optional<std::string>& draft_model, std::string_view incompatible, bool kernels_ok) {
    Choice c;
    if (requested == Source::mtp) {
        if (draft_model) c.warning = "WHIRL_DRAFT_MODEL is set but --draft is mtp: the DFlash2 drafter is not loaded";
        return c;
    }
    if (!draft_model) {
        c.warning = "--draft dflash needs the DFlash2 drafter GGUF in WHIRL_DRAFT_MODEL: using MTP drafts";
        return c;
    }
    if (!incompatible.empty()) {
        c.warning = "--draft dflash: " + *draft_model + " does not fit this model (" + std::string(incompatible) + "): using MTP drafts";
        return c;
    }
    if (!kernels_ok) {
        c.warning = "--draft dflash: the GPU code object has no DFlash2 kernels (dflash_*): using MTP drafts";
        return c;
    }
    c.source = Source::dflash;
    return c;
}

// relaxacc (fast mode) relaxes MTP drafts only; n-gram and DFlash2 drafts keep exact acceptance.
inline bool relaxApplies(Source s, bool relax_on) { return relax_on && s == Source::mtp; }

// Settings from the environment (WHIRL_DFLASH_PMIN / _NMIN / _WARM / _CAPTURE_ROWS).
struct Params {
    float p_min = 0;                           // walk stops at a token with probability below p_min
    std::uint32_t n_min = 0;                   // fewer drafts than n_min: none
    std::uint32_t warm = 64;                   // server: rows after a restore before drafting again
    std::optional<std::uint32_t> capture_rows; // default: the drafter's window (2048 without one)
};

namespace detail {
inline bool parseU32(const std::string& s, std::uint32_t& out) {
    if (s.empty() || s.size() > 10) return false;
    std::uint64_t v = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') return false;
        v = v * 10 + static_cast<std::uint64_t>(ch - '0');
    }
    if (v > 0xffffffffull) return false;
    out = static_cast<std::uint32_t>(v);
    return true;
}
inline bool parseProb(const std::string& s, float& out) {
    // decimal fraction in [0, 1]: digits, at most one '.', no sign / exponent
    if (s.empty() || s.size() > 16) return false;
    double v = 0, scale = 0;
    bool digit = false;
    for (char ch : s) {
        if (ch == '.') {
            if (scale != 0) return false;
            scale = 1;
            continue;
        }
        if (ch < '0' || ch > '9') return false;
        digit = true;
        if (scale == 0) v = v * 10 + (ch - '0');
        else v += (ch - '0') * (scale /= 10);
    }
    if (!digit || v > 1.0) return false;
    out = static_cast<float>(v);
    return true;
}
}  // namespace detail

// env(name) -> std::optional<std::string> for the name without the WHIRL_ prefix ("DFLASH_PMIN").
// An invalid value keeps the default and adds a warning.
template <class Env>
Params parseParams(Env&& env, std::vector<std::string>& warnings) {
    Params p;
    if (auto v = env("DFLASH_PMIN")) {
        if (!detail::parseProb(*v, p.p_min)) warnings.push_back("WHIRL_DFLASH_PMIN=" + *v + " ignored (a probability 0..1)");
    }
    if (auto v = env("DFLASH_NMIN")) {
        if (!detail::parseU32(*v, p.n_min)) warnings.push_back("WHIRL_DFLASH_NMIN=" + *v + " ignored (a draft count)");
    }
    if (auto v = env("DFLASH_WARM")) {
        if (!detail::parseU32(*v, p.warm)) warnings.push_back("WHIRL_DFLASH_WARM=" + *v + " ignored (a row count)");
    }
    if (auto v = env("DFLASH_CAPTURE_ROWS")) {
        std::uint32_t r = 0;
        if (detail::parseU32(*v, r) && r > 0) p.capture_rows = r;
        else warnings.push_back("WHIRL_DFLASH_CAPTURE_ROWS=" + *v + " ignored (a row count > 0)");
    }
    return p;
}

// Capture rows per target forward: requested (or the drafter's window, 2048 without one), at most
// the forward's batch (Model::enableCaptureLayers clamps the same way).
inline std::uint32_t captureRows(const Params& p, std::uint32_t window, std::uint32_t max_batch) {
    const std::uint32_t want = p.capture_rows ? *p.capture_rows : (window > 0 ? window : 2048u);
    return std::min(want, max_batch);
}

// Ring rows of a slot's drafter context: the drafter's window, 2048 without one.
inline std::uint32_t ringRows(std::uint32_t window) { return window > 0 ? window : 2048u; }

// Positions p0 .. p0 + m - 1 go to ring rows p % ring: [start, start + first) then [0, second).
struct RingPieces {
    std::uint32_t start = 0, first = 0, second = 0;
};
inline RingPieces ringPieces(std::uint32_t p0, std::uint32_t m, std::uint32_t ring) {
    RingPieces r;
    if (ring == 0 || m == 0) return r;
    m = std::min(m, ring);
    r.start = p0 % ring;
    r.first = std::min(m, ring - r.start);
    r.second = m - r.first;
    return r;
}

// DFlash2 drafts per slot in a verify of `slots` slots: block - 1, at most the literal-draft
// limit, the snapshot sets per slot and the verify rows per slot minus the slot's own row; the
// unfused decode path snapshots row 0 only (1 draft).
inline std::uint32_t draftLimit(std::uint32_t block, std::uint32_t max_literal, std::uint32_t snap_cap, std::uint32_t vrows,
                                std::uint32_t slots, bool fused) {
    if (block < 2 || slots == 0) return 0;
    const std::uint32_t per_rows = vrows / slots > 0 ? vrows / slots - 1 : 0u;
    std::uint32_t lim = std::min({block - 1, max_literal, snap_cap / slots, per_rows});
    if (!fused) lim = std::min(lim, 1u);
    return lim;
}

// The drafter context of one server slot: positions lo .. hi - 1 form one unbroken run.
struct Range {
    std::uint32_t lo = 0, hi = 0;
};

// Drafts with DFlash2 once the run ends at pos (the next position) and is at least
// min(pos, warm) rows long (after a restore the earlier rows have no features).
inline bool ready(const Range& r, std::uint32_t pos, std::uint32_t warm) { return r.hi == pos && pos - r.lo >= std::min(pos, warm); }

// One ingest after a forward / verify of run_rows rows whose last capture_n rows were captured:
// rows r0 .. r0 + n - 1 of it are positions pos0 .. of the slot.
struct Ingest {
    bool any = false;            // some of these rows were captured
    std::uint32_t cap_row = 0;   // first capture row to ingest
    std::uint32_t n = 0;         // rows to ingest
    std::uint32_t pos = 0;       // position of cap_row
    std::uint32_t gap = 0;       // positions without features (counted, the run restarts)
};

// Plans the ingest and updates the slot's range (hi = pos0 + n whatever was captured).
inline Ingest planIngest(Range& r, std::uint32_t capture_n, std::uint32_t run_rows, std::uint32_t r0, std::uint32_t n, std::uint32_t pos0) {
    Ingest g;
    if (n == 0) return g;
    const std::uint32_t cn = std::min(capture_n, run_rows);
    const std::uint32_t c0 = run_rows - cn;  // first captured row
    const std::uint32_t a = std::max(r0, c0), b = r0 + n;
    if (a >= b) {
        g.gap = n;  // nothing of this slot's rows in the capture
        return g;
    }
    const std::uint32_t p = pos0 + (a - r0);
    if (p > r.hi) {
        g.gap = p - r.hi;
        r.lo = p;
    } else if (p < r.lo) {
        r.lo = p;
    }
    g.any = true;
    g.cap_row = a - c0;
    g.n = b - a;
    g.pos = p;
    r.hi = pos0 + n;
    return g;
}

}  // namespace whirl::dflash
