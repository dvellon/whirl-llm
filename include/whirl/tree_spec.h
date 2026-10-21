// Pure planning and acceptance rules for the single-slot beam-2 speculative
// prototype.  Device/KV state management lives in the server; this header is
// deliberately host-only and deterministic so the tree semantics can be
// tested without a GPU.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace whirl::tree_spec {

inline constexpr std::uint32_t max_depth = 8;
inline constexpr std::uint32_t branches = 2;

// The primary verify follows the same token chain as an ordinary linear
// speculative verify, so it may write its speculative suffix directly into
// the active slot's already-owned KV pages.  Only the alternate branch needs
// a private copy-on-write tail: if it wins, that owned tail replaces the
// active one.  Keeping this rule beside the pure tree plan makes ownership
// explicit and host-testable.
inline constexpr bool ownsKvTail(std::uint32_t branch) {
    if (branch >= branches) throw std::out_of_range("tree branch");
    return branch == 1;
}

struct Branch {
    std::array<std::uint32_t, max_depth> drafts{};
    std::uint32_t n = 0;
};

// Row layout intentionally consists of two ordinary linear verifies.  The
// primary branch may be deeper than the alternate: a short second root buys
// most top-2 recovery without paying two full target verifies every cycle.
struct Plan {
    std::uint32_t depth = 0;
    std::array<Branch, branches> branch{};

    bool valid() const {
        if (depth == 0 || depth > max_depth) return false;
        return branch[0].n == depth && branch[1].n > 0 && branch[1].n <= depth;
    }
    std::uint32_t row0(std::uint32_t b) const {
        if (b >= branches) throw std::out_of_range("tree branch");
        return b == 0 ? 0u : branch[0].n + 1;
    }
    std::uint32_t rows() const { return branch[0].n + branch[1].n + branches; }

    // Dense target work needs only one copy of the shared root.  The public
    // row layout above deliberately remains two ordinary linear verifies so
    // acceptance, diagnostics, and control-buffer ABI do not change.  These
    // helpers define the lossless DAG projection used by the optimized
    // verifier: branch 0 is [0 .. n0], branch 1 reuses row 0 and owns
    // [n0 + 1 .. n0 + n1].
    std::uint32_t uniqueRows() const {
        if (!valid()) throw std::invalid_argument("invalid tree plan");
        return branch[0].n + branch[1].n + 1;
    }
    std::uint32_t uniqueRow(std::uint32_t b, std::uint32_t local) const {
        if (!valid()) throw std::invalid_argument("invalid tree plan");
        if (b >= branches) throw std::out_of_range("tree branch");
        if (local > branch[b].n) throw std::out_of_range("tree branch row");
        if (b == 0 || local == 0) return local;
        return branch[0].n + local;
    }
    std::uint32_t uniqueRowFromExpanded(std::uint32_t row) const {
        if (!valid()) throw std::invalid_argument("invalid tree plan");
        if (row >= rows()) throw std::out_of_range("tree expanded row");
        if (row <= branch[0].n) return row;
        return uniqueRow(1, row - row0(1));
    }
};

struct Accept {
    std::uint32_t branch = 0;
    std::uint32_t accepted = 0;  // accepted draft tokens
    std::uint32_t chosen = 0;    // target token after the accepted prefix
    std::uint32_t emitted = 0;   // accepted + chosen, clipped to the budget
    bool has_chosen = false;
};

// target contains one target argmax for each verify row in Plan::row layout.
// The branch with the longest accepted draft prefix wins; ties choose branch
// 0.  Without a relaxed mask this is the exact-match rule.
// A budget ending inside an all-matching draft prefix emits no extra target
// token, which keeps max_tokens handling independent of speculative depth.
// `relaxed` optionally marks draft rows that the caller accepted by the fast
// mode's relaxed-greedy rule.  Exact matches are always accepted.  Keeping the
// policy decision outside this host-only planner lets the server apply the
// same logits and repetition checks as its linear speculative path.
inline Accept accept(const Plan& plan, std::span<const std::uint32_t> target, std::uint32_t budget,
                     std::span<const std::uint8_t> relaxed = {}) {
    if (!plan.valid()) throw std::invalid_argument("invalid tree plan");
    if (target.size() < plan.rows()) throw std::invalid_argument("short tree target rows");
    if (!relaxed.empty() && relaxed.size() < plan.rows()) throw std::invalid_argument("short tree relaxed rows");

    std::array<std::uint32_t, branches> matched{};
    for (std::uint32_t b = 0; b < branches; ++b) {
        const std::uint32_t r0 = plan.row0(b);
        const std::uint32_t lim = std::min(plan.branch[b].n, budget);
        while (matched[b] < lim &&
               (target[r0 + matched[b]] == plan.branch[b].drafts[matched[b]] ||
                (!relaxed.empty() && relaxed[r0 + matched[b]] != 0)))
            ++matched[b];
    }

    Accept out;
    out.branch = matched[1] > matched[0] ? 1u : 0u;
    out.accepted = matched[out.branch];
    out.has_chosen = out.accepted < budget;
    if (out.has_chosen) out.chosen = target[plan.row0(out.branch) + out.accepted];
    out.emitted = out.accepted + static_cast<std::uint32_t>(out.has_chosen);
    return out;
}

}  // namespace whirl::tree_spec
