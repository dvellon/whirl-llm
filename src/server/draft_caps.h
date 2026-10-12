// MTP draft caps under load when the DeltaNet verify keeps snapshot sets (no replay; on the
// Radeon 8060S / gfx1151, whose kernel table has no replay kernels), and the snapshot sets the
// server allocates for them.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/model.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

namespace whirl::server {

struct SnapDraftCaps {
    std::array<std::uint32_t, qwen35::gdn_max_seg + 1> caps{};  // max MTP drafts per cycle with k decoding slots
    std::uint32_t sets = 0;      // snapshot sets to allocate
    std::uint32_t lit_sets = 0;  // sets the literal (n-gram / DFlash2) drafts may use; 0 = all
};

// Exact acceptance (relaxed == false): k slots draft at most min(caps[k], 8 / k), as always (the
// text does not depend on drafts; on the 8060S 3 drafts at 4 slots was 7% slower in balance).
// relaxacc (CONC-8S): k slots draft at most min(caps[k], gdn_max_snap / k) (= the replay path's
// {8, 7, 4, 3} for 1..4 slots): with 8 / k three or four decoding slots drafted 2 per cycle where a
// solo request drafts 3, so relaxacc kept other drafts and 4 concurrent requests != each alone.
// The literal drafts keep the budget the old caps gave them, so a solo request makes the same draft
// choices as before. user_caps (WHIRL_MTP_BATCH_DRAFTS) are used as given and leave the literal
// budget at all sets; snap_env (WHIRL_SNAP_SETS) raises both.
inline SnapDraftCaps snapshotDraftCaps(const std::array<std::uint32_t, qwen35::gdn_max_seg + 1>& caps, std::uint32_t drafts,
                                       std::uint32_t parallel, bool user_caps, std::optional<std::uint32_t> snap_env,
                                       bool relaxed) {
    SnapDraftCaps r;
    r.caps = caps;
    const std::uint32_t P = std::min(parallel, qwen35::gdn_max_seg);
    std::uint32_t need_old = drafts;  // the sets the old 8 / k caps needed
    for (std::uint32_t k = 1; k <= P; ++k) need_old = std::max(need_old, k * std::min({caps[k], drafts, k > 1 ? 8 / k : drafts}));
    if (!user_caps)
        for (std::uint32_t k = 2; k <= qwen35::gdn_max_seg; ++k) r.caps[k] = std::min(r.caps[k], relaxed ? qwen35::gdn_max_snap / k : 8 / k);
    std::uint32_t need = drafts;
    for (std::uint32_t k = 1; k <= P; ++k) need = std::max(need, k * std::min(r.caps[k], drafts));
    if (!user_caps && relaxed) r.lit_sets = std::min(qwen35::gdn_max_snap, need_old);
    if (snap_env) {
        const std::uint32_t x = std::min(*snap_env, qwen35::gdn_max_snap);
        need = std::max(need, x);
        if (r.lit_sets > 0) r.lit_sets = std::max(r.lit_sets, x);
    }
    r.sets = std::min(qwen35::gdn_max_snap, need);
    return r;
}

}  // namespace whirl::server
