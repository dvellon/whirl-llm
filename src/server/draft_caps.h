// MTP draft caps under load when the DeltaNet verify keeps snapshot sets (no replay; on the
// Radeon 8060S / gfx1151, whose kernel table has no replay kernels), and the snapshot sets the
// server allocates for them.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/dflash_policy.h"
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

// FR-SRV P2-5: snapshot sets for the literal drafts (DFlash2 blocks, n-gram) with --draft dflash
// when the DeltaNet verify keeps snapshots (no replay: gfx1151 / Radeon 8060S, WHIRL_GDN_REPLAY=0).
// The engine caps literal drafts by the sets that exist, and the model only allocates sets for the
// drafts of the verify at hand: with none allocated up front no draft was ever made. Enough for one
// full block (max_literal = block - 1), WHIRL_SNAP_SETS raises it, at most gdn_max_snap.
inline std::uint32_t literalSnapSets(std::uint32_t max_literal, std::optional<std::uint32_t> snap_env) {
    std::uint32_t n = std::max<std::uint32_t>(max_literal, 1);
    if (snap_env) n = std::max(n, *snap_env);
    return std::min(qwen35::gdn_max_snap, n);
}

// FR-DFL P3-4: the snapshot sets whirl-server allocates for the literal drafts right after the
// DFlash2 setup and BEFORE sizing the KV pool (server_main), so the pool is sized after them and the
// engine (which caps literal drafts by the sets that exist) drafts at all: --draft dflash without
// DeltaNet replay only (with MTP the MTP caps decide; with replay no sets are needed). 0 = none.
inline std::uint32_t dflashSnapSets(bool use_mtp, bool use_dflash, bool replay, std::uint32_t block, bool ngram,
                                    std::optional<std::uint32_t> snap_env) {
    if (use_mtp || !use_dflash || replay) return 0;
    const std::uint32_t max_lit = std::max(block > 0 ? block - 1 : 0u, ngram ? qwen35::max_ng_drafts : 0u);
    return literalSnapSets(std::min(max_lit, qwen35::gdn_max_snap), snap_env);
}

// FR-DFL P3-1: what --draft dflash adds to the VRAM fit check (vram::FitInput::fixed): drafter weights,
// target-feature capture, per-slot context rings, draft scratch and, without replay, the literal-draft
// snapshot sets (dflash::memory over the drafter's metadata).
inline ::whirl::dflash::Mem dflashFitBytes(const qwen35::DFlashConfig& dc, std::uint64_t weights, const qwen35::Config& target,
                                           std::uint32_t capture_rows, std::uint32_t slots, std::uint32_t snap_sets,
                                           std::uint64_t snap_set_bytes) {
    ::whirl::dflash::MemInput in;
    in.weights = weights;
    in.n_layer = dc.n_layer;
    in.n_embd = dc.n_embd;
    in.n_head = dc.n_head;
    in.n_head_kv = dc.n_head_kv;
    in.head_dim = dc.head_dim;
    in.n_ff = dc.n_ff;
    in.block = dc.block;
    in.window = dc.window;
    in.conv_k = dc.conv_k;
    in.conv_group = dc.conv_group;
    in.sel_topk = dc.sel_topk;
    in.sel_rank = dc.sel_rank;
    in.n_target_layers = static_cast<std::uint32_t>(dc.target_layers.size());
    in.target_embd = target.n_embd;
    in.vocab = target.n_vocab;
    in.capture_rows = capture_rows;
    in.slots = slots;
    in.snap_sets = snap_sets;
    in.snap_set_bytes = snap_set_bytes;
    return ::whirl::dflash::memory(in);
}

}  // namespace whirl::server
