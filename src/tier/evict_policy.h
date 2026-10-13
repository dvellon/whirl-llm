// RAM-tier eviction policy: when RAM pressure pushes an entry out of the pinned RAM tier before
// its SSD write is due, is it written to the SSD tier (asynchronously) or dropped?
// SPDX-License-Identifier: Apache-2.0
//
// Pure function (no tier state, no OS calls) so the host tests can check every case directly
// (tests/server/tier_tests.cpp testEvictPolicy); Tier::evictWanted fills EvictFacts from an Entry.
//
//   never written: demoted (probably superseded), SSD tier off, empty, or < min_tokens
//   --kv-ssd-mode shutdown: written only if the entry was reused (restored, resumed, or a shared
//       prefix checkpoint hit) -- a one-shot prompt is never written, and with no RAM pressure
//       nothing is written while running (TIER-EVICT). The same rule picks what the idle write
//       (idle_flush.h, TIER-IDLE) and the flush when WHIRL is closed (closeWriteWanted, CLOSE-REUSED)
//       write, so in shutdown mode a one-shot prompt never reaches the SSD.
//   --kv-ssd-mode always: written if it passes the seen-once filter (reused, or filter off)
//   in both modes: an entry whose SSD copy is already current (on SSD, nothing dirty) is not rewritten

#pragma once

#include <cstdint>

namespace whirl::tier {

struct EvictFacts {
    bool ssd_on = false;         // an SSD tier directory is open for writing
    bool shutdown_only = false;  // --kv-ssd-mode shutdown (Config::ssd_shutdown_only)
    bool seen_filter = true;     // Config::ssd_seen_filter (always mode)
    bool demoted = false;        // Entry::demoted
    bool reused = false;         // Entry::reused
    std::uint32_t n_tok = 0;
    std::uint32_t min_tokens = 0;
    bool on_ssd = false;  // Entry::ssd: an SSD copy exists
    bool dirty = false;   // Entry::hasDirty(): blocks or metadata changed since that copy
};

inline bool evictWriteWanted(const EvictFacts& f) {
    if (f.demoted) return false;
    if (!f.ssd_on || f.n_tok == 0 || f.n_tok < f.min_tokens) return false;
    const bool worth = f.shutdown_only ? f.reused : (!f.seen_filter || f.reused);
    if (!worth) return false;
    return !f.on_ssd || f.dirty;  // clean: nothing to write
}

// CLOSE-REUSED: the flush when WHIRL is closed (graceful stop) in --kv-ssd-mode shutdown writes only what
// the rule above writes in shutdown mode -- reused entries -- with one difference: a demoted entry (kept
// whole but evicted first, e.g. the main agent's conversation while a subagent with the same system prompt
// ran) is still written if it was reused, since the close flush is its last chance to reach the SSD. The
// always-mode flush keeps its own seen-once filter (Tier::seenOk) and does not use this.
inline bool closeWriteWanted(EvictFacts f) {
    f.shutdown_only = true;
    f.demoted = false;
    return evictWriteWanted(f);
}

}  // namespace whirl::tier
