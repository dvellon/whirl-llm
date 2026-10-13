// Idle write (TIER-IDLE): in --kv-ssd-mode shutdown, once the server has been idle for a while (no
// request running, queued or being built), the entries the close flush would most want -- the reused
// ones -- are written to SSD in the background, so a close that allows only ~5 s (console close,
// logoff, OS shutdown) has little or nothing left to write.
// SPDX-License-Identifier: Apache-2.0
//
// Pure (no tier state, no OS calls, the caller passes the clock) so the host tests can drive it with a
// fake clock (tests/server/tier_tests.cpp testIdleFlushPolicy). The engine owns one IdleFlushClock and
// sets Tier::idle_writing when it fires; Tier::tick picks the entries with idleWriteWanted.
//
//   when:  once per idle period, after idle_flush_ms_default (300 s) of continuous idleness; any request
//          ends the period (the next idle period may fire again). Never in --kv-ssd-mode always (entries
//          are written after their delay there) or off.
//   what:  the eviction-write rule of evict_policy.h in shutdown mode: reused, not demoted, >= min_tokens,
//          not already current on SSD. A one-shot prompt is never written (the close flush still writes
//          it when WHIRL is closed).
//   how:   one write at a time, and a request that arrives stops the pass (Tier::idle_writing cleared):
//          a new request waits on at most the one write already running on the IO thread.
//
// WHIRL_KV_IDLE_FLUSH_MS (hidden, tests only) overrides the 300 s; 0 turns the idle write off.

#pragma once

#include <cstdint>
#include <optional>

#include "tier/evict_policy.h"

namespace whirl::tier {

inline constexpr std::uint64_t idle_flush_ms_default = 300000;

// The idle write runs at all: shutdown mode, an SSD tier open for writing, a nonzero delay.
inline bool idleFlushEnabled(bool shutdown_only, bool ssd_on, std::uint64_t after_ms) {
    return shutdown_only && ssd_on && after_ms > 0;
}

// Per entry: the shutdown-mode eviction rule (f.shutdown_only is forced on).
inline bool idleWriteWanted(EvictFacts f) {
    f.shutdown_only = true;
    return evictWriteWanted(f);
}

// One idle period at a time. now_ms: any monotonic millisecond clock (the caller's).
class IdleFlushClock {
public:
    // A request is running, queued or being built: the idle period (if any) ends.
    void activity() {
        idle_ = false;
        fired_ = false;
    }
    // Called while idle. true exactly once per idle period: when idle for >= after_ms.
    bool poll(std::int64_t now_ms, std::uint64_t after_ms) {
        if (!idle_) {
            idle_ = true;
            fired_ = false;
            since_ = now_ms;
        }
        if (fired_ || after_ms == 0) return false;
        if (now_ms - since_ < static_cast<std::int64_t>(after_ms)) return false;
        fired_ = true;
        return true;
    }
    // When poll would fire (nullopt: not idle, already fired this period, or off).
    std::optional<std::int64_t> deadline(std::uint64_t after_ms) const {
        if (!idle_ || fired_ || after_ms == 0) return std::nullopt;
        return since_ + static_cast<std::int64_t>(after_ms);
    }
    bool idle() const { return idle_; }
    bool fired() const { return fired_; }

private:
    bool idle_ = false;
    bool fired_ = false;
    std::int64_t since_ = 0;
};

}  // namespace whirl::tier
