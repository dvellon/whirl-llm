// Device-local memory accounting per device class: "how much VRAM did this step take?"
// SPDX-License-Identifier: Apache-2.0
//
// The answer needs a different counter per device class on Windows (measured on the R9700 and the
// Radeon 8060S with HIP 7.2, audit/VIS-8S-2.md):
//  - discrete GPU (R9700): hipMemGetInfo's free is this process's view of its VRAM (another
//    process's allocations do not move it) and returns exactly on hipFree; pinned host memory
//    (hipHostMalloc) does not move it (it shows as WDDM non-local / GPU "Shared Usage").
//  - integrated GPU (Radeon 8060S, UMA; the BIOS carve-out and system RAM are separate pools):
//    hipMemGetInfo's free drops by TWICE the size of every pinned host allocation, and DXGI's
//    LOCAL segment group counts it too (a UMA adapter reports all of its memory as local), although
//    the pages are system RAM (GPU "Shared Usage"), not the carve-out. The carve-out this process
//    holds is its GPU "Dedicated Usage" (PDH \GPU Process Memory(pid_P_luid_L_phys_0)\Dedicated
//    Usage, the number Task Manager shows). It counts the runtime's committed heap: a device
//    allocation of >= a few MiB shows at once; small ones can land in an already committed chunk,
//    and a freed block stays committed (reused by the next allocation), so it suits "what did this
//    step take" (load), not "was it given back" (release: hipMemGetInfo answers that on both
//    classes; on an integrated GPU it also counts pinned memory allocated in between, twice).
#pragma once

#include <cstdint>

namespace whirl::hip {

enum class MemClass { discrete, integrated };

struct LocalMemSnap {
    MemClass cls = MemClass::discrete;
    std::uint64_t hip_free = 0;        // hipMemGetInfo free (WHIRL_VRAM_LIMIT_MB applied)
    bool pdh_ok = false;               // per-process GPU counters read
    std::uint64_t dedicated = 0;       // this process's GPU "Dedicated Usage" on the current adapter
    std::uint64_t shared = 0;          // this process's GPU "Shared Usage" on the current adapter
};

// The current device's class and counters now.
LocalMemSnap localMemSnap();

struct LocalMemTaken {
    double mib = 0;              // device-local memory taken from a to b (negative: given back)
    const char* how = "";        // which counter, for logs
    double hip_free_mib = 0;     // hipMemGetInfo free drop, always reported
    double shared_mib = 0;       // "Shared Usage" growth (pinned / system memory), 0 if unknown
};

// Rule per class (pure; tests/unit): discrete -> hipMemGetInfo free drop; integrated -> growth
// of "Dedicated Usage" (carve-out), falling back to the hipMemGetInfo drop when the counters
// could not be read (that fallback over-reports pinned memory, so it can fail, never pass wrongly).
inline LocalMemTaken localMemTaken(const LocalMemSnap& a, const LocalMemSnap& b) {
    constexpr double kMiB = 1048576.0;
    LocalMemTaken t;
    t.hip_free_mib = (static_cast<double>(a.hip_free) - static_cast<double>(b.hip_free)) / kMiB;
    const bool pdh = a.pdh_ok && b.pdh_ok;
    if (pdh) t.shared_mib = (static_cast<double>(b.shared) - static_cast<double>(a.shared)) / kMiB;
    if (b.cls == MemClass::integrated && pdh) {
        t.mib = (static_cast<double>(b.dedicated) - static_cast<double>(a.dedicated)) / kMiB;
        t.how = "integrated GPU: carve-out (this process's GPU Dedicated Usage)";
    } else if (b.cls == MemClass::integrated) {
        t.mib = t.hip_free_mib;
        t.how = "integrated GPU, GPU process counters unreadable: hipMemGetInfo, counts pinned system RAM twice";
    } else {
        t.mib = t.hip_free_mib;
        t.how = "discrete GPU: hipMemGetInfo";
    }
    return t;
}

// Gate limits per device class (whirl-server-gate --suite vis; measurements in audit/VIS-8S-2.md).
//
// mmproj load (a pinned host copy of pinned_bytes, no device buffer): 2 MiB on a discrete GPU
// (hipMemGetInfo, measured 0.0). An integrated GPU also keeps the GPU page tables that map the
// pinned pages in the carve-out: 8 bytes per 4 KiB page (pinned / 512; measured +1.86 MiB for
// 887 MiB, +7.4 MiB for 3,548 MiB, 0.13 MiB for 64 MiB), so its limit is 2 MiB plus that.
inline double loadLimitMiB(MemClass c, std::uint64_t pinned_bytes) {
    constexpr double kMiB = 1048576.0;
    return c == MemClass::integrated ? 2.0 + static_cast<double>(pinned_bytes) / 512.0 / kMiB : 2.0;
}
// Idle release, hipMemGetInfo "before the first image" vs after the release (whole workload, once):
// discrete 8 MiB (measured 0.0 MiB over 20 cycles). Integrated 16 MiB: the HIP runtime grows its
// own memory once, by 8.3-8.8 MiB, at the first long (2,040-token) image prompt with the prefix
// cache on (GPU Dedicated +6.3 MiB, Shared +1.4 MiB counted twice by hipMemGetInfo), then stays
// flat (20 cycles). A leak per load/unload cycle is caught by releaseGrowthLimitMiB on both.
inline double releaseLimitMiB(MemClass c) { return c == MemClass::integrated ? 16.0 : 8.0; }
// growth of that delta over further encoder load/unload cycles (both classes; measured 0.0)
inline constexpr double releaseGrowthLimitMiB = 2.0;

}  // namespace whirl::hip
