// Identity of everything that decides the bits of a host tier entry (moved out
// of server_main.cpp, refactor R-1). An entry is restored only by a process
// with the same fingerprint.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace whirl::cache {

struct FingerprintInputs {
    std::uint64_t exe_id = 0;
    std::string model_base;  // model file name (no directory)
    std::uint64_t tensors = 0, bytes = 0;
    std::string kv_name;
    bool use_mtp = false;
    std::uint64_t n_gdn = 0;
    std::uint64_t ck_bytes = 0, page_bytes = 0;
    std::uint64_t max_batch = 0;
    // GPU code object (hip DeviceInfo::gcn_arch, e.g. "gfx1201"): one exe carries kernels for several
    // GPUs whose bits differ, so an entry written on one GPU is never restored on another (FIX-NUM P2-2)
    std::string gpu_arch;
    std::string gpu_name;    // device name (DeviceInfo::name), e.g. "AMD Radeon AI PRO R9700"
    std::string kernel_set;  // code object family the model loaded (hip::archName + its size)
    // numerics mode and its items (numerics::logLine): a --balance run's KV /
    // DeltaNet state must never be restored by a --precise process
    std::string numerics;
    // WHIRL_* environment (any order; filtered by fingerprintEnvKept)
    std::vector<std::pair<std::string, std::string>> env;
    // tests only: hash the format from before FIX-NUM P2-2 (no GPU identity; the three GPU fields
    // must then be empty). Without it an empty gpu_arch or kernel_set is an error (FIX-TEST-A,
    // FR-TEST P2-3: an empty GPU identity used to fall back silently to the old format, so two GPUs
    // would share SSD tier files).
    bool legacy_no_gpu = false;
};

// Throws std::invalid_argument when the GPU identity is missing (see legacy_no_gpu).
std::uint64_t tierFingerprint(const FingerprintInputs& in);

// What the server gathers for the tier fingerprint (server_main.cpp fills it from the loaded model,
// hip::describeDevice and the embedded code object; kept free of HIP so the CPU tests can call it).
struct ServerIdentity {
    std::uint64_t exe_id = 0;
    std::string model_path;  // as given on the command line
    std::uint64_t tensors = 0, bytes = 0;
    std::string kv_name;
    bool use_mtp = false;
    std::uint64_t n_gdn = 0, ck_bytes = 0, page_bytes = 0, max_batch = 0;
    std::string numerics;
    std::string gpu_arch, gpu_name;     // hip::DeviceInfo gcn_arch / name of the current device
    std::string kernel_arch;            // hip::archName of the code object the model loaded
    std::uint64_t kernel_co_bytes = 0;  // that code object's size (0: none embedded)
    std::vector<std::pair<std::string, std::string>> env;
};

// The server's fingerprint inputs (FIX-TEST-A, FR-TEST P2-3: this assembly was inline in
// server_main.cpp and untested). Throws std::invalid_argument when gpu_arch, kernel_arch or the code
// object size is missing.
FingerprintInputs serverFingerprintInputs(const ServerIdentity& s);

// A WHIRL_* variable that changes numerics (not a path / size / logging knob).
bool fingerprintEnvKept(std::string_view key);

// The process environment's kept WHIRL_* variables (empty off Windows).
std::vector<std::pair<std::string, std::string>> fingerprintEnv();

}  // namespace whirl::cache
