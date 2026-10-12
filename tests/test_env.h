// Test hygiene for the CPU test binaries: clear the developer shell's WHIRL_* variables before any
// test runs, so a leftover WHIRL_MOE_FP8 / WHIRL_MODE / WHIRL_KV ... cannot change what the loader,
// the numerics plan or the server under test does (FIX-TEST-A, FR-TEST P3-11). Kept: WHIRL_TEST_*
// (test inputs: tokenizer GGUF, temp dir, model paths) and the test-selection knobs named by the caller.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdio>
#include <cwchar>
#include <cwctype>
#include <initializer_list>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef near  // windows.h legacy macros clash with test identifiers
#undef far
#endif

namespace whirl::test {

// Returns the names it cleared (printed once so a run log shows them).
inline std::vector<std::wstring> clearWhirlEnv(std::initializer_list<const wchar_t*> keep = {}) {
    std::vector<std::wstring> cleared;
#ifdef _WIN32
    std::vector<std::wstring> names;
    if (wchar_t* blk = GetEnvironmentStringsW()) {
        for (const wchar_t* p = blk; *p; p += std::wcslen(p) + 1) {
            const std::wstring s(p);
            const std::size_t eq = s.find(L'=', 1);
            if (eq == std::wstring::npos) continue;
            names.push_back(s.substr(0, eq));
        }
        FreeEnvironmentStringsW(blk);
    }
    for (const std::wstring& n : names) {
        std::wstring up = n;
        for (wchar_t& c : up) c = static_cast<wchar_t>(std::towupper(c));
        if (up.rfind(L"WHIRL_", 0) != 0 || up.rfind(L"WHIRL_TEST_", 0) == 0) continue;
        bool kept = false;
        for (const wchar_t* k : keep) kept = kept || up == k;
        if (kept) continue;
        SetEnvironmentVariableW(n.c_str(), nullptr);
        _wputenv_s(n.c_str(), L"");  // the CRT copy that getenv reads
        cleared.push_back(n);
    }
    if (!cleared.empty()) {
        std::printf("test env: cleared");
        for (const auto& n : cleared) std::printf(" %ls", n.c_str());
        std::printf(" (FR-TEST P3-11)\n");
    }
#endif
    return cleared;
}

}  // namespace whirl::test
