// Load-failure diagnosis for a DLL (see dll_deps.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.

#include "release/dll_deps.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace whirl::app::dlldeps {

namespace {

std::string narrowUtf8(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<std::size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

std::wstring widenAscii(const std::string& s) { return std::wstring(s.begin(), s.end()); }

std::string lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

bool isApiSet(const std::string& dll) {
    const std::string l = lower(dll);
    return l.rfind("api-ms-", 0) == 0 || l.rfind("ext-ms-", 0) == 0;
}

template <class T>
bool rd(const std::vector<unsigned char>& b, std::size_t off, T& v) {
    if (off > b.size() || b.size() - off < sizeof(T)) return false;
    std::memcpy(&v, b.data() + off, sizeof(T));
    return true;
}

void addUnique(std::vector<std::string>& v, const std::string& s) {
    const std::string l = lower(s);
    for (const std::string& x : v)
        if (lower(x) == l) return;
    v.push_back(s);
}

// true when `name` loads; with `loaded_err` the GetLastError of a failure
bool tryLoad(const std::wstring& name, DWORD* err) {
    const UINT old_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX);
    HMODULE h = LoadLibraryW(name.c_str());
    *err = h == nullptr ? GetLastError() : 0;
    SetErrorMode(old_mode);
    return h != nullptr;  // kept loaded: the process exits right after the diagnosis
}

}  // namespace

std::vector<std::string> peImports(const std::wstring& path) {
    std::vector<std::string> out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    const std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::uint16_t mz = 0;
    std::uint32_t pe = 0, sig = 0;
    if (!rd(b, 0, mz) || mz != 0x5a4d || !rd(b, 0x3c, pe) || !rd(b, pe, sig) || sig != 0x00004550) return out;
    std::uint16_t nsec = 0, opt_size = 0, magic = 0;
    if (!rd(b, pe + 6, nsec) || !rd(b, pe + 20, opt_size)) return out;
    const std::size_t opt = static_cast<std::size_t>(pe) + 24;
    if (!rd(b, opt, magic) || magic != 0x20b) return out;  // PE32+ only
    struct Sec {
        std::uint32_t va, vs, raw, rawsize;
    };
    std::vector<Sec> secs;
    for (std::uint16_t i = 0; i < nsec; ++i) {
        const std::size_t s = opt + opt_size + 40u * i;
        Sec x{};
        if (!rd(b, s + 8, x.vs) || !rd(b, s + 12, x.va) || !rd(b, s + 16, x.rawsize) || !rd(b, s + 20, x.raw)) return out;
        secs.push_back(x);
    }
    auto rva2off = [&](std::uint32_t rva, std::size_t& off) {
        for (const Sec& s : secs) {
            const std::uint32_t span = std::max(s.vs, s.rawsize);
            if (rva >= s.va && rva - s.va < span) {
                off = static_cast<std::size_t>(rva - s.va) + s.raw;
                return off < b.size();
            }
        }
        return false;
    };
    auto cstr = [&](std::size_t off) {
        std::string s;
        while (off < b.size() && b[off] != 0 && s.size() < 260) s += static_cast<char>(b[off++]);
        return s;
    };
    // data directory 1: imports (20-byte descriptors, name RVA at +12);
    // 13: delay imports (32-byte descriptors, name RVA at +4)
    const struct {
        int idx;
        std::size_t size, name_at;
    } dirs[] = {{1, 20, 12}, {13, 32, 4}};
    for (const auto& d : dirs) {
        std::uint32_t rva = 0;
        if (!rd(b, opt + 112 + 8u * d.idx, rva) || rva == 0) continue;
        std::size_t off = 0;
        if (!rva2off(rva, off)) continue;
        for (int n = 0; n < 4096; ++n, off += d.size) {
            std::uint32_t nrva = 0;
            if (!rd(b, off + d.name_at, nrva) || nrva == 0) break;
            std::size_t noff = 0;
            if (!rva2off(nrva, noff)) break;
            const std::string name = cstr(noff);
            if (!name.empty()) addUnique(out, name);
        }
    }
    return out;
}

std::wstring findDll(const std::wstring& dll) {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = SearchPathW(nullptr, dll.c_str(), nullptr, static_cast<DWORD>(std::size(buf)), buf, nullptr);
    if (n == 0 || n >= std::size(buf)) return {};
    return std::wstring(buf, n);
}

Diagnosis diagnose(const std::wstring& dll) {
    Diagnosis d;
    const bool is_path = dll.find_first_of(L"\\/") != std::wstring::npos;
    if (is_path) {
        if (GetFileAttributesW(dll.c_str()) != INVALID_FILE_ATTRIBUTES) d.found_at = dll;
    } else {
        d.found_at = findDll(dll);
    }
    if (d.found_at.empty()) return d;
    for (const std::string& imp : peImports(d.found_at)) {
        if (isApiSet(imp)) continue;
        DWORD err = 0;
        if (tryLoad(widenAscii(imp), &err)) continue;
        const std::wstring at = findDll(widenAscii(imp));
        if (at.empty()) {
            addUnique(d.missing, imp);
            continue;
        }
        // present but not loadable: one level further down
        bool sub_missing = false;
        for (const std::string& sub : peImports(at)) {
            if (isApiSet(sub)) continue;
            DWORD e2 = 0;
            if (tryLoad(widenAscii(sub), &e2)) continue;
            if (findDll(widenAscii(sub)).empty()) {
                addUnique(d.missing, sub);
                sub_missing = true;
            }
        }
        if (!sub_missing) addUnique(d.unloadable, imp);
    }
    return d;
}

std::string message126(const std::string& dll_name, const Diagnosis& d, const char* driver_advice) {
    if (d.found_at.empty())
        return std::format("error: the AMD GPU runtime ({}) was not found, so WHIRL cannot use the GPU.\n"
                           "  It is installed by the AMD graphics driver.\n{}",
                           dll_name, driver_advice);
    const std::string at = narrowUtf8(d.found_at);
    if (d.missing.empty() && d.unloadable.empty())
        return std::format("error: the AMD GPU runtime ({}) was found at {}, but it could not be loaded\n"
                           "  (a DLL it needs is missing), so WHIRL cannot use the GPU.\n{}",
                           dll_name, at, driver_advice);
    std::string list;
    for (const std::string& m : d.missing) list += (list.empty() ? "" : ", ") + m;
    for (const std::string& m : d.unloadable) list += (list.empty() ? "" : ", ") + m + " (present, does not load)";
    std::string s = std::format("error: the AMD GPU runtime ({}) was found at {},\n"
                                "  but it cannot be loaded because a DLL it needs is missing: {}.\n",
                                dll_name, at, list);
    bool kpack = false, vcrt = false;
    for (const std::string& m : d.missing) {
        const std::string l = lower(m);
        if (l == "rocm_kpack.dll") kpack = true;
        if (l.rfind("vcruntime140", 0) == 0 || l.rfind("msvcp140", 0) == 0) vcrt = true;
    }
    if (kpack)
        s += "  rocm_kpack.dll is part of AMD's ROCm runtime; the AMD Software: Adrenalin Edition for ROCm\n"
             "  driver installs a GPU runtime that needs it but does not install it. WHIRL ships it in its zip:\n"
             "  put rocm_kpack.dll from the WHIRL zip next to whirl.exe and whirl-server.exe, or unzip WHIRL\n"
             "  again into a new folder.\n";
    if (vcrt)
        s += "  The Microsoft Visual C++ Redistributable (x64) is missing; install it from\n"
             "  https://aka.ms/vs/17/release/vc_redist.x64.exe and run the command again.\n";
    if (!kpack && !vcrt) s += driver_advice;
    return s;
}

}  // namespace whirl::app::dlldeps
