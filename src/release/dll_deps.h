// Why a DLL fails to load with ERROR_MOD_NOT_FOUND (126): the DLL itself is
// missing, or it is present and one of the DLLs it imports is missing.
// Used by the GPU runtime pre-flight check (release.cpp): the ROCm 10.1
// driver's amdhip64_7.dll (10.0.3684) imports rocm_kpack.dll, which that driver
// does not install (GitHub issue #2), and LoadLibrary reports both cases as 126.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Windows only; no HIP.
#pragma once

#include <string>
#include <vector>

namespace whirl::app::dlldeps {

// DLL names imported by the PE32+ image at `path` (import directory, then the
// delay-import directory), as written in the file. Empty when the file cannot
// be read or is not a PE32+ image.
std::vector<std::string> peImports(const std::wstring& path);

// Full path of `dll` in the standard search order (SearchPathW: the exe's
// folder, the current folder, System32, Windows, PATH), or empty.
std::wstring findDll(const std::wstring& dll);

struct Diagnosis {
    std::wstring found_at;                // where the DLL itself is ("" = not found)
    std::vector<std::string> missing;     // imports (direct or one level down) that are not found
    std::vector<std::string> unloadable;  // imports that exist but do not load for another reason
};

// Diagnoses a load failure of `dll` (a name, searched with findDll, or a full
// path). For each import that does not load, looks one level further down,
// so "rocm_kpack.dll is present but vcruntime140.dll is missing" is told apart
// from "rocm_kpack.dll is missing". Loads the imports that are present (the
// caller is about to exit anyway). API-set names (api-ms-*, ext-ms-*) are skipped.
Diagnosis diagnose(const std::wstring& dll);

// User-facing message for a 126 failure of `dll_name`, from a diagnosis.
// `driver_advice` is appended when the DLL itself is not found.
std::string message126(const std::string& dll_name, const Diagnosis& d, const char* driver_advice);

}  // namespace whirl::app::dlldeps
