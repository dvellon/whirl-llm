# Radeon 8060S / gfx1151 performance lab

This fork carries an opt-in Windows HIP performance stack for the Ryzen AI Max+ 395 / Radeon 8060S. The current reproducible point uses Swift-1.5 Qwen3.8 27B MXFP4, one slot, an 8,238-token prompt, and a forced 1,000-token response.

## Current measured point

| Task | Prefill | Decode | Prompt / output |
|---|---:|---:|---:|
| Python coding | 554.05 tok/s | **53.52 tok/s** | 8,238 / 1,000 |
| Technical prose | 556.27 tok/s | **42.20 tok/s** | 8,238 / 1,000 |

Hardware and software: Ryzen AI Max+ 395, Radeon 8060S (`gfx1151`), 128 GB LPDDR5X with 32 GB dedicated and up to 96 GB shared graphics memory, Windows, HIP SDK 7.2, AMD driver `32.0.31041.1004`. Source commit: `635a8bc8fbf5474442f218a73d3e6def5d475b58`.

The coding run accepted 821 of 1,602 MTP drafts over 178 target-verification cycles (5.61 output tokens/cycle). Both runs used fresh server processes and reported zero cached prompt tokens.

`--fast` includes WHIRL's documented relaxed-greedy MTP acceptance. It is a speed/quality mode and is **not** byte-equivalent to plain greedy decoding. Use `--balance` for the exact-output gate. The measured balance tree retains the recorded exact greedy digest but is slower.

These tasks match the public 8K/1K/fresh-process/cache-off categories used by other public reports, but their exact original prompts were not available. This is protocol-equivalent evidence, not a claim of byte-identical reproduction or a universal record.

## Build the measured configuration

Install Visual Studio 2022 Build Tools with the C++ x64 workload, CMake,
Ninja, and AMD HIP SDK 7.2. From a normal PowerShell prompt in a clean clone
of this branch, build only the Radeon 8060S code object:

```powershell
.\build.bat Release `
  -DWHIRL_WITH_HIP=ON `
  -DWHIRL_GPU_ARCHS=gfx1151 `
  "-DWHIRL_HIP_ROOT=C:\Program Files\AMD\ROCm\7.2"
```

The resulting binaries are under `build\Release`. The public number was
measured with AMD driver `32.0.31041.1004`; record a different driver rather
than silently comparing it as the same setup.

## Reproduce the 8K/1K run

Download `Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf` from the project's
[documented model source](https://huggingface.co/tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF), then run:

```powershell
.\tools\bench\gfx1151_public_8k.ps1 `
  -Build .\build\Release `
  -Model C:\models\Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf
```

The script refuses a dirty source tree, resolves the exact commit, calibrates both prompts to exactly 8,238 tokenizer-visible tokens in a disposable process, then starts a fresh server per measured task. It saves the response, timing JSON, SHA-256 output digest, stdout/stderr logs, source commit, hardware inventory, and complete summary under `whirl-gfx1151-results`.

Publish the complete result directory when reporting a number. At minimum,
retain `summary.json`, the per-task timing JSON, response digests, server logs,
`source-commit.txt`, and hardware inventory. A screenshot or a lone tok/s value
is not equivalent evidence.

The active opt-ins are:

```text
WHIRL_MXFP4_NT11_W16=1
WHIRL_Q6K_HEAD_NT11_W16=1
WHIRL_TREE_VERIFY_GRAPH=1
WHIRL_NO_PREFIX_CACHE=1
--fast --tree beam2:8x1 -np 1 --ctx-per-slot 262144 --kv-ssd-mode off --kv-ram-mb 0
```

## Related contribution

Branch `perf/gfx1151-dflash-q4-v1` adds native Q4_K selector-codebook support for DFlash2. The Q4 helper loads at 1.05 GiB instead of the Q8 helper's 1.90 GiB and produces the same exact-decoding output digest in the measured code task. Its 43.36 tok/s does not beat the MTP tree, but the compatibility patch is independently GPU-tested and useful beyond this benchmark.
