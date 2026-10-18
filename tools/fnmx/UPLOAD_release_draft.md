# Upload steps (NOT executed: the build failed the KL acceptance rule on 2026-10-19; run only after the owner accepts)

Repo: `tsaipifong/Swift-1.5-Qwen3.8-Flash-Next-MXFP4-GGUF` (create PRIVATE). This file is not uploaded.
All commands in PowerShell. Uploads only through `C:\ClaudeWorkspace\tools\hf\hf_run.ps1` (it holds the token).

## 0. Split (Hugging Face 50 GB per-file limit), then check that llama.cpp loads shard 1

```powershell
$bin = "C:\ClaudeWorkspace\tools\llamacpp_fn_bin\cpu"
$g   = "C:\ClaudeWorkspace\models\fnmx\gguf"
$out = "C:\ClaudeWorkspace\models\fnmx\split"; New-Item -ItemType Directory -Force $out
& "$bin\llama-gguf-split.exe" --split --split-max-size 45G "$g\Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-IQ4_NL.gguf" "$out\Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-IQ4_NL"
& "$bin\llama-gguf-split.exe" --split --split-max-size 45G "$g\Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-Q8_0.gguf"   "$out\Swift-1.5-Qwen3.8-Flash-Next-MXFP4-ple-Q8_0"
```

Disk: each split needs as much free space as the file (99.75 / 125.35 GB); delete the unsplit file after the shard-1 load test passes. Keep >= 30 GB free.
Then: load test from shard 1 (llama-completion, 16 tokens), `sha256sum` every shard + the mmproj into `SHA256SUMS` in this folder, and replace `0000N-of-0000M` in README.md with the real shard names.

## 1. Create the repo (private) and upload card + licenses

```powershell
$env:PATH = "C:\myhermes\venv\Scripts;$env:PATH"
$run  = "C:\ClaudeWorkspace\tools\hf\hf_run.ps1"
$repo = "tsaipifong/Swift-1.5-Qwen3.8-Flash-Next-MXFP4-GGUF"
& $run repos create $repo --type model --private --exist-ok
& $run upload $repo "C:\ClaudeWorkspace\hf-release\Swift-1.5-Qwen3.8-Flash-Next-MXFP4-GGUF" . --exclude "UPLOAD.md" --commit-message "Add model card, LICENSE, LICENSE-QWEN, NOTICE, RECIPE, SHA256SUMS, patch"
```

## 2. Upload the files (separate commits per build)

```powershell
$s = "C:\ClaudeWorkspace\models\fnmx\split"
& $run upload $repo "C:\ClaudeWorkspace\models\fnmx\gguf\mmproj-Swift-1.5-Qwen3.8-Flash-Next-F16.gguf" mmproj-Swift-1.5-Qwen3.8-Flash-Next-F16.gguf --commit-message "Add vision projector (F16)"
& $run upload $repo $s . --include "*ple-IQ4_NL*" --commit-message "Add MXFP4 build, n-gram table IQ4_NL"
& $run upload $repo $s . --include "*ple-Q8_0*"   --commit-message "Add MXFP4 build, n-gram table Q8_0"
```

## 3. Verify every file

`& $run models ls $repo --json` returns `size` and `lfs.sha256` per file; compare with `SHA256SUMS` (all must match). The main session reviews the card before the repo is made public.
