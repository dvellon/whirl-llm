# Upload steps — 2026-10-19 rebuild (REQUANT-1). NOT RUN. For the main session.

Repo: `tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF` (already **public**). This upload **replaces** the three
language-model files in place (same names) and updates the card, so it goes live as soon as the commit lands.
The vision projector and the LICENSE files are not re-uploaded (unchanged).

- Traffic: about 47.4 GB upload (3 x ~15.8 GB). **Not on 2026-10-20** (no big transfers that day); check
  `whirl-work\internal_docs\TRAFFIC_LEDGER.md` before and after.
- `hf_run.ps1` reads the token from Windows Credential Manager, passes it only to the child `hf` process, then clears it.
- This file (`UPLOAD.md`) and the backup folder are **not** uploaded.
- The local LM Studio copies (`C:\Users\tsaip\.lmstudio\models\ukisai\Swift-1.5-Qwen3.8-27b-MXFP4\`) are still the
  first-release files (owner instruction: keep them). The new files are in the staging folder below.

## 0. Prerequisites

```powershell
$env:PATH = "C:\myhermes\venv\Scripts;$env:PATH"
hf version                                          # expect version=1.32.0
C:\ClaudeWorkspace\tools\hf\hf_run.ps1 auth whoami  # expect tsaipifong
```

## 1. Local pre-check (size + SHA256 of the staged files vs SHA256SUMS)

```powershell
$stage = "C:\ClaudeWorkspace\models\requant1\staging\swift"
$card  = "C:\ClaudeWorkspace\hf-release\Swift-1.5-Qwen3.8-27b-MXFP4-GGUF"
$expected = @{
  "Swift-1.5-Qwen3.8-27B-MXFP4-A-outQ6_K.gguf" = @(15815469376, "c765544dee05261e4161281cfc298c493087d623b89c3753675d43255b42e09f")
  "Swift-1.5-Qwen3.8-27B-MXFP4-B-outQ8_0.gguf" = @(16123386176, "4b5f70adf8b462a79920e76c821e76c3875967b0925d78fa59fce01ec287e933")
  "Swift-1.5-Qwen3.8-27B-MXFP4-C-outQ4_K.gguf" = @(15487686976, "d24f9692a4db9dda2fe84a6fdcde9f5007e4bc93e9e905e67d1cbfc6ee85edbc")
}
foreach ($f in $expected.Keys) {
  $p = Join-Path $stage $f
  $len = (Get-Item $p).Length
  $sha = (Get-FileHash $p -Algorithm SHA256).Hash.ToLower()
  "{0}  size {1}  sha {2}" -f $f, ($(if ($len -eq $expected[$f][0]) {"OK"} else {"MISMATCH $len"})), ($(if ($sha -eq $expected[$f][1]) {"OK"} else {"MISMATCH $sha"}))
}
```

## 2. Upload (one commit: three GGUFs + card files)

Build an upload folder of hard links (no extra disk space) and copies of the card files, then upload it as one commit,
so the card and the files change together:

```powershell
$up = "C:\ClaudeWorkspace\models\requant1\upload_swift"
New-Item -ItemType Directory -Force $up | Out-Null
foreach ($f in $expected.Keys) { New-Item -ItemType HardLink -Path (Join-Path $up $f) -Target (Join-Path $stage $f) -Force | Out-Null }
foreach ($f in @("README.md", "NOTICE", "RECIPE.md", "SOURCE_SHA256.txt", "SHA256SUMS", "mxfp4_wsearch.patch")) { Copy-Item (Join-Path $card $f) $up -Force }
Get-ChildItem $up | Select-Object Name, Length

C:\ClaudeWorkspace\tools\hf\hf_run.ps1 upload tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF $up . `
  --commit-message "Rebuild A/B/C with the imatrix-weighted MXFP4 quantizer (KLD vs BF16 -17 to -19%); card, NOTICE, RECIPE, SHA256SUMS"
```

If the upload is interrupted, re-run the same command (identical files are skipped).

## 3. Post-upload verification

```powershell
$repo = "tsaipifong/Swift-1.5-Qwen3.8-27b-MXFP4-GGUF"
$remote = C:\ClaudeWorkspace\tools\hf\hf_run.ps1 models ls $repo --json | Out-String | ConvertFrom-Json
$remote | Select-Object path, size, @{n="sha256";e={$_.lfs.sha256}}
```

- [ ] The three GGUFs: size and LFS SHA256 equal step 1 (and `SHA256SUMS`).
- [ ] `mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf` still `71988a54…` / 927,607,584 bytes; `LICENSE`, `LICENSE-APACHE-2.0` unchanged.
- [ ] New files present: `RECIPE.md`, `SOURCE_SHA256.txt`, `SHA256SUMS`, `mxfp4_wsearch.patch`; no `UPLOAD.md`.
- [ ] Card renders (Quality table, files table with the new hashes).
- [ ] Then delete the upload folder of hard links (`Remove-Item -Recurse $up`; the staged files stay).
- [ ] Update `TRAFFIC_LEDGER.md`.
- [ ] After the upload: install the new files into the LM Studio folder (or point `WHIRL_GATE_SWIFT` at the staged A) before
      running WHIRL gates — the gfx1151 Swift golden on branch `requant1` now describes the new A file.
