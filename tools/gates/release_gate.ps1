# SPDX-License-Identifier: Apache-2.0
# Release / rc gate: the canonical set of checks that must all pass before an rc merge or a release
# (docs/dev/testing.md). Versioned here (FIX-TEST-A, FR-TEST P2-1 / P2-4 / P2-5 / P2-6 / P3-12): the
# per-rc copies outside the repo only logged exit codes, wrote DONE markers unconditionally and left the
# verdict to whoever read the logs.
#
#   pwsh tools/gates/release_gate.ps1 -Device r9700|8060s [-Build DIR] [-ExpectCommit SHA] [-Steps a,b,...]
#                                     [-Work DIR] [-Tag NAME] [-InjectFail] [-List]
#
# Verdict: every step has an exit code AND an output rule (e.g. "(ALL PASS)", "seqtest: ok", three equal
# DFlash2 hashes with drafts accepted); a step that does not run counts as failed. At the end a table of
# all steps is printed and written to <Work>\<Tag>_summary.txt, plus PASS_<Tag> or FAIL_<Tag> (never
# both, never a plain DONE).
# Exit: 0 every selected step passed; 1 a step failed; 2 setup error (missing exe / model / text root,
#       wrong commit, unknown step name, 0 steps selected).
#
# Machine paths come from the environment (a local wrapper sets them; nothing here is machine specific):
#   WHIRL_GATE_DENSE   Qwen3.8-27B Q4_K_M GGUF          WHIRL_GATE_MOE     Ornith-1.5-35B-A3B MXFP4 GGUF
#   WHIRL_GATE_MX      a dense MXFP4 GGUF (kernel test)  WHIRL_GATE_DFLASH  DFlash2 drafter GGUF
#   WHIRL_GATE_TEXT_ROOT  llama.cpp checkout (server gate texts)
#   WHIRL_GATE_LOCK_SCRIPT  optional: a script defining Enter-R9700Lock; each GPU step then holds
#                           that lock for its own duration only (other agents interleave)
# Steps (-List prints them): CPU test binaries, gate / golden self-tests, golden xcheck; kernel test
# --require-all; selftest dense + MoE x 3 modes; golden check; seqtest dense x 3 modes and MoE precise +
# balance; bench DFlash2 x 3 modes (hashes equal, drafts > 0); server gate balance + fast basic,mt_cache,
# precise basic with --cli, fast_stream dense + MoE, tier,sys,restore_conc,pool in balance and precise
# (--kv auto: the formats the modes ship, not a pinned q8v).
# -InjectFail: the golden steps compare against a copy of the references with one hash changed (a real
# check that must turn the gate red; used to prove the gate fails).
param(
    [ValidateSet("r9700", "8060s")][string]$Device = "r9700",
    [string]$Build = "",
    [string]$ExpectCommit = "",
    [string[]]$Steps = @(),
    [string]$Work = "",
    [string]$Tag = "",
    [switch]$InjectFail,
    [switch]$List
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Build) { $Build = Join-Path $repo "build\Release" }
if (-not $Tag) { $Tag = "gate_$Device" }
if (-not $Work) { $Work = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-release-gate" }
$Steps = @($Steps | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$arch = if ($Device -eq "8060s") { "gfx1151" } else { "gfx1201" }
$port = if ($Device -eq "8060s") { 8095 } else { 8093 }
$Q = $env:WHIRL_GATE_DENSE; $MX = $env:WHIRL_GATE_MOE; $SW = $env:WHIRL_GATE_MX; $DF = $env:WHIRL_GATE_DFLASH
$exe = @{}
foreach ($n in "whirl", "whirl-server", "whirl-server-gate", "whirl-kernel-test", "whirl-tests", "whirl-model-tests", "whirl-cache-tests",
               "whirl-vision-tests", "whirl-tier-tests", "whirl-server-tests") { $exe[$n] = Join-Path $Build "$n.exe" }

# ---------------------------------------------------------------------------------------------------
# step table: name, gpu, command (array: exe + args), rule (scriptblock over the output lines + exit code)
$S = [System.Collections.Generic.List[object]]::new()
function AddStep([string]$name, [bool]$gpu, [string[]]$cmd, [scriptblock]$rule, [hashtable]$envv = @{}) {
    $S.Add([pscustomobject]@{ name = $name; gpu = $gpu; cmd = $cmd; rule = $rule; env = $envv })
}
$exit0 = { param($o, $c) $c -eq 0 }
$pwsh = (Get-Process -Id $PID).Path
foreach ($t in "whirl-tests", "whirl-model-tests", "whirl-cache-tests", "whirl-vision-tests", "whirl-tier-tests") {
    AddStep "cpu_$($t -replace '^whirl-','' -replace '-tests$','')" $false @($exe[$t]) { param($o, $c) $c -eq 0 -and ($o -match "\b[1-9]\d* passed, 0 failed") }
}
AddStep "cpu_server" $false @($exe["whirl-server-tests"], "--gguf", $Q) { param($o, $c) $c -eq 0 -and ($o -match "server tests: [1-9]\d* passed, 0 failed") }
AddStep "cpu_gate_selftest" $false @($exe["whirl-server-gate"], "--selftest") { param($o, $c) $c -eq 0 -and ($o -match ", 0 failed") }
AddStep "cpu_gate_nosuite" $false @($exe["whirl-server-gate"], "--exe", $exe["whirl-server"], "--model", $Q) { param($o, $c) $c -eq 2 }
AddStep "cpu_server_only_typo" $false @($exe["whirl-server-tests"], "--gguf", $Q, "--only", "no_such_test") { param($o, $c) $c -eq 2 }
AddStep "cpu_golden_selftest" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\golden\golden_selftest.ps1")) { param($o, $c) $c -eq 0 -and ($o -match "all cases as expected") }
AddStep "cpu_golden_xcheck" $false @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\golden\golden.ps1"), "xcheck", "-Arch", $arch) { param($o, $c) $c -eq 0 -and ($o -match "file\(s\) ok") }
$ktArgs = @($exe["whirl-kernel-test"], "--device", $Device, "--require-all", "--q4", $Q, "--mx", $SW, "--moemx", $MX)
AddStep "kt_full" $true $ktArgs { param($o, $c) $c -eq 0 -and ($o -match "^PASS: 0 failing") -and ($o -match "--require-all: 0 required kernel") }
foreach ($m in "dense", "moe") {
    foreach ($mode in "precise", "balance", "fast") {
        $mp = if ($m -eq "dense") { $Q } else { $MX }
        AddStep "st_${m}_$mode" $true @($exe["whirl"], "selftest", $mp, "--$mode", "--device", $Device) { param($o, $c) $c -eq 0 -and ($o -match "^selftest: ok") }
    }
}
foreach ($m in "dense", "moe") {
    AddStep "golden_$m" $true @($pwsh, "-NoProfile", "-File", (Join-Path $repo "tools\golden\golden.ps1"), "check", "-Arch", $arch, "-Models", $m, "-Exe", $exe["whirl"], "-GoldenRoot", "@GOLDEN@") `
        { param($o, $c) $c -eq 0 -and ($o -match "golden check $arch`: ok") } @{ WHIRL_GOLDEN_DENSE = $Q; WHIRL_GOLDEN_MOE = $MX }
}
foreach ($p in @(@("dense", "precise"), @("dense", "balance"), @("dense", "fast"), @("moe", "precise"), @("moe", "balance"))) {
    $mp = if ($p[0] -eq "dense") { $Q } else { $MX }
    AddStep "seq_$($p[0])_$($p[1])" $true @($exe["whirl"], "seqtest", $mp, "--decode", "24", "--$($p[1])", "--device", $Device) { param($o, $c) $c -eq 0 -and ($o -match "^seqtest: ok") }
}
$dflRule = {
    param($o, $c)
    $h = @($o | ForEach-Object { if ($_ -match "hash ([0-9a-f]{16})") { $Matches[1] } })
    $dr = @($o | ForEach-Object { if ($_ -match "drafts (\d+)/(\d+)") { [int]$Matches[1] } })
    $c -eq 0 -and $h.Count -eq 3 -and @($h | Select-Object -Unique).Count -eq 1 -and $dr.Count -ge 1 -and ($dr | Measure-Object -Minimum).Minimum -gt 0
}
if ($Device -eq "r9700") {
    foreach ($mode in "precise", "balance", "fast") {
        AddStep "dfl_$mode" $true @($exe["whirl"], "bench", $Q, "--$mode", "--draft", "dflash", "--draft-model", $DF, "--prefill", "512", "--decode", "160",
                                "--modes", "plain,dflash,dflash-ngram", "--device", $Device) $dflRule
    }
}
$gateRule = { param($o, $c) $c -eq 0 -and ($o -match "== whirl-server-gate: [1-9]\d* passed, 0 failed \(ALL PASS\)") }
function GateCmd([string]$model, [string]$suites, [string]$mode, [string[]]$extra) {
    $a = @($exe["whirl-server-gate"], "--exe", $exe["whirl-server"], "--kind", "whirl", "--model", $model, "--suite", $suites,
           "--port", "$port", "--work", "@WORK@", "--kv", "auto")
    if ($mode) { $a += @("--server-env", "WHIRL_MODE=$mode") }
    if ($Device -eq "8060s") { $a += @("--server-env", "WHIRL_DEVICE=8060s") }
    return @($a + @($extra | Where-Object { $_ }))
}
$ssdOff = @("--server-env", "WHIRL_KV_SSD_MODE=off")
$ram8s = @(if ($Device -eq "8060s") { "--server-env"; "WHIRL_KV_RAM_MB=8192" })
AddStep "srv_balance" $true (GateCmd $Q "basic,mt_cache" "balance" $ssdOff) $gateRule
AddStep "srv_fast" $true (GateCmd $Q "basic,mt_cache" "fast" $ssdOff) $gateRule
AddStep "srv_precise_cli" $true (GateCmd $Q "basic" "precise" ($ssdOff + @("--cli", $exe["whirl"]))) $gateRule
AddStep "srv_fs_dense" $true (GateCmd $Q "fast_stream" "" $ssdOff) $gateRule
AddStep "srv_fs_moe" $true (GateCmd $MX "fast_stream" "" $ssdOff) $gateRule
# the cache suites run their own SSD dirs under --work (deleted after the step)
AddStep "srv_cache_balance" $true (GateCmd $Q "tier,sys,restore_conc,pool" "balance" $ram8s) $gateRule
AddStep "srv_cache_precise" $true (GateCmd $Q "tier,sys,restore_conc,pool" "precise" $ram8s) $gateRule

if ($List) { $S | ForEach-Object { "{0,-22} {1}" -f $_.name, ($(if ($_.gpu) { "GPU" } else { "CPU" })) }; exit 0 }

# ---------------------------------------------------------------------------------------------------
# selection and preflight (setup errors exit 2 before any step runs)
foreach ($n in $Steps) { if (-not ($S | Where-Object { $_.name -eq $n })) { Write-Host "release gate: unknown step '$n' (-List shows them)"; exit 2 } }
$sel = @($S | Where-Object { $Steps.Count -eq 0 -or $Steps -contains $_.name })
if ($sel.Count -eq 0) { Write-Host "release gate: 0 steps selected"; exit 2 }
$setup = @()
foreach ($k in $exe.Keys) { if (-not (Test-Path $exe[$k])) { $setup += "missing $($exe[$k])" } }
foreach ($p in @(@("WHIRL_GATE_DENSE", $Q), @("WHIRL_GATE_MOE", $MX), @("WHIRL_GATE_MX", $SW), @("WHIRL_GATE_DFLASH", $DF), @("WHIRL_GATE_TEXT_ROOT", $env:WHIRL_GATE_TEXT_ROOT))) {
    if (-not $p[1] -or -not (Test-Path $p[1])) { $setup += "$($p[0]) not set or not found ($($p[1]))" }
}
$head = (& git -C $repo rev-parse HEAD 2>$null)
$dirty = @(& git -C $repo status --porcelain --untracked-files=no 2>$null).Count
if ($ExpectCommit -and -not ($head -and $head.StartsWith($ExpectCommit))) { $setup += "tree $repo is at $head, expected $ExpectCommit" }
if ($setup.Count -gt 0) { $setup | ForEach-Object { Write-Host "release gate setup: $_" }; exit 2 }
New-Item -ItemType Directory -Force $Work | Out-Null
$logDir = Join-Path $Work $Tag
New-Item -ItemType Directory -Force $logDir | Out-Null
Remove-Item (Join-Path $Work "PASS_$Tag"), (Join-Path $Work "FAIL_$Tag") -ErrorAction SilentlyContinue
$whirlTime = (Get-Item $exe["whirl"]).LastWriteTime.ToString("s")
$hdr = "release gate $Tag on $Device ($arch): tree $repo commit $head (dirty tracked files: $dirty), exe $Build (whirl.exe $whirlTime), $(Get-Date -Format s)"
Write-Host $hdr
$golden = Join-Path $repo "tests\golden"
if ($InjectFail) {
    # a copy of the references with one step hash changed: the golden steps must fail
    $golden = Join-Path $Work "golden_injected"
    Remove-Item -Recurse -Force $golden -ErrorAction SilentlyContinue
    Copy-Item -Recurse (Join-Path $repo "tests\golden") $golden
    $f = Join-Path $golden "$arch\dense\precise_single.txt"
    $l = @(Get-Content $f)
    $l = @($l | ForEach-Object { if ($_ -like "G short.step01 *") { $_ -replace "lh=[0-9a-f]+", "lh=0123456789abcdef" } else { $_ } })
    [System.IO.File]::WriteAllLines($f, [string[]]$l)
    Write-Host "release gate: -InjectFail: golden references copied to $golden with dense/precise_single short.step01 lh changed"
}

# clean environment for the steps: no stray WHIRL_* (a WHIRL_MODE / WHIRL_KV in the shell would change
# what is tested); restored at the end
$saved = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -like "WHIRL_*" -and $_.Name -notlike "WHIRL_GATE_*" -and $_.Name -ne "WHIRL_R9700_LOCK_HELD" })) {
    $saved[$v.Name] = $v.Value; [Environment]::SetEnvironmentVariable($v.Name, $null)
}
if (-not $env:AMD_LOG_LEVEL) { $env:AMD_LOG_LEVEL = "1" }
$env:PYTHONIOENCODING = "utf-8"
$lock = $env:WHIRL_GATE_LOCK_SCRIPT
$rows = @()
try {
    foreach ($s in $sel) {
        $t0 = Get-Date
        $wd = Join-Path $Work "work_$($s.name)"
        $cmd = @($s.cmd | ForEach-Object { $_ -replace "@WORK@", $wd -replace "@GOLDEN@", $golden })
        $log = Join-Path $logDir "$($s.name).txt"
        $c = -1
        foreach ($k in $s.env.Keys) { [Environment]::SetEnvironmentVariable($k, $s.env[$k]) }
        try {
            # hold the R9700 lock for this step only, in this process (a child pwsh in between lost the
            # server gate's output after its first Ctrl+C server stop)
            $held = $null
            if ($s.gpu -and $lock -and -not $env:WHIRL_R9700_LOCK_HELD -and $Device -eq "r9700") {
                . $lock
                $held = Enter-R9700Lock
            }
            try {
                & $cmd[0] @($cmd | Select-Object -Skip 1) *> $log
                $c = $LASTEXITCODE
            } finally {
                if ($held) { $held.Close(); Remove-Item Env:WHIRL_R9700_LOCK_HELD -ErrorAction SilentlyContinue }
            }
        } catch {
            "release gate: step did not run: $_" | Add-Content $log
            $c = -1
        }
        foreach ($k in $s.env.Keys) { [Environment]::SetEnvironmentVariable($k, $null) }
        $out = @(Get-Content $log -ErrorAction SilentlyContinue)
        $ok = $false
        try { $ok = [bool](& $s.rule $out $c) } catch { $ok = $false }
        Remove-Item -Recurse -Force $wd -ErrorAction SilentlyContinue  # server gate work dirs (SSD tier files)
        $sec = [int]((Get-Date) - $t0).TotalSeconds
        $last = ($out | Where-Object { $_ -match "\S" -and $_ -notmatch "^:1:" } | Select-Object -Last 1)
        $row = "{0,-4} {1,-22} exit {2,3}  {3,5} s  {4}" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $s.name, $c, $sec, "$last".Trim()
        Write-Host $row
        $rows += [pscustomobject]@{ ok = $ok; line = $row }
    }
} finally {
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
}
$nf = @($rows | Where-Object { -not $_.ok }).Count
$verdict = if ($rows.Count -eq 0) { "FAIL (no step ran)" } elseif ($nf -gt 0) { "FAIL ($nf of $($rows.Count) steps failed)" } else { "PASS ($($rows.Count) steps)" }
$summary = @($hdr) + @($rows | ForEach-Object { $_.line }) + @("release gate $Tag`: $verdict")
[System.IO.File]::WriteAllLines((Join-Path $Work "${Tag}_summary.txt"), [string[]]$summary)
Write-Host "release gate $Tag`: $verdict  (logs: $logDir)"
if ($rows.Count -gt 0 -and $nf -eq 0) { "PASS $(Get-Date -Format s) $head" | Set-Content (Join-Path $Work "PASS_$Tag"); exit 0 }
"FAIL $(Get-Date -Format s) $head $verdict" | Set-Content (Join-Path $Work "FAIL_$Tag")
exit 1
