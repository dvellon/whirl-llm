# SPDX-License-Identifier: Apache-2.0
# Golden-hash regression gate (developers). Runs `whirl golden` for every
# {model} x {mode} x {suite} and records the "G ..." lines under
# tests/golden/<arch>/<model>/<mode>_<suite>.txt, or diffs a new run against them.
#
#   pwsh tools/golden/golden.ps1 check  [-Arch gfx1201|gfx1151] [-Models dense,moe] [-Modes precise,balance,fast] [-Suites single,pair]
#   pwsh tools/golden/golden.ps1 record [-Arch ...] ...
#
# Exit code: 0 all match, 1 mismatch (first differing step per file printed), 2 a run failed / setup error.
# Model files: $env:WHIRL_GOLDEN_DENSE / $env:WHIRL_GOLDEN_MOE, else the default LM Studio paths below.
# The executable: -Exe, else build\Release\whirl.exe of this checkout.
# A golden file stores the model file name and size; check refuses a different model. The model is
# also identified by a hash of its size + first and last 4 MiB (tests/golden/models.txt, written by record).
# The gate cannot pass vacuously: an empty model / mode / suite list, a run without "G" lines, an
# empty reference file, or a suite ngram run that never drafted from the n-gram table (ng_cycles=0)
# is an error (exit 2); a step line is a "G" line other than G model / G header / G info. record writes the
# references and models.txt only when every case ran. CPU self-test: tools/golden/golden_selftest.ps1. The runner clears every WHIRL_* environment variable (a stray WHIRL_HIP_DEVICE
# or WHIRL_MODE would change what is measured; --device selects the GPU) and restores them on exit.
# Suites: single, pair, ngram (forced n-gram MTP drafts on a copy-the-text prompt); all three are the
# default for both architectures.
# record needs -KlReport: the JSON report of a passing KL gate run (tools/klgate/klgate.py: balance and fast
# vs precise at 8k / 32k / 96k / 128k) made with the same whirl.exe (SHA-256), on the same GPU family, covering
# every model being recorded. Without it record writes nothing and exits 2 (docs/dev/testing.md, FIX-TEST-B).
param(
    [Parameter(Mandatory = $true, Position = 0)][ValidateSet("check", "record")][string]$Action,
    [ValidateSet("gfx1201", "gfx1151")][string]$Arch = "gfx1201",
    [string[]]$Models = @("dense", "moe"),
    [string[]]$Modes = @("precise", "balance", "fast"),
    [string[]]$Suites = @(),
    [string]$Exe = "",
    [string]$Out = "",  # check: also keep the new outputs here (default: a temp folder)
    [string]$GoldenRoot = "",  # reference tree (default tests\golden of this checkout; golden_selftest.ps1 uses a temp one)
    [string]$KlReport = ""  # record: a passing tools/klgate report (--report) of this exe on this arch, for these models
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Exe) { $Exe = Join-Path $root "build\Release\whirl.exe" }
if (-not (Test-Path $Exe)) { Write-Host "golden: no executable at $Exe (build first or pass -Exe)"; exit 2 }
# accept comma lists given as one string (pwsh -File passes "a,b" as one element)
$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Modes = @($Modes | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Suites = @($Suites | ForEach-Object { $_ -split "," } | Where-Object { $_ })
if ($Suites.Count -eq 0 -and -not $PSBoundParameters.ContainsKey("Suites")) {
    $Suites = @("single", "pair", "ngram")
}
if ($Models.Count -eq 0 -or $Modes.Count -eq 0 -or $Suites.Count -eq 0) {
    Write-Host "golden: empty model / mode / suite list (models=$($Models.Count) modes=$($Modes.Count) suites=$($Suites.Count)): nothing would be checked"
    exit 2
}
foreach ($s in $Suites) { if ($s -notin @("single", "pair", "ngram")) { Write-Host "golden: unknown suite '$s'"; exit 2 } }

$lms = Join-Path $env:USERPROFILE ".lmstudio\models"
$modelPath = @{
    dense = if ($env:WHIRL_GOLDEN_DENSE) { $env:WHIRL_GOLDEN_DENSE } else { Join-Path $lms "unsloth\Qwen3.8-27B-GGUF\Qwen3.8-27B-UD-Q4_K_M.gguf" }
    moe   = if ($env:WHIRL_GOLDEN_MOE) { $env:WHIRL_GOLDEN_MOE } else { Join-Path $lms "ornith-ai\Ornith-1.5-35B-A3B-MXFP4\Ornith-1.5-35B-A3B-MXFP4.gguf" }
}
# FIX-TEST-B: no reference is recorded without a passing KL gate report for this exe / arch / models
function KlReportError([string]$path) {
    if (-not $path) { return "no -KlReport (run tools/klgate/klgate.py --report on this exe first)" }
    if (-not (Test-Path $path)) { return "KL report $path not found" }
    try { $r = Get-Content $path -Raw | ConvertFrom-Json } catch { return "KL report $path is not JSON" }
    if ($r.pass -ne $true) { return "KL report $path did not pass ($($r.summary))" }
    $h = (Get-FileHash -Algorithm SHA256 $Exe).Hash.ToLower()
    if ("$($r.exe_sha256)".ToLower() -ne $h) { return "KL report $path is for another exe (sha256 $($r.exe_sha256), this exe $h)" }
    $is8060 = "$($r.device)" -eq "8060s"
    if (($Arch -eq "gfx1151") -ne $is8060) { return "KL report $path was measured on device '$($r.device)', not on $Arch" }
    foreach ($md in @("balance", "fast")) { if ($md -notin @($r.modes)) { return "KL report $path has no $md cases" } }
    foreach ($L in @(8192, 32768, 98304, 131072)) { if ($L -notin @($r.lengths)) { return "KL report $path does not cover length $L" } }
    $names = @($r.models.PSObject.Properties | ForEach-Object { "$($_.Value)" })
    foreach ($m in $Models) {
        $b = Split-Path -Leaf $modelPath[$m]
        if ($b -notin $names) { return "KL report $path does not cover model $b" }
    }
    return ""
}
if ($Action -eq "record") {
    $klErr = KlReportError $KlReport
    if ($klErr) { Write-Host "golden: record refused: $klErr"; exit 2 }
    Write-Host "golden: KL gate report $KlReport passed for this exe ($Arch)"
}

# (model paths above were read from WHIRL_GOLDEN_*; now drop every WHIRL_* variable, restored in the finally below)
$savedEnv = @{}
foreach ($v in @(Get-ChildItem Env: | Where-Object { $_.Name -like "WHIRL_*" -and $_.Name -ne "WHIRL_R9700_LOCK_HELD" })) {
    $savedEnv[$v.Name] = $v.Value
    [Environment]::SetEnvironmentVariable($v.Name, $null)
}
if (-not $env:AMD_LOG_LEVEL) { $env:AMD_LOG_LEVEL = "1" }
if (-not $GoldenRoot) { $GoldenRoot = Join-Path $root "tests\golden" }
$goldDir = Join-Path $GoldenRoot $Arch
if (-not $Out) { $Out = Join-Path ([System.IO.Path]::GetTempPath()) "whirl-golden-$Arch" }
New-Item -ItemType Directory -Force $Out | Out-Null

# first differing field of two "G ..." lines, e.g. "lh 1a2b... -> 3c4d..."
function FieldDiff([string]$a, [string]$b) {
    $fa = $a -split " "; $fb = $b -split " "
    $d = @()
    for ($i = 0; $i -lt [Math]::Max($fa.Count, $fb.Count); $i++) {
        $x = if ($i -lt $fa.Count) { $fa[$i] } else { "" }
        $y = if ($i -lt $fb.Count) { $fb[$i] } else { "" }
        if ($x -ne $y) {
            $kx = ($x -split "=", 2); $ky = ($y -split "=", 2)
            if ($kx[0] -eq $ky[0] -and $kx.Count -eq 2 -and $kx[0] -eq "ids") {
                # token lists: first differing position
                $ta = $kx[1] -split ","; $tb = $ky[1] -split ","
                $p = 0
                while ($p -lt $ta.Count -and $p -lt $tb.Count -and $ta[$p] -eq $tb[$p]) { $p++ }
                $ea = if ($p -lt $ta.Count) { $ta[$p] } else { "(end)" }
                $eb = if ($p -lt $tb.Count) { $tb[$p] } else { "(end)" }
                $d += "ids differ at token $p ($ea -> $eb)"
            } else {
                $d += "$x -> $y"
            }
        }
    }
    return ($d -join "; ")
}

# size + first and last 4 MiB, SHA-256 (a renamed or re-quantized file of the same size still differs)
function ModelHash([string]$path) {
    $fs = [System.IO.File]::OpenRead($path)
    try {
        $len = $fs.Length
        $n = [int][Math]::Min([long]4194304, $len)
        $head = New-Object byte[] $n
        [void]$fs.Read($head, 0, $n)
        $tail = New-Object byte[] $n
        [void]$fs.Seek($len - $n, [System.IO.SeekOrigin]::Begin)
        [void]$fs.Read($tail, 0, $n)
        $all = New-Object byte[] (2 * $n + 8)
        [Array]::Copy($head, 0, $all, 0, $n)
        [Array]::Copy($tail, 0, $all, $n, $n)
        [Array]::Copy([BitConverter]::GetBytes([long]$len), 0, $all, 2 * $n, 8)
        return ([BitConverter]::ToString([System.Security.Cryptography.SHA256]::HashData($all)) -replace "-", "").ToLower()
    } finally { $fs.Dispose() }
}
$hashFile = Join-Path $GoldenRoot "models.txt"
$hashes = @{}
if (Test-Path $hashFile) { foreach ($l in (Get-Content $hashFile)) { $p = $l -split " "; if ($p.Count -ge 4) { $hashes[$p[0]] = $p } } }
$hashesChanged = $false
# step lines: every "G ..." line except the model / header / info lines (a file or run without one proves nothing)
function StepCount([string[]]$l) { return @($l | Where-Object { $_ -like "G *" -and $_ -notmatch "^G (model|header|info)( |$)" }).Count }
# record: references are staged here and written with models.txt only when every case ran
$staged = [ordered]@{}

$fail = 0; $err = 0; $ran = 0; $t0 = Get-Date
try {
foreach ($mk in $Models) {
    $mp = $modelPath[$mk]
    if (-not $mp -or -not (Test-Path $mp)) { Write-Host "golden: model '$mk' not found ($mp)"; $err++; continue }
    $mi = Get-Item $mp
    $mline = "G model file=$($mi.Name) bytes=$($mi.Length)"
    $mh = ModelHash $mp
    if ($Action -eq "record") {
        $hashes[$mk] = @($mk, $mi.Name, "$($mi.Length)", $mh); $hashesChanged = $true
    } elseif (-not $hashes.ContainsKey($mk)) {
        Write-Host "golden: no recorded hash for model '$mk' in $hashFile (run record)"; $err++; continue
    } elseif ($hashes[$mk][1] -ne $mi.Name -or $hashes[$mk][2] -ne "$($mi.Length)" -or $hashes[$mk][3] -ne $mh) {
        Write-Host "golden: model '$mk' ($mp) is not the recorded file (name/size/hash differ from $hashFile)"; $err++; continue
    }
    foreach ($mode in $Modes) {
        foreach ($suite in $Suites) {
            $name = "$mk/${mode}_$suite"
            $ts = Get-Date
            $raw = & $Exe golden $mp --suite $suite "--$mode" --device $Arch 2>&1 | ForEach-Object { "$_" }
            $rc = $LASTEXITCODE
            $sec = ((Get-Date) - $ts).TotalSeconds
            $glines = @($raw | Where-Object { $_ -like "G *" })
            $lines = @($mline) + $glines
            $new = Join-Path $Out "$mk`_${mode}_$suite.txt"
            [System.IO.File]::WriteAllLines($new, [string[]]$lines)
            $diag = @($raw | Where-Object { $_ -notlike "G *" })
            if ($rc -ne 0 -or ($diag -match "hipError")) {
                Write-Host ("FAIL  {0,-24} whirl exit {1} after {2:N0} s" -f $name, $rc, $sec)
                $diag | Select-Object -Last 15 | ForEach-Object { Write-Host "      $_" }
                $err++
                if ($diag -match "hipError.*719|719.*hip|hip.*719") { Write-Host "golden: hipError 719, stopping"; exit 2 }
                continue
            }
            # a run that printed (almost) nothing must not become, or be compared with, a reference
            if ($glines.Count -lt 2 -or -not ($glines[0] -like "G header *") -or (StepCount $glines) -lt 1) {
                Write-Host ("FAIL  {0,-24} no golden output ({1} G lines, {2} step lines)" -f $name, $glines.Count, (StepCount $glines))
                $diag | Select-Object -Last 10 | ForEach-Object { Write-Host "      $_" }
                $err++
                continue
            }
            if ($suite -eq "ngram") {
                $ng = @($glines | Where-Object { $_ -match "ng_cycles=(\d+)" } | ForEach-Object { [int]$Matches[1] })
                if ($ng.Count -eq 0 -or ($ng | Measure-Object -Maximum).Maximum -le 0) {
                    Write-Host ("FAIL  {0,-24} the n-gram path was not exercised (ng_cycles=0): the case proves nothing" -f $name)
                    $err++
                    continue
                }
            }
            $ran++
            $gold = Join-Path $goldDir "$mk\${mode}_$suite.txt"
            if ($Action -eq "record") {
                $staged[$gold] = [string[]]$lines
                Write-Host ("REC   {0,-24} {1} lines, {2:N0} s (staged)" -f $name, $lines.Count, $sec)
                continue
            }
            if (-not (Test-Path $gold)) { Write-Host "MISS  $name (no golden file $gold; run record)"; $err++; continue }
            $ref = @(Get-Content $gold)
            if ((StepCount $ref) -lt 1) { Write-Host "FAIL  $name golden file $gold is empty or has no step lines"; $err++; continue }
            $bad = $null
            for ($i = 0; $i -lt [Math]::Max($ref.Count, $lines.Count); $i++) {
                $x = if ($i -lt $ref.Count) { $ref[$i] } else { "(missing)" }
                $y = if ($i -lt $lines.Count) { $lines[$i] } else { "(missing)" }
                if ($x -ne $y) { $bad = $i; break }
            }
            if ($null -eq $bad) {
                Write-Host ("OK    {0,-24} {1} lines, {2:N0} s" -f $name, $lines.Count, $sec)
            } else {
                $fail++
                $x = if ($bad -lt $ref.Count) { $ref[$bad] } else { "(missing)" }
                $y = if ($bad -lt $lines.Count) { $lines[$bad] } else { "(missing)" }
                $step = (($x -split " ")[1])
                $ndiff = 0
                for ($i = 0; $i -lt [Math]::Max($ref.Count, $lines.Count); $i++) {
                    if (($i -lt $ref.Count) -and ($i -lt $lines.Count) -and $ref[$i] -eq $lines[$i]) { continue }
                    $ndiff++
                }
                Write-Host ("DIFF  {0,-24} first at line {1} ({2}); {3} of {4} lines differ" -f $name, ($bad + 1), $step, $ndiff, $ref.Count)
                Write-Host "      golden: $x"
                Write-Host "      now:    $y"
                Write-Host "      change: $(FieldDiff $x $y)"
                Write-Host "      (full new output: $new)"
            }
        }
    }
}
} finally {
    foreach ($k in $savedEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $savedEnv[$k]) }
}
# record: the references and models.txt are written together, and only when no case failed (a partial
# record would leave new references next to a stale model hash)
if ($Action -eq "record" -and $err -eq 0) {
    foreach ($gold in $staged.Keys) {
        New-Item -ItemType Directory -Force (Split-Path $gold) | Out-Null
        [System.IO.File]::WriteAllLines($gold, $staged[$gold])
    }
    if ($hashesChanged) {
        New-Item -ItemType Directory -Force (Split-Path $hashFile) | Out-Null
        [System.IO.File]::WriteAllLines($hashFile, [string[]]@($hashes.Keys | Sort-Object | ForEach-Object { $hashes[$_] -join " " }))
    }
} elseif ($Action -eq "record" -and $staged.Count -gt 0) {
    Write-Host "golden record: $err case(s) failed: nothing written ($($staged.Count) staged reference(s) discarded, models.txt unchanged)"
}
$tot = ((Get-Date) - $t0).TotalSeconds
if ($ran -eq 0) { Write-Host "golden: no case ran: refusing to report success"; exit 2 }
if ($err -gt 0) { Write-Host ("golden {0} {1}: {2} run(s) failed, {3} mismatch(es), {4:N0} s" -f $Action, $Arch, $err, $fail, $tot); exit 2 }
if ($fail -gt 0) { Write-Host ("golden check {0}: {1} mismatch(es), {2:N0} s" -f $Arch, $fail, $tot); exit 1 }
Write-Host ("golden {0} {1}: ok, {2:N0} s" -f $Action, $Arch, $tot)
exit 0
