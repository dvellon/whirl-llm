# SPDX-License-Identifier: Apache-2.0
# CPU-only self-test of tools/golden/golden.ps1 (no GPU, no model): runs the gate against
# fake_whirl.cmd and a small fake model file in a temporary golden root and checks its verdicts.
#   pwsh tools/golden/golden_selftest.ps1      (exit 0 = all cases as expected, 1 = a case failed)
# Cases (REVIEW-RC4-C P3-4 / P3-5):
#   - record + check of a normal run: exit 0
#   - a run with only "G header" + "G info" (no step lines): exit 2, against any reference
#   - a reference with only model + header + info lines: exit 2 (it has no step lines)
#   - a record run where one case fails: exit 2 and no reference / models.txt is written
$ErrorActionPreference = "Stop"
$gate = Join-Path $PSScriptRoot "golden.ps1"
$fake = Join-Path $PSScriptRoot "fake_whirl.cmd"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("whirl-golden-selftest-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
$model = Join-Path $tmp "fake-model.gguf"
[System.IO.File]::WriteAllBytes($model, [byte[]](1..200 | ForEach-Object { [byte]($_ % 256) }))
$savedDense = $env:WHIRL_GOLDEN_DENSE
$env:WHIRL_GOLDEN_DENSE = $model
$bad = 0

function RunGate([string]$action, [string]$root, [hashtable]$fakeEnv) {
    foreach ($k in @("FAKEGOLD_FAIL_SUITE", "FAKEGOLD_NOSTEPS")) { [Environment]::SetEnvironmentVariable($k, $null) }
    foreach ($k in $fakeEnv.Keys) { [Environment]::SetEnvironmentVariable($k, $fakeEnv[$k]) }
    $env:WHIRL_GOLDEN_DENSE = $model
    $out = & pwsh -NoProfile -File $gate $action -Models dense -Modes precise -Suites "single,pair" -Exe $fake `
        -GoldenRoot $root -Out (Join-Path $tmp "out") 2>&1 | ForEach-Object { "$_" }
    $rc = $LASTEXITCODE
    foreach ($k in @("FAKEGOLD_FAIL_SUITE", "FAKEGOLD_NOSTEPS")) { [Environment]::SetEnvironmentVariable($k, $null) }
    return @{ rc = $rc; out = $out }
}
function Expect([string]$name, $r, [int]$want, [scriptblock]$extra = { $true }) {
    $ok = ($r.rc -eq $want) -and (& $extra)
    Write-Host ("[{0}] {1} (exit {2}, want {3})" -f ($(if ($ok) { "PASS" } else { "FAIL" })), $name, $r.rc, $want)
    if (-not $ok) { $r.out | Select-Object -Last 8 | ForEach-Object { Write-Host "      $_" }; $script:bad++ }
}

try {
    # 1) normal record + check
    $g1 = Join-Path $tmp "g1"
    $r = RunGate "record" $g1 @{}
    Expect "record of a normal run" $r 0 { (Test-Path (Join-Path $g1 "gfx1201\dense\precise_single.txt")) -and (Test-Path (Join-Path $g1 "models.txt")) }
    $r = RunGate "check" $g1 @{}
    Expect "check of the same run" $r 0
    # 2) run without step lines vs a normal reference
    $r = RunGate "check" $g1 @{ FAKEGOLD_NOSTEPS = "1" }
    Expect "run with header + info only (no step lines) is an error" $r 2
    # 3) reference without step lines (model + header + info) vs a run without step lines
    foreach ($s in @("single", "pair")) {
        $f = Join-Path $g1 "gfx1201\dense\precise_$s.txt"
        $l = @(Get-Content $f)
        [System.IO.File]::WriteAllLines($f, [string[]]@($l[0..2]))
    }
    $r = RunGate "check" $g1 @{ FAKEGOLD_NOSTEPS = "1" }
    Expect "reference with model + header + info only is an error" $r 2
    # 4) record with one failing case: nothing is written
    $g2 = Join-Path $tmp "g2"
    $r = RunGate "record" $g2 @{ FAKEGOLD_FAIL_SUITE = "pair" }
    Expect "record with a failing case writes no reference and no models.txt" $r 2 {
        -not (Test-Path (Join-Path $g2 "gfx1201\dense\precise_single.txt")) -and -not (Test-Path (Join-Path $g2 "models.txt")) }
} finally {
    $env:WHIRL_GOLDEN_DENSE = $savedDense
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
if ($bad -gt 0) { Write-Host "golden selftest: $bad case(s) failed"; exit 1 }
Write-Host "golden selftest: all cases as expected"
exit 0
