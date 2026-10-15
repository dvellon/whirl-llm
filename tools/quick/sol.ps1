# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1b: speed-of-light calculator (thin wrapper of sol.py; CPU only, seconds).
#   pwsh -File tools\quick\sol.ps1 -Rgp <capture prefix or *_dispatches.csv> [-Device 8s|r97] [-Model dense|moe] [-Ctx N] [-Md]
#   pwsh -File tools\quick\sol.ps1 -Theory -Model dense|moe -Device 8s|r97 -Phase decode|verify|prefill -Ctx N [-Rows 4|2048]
#   pwsh -File tools\quick\sol.ps1 -Fl1a [-Md]
param([string]$Rgp = '', [switch]$Theory, [switch]$Fl1a, [ValidateSet('8s','r97')][string]$Device = '8s',
      [ValidateSet('dense','moe')][string]$Model = 'dense', [ValidateSet('decode','verify','prefill')][string]$Phase = 'decode',
      [int]$Ctx = -1, [int]$Rows = 0, [string]$Kv = 'q8h', [switch]$Md)
$a = @("$PSScriptRoot\sol.py", '--device', $Device, '--model', $Model, '--phase', $Phase, '--kv', $Kv)
if ($Rgp) { $a += @('--rgp', $Rgp) }; if ($Theory) { $a += '--theory' }; if ($Fl1a) { $a += '--fl1a' }
if ($Ctx -ge 0) { $a += @('--ctx', $Ctx) }; if ($Rows -gt 0) { $a += @('--rows', $Rows) }; if ($Md) { $a += '--md' }
& python -I @a; exit $LASTEXITCODE
