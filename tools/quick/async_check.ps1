# SPDX-License-Identifier: Apache-2.0
# FASTLANE / REF-MINE T0-A: golden suites with and without HIP_LAUNCH_BLOCKING=1 must be byte-identical (see async_check.py).
#   pwsh -File tools\quick\async_check.ps1 -Device r97|8s [-Models dense,moe] [-Modes precise,balance,fast] [-Exe whirl.exe]
param([ValidateSet('r97','8s')][string]$Device = 'r97', [string]$Models = 'dense,moe', [string]$Modes = 'precise,balance,fast',
      [string]$Suites = 'single,pair,ngram', [string]$Exe = '')
$a = @("$PSScriptRoot\async_check.py", '--device', $Device, '--models', $Models, '--modes', $Modes, '--suites', $Suites)
if ($Exe) { $a += @('--exe', $Exe) }
& python -I @a; exit $LASTEXITCODE
