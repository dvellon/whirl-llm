# SPDX-License-Identifier: Apache-2.0
# FASTLANE: RGP capture + per-dispatch CSV + per-kernel summary of one WHIRL prefill chunk / decode steps (see rgp_capture.py).
#   pwsh -File tools\quick\rgp_capture.ps1 -Device r97|8s -Phase decode|prefill|both [-Ctx 2048] [-Mode balance] [-Exe whirl.exe] [-Buffer high] [-WEnv 'K=V;K2=V2']
param([ValidateSet('r97','8s')][string]$Device = 'r97', [ValidateSet('decode','prefill','both')][string]$Phase = 'both',
      [int]$Ctx = 2048, [string]$Mode = 'balance', [string]$Exe = '', [string]$Buffer = 'high', [string]$WEnv = '', [switch]$NoMtp)
$a = @("$PSScriptRoot\rgp_capture.py", '--device', $Device, '--phase', $Phase, '--ctx', $Ctx, '--mode', $Mode, '--buffer', $Buffer)
if ($Exe) { $a += @('--exe', $Exe) }; if ($WEnv) { $a += @('--env', $WEnv) }; if ($NoMtp) { $a += '--no-mtp' }
& python -I @a; exit $LASTEXITCODE
