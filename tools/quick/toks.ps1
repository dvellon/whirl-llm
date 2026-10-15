# SPDX-License-Identifier: Apache-2.0
# FASTLANE: 5-minute tok/s quick test (decode at 2k / 32k, prefill at 8k; REL3 prompts and client, see toks.py).
#   pwsh -File tools\quick\toks.ps1 -Device r97|8s [-Mode balance] [-Exe whirl.exe] [-Lengths 2048,8192,32768] [-WEnv 'WHIRL_KV=q8h'] [-Tag x]
param([ValidateSet('r97','8s')][string]$Device = 'r97', [string]$Mode = 'balance', [string]$Exe = '',
      [string]$Lengths = '2048,8192,32768', [string]$WEnv = '', [string]$Tag = '')
$a = @("$PSScriptRoot\toks.py", '--device', $Device, '--mode', $Mode, '--lengths', $Lengths)
if ($Exe) { $a += @('--exe', $Exe) }; if ($WEnv) { $a += @('--env', $WEnv) }; if ($Tag) { $a += @('--tag', $Tag) }
& python -I @a; exit $LASTEXITCODE
