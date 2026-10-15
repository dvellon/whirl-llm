# SPDX-License-Identifier: Apache-2.0
# FASTLANE: KL quick check (dense, code + zh, 2k + 8k, one 512-row window each, mode vs precise; see kl_quick.py).
#   pwsh -File tools\quick\kl_quick.ps1 -Device r97|8s [-Mode balance|fast|balance,fast] [-Exe whirl.exe] [-WEnv 'WHIRL_KV=q8h'] [-Prompts code,zh]
param([ValidateSet('r97','8s')][string]$Device = 'r97', [string]$Mode = 'balance', [string]$Exe = '', [string]$WEnv = '',
      [string]$Prompts = 'code,zh', [string]$Lengths = '2048,8192', [string]$Tag = '')
$a = @("$PSScriptRoot\kl_quick.py", '--device', $Device, '--modes', $Mode, '--prompts', $Prompts, '--lengths', $Lengths)
if ($Exe) { $a += @('--exe', $Exe) }; if ($WEnv) { $a += @('--env', $WEnv) }; if ($Tag) { $a += @('--tag', $Tag) }
& python -I @a; exit $LASTEXITCODE
