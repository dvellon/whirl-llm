# SPDX-License-Identifier: Apache-2.0
# FASTLANE FL-1b: automatic gate cascade (thin wrapper of gate.py; see there for the stages and rules).
#   pwsh -File tools\quick\gate.ps1 -Device r97|8s [-Exe DIR|whirl.exe] [-Worktree DIR -Build [-TouchCpp]] [-Mode balance]
#        [-WEnv 'K=V;K2=V2'] [-Check bitwise|nmse] [-Kt families|none] [-Models dense[,moe]] [-Stages build,bitwise,kl,perf] [-Tag x]
#   pwsh -File tools\quick\gate.ps1 -Device 8s -RecordPerfRef [-Mode balance]
param([ValidateSet('r97','8s')][string]$Device = 'r97', [string]$Exe = '', [string]$Worktree = '', [switch]$Build, [switch]$TouchCpp,
      [ValidateSet('precise','balance','fast')][string]$Mode = 'balance', [string]$WEnv = '', [ValidateSet('bitwise','nmse')][string]$Check = 'bitwise',
      [string]$Kt = '', [string]$Models = 'dense', [string]$Suites = 'single,pair,ngram', [string]$Stages = 'build,bitwise,kl,perf',
      [double]$PerfTol = 4.0, [switch]$RecordPerfRef, [string]$Tag = 'cand')
$a = @("$PSScriptRoot\gate.py", '--device', $Device, '--mode', $Mode, '--check', $Check, '--models', $Models, '--suites', $Suites,
       '--stages', $Stages, '--perf-tol', $PerfTol, '--tag', $Tag)
if ($Exe) { $a += @('--exe', $Exe) }; if ($Worktree) { $a += @('--worktree', $Worktree) }; if ($Build) { $a += '--build' }
if ($TouchCpp) { $a += '--touch-cpp' }; if ($WEnv) { $a += @('--env', $WEnv) }; if ($Kt) { $a += @('--kt', $Kt) }
if ($RecordPerfRef) { $a += '--record-perf-ref' }
& python -I @a; exit $LASTEXITCODE
