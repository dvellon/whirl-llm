# REQUANT-1: start whirl-server on the 8060S, run wtests.py, stop it (only the PID we started).
#   pwsh -File serve_test.ps1 -Model M.gguf -Tag NAME [-Mmproj P] [-Mode balance] [-Plain] [-TextOnly]
param([Parameter(Mandatory)][string]$Model, [Parameter(Mandatory)][string]$Tag, [string]$Mmproj = '', [string]$Mode = 'balance', [switch]$Plain, [switch]$TextOnly)
$ErrorActionPreference = 'Stop'
$exe = 'C:\ClaudeWorkspace\whirl-work\wt_rel020f\build\Release\whirl-server.exe'
$LG = 'C:\ClaudeWorkspace\models\requant1\logs\wsrv'; New-Item -ItemType Directory -Force $LG | Out-Null
$log = "$LG\$Tag.server.log"; $port = 8097; $url = "http://127.0.0.1:$port"
$sa = @($Model, '--port', $port, '--log-file', $log, '--kv-ssd-mode', 'off', '-np', 1, '--ctx-per-slot', 32768, "--$Mode")
if ($Mmproj) { $sa += @('--mmproj', $Mmproj) }
$env:WHIRL_DEVICE = '8060s'
if ($Plain) { $env:WHIRL_MTP = '0'; $env:WHIRL_NGRAM = '0' } else { Remove-Item Env:WHIRL_MTP, Env:WHIRL_NGRAM -ErrorAction SilentlyContinue }
$proc = Start-Process -FilePath $exe -ArgumentList $sa -PassThru -WindowStyle Minimized
"$($proc.Id) $(Get-Date -Format s) whirl-server $Tag" | Add-Content 'C:\ClaudeWorkspace\whirl-work\progress\requant1.pids'
try {
  $dead = (Get-Date).AddSeconds(900)
  while ($true) {
    if ($proc.HasExited) { throw "server exited (code $($proc.ExitCode)); see $log" }
    try { $h = Invoke-WebRequest "$url/health" -TimeoutSec 5 -SkipHttpErrorCheck; if ($h.StatusCode -eq 200) { break } } catch { }
    if ((Get-Date) -gt $dead) { throw 'server not ready after 900 s' }
    Start-Sleep -Seconds 2
  }
  if (-not (Select-String -Path $log -Pattern 'gfx1151' -Quiet)) { throw "device check: gfx1151 not in $log" }
  $a = @('C:\ClaudeWorkspace\whirl-work\wt_requant1\tools\requant1\wtests.py', $port, "$LG\$Tag.json")
  if ($TextOnly) { $a += '--text-only' }
  & 'C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe' @a
  if ($LASTEXITCODE -ne 0) { throw "wtests failed $LASTEXITCODE" }
} finally {
  if (-not $proc.HasExited) {
    pwsh -NoProfile -File 'C:\ClaudeWorkspace\whirl-cloud\tools\rel013\stop_server.ps1' -ProcessId $proc.Id
    if (-not $proc.WaitForExit(60000)) { Stop-Process -Id $proc.Id -Force }
  }
  Remove-Item Env:WHIRL_MTP, Env:WHIRL_NGRAM -ErrorAction SilentlyContinue
}
