# REQUANT-1: llama.cpp b11719 Vulkan llama-server on Vulkan0 (8060S) with the projector, run wtests.py, stop (our PID only).
param([Parameter(Mandatory)][string]$Model, [Parameter(Mandatory)][string]$Mmproj, [Parameter(Mandatory)][string]$Tag)
$ErrorActionPreference = 'Stop'
$dir = 'C:\ClaudeWorkspace\tools\llamacpp_fn_bin\vulkan'; $LG = 'C:\ClaudeWorkspace\models\requant1\logs\wsrv'; $port = 8096
$log = "$LG\$Tag.llama.log"
$sa = @('-m', $Model, '--mmproj', $Mmproj, '--device', 'Vulkan0', '-ngl', 99, '-c', 8192, '-np', 1, '-fa', 'on', '--port', $port, '--host', '127.0.0.1', '--jinja', '--no-webui', '-lv', 4,
        '--spec-type', 'draft-mtp', '--spec-draft-n-max', 3, '--spec-draft-n-min', 0, '--spec-draft-p-min', 0.3)
$proc = Start-Process -FilePath "$dir\llama-server.exe" -ArgumentList $sa -WorkingDirectory $dir -PassThru -WindowStyle Minimized -RedirectStandardError $log
"$($proc.Id) $(Get-Date -Format s) llama-server $Tag" | Add-Content 'C:\ClaudeWorkspace\whirl-work\progress\requant1.pids'
try {
  $dead = (Get-Date).AddSeconds(600)
  while ($true) {
    if ($proc.HasExited) { throw "llama-server exited ($($proc.ExitCode)); see $log" }
    try { $h = Invoke-WebRequest "http://127.0.0.1:$port/health" -TimeoutSec 5 -SkipHttpErrorCheck; if ($h.StatusCode -eq 200) { break } } catch { }
    if ((Get-Date) -gt $dead) { throw 'not ready' }
    Start-Sleep 2
  }
  if (-not (Select-String -Path $log -Pattern '8060S' -Quiet)) { throw "device check: 8060S not in $log" }
  & 'C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe' 'C:\ClaudeWorkspace\whirl-work\wt_requant1\tools\requant1\wtests.py' $port "$LG\$Tag.json"
} finally { if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; $null = $proc.WaitForExit(30000) } }
