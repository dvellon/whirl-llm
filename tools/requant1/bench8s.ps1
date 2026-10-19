# REQUANT-1 speed check (8060S): one whirl-server start, b8s.py sweep (2k / 32k, prefill + decode, warm-up), stop.
#   pwsh -File bench8s.ps1 -Model M.gguf -Label old|new -ModelName swift|ornith -Round N [-Mode balance]
param([Parameter(Mandatory)][string]$Model, [Parameter(Mandatory)][string]$Label, [Parameter(Mandatory)][string]$ModelName,
      [string]$Round = '1', [string]$Mode = 'balance', [string]$Lengths = '2048,32768', [switch]$Plain)
$ErrorActionPreference = 'Stop'
$exe = 'C:\ClaudeWorkspace\whirl-work\wt_rel020f\build\Release\whirl-server.exe'
$LG = 'C:\ClaudeWorkspace\models\requant1\logs\bench'; New-Item -ItemType Directory -Force $LG | Out-Null
$OUT = "$LG\results_$ModelName.tsv"
$name = "$(Get-Date -Format 'yyyyMMdd-HHmmss')_${ModelName}_${Label}_r$Round"
$log = "$LG\$name.server.log"; $port = 8098; $url = "http://127.0.0.1:$port"
$env:WHIRL_DEVICE = '8060s'
Remove-Item Env:WHIRL_MTP, Env:WHIRL_NGRAM -ErrorAction SilentlyContinue
if ($Plain) { $env:WHIRL_MTP = '0'; $env:WHIRL_NGRAM = '0'; $Label = "$Label-plain" }
$sa = @($Model, '--port', $port, '--log-file', $log, '--kv-ssd-mode', 'off', '-np', 1, '--ctx-per-slot', 40960, "--$Mode")
"WHIRL_DEVICE=8060s $exe $($sa -join ' ')" | Set-Content "$LG\$name.cmd.txt"
$proc = Start-Process -FilePath $exe -ArgumentList $sa -PassThru -WindowStyle Minimized
"$($proc.Id) $(Get-Date -Format s) whirl-server $name" | Add-Content 'C:\ClaudeWorkspace\whirl-work\progress\requant1.pids'
try {
  $dead = (Get-Date).AddSeconds(900)
  while ($true) {
    if ($proc.HasExited) { throw "server exited (code $($proc.ExitCode)); see $log" }
    try { $h = Invoke-WebRequest "$url/health" -TimeoutSec 5 -SkipHttpErrorCheck; if ($h.StatusCode -eq 200) { break } } catch { }
    if ((Get-Date) -gt $dead) { throw 'server not ready after 900 s' }
    Start-Sleep -Seconds 2
  }
  & 'C:\ClaudeWorkspace\tools\pyref\Scripts\python.exe' 'C:\ClaudeWorkspace\whirl-work\base8s\b8s.py' sweep --url $url --engine "whirl020-$Label" `
     --model $ModelName --mode $Mode --out $OUT --round $Round --lengths $Lengths --max-tokens 256 --expect-device gfx1151 --server-log $log
  if ($LASTEXITCODE -ne 0) { throw "b8s.py exit $LASTEXITCODE" }
} finally {
  if (-not $proc.HasExited) {
    pwsh -NoProfile -File 'C:\ClaudeWorkspace\whirl-cloud\tools\rel013\stop_server.ps1' -ProcessId $proc.Id
    if (-not $proc.WaitForExit(60000)) { Stop-Process -Id $proc.Id -Force }
  }
}
