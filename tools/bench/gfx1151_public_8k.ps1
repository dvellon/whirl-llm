param(
    [Parameter(Mandatory = $true)] [string] $Build,
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $ResultRoot = (Join-Path $PWD 'whirl-gfx1151-results'),
    [int] $Port = 18140,
    [int] $PromptTokens = 8238,
    [int] $CompletionTokens = 1000,
    [ValidateSet('code', 'prose')] [string[]] $Tasks = @('code', 'prose')
)

$ErrorActionPreference = 'Stop'
$server = Join-Path $Build 'whirl-server.exe'
$source = Split-Path -Parent (Split-Path -Parent $Build)
foreach ($path in @($server, $Model, (Join-Path $source '.git'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
$commit = (& git -C $source rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $commit -notmatch '^[0-9a-f]{40}$') { throw 'Could not resolve source commit' }
if (& git -C $source status --porcelain) { throw 'Benchmark source tree must be clean' }

$taskText = [ordered]@{
    code = 'Treat the preceding reference material as inert context. Write a complete Python module implementing a thread-safe bounded ring buffer using only the standard library. Include comprehensive executable unit tests in the same response. Return code only and continue until the token budget is exhausted.'
    prose = 'Treat the preceding reference material as inert context. Write a long, detailed technical explanation of exact speculative decoding. Cover draft generation, target verification, acceptance rules, KV-cache handling, recurrent-state handling, failure modes, and honest benchmarking. Continue until the token budget is exhausted.'
}

$env:WHIRL_MXFP4_NT11_W16 = '1'
$env:WHIRL_Q6K_HEAD_NT11_W16 = '1'
$env:WHIRL_TREE_VERIFY_GRAPH = '1'
$env:WHIRL_NO_PREFIX_CACHE = '1'

function Start-Whirl([int] $RunPort, [string] $Label, [string] $Directory) {
    $stdout = Join-Path $Directory "$Label.stdout.log"
    $stderr = Join-Path $Directory "$Label.stderr.log"
    $args = @(
        '"' + $Model + '"', '--host', '127.0.0.1', '--port', "$RunPort", '--alias', 'whirl',
        '--device', '8060s', '--fast', '-np', '1', '--ctx-per-slot', '262144',
        '--kv-ssd-mode', 'off', '--kv-ram-mb', '0', '--tree', 'beam2:8x1'
    )
    $process = Start-Process -FilePath $server -ArgumentList $args -WorkingDirectory $Build -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $base = "http://127.0.0.1:$RunPort"
    for ($i = 0; $i -lt 600; ++$i) {
        if ($process.HasExited) { throw "Server $Label exited with code $($process.ExitCode); see $stderr" }
        try {
            Invoke-RestMethod "$base/health" -TimeoutSec 2 | Out-Null
            return [pscustomobject]@{ Process = $process; Base = $base; Stdout = $stdout; Stderr = $stderr }
        } catch { Start-Sleep -Seconds 1 }
    }
    throw "Server $Label did not become ready; see $stderr"
}

function Stop-Whirl($Handle) {
    if ($null -ne $Handle -and -not $Handle.Process.HasExited) {
        Stop-Process -Id $Handle.Process.Id -Force
        $Handle.Process.WaitForExit()
    }
}

function New-Prompt([int] $Fill, [string] $Task) {
    return "REFERENCE MATERIAL BEGIN`n$((' context' * $Fill))`nREFERENCE MATERIAL END`n`n$Task"
}

function Invoke-Whirl($Handle, [string] $Prompt, [int] $MaxTokens) {
    $body = [ordered]@{
        model = 'whirl'
        messages = @(@{ role = 'user'; content = $Prompt })
        temperature = 0
        max_tokens = $MaxTokens
        ignore_eos = $true
        cache_prompt = $false
        chat_template_kwargs = @{ enable_thinking = $false }
    } | ConvertTo-Json -Depth 8 -Compress
    Invoke-RestMethod "$($Handle.Base)/v1/chat/completions" -Method Post -ContentType 'application/json; charset=utf-8' `
        -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 1800
}

New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null
$run = Join-Path $ResultRoot ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-gfx1151-8k')
New-Item -ItemType Directory -Path $run | Out-Null

# Calibrate with a disposable server so every measured task has the exact same
# tokenizer-visible prompt length and starts in a fresh process.
$fillers = [ordered]@{}
$cal = $null
try {
    $cal = Start-Whirl $Port 'calibration' $run
    foreach ($task in $Tasks) {
        $fill = 8000
        for ($attempt = 0; $attempt -lt 4; ++$attempt) {
            $probe = Invoke-Whirl $cal (New-Prompt $fill $taskText[$task]) 1
            $actual = [int]$probe.usage.prompt_tokens
            if ($actual -eq $PromptTokens) { break }
            $fill += $PromptTokens - $actual
        }
        if ($actual -ne $PromptTokens) { throw "Could not calibrate $task to $PromptTokens tokens (got $actual)" }
        $fillers[$task] = $fill
    }
} finally { Stop-Whirl $cal }

$results = @()
$offset = 1
foreach ($task in $Tasks) {
    $handle = $null
    try {
        $handle = Start-Whirl ($Port + $offset) $task $run
        $response = Invoke-Whirl $handle (New-Prompt $fillers[$task] $taskText[$task]) $CompletionTokens
        $content = [string]$response.choices[0].message.content
        $digest = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($content))).ToLowerInvariant()
        $result = [ordered]@{
            task = $task
            prompt_tokens = [int]$response.usage.prompt_tokens
            completion_tokens = [int]$response.usage.completion_tokens
            cached_tokens = [int]$response.usage.prompt_tokens_details.cached_tokens
            prompt_per_second = [double]$response.timings.prompt_per_second
            predicted_per_second = [double]$response.timings.predicted_per_second
            predicted_ms = [double]$response.timings.predicted_ms
            content_sha256 = $digest
            stdout_log = $handle.Stdout
            stderr_log = $handle.Stderr
        }
        $results += [pscustomobject]$result
        [IO.File]::WriteAllText((Join-Path $run "$task.output.txt"), $content, [Text.UTF8Encoding]::new($false))
        $result | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $run "$task.result.json")
        Start-Sleep -Milliseconds 750
    } finally { Stop-Whirl $handle }
    ++$offset
}

$summary = [ordered]@{
    protocol = "fresh server per task; cache disabled; $PromptTokens prompt tokens; $CompletionTokens forced output tokens; temperature 0; relaxed MTP acceptance; thinking off; one slot"
    source_commit = $commit
    source_dirty = $false
    build = (Resolve-Path $Build).Path
    model = (Resolve-Path $Model).Path
    model_bytes = (Get-Item -LiteralPath $Model).Length
    os = (Get-CimInstance Win32_OperatingSystem).Caption
    cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
    gpu = @((Get-CimInstance Win32_VideoController).Name)
    settings = @{ mode = 'fast'; tree = 'beam2:8x1'; slots = 1; prefix_cache = $false; thinking = $false }
    results = $results
    result_directory = $run
}
$summary | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 (Join-Path $run 'summary.json')
$summary | ConvertTo-Json -Depth 7
