param(
    [string]$Source = "E:\youtuber-clone",
    [string]$Destination = "E:\youtuber-studio",
    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"
$Source = [IO.Path]::GetFullPath($Source)
$Destination = [IO.Path]::GetFullPath($Destination)
$Parent = Split-Path -Parent $Destination
$Leaf = Split-Path -Leaf $Destination
$Stage = Join-Path $Parent "$Leaf.staging"
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Backup = Join-Path $Parent "$Leaf.previous-$Stamp"

if ($Source -eq $Destination -or $Source.StartsWith("$Destination\", [StringComparison]::OrdinalIgnoreCase)) {
    throw "Destination cannot equal or contain Source"
}
if ((Split-Path -Parent $Stage) -ne $Parent -or (Split-Path -Parent $Backup) -ne $Parent) {
    throw "Stage and backup must remain direct siblings of Destination"
}

$RuntimeFiles = @(
    "__init__.py", "embed.py", "retrieve.py", "store.py", "types.py", "prompt.py",
    "expand.py", "card_manifest.py", "candidates.py", "context.py", "verifier.py",
    "evidence_pipeline.py", "retrieve_server.py"
)
$DataJobs = @(
    @("$Source\data\clean", "$Stage\data\clean"),
    @("$Source\data\rag\qdrant", "$Stage\data\rag\qdrant"),
    @("$Source\data\tts\wavs", "$Stage\data\tts\wavs"),
    @("$Source\models\xtts_speaker_v2", "$Stage\models\xtts_speaker_v2")
)

Write-Host "Source:      $Source"
Write-Host "Stage:       $Stage"
Write-Host "Destination: $Destination"
Write-Host "Rollback:    $Backup"
Write-Host "RAG modules: $($RuntimeFiles -join ', ')"
if ($PlanOnly) {
    Write-Host "PLAN ONLY -- no files or processes changed"
    return
}

if (Test-Path -LiteralPath $Stage) {
    $resolvedStage = [IO.Path]::GetFullPath($Stage)
    if ((Split-Path -Parent $resolvedStage) -ne $Parent -or -not (Split-Path -Leaf $resolvedStage).EndsWith(".staging")) {
        throw "Refusing to remove unsafe stage path $resolvedStage"
    }
    Remove-Item -LiteralPath $resolvedStage -Recurse -Force
}
New-Item -ItemType Directory -Force $Stage, "$Stage\ui", "$Stage\rag", "$Stage\pipeline", "$Stage\logs" | Out-Null

Copy-Item -LiteralPath "$Source\ui\speaker_studio.py" -Destination "$Stage\ui\" -Force
Copy-Item -LiteralPath "$Source\ui\chat_runtime.py" -Destination "$Stage\ui\" -Force
Copy-Item -LiteralPath "$Source\ui\__init__.py" -Destination "$Stage\ui\" -Force
foreach ($file in $RuntimeFiles) {
    Copy-Item -LiteralPath "$Source\rag\$file" -Destination "$Stage\rag\" -Force
}
foreach ($file in "__init__.py", "config.py") {
    Copy-Item -LiteralPath "$Source\pipeline\$file" -Destination "$Stage\pipeline\" -Force
}
Copy-Item -LiteralPath "$Source\config.yaml" -Destination $Stage -Force
Copy-Item -LiteralPath "$Source\START_SPEAKER.bat" -Destination $Stage -Force
Copy-Item -LiteralPath "$Source\START_SPEAKER_PAYLAS.bat" -Destination $Stage -Force
Copy-Item -LiteralPath "$Source\docs\RUNBOOK.md" -Destination "$Stage\README.md" -Force

foreach ($job in $DataJobs) {
    robocopy $job[0] $job[1] /MIR /NFL /NDL /NJH /R:2 /W:2 | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed ($($job[0]) -> $($job[1])) with exit $LASTEXITCODE"
    }
}

$CatalogSource = "$Source\data\cards\speaker_cards.json"
$ManifestSource = "$Source\data\cards\speaker_cards.index_manifest.json"
$CardsEnabled = (Test-Path -LiteralPath $CatalogSource) -and (Test-Path -LiteralPath $ManifestSource)
if ($CardsEnabled) {
    New-Item -ItemType Directory -Force "$Stage\data\cards" | Out-Null
    Copy-Item -LiteralPath $CatalogSource -Destination "$Stage\data\cards\speaker_cards.json" -Force
    Copy-Item -LiteralPath $ManifestSource -Destination "$Stage\data\cards\speaker_cards.index_manifest.json" -Force
    $Python = "$Source\.venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "RAG Python is missing: $Python"
    }
    $Relocate = @'
import sys
from pathlib import Path
from rag.card_manifest import relocate_card_manifest, validate_card_hint_manifest
catalog, manifest, store = map(Path, sys.argv[1:4])
relocate_card_manifest(
    catalog_path=catalog,
    manifest_path=manifest,
    destination_store_path=store,
    collection="speaker_cards",
    embedder="BAAI/bge-m3",
)
status = validate_card_hint_manifest(
    catalog_path=catalog,
    manifest_path=manifest,
    collection="speaker_cards",
    embedder="BAAI/bge-m3",
    store_path=store,
)
if not status.usable:
    raise SystemExit(f"relocated card manifest invalid: {status.reason}")
'@
    $env:PYTHONPATH = $Source
    & $Python -c $Relocate "$Stage\data\cards\speaker_cards.json" "$Stage\data\cards\speaker_cards.index_manifest.json" "$Destination\data\rag\qdrant"
    if ($LASTEXITCODE -ne 0) {
        throw "card manifest validation or relocation failed"
    }
    Write-Host "Validated card pair staged; cards remain search hints only"
} else {
    Write-Host "Card catalog/manifest pair unavailable; runnable will report clean-only hint fallback"
}

$MustExist = @(
    "$Stage\ui\speaker_studio.py", "$Stage\ui\chat_runtime.py",
    "$Stage\rag\retrieve_server.py", "$Stage\rag\evidence_pipeline.py",
    "$Stage\rag\verifier.py", "$Stage\pipeline\config.py", "$Stage\config.yaml",
    "$Stage\START_SPEAKER.bat", "$Stage\README.md",
    "$Stage\models\xtts_speaker_v2\best_model.pth", "$Stage\data\rag\qdrant", "$Stage\data\clean"
)
$Missing = $MustExist | Where-Object { -not (Test-Path -LiteralPath $_) }
if ($Missing) {
    throw "Staging verification failed; missing: $($Missing -join ', ')"
}
foreach ($directory in "$Stage\data\rag\qdrant", "$Stage\data\clean", "$Stage\data\tts\wavs") {
    if (-not (Get-ChildItem -LiteralPath $directory -Recurse -File | Select-Object -First 1)) {
        throw "Staging verification failed; empty: $directory"
    }
}

foreach ($port in 7861, 7670) {
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) {
        continue
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
    $commandLine = [string]$process.CommandLine
    $executablePath = [string]$process.ExecutablePath
    $known = $commandLine.Contains("speaker_studio.py") -or $commandLine.Contains("rag.retrieve_server")
    $scoped = (
        $commandLine.Contains($Source) -or $commandLine.Contains($Destination) -or
        $executablePath.Contains($Source) -or $executablePath.Contains($Destination)
    )
    if (-not ($known -and $scoped)) {
        throw "Port $port belongs to unknown process $($listener.OwningProcess): $commandLine"
    }
    Stop-Process -Id $listener.OwningProcess -Force -Confirm:$false
}

$BackupCreated = $false
if (Test-Path -LiteralPath $Destination) {
    Move-Item -LiteralPath $Destination -Destination $Backup
    $BackupCreated = $true
}
try {
    Move-Item -LiteralPath $Stage -Destination $Destination
} catch {
    $SwapFailure = $_
    if ($BackupCreated -and -not (Test-Path -LiteralPath $Destination) -and (Test-Path -LiteralPath $Backup)) {
        try {
            Move-Item -LiteralPath $Backup -Destination $Destination
        } catch {
            throw "Runnable swap failed and rollback failed. Stage error: $SwapFailure. Rollback error: $_"
        }
        throw "Runnable swap failed; previous runnable restored: $SwapFailure"
    }
    throw "Runnable swap failed before a rollback was available: $SwapFailure"
}
Write-Host "BUILD OK -- $Destination"
if (Test-Path -LiteralPath $Backup) {
    Write-Host "Rollback retained at $Backup"
}
