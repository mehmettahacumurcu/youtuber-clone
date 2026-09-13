[CmdletBinding()]
param([switch]$Clean, [switch]$PackageOnly, [switch]$StudioOnly)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StudioProject = Join-Path $RepositoryRoot "packaging/runtime/studio"
$VoiceProject = Join-Path $RepositoryRoot "packaging/runtime/voice"
$StudioEnvironment = Join-Path $RepositoryRoot ".venv-build-studio"
$VoiceEnvironment = Join-Path $RepositoryRoot ".venv-build-voice"
$BuildRoot = Join-Path $RepositoryRoot "build/pyinstaller"
$WorkerRoot = Join-Path $RepositoryRoot "dist/workers"
$VendorRoot = Join-Path $RepositoryRoot "build/vendor/applio"

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) { throw "$Operation failed with exit code $LASTEXITCODE" }
}

function Assert-ContainedGeneratedPath([string]$Path, [string[]]$AllowedRoots) {
    $candidate = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $allowed = $AllowedRoots | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\') }
    if ($candidate -notin $allowed) { throw "Refusing to clean path outside declared generated roots: $candidate" }
    if ($candidate -eq [IO.Path]::GetPathRoot($candidate)) { throw "Refusing to clean a filesystem root" }
}

function Assert-NoReparsePoints([string]$Path) {
    $rootItem = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (($rootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to traverse reparse point: $($rootItem.FullName)"
    }
    if (-not $rootItem.PSIsContainer) { return }

    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($rootItem.FullName)
    while ($pending.Count -gt 0) {
        $directory = $pending.Pop()
        foreach ($item in @(Get-ChildItem -LiteralPath $directory -Force -ErrorAction Stop)) {
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing to traverse reparse point: $($item.FullName)"
            }
            if ($item.PSIsContainer) { $pending.Push($item.FullName) }
        }
    }
}

function Remove-GeneratedDirectory([string]$Path) {
    $allowed = @($BuildRoot, $WorkerRoot)
    Assert-ContainedGeneratedPath $Path $allowed
    if (Test-Path -LiteralPath $Path) {
        Assert-NoReparsePoints $Path
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Remove-ForbiddenBundleDebris([string]$Bundle) {
    $allowed = @(
        (Join-Path $WorkerRoot "studio-worker"),
        (Join-Path $WorkerRoot "voice-worker")
    )
    Assert-ContainedGeneratedPath $Bundle $allowed
    if (-not (Test-Path -LiteralPath $Bundle -PathType Container)) {
        throw "Worker bundle was not produced: $Bundle"
    }
    Assert-NoReparsePoints $Bundle
    $forbiddenDirectories = @("test", "tests", "testing", "test_data", "demos", "notebooks", "__pycache__", ".pytest_cache")
    Get-ChildItem -LiteralPath $Bundle -Directory -Recurse -Force |
        Where-Object { $_.Name.ToLowerInvariant() -in $forbiddenDirectories } |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object {
            if (Test-Path -LiteralPath $_.FullName) {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force
            }
        }
    Get-ChildItem -LiteralPath $Bundle -File -Recurse -Force |
        Where-Object { $_.Length -eq 0 } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    Get-ChildItem -LiteralPath $Bundle -File -Recurse -Force |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".ipynb", ".log", ".pyc") } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

Set-Location $RepositoryRoot
$env:PYTHONNOUSERSITE = "1"
if ($Clean -and $PackageOnly) { throw "-Clean and -PackageOnly cannot be combined" }
if ($Clean -and $StudioOnly) { throw "-Clean and -StudioOnly cannot be combined" }
if ($Clean) {
    Remove-GeneratedDirectory $BuildRoot
    Remove-GeneratedDirectory $WorkerRoot
}

$StudioPython = Join-Path $StudioEnvironment "Scripts/python.exe"
$VoicePython = Join-Path $VoiceEnvironment "Scripts/python.exe"
if (-not $PackageOnly) {
    $env:UV_PROJECT_ENVIRONMENT = $StudioEnvironment
    & uv sync --project $StudioProject --frozen
    Assert-NativeSuccess "Studio uv sync"
    if (-not $StudioOnly) {
        $env:UV_PROJECT_ENVIRONMENT = $VoiceEnvironment
        & uv sync --project $VoiceProject --frozen
        Assert-NativeSuccess "Voice uv sync"
    }
    Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue

    if (-not $StudioOnly) {
        & $VoicePython "scripts/fetch_applio_inference.py" --destination $VendorRoot
        Assert-NativeSuccess "Pinned Applio staging"
    }

    New-Item -ItemType Directory -Force -Path $BuildRoot, $WorkerRoot | Out-Null
    & $StudioPython -m PyInstaller --noconfirm --clean --distpath $WorkerRoot --workpath (Join-Path $BuildRoot "studio") "packaging/pyinstaller/studio-worker.spec"
    Assert-NativeSuccess "Studio PyInstaller build"
    if (-not $StudioOnly) {
        & $VoicePython -m PyInstaller --noconfirm --clean --distpath $WorkerRoot --workpath (Join-Path $BuildRoot "voice") "packaging/pyinstaller/voice-worker.spec"
        Assert-NativeSuccess "Voice PyInstaller build"
    }
}

$StudioBundle = Join-Path $WorkerRoot "studio-worker"
$VoiceBundle = Join-Path $WorkerRoot "voice-worker"
Remove-ForbiddenBundleDebris $StudioBundle
if (-not $StudioOnly) {
    Remove-ForbiddenBundleDebris $VoiceBundle
}
& $StudioPython -m scripts.write_component_manifest --bundle $StudioBundle --component studio_runtime --entrypoint studio-worker.exe --zip (Join-Path $WorkerRoot "studio-worker.zip")
Assert-NativeSuccess "Studio component manifest and ZIP"
$Bundles = @($StudioBundle)
if (-not $StudioOnly) {
    & $StudioPython -m scripts.write_component_manifest --bundle $VoiceBundle --component voice_runtime --entrypoint voice-worker.exe --zip (Join-Path $WorkerRoot "voice-worker.zip")
    Assert-NativeSuccess "Voice component manifest and ZIP"
    $Bundles += $VoiceBundle
}

$Verification = "from pathlib import Path; from distribution.models import ComponentManifest; from distribution.hashing import verify_component_tree; root=Path(r'__ROOT__'); manifest=ComponentManifest.model_validate_json((root/'component-manifest.json').read_text(encoding='utf-8')); verify_component_tree(root, manifest)"
foreach ($Bundle in $Bundles) {
    & $StudioPython -c ($Verification.Replace("__ROOT__", $Bundle))
    Assert-NativeSuccess "Component tree verification"
}

Write-Host "Built deterministic workers under $WorkerRoot"
