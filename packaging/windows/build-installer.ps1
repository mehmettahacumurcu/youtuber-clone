[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$')]
    [string] $Version,

    [Parameter(Mandatory = $true)]
    [string] $BootstrapCatalog,

    [switch] $FixtureMode,

    [string] $ArtifactsRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ExpectedInnoVersion = "7.0.2"
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($ArtifactsRoot)) {
    $ArtifactsRoot = Join-Path $RepoRoot 'artifacts\installer'
}
$ArtifactsRoot = [IO.Path]::GetFullPath($ArtifactsRoot)

function Assert-ChildPath([string] $Parent, [string] $Candidate, [string] $Description) {
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    $candidateFull = [IO.Path]::GetFullPath($Candidate)
    if (-not $candidateFull.StartsWith($parentFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "$Description must remain under $Parent"
    }
}

function Invoke-Checked([string] $File, [string[]] $Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File failed with exit code $LASTEXITCODE"
    }
}

function Test-CanonicalHuggingFaceResolveUrl([string] $Value) {
    $uri = $null
    if (-not [Uri]::TryCreate($Value, [UriKind]::Absolute, [ref] $uri)) { return $false }
    if ($uri.Scheme -cne 'https' -or $uri.Host -ine 'huggingface.co' -or
        -not $uri.IsDefaultPort -or $uri.IsLoopback -or $uri.UserInfo.Length -ne 0 -or
        $uri.Query.Length -ne 0 -or $uri.Fragment.Length -ne 0) {
        return $false
    }

    $escapedPath = $uri.GetComponents([UriComponents]::Path, [UriFormat]::UriEscaped)
    $schemeDelimiter = $uri.OriginalString.IndexOf('://', [StringComparison]::Ordinal)
    $originalPathStart = if ($schemeDelimiter -lt 0) { -1 } else { $uri.OriginalString.IndexOf('/', $schemeDelimiter + 3) }
    if ($originalPathStart -lt 0 -or $uri.OriginalString.Substring($originalPathStart) -cne "/$escapedPath") {
        return $false
    }
    $segments = @($escapedPath -split '/', 0, 'SimpleMatch')
    if ($segments.Count -lt 5 -or
        $segments[0] -cnotmatch '^[A-Za-z0-9_.-]+$' -or
        $segments[1] -cnotmatch '^[A-Za-z0-9_.-]+$' -or
        $segments[2] -cne 'resolve' -or
        $segments[3] -cnotmatch '^[0-9a-f]{40}$') {
        return $false
    }

    foreach ($segment in $segments[4..($segments.Count - 1)]) {
        if ([string]::IsNullOrWhiteSpace($segment)) { return $false }
        try { $decoded = [Uri]::UnescapeDataString($segment) }
        catch { return $false }
        if ($decoded -in @('.', '..') -or $decoded.Contains('/') -or $decoded.Contains('\') -or
            [Uri]::EscapeDataString($decoded) -cne $segment) {
            return $false
        }
    }
    return $true
}

function Find-Iscc {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 7\ISCC.exe'),
        (Join-Path $env:ProgramFiles 'Inno Setup 7\ISCC.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 7\ISCC.exe')
    )
    $iscc = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
    if (-not $iscc) { throw 'ISCC.exe from Inno Setup 7 was not found.' }

    $signature = Get-AuthenticodeSignature -LiteralPath $iscc
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Pyrsys B\.V\.') {
        throw 'ISCC.exe does not carry the expected valid official signature.'
    }

    $registered = @(
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*'
    ) | ForEach-Object { Get-ItemProperty $_ -ErrorAction SilentlyContinue } |
        Where-Object {
            $_.PSObject.Properties['DisplayName'] -and $_.PSObject.Properties['DisplayVersion'] -and
            $_.DisplayName -like 'Inno Setup 7*' -and $_.DisplayVersion -eq $ExpectedInnoVersion
        }
    if (-not $registered) { throw "ISCC.exe is not registered as Inno Setup $ExpectedInnoVersion." }
    return [IO.Path]::GetFullPath($iscc)
}

$CatalogPath = [IO.Path]::GetFullPath($BootstrapCatalog)
if (-not (Test-Path -LiteralPath $CatalogPath -PathType Leaf)) { throw "Bootstrap catalog not found: $CatalogPath" }
if (-not $FixtureMode -and $CatalogPath.StartsWith((Join-Path $RepoRoot 'tests\fixtures'), [StringComparison]::OrdinalIgnoreCase)) {
    throw 'A fixture bootstrap catalog requires -FixtureMode.'
}
$catalog = Get-Content -LiteralPath $CatalogPath -Raw -Encoding UTF8 | ConvertFrom-Json
$catalogFields = @($catalog.PSObject.Properties.Name | Sort-Object)
$expectedFields = @('environment', 'manifest_sha256', 'manifest_url', 'schema')
if (Compare-Object $catalogFields $expectedFields) { throw 'Bootstrap catalog fields do not match the frozen contract.' }
if ($catalog.schema -ne 'youtuber.bootstrap-catalog.v1' -or $catalog.environment -ne 'production') {
    throw 'Bootstrap catalog must use the production v1 contract.'
}
if (-not (Test-CanonicalHuggingFaceResolveUrl ([string] $catalog.manifest_url))) {
    throw 'Production bootstrap manifest must use a canonical immutable Hugging Face resolve URL.'
}
if ($catalog.manifest_sha256 -notmatch '^[0-9a-f]{64}$' -or $catalog.manifest_sha256 -match '^0{64}$') {
    throw 'Bootstrap manifest SHA-256 is invalid.'
}

$Iscc = Find-Iscc
$BuildRoot = Join-Path $ArtifactsRoot $Version
$PublishDir = Join-Path $BuildRoot 'publish'
$OutputDir = Join-Path $BuildRoot 'output'
Assert-ChildPath $ArtifactsRoot $BuildRoot 'Build root'
if (Test-Path -LiteralPath $BuildRoot) { Remove-Item -LiteralPath $BuildRoot -Recurse -Force }
New-Item -ItemType Directory -Path $PublishDir, $OutputDir -Force | Out-Null

Push-Location $RepoRoot
try {
    # Release gate: dotnet test
    Invoke-Checked 'dotnet' @('test', 'app/YouTuberStudio.sln', '-c', 'Release', '--no-restore')
    $python = Join-Path $RepoRoot '.venv-dist-dev\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Packaging test Python environment is missing.' }
    Invoke-Checked $python @('-m', 'pytest', 'tests/packaging/test_inno_contract.py', 'tests/packaging/test_launcher_project_contract.py', '-q', '-p', 'no:cacheprovider')

    # Self-contained release: dotnet publish
    $publishArguments = @(
        'publish', 'app/src/YouTuber.Launcher/YouTuber.Launcher.csproj',
        '-c', 'Release', '-r', 'win-x64', '--self-contained', 'true',
        '--no-restore', '-o', $PublishDir,
        '/p:SelfContained=true',
        '/p:UsingDevelopmentBootstrapCatalog=false',
        "/p:BootstrapCatalogPath=$CatalogPath"
    )
    if ($FixtureMode) { $publishArguments += '/p:FixtureMode=true'; $publishArguments += '/p:FixtureHarness=true' }
    Invoke-Checked 'dotnet' $publishArguments
    if ($FixtureMode) {
        Invoke-Checked 'dotnet' @('build', 'app/tests/YouTuber.WorkerFixture/YouTuber.WorkerFixture.csproj', '-c', 'Release', '--no-restore')
        $fixtureOutput = Join-Path $RepoRoot 'app/tests/YouTuber.WorkerFixture/bin/Release/net8.0-windows'
        Copy-Item -Path (Join-Path $fixtureOutput '*') -Destination $PublishDir -Recurse -Force
    }
}
finally { Pop-Location }

$launcher = Join-Path $PublishDir 'YouTuberStudio.exe'
$launcherAssembly = Join-Path $PublishDir 'YouTuberStudio.dll'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf) -or -not (Test-Path -LiteralPath $launcherAssembly -PathType Leaf)) {
    throw 'Self-contained launcher publish output is incomplete.'
}
$assemblyText = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes($launcherAssembly))
if (-not $assemblyText.Contains([string] $catalog.manifest_url)) {
    throw 'Published launcher does not contain the selected production bootstrap manifest URL.'
}
if ($assemblyText.Contains('"environment": "development"') -or $assemblyText.Contains('https://127.0.0.1/manifest.json')) {
    throw 'Published launcher contains the development bootstrap catalog.'
}
$developmentCatalogPath = [IO.Path]::GetFullPath((Join-Path $RepoRoot 'app\src\YouTuber.Launcher\Resources\bootstrap-catalog.json'))
if ($assemblyText.Contains($developmentCatalogPath)) { throw 'Published launcher contains a development catalog path.' }

$iss = Join-Path $PSScriptRoot 'YouTuberStudio.iss'
$assets = Join-Path $PSScriptRoot 'assets'
Invoke-Checked $Iscc @(
    '/Qp',
    "/DAppVersion=$Version",
    "/DPublishDir=$PublishDir",
    "/DAssetsDir=$assets",
    "/DBootstrapCatalog=$CatalogPath",
    "/DInstallerOutputDir=$OutputDir",
    $(if ($FixtureMode) { '/DFixtureMode=1' }),
    $iss
)

$installer = Join-Path $OutputDir "YouTuberStudio-Setup-$Version.exe"
if (-not (Test-Path -LiteralPath $installer -PathType Leaf)) { throw "Installer output is missing: $installer" }
$hash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
$metadataPath = Join-Path $OutputDir 'installer-metadata.json'
$metadata = [ordered]@{
    schema = 'youtuber.installer-metadata.v1'
    version = $Version
    fixture_mode = [bool] $FixtureMode
    filename = [IO.Path]::GetFileName($installer)
    size = (Get-Item -LiteralPath $installer).Length
    sha256 = $hash
    bootstrap_catalog_sha256 = (Get-FileHash -LiteralPath $CatalogPath -Algorithm SHA256).Hash.ToLowerInvariant()
    inno_setup_version = $ExpectedInnoVersion
}
$metadata | ConvertTo-Json | Set-Content -LiteralPath $metadataPath -Encoding UTF8

Write-Host "Installer: $installer"
Write-Host "SHA256:    $hash"
Write-Host "Metadata:  $metadataPath"
