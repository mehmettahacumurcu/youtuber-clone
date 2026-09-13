[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateScript({ Test-Path $_ -PathType Leaf })][string]$Installer,
    [Parameter(Mandatory)][string]$DataRoot,
    [Parameter(Mandatory)][uri]$FixtureManifestUrl,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{64}$')][string]$FixtureManifestSha256,
    [Parameter(Mandatory)][ValidateSet('success','failure')][string]$ExpectedOutcome,
    [Parameter(Mandatory)][ValidateRange(30,3600)][int]$TimeoutSeconds,
    [ValidateSet('fresh-install','interrupted-download','corrupted-component','insufficient-disk','unsupported-gpu','missing-webview2','missing-ollama','update-rollback','app-only-uninstall','full-owned-data-uninstall')][string]$Scenario = 'fresh-install',
    [Parameter(Mandatory)][string]$EvidenceRoot
)

$ErrorActionPreference = 'Stop'
if ($Scenario -notin @('fresh-install','app-only-uninstall','full-owned-data-uninstall')) { throw "Scenario '$Scenario' requires an external clean-VM/hardware fixture and is not implemented locally." }
$Installer = [IO.Path]::GetFullPath($Installer)
$DataRoot = [IO.Path]::GetFullPath($DataRoot)
$EvidenceRoot = [IO.Path]::GetFullPath($EvidenceRoot)
New-Item -ItemType Directory -Force -Path $EvidenceRoot | Out-Null

function Get-FileReceipt([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return @{ path = $Path; sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
}

function Get-Inventory([string]$Root) {
    if (-not (Test-Path -LiteralPath $Root)) { return @() }
    return @(Get-ChildItem -LiteralPath $Root -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        @{ path = $_.FullName.Substring($Root.TrimEnd('\').Length).TrimStart('\'); size = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
}

# Deliberately contains no repository-relative lookup: after this point the installed app and fixture URL are the only inputs.
$started = Get-Date
$installRoot = Join-Path $EvidenceRoot 'installed-app'
$installerLog = Join-Path $EvidenceRoot 'inno-install.log'
$launcher = Join-Path $installRoot 'YouTuberStudio.exe'
$launcherProcess = $null
$installerProcess = $null
$workerPids = @()
$workerPorts = @()
$cleanup = @()
$preUninstall = $null
$failureMessage = $null
$evidencePath = Join-Path $EvidenceRoot 'clean-install-evidence.json'
$emptyReceipts = @()
$evidence = [ordered]@{
    schema = 'youtuber.clean-install.v1'; scenario = $Scenario; expected_outcome = $ExpectedOutcome; installer = $Installer; data_root = $DataRoot
    fixture_manifest_url = $FixtureManifestUrl.AbsoluteUri; fixture_manifest_sha256 = $FixtureManifestSha256
    started_utc = $started.ToUniversalTime().ToString('o'); finished_utc = $null; installer_exit_code = $null
    failure = $null; 'launcher-logs' = @(); 'worker-logs' = @(); inventory = @(); versions = $emptyReceipts; hashes = $emptyReceipts; ports = @(); 'process-cleanup' = @()
    pre_uninstall = $null; uninstall = $null
}

try {
    $installArguments = @('/VERYSILENT', '/SUPPRESSMSGBOXES', ('/DIR="' + $installRoot + '"'), ("/LOG=$installerLog"))
    $previousFixtureDataRoot = $env:YOUTUBER_FIXTURE_DATA_ROOT
    try {
        $env:YOUTUBER_FIXTURE_DATA_ROOT = $DataRoot
        $installerProcess = Start-Process -FilePath $Installer -ArgumentList $installArguments -PassThru
    }
    finally {
        $env:YOUTUBER_FIXTURE_DATA_ROOT = $previousFixtureDataRoot
    }
    if (-not $installerProcess.WaitForExit($TimeoutSeconds * 1000)) { Stop-Process -Id $installerProcess.Id -Force; throw "Installer exceeded timeout; see $installerLog" }
    $evidence.installer_exit_code = $installerProcess.ExitCode
    if (($ExpectedOutcome -eq 'success' -and $installerProcess.ExitCode -ne 0) -or ($ExpectedOutcome -eq 'failure' -and $installerProcess.ExitCode -eq 0)) { throw 'Installer outcome did not match expectation.' }
    if ($ExpectedOutcome -eq 'failure') { return }
    if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) { throw 'Fixture installer did not install the launcher.' }

    $fixtureArguments = @('--fixture-workers', "--data-root=$DataRoot", "--fixture-manifest-url=$FixtureManifestUrl", "--fixture-manifest-sha256=$FixtureManifestSha256")
    $launcherProcess = Start-Process -FilePath $launcher -ArgumentList $fixtureArguments -PassThru
    $readyMarker = Join-Path $DataRoot 'state\studio-ready.marker'
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while (-not (Test-Path -LiteralPath $readyMarker) -and (Get-Date) -lt $deadline -and -not $launcherProcess.HasExited) { Start-Sleep -Milliseconds 100 }
    if (-not (Test-Path -LiteralPath $readyMarker)) { throw 'Fixture launcher did not reach durable worker readiness.' }
    $voicePidFile = Join-Path $DataRoot 'state\voice.pid'; $studioPidFile = Join-Path $DataRoot 'state\studio.pid'
    $voicePortFile = Join-Path $DataRoot 'state\voice.port'; $studioPortFile = Join-Path $DataRoot 'state\studio.port'
    foreach ($receipt in @($voicePidFile,$studioPidFile,$voicePortFile,$studioPortFile)) { if (-not (Test-Path -LiteralPath $receipt)) { throw "Missing worker receipt: $receipt" } }
    $workerPids = @([int](Get-Content -LiteralPath $voicePidFile), [int](Get-Content -LiteralPath $studioPidFile))
    $workerPorts = @([int](Get-Content -LiteralPath $voicePortFile), [int](Get-Content -LiteralPath $studioPortFile))

    Stop-Process -Id $launcherProcess.Id -ErrorAction SilentlyContinue
    $launcherProcess.WaitForExit(10000) | Out-Null
    $cleanupDeadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $cleanupDeadline -and (@($workerPids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0)) { Start-Sleep -Milliseconds 100 }
    foreach ($workerPid in $workerPids) { if (Get-Process -Id $workerPid -ErrorAction SilentlyContinue) { throw "Worker $workerPid survived launcher cleanup." } }
    while ((Get-Date) -lt $cleanupDeadline -and (@($workerPorts | Where-Object { Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue }).Count -gt 0)) { Start-Sleep -Milliseconds 100 }
    foreach ($port in $workerPorts) { if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "Worker port $port remained open." } }

    if ($Scenario -in @('app-only-uninstall','full-owned-data-uninstall')) {
        $state = Join-Path $DataRoot 'state'; New-Item -ItemType Directory -Force -Path $state | Out-Null
        $owned = Join-Path $DataRoot 'owned.txt'; $neighbor = Join-Path $DataRoot 'unowned-neighbor.txt'
        Set-Content -LiteralPath $owned -Value 'owned'; Set-Content -LiteralPath $neighbor -Value 'preserve'
        @{ schema = 'youtuber.ownership.v1'; install_id = 'f324e3a6-7c5d-4bdf-92dd-f00a307b3f1e'; owned_relative_paths = @('owned.txt') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $state 'ownership.json') -Encoding utf8
    }

    $preUninstall = [ordered]@{
        inventory = Get-Inventory $installRoot
        versions = @($Installer,$launcher | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { @{ path = $_; version = [Diagnostics.FileVersionInfo]::GetVersionInfo($_).FileVersion } })
        hashes = @($Installer,$launcher,(Join-Path $installRoot 'YouTuber.WorkerFixture.dll'),(Join-Path $installRoot 'resources\bootstrap-catalog.json') | ForEach-Object { Get-FileReceipt $_ } | Where-Object { $null -ne $_ })
    }
    $evidence.inventory = $preUninstall.inventory; $evidence.versions = $preUninstall.versions; $evidence.hashes = $preUninstall.hashes; $evidence.pre_uninstall = $preUninstall

    $uninstaller = Get-ChildItem -LiteralPath $installRoot -Filter 'unins*.exe' | Select-Object -First 1
    if ($null -eq $uninstaller) { throw 'Installed Inno uninstaller is missing.' }
    $uninstallArguments = @('/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART')
    if ($Scenario -eq 'full-owned-data-uninstall') { $uninstallArguments += '/REMOVEOWNEDDATA' }
    $uninstallProcess = Start-Process -FilePath $uninstaller.FullName -ArgumentList $uninstallArguments -PassThru -Wait
    $evidence.uninstall = @{ executable = $uninstaller.FullName; arguments = $uninstallArguments; exit_code = $uninstallProcess.ExitCode }
    if ($uninstallProcess.ExitCode -ne 0) { throw 'Installed uninstaller failed.' }
    if ((Test-Path -LiteralPath $launcher) -or (Test-Path -LiteralPath $uninstaller.FullName)) { throw 'Installed uninstaller did not remove launcher binaries and itself.' }
    if ($Scenario -in @('app-only-uninstall','full-owned-data-uninstall')) {
        if (-not (Test-Path -LiteralPath $neighbor)) { throw 'Unowned neighbor was deleted.' }
        if ($Scenario -eq 'app-only-uninstall' -and -not (Test-Path -LiteralPath $owned)) { throw 'App-only uninstall removed owned data.' }
        if ($Scenario -eq 'full-owned-data-uninstall' -and (Test-Path -LiteralPath $owned)) { throw 'Full owned-data uninstall retained owned data.' }
    }
}
catch {
    $failureMessage = $_.Exception.ToString()
    throw
}
finally {
    if ($null -ne $launcherProcess -and -not $launcherProcess.HasExited) { Stop-Process -Id $launcherProcess.Id -ErrorAction SilentlyContinue; $launcherProcess.WaitForExit(10000) | Out-Null }
    foreach ($workerPid in $workerPids) {
        $running = Get-Process -Id $workerPid -ErrorAction SilentlyContinue
        if ($null -ne $running) { Stop-Process -Id $workerPid -Force -ErrorAction SilentlyContinue }
        $cleanup += @{ pid = $workerPid; exited = -not [bool](Get-Process -Id $workerPid -ErrorAction SilentlyContinue) }
    }
    $evidence.finished_utc = (Get-Date).ToUniversalTime().ToString('o')
    $evidence.failure = $failureMessage
    $evidence.'launcher-logs' = @(Get-ChildItem -LiteralPath $DataRoot -Recurse -File -Filter '*.log' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    $evidence.'worker-logs' = @(Get-ChildItem -LiteralPath $DataRoot -Recurse -File -Include 'voice.*.log','studio.*.log' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    $evidence.ports = @($workerPorts | ForEach-Object { @{ port = $_; listener_absent = -not [bool](Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue) } })
    $evidence.'process-cleanup' = $cleanup
    $evidence | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $evidencePath -Encoding utf8
}
