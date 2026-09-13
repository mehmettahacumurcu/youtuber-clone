# Clean-machine release test

Run only in Windows Sandbox or a clean VM with a local fixture artifact server. The external gate is a supported GPU/hardware matrix; this repository harness does not fake that evidence.

```powershell
& scripts\test_clean_install.ps1 -Installer artifacts\installer\0.0.1-test\output\YouTuberStudio-Setup-0.0.1-test.exe -DataRoot C:\YouTuberData -FixtureManifestUrl http://127.0.0.1:8080/manifest.json -FixtureManifestSha256 <64-lowercase-hex> -ExpectedOutcome success -TimeoutSeconds 900 -EvidenceRoot C:\evidence
```

Run fresh install, interrupted download, corrupted component, insufficient disk, unsupported GPU, missing WebView2, missing Ollama, update rollback, app-only uninstall, and full owned-data uninstall. Archive the generated evidence JSON with fixture-server logs.
