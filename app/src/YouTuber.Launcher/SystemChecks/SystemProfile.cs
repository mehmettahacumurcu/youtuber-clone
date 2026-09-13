namespace YouTuber.Launcher.SystemChecks;

public sealed record SystemProfile(
    int WindowsBuild,
    NvidiaProbeResult Nvidia,
    long PhysicalMemoryBytes,
    IReadOnlyDictionary<string, long> AvailableBytesByDrive,
    string DataDriveRoot,
    string OllamaStoreDriveRoot,
    PrerequisiteStatus Prerequisites);

public sealed record NvidiaGpu(string Name, long VramBytes, string DriverVersion);

public sealed record NvidiaProbeResult(
    bool IsAvailable,
    IReadOnlyList<NvidiaGpu> Gpus,
    string Diagnostic)
{
    public NvidiaGpu? SelectedGpu => Gpus.OrderByDescending(gpu => gpu.VramBytes).FirstOrDefault();

    public static NvidiaProbeResult Success(IReadOnlyList<NvidiaGpu> gpus) => new(
        gpus.Count > 0,
        gpus,
        gpus.Count > 0
            ? string.Join("; ", gpus.Select(gpu => $"{gpu.Name} ({gpu.VramBytes} bytes, driver {gpu.DriverVersion})"))
            : "nvidia-smi returned no GPUs.");

    public static NvidiaProbeResult Failure(string diagnostic) => new(false, [], diagnostic);
}

public sealed record PrerequisiteStatus(
    bool OllamaAvailable,
    bool WebView2Available,
    string OllamaDiagnostic = "",
    string WebView2Diagnostic = "");

public enum PreflightStatus
{
    Pass,
    Warning,
    Block,
}

public sealed record PreflightCheck(
    string Name,
    PreflightStatus Status,
    long? ActualValue,
    long? RequiredValue,
    string Action);

public sealed record PreflightReport(IReadOnlyList<PreflightCheck> Checks)
{
    public PreflightStatus OverallStatus => Checks.Any(check => check.Status == PreflightStatus.Block)
        ? PreflightStatus.Block
        : Checks.Any(check => check.Status == PreflightStatus.Warning)
            ? PreflightStatus.Warning
            : PreflightStatus.Pass;
}
