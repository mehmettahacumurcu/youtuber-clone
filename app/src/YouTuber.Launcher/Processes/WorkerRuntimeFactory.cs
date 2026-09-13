using System.IO;
using YouTuber.Launcher.Ollama;

namespace YouTuber.Launcher.Processes;

/// <summary>Production launch composition: one retained reservation per local listener.</summary>
public sealed class WorkerRuntimeFactory : IDisposable
{
    private readonly LauncherPorts _ports;
    public WorkerRuntimeFactory(LauncherPorts? ports = null) => _ports = ports ?? LauncherPorts.Create();
    public WorkerRuntime Create(string dataRoot, WorkerDefinition voice, WorkerDefinition studio)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataRoot);
        var normalizedDataRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(dataRoot));
        var workers = _ports.Compose(voice, studio);
        var settings = new Dictionary<string, string>(workers.Settings, StringComparer.Ordinal)
        {
            ["YOUTUBER_DATA_ROOT"] = normalizedDataRoot,
            ["YOUTUBER_CACHE_ROOT"] = normalizedDataRoot,
        };
        var configuredVoice = ConfigureWorker(workers.Voice, normalizedDataRoot, settings);
        var configuredStudio = ConfigureWorker(workers.Studio, normalizedDataRoot, settings);
        return new WorkerRuntime(
            OllamaInstallation.New(normalizedDataRoot, _ports.OllamaPort),
            _ports.OllamaReservation,
            _ports.RagReservation,
            configuredVoice,
            configuredStudio,
            settings);
    }

    private static WorkerDefinition ConfigureWorker(WorkerDefinition worker, string dataRoot, IReadOnlyDictionary<string, string> settings)
    {
        var installRoot = Path.TrimEndingDirectorySeparator(Path.GetFullPath(
            worker.WorkingDirectory ?? Path.GetDirectoryName(Path.GetFullPath(worker.ExecutablePath)) ?? dataRoot));
        var environment = new Dictionary<string, string>(settings, StringComparer.Ordinal)
        {
            ["YOUTUBER_INSTALL_ROOT"] = installRoot,
        };
        return worker.WithEnvironment(environment);
    }
    public void Dispose() => _ports.Dispose();
}

public sealed record WorkerRuntime(
    OllamaInstallation Ollama,
    PortReservation OllamaReservation,
    PortReservation RagReservation,
    WorkerDefinition Voice,
    WorkerDefinition Studio,
    IReadOnlyDictionary<string, string> Settings);
