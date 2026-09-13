using YouTuber.Launcher.Ollama;
using YouTuber.Launcher.Prerequisites;

namespace YouTuber.Launcher.Processes;

/// <summary>Starts the one-shot port composition produced by <see cref="WorkerRuntimeFactory"/>.</summary>
public sealed class WorkerRuntimeOrchestrator(OllamaManager ollama, WorkerSupervisor workers) : IAsyncDisposable
{
    private bool _ollamaStarted;

    public async Task<InstallerRunResult> InstallAndStartAsync(
        OllamaInstallerArtifact artifact,
        WorkerRuntime runtime,
        IInstallerRunner runner,
        string sessionSecret,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        ArgumentNullException.ThrowIfNull(runtime);
        ArgumentNullException.ThrowIfNull(runner);
        var result = await ollama.InstallNewComposedAsync(
            artifact,
            runtime.Ollama,
            runtime.OllamaReservation,
            runner,
            cancellationToken);
        if (result.TimedOut || result.ExitCode != 0) return result;
        _ollamaStarted = true;
        try
        {
            await workers.StartAsync(runtime.Voice, runtime.Studio, sessionSecret, runtime.Settings, cancellationToken);
        }
        catch
        {
            await StopOwnedOllamaAfterFailureAsync();
            throw;
        }
        return result;
    }

    public async Task StartAsync(VerifiedOllamaServer server, WorkerRuntime runtime, string sessionSecret, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(server);
        ArgumentNullException.ThrowIfNull(runtime);
        await ollama.StartControlledServeAsync(server, runtime.Ollama, runtime.OllamaReservation, cancellationToken);
        _ollamaStarted = true;
        try
        {
            await workers.StartAsync(runtime.Voice, runtime.Studio, sessionSecret, runtime.Settings, cancellationToken);
        }
        catch
        {
            await StopOwnedOllamaAfterFailureAsync();
            throw;
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            await workers.StopAsync(cancellationToken);
        }
        finally
        {
            if (_ollamaStarted)
            {
                await ollama.StopControlledServeAsync(CancellationToken.None);
                _ollamaStarted = false;
            }
        }
    }

    private async Task StopOwnedOllamaAfterFailureAsync()
    {
        if (!_ollamaStarted) return;
        try
        {
            await ollama.StopControlledServeAsync(CancellationToken.None);
        }
        finally
        {
            _ollamaStarted = false;
        }
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            await workers.DisposeAsync();
        }
        finally
        {
            await StopOwnedOllamaAfterFailureAsync();
        }
    }
}
