using YouTuber.Launcher.Processes;

namespace YouTuber.Launcher.Web;

public sealed record LauncherSessionPlan(
    string DataRoot,
    WorkerDefinition Voice,
    WorkerDefinition Studio,
    IReadOnlyDictionary<string, string> Settings,
    string SessionSecret,
    Uri? OllamaEndpoint = null,
    IDisposable? Resources = null) : IDisposable
{
    public void Dispose() => Resources?.Dispose();
}

public interface IStudioShellHost
{
    void AttachWorkerSupervisor(IWorkerSupervisor supervisor);
    Task StartStudioAsync(StudioOrigin origin, string dataRoot, string sessionSecret, CancellationToken cancellationToken = default);
    void ShowStartupFailure(string message);
}

public sealed class LauncherStartupController(IWorkerSupervisor workers)
{
    public async Task StartAsync(IStudioShellHost shell, LauncherSessionPlan plan, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(shell);
        ArgumentNullException.ThrowIfNull(plan);
        SessionSecret.Decode(plan.SessionSecret);
        shell.AttachWorkerSupervisor(workers);
        await workers.StartAsync(plan.Voice, plan.Studio, plan.SessionSecret, plan.Settings, cancellationToken);
        if (workers.State != WorkerSupervisorState.Running) throw new InvalidOperationException("Local workers did not reach the running state.");
        await shell.StartStudioAsync(new StudioOrigin(plan.Studio.HealthEndpoint.Port), plan.DataRoot, plan.SessionSecret, cancellationToken);
    }

    public async Task AttachRunningAsync(IStudioShellHost shell, LauncherSessionPlan plan, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(shell);
        ArgumentNullException.ThrowIfNull(plan);
        SessionSecret.Decode(plan.SessionSecret);
        if (workers.State != WorkerSupervisorState.Running) throw new InvalidOperationException("The retained production runtime is not running.");
        shell.AttachWorkerSupervisor(workers);
        await shell.StartStudioAsync(new StudioOrigin(plan.Studio.HealthEndpoint.Port), plan.DataRoot, plan.SessionSecret, cancellationToken);
    }
}
