using System.Net;
using System.Net.Http.Headers;
using System.Net.Http;

namespace YouTuber.Launcher.Processes;

public interface IWorkerHealthClient
{
    Task WaitForLiveAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken);
    Task WaitForReadyAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken);
}

public sealed class HttpWorkerHealthClient : IWorkerHealthClient, IDisposable
{
    private readonly HttpClient _http;
    public HttpWorkerHealthClient(HttpMessageHandler? handler = null) => _http = new HttpClient(handler ?? new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false }, disposeHandler: true);
    public Task WaitForLiveAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken)
        => WaitForHealthAsync(definition, "health/live", secret, timeout, cancellationToken);
    public Task WaitForReadyAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken)
        => WaitForHealthAsync(definition, "health/ready", secret, timeout, cancellationToken);

    private async Task WaitForHealthAsync(WorkerDefinition definition, string relativePath, string secret, TimeSpan timeout, CancellationToken cancellationToken)
    {
        var deadline = DateTimeOffset.UtcNow + timeout;
        while (DateTimeOffset.UtcNow < deadline)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, new Uri(definition.HealthEndpoint, relativePath));
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", secret);
                var remaining = deadline - DateTimeOffset.UtcNow;
                if (remaining <= TimeSpan.Zero) break;
                using var requestCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                requestCancellation.CancelAfter(remaining);
                using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, requestCancellation.Token);
                if (response.IsSuccessStatusCode) return;
            }
            catch (HttpRequestException) { }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested) { break; }
            await Task.Delay(TimeSpan.FromMilliseconds(100), cancellationToken);
        }
        throw new TimeoutException($"{definition.Name} did not report {relativePath} before the startup timeout.");
    }
    public void Dispose() => _http.Dispose();
}

public interface IWorkerSupervisor : IAsyncDisposable
{
    WorkerSupervisorState State { get; }
    event EventHandler? RecoveryRequired;
    event EventHandler<WorkerRestartEventArgs>? WorkerRestarting;
    event EventHandler<WorkerRestartEventArgs>? WorkerRestarted;
    Task StartAsync(WorkerDefinition voice, WorkerDefinition studio, string sessionSecret, IReadOnlyDictionary<string, string>? settings = null, CancellationToken cancellationToken = default);
    Task StopAsync(CancellationToken cancellationToken = default);
}

public sealed class WorkerSupervisor : IWorkerSupervisor
{
    private readonly IWorkerProcessLauncher _launcher;
    private readonly IWorkerHealthClient _health;
    private readonly IWorkerJob _job;
    private readonly IWorkerDependencyHealth _dependencies;
    private readonly IWorkerActivityCoordinator _activity;
    private readonly Dictionary<string, RunningWorker> _workers = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _gate = new();
    private readonly SemaphoreSlim _lifecycle = new(1, 1);
    private bool _stopping;

    public WorkerSupervisor(IWorkerProcessLauncher? launcher = null, IWorkerHealthClient? health = null, IWorkerJob? job = null, IWorkerDependencyHealth? dependencies = null, IWorkerActivityCoordinator? activity = null)
    {
        _launcher = launcher ?? new ProcessWorkerProcessLauncher();
        _health = health ?? new HttpWorkerHealthClient();
        _job = job ?? new WindowsJob();
        _dependencies = dependencies ?? new NoopWorkerDependencyHealth();
        _activity = activity ?? new WorkerActivityCoordinator();
    }

    public WorkerSupervisorState State { get; private set; } = WorkerSupervisorState.Stopped;
    public IWorkerActivityCoordinator Activity => _activity;
    public event EventHandler? RecoveryRequired;
    public event EventHandler<WorkerRestartEventArgs>? WorkerRestarting;
    public event EventHandler<WorkerRestartEventArgs>? WorkerRestarted;

    public async Task StartAsync(WorkerDefinition voice, WorkerDefinition studio, string sessionSecret, IReadOnlyDictionary<string, string>? settings = null, CancellationToken cancellationToken = default)
    {
        await _lifecycle.WaitAsync(cancellationToken);
        try
        {
        if (_stopping) throw new InvalidOperationException("The worker supervisor has stopped.");
        ValidateSecret(sessionSecret);
        State = WorkerSupervisorState.Starting;
        try
        {
            await StartWorkerCoreAsync(voice, sessionSecret, settings, restartCount: 0, cancellationToken);
            await _health.WaitForLiveAsync(voice, sessionSecret, voice.StartupTimeout, cancellationToken);
            await _dependencies.WaitForReadyAsync(cancellationToken);
            await _health.WaitForLiveAsync(voice, sessionSecret, voice.StartupTimeout, cancellationToken);
            await StartWorkerCoreAsync(studio, sessionSecret, settings, restartCount: 0, cancellationToken);
            State = WorkerSupervisorState.Running;
        }
        catch
        {
            State = WorkerSupervisorState.Faulted;
            await StopCoreAsync(CancellationToken.None);
            throw;
        }
        }
        finally { _lifecycle.Release(); }
    }

    public async Task StartWorkerAsync(WorkerDefinition definition, string sessionSecret, IReadOnlyDictionary<string, string>? settings = null, CancellationToken cancellationToken = default)
    {
        await _lifecycle.WaitAsync(cancellationToken);
        try
        {
            if (_stopping) throw new InvalidOperationException("The worker supervisor has stopped.");
            await StartWorkerCoreAsync(definition, sessionSecret, settings, restartCount: 0, cancellationToken);
        }
        finally { _lifecycle.Release(); }
    }

    private async Task StartWorkerCoreAsync(WorkerDefinition definition, string sessionSecret, IReadOnlyDictionary<string, string>? settings, int restartCount, CancellationToken cancellationToken, int handoffRetries = 0)
    {
        ArgumentNullException.ThrowIfNull(definition);
        ValidateSecret(sessionSecret);
        lock (_gate)
        {
            if (_workers.ContainsKey(definition.Name)) throw new InvalidOperationException("A worker with this name is already running.");
        }
        var environment = BuildEnvironment(definition, sessionSecret, settings);
        var logs = CreateLogs(definition, sessionSecret);
        IWorkerProcess? process = null;
        try
        {
            process = _launcher.Start(definition, environment, logs.Stdout, logs.Stderr);
            _job.Assign(process);
            process.Resume();
            await _health.WaitForLiveAsync(definition, sessionSecret, definition.StartupTimeout, cancellationToken);
            await _health.WaitForReadyAsync(definition, sessionSecret, definition.StartupTimeout, cancellationToken);
            var running = new RunningWorker(definition, process, sessionSecret, settings, logs) { Restarts = restartCount };
            lock (_gate) _workers[definition.Name] = running;
            _ = MonitorAsync(running);
        }
        catch (WorkerPortHandoffException) when (
            definition.PortReservation is not null &&
            !string.IsNullOrWhiteSpace(definition.PortEnvironmentVariable) &&
            handoffRetries < 1)
        {
            logs.Dispose.Dispose();
            if (process is not null)
            {
                try { await process.KillTreeAsync(CancellationToken.None); } catch { }
                await process.DisposeAsync();
            }
            await StartWorkerCoreAsync(definition.ReReservePort(), sessionSecret, settings, restartCount, cancellationToken, handoffRetries + 1);
        }
        catch (TimeoutException) when (
            process is not null &&
            process.HasExited &&
            definition.PortReservation is not null &&
            !string.IsNullOrWhiteSpace(definition.PortEnvironmentVariable) &&
            handoffRetries < 1)
        {
            logs.Dispose.Dispose();
            if (process is not null)
            {
                try { await process.KillTreeAsync(CancellationToken.None); } catch { }
                await process.DisposeAsync();
            }
            await StartWorkerCoreAsync(definition.ReReservePort(), sessionSecret, settings, restartCount, cancellationToken, handoffRetries + 1);
        }
        catch
        {
            logs.Dispose.Dispose();
            if (process is not null)
            {
                try { await process.KillTreeAsync(CancellationToken.None); } catch { }
                await process.DisposeAsync();
            }
            throw;
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        await _lifecycle.WaitAsync(cancellationToken);
        try { await StopCoreAsync(cancellationToken); }
        finally { _lifecycle.Release(); }
    }

    private async Task StopCoreAsync(CancellationToken cancellationToken)
    {
        _stopping = true;
        RunningWorker[] workers;
        lock (_gate) { workers = _workers.Values.ToArray(); _workers.Clear(); }
        _job.Dispose(); // Kernel kill-on-close covers descendants even if a Process instance is incomplete.
        foreach (var worker in workers)
        {
            try { await worker.Process.KillTreeAsync(cancellationToken); } catch (InvalidOperationException) { }
            await worker.Process.DisposeAsync();
            worker.Logs.Dispose.Dispose();
        }
        State = WorkerSupervisorState.Stopped;
    }

    private async Task MonitorAsync(RunningWorker worker)
    {
        try { await worker.Process.WaitForExitAsync(CancellationToken.None); }
        catch { return; }
        await _lifecycle.WaitAsync();
        try
        {
        if (_stopping) return;
        lock (_gate)
        {
            if (!_workers.TryGetValue(worker.Definition.Name, out var current) || !ReferenceEquals(current, worker)) return;
            _workers.Remove(worker.Definition.Name);
        }
        await worker.Process.DisposeAsync();
        worker.Logs.Dispose.Dispose();
        if (worker.Restarts == 0 && TryAcquireRestartLease(out var restartLease))
        {
            try
            {
                WorkerRestarting?.Invoke(this, new WorkerRestartEventArgs(worker.Definition.Name));
                using (restartLease)
                {
                worker.Restarts++;
                await StartWorkerCoreAsync(worker.Definition, worker.Secret, worker.Settings, worker.Restarts, CancellationToken.None);
                }
                WorkerRestarted?.Invoke(this, new WorkerRestartEventArgs(worker.Definition.Name));
                return;
            }
            catch { State = WorkerSupervisorState.RecoveryRequired; RecoveryRequired?.Invoke(this, EventArgs.Empty); return; }
        }
        State = WorkerSupervisorState.RecoveryRequired;
        RecoveryRequired?.Invoke(this, EventArgs.Empty);
        }
        finally { _lifecycle.Release(); }
    }

    private bool TryAcquireRestartLease(out IDisposable? lease)
    {
        return _activity.TryAcquireRestartLease(out lease);
    }

    private static IReadOnlyDictionary<string, string> BuildEnvironment(WorkerDefinition definition, string secret, IReadOnlyDictionary<string, string>? settings)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        void Copy(IReadOnlyDictionary<string, string> values)
        {
            foreach (var value in values)
            {
                if (value.Key.StartsWith("YOUTUBER_", StringComparison.Ordinal)) result[value.Key] = value.Value;
            }
        }
        if (settings is not null) Copy(settings);
        // Definition-scoped bindings win so a handoff retry can replace its own ports.
        Copy(definition.Environment);
        result["YOUTUBER_SESSION_SECRET"] = secret;
        return result;
    }

    private static (Action<string> Stdout, Action<string> Stderr, IDisposable Dispose) CreateLogs(WorkerDefinition definition, string secret)
    {
        if (string.IsNullOrWhiteSpace(definition.LogDirectory)) return (_ => { }, _ => { }, new DisposableAction(() => { }));
        var stdout = new RotatingLogWriter(definition.LogDirectory, definition.Name + ".stdout.log");
        var stderr = new RotatingLogWriter(definition.LogDirectory, definition.Name + ".stderr.log");
        string Scrub(string value) => value.Replace(secret, "[redacted]", StringComparison.Ordinal);
        return (line => stdout.WriteLine(Scrub(line)), line => stderr.WriteLine(Scrub(line)), new DisposableAction(() => { stdout.Dispose(); stderr.Dispose(); }));
    }

    private static void ValidateSecret(string secret)
    {
        if (SessionSecret.Decode(secret).Length != SessionSecret.ByteLength) throw new ArgumentException("Invalid worker session credential.", nameof(secret));
    }

    private sealed class RunningWorker(WorkerDefinition definition, IWorkerProcess process, string secret, IReadOnlyDictionary<string, string>? settings, (Action<string> Stdout, Action<string> Stderr, IDisposable Dispose) logs)
    {
        public WorkerDefinition Definition { get; } = definition;
        public IWorkerProcess Process { get; } = process;
        public string Secret { get; } = secret;
        public IReadOnlyDictionary<string, string>? Settings { get; } = settings;
        public (Action<string> Stdout, Action<string> Stderr, IDisposable Dispose) Logs { get; } = logs;
        public int Restarts { get; set; }
    }
    private sealed class DisposableAction(Action action) : IDisposable { public void Dispose() => action(); }
    public async ValueTask DisposeAsync()
    {
        await StopAsync(CancellationToken.None);
        if (_health is IDisposable disposable) disposable.Dispose();
        if (_dependencies is IDisposable dependency) dependency.Dispose();
    }
}

public sealed record WorkerRestartEventArgs(string WorkerName);
