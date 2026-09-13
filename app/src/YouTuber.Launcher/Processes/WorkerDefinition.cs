using System.Net;
using System.Net.Http;
using YouTuber.Launcher.Ollama;

namespace YouTuber.Launcher.Processes;

public sealed class WorkerDefinition
{
    public WorkerDefinition(string name, string executablePath, Uri healthEndpoint)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(executablePath);
        ArgumentNullException.ThrowIfNull(healthEndpoint);
        if (!healthEndpoint.IsLoopback || healthEndpoint.Scheme != Uri.UriSchemeHttp)
        {
            throw new ArgumentException("Worker health endpoints must be HTTP loopback endpoints.", nameof(healthEndpoint));
        }
        Name = name;
        ExecutablePath = executablePath;
        HealthEndpoint = healthEndpoint;
    }

    public string Name { get; }
    public string ExecutablePath { get; }
    public Uri HealthEndpoint { get; }
    public IReadOnlyList<string> Arguments { get; init; } = [];
    public IReadOnlyDictionary<string, string> Environment { get; init; } = new Dictionary<string, string>();
    public string? WorkingDirectory { get; init; }
    public string? LogDirectory { get; init; }
    public PortReservation? PortReservation { get; init; }
    public string? PortEnvironmentVariable { get; init; }
    public IReadOnlyList<WorkerPortBinding> AdditionalPortBindings { get; init; } = [];
    public IReadOnlyList<PortReservation> AdditionalPortReservations => AdditionalPortBindings.Select(binding => binding.Reservation).ToArray();
    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromSeconds(30);

    public WorkerDefinition WithPort(PortReservation reservation, string portEnvironmentVariable)
    {
        ArgumentNullException.ThrowIfNull(reservation);
        ArgumentException.ThrowIfNullOrWhiteSpace(portEnvironmentVariable);
        var environment = new Dictionary<string, string>(Environment, StringComparer.Ordinal) { [portEnvironmentVariable] = reservation.Port.ToString(System.Globalization.CultureInfo.InvariantCulture) };
        var endpoint = new UriBuilder(HealthEndpoint) { Port = reservation.Port }.Uri;
        return new WorkerDefinition(Name, ExecutablePath, endpoint)
        {
            Arguments = Arguments,
            Environment = environment,
            WorkingDirectory = WorkingDirectory,
            LogDirectory = LogDirectory,
            PortReservation = reservation,
            PortEnvironmentVariable = portEnvironmentVariable,
            AdditionalPortBindings = AdditionalPortBindings,
            StartupTimeout = StartupTimeout,
        };
    }

    public WorkerDefinition WithAdditionalPort(PortReservation reservation, string portEnvironmentVariable)
    {
        ArgumentNullException.ThrowIfNull(reservation);
        ArgumentException.ThrowIfNullOrWhiteSpace(portEnvironmentVariable);
        var environment = new Dictionary<string, string>(Environment, StringComparer.Ordinal)
        {
            [portEnvironmentVariable] = reservation.Port.ToString(System.Globalization.CultureInfo.InvariantCulture),
        };
        return Copy(environment: environment, additionalPortBindings: [.. AdditionalPortBindings, new WorkerPortBinding(reservation, portEnvironmentVariable)]);
    }

    public WorkerDefinition WithEnvironment(IReadOnlyDictionary<string, string> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        var environment = new Dictionary<string, string>(Environment, StringComparer.Ordinal);
        foreach (var pair in values) environment[pair.Key] = pair.Value;
        return Copy(environment: environment);
    }

    internal IReadOnlyList<PortReservation> AllPortReservations
        => PortReservation is null ? AdditionalPortReservations : [PortReservation, .. AdditionalPortReservations];

    public WorkerDefinition ReReservePort()
    {
        if (PortReservation is null || string.IsNullOrWhiteSpace(PortEnvironmentVariable)) throw new InvalidOperationException("This worker has no re-reservable port binding.");
        var allocation = new PortAllocator().Reserve(1 + AdditionalPortBindings.Count);
        var updated = WithPort(allocation[0], PortEnvironmentVariable);
        updated = updated.Copy(additionalPortBindings: []);
        for (var index = 0; index < AdditionalPortBindings.Count; index++)
            updated = updated.WithAdditionalPort(allocation[index + 1], AdditionalPortBindings[index].EnvironmentVariable);
        return updated;
    }

    private WorkerDefinition Copy(
        IReadOnlyDictionary<string, string>? environment = null,
        IReadOnlyList<WorkerPortBinding>? additionalPortBindings = null)
        => new(Name, ExecutablePath, HealthEndpoint)
        {
            Arguments = Arguments,
            Environment = environment ?? Environment,
            WorkingDirectory = WorkingDirectory,
            LogDirectory = LogDirectory,
            PortReservation = PortReservation,
            PortEnvironmentVariable = PortEnvironmentVariable,
            AdditionalPortBindings = additionalPortBindings ?? AdditionalPortBindings,
            StartupTimeout = StartupTimeout,
        };
}

public sealed record WorkerPortBinding(PortReservation Reservation, string EnvironmentVariable);

public interface IWorkerDependencyHealth
{
    Task WaitForReadyAsync(CancellationToken cancellationToken);
}

/// <summary>Uses the existing trusted Ollama client without taking ownership of its process.</summary>
public sealed class OllamaWorkerDependencyHealth(OllamaClient ollama) : IWorkerDependencyHealth
{
    public Task WaitForReadyAsync(CancellationToken cancellationToken) => ollama.EnsureHealthyAsync(cancellationToken);
}

public sealed class NoopWorkerDependencyHealth : IWorkerDependencyHealth
{
    public Task WaitForReadyAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}

public sealed class LoopbackOllamaDependencyHealth : IWorkerDependencyHealth, IDisposable
{
    private readonly Uri _status;
    private readonly HttpClient _http = new(new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false }, disposeHandler: true) { Timeout = TimeSpan.FromSeconds(2) };

    public LoopbackOllamaDependencyHealth(Uri endpoint)
    {
        OllamaInstallation.ValidateLoopbackEndpoint(endpoint);
        _status = new Uri(endpoint, "api/version");
    }

    public async Task WaitForReadyAsync(CancellationToken cancellationToken)
    {
        var deadline = DateTimeOffset.UtcNow + TimeSpan.FromSeconds(30);
        while (DateTimeOffset.UtcNow < deadline)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                using var response = await _http.GetAsync(_status, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                if (response.IsSuccessStatusCode && response.RequestMessage?.RequestUri == _status) return;
            }
            catch (HttpRequestException) { }
            catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested) { }
            await Task.Delay(250, cancellationToken);
        }
        throw new TimeoutException("The configured local Ollama service did not become ready.");
    }

    public void Dispose() => _http.Dispose();
}

public interface IWorkerActivityCoordinator
{
    bool HasActiveChatOrVoiceJob { get; }
    IDisposable AcquireJobLease();
    bool TryAcquireRestartLease(out IDisposable? lease);
}

/// <summary>Shared admission gate for UI jobs and supervisor crash recovery.</summary>
public sealed class WorkerActivityCoordinator : IWorkerActivityCoordinator
{
    private readonly object _gate = new();
    private int _jobs;
    private bool _restarting;
    public bool HasActiveChatOrVoiceJob { get { lock (_gate) return _jobs != 0; } }
    public IDisposable AcquireJobLease()
    {
        lock (_gate)
        {
            if (_restarting) throw new InvalidOperationException("Worker recovery is in progress.");
            _jobs++;
            return new Lease(this, false);
        }
    }
    public bool TryAcquireRestartLease(out IDisposable? lease)
    {
        lock (_gate)
        {
            if (_jobs != 0 || _restarting) { lease = null; return false; }
            _restarting = true; lease = new Lease(this, true); return true;
        }
    }
    private void Release(bool restart) { lock (_gate) { if (restart) _restarting = false; else _jobs--; } }
    private sealed class Lease(WorkerActivityCoordinator owner, bool restart) : IDisposable { private WorkerActivityCoordinator? _owner = owner; public void Dispose() { Interlocked.Exchange(ref _owner, null)?.Release(restart); } }
}

public enum WorkerSupervisorState { Stopped, Starting, Running, RecoveryRequired, Faulted }
