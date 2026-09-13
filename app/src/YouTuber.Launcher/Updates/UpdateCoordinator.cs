using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using YouTuber.Launcher.Processes;

namespace YouTuber.Launcher.Updates;

public enum LocalWorker { Studio, Voice }
public enum UpdateApplyStatus { NoChanges, Rejected, Postponed, Applied }

public interface IWorkerMaintenanceClient
{
    Task<IWorkerMaintenanceLease?> TryAcquireAsync(LocalWorker worker, CancellationToken cancellationToken = default);
}

public interface IWorkerMaintenanceLease : IAsyncDisposable
{
    Task LeaseLost { get; }
}

public interface IUpdateActivator
{
    Task ActivateAsync(UpdatePlan plan, CancellationToken cancellationToken = default);
}

/// <summary>
/// Stops the launcher when an activation ignored cancellation after a worker maintenance
/// lease was lost.  Continuing would allow an unfenced pointer commit while workers are
/// no longer guaranteed idle; the activation transaction is recovered on the next start.
/// </summary>
public interface IUpdateFailureTerminator
{
    void Terminate(string message);
}

public sealed class FailFastUpdateFailureTerminator : IUpdateFailureTerminator
{
    public void Terminate(string message) => Environment.FailFast(message);
}

public sealed class HttpWorkerMaintenanceClient : IWorkerMaintenanceClient, IDisposable
{
    private const int MaximumResponseBytes = 4096;
    private readonly HttpClient _http;
    private readonly IReadOnlyDictionary<LocalWorker, Uri> _origins;
    private readonly string _secret;

    public HttpWorkerMaintenanceClient(Uri studioEndpoint, Uri voiceEndpoint, string sessionSecret, HttpMessageHandler? handler = null)
    {
        SessionSecret.Decode(sessionSecret);
        _secret = sessionSecret;
        _origins = new Dictionary<LocalWorker, Uri>
        {
            [LocalWorker.Studio] = ExactOrigin(studioEndpoint),
            [LocalWorker.Voice] = ExactOrigin(voiceEndpoint),
        };
        _http = new HttpClient(handler ?? new HttpClientHandler { AllowAutoRedirect = false, UseProxy = false }, disposeHandler: true)
        {
            Timeout = TimeSpan.FromSeconds(5),
        };
    }

    public async Task<IWorkerMaintenanceLease?> TryAcquireAsync(LocalWorker worker, CancellationToken cancellationToken = default)
    {
        var endpoint = new Uri(_origins[worker], "v1/maintenance/acquire");
        using var request = AuthorizedPost(endpoint, new ByteArrayContent([]));
        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        if (response.StatusCode == HttpStatusCode.Conflict && response.RequestMessage?.RequestUri == endpoint) return null;
        EnsureDirectJson(response, endpoint, worker);
        using var document = await ReadBoundedJsonAsync(response, worker, cancellationToken);
        if (!document.RootElement.TryGetProperty("lease_token", out var tokenElement)
            || tokenElement.GetString() is not { Length: >= 32 and <= 128 } token
            || token.Any(character => !(char.IsAsciiLetterOrDigit(character) || character is '_' or '-'))
            || !document.RootElement.TryGetProperty("expires_in_seconds", out var expiryElement)
            || !expiryElement.TryGetDouble(out var expirySeconds) || expirySeconds is < 1 or > 300)
        {
            throw new InvalidOperationException($"{worker} returned an invalid maintenance lease.");
        }
        return new RemoteLease(this, worker, token, TimeSpan.FromSeconds(expirySeconds));
    }

    private async Task RenewAsync(LocalWorker worker, string token, CancellationToken cancellationToken)
    {
        var endpoint = new Uri(_origins[worker], "v1/maintenance/renew");
        using var response = await SendTokenAsync(endpoint, token, cancellationToken);
        EnsureDirectJson(response, endpoint, worker);
        using var document = await ReadBoundedJsonAsync(response, worker, cancellationToken);
        if (!document.RootElement.TryGetProperty("status", out var status) || status.GetString() != "renewed")
            throw new InvalidOperationException($"{worker} did not renew its maintenance lease.");
    }

    private async ValueTask ReleaseAsync(LocalWorker worker, string token)
    {
        var endpoint = new Uri(_origins[worker], "v1/maintenance/release");
        using var response = await SendTokenAsync(endpoint, token, CancellationToken.None);
        EnsureDirectJson(response, endpoint, worker);
        using var document = await ReadBoundedJsonAsync(response, worker, CancellationToken.None);
        if (!document.RootElement.TryGetProperty("status", out var status) || status.GetString() != "released")
            throw new InvalidOperationException($"{worker} did not release its maintenance lease.");
    }

    private async Task<HttpResponseMessage> SendTokenAsync(Uri endpoint, string token, CancellationToken cancellationToken)
    {
        var payload = JsonSerializer.SerializeToUtf8Bytes(new Dictionary<string, string> { ["lease_token"] = token });
        using var content = new ByteArrayContent(payload);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        using var request = AuthorizedPost(endpoint, content);
        return await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
    }

    private HttpRequestMessage AuthorizedPost(Uri endpoint, HttpContent content)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, endpoint) { Content = content };
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _secret);
        return request;
    }

    private static void EnsureDirectJson(HttpResponseMessage response, Uri endpoint, LocalWorker worker)
    {
        if (!response.IsSuccessStatusCode || response.RequestMessage?.RequestUri != endpoint
            || response.Content.Headers.ContentType?.MediaType != "application/json")
        {
            throw new InvalidOperationException($"{worker} maintenance did not return a direct authenticated JSON response.");
        }
    }

    private static async Task<JsonDocument> ReadBoundedJsonAsync(HttpResponseMessage response, LocalWorker worker, CancellationToken cancellationToken)
    {
        if (response.Content.Headers.ContentLength is > MaximumResponseBytes)
            throw new InvalidOperationException($"{worker} maintenance response is too large.");
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        var bytes = new byte[MaximumResponseBytes + 1];
        var length = 0;
        while (length < bytes.Length)
        {
            var read = await stream.ReadAsync(bytes.AsMemory(length), cancellationToken);
            if (read == 0) break;
            length += read;
        }
        if (length > MaximumResponseBytes)
            throw new InvalidOperationException($"{worker} maintenance response is too large.");
        return JsonDocument.Parse(bytes.AsMemory(0, length));
    }

    private static Uri ExactOrigin(Uri endpoint)
    {
        ArgumentNullException.ThrowIfNull(endpoint);
        if (!endpoint.IsAbsoluteUri || endpoint.Scheme != Uri.UriSchemeHttp || endpoint.Host != "127.0.0.1"
            || endpoint.Port is <= 0 or > 65535 || !string.IsNullOrEmpty(endpoint.UserInfo)
            || !string.IsNullOrEmpty(endpoint.Query) || !string.IsNullOrEmpty(endpoint.Fragment)
            || endpoint.AbsolutePath != "/")
        {
            throw new ArgumentException("Worker maintenance endpoint must be an exact HTTP 127.0.0.1 origin.", nameof(endpoint));
        }
        return endpoint;
    }

    public void Dispose() => _http.Dispose();

    private sealed class RemoteLease : IWorkerMaintenanceLease
    {
        private readonly HttpWorkerMaintenanceClient _owner;
        private readonly LocalWorker _worker;
        private readonly string _token;
        private readonly TimeSpan _heartbeatInterval;
        private readonly CancellationTokenSource _stop = new();
        private readonly TaskCompletionSource _lost = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly Task _heartbeat;
        private int _released;

        public RemoteLease(HttpWorkerMaintenanceClient owner, LocalWorker worker, string token, TimeSpan lifetime)
        {
            _owner = owner;
            _worker = worker;
            _token = token;
            _heartbeatInterval = TimeSpan.FromMilliseconds(Math.Max(100, lifetime.TotalMilliseconds / 3));
            _heartbeat = HeartbeatAsync();
        }

        public Task LeaseLost => _lost.Task;

        private async Task HeartbeatAsync()
        {
            try
            {
                while (true)
                {
                    await Task.Delay(_heartbeatInterval, _stop.Token);
                    await _owner.RenewAsync(_worker, _token, _stop.Token);
                }
            }
            catch (OperationCanceledException) when (_stop.IsCancellationRequested) { }
            catch { _lost.TrySetResult(); }
        }

        public async ValueTask DisposeAsync()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0) return;
            _stop.Cancel();
            try
            {
                await _heartbeat;
                await _owner.ReleaseAsync(_worker, _token);
            }
            finally { _stop.Dispose(); }
        }
    }
}

public sealed class UpdateCoordinator
{
    private readonly IWorkerMaintenanceClient _workers;
    private readonly IUpdateActivator _activator;
    private readonly TimeSpan _leaseLossCancellationTimeout;
    private readonly IUpdateFailureTerminator _terminator;

    public UpdateCoordinator(
        IWorkerMaintenanceClient workers,
        IUpdateActivator activator,
        TimeSpan? leaseLossCancellationTimeout = null,
        IUpdateFailureTerminator? terminator = null)
    {
        _workers = workers ?? throw new ArgumentNullException(nameof(workers));
        _activator = activator ?? throw new ArgumentNullException(nameof(activator));
        _leaseLossCancellationTimeout = leaseLossCancellationTimeout ?? TimeSpan.FromSeconds(30);
        if (_leaseLossCancellationTimeout <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(leaseLossCancellationTimeout));
        _terminator = terminator ?? new FailFastUpdateFailureTerminator();
    }

    public async Task<UpdateApplyStatus> ApplyAsync(UpdatePlan plan, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(plan);
        if (plan.IsRejected) return UpdateApplyStatus.Rejected;
        if (plan.Components.Count == 0) return UpdateApplyStatus.NoChanges;

        var leases = new List<IWorkerMaintenanceLease>();
        Exception? releaseError = null;
        try
        {
            foreach (var worker in new[] { LocalWorker.Studio, LocalWorker.Voice })
            {
                var lease = await _workers.TryAcquireAsync(worker, cancellationToken);
                if (lease is null) return UpdateApplyStatus.Postponed;
                leases.Add(lease);
            }
            using var activationCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            var activation = _activator.ActivateAsync(plan, activationCancellation.Token);
            var firstLeaseLoss = Task.WhenAny(leases.Select(lease => lease.LeaseLost));
            var firstCompletion = await Task.WhenAny(activation, firstLeaseLoss);
            if (firstCompletion == firstLeaseLoss)
            {
                activationCancellation.Cancel();
                var cancellationAcknowledged = await Task.WhenAny(activation, Task.Delay(_leaseLossCancellationTimeout, CancellationToken.None));
                if (cancellationAcknowledged != activation)
                {
                    _terminator.Terminate("Update activation ignored cancellation after a worker maintenance lease was lost.");
                    throw new TimeoutException("Update activation did not acknowledge cancellation after a worker maintenance lease was lost.");
                }
                try { await activation; }
                catch (OperationCanceledException) when (activationCancellation.IsCancellationRequested) { }
                throw new InvalidOperationException("A worker maintenance lease was lost during activation.");
            }
            await activation;
            return UpdateApplyStatus.Applied;
        }
        finally
        {
            for (var index = leases.Count - 1; index >= 0; index--)
            {
                try { await leases[index].DisposeAsync(); }
                catch (Exception exception) { releaseError ??= exception; }
            }
            if (releaseError is not null) throw new InvalidOperationException("A worker maintenance lease could not be released.", releaseError);
        }
    }
}

public sealed record ComponentVersionState(string RelativePath, DateTimeOffset ActivatedAt, bool Healthy);
public sealed record VersionRetentionPlan(ComponentVersionState Current, ComponentVersionState? Rollback, IReadOnlyList<ComponentVersionState> Deletable);

public static class VersionRetentionPlanner
{
    public static VersionRetentionPlan Plan(string currentRelativePath, IReadOnlyList<ComponentVersionState> versions)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(currentRelativePath);
        ArgumentNullException.ThrowIfNull(versions);
        var current = versions.SingleOrDefault(version => string.Equals(version.RelativePath, currentRelativePath, StringComparison.Ordinal))
            ?? throw new InvalidOperationException("The active component version is absent from retention state.");
        var rollback = versions.Where(version => version.Healthy && !string.Equals(version.RelativePath, current.RelativePath, StringComparison.Ordinal)).OrderByDescending(version => version.ActivatedAt).FirstOrDefault();
        var deletable = current.Healthy && rollback is not null
            ? versions.Where(version => !string.Equals(version.RelativePath, current.RelativePath, StringComparison.Ordinal)
                                        && !string.Equals(version.RelativePath, rollback.RelativePath, StringComparison.Ordinal)).ToArray()
            : [];
        return new VersionRetentionPlan(current, rollback, deletable);
    }
}
