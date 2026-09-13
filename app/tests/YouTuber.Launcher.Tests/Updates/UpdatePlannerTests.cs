using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Updates;
using Xunit;

namespace YouTuber.Launcher.Tests.Updates;

public sealed class UpdatePlannerTests
{
    private const string A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    private const string B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    [Fact]
    public void Matching_revision_and_hash_is_a_no_op()
    {
        var manifest = Manifest(("rag", Component("r2", B, 1, 10)));
        var plan = new UpdatePlanner().Plan(manifest, new Dictionary<string, InstalledComponent> { ["rag"] = new("r2", B, 1) }, 100);
        Assert.Empty(plan.Components);
        Assert.False(plan.IsRejected);
    }

    [Theory]
    [InlineData("rag")]
    [InlineData("voice_model")]
    [InlineData("studio_runtime")]
    public void A_single_compatible_component_change_remains_independent(string changed)
    {
        var manifest = Manifest(
            ("rag", Component("new-rag", B, 1, 10)),
            ("voice_model", Component("new-voice", B, 1, 10)),
            ("llm", Component("llm", A, 1, 10)),
            ("voice_runtime", Component("voice-runtime", A, 1, 10)),
            ("studio_runtime", Component("studio-runtime", B, 1, 10)));
        var installed = manifest.Components.ToDictionary(
            pair => pair.Key,
            pair => new InstalledComponent(pair.Value.Revision, pair.Value.Sha256, pair.Value.Compatibility.RuntimeApi),
            StringComparer.Ordinal);
        installed[changed] = installed[changed] with { Revision = "old", Sha256 = A };

        var plan = new UpdatePlanner().Plan(manifest, installed, 1_000);

        Assert.Equal([changed], plan.Components.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void Runtime_api_change_coordinates_only_declared_runtime_components()
    {
        var manifest = Manifest(
            ("rag", Component("rag-v2", B, 2, 10)),
            ("llm", Component("llm", A, 1, 10)),
            ("voice_model", Component("voice", A, 1, 10)),
            ("voice_runtime", Component("vr2", B, 2, 10)),
            ("studio_runtime", Component("sr2", B, 2, 10)));
        var installed = new Dictionary<string, InstalledComponent>
        {
            ["rag"] = new("rag-v1", A, 1),
            ["llm"] = new("llm", A, 1),
            ["voice_model"] = new("voice", A, 1),
            ["voice_runtime"] = new("vr1", A, 1),
            ["studio_runtime"] = new("sr1", A, 1),
        };

        var plan = new UpdatePlanner().Plan(manifest, installed, 1_000);

        Assert.Equal(["rag", "studio_runtime", "voice_runtime"], plan.Components.Order(StringComparer.Ordinal));
        Assert.DoesNotContain("llm", plan.Components);
        Assert.DoesNotContain("voice_model", plan.Components);
    }

    [Fact]
    public void Runtime_api_change_is_rejected_when_a_coordinated_runtime_is_absent()
    {
        var manifest = Manifest(
            ("rag", Component("rag-v2", B, 2, 10)),
            ("studio_runtime", Component("sr2", B, 2, 10)));
        var installed = new Dictionary<string, InstalledComponent>
        {
            ["rag"] = new("rag-v1", A, 1),
            ["studio_runtime"] = new("sr1", A, 1),
        };

        var plan = new UpdatePlanner().Plan(manifest, installed, 1_000);

        Assert.True(plan.IsRejected);
        Assert.Empty(plan.Components);
        Assert.Contains("runtime", plan.Rejection, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Runtime_api_metadata_change_cannot_bypass_coordinated_runtime_planning()
    {
        var manifest = Manifest(
            ("rag", Component("rag", A, 2, 10)),
            ("voice_runtime", Component("vr2", B, 2, 10)),
            ("studio_runtime", Component("sr2", B, 2, 10)));
        var installed = new Dictionary<string, InstalledComponent>
        {
            ["rag"] = new("rag", A, 1),
            ["voice_runtime"] = new("vr1", A, 1),
            ["studio_runtime"] = new("sr1", A, 1),
        };

        var plan = new UpdatePlanner().Plan(manifest, installed, 1_000);

        Assert.Equal(["rag", "studio_runtime", "voice_runtime"], plan.Components.Order(StringComparer.Ordinal));
    }

    [Fact]
    public void Insufficient_peak_space_rejects_before_any_download_is_scheduled()
    {
        var manifest = Manifest(("rag", Component("new", B, 1, 101)));
        var plan = new UpdatePlanner().Plan(manifest, new Dictionary<string, InstalledComponent> { ["rag"] = new("old", A, 1) }, 100);
        Assert.True(plan.IsRejected);
        Assert.Empty(plan.Components);
        Assert.Contains("rollback", plan.Rejection, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Active_voice_job_postpones_activation_until_both_workers_report_idle()
    {
        var probe = new RecordingMaintenanceClient(voiceBusy: true);
        var activator = new RecordingActivator();
        var coordinator = new UpdateCoordinator(probe, activator);
        var plan = new UpdatePlan(new HashSet<string>(["rag"], StringComparer.Ordinal), 10);

        var result = await coordinator.ApplyAsync(plan);

        Assert.Equal(UpdateApplyStatus.Postponed, result);
        Assert.Equal([LocalWorker.Studio, LocalWorker.Voice], probe.Requests.Order());
        Assert.Equal(0, probe.Active);
        Assert.Equal([LocalWorker.Studio], probe.Released);
        Assert.Equal(0, activator.Calls);
    }

    [Fact]
    public async Task Maintenance_leases_remain_held_for_the_entire_activation()
    {
        var maintenance = new RecordingMaintenanceClient();
        var activator = new RecordingActivator(() => Assert.Equal(2, maintenance.Active));
        var coordinator = new UpdateCoordinator(maintenance, activator);

        var result = await coordinator.ApplyAsync(new UpdatePlan(new HashSet<string>(["rag"], StringComparer.Ordinal), 10));

        Assert.Equal(UpdateApplyStatus.Applied, result);
        Assert.Equal(0, maintenance.Active);
        Assert.Equal([LocalWorker.Studio, LocalWorker.Voice], maintenance.Acquired);
        Assert.Equal([LocalWorker.Voice, LocalWorker.Studio], maintenance.Released);
    }

    [Fact]
    public async Task Activation_failure_still_releases_both_worker_leases()
    {
        var maintenance = new RecordingMaintenanceClient();
        var coordinator = new UpdateCoordinator(
            maintenance,
            new RecordingActivator(() => throw new InvalidOperationException("activation failed")));

        await Assert.ThrowsAsync<InvalidOperationException>(() => coordinator.ApplyAsync(
            new UpdatePlan(new HashSet<string>(["rag"], StringComparer.Ordinal), 10)));

        Assert.Equal(0, maintenance.Active);
        Assert.Equal([LocalWorker.Voice, LocalWorker.Studio], maintenance.Released);
    }

    [Fact]
    public async Task Lost_heartbeat_cancels_activation_and_releases_every_acquired_lease()
    {
        var maintenance = new RecordingMaintenanceClient();
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var coordinator = new UpdateCoordinator(maintenance, new AsyncActivator(async cancellationToken =>
        {
            started.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
        }));
        var applying = coordinator.ApplyAsync(new UpdatePlan(new HashSet<string>(["rag"], StringComparer.Ordinal), 10));
        await started.Task;

        maintenance.Lose(LocalWorker.Studio);

        await Assert.ThrowsAsync<InvalidOperationException>(() => applying);
        Assert.Equal(0, maintenance.Active);
        Assert.Equal([LocalWorker.Voice, LocalWorker.Studio], maintenance.Released);
    }

    [Fact]
    public async Task Lost_heartbeat_with_a_non_cooperative_activation_fences_the_process_after_the_cancellation_deadline()
    {
        var maintenance = new RecordingMaintenanceClient();
        var started = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var terminator = new RecordingTerminator();
        var coordinator = new UpdateCoordinator(
            maintenance,
            new AsyncActivator(_ => { started.SetResult(); return Task.Delay(Timeout.InfiniteTimeSpan); }),
            TimeSpan.FromMilliseconds(25),
            terminator);
        var applying = coordinator.ApplyAsync(new UpdatePlan(new HashSet<string>(["rag"], StringComparer.Ordinal), 10));
        await started.Task;

        maintenance.Lose(LocalWorker.Studio);

        await Assert.ThrowsAsync<TimeoutException>(() => applying);
        Assert.Equal(1, terminator.Calls);
        Assert.Equal(0, maintenance.Active);
        Assert.Equal([LocalWorker.Voice, LocalWorker.Studio], maintenance.Released);
    }

    [Fact]
    public async Task Http_maintenance_client_uses_bearer_and_exact_acquire_release_contract()
    {
        var secret = SessionSecret.Create();
        var handler = new RecordingHandler(request =>
        {
            Assert.Equal(new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", secret), request.Headers.Authorization);
            var release = request.RequestUri!.AbsolutePath.EndsWith("/release", StringComparison.Ordinal);
            return new System.Net.Http.HttpResponseMessage(System.Net.HttpStatusCode.OK)
            {
                Content = new System.Net.Http.StringContent(
                    release ? "{\"status\":\"released\"}" : "{\"lease_token\":\"abcdefghijklmnopqrstuvwxyzABCDEFGH123456789\",\"expires_in_seconds\":30}",
                    System.Text.Encoding.UTF8,
                    "application/json"),
            };
        });
        using var client = new HttpWorkerMaintenanceClient(new Uri("http://127.0.0.1:31002/"), new Uri("http://127.0.0.1:31001/"), secret, handler);

        var lease = await client.TryAcquireAsync(LocalWorker.Studio);
        Assert.NotNull(lease);
        await lease!.DisposeAsync();

        Assert.Equal(
            ["http://127.0.0.1:31002/v1/maintenance/acquire", "http://127.0.0.1:31002/v1/maintenance/release"],
            handler.Requests.Select(request => request.RequestUri!.AbsoluteUri));
    }

    [Fact]
    public async Task Http_maintenance_client_rejects_an_oversized_lease_document()
    {
        var secret = SessionSecret.Create();
        var handler = new RecordingHandler(_ => new System.Net.Http.HttpResponseMessage(System.Net.HttpStatusCode.OK)
        {
            Content = new System.Net.Http.StringContent("{\"lease_token\":\"" + new string('x', 5000) + "\"}", System.Text.Encoding.UTF8, "application/json"),
        });
        using var client = new HttpWorkerMaintenanceClient(new Uri("http://127.0.0.1:31002/"), new Uri("http://127.0.0.1:31001/"), secret, handler);

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.TryAcquireAsync(LocalWorker.Studio));
    }

    [Fact]
    public void Retention_deletes_older_versions_only_after_current_and_one_rollback_are_healthy()
    {
        var now = DateTimeOffset.UtcNow;
        var current = new ComponentVersionState("runtime/current", now, true);
        var rollback = new ComponentVersionState("runtime/previous", now.AddDays(-1), true);
        var old = new ComponentVersionState("runtime/old", now.AddDays(-2), true);

        var plan = VersionRetentionPlanner.Plan(current.RelativePath, [old, current, rollback]);

        Assert.Same(rollback, plan.Rollback);
        Assert.Equal([old], plan.Deletable);
        Assert.Empty(VersionRetentionPlanner.Plan(current.RelativePath, [current with { Healthy = false }, rollback, old]).Deletable);
    }

    private static DistributionManifest Manifest(params (string Name, DistributionComponent Component)[] values)
        => new("youtuber.distribution.v1", "1.0.0", new MinimumRequirements("10.0.19045", 1, 1, 1, 1), values.ToDictionary(value => value.Name, value => value.Component, StringComparer.Ordinal));

    private static DistributionComponent Component(string revision, string hash, int runtimeApi, long peakSpace)
        => new("owner/repo", revision, "component.zip", 1, hash, "component", 1, 1, peakSpace, "worker.exe", ["--healthcheck"], new Compatibility(runtimeApi), ["NOTICE.txt"]);

    private sealed class RecordingMaintenanceClient(bool voiceBusy = false) : IWorkerMaintenanceClient
    {
        private readonly Dictionary<LocalWorker, DelegateLease> _leases = [];
        public List<LocalWorker> Acquired { get; } = [];
        public List<LocalWorker> Released { get; } = [];
        public IReadOnlyList<LocalWorker> Requests => Acquired;
        public int Active { get; private set; }
        public Task<IWorkerMaintenanceLease?> TryAcquireAsync(LocalWorker worker, CancellationToken cancellationToken = default)
        {
            Acquired.Add(worker);
            if (worker == LocalWorker.Voice && voiceBusy) return Task.FromResult<IWorkerMaintenanceLease?>(null);
            Active++;
            var lease = new DelegateLease(() => { Active--; Released.Add(worker); });
            _leases[worker] = lease;
            return Task.FromResult<IWorkerMaintenanceLease?>(lease);
        }
        public void Lose(LocalWorker worker) => _leases[worker].Lose();
    }

    private sealed class DelegateLease(Action release) : IWorkerMaintenanceLease
    {
        private readonly TaskCompletionSource _lost = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public Task LeaseLost => _lost.Task;
        public void Lose() => _lost.TrySetResult();
        public ValueTask DisposeAsync() { release(); return ValueTask.CompletedTask; }
    }

    private sealed class AsyncActivator(Func<CancellationToken, Task> activate) : IUpdateActivator
    {
        public Task ActivateAsync(UpdatePlan plan, CancellationToken cancellationToken = default) => activate(cancellationToken);
    }

    private sealed class RecordingActivator(Action? onActivate = null) : IUpdateActivator
    {
        public int Calls { get; private set; }
        public Task ActivateAsync(UpdatePlan plan, CancellationToken cancellationToken = default) { Calls++; onActivate?.Invoke(); return Task.CompletedTask; }
    }

    private sealed class RecordingTerminator : IUpdateFailureTerminator
    {
        public int Calls { get; private set; }
        public void Terminate(string message) => Calls++;
    }

    private sealed class RecordingHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        public List<HttpRequestMessage> Requests { get; } = [];
        public HttpRequestMessage? LastRequest { get; private set; }
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            LastRequest = request;
            Requests.Add(request);
            var response = respond(request);
            response.RequestMessage = request;
            return Task.FromResult(response);
        }
    }
}
