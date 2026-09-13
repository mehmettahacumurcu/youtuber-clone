using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.Ollama;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Setup;
using YouTuber.Launcher.Downloads;
using YouTuber.Launcher.Uninstall;
using Xunit;

namespace YouTuber.Launcher.Tests.Setup;

public sealed class ProductionRuntimeSessionTests
{
    [Fact]
    public async Task Restart_health_starts_owned_Ollama_checks_both_models_and_both_workers()
    {
        using var root = new TemporaryDirectory();
        var events = new List<string>();
        IReadOnlyDictionary<string, string>? launchedSettings = null;
        WorkerDefinition? launchedVoice = null;
        WorkerDefinition? launchedStudio = null;
        await using var session = Session(
            root.Path,
            events,
            captureLaunch: (voice, studio, settings) =>
            {
                launchedVoice = voice;
                launchedStudio = studio;
                launchedSettings = settings;
            });

        await session.EnsureHealthyAsync(FixtureManifest(), CancellationToken.None);

        Assert.Equal(["workers:create", "ollama:start-existing", "ollama:models", "workers:start", "ready:voice", "ready:studio"], events);
        Assert.True(session.HasRunningSession);
        Assert.NotEqual(11434, session.SessionPlan!.OllamaEndpoint!.Port);
        Assert.NotNull(launchedSettings);
        Assert.Equal(session.SessionPlan.OllamaEndpoint.GetLeftPart(UriPartial.Authority), launchedSettings!["YOUTUBER_OLLAMA_ORIGIN"]);
        Assert.Equal("speaker-v5-a636", launchedSettings["YOUTUBER_ANSWER_MODEL"]);
        Assert.Equal("qwen3:4b", launchedSettings["YOUTUBER_VERIFIER_MODEL"]);
        foreach (var key in new[] { "YOUTUBER_DATA_ROOT", "YOUTUBER_CACHE_ROOT", "YOUTUBER_STUDIO_PORT", "YOUTUBER_RAG_PORT", "YOUTUBER_VOICE_PORT" })
            Assert.True(launchedSettings.ContainsKey(key), $"missing launcher worker setting: {key}");
        Assert.True(Path.IsPathFullyQualified(launchedVoice!.Environment["YOUTUBER_INSTALL_ROOT"]));
        Assert.True(Path.IsPathFullyQualified(launchedStudio!.Environment["YOUTUBER_INSTALL_ROOT"]));
    }

    [Fact]
    public async Task Failed_worker_health_stops_workers_and_owned_Ollama_and_never_exposes_a_session()
    {
        using var root = new TemporaryDirectory();
        var events = new List<string>();
        await using var session = Session(root.Path, events, failStudioHealthAttempts: 1);

        await Assert.ThrowsAsync<InvalidOperationException>(() => session.EnsureHealthyAsync(FixtureManifest(), CancellationToken.None));

        Assert.Contains("workers:stop", events);
        Assert.Contains("ollama:stop", events);
        Assert.False(session.HasRunningSession);
        Assert.Null(session.SessionPlan);
    }

    [Fact]
    public async Task Retry_after_prepared_model_health_failure_restarts_owned_Ollama_before_verification()
    {
        using var root = new TemporaryDirectory();
        var events = new List<string>();
        await using var session = Session(root.Path, events, failStudioHealthAttempts: 1);
        var manifest = FixtureManifest();
        await session.EnsureModelsAsync(manifest, CancellationToken.None);
        await Assert.ThrowsAsync<InvalidOperationException>(() => session.EnsureHealthyAsync(manifest, CancellationToken.None));

        events.Clear();
        await session.EnsureHealthyAsync(manifest, CancellationToken.None);

        Assert.Equal(["workers:create", "ollama:start-existing", "ollama:models", "workers:start", "ready:voice", "ready:studio"], events);
        Assert.True(session.HasRunningSession);
    }

    [Fact]
    public async Task Cancellation_during_health_cleans_up_the_owned_runtime()
    {
        using var root = new TemporaryDirectory();
        var events = new List<string>();
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await using var session = Session(root.Path, events);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.EnsureHealthyAsync(FixtureManifest(), cancelled.Token));

        Assert.Contains("workers:stop", events);
        Assert.Contains("ollama:stop", events);
        Assert.False(session.HasRunningSession);
    }

    [Fact]
    public async Task Worker_and_Ollama_cleanup_failures_do_not_replace_the_original_health_failure()
    {
        using var root = new TemporaryDirectory();
        var events = new List<string>();
        await using var session = Session(root.Path, events, failStudioHealthAttempts: 1, failWorkerDispose: true, failOllamaStop: true);

        var failure = await Assert.ThrowsAsync<InvalidOperationException>(() => session.EnsureHealthyAsync(FixtureManifest(), CancellationToken.None));

        Assert.Equal("studio health failed", failure.Message);
        Assert.Contains("workers:stop", events);
        Assert.Contains("ollama:stop", events);
    }

    [Fact]
    public async Task Closing_a_healthy_session_stops_workers_and_the_launcher_owned_Ollama_process()
    {
        using var root = new TemporaryDirectory();
        var events = new List<string>();
        var session = Session(root.Path, events);
        await session.EnsureHealthyAsync(FixtureManifest(), CancellationToken.None);

        await session.DisposeAsync();

        Assert.Contains("workers:stop", events);
        Assert.Contains("ollama:stop", events);
        Assert.False(session.HasRunningSession);
    }

    [Fact]
    public async Task Partial_Ollama_download_is_owned_before_a_later_artifact_download_fails()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        await using var ollama = new ControlledOllamaRuntime(paths, downloads: new FailSecondDownload(), installationProbe: new RecordingInstallationProbe(null));
        using var runtimeFactory = new WorkerRuntimeFactory();
        var runtime = runtimeFactory.Create(
            paths.DataRoot,
            new WorkerDefinition("voice", "voice.exe", new Uri("http://127.0.0.1:1/")),
            new WorkerDefinition("studio", "studio.exe", new Uri("http://127.0.0.1:1/")));

        await Assert.ThrowsAsync<IOException>(() => ollama.EnsureModelsAsync(FixtureManifest(), runtime, CancellationToken.None));

        var partialGguf = Path.Combine(paths.DownloadsRoot, "qwen3-verifier.gguf");
        Assert.Contains(partialGguf, new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData).Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task CompatibleDetectedOllamaIsSelectedAndNeverReinstalledOnRetry()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var detected = OllamaInstallation.Existing(
            @"C:\Program Files\Ollama\ollama.exe",
            new Uri("http://127.0.0.1:11434"),
            "0.9.1",
            @"C:\Users\person\.ollama\models");
        var probe = new RecordingInstallationProbe(detected);
        var operations = new RecordingOllamaRuntimeOperations();
        var downloads = new RecordingDownloadClient();
        await using var ollama = new ControlledOllamaRuntime(
            paths,
            downloads: downloads,
            installationProbe: probe,
            operations: operations);
        using var runtimeFactory = new WorkerRuntimeFactory();
        var runtime = runtimeFactory.Create(
            paths.DataRoot,
            new WorkerDefinition("voice", "voice.exe", new Uri("http://127.0.0.1:1/")),
            new WorkerDefinition("studio", "studio.exe", new Uri("http://127.0.0.1:1/")));
        var manifest = FixtureManifest();

        await ollama.EnsureModelsAsync(manifest, runtime, CancellationToken.None);
        await ollama.EnsureModelsAsync(manifest, runtime, CancellationToken.None);

        Assert.Equal(1, probe.Calls);
        Assert.Equal(1, operations.SelectionCalls);
        Assert.Equal(1, operations.StartExistingCalls);
        Assert.Equal(0, operations.InstallCalls);
        Assert.Equal(1, operations.EnsureModelCalls);
        Assert.Equal(1, operations.VerifyModelCalls);
        Assert.Equal(3, downloads.Requests.Count);
        Assert.DoesNotContain(downloads.Requests, request => request.TargetPath.EndsWith("ollama-installer.exe", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(runtime.Ollama.ApiEndpoint, operations.SelectedInstallation!.ApiEndpoint);
        Assert.Equal(runtime.Ollama.ModelDirectory, operations.SelectedInstallation.ModelDirectory);
        Assert.Equal(runtime.Ollama.ModelDirectory, operations.SelectedInstallation.Environment["OLLAMA_MODELS"]);
    }

    [Fact]
    public async Task RetryAfterSuccessfulInstallAndFailedModelImportDoesNotRunInstallerAgain()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var installerBytes = new byte[] { 0x42, 0x19, 0x73 };
        var manifest = FixtureManifestWithInstaller(installerBytes);
        var probe = new RecordingInstallationProbe(null);
        var operations = new RecordingOllamaRuntimeOperations { EnsureFailuresRemaining = 1 };
        var downloads = new InstallerAwareDownloadClient(installerBytes);
        await using var ollama = new ControlledOllamaRuntime(
            paths,
            downloads: downloads,
            authenticode: new AuthenticodeVerifier(new AlwaysTrustedOllamaPublisher()),
            installationProbe: probe,
            operations: operations);
        using var runtimeFactory = new WorkerRuntimeFactory();
        var runtime = runtimeFactory.Create(
            paths.DataRoot,
            new WorkerDefinition("voice", "voice.exe", new Uri("http://127.0.0.1:1/")),
            new WorkerDefinition("studio", "studio.exe", new Uri("http://127.0.0.1:1/")));

        await Assert.ThrowsAsync<IOException>(() => ollama.EnsureModelsAsync(manifest, runtime, CancellationToken.None));
        await ollama.EnsureModelsAsync(manifest, runtime, CancellationToken.None);

        Assert.Equal(1, operations.InstallCalls);
        Assert.Equal(1, operations.StartExistingCalls);
        Assert.Equal(2, operations.EnsureModelCalls);
        Assert.Equal(1, downloads.Requests.Count(request => request.TargetPath.EndsWith("ollama-installer.exe", StringComparison.OrdinalIgnoreCase)));
    }

    private static ProductionRuntimeSession Session(
        string root,
        List<string> events,
        int failStudioHealthAttempts = 0,
        bool failWorkerDispose = false,
        bool failOllamaStop = false,
        Action<WorkerDefinition, WorkerDefinition, IReadOnlyDictionary<string, string>?>? captureLaunch = null)
    {
        var paths = AppPaths.ForBaseDirectory(root);
        return new ProductionRuntimeSession(
            paths,
            new RecordingRuntimeComposer(paths),
            new RecordingOllamaRuntime(events, failOllamaStop),
            health: new RecordingHealth(events, failStudioHealthAttempts),
            workerFactory: _ =>
            {
                events.Add("workers:create");
                return new RecordingWorkers(events, failWorkerDispose, captureLaunch);
            });
    }

    private static FetchedDistributionManifest FixtureManifest()
        => new(ManifestValidator.ParseDistribution(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "distribution-valid.json"))), new string('c', 64));

    private static FetchedDistributionManifest FixtureManifestWithInstaller(byte[] installerBytes)
    {
        var fetched = FixtureManifest();
        var metadata = fetched.Manifest.Ollama!;
        var installer = metadata.Installer with
        {
            Size = installerBytes.Length,
            Sha256 = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(installerBytes)).ToLowerInvariant(),
        };
        return fetched with { Manifest = fetched.Manifest with { Ollama = metadata with { Installer = installer } } };
    }

    private sealed class RecordingRuntimeComposer(AppPaths paths) : IProductionRuntimeComposer
    {
        private readonly WorkerRuntimeFactory _factory = new();
        public Task<WorkerRuntime> ComposeAsync(CancellationToken cancellationToken)
        {
            var voice = new WorkerDefinition("voice", "voice.exe", new Uri("http://127.0.0.1:1/"));
            var studio = new WorkerDefinition("studio", "studio.exe", new Uri("http://127.0.0.1:1/"));
            return Task.FromResult(_factory.Create(paths.DataRoot, voice, studio));
        }
        public void Dispose() => _factory.Dispose();
    }

    private sealed class RecordingOllamaRuntime(List<string> events, bool failStop = false) : IControlledOllamaRuntime
    {
        public Task EnsureModelsAsync(FetchedDistributionManifest manifest, WorkerRuntime runtime, CancellationToken cancellationToken) { events.Add("ollama:ensure"); return Task.CompletedTask; }
        public Task StartExistingAsync(FetchedDistributionManifest manifest, WorkerRuntime runtime, CancellationToken cancellationToken) { events.Add("ollama:start-existing"); return Task.CompletedTask; }
        public Task VerifyModelsAsync(FetchedDistributionManifest manifest, WorkerRuntime runtime, CancellationToken cancellationToken) { events.Add("ollama:models"); return Task.CompletedTask; }
        public Task StopAsync(CancellationToken cancellationToken)
        {
            events.Add("ollama:stop");
            return failStop ? Task.FromException(new InvalidOperationException("ollama cleanup failed")) : Task.CompletedTask;
        }
        public ValueTask DisposeAsync() { events.Add("ollama:stop"); return ValueTask.CompletedTask; }
    }

    private sealed class RecordingInstallationProbe(OllamaInstallation? detected) : IOllamaInstallationProbe
    {
        public int Calls { get; private set; }
        public Task<OllamaInstallation?> DetectAsync(CancellationToken cancellationToken = default)
        {
            Calls++;
            return Task.FromResult(detected);
        }
    }

    private sealed class RecordingOllamaRuntimeOperations : IOllamaRuntimeOperations
    {
        public int SelectionCalls { get; private set; }
        public int StartExistingCalls { get; private set; }
        public int InstallCalls { get; private set; }
        public int EnsureModelCalls { get; private set; }
        public int VerifyModelCalls { get; private set; }
        public OllamaInstallation? SelectedInstallation { get; private set; }
        public int EnsureFailuresRemaining { get; set; }

        public OllamaInstallation SelectInstallation(OllamaInstallation? detected, OllamaInstallation composed)
        {
            SelectionCalls++;
            return SelectedInstallation = composed with
            {
                IsExisting = detected is not null,
                BinaryPath = detected?.BinaryPath,
                Version = detected?.Version,
                BinaryTrusted = detected?.BinaryTrusted ?? false,
            };
        }

        public Task StartInstalledAsync(OllamaPublication publication, OllamaInstallation installation, PortReservation reservation, CancellationToken cancellationToken)
        {
            StartExistingCalls++;
            return Task.CompletedTask;
        }

        public Task<InstallerRunResult> InstallNewAsync(OllamaInstallerArtifact artifact, OllamaInstallation installation, PortReservation reservation, IInstallerRunner installer, CancellationToken cancellationToken)
        {
            InstallCalls++;
            return Task.FromResult(new InstallerRunResult(0, string.Empty, string.Empty, false));
        }

        public Task EnsureModelsAsync(OllamaInstallation installation, OllamaImportArtifacts artifacts, OllamaModelRolesMetadata roles, CancellationToken cancellationToken)
        {
            EnsureModelCalls++;
            if (EnsureFailuresRemaining-- > 0) throw new IOException("model import failed after installer success");
            return Task.CompletedTask;
        }

        public Task VerifyModelsAsync(OllamaInstallation installation, OllamaModelRolesMetadata roles, CancellationToken cancellationToken)
        {
            VerifyModelCalls++;
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
    }

    private sealed class RecordingDownloadClient : ISetupDownloadClient
    {
        public List<DownloadRequest> Requests { get; } = [];
        public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            Requests.Add(request);
            return Task.CompletedTask;
        }
        public void Dispose() { }
    }

    private sealed class InstallerAwareDownloadClient(byte[] installerBytes) : ISetupDownloadClient
    {
        public List<DownloadRequest> Requests { get; } = [];
        public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            Requests.Add(request);
            Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
            File.WriteAllBytes(request.TargetPath, request.TargetPath.EndsWith("ollama-installer.exe", StringComparison.OrdinalIgnoreCase) ? installerBytes : new byte[checked((int)request.ExpectedSize)]);
            return Task.CompletedTask;
        }
        public void Dispose() { }
    }

    private sealed class AlwaysTrustedOllamaPublisher : IWinVerifyTrust
    {
        public AuthenticodeTrustResult Verify(string path) => new(true, OllamaInstallation.ExpectedPublisher, null);
    }

    private sealed class RecordingWorkers(
        List<string> events,
        bool failDispose = false,
        Action<WorkerDefinition, WorkerDefinition, IReadOnlyDictionary<string, string>?>? captureLaunch = null) : IWorkerSupervisor
    {
        private bool _disposed;
        public WorkerSupervisorState State { get; private set; }
        public event EventHandler? RecoveryRequired { add { } remove { } }
        public event EventHandler<WorkerRestartEventArgs>? WorkerRestarting { add { } remove { } }
        public event EventHandler<WorkerRestartEventArgs>? WorkerRestarted { add { } remove { } }
        public Task StartAsync(WorkerDefinition voice, WorkerDefinition studio, string sessionSecret, IReadOnlyDictionary<string, string>? settings = null, CancellationToken cancellationToken = default)
        {
            if (_disposed) throw new ObjectDisposedException(nameof(RecordingWorkers));
            cancellationToken.ThrowIfCancellationRequested();
            captureLaunch?.Invoke(voice, studio, settings);
            events.Add("workers:start");
            State = WorkerSupervisorState.Running;
            return Task.CompletedTask;
        }
        public Task StopAsync(CancellationToken cancellationToken = default) { events.Add("workers:stop"); State = WorkerSupervisorState.Stopped; return Task.CompletedTask; }
        public ValueTask DisposeAsync()
        {
            events.Add("workers:stop");
            State = WorkerSupervisorState.Stopped;
            _disposed = true;
            return failDispose ? ValueTask.FromException(new InvalidOperationException("worker cleanup failed")) : ValueTask.CompletedTask;
        }
    }

    private sealed class FailSecondDownload : ISetupDownloadClient
    {
        private int _count;
        public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            if (++_count == 2)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
                File.WriteAllText(request.TargetPath, "partial");
                throw new IOException("second artifact failed after partial write");
            }
            Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
            File.WriteAllBytes(request.TargetPath, new byte[checked((int)request.ExpectedSize)]);
            return Task.CompletedTask;
        }
        public void Dispose() { }
    }

    private sealed class RecordingHealth(List<string> events, int failStudioAttempts) : IWorkerHealthClient
    {
        private int _remainingStudioFailures = failStudioAttempts;
        public Task WaitForLiveAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken) => Task.CompletedTask;
        public Task WaitForReadyAsync(WorkerDefinition definition, string secret, TimeSpan timeout, CancellationToken cancellationToken)
        {
            events.Add("ready:" + definition.Name);
            if (definition.Name == "studio" && _remainingStudioFailures-- > 0) throw new InvalidOperationException("studio health failed");
            return Task.CompletedTask;
        }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory() { Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-runtime-session-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(Path); }
        public string Path { get; }
        public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
    }
}
