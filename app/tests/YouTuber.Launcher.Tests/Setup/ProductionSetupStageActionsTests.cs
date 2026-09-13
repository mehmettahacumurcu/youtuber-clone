using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.Setup;
using YouTuber.Launcher.SystemChecks;
using YouTuber.Launcher.Uninstall;
using YouTuber.Launcher.Downloads;
using YouTuber.Launcher.Activation;
using System.Collections.Immutable;
using YouTuber.Launcher.ViewModels;
using Xunit;

namespace YouTuber.Launcher.Tests.Setup;

public sealed class ProductionSetupStageActionsTests
{
    [Fact]
    public async Task Former_no_op_stages_execute_their_concrete_provisioners_once()
    {
        using var root = new TemporaryDirectory();
        var calls = new List<string>();
        var manifest = FixtureManifest();
        await using var actions = new ProductionSetupStageActions(
            AppPaths.ForBaseDirectory(root.Path),
            new FixedManifestSource(manifest),
            new RecordingPrerequisites(calls),
            new RecordingComponents(calls),
            new RecordingOllama(calls),
            new RecordingHealth(calls),
            new PassingProbe(),
            new PreflightEvaluator());

        await actions.PrerequisitesAsync(CancellationToken.None);
        await actions.DownloadAsync(CancellationToken.None);
        await actions.VerifyAsync(CancellationToken.None);
        await actions.ActivateAsync(CancellationToken.None);
        await actions.OllamaModelsAsync(CancellationToken.None);
        await actions.EndToEndHealthAsync(CancellationToken.None);

        Assert.Equal(["prerequisites", "download", "verify", "activate", "ollama:qwen3:4b:speaker-v5-a636", "health"], calls);
    }

    [Fact]
    public async Task Cancellation_and_health_failure_are_propagated_without_a_false_success()
    {
        using var root = new TemporaryDirectory();
        var health = new RecordingHealth([], fail: true);
        await using var actions = new ProductionSetupStageActions(
            AppPaths.ForBaseDirectory(root.Path),
            new FixedManifestSource(FixtureManifest()),
            new RecordingPrerequisites([]),
            new RecordingComponents([]),
            new RecordingOllama([]),
            health,
            new PassingProbe(),
            new PreflightEvaluator());

        await Assert.ThrowsAsync<InvalidOperationException>(() => actions.EndToEndHealthAsync(CancellationToken.None));
        Assert.False(actions.HasRunningSession);
    }

    [Fact]
    public async Task Component_archive_is_owned_immediately_after_download_before_activation()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        using var components = new ProductionComponentProvisioner(paths, new WritingDownload());

        await components.DownloadAsync(FixtureManifest(), CancellationToken.None);

        var archive = Path.Combine(paths.DownloadsRoot, "voice_runtime.zip");
        Assert.Contains(archive, new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData).Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Partially_written_component_archive_is_owned_when_download_throws()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        using var components = new ProductionComponentProvisioner(paths, new PartialFailureDownload());

        await Assert.ThrowsAsync<IOException>(() => components.DownloadAsync(FixtureManifest(), CancellationToken.None));

        var archive = Path.Combine(paths.DownloadsRoot, "voice_runtime.zip");
        Assert.Contains(archive, new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData).Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Activation_recovers_a_stale_transaction_before_starting_a_new_one_without_rewriting_setup_journal()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var store = new ActiveComponentsStore(paths.DataRoot);
        var staleStaging = Path.Combine(store.ActivationRoot, "runtime-old.staging-crash");
        Directory.CreateDirectory(staleStaging);
        await store.SaveTransactionAsync(new ActivationTransaction(
            "voice_runtime",
            DistributionInstallRoot.Runtime,
            "0.9.0",
            new string('b', 64),
            "runtime/0.9.0-bbbbbbbbbbbb",
            staleStaging,
            ActivationPhase.Extracted,
            null));
        var journal = new AtomicFileSetupJournal(paths.SetupJournalFile);
        journal.Save(SetupState.Initial with { Revision = 1 });
        using var components = new ProductionComponentProvisioner(paths, new WritingDownload());

        await Assert.ThrowsAsync<ComponentActivationException>(() => components.ActivateAsync(FixtureManifest(), CancellationToken.None));

        Assert.False(Directory.Exists(staleStaging));
        Assert.Empty(await store.LoadTransactionsAsync());
        Assert.Equal(1, journal.Load()!.Revision);
    }

    [Fact]
    public async Task Production_download_close_waits_until_resume_metadata_is_durable()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var download = new DurablePauseDownload();
        using var components = new ProductionComponentProvisioner(paths, download);
        var operations = Assert.IsAssignableFrom<ISetupOperationControl>(components);
        var run = components.DownloadAsync(FixtureManifest(), CancellationToken.None);
        await download.Started.Task;

        var pause = operations.PauseDownloadsAsync(CancellationToken.None);
        await download.CancellationObserved.Task;
        Assert.False(pause.IsCompleted);
        download.AllowPersistence.TrySetResult();

        Assert.Equal(DownloadPauseResult.PausedAndPersisted, await pause);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
        var request = download.Request!;
        var metadata = ResumeMetadata.Load(request.TargetPath + ".partial.json");
        Assert.NotNull(metadata);
        Assert.True(metadata.Matches(request, 1));
    }

    [Fact]
    public async Task Production_coordinator_uses_durable_recovery_evidence_and_real_operation_control()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        Directory.CreateDirectory(paths.DownloadsRoot);
        await File.WriteAllBytesAsync(Path.Combine(paths.DownloadsRoot, "voice_runtime.zip"), [1]);
        var journal = new AtomicFileSetupJournal(paths.SetupJournalFile);
        journal.Save(StateAt(SetupStage.Verify) with { Revision = 1 });
        var components = new ProductionComponentProvisioner(paths, new WritingDownload());
        await using var actions = new ProductionSetupStageActions(
            paths,
            new FixedManifestSource(FixtureManifest()),
            new RecordingPrerequisites([]),
            components,
            new RecordingOllama([]),
            new RecordingHealth([]),
            new PassingProbe(),
            new PreflightEvaluator());

        var coordinator = await actions.CreateCoordinatorAsync(journal, CancellationToken.None);

        Assert.Equal(SetupStage.Download, coordinator.State.Stage);
        coordinator.ReportComponent(new ComponentProgress("voice_runtime", "1.0.0", 1, 2, 1, null, "Downloading"));
        Assert.True(coordinator.CanOperateComponent("voice_runtime", "Downloading"));
    }

    [Fact]
    public async Task Production_ui_commands_pause_resume_and_retry_the_real_component_download()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var journal = new AtomicFileSetupJournal(paths.SetupJournalFile);
        journal.Save(StateAt(SetupStage.Download) with { Revision = 1 });
        var download = new DurablePauseDownload();
        var components = new ProductionComponentProvisioner(paths, download);
        await using var actions = new ProductionSetupStageActions(
            paths,
            new FixedManifestSource(FixtureManifest()),
            new RecordingPrerequisites([]),
            components,
            new RecordingOllama([]),
            new RecordingHealth([]),
            new PassingProbe(),
            new PreflightEvaluator());
        var coordinator = await actions.CreateCoordinatorAsync(journal, CancellationToken.None);
        var viewModel = new SetupViewModel(coordinator, new ImmediateUiDispatcher());
        var run = actions.DownloadAsync(CancellationToken.None);
        await download.Started.Task;
        var component = Assert.Single(viewModel.Components);

        var pause = component.PauseCommand.ExecuteAsync();
        await download.CancellationObserved.Task;
        download.AllowPersistence.TrySetResult();
        await pause;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => run);
        Assert.Equal("Paused", component.Status);

        download.CompleteSubsequentCalls = true;
        await component.ResumeCommand.ExecuteAsync();
        Assert.Equal(2, download.Calls);
        Assert.True(File.Exists(Path.Combine(paths.DownloadsRoot, "voice_runtime.zip")));

        await Assert.ThrowsAsync<InvalidDataException>(() => actions.VerifyAsync(CancellationToken.None));
        Assert.Equal("Failed", component.Status);
        await component.RetryCommand.ExecuteAsync();
        Assert.Equal(3, download.Calls);
    }

    [Fact]
    public async Task Production_activate_close_waits_for_the_real_recovery_safe_point_and_persists_resume_state()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var hooks = new BlockingRecoveryHooks();
        var runtime = new ComponentActivator(new ActiveComponentsStore(paths.DataRoot), hooks: hooks);
        using var components = new ProductionComponentProvisioner(paths, new WritingDownload(), runtime);
        var journal = new AtomicFileSetupJournal(paths.SetupJournalFile);
        journal.Save(StateAt(SetupStage.Activate) with { Revision = 1 });
        var coordinator = new SetupCoordinator(journal, components);
        var activation = components.ActivateAsync(FixtureManifest(), CancellationToken.None);

        await hooks.RecoveryEntered.Task.WaitAsync(TimeSpan.FromSeconds(2));
        var close = coordinator.RequestCloseAsync();
        Assert.False(close.IsCompleted);
        hooks.AllowRecovery.TrySetResult();

        await Assert.ThrowsAsync<ComponentActivationException>(() => activation);
        Assert.True(await close);
        Assert.True(journal.Load()!.IsPaused);
        Assert.Equal(SetupStage.Activate, journal.Load()!.Stage);
    }

    [Fact]
    public async Task Restarted_production_coordinator_resumes_the_download_discovered_by_durable_probe_evidence()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var fetched = FixtureManifest();
        var componentMetadata = fetched.Manifest.Components["voice_runtime"];
        var request = new DownloadRequest(
            HuggingFaceUrl.Create(componentMetadata.Repo, componentMetadata.Revision, componentMetadata.Path),
            Path.Combine(paths.DownloadsRoot, "voice_runtime.zip"),
            componentMetadata.Size,
            componentMetadata.Sha256);
        Directory.CreateDirectory(paths.DownloadsRoot);
        await File.WriteAllBytesAsync(request.TargetPath + ".partial", [1]);
        ResumeMetadata.Create(request, 1, "etag").Save(request.TargetPath + ".partial.json");
        var journal = new AtomicFileSetupJournal(paths.SetupJournalFile);
        journal.Save(StateAt(SetupStage.Verify) with { Revision = 1 });
        var download = new WritingDownload();
        var components = new ProductionComponentProvisioner(paths, download);
        await using var actions = new ProductionSetupStageActions(
            paths,
            new FixedManifestSource(fetched),
            new RecordingPrerequisites([]),
            components,
            new RecordingOllama([]),
            new RecordingHealth([]),
            new PassingProbe(),
            new PreflightEvaluator());
        var coordinator = await actions.CreateCoordinatorAsync(journal, CancellationToken.None);
        var viewModel = new SetupViewModel(coordinator, new ImmediateUiDispatcher());
        var paused = Assert.Single(viewModel.Components);

        Assert.Equal("Paused", paused.Status);
        await paused.ResumeCommand.ExecuteAsync();

        Assert.Equal(1, download.Calls);
    }

    private static FetchedDistributionManifest FixtureManifest()
    {
        var manifest = ManifestValidator.ParseDistribution(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "distribution-valid.json")));
        return new(manifest, new string('c', 64));
    }

    private static SetupState StateAt(SetupStage stage) => SetupState.Initial with
    {
        Stage = stage,
        TermsAccepted = true,
        CompletedStages = Enum.GetValues<SetupStage>().Where(value => value < stage).ToImmutableHashSet(),
    };

    private sealed class FixedManifestSource(FetchedDistributionManifest value) : IDistributionManifestSource
    {
        public Task<FetchedDistributionManifest> GetAsync(CancellationToken cancellationToken) => Task.FromResult(value);
        public void Dispose() { }
    }

    private sealed class RecordingPrerequisites(List<string> calls) : ISetupPrerequisiteProvisioner
    {
        public Task EnsureAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken) { calls.Add("prerequisites"); return Task.CompletedTask; }
    }

    private sealed class RecordingComponents(List<string> calls) : ISetupComponentProvisioner
    {
        public Task DownloadAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken) { calls.Add("download"); return Task.CompletedTask; }
        public Task VerifyAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken) { calls.Add("verify"); return Task.CompletedTask; }
        public Task ActivateAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken) { calls.Add("activate"); return Task.CompletedTask; }
    }

    private sealed class RecordingOllama(List<string> calls) : ISetupOllamaProvisioner
    {
        public Task EnsureModelsAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken)
        {
            calls.Add($"ollama:{manifest.Manifest.Ollama!.Models.Verifier.Name}:{manifest.Manifest.Ollama.Models.Speaker.Name}");
            return Task.CompletedTask;
        }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class RecordingHealth(List<string> calls, bool fail = false) : ISetupEndToEndHealth
    {
        public bool HasRunningSession { get; private set; }
        public Task EnsureHealthyAsync(FetchedDistributionManifest manifest, CancellationToken cancellationToken)
        {
            calls.Add("health");
            if (fail) throw new InvalidOperationException("health failed");
            HasRunningSession = true;
            return Task.CompletedTask;
        }
        public ValueTask DisposeAsync() { HasRunningSession = false; return ValueTask.CompletedTask; }
    }

    private sealed class PassingProbe : ISystemProbe
    {
        public Task<SystemProfile> ProbeAsync(AppPaths paths, CancellationToken cancellationToken = default)
        {
            var drive = Path.GetPathRoot(paths.DataRoot)!;
            return Task.FromResult(new SystemProfile(
                22631,
                NvidiaProbeResult.Success([new NvidiaGpu("RTX", 12L << 30, "test")]),
                64L << 30,
                new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase) { [drive] = 100L << 30 },
                drive,
                drive,
                new PrerequisiteStatus(true, true)));
        }
    }

    private sealed class WritingDownload : ISetupDownloadClient
    {
        public int Calls { get; private set; }
        public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            Calls++;
            Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
            File.WriteAllBytes(request.TargetPath, new byte[checked((int)request.ExpectedSize)]);
            return Task.CompletedTask;
        }
        public void Dispose() { }
    }

    private sealed class PartialFailureDownload : ISetupDownloadClient
    {
        public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
            File.WriteAllText(request.TargetPath, "partial");
            throw new IOException("download failed after partial write");
        }
        public void Dispose() { }
    }

    private sealed class DurablePauseDownload : ISetupDownloadClient
    {
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource CancellationObserved { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource AllowPersistence { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public DownloadRequest? Request { get; private set; }
        public bool CompleteSubsequentCalls { get; set; }
        public int Calls { get; private set; }

        public async Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            Calls++;
            Request = request;
            Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
            if (Calls > 1 && CompleteSubsequentCalls)
            {
                await File.WriteAllBytesAsync(request.TargetPath, [(byte)Calls], CancellationToken.None);
                File.Delete(request.TargetPath + ".partial");
                File.Delete(request.TargetPath + ".partial.json");
                return;
            }
            await File.WriteAllBytesAsync(request.TargetPath + ".partial", [1], CancellationToken.None);
            Started.TrySetResult();
            try
            {
                await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                CancellationObserved.TrySetResult();
                await AllowPersistence.Task;
                ResumeMetadata.Create(request, 1, "etag").Save(request.TargetPath + ".partial.json");
                throw;
            }
        }

        public void Dispose() { }
    }

    private sealed class BlockingRecoveryHooks : IActivationRaceHooks
    {
        public TaskCompletionSource RecoveryEntered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource AllowRecovery { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public async Task BeforeRecoveryAsync(CancellationToken cancellationToken)
        {
            RecoveryEntered.TrySetResult();
            await AllowRecovery.Task.WaitAsync(cancellationToken);
        }

        public Task BeforeHealthcheckAsync(string stagingDirectory, CancellationToken cancellationToken) => Task.CompletedTask;
        public Task BeforeFinalMoveAsync(string stagingDirectory, CancellationToken cancellationToken) => Task.CompletedTask;
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory() { Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-stage-actions-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(Path); }
        public string Path { get; }
        public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
    }
}
