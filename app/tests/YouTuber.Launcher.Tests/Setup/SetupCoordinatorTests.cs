using System.Collections.Immutable;
using YouTuber.Launcher.Setup;
using Xunit;

namespace YouTuber.Launcher.Tests.Setup;

public sealed class SetupCoordinatorTests
{
    [Theory]
    [InlineData(SetupErrorCategory.UnsupportedHardware)] [InlineData(SetupErrorCategory.InsufficientSpace)] [InlineData(SetupErrorCategory.SmartScreen)] [InlineData(SetupErrorCategory.NetworkRetry)] [InlineData(SetupErrorCategory.HashMismatch)] [InlineData(SetupErrorCategory.PrerequisiteElevation)]
    public void Setup_errors_are_explicit_and_actionable(SetupErrorCategory category) => Assert.DoesNotContain("something went wrong", SetupErrors.Message(category), StringComparison.OrdinalIgnoreCase);
    [Fact]
    public void Cannot_skip_an_ordered_stage() => Assert.Throws<InvalidOperationException>(() => new SetupCoordinator(new MemoryJournal()).Complete(SetupStage.Download));

    [Fact]
    public void Terms_cannot_be_accepted_before_the_terms_stage()
        => Assert.Throws<InvalidOperationException>(() => new SetupCoordinator(new MemoryJournal()).AcceptTerms());

    [Fact]
    public void Blocking_preflight_prevents_downloads()
    {
        var coordinator = new SetupCoordinator(new MemoryJournal());
        coordinator.FailPreflight("Unsupported hardware");
        Assert.Throws<InvalidOperationException>(() => coordinator.Complete(SetupStage.Preflight));
    }

    [Fact]
    public void Pause_persists_partial_component_state()
    {
        var journal = new MemoryJournal();
        var coordinator = new SetupCoordinator(journal);
        coordinator.ReportComponent(new ComponentProgress("voice", "1", 10, 100, 5, TimeSpan.FromSeconds(18), "Downloading"));
        coordinator.Pause();
        Assert.True(journal.State!.IsPaused);
        Assert.Equal(10, journal.State.Components.Single().BytesReceived);
    }

    [Fact]
    public async Task Retry_targets_only_the_failed_component_after_operation_confirmation()
    {
        var coordinator = new SetupCoordinator(new MemoryJournal(), new RecordingOperations());
        coordinator.ReportComponent(new ComponentProgress("voice", "1", 0, 100, 0, null, "Failed", "hash mismatch"));
        coordinator.ReportComponent(new ComponentProgress("studio", "1", 50, 100, 5, null, "Downloading"));
        Assert.True(await coordinator.RetryComponentAsync("voice"));
        Assert.Equal("Queued", coordinator.State.Components.Single(component => component.Name == "voice").Status);
        Assert.Equal("Downloading", coordinator.State.Components.Single(component => component.Name == "studio").Status);
    }

    [Fact]
    public void Restart_resumes_at_the_earliest_incomplete_stage()
    {
        var journal = new MemoryJournal { State = SetupState.Initial with { Stage = SetupStage.ChooseDataRoot, CompletedStages = ImmutableHashSet.Create(SetupStage.Preflight) } };
        var coordinator = new SetupCoordinator(journal);
        Assert.Equal(SetupStage.ChooseDataRoot, coordinator.State.Stage);
    }

    [Fact]
    public void Restart_keeps_a_blocking_preflight() => Assert.Equal(SetupStage.Preflight, new SetupCoordinator(new MemoryJournal { State = SetupState.Initial with { BlockingError = "disk" } }).State.Stage);

    [Fact]
    public void Restart_returns_to_terms_when_terms_is_earliest_incomplete()
        => Assert.Equal(SetupStage.AcceptTerms, new SetupCoordinator(new MemoryJournal { State = SetupState.Initial with { CompletedStages = ImmutableHashSet.Create(SetupStage.Preflight, SetupStage.ChooseDataRoot) } }).State.Stage);

    [Fact]
    public void Persisted_terms_checkpoint_cannot_bypass_missing_acceptance()
    {
        var completed = Enum.GetValues<SetupStage>().Where(stage => stage < SetupStage.Download).ToImmutableHashSet();
        var coordinator = new SetupCoordinator(new MemoryJournal
        {
            State = SetupState.Initial with { Stage = SetupStage.Download, TermsAccepted = false, CompletedStages = completed },
        });

        Assert.Equal(SetupStage.AcceptTerms, coordinator.State.Stage);
        Assert.DoesNotContain(SetupStage.AcceptTerms, coordinator.State.CompletedStages);
        Assert.DoesNotContain(SetupStage.Prerequisites, coordinator.State.CompletedStages);
    }

    [Fact]
    public void Ready_flag_fails_closed_when_prior_stage_checkpoints_are_incomplete()
    {
        var coordinator = new SetupCoordinator(new MemoryJournal
        {
            State = SetupState.Initial with { Stage = SetupStage.Ready, TermsAccepted = true, ReadyRecorded = true, CompletedStages = ImmutableHashSet.Create(SetupStage.Preflight) },
        });

        Assert.False(coordinator.State.ReadyRecorded);
        Assert.Equal(SetupStage.ChooseDataRoot, coordinator.State.Stage);
    }

    [Fact]
    public void Failed_ready_persistence_does_not_publish_ready()
    {
        var journal = new FailingJournal { State = SetupState.Initial with { Stage = SetupStage.Ready, TermsAccepted = true, CompletedStages = Enum.GetValues<SetupStage>().Where(stage => stage != SetupStage.Ready).ToImmutableHashSet() } }; var coordinator = new SetupCoordinator(journal); journal.Fail = true;
        Assert.Throws<IOException>(() => coordinator.Complete(SetupStage.Ready)); Assert.False(coordinator.State.ReadyRecorded);
    }

    [Fact]
    public async Task Download_close_fails_closed_without_an_operation_controller()
    {
        var coordinator = new SetupCoordinator(new MemoryJournal { State = StateAt(SetupStage.Download) });

        Assert.False(await coordinator.RequestCloseAsync());
        Assert.False(coordinator.State.IsPaused);
    }

    [Theory]
    [InlineData(DownloadPauseResult.PausedAndPersisted, true)]
    [InlineData(DownloadPauseResult.Failed, false)]
    public async Task Download_close_requires_a_confirmed_durable_pause(DownloadPauseResult result, bool expected)
    {
        var operations = new RecordingOperations { DownloadResult = result };
        var coordinator = new SetupCoordinator(new MemoryJournal { State = StateAt(SetupStage.Download) }, operations);

        Assert.Equal(expected, await coordinator.RequestCloseAsync());
        Assert.Equal(expected, coordinator.State.IsPaused);
    }

    [Fact]
    public async Task Download_close_denies_exit_when_pause_throws()
    {
        var operations = new RecordingOperations { Failure = new IOException("pause failed") };
        var coordinator = new SetupCoordinator(new MemoryJournal { State = StateAt(SetupStage.Download) }, operations);

        Assert.False(await coordinator.RequestCloseAsync());
        Assert.False(coordinator.State.IsPaused);
    }

    [Theory]
    [InlineData(ActivationCloseResult.AtomicCommitCompleted, true)]
    [InlineData(ActivationCloseResult.RolledBackSafe, true)]
    [InlineData(ActivationCloseResult.Unsafe, false)]
    public async Task Activation_close_requires_an_explicit_safe_result(ActivationCloseResult result, bool expected)
    {
        var operations = new RecordingOperations { ActivationResult = result };
        var coordinator = new SetupCoordinator(new MemoryJournal { State = StateAt(SetupStage.Activate) }, operations);

        Assert.Equal(expected, await coordinator.RequestCloseAsync());
        Assert.Equal(expected, coordinator.State.IsPaused);
    }

    [Fact]
    public async Task Progress_racing_confirmed_close_pause_preserves_latest_bytes_and_pause_state()
    {
        var operations = new BlockingOperations();
        var coordinator = new SetupCoordinator(new MemoryJournal { State = StateAt(SetupStage.Download) }, operations);
        coordinator.ReportComponent(new ComponentProgress("voice", "1", 1, 10, 1, null, "Downloading"));

        var close = coordinator.RequestCloseAsync();
        await operations.PauseEntered.Task;
        await Task.Run(() => coordinator.ReportComponent(new ComponentProgress("voice", "1", 9, 10, 1, null, "Downloading")));
        operations.AllowPause.TrySetResult();

        Assert.True(await close);
        Assert.True(coordinator.State.IsPaused);
        Assert.Equal(9, coordinator.State.Components.Single().BytesReceived);
    }

    [Fact]
    public async Task Concurrent_progress_transitions_are_serialized_without_lost_components()
    {
        var coordinator = new SetupCoordinator(new MemoryJournal());

        await Task.WhenAll(Enumerable.Range(0, 50).Select(index => Task.Run(() =>
            coordinator.ReportComponent(new ComponentProgress($"component-{index}", "1", index, 50, 1, null, "Downloading")))));

        Assert.Equal(50, coordinator.State.Components.Count);
        Assert.Equal(50, coordinator.State.Revision);
    }

    [Fact]
    public async Task Rejected_component_operation_does_not_publish_a_fake_status_change()
    {
        var operations = new RecordingOperations { ComponentResult = ComponentOperationResult.Rejected };
        var coordinator = new SetupCoordinator(new MemoryJournal(), operations);
        coordinator.ReportComponent(new ComponentProgress("voice", "1", 1, 10, 1, null, "Downloading"));

        Assert.False(await coordinator.PauseComponentAsync("voice"));
        Assert.Equal("Downloading", coordinator.State.Components.Single().Status);
    }

    private static SetupState StateAt(SetupStage stage) => SetupState.Initial with
    {
        Stage = stage,
        TermsAccepted = true,
        CompletedStages = Enum.GetValues<SetupStage>().Where(value => value < stage).ToImmutableHashSet(),
    };

    private class MemoryJournal : ISetupJournal { public SetupState? State { get; set; } public SetupState? Load() => State; public virtual void Save(SetupState state) => State = state; }
    private sealed class FailingJournal : MemoryJournal { public bool Fail; public override void Save(SetupState state) { if (Fail) throw new IOException(); base.Save(state); } }
    private sealed class RecordingOperations : ISetupOperationControl
    {
        public DownloadPauseResult DownloadResult { get; init; } = DownloadPauseResult.PausedAndPersisted;
        public ActivationCloseResult ActivationResult { get; init; } = ActivationCloseResult.AtomicCommitCompleted;
        public Exception? Failure { get; init; }
        public ComponentOperationResult ComponentResult { get; init; } = ComponentOperationResult.ConfirmedDurable;
        public Task<DownloadPauseResult> PauseDownloadsAsync(CancellationToken cancellationToken)
            => Failure is null ? Task.FromResult(DownloadResult) : Task.FromException<DownloadPauseResult>(Failure);
        public Task<ActivationCloseResult> WaitForActivationSafePointAsync(CancellationToken cancellationToken)
            => Failure is null ? Task.FromResult(ActivationResult) : Task.FromException<ActivationCloseResult>(Failure);
        public Task<ComponentOperationResult> PauseComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentResult);
        public Task<ComponentOperationResult> ResumeComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentResult);
        public Task<ComponentOperationResult> RetryComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentResult);
    }
    private sealed class BlockingOperations : ISetupOperationControl
    {
        public TaskCompletionSource PauseEntered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource AllowPause { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public async Task<DownloadPauseResult> PauseDownloadsAsync(CancellationToken cancellationToken)
        {
            PauseEntered.TrySetResult();
            await AllowPause.Task.WaitAsync(cancellationToken);
            return DownloadPauseResult.PausedAndPersisted;
        }
        public Task<ActivationCloseResult> WaitForActivationSafePointAsync(CancellationToken cancellationToken) => Task.FromResult(ActivationCloseResult.AtomicCommitCompleted);
        public Task<ComponentOperationResult> PauseComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentOperationResult.ConfirmedDurable);
        public Task<ComponentOperationResult> ResumeComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentOperationResult.ConfirmedDurable);
        public Task<ComponentOperationResult> RetryComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentOperationResult.ConfirmedDurable);
    }
}
