using System.Collections.Immutable;
using YouTuber.Launcher.Setup;
using Xunit;

namespace YouTuber.Launcher.Tests.Setup;

public sealed class FirstRunSetupRunnerTests
{
    [Fact]
    public async Task Runs_the_ordered_setup_stages_and_stops_for_explicit_terms_acceptance()
    {
        var actions = new RecordingActions();
        var runner = new FirstRunSetupRunner(new SetupCoordinator(new MemoryJournal()), actions);

        var waiting = await runner.RunAsync();

        Assert.True(waiting.RequiresTermsAcceptance);
        Assert.Equal(SetupStage.AcceptTerms, waiting.Stage);
        Assert.Equal([SetupStage.Preflight, SetupStage.ChooseDataRoot], actions.Executed);

        runner.AcceptTerms();
        var completed = await runner.RunAsync();

        Assert.True(completed.IsReady);
        Assert.Equal(SetupStage.Ready, completed.Stage);
        Assert.Equal(
        [
            SetupStage.Preflight, SetupStage.ChooseDataRoot, SetupStage.Prerequisites,
            SetupStage.FetchCatalog, SetupStage.Download, SetupStage.Verify,
            SetupStage.Activate, SetupStage.OllamaModels, SetupStage.EndToEndHealth,
        ], actions.Executed);
    }

    [Fact]
    public async Task Does_not_run_a_stage_or_checkpoint_it_when_its_action_fails()
    {
        var coordinator = new SetupCoordinator(new MemoryJournal { State = StateAt(SetupStage.Download) });
        var actions = new RecordingActions { FailureStage = SetupStage.Download };
        var runner = new FirstRunSetupRunner(coordinator, actions);

        await Assert.ThrowsAsync<InvalidOperationException>(() => runner.RunAsync());

        Assert.Equal([SetupStage.Download], actions.Executed);
        Assert.Equal(SetupStage.Download, coordinator.State.Stage);
        Assert.DoesNotContain(SetupStage.Download, coordinator.State.CompletedStages);
    }

    [Fact]
    public async Task Does_not_continue_after_a_durable_checkpoint_write_fails()
    {
        var journal = new FailingJournal { FailSaves = true };
        var coordinator = new SetupCoordinator(journal);
        var actions = new RecordingActions();
        var runner = new FirstRunSetupRunner(coordinator, actions);

        await Assert.ThrowsAsync<IOException>(() => runner.RunAsync());

        Assert.Equal([SetupStage.Preflight], actions.Executed);
        Assert.Equal(SetupStage.Preflight, coordinator.State.Stage);
        Assert.DoesNotContain(SetupStage.Preflight, coordinator.State.CompletedStages);
    }

    [Fact]
    public async Task Checkpoint_failure_after_model_work_stops_owned_runtime_before_returning_the_error()
    {
        var journal = new FailingJournal { State = StateAt(SetupStage.OllamaModels), FailSaves = true };
        var actions = new RecordingActions();
        var runner = new FirstRunSetupRunner(new SetupCoordinator(journal), actions);

        await Assert.ThrowsAsync<IOException>(() => runner.RunAsync());

        Assert.Equal([SetupStage.OllamaModels], actions.Executed);
        Assert.Equal(1, actions.FailedRunCleanups);
    }

    [Fact]
    public async Task Failed_end_to_end_health_never_records_Ready()
    {
        var coordinator = new SetupCoordinator(new MemoryJournal { State = StateAt(SetupStage.EndToEndHealth) });
        var runner = new FirstRunSetupRunner(coordinator, new RecordingActions { FailureStage = SetupStage.EndToEndHealth });

        await Assert.ThrowsAsync<InvalidOperationException>(() => runner.RunAsync());

        Assert.Equal(SetupStage.EndToEndHealth, coordinator.State.Stage);
        Assert.False(coordinator.State.ReadyRecorded);
        Assert.DoesNotContain(SetupStage.EndToEndHealth, coordinator.State.CompletedStages);
    }

    [Fact]
    public async Task Durable_ready_health_failure_persists_an_activate_repair_checkpoint_and_retry_succeeds()
    {
        var journal = new MemoryJournal
        {
            State = StateAt(SetupStage.Ready) with { ReadyRecorded = true },
        };
        var coordinator = new SetupCoordinator(journal);
        var actions = new RecordingActions { FailureStage = SetupStage.EndToEndHealth };
        var runner = new FirstRunSetupRunner(coordinator, actions);

        await Assert.ThrowsAsync<InvalidOperationException>(() => runner.RevalidateReadyAsync());

        Assert.Equal(SetupStage.Activate, journal.State!.Stage);
        Assert.False(journal.State.ReadyRecorded);
        Assert.DoesNotContain(SetupStage.Activate, journal.State.CompletedStages);

        var retry = await runner.RunAsync();

        Assert.True(retry.IsReady);
        Assert.Equal(
        [
            SetupStage.EndToEndHealth,
            SetupStage.Activate,
            SetupStage.OllamaModels,
            SetupStage.EndToEndHealth,
        ], actions.Executed);
    }

    [Fact]
    public async Task Restart_runs_from_the_earliest_incomplete_checkpoint()
    {
        var journal = new MemoryJournal
        {
            State = SetupState.Initial with
            {
                Stage = SetupStage.Verify,
                TermsAccepted = true,
                CompletedStages = ImmutableHashSet.Create(SetupStage.Preflight),
            },
        };
        var actions = new RecordingActions();
        var runner = new FirstRunSetupRunner(new SetupCoordinator(journal), actions);

        var result = await runner.RunAsync();

        Assert.True(result.IsReady);
        Assert.Equal(
        [
            SetupStage.ChooseDataRoot, SetupStage.Prerequisites, SetupStage.FetchCatalog,
            SetupStage.Download, SetupStage.Verify, SetupStage.Activate,
            SetupStage.OllamaModels, SetupStage.EndToEndHealth,
        ], actions.Executed);
    }

    private static SetupState StateAt(SetupStage stage) => SetupState.Initial with
    {
        Stage = stage,
        TermsAccepted = true,
        CompletedStages = Enum.GetValues<SetupStage>().Where(value => value < stage).ToImmutableHashSet(),
    };

    private class MemoryJournal : ISetupJournal
    {
        public SetupState? State { get; set; }
        public virtual void Save(SetupState state) => State = state;
        public SetupState? Load() => State;
    }

    private sealed class FailingJournal : MemoryJournal
    {
        public bool FailSaves { get; init; }
        public override void Save(SetupState state)
        {
            if (FailSaves) throw new IOException("durable storage unavailable");
            base.Save(state);
        }
    }

    private sealed class RecordingActions : ISetupStageActions, ISetupRunFailureCleanup
    {
        public List<SetupStage> Executed { get; } = [];
        public SetupStage? FailureStage { get; set; }
        public int FailedRunCleanups { get; private set; }

        public Task PreflightAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.Preflight);
        public Task ChooseDataRootAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.ChooseDataRoot);
        public Task PrerequisitesAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.Prerequisites);
        public Task FetchCatalogAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.FetchCatalog);
        public Task DownloadAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.Download);
        public Task VerifyAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.Verify);
        public Task ActivateAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.Activate);
        public Task OllamaModelsAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.OllamaModels);
        public Task EndToEndHealthAsync(CancellationToken cancellationToken) => RunAsync(SetupStage.EndToEndHealth);
        public Task CleanupAfterFailedRunAsync()
        {
            FailedRunCleanups++;
            return Task.CompletedTask;
        }

        private Task RunAsync(SetupStage stage)
        {
            Executed.Add(stage);
            if (FailureStage != stage) return Task.CompletedTask;
            FailureStage = null;
            return Task.FromException(new InvalidOperationException($"{stage} failed"));
        }
    }
}
