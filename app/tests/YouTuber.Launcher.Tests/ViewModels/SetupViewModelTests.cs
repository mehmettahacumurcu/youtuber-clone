using System.Collections.Immutable;
using YouTuber.Launcher.Setup;
using YouTuber.Launcher.ViewModels;
using Xunit;
namespace YouTuber.Launcher.Tests.ViewModels;
public sealed class SetupViewModelTests
{
    [Fact]
    public async Task Background_progress_is_marshaled_through_the_ui_dispatcher()
    {
        var coordinator = new SetupCoordinator(new Journal()); var dispatcher = new RecordingDispatcher(); var viewModel = new SetupViewModel(coordinator, dispatcher);
        await Task.Run(() => coordinator.ReportComponent(new ComponentProgress("voice", "1", 1, 2, 1, null, "Downloading")));
        Assert.Equal(1, dispatcher.Posts); Assert.Single(viewModel.Components);
    }
    [Fact]
    public void Out_of_order_dispatch_callbacks_cannot_regress_a_newer_progress_revision()
    {
        var coordinator = new SetupCoordinator(new Journal());
        var dispatcher = new QueuedDispatcher();
        var viewModel = new SetupViewModel(coordinator, dispatcher);
        coordinator.ReportComponent(new ComponentProgress("voice", "1", 1, 10, 1, null, "Downloading"));
        coordinator.ReportComponent(new ComponentProgress("voice", "1", 9, 10, 1, null, "Downloading"));

        dispatcher.Actions[1]();
        dispatcher.Actions[0]();

        Assert.Equal(9, Assert.Single(viewModel.Components).BytesReceived);
    }
    [Fact]
    public void Chat_stays_closed_until_ready_is_durably_recorded()
    {
        var journal = new Journal(); var coordinator = new SetupCoordinator(journal); var viewModel = new SetupViewModel(coordinator);
        Assert.False(viewModel.CanOpenChat);
        coordinator.Complete(SetupStage.Preflight); coordinator.Complete(SetupStage.ChooseDataRoot); coordinator.AcceptTerms(); coordinator.Complete(SetupStage.AcceptTerms);
        while (coordinator.State.Stage != SetupStage.Ready) coordinator.Complete(coordinator.State.Stage);
        Assert.True(viewModel.CanOpenChat); Assert.True(journal.State!.ReadyRecorded);
    }
    [Fact]
    public async Task Component_commands_pause_resume_and_retry_the_target_component()
    {
        var coordinator = new SetupCoordinator(new Journal(), new RecordingOperations());
        var viewModel = new SetupViewModel(coordinator);
        coordinator.ReportComponent(new ComponentProgress("voice", "1", 5, 10, 2, TimeSpan.FromSeconds(3), "Downloading"));
        var component = Assert.Single(viewModel.Components);

        Assert.True(component.PauseCommand.CanExecute(null));
        await component.PauseCommand.ExecuteAsync();
        Assert.True(coordinator.State.IsPaused);
        Assert.Equal("Paused", coordinator.State.Components.Single().Status);
        Assert.True(component.ResumeCommand.CanExecute(null));
        await component.ResumeCommand.ExecuteAsync();
        Assert.False(coordinator.State.IsPaused);
        Assert.Equal("Downloading", coordinator.State.Components.Single().Status);

        coordinator.ReportComponent(new ComponentProgress("voice", "1", 5, 10, 0, null, "Failed", "hash mismatch"));
        Assert.True(component.RetryCommand.CanExecute(null));
        await component.RetryCommand.ExecuteAsync();
        Assert.Equal("Queued", coordinator.State.Components.Single().Status);
        Assert.Equal("5 B / 10 B", component.TransferredText);
    }

    [Fact]
    public async Task View_model_exposes_the_fail_closed_window_close_contract()
    {
        var coordinator = new SetupCoordinator(new Journal { State = StateAt(SetupStage.Download) });
        var viewModel = new SetupViewModel(coordinator);

        Assert.False(await viewModel.RequestCloseAsync());
    }

    [Fact]
    public void Accepting_terms_from_the_view_model_updates_the_durable_setup_state()
    {
        var journal = new Journal { State = SetupState.Initial with
        {
            Stage = SetupStage.AcceptTerms,
            CompletedStages = ImmutableHashSet.Create(SetupStage.Preflight, SetupStage.ChooseDataRoot),
        } };
        var viewModel = new SetupViewModel(new SetupCoordinator(journal));

        Assert.True(viewModel.CanAcceptTerms);
        viewModel.AcceptTerms();

        Assert.False(viewModel.CanAcceptTerms);
        Assert.True(journal.State!.TermsAccepted);
    }

    private static SetupState StateAt(SetupStage stage) => SetupState.Initial with
    {
        Stage = stage,
        TermsAccepted = true,
        CompletedStages = Enum.GetValues<SetupStage>().Where(value => value < stage).ToImmutableHashSet(),
    };
    private sealed class Journal : ISetupJournal { public SetupState? State; public SetupState? Load() => State; public void Save(SetupState state) => State = state; }
    private sealed class RecordingDispatcher : IUiDispatcher { public int Posts; public void Post(Action action) { Posts++; action(); } }
    private sealed class QueuedDispatcher : IUiDispatcher { public List<Action> Actions { get; } = []; public void Post(Action action) => Actions.Add(action); }
    private sealed class RecordingOperations : ISetupOperationControl
    {
        public Task<DownloadPauseResult> PauseDownloadsAsync(CancellationToken cancellationToken) => Task.FromResult(DownloadPauseResult.PausedAndPersisted);
        public Task<ActivationCloseResult> WaitForActivationSafePointAsync(CancellationToken cancellationToken) => Task.FromResult(ActivationCloseResult.AtomicCommitCompleted);
        public Task<ComponentOperationResult> PauseComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentOperationResult.ConfirmedDurable);
        public Task<ComponentOperationResult> ResumeComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentOperationResult.ConfirmedDurable);
        public Task<ComponentOperationResult> RetryComponentAsync(string component, CancellationToken cancellationToken) => Task.FromResult(ComponentOperationResult.ConfirmedDurable);
    }
}
