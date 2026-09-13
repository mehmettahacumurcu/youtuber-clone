namespace YouTuber.Launcher.Setup;

/// <summary>Executes the durable first-run checkpoints in their required order.</summary>
public sealed class FirstRunSetupRunner
{
    private readonly SetupCoordinator _coordinator;
    private readonly ISetupStageActions _actions;

    public FirstRunSetupRunner(SetupCoordinator coordinator, ISetupStageActions actions)
    {
        _coordinator = coordinator ?? throw new ArgumentNullException(nameof(coordinator));
        _actions = actions ?? throw new ArgumentNullException(nameof(actions));
    }

    public void AcceptTerms()
    {
        _coordinator.AcceptTerms();
        _coordinator.Complete(SetupStage.AcceptTerms);
    }

    public async Task<SetupRunResult> RunAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            while (true)
            {
                var stage = _coordinator.State.Stage;
                if (stage == SetupStage.Ready) return SetupRunResult.Ready;
                if (stage == SetupStage.AcceptTerms)
                {
                    if (!_coordinator.State.TermsAccepted) return new SetupRunResult(false, true, stage);
                    _coordinator.Complete(SetupStage.AcceptTerms);
                    continue;
                }

                await ExecuteAsync(stage, cancellationToken);
                _coordinator.Complete(stage);
            }
        }
        catch
        {
            if (_actions is ISetupRunFailureCleanup cleanup)
            {
                try { await cleanup.CleanupAfterFailedRunAsync(); }
                catch { }
            }
            throw;
        }
    }

    public async Task<SetupRunResult> RevalidateReadyAsync(CancellationToken cancellationToken = default)
    {
        if (_coordinator.State is not { Stage: SetupStage.Ready, ReadyRecorded: true })
            throw new InvalidOperationException("Only a durably Ready setup can be revalidated.");

        try
        {
            await _actions.EndToEndHealthAsync(cancellationToken);
            return SetupRunResult.Ready;
        }
        catch
        {
            _coordinator.RewindToRepair();
            if (_actions is ISetupRunFailureCleanup cleanup)
            {
                try { await cleanup.CleanupAfterFailedRunAsync(); }
                catch { }
            }
            throw;
        }
    }

    private Task ExecuteAsync(SetupStage stage, CancellationToken cancellationToken) => stage switch
    {
        SetupStage.Preflight => _actions.PreflightAsync(cancellationToken),
        SetupStage.ChooseDataRoot => _actions.ChooseDataRootAsync(cancellationToken),
        SetupStage.Prerequisites => _actions.PrerequisitesAsync(cancellationToken),
        SetupStage.FetchCatalog => _actions.FetchCatalogAsync(cancellationToken),
        SetupStage.Download => _actions.DownloadAsync(cancellationToken),
        SetupStage.Verify => _actions.VerifyAsync(cancellationToken),
        SetupStage.Activate => _actions.ActivateAsync(cancellationToken),
        SetupStage.OllamaModels => _actions.OllamaModelsAsync(cancellationToken),
        SetupStage.EndToEndHealth => _actions.EndToEndHealthAsync(cancellationToken),
        SetupStage.AcceptTerms => throw new InvalidOperationException("Terms must be accepted explicitly before setup continues."),
        SetupStage.Ready => Task.CompletedTask,
        _ => throw new ArgumentOutOfRangeException(nameof(stage)),
    };
}

public sealed record SetupRunResult(bool IsReady, bool RequiresTermsAcceptance, SetupStage Stage)
{
    public static SetupRunResult Ready { get; } = new(true, false, SetupStage.Ready);
}

/// <summary>Production composition supplies the concrete operation behind each durable checkpoint.</summary>
public interface ISetupStageActions
{
    Task PreflightAsync(CancellationToken cancellationToken);
    Task ChooseDataRootAsync(CancellationToken cancellationToken);
    Task PrerequisitesAsync(CancellationToken cancellationToken);
    Task FetchCatalogAsync(CancellationToken cancellationToken);
    Task DownloadAsync(CancellationToken cancellationToken);
    Task VerifyAsync(CancellationToken cancellationToken);
    Task ActivateAsync(CancellationToken cancellationToken);
    Task OllamaModelsAsync(CancellationToken cancellationToken);
    Task EndToEndHealthAsync(CancellationToken cancellationToken);
}

public interface ISetupRunFailureCleanup
{
    Task CleanupAfterFailedRunAsync();
}
