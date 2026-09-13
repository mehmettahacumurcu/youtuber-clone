using System.Collections.Immutable;

namespace YouTuber.Launcher.Setup;

public sealed class SetupCoordinator
{
    private readonly ISetupJournal _journal;
    private readonly ISetupOperationControl? _operations;
    private readonly object _gate = new();
    private SetupState _state;

    public SetupCoordinator(ISetupJournal journal, ISetupOperationControl? operations = null)
    {
        _journal = journal ?? throw new ArgumentNullException(nameof(journal));
        _operations = operations;
        _state = EarliestIncomplete(journal.Load() ?? SetupState.Initial);
    }

    public SetupState State { get { lock (_gate) return _state; } }
    public event EventHandler<SetupState>? Changed;

    public static async Task<SetupCoordinator> CreateAsync(
        ISetupJournal journal,
        ISetupRecoveryProbe recovery,
        ISetupOperationControl? operations = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(recovery);
        var coordinator = new SetupCoordinator(journal, operations);
        var evidence = await recovery.ProbeAsync(cancellationToken);
        var state = coordinator.State;
        if (state.CompletedStages.Contains(SetupStage.Download) && !evidence.DownloadArtifactsPresent)
        {
            coordinator.RecoverAt(SetupStage.Download, evidence.ResumableComponents);
        }
        else if (state.Stage == SetupStage.Download && evidence.ResumableComponents.Count != 0)
        {
            coordinator.RecoverAt(SetupStage.Download, evidence.ResumableComponents);
        }
        else
        {
            var activationWasReached = state.Stage >= SetupStage.Activate || state.CompletedStages.Contains(SetupStage.Activate);
            if (activationWasReached && (evidence.UnfinishedActivation || (state.CompletedStages.Contains(SetupStage.Activate) && !evidence.ActivePointersComplete)))
            {
                coordinator.RecoverAt(SetupStage.Activate);
            }
        }
        return coordinator;
    }

    public void Complete(SetupStage stage) => Update(state =>
    {
        if (stage != state.Stage) throw new InvalidOperationException("Setup stages must complete in order.");
        if (stage == SetupStage.Preflight && state.BlockingError is not null) throw new InvalidOperationException(state.BlockingError);
        if (stage == SetupStage.AcceptTerms && !state.TermsAccepted) throw new InvalidOperationException("Terms must be accepted before setup continues.");
        var next = stage == SetupStage.Ready ? SetupStage.Ready : (SetupStage)((int)stage + 1);
        return state with
        {
            Stage = next,
            IsPaused = false,
            ReadyRecorded = next == SetupStage.Ready,
            CompletedStages = state.CompletedStages.Add(stage),
        };
    });

    public void AcceptTerms() => Update(state =>
    {
        if (state.Stage != SetupStage.AcceptTerms) throw new InvalidOperationException("Terms cannot be accepted before the terms stage.");
        return state with { TermsAccepted = true };
    });

    public void FailPreflight(string message)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message);
        Update(state => state.Stage == SetupStage.Preflight ? state with { BlockingError = message } : throw new InvalidOperationException());
    }

    public void ClearPreflightFailure() => Update(state =>
        state.Stage == SetupStage.Preflight ? state with { BlockingError = null } : throw new InvalidOperationException());

    public void Pause() => Update(state => state with { IsPaused = true });
    public void Resume() => Update(state => state with { IsPaused = false });

    internal void RewindToRepair() => RecoverAt(SetupStage.Activate);

    public void ReportComponent(ComponentProgress progress)
    {
        ArgumentNullException.ThrowIfNull(progress);
        Update(state => state with
        {
            Components = state.Components.Where(component => component.Name != progress.Name).Append(progress).ToArray(),
        });
    }

    public Task<bool> PauseComponentAsync(string component, CancellationToken cancellationToken = default) =>
        RunComponentOperationAsync(component, "Downloading", "Paused", (operations, name, token) => operations.PauseComponentAsync(name, token), cancellationToken);

    public Task<bool> ResumeComponentAsync(string component, CancellationToken cancellationToken = default) =>
        RunComponentOperationAsync(component, "Paused", "Downloading", (operations, name, token) => operations.ResumeComponentAsync(name, token), cancellationToken);

    public Task<bool> RetryComponentAsync(string component, CancellationToken cancellationToken = default) =>
        RunComponentOperationAsync(component, "Failed", "Queued", (operations, name, token) => operations.RetryComponentAsync(name, token), cancellationToken, clearError: true);

    public bool CanOperateComponent(string component, string expectedStatus) =>
        _operations is not null && State.Components.Any(value => value.Name == component && value.Status == expectedStatus);

    public async Task<bool> RequestCloseAsync(CancellationToken cancellationToken = default)
    {
        var stage = State.Stage;
        if (stage == SetupStage.Download)
        {
            if (_operations is null) return false;
            try
            {
                if (await _operations.PauseDownloadsAsync(cancellationToken) != DownloadPauseResult.PausedAndPersisted) return false;
                return TryUpdate(state => state.Stage == SetupStage.Download ? state with { IsPaused = true } : null);
            }
            catch (Exception) { return false; }
        }
        if (stage == SetupStage.Activate)
        {
            if (_operations is null) return false;
            try
            {
                var result = await _operations.WaitForActivationSafePointAsync(cancellationToken);
                if (result is not (ActivationCloseResult.AtomicCommitCompleted or ActivationCloseResult.RolledBackSafe)) return false;
                if (State.Stage != SetupStage.Activate) return State.Stage > SetupStage.Activate;
                return TryUpdate(state => state.Stage == SetupStage.Activate ? state with { IsPaused = true } : null);
            }
            catch (Exception) { return false; }
        }
        return true;
    }

    private async Task<bool> RunComponentOperationAsync(
        string component,
        string expectedStatus,
        string replacementStatus,
        Func<ISetupOperationControl, string, CancellationToken, Task<ComponentOperationResult>> operation,
        CancellationToken cancellationToken,
        bool clearError = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(component);
        if (_operations is null || !State.Components.Any(value => value.Name == component && value.Status == expectedStatus)) return false;
        try
        {
            if (await operation(_operations, component, cancellationToken) != ComponentOperationResult.ConfirmedDurable) return false;
            var observed = State.Components.FirstOrDefault(value => value.Name == component);
            if (observed is not null && observed.Status != expectedStatus) return true;
            return TryUpdate(state =>
            {
                var changed = false;
                var components = state.Components.Select(value =>
                {
                    if (value.Name != component || value.Status != expectedStatus) return value;
                    changed = true;
                    return value with { Status = replacementStatus, Error = clearError ? null : value.Error };
                }).ToArray();
                if (!changed) return null;
                return state with
                {
                    Components = components,
                    IsPaused = components.Any(value => value.Status == "Paused"),
                };
            });
        }
        catch (Exception) { return false; }
    }

    private void RecoverAt(SetupStage stage, IReadOnlyList<ComponentProgress>? recoveredComponents = null) => Update(state =>
    {
        if (state.Stage < stage && !state.CompletedStages.Contains(stage)) return state;
        var components = recoveredComponents is null
            ? state.Components
            : state.Components.Where(existing => recoveredComponents.All(recovered => recovered.Name != existing.Name)).Concat(recoveredComponents).ToArray();
        return state with
        {
            Stage = stage,
            ReadyRecorded = false,
            IsPaused = stage == SetupStage.Download && recoveredComponents?.Count > 0,
            Components = components,
            CompletedStages = state.CompletedStages.Where(value => value < stage).ToImmutableHashSet(),
        };
    });

    private void Update(Func<SetupState, SetupState> transition)
    {
        if (!TryUpdate(state => transition(state))) throw new InvalidOperationException("The setup state transition was rejected.");
    }

    private bool TryUpdate(Func<SetupState, SetupState?> transition)
    {
        SetupState? published = null;
        lock (_gate)
        {
            for (var attempt = 0; attempt < 5; attempt++)
            {
                var next = transition(_state);
                if (next is null) return false;
                next = next with { Revision = checked(_state.Revision + 1) };
                try
                {
                    _journal.Save(next);
                    _state = next;
                    published = next;
                    break;
                }
                catch (SetupJournalConflictException)
                {
                    _state = EarliestIncomplete(_journal.Load() ?? SetupState.Initial);
                }
            }
        }
        if (published is null) return false;
        Changed?.Invoke(this, published);
        return true;
    }

    private static SetupState EarliestIncomplete(SetupState state)
    {
        if (state.BlockingError is not null)
        {
            return state with { Stage = SetupStage.Preflight, ReadyRecorded = false, CompletedStages = ImmutableHashSet<SetupStage>.Empty };
        }

        var contiguous = ImmutableHashSet<SetupStage>.Empty;
        foreach (var stage in Enum.GetValues<SetupStage>().Where(stage => stage != SetupStage.Ready))
        {
            if (!state.CompletedStages.Contains(stage) || (stage == SetupStage.AcceptTerms && !state.TermsAccepted))
            {
                return state with { Stage = stage, ReadyRecorded = false, CompletedStages = contiguous };
            }
            contiguous = contiguous.Add(stage);
        }
        return state with { Stage = SetupStage.Ready, ReadyRecorded = state.ReadyRecorded, CompletedStages = contiguous };
    }
}
