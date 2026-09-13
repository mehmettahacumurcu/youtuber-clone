using System.Collections.Immutable;

namespace YouTuber.Launcher.Setup;

public sealed record ComponentProgress(string Name, string Version, long BytesReceived, long TotalBytes, double BytesPerSecond, TimeSpan? Eta, string Status, string? Error = null);
public sealed record SetupState(SetupStage Stage, bool TermsAccepted, bool IsPaused, string? BlockingError, IReadOnlyList<ComponentProgress> Components, bool ReadyRecorded)
{
    public ImmutableHashSet<SetupStage> CompletedStages { get; init; } = ImmutableHashSet<SetupStage>.Empty;
    public long Revision { get; init; }
    public static SetupState Initial { get; } = new(SetupStage.Preflight, false, false, null, [], false);
}
public interface ISetupJournal { SetupState? Load(); void Save(SetupState state); }
public interface ISetupOperationControl
{
    Task<DownloadPauseResult> PauseDownloadsAsync(CancellationToken cancellationToken);
    Task<ActivationCloseResult> WaitForActivationSafePointAsync(CancellationToken cancellationToken);
    Task<ComponentOperationResult> PauseComponentAsync(string component, CancellationToken cancellationToken);
    Task<ComponentOperationResult> ResumeComponentAsync(string component, CancellationToken cancellationToken);
    Task<ComponentOperationResult> RetryComponentAsync(string component, CancellationToken cancellationToken);
}
public enum DownloadPauseResult { PausedAndPersisted, Failed }
public enum ActivationCloseResult { AtomicCommitCompleted, RolledBackSafe, Unsafe }
public enum ComponentOperationResult { ConfirmedDurable, Rejected }
public interface ISetupRecoveryProbe { Task<SetupRecoveryEvidence> ProbeAsync(CancellationToken cancellationToken = default); }
public sealed record SetupRecoveryEvidence(bool DownloadArtifactsPresent, IReadOnlyList<ComponentProgress> ResumableComponents, bool UnfinishedActivation, bool ActivePointersComplete);
public enum SetupErrorCategory { UnsupportedHardware, InsufficientSpace, SmartScreen, NetworkRetry, HashMismatch, PrerequisiteElevation }
public static class SetupErrors
{
    public static string Message(SetupErrorCategory category) => category switch
    {
        SetupErrorCategory.UnsupportedHardware => "This hardware is not supported. Review GPU and system requirements.",
        SetupErrorCategory.InsufficientSpace => "Insufficient disk space. Free space or choose another data location.",
        SetupErrorCategory.SmartScreen => "Windows SmartScreen blocked a prerequisite. Review the verified publisher prompt and retry.",
        SetupErrorCategory.NetworkRetry => "Network download failed. Check the connection and retry the affected component.",
        SetupErrorCategory.HashMismatch => "Downloaded files did not match their hash. Retry to download a fresh verified copy.",
        SetupErrorCategory.PrerequisiteElevation => "A prerequisite needs administrator elevation. Approve the Windows prompt and retry.",
        _ => throw new ArgumentOutOfRangeException(nameof(category)),
    };
}
public interface IUiDispatcher { void Post(Action action); }
public sealed class ImmediateUiDispatcher : IUiDispatcher { public void Post(Action action) => action(); }
