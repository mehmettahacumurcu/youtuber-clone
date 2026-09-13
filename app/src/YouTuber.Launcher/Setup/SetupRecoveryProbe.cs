using System.IO;
using YouTuber.Launcher.Activation;
using YouTuber.Launcher.Downloads;

namespace YouTuber.Launcher.Setup;

public sealed record SetupDownloadArtifact(string Name, string Version, DownloadRequest Request);

/// <summary>Reconciles setup checkpoints with the durable downloader and activation journals.</summary>
public sealed class SetupRecoveryProbe : ISetupRecoveryProbe
{
    private readonly IReadOnlyList<SetupDownloadArtifact> _downloads;
    private readonly ActiveComponentsStore _activeComponents;
    private readonly IReadOnlySet<string> _requiredActiveComponents;

    public SetupRecoveryProbe(
        IEnumerable<DownloadRequest> downloads,
        ActiveComponentsStore activeComponents,
        IEnumerable<string> requiredActiveComponents)
        : this(
            (downloads ?? throw new ArgumentNullException(nameof(downloads))).Select(request => new SetupDownloadArtifact(Path.GetFileName(request.TargetPath), string.Empty, request)),
            activeComponents,
            requiredActiveComponents)
    {
    }

    public SetupRecoveryProbe(
        IEnumerable<SetupDownloadArtifact> downloads,
        ActiveComponentsStore activeComponents,
        IEnumerable<string> requiredActiveComponents)
    {
        ArgumentNullException.ThrowIfNull(downloads);
        ArgumentNullException.ThrowIfNull(activeComponents);
        ArgumentNullException.ThrowIfNull(requiredActiveComponents);
        _downloads = downloads.ToArray();
        _activeComponents = activeComponents;
        _requiredActiveComponents = requiredActiveComponents.ToHashSet(StringComparer.Ordinal);
        foreach (var artifact in _downloads)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(artifact.Name);
            artifact.Request.Validate();
        }
    }

    public async Task<SetupRecoveryEvidence> ProbeAsync(CancellationToken cancellationToken = default)
    {
        var complete = true;
        var resumable = new List<ComponentProgress>();
        foreach (var artifact in _downloads)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var request = artifact.Request;
            var finalLength = File.Exists(request.TargetPath) ? new FileInfo(request.TargetPath).Length : -1;
            if (finalLength == request.ExpectedSize)
            {
                try
                {
                    if (await Sha256Verifier.MatchesAsync(request.TargetPath, request.ExpectedSha256, cancellationToken)) continue;
                }
                catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
                {
                    // A completed artifact that cannot be read is not durable recovery evidence.
                }
            }

            complete = false;
            var partialPath = request.TargetPath + ".partial";
            var partialLength = File.Exists(partialPath) ? new FileInfo(partialPath).Length : 0;
            var metadata = ResumeMetadata.Load(partialPath + ".json");
            if (partialLength > 0 && metadata is not null && metadata.Matches(request, partialLength))
            {
                resumable.Add(new ComponentProgress(
                    artifact.Name,
                    artifact.Version,
                    partialLength,
                    request.ExpectedSize,
                    0,
                    null,
                    "Paused"));
            }
        }

        var transactions = await _activeComponents.LoadTransactionsAsync(cancellationToken);
        var active = await _activeComponents.LoadAsync(cancellationToken);
        var pointersComplete = _requiredActiveComponents.All(active.Components.ContainsKey);
        return new SetupRecoveryEvidence(complete, resumable, transactions.Count != 0, pointersComplete);
    }
}
