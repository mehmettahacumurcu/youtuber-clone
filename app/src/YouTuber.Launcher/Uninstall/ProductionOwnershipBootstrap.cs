using System.IO;
using YouTuber.Launcher.Configuration;

namespace YouTuber.Launcher.Uninstall;

/// <summary>
/// Durably declares launcher-owned files and exclusive trees before production code can create them.
/// A crash after any later write therefore still leaves enough inventory for a safe full uninstall.
/// </summary>
public static class ProductionOwnershipBootstrap
{
    public static async Task InitializeAsync(AppPaths paths, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var ownership = new OwnershipStore(paths.DataRoot, UninstallPreparation.InstallId);

        var exactFiles = new[]
        {
            paths.SetupJournalFile,
            paths.SetupJournalFile + ".lock",
            paths.SettingsFile,
            paths.ActiveComponentsFile,
            Path.Combine(paths.StateRoot, "activation.lock"),
            Path.Combine(paths.StateRoot, "uninstall-residual.json"),
        };
        foreach (var file in exactFiles)
        {
            await ownership.ReserveOwnedFileAsync(file, cancellationToken);
        }

        var exclusiveTrees = new[]
        {
            paths.LogsRoot,
            paths.GeneratedAudioRoot,
            paths.DownloadsRoot,
            Path.Combine(paths.StateRoot, "webview2"),
            Path.Combine(paths.StateRoot, "activation-staging"),
            Path.Combine(paths.StateRoot, "activation-transactions"),
            Path.Combine(paths.StateRoot, "asset-staging"),
        };
        foreach (var tree in exclusiveTrees)
        {
            await ownership.ReserveExclusiveTreeRootAsync(tree, cancellationToken);
        }
    }
}
