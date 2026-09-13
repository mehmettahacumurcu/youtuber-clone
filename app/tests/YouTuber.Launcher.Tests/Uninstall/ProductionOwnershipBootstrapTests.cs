using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Uninstall;
using Xunit;

namespace YouTuber.Launcher.Tests.Uninstall;

public sealed class ProductionOwnershipBootstrapTests
{
    [Fact]
    public async Task Reserves_journal_lock_and_every_launcher_exclusive_tree_before_creation()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var dataRoot = Path.Combine(directory.Path, "custom-data");
        var paths = AppPaths.ForBaseDirectory(localAppData, dataRoot);

        await ProductionOwnershipBootstrap.InitializeAsync(paths);

        var ownedFiles = PopulateProductionSessionTree(paths);
        var plan = new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData);
        foreach (var ownedFile in ownedFiles)
            Assert.Contains(ownedFile, plan.Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Bootstrap_is_idempotent_for_an_existing_installation()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var paths = AppPaths.ForBaseDirectory(localAppData, Path.Combine(directory.Path, "custom-data"));

        await ProductionOwnershipBootstrap.InitializeAsync(paths);
        await ProductionOwnershipBootstrap.InitializeAsync(paths);

        Assert.True(File.Exists(Path.Combine(paths.StateRoot, "ownership.json")));
    }

    [Fact]
    public async Task Bootstrap_never_adopts_a_preexisting_exact_file_without_prior_ownership()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var paths = AppPaths.ForBaseDirectory(localAppData, Path.Combine(directory.Path, "custom-data"));
        Directory.CreateDirectory(paths.StateRoot);
        File.WriteAllText(paths.SetupJournalFile, "neighbor data");

        await Assert.ThrowsAsync<InvalidDataException>(() => ProductionOwnershipBootstrap.InitializeAsync(paths));

        Assert.Equal("neighbor data", File.ReadAllText(paths.SetupJournalFile));
        var plan = new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData);
        Assert.DoesNotContain(paths.SetupJournalFile, plan.Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Real_production_session_tree_full_uninstall_removes_owned_outputs_but_preserves_neighbor()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var paths = AppPaths.ForBaseDirectory(localAppData, Path.Combine(directory.Path, "custom-data"));
        await ProductionOwnershipBootstrap.InitializeAsync(paths);
        var ownedFiles = PopulateProductionSessionTree(paths);
        var neighbor = Path.Combine(paths.DataRoot, "keep-my-notes.txt");
        File.WriteAllText(neighbor, "not launcher owned");

        var plan = new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData);
        new OwnedDataDeletionExecutor().Execute(paths.DataRoot, plan);

        foreach (var ownedFile in ownedFiles) Assert.False(File.Exists(ownedFile));
        Assert.True(File.Exists(neighbor));
        Assert.False(File.Exists(Path.Combine(paths.StateRoot, "ownership.json")));
    }

    private static string[] PopulateProductionSessionTree(AppPaths paths)
    {
        var files = new[]
        {
            paths.SetupJournalFile,
            paths.SetupJournalFile + ".lock",
            Path.Combine(paths.LogsRoot, "studio.log"),
            Path.Combine(paths.StateRoot, "webview2", "Default", "Preferences"),
            Path.Combine(paths.GeneratedAudioRoot, "answer.wav"),
            Path.Combine(paths.DownloadsRoot, "runtime.zip.part"),
            Path.Combine(paths.StateRoot, "activation-staging", "runtime", "payload.bin"),
            Path.Combine(paths.StateRoot, "activation-transactions", "runtime.json"),
            Path.Combine(paths.StateRoot, "asset-staging", "rag", "index.bin"),
        };
        foreach (var file in files)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(file)!);
            File.WriteAllText(file, "owned");
        }
        return files;
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-production-owned-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); }
            catch { }
        }
    }
}
