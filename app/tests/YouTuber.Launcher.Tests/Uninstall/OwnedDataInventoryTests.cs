using System.Text.Json;
using YouTuber.Launcher.Uninstall;
using Xunit;

namespace YouTuber.Launcher.Tests.Uninstall;

public sealed class OwnedDataInventoryTests
{
    private const string InstallId = "dd8ad339-2d57-4b60-9770-c9e606380bb8";

    [Fact]
    public async Task Ownership_store_atomically_creates_and_merges_exact_owned_paths()
    {
        using var directory = new TemporaryDirectory();
        var firstDirectory = Directory.CreateDirectory(Path.Combine(directory.Path, "runtime", "1.0.0")).FullName;
        var secondDirectory = Directory.CreateDirectory(Path.Combine(directory.Path, "rag")).FullName;
        var first = Path.Combine(firstDirectory, "worker.exe");
        var second = Path.Combine(secondDirectory, "index.bin");
        var neighbor = Path.Combine(secondDirectory, "user-notes.txt");
        File.WriteAllText(first, "worker");
        File.WriteAllText(second, "index");
        File.WriteAllText(neighbor, "user");

        await Task.WhenAll(
            new OwnershipStore(directory.Path, InstallId).RegisterExistingPathsAsync([first]),
            new OwnershipStore(directory.Path, InstallId).RegisterExistingPathsAsync([second]));

        var plan = new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData);
        Assert.Contains(first, plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.Contains(second, plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain(neighbor, plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.Contains(plan.Directories, entry => string.Equals(entry.Path, firstDirectory, StringComparison.OrdinalIgnoreCase));
        Assert.Contains(plan.Directories, entry => string.Equals(entry.Path, secondDirectory, StringComparison.OrdinalIgnoreCase));
        Assert.Empty(Directory.EnumerateFiles(Path.Combine(directory.Path, "state"), "*.tmp"));
    }

    [Fact]
    public void App_only_uninstall_retains_all_data()
    {
        using var directory = new TemporaryDirectory();
        WriteManifest(directory.Path, InstallId, ["rag/owned.bin"]);
        Directory.CreateDirectory(Path.Combine(directory.Path, "rag"));
        File.WriteAllText(Path.Combine(directory.Path, "rag", "owned.bin"), "owned");
        Assert.Empty(new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppOnly).Files);
    }

    [Fact]
    public void Full_uninstall_selects_exact_owned_files_but_preserves_neighboring_ollama_models()
    {
        using var directory = new TemporaryDirectory();
        WriteManifest(directory.Path, InstallId, ["models/ollama/youtuber-owned.bin", "models/ollama"]);
        var models = Directory.CreateDirectory(Path.Combine(directory.Path, "models", "ollama")).FullName;
        var owned = Path.Combine(models, "youtuber-owned.bin");
        var unrelated = Path.Combine(models, "unrelated-model.bin");
        File.WriteAllText(owned, "owned");
        File.WriteAllText(unrelated, "unrelated");

        var plan = new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData);

        Assert.Contains(owned, plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.DoesNotContain(unrelated, plan.Files, StringComparer.OrdinalIgnoreCase);
        Assert.Contains(plan.Directories, entry => string.Equals(entry.Path, models, StringComparison.OrdinalIgnoreCase) && entry.DeleteOnlyIfEmpty);
    }

    [Fact]
    public async Task Exclusive_tree_registration_owns_model_files_created_after_the_durable_registration()
    {
        using var directory = new TemporaryDirectory();
        var models = Directory.CreateDirectory(Path.Combine(directory.Path, "models", "ollama")).FullName;
        var neighbor = Path.Combine(directory.Path, "models", "user-model.bin");
        File.WriteAllText(neighbor, "user");
        var store = new OwnershipStore(directory.Path, InstallId);
        await store.RegisterExclusiveTreeRootAsync(models);

        var crashOutput = Path.Combine(models, "blobs", "sha256-model");
        Directory.CreateDirectory(Path.GetDirectoryName(crashOutput)!);
        File.WriteAllText(crashOutput, "created by Ollama immediately before process loss");
        var plan = new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData);

        Assert.Contains(crashOutput, plan.Files, StringComparer.OrdinalIgnoreCase);
        new OwnedDataDeletionExecutor().Execute(directory.Path, plan);
        Assert.False(Directory.Exists(models));
        Assert.True(File.Exists(neighbor));
    }

    [Fact]
    public async Task Reserved_exclusive_tree_owns_an_asset_atomically_moved_into_place_after_registration()
    {
        using var directory = new TemporaryDirectory();
        var target = Path.Combine(directory.Path, "rag", "rag", "1.0.0-abcdef");
        var store = new OwnershipStore(directory.Path, InstallId);
        await store.ReserveExclusiveTreeRootAsync(target);

        Directory.CreateDirectory(target);
        var crashOutput = Path.Combine(target, "index.bin");
        File.WriteAllText(crashOutput, "moved before process loss");
        var plan = new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData);

        Assert.Contains(crashOutput, plan.Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public void Missing_marker_or_another_install_id_never_selects_data()
    {
        using var directory = new TemporaryDirectory();
        Assert.Empty(new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData).Files);
        WriteManifest(directory.Path, Guid.NewGuid().ToString(), ["rag/owned.bin"]);
        Assert.Empty(new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData).Files);
    }

    [Fact]
    public void Full_uninstall_selects_the_matching_ownership_marker_even_when_owned_targets_are_already_absent()
    {
        using var directory = new TemporaryDirectory();
        WriteManifest(directory.Path, InstallId, ["rag/already-absent.bin"]);

        var plan = new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData);

        Assert.Single(plan.Files);
        Assert.EndsWith(Path.Combine("state", "ownership.json"), plan.Files[0], StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("../outside")]
    [InlineData("rag/../../outside")]
    [InlineData("C:/outside")]
    [InlineData("rag\\outside")]
    public void Traversal_or_noncanonical_owned_paths_fail_closed(string unsafePath)
    {
        using var directory = new TemporaryDirectory();
        WriteManifest(directory.Path, InstallId, [unsafePath]);
        Assert.Throws<InvalidDataException>(() => new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData));
    }

    [Fact]
    public void Reparse_point_owned_paths_fail_closed_and_preserve_the_ownership_marker()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var directory = new TemporaryDirectory();
        using var outside = new TemporaryDirectory();
        try { Directory.CreateSymbolicLink(Path.Combine(directory.Path, "linked"), outside.Path); }
        catch (UnauthorizedAccessException) { return; }
        WriteManifest(directory.Path, InstallId, ["linked"]);

        Assert.Throws<InvalidDataException>(() =>
            new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData));
        Assert.True(File.Exists(Path.Combine(directory.Path, "state", "ownership.json")));
    }

    [Fact]
    public async Task Preexisting_ownership_lock_is_never_adopted()
    {
        using var directory = new TemporaryDirectory();
        var state = Directory.CreateDirectory(Path.Combine(directory.Path, "state")).FullName;
        var ownershipLock = Path.Combine(state, "ownership.lock");
        File.WriteAllText(ownershipLock, "neighbor lock");

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            new OwnershipStore(directory.Path, InstallId).ReserveOwnedFileAsync(Path.Combine(state, "future.json")));

        Assert.Equal("neighbor lock", File.ReadAllText(ownershipLock));
        Assert.False(File.Exists(Path.Combine(state, "ownership.json")));
    }

    [Fact]
    public void Executor_deletes_only_exact_owned_files_and_empty_directories()
    {
        using var directory = new TemporaryDirectory();
        WriteManifest(directory.Path, InstallId, ["owned/file.bin", "owned"]);
        var ownedDirectory = Directory.CreateDirectory(Path.Combine(directory.Path, "owned")).FullName;
        var ownedFile = Path.Combine(ownedDirectory, "file.bin");
        var neighbor = Path.Combine(ownedDirectory, "neighbor.bin");
        File.WriteAllText(ownedFile, "owned");
        File.WriteAllText(neighbor, "user");
        var plan = new OwnedDataInventory().Plan(directory.Path, InstallId, UninstallChoice.AppAndOwnedData);

        new OwnedDataDeletionExecutor().Execute(directory.Path, plan);

        Assert.False(File.Exists(ownedFile));
        Assert.True(File.Exists(neighbor));
        Assert.True(Directory.Exists(ownedDirectory));
        Assert.False(File.Exists(Path.Combine(directory.Path, "state", "ownership.json")));
        var residualReport = Path.Combine(directory.Path, "state", "uninstall-residual.json");
        Assert.True(File.Exists(residualReport));
        Assert.Contains("owned", File.ReadAllText(residualReport), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Preparation_without_cli_root_uses_locator_and_full_uninstall_preserves_custom_root_neighbor()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var dataRoot = Path.Combine(directory.Path, "custom-data");
        var locator = YouTuber.Launcher.Configuration.DataRootLocator.ForBaseDirectory(localAppData);
        await locator.SaveAsync(dataRoot);
        var paths = YouTuber.Launcher.Configuration.AppPaths.ForBaseDirectory(localAppData, dataRoot);
        await ProductionOwnershipBootstrap.InitializeAsync(paths);
        var owned = Path.Combine(paths.LogsRoot, "launcher.log");
        Directory.CreateDirectory(paths.LogsRoot);
        File.WriteAllText(owned, "owned");
        var neighbor = Path.Combine(paths.DataRoot, "user-neighbor.txt");
        File.WriteAllText(neighbor, "user");

        await UninstallPreparation.RunAsync(
            ["--prepare-uninstall", "--uninstall-choice=app-and-owned-data"],
            locator);

        Assert.False(File.Exists(owned));
        Assert.True(File.Exists(neighbor));
        Assert.False(File.Exists(locator.LocatorFile));
        Assert.False(Directory.Exists(locator.BootstrapRoot));
    }

    [Fact]
    public async Task App_only_uninstall_retains_locator_and_custom_data()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var dataRoot = Path.Combine(directory.Path, "custom-data");
        var locator = YouTuber.Launcher.Configuration.DataRootLocator.ForBaseDirectory(localAppData);
        await locator.SaveAsync(dataRoot);
        Directory.CreateDirectory(dataRoot);
        var userData = Path.Combine(dataRoot, "keep.bin");
        File.WriteAllText(userData, "keep");

        await UninstallPreparation.RunAsync(
            ["--prepare-uninstall", "--uninstall-choice=app-only"],
            locator);

        Assert.Equal(Path.GetFullPath(dataRoot), locator.LoadRequired());
        Assert.True(File.Exists(userData));
    }

    [Fact]
    public async Task Full_uninstall_without_matching_ownership_marker_retains_locator()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var dataRoot = Path.Combine(directory.Path, "custom-data");
        var locator = YouTuber.Launcher.Configuration.DataRootLocator.ForBaseDirectory(localAppData);
        await locator.SaveAsync(dataRoot);

        await UninstallPreparation.RunAsync(
            ["--prepare-uninstall", "--uninstall-choice=app-and-owned-data"],
            locator);

        Assert.Equal(Path.GetFullPath(dataRoot), locator.LoadRequired());
    }

    [Fact]
    public async Task Full_uninstall_with_residual_report_retains_locator_for_follow_up()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var dataRoot = Path.Combine(directory.Path, "custom-data");
        var locator = YouTuber.Launcher.Configuration.DataRootLocator.ForBaseDirectory(localAppData);
        await locator.SaveAsync(dataRoot);
        var paths = YouTuber.Launcher.Configuration.AppPaths.ForBaseDirectory(localAppData, dataRoot);
        await ProductionOwnershipBootstrap.InitializeAsync(paths);
        var neighbor = Path.Combine(paths.StateRoot, "user-neighbor.txt");
        File.WriteAllText(neighbor, "keep");

        await UninstallPreparation.RunAsync(
            ["--prepare-uninstall", "--uninstall-choice=app-and-owned-data"],
            locator);

        Assert.True(File.Exists(neighbor));
        Assert.True(File.Exists(Path.Combine(paths.StateRoot, "uninstall-residual.json")));
        Assert.Equal(Path.GetFullPath(dataRoot), locator.LoadRequired());
    }

    [Fact]
    public async Task Corrupt_locator_fails_safe_without_touching_possible_owned_data()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var dataRoot = Directory.CreateDirectory(Path.Combine(directory.Path, "custom-data")).FullName;
        var owned = Path.Combine(dataRoot, "owned.bin");
        File.WriteAllText(owned, "must survive locator failure");
        var locator = YouTuber.Launcher.Configuration.DataRootLocator.ForBaseDirectory(localAppData);
        Directory.CreateDirectory(Path.GetDirectoryName(locator.LocatorFile)!);
        File.WriteAllText(locator.LocatorFile, "{broken");

        await Assert.ThrowsAsync<InvalidDataException>(() => UninstallPreparation.RunAsync(
            ["--prepare-uninstall", "--uninstall-choice=app-and-owned-data"],
            locator));

        Assert.True(File.Exists(owned));
    }

    [Fact]
    public async Task Preparation_rejects_unknown_or_ambiguous_arguments_before_deletion()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => UninstallPreparation.RunAsync(["--prepare-uninstall", "--uninstall-choice=everything"]));
        await Assert.ThrowsAsync<ArgumentException>(() => UninstallPreparation.RunAsync(["--prepare-uninstall", "--uninstall-choice=app-only", "--unknown"]));
        await Assert.ThrowsAsync<ArgumentException>(() => UninstallPreparation.RunAsync(["--prepare-uninstall", "--uninstall-choice-file=C:\\outside"]));
    }

    [Fact]
    public async Task Preparation_rejects_choice_file_content_that_is_not_an_exact_choice()
    {
        var choiceFile = Path.Combine(AppContext.BaseDirectory, "uninstall-choice.txt");
        var previous = File.Exists(choiceFile) ? File.ReadAllText(choiceFile) : null;
        try
        {
            File.WriteAllText(choiceFile, "app-only\n");
            await Assert.ThrowsAsync<ArgumentException>(() => UninstallPreparation.RunAsync(["--prepare-uninstall", "--uninstall-choice-file"]));
        }
        finally
        {
            if (previous is null) File.Delete(choiceFile); else File.WriteAllText(choiceFile, previous);
        }
    }

    private static void WriteManifest(string root, string installId, string[] paths)
    {
        var state = Directory.CreateDirectory(Path.Combine(root, "state")).FullName;
        File.WriteAllText(Path.Combine(state, "ownership.json"), JsonSerializer.Serialize(new { schema = "youtuber.ownership.v1", install_id = installId, owned_relative_paths = paths }));
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory() { Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-owned-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(Path); }
        public string Path { get; }
        public void Dispose() { try { Directory.Delete(Path, true); } catch { } }
    }
}
