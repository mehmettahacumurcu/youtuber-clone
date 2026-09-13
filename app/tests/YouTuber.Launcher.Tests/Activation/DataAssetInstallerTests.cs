using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using YouTuber.Launcher.Activation;
using Xunit;

namespace YouTuber.Launcher.Tests.Activation;

public sealed class DataAssetInstallerTests
{
    [Fact]
    public void Parses_the_shared_data_asset_inventory_fixture()
    {
        var inventory = DataAssetInventory.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "asset-manifest.json")));

        Assert.Equal("rag", inventory.Component);
        Assert.Equal("index/corpus.txt", Assert.Single(inventory.Files).Path);
    }

    [Fact]
    public async Task Installs_an_exact_inventory_below_the_fixed_data_root_and_records_the_active_pointer()
    {
        using var fixture = new AssetFixture();
        var installer = new DataAssetInstaller(new ActiveComponentsStore(fixture.DataRoot));

        await installer.InstallAsync(fixture.Request("rag"));

        var active = await new ActiveComponentsStore(fixture.DataRoot).LoadAsync();
        var pointer = active.Components["rag"];
        Assert.StartsWith("rag/rag/1.0.0-", pointer.RelativePath, StringComparison.Ordinal);
        Assert.Equal(fixture.ManifestHash, pointer.ManifestHash);
        var target = Path.Combine(fixture.DataRoot, pointer.RelativePath.Replace('/', Path.DirectorySeparatorChar));
        Assert.Equal("grounded corpus", await File.ReadAllTextAsync(Path.Combine(target, "index", "corpus.txt")));
        Assert.True(File.Exists(Path.Combine(fixture.DataRoot, "state", "ownership.json")));
    }

    [Fact]
    public async Task Rejects_an_archive_with_a_file_not_listed_in_the_signed_inventory()
    {
        using var fixture = new AssetFixture(includeExtraFile: true);
        var installer = new DataAssetInstaller(new ActiveComponentsStore(fixture.DataRoot));

        await Assert.ThrowsAsync<ComponentActivationException>(() => installer.InstallAsync(fixture.Request("models")));

        Assert.Empty((await new ActiveComponentsStore(fixture.DataRoot).LoadAsync()).Components);
    }

    [Fact]
    public async Task Rejects_an_inventory_total_that_does_not_match_the_signed_install_size()
    {
        using var fixture = new AssetFixture();
        var installer = new DataAssetInstaller(new ActiveComponentsStore(fixture.DataRoot));
        var request = fixture.Request("rag") with { InstallSize = fixture.InstallSize + 1 };

        await Assert.ThrowsAsync<ComponentActivationException>(() => installer.InstallAsync(request));

        Assert.Empty((await new ActiveComponentsStore(fixture.DataRoot).LoadAsync()).Components);
    }

    [Fact]
    public void Data_asset_inventory_rejects_an_overflowing_file_size_total()
    {
        var json = $$"""{"schema":"youtuber.asset.v1","component":"rag","version":"1.0.0","files":[{"path":"first.bin","size":{{long.MaxValue}},"sha256":"{{new string('a', 64)}}"},{"path":"second.bin","size":1,"sha256":"{{new string('b', 64)}}"}]}""";

        Assert.Throws<ComponentActivationException>(() => DataAssetInventory.Parse(json));
    }

    [Fact]
    public async Task Restart_removes_the_single_stale_transaction_stage_before_extracting_again()
    {
        using var fixture = new AssetFixture();
        var staging = Path.Combine(fixture.DataRoot, "state", "asset-staging", "rag.staging");
        Directory.CreateDirectory(staging);
        await File.WriteAllTextAsync(Path.Combine(staging, "partial-model.bin"), "interrupted extraction");
        var installer = new DataAssetInstaller(new ActiveComponentsStore(fixture.DataRoot));

        await installer.InstallAsync(fixture.Request("rag"));

        Assert.False(Directory.Exists(staging));
        Assert.Single((await new ActiveComponentsStore(fixture.DataRoot).LoadAsync()).Components);
        Assert.Empty(Directory.EnumerateDirectories(Path.Combine(fixture.DataRoot, "state", "asset-staging")));
    }

    [Fact]
    public async Task Stale_cleanup_preserves_a_neighbor_with_a_non_guid_legacy_prefix_suffix()
    {
        using var fixture = new AssetFixture();
        var neighbor = Path.Combine(fixture.DataRoot, "state", "asset-staging", "rag.staging-user-backup");
        Directory.CreateDirectory(neighbor);
        var marker = Path.Combine(neighbor, "keep.txt");
        await File.WriteAllTextAsync(marker, "neighbor");

        await new DataAssetInstaller(new ActiveComponentsStore(fixture.DataRoot)).InstallAsync(fixture.Request("rag"));

        Assert.True(File.Exists(marker));
    }

    [Fact]
    public async Task Stale_cleanup_rejects_a_reparse_ancestor_without_deleting_the_external_tree()
    {
        if (!OperatingSystem.IsWindows()) return;
        using var fixture = new AssetFixture();
        var outside = Path.Combine(Path.GetTempPath(), "youtuber-asset-outside-" + Guid.NewGuid().ToString("N"));
        var externalStage = Path.Combine(outside, "rag.staging");
        Directory.CreateDirectory(externalStage);
        var marker = Path.Combine(externalStage, "keep.txt");
        await File.WriteAllTextAsync(marker, "external");
        var state = Directory.CreateDirectory(Path.Combine(fixture.DataRoot, "state")).FullName;
        try
        {
            try { Directory.CreateSymbolicLink(Path.Combine(state, "asset-staging"), outside); }
            catch (UnauthorizedAccessException) { return; }

            await Assert.ThrowsAnyAsync<Exception>(() => new DataAssetInstaller(new ActiveComponentsStore(fixture.DataRoot)).InstallAsync(fixture.Request("rag")));
            Assert.True(File.Exists(marker));
        }
        finally
        {
            try { Directory.Delete(outside, recursive: true); } catch { }
        }
    }

    [Theory]
    [InlineData("runtime")]
    [InlineData("elsewhere")]
    public async Task Rejects_non_data_install_roots(string installRoot)
    {
        using var fixture = new AssetFixture();
        var installer = new DataAssetInstaller(new ActiveComponentsStore(fixture.DataRoot));

        await Assert.ThrowsAsync<ComponentActivationException>(() => installer.InstallAsync(fixture.Request(installRoot)));
    }

    private sealed class AssetFixture : IDisposable
    {
        public AssetFixture(bool includeExtraFile = false)
        {
            DataRoot = Path.Combine(Path.GetTempPath(), "youtuber-asset-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(DataRoot);
            ArchivePath = Path.Combine(DataRoot, "downloads", "rag.zip");
            Directory.CreateDirectory(Path.GetDirectoryName(ArchivePath)!);
            var payload = Encoding.UTF8.GetBytes("grounded corpus");
            var inventory = JsonSerializer.SerializeToUtf8Bytes(new
            {
                schema = "youtuber.asset.v1",
                component = "rag",
                version = "1.0.0",
                files = new[] { new { path = "index/corpus.txt", size = payload.Length, sha256 = Hash(payload) } },
            });
            ManifestHash = Hash(inventory);
            using (var archive = ZipFile.Open(ArchivePath, ZipArchiveMode.Create))
            {
                Write(archive, "rag-archive/asset-manifest.json", inventory);
                Write(archive, "rag-archive/index/corpus.txt", payload);
                if (includeExtraFile) Write(archive, "rag-archive/extra.txt", Encoding.UTF8.GetBytes("not inventoried"));
            }
            ArchiveHash = Hash(File.ReadAllBytes(ArchivePath));
            using var written = ZipFile.OpenRead(ArchivePath);
            ExpandedSize = written.Entries.Where(entry => !entry.FullName.EndsWith("/", StringComparison.Ordinal))
                .Aggregate(0L, (total, entry) => checked(total + entry.Length));
            InstallSize = payload.Length;
        }

        public string DataRoot { get; }
        public string ArchivePath { get; }
        public string ManifestHash { get; }
        public string ArchiveHash { get; }
        public long ExpandedSize { get; }
        public long InstallSize { get; }

        public DataAssetInstallRequest Request(string installRoot) => new("rag", "1.0.0", ArchivePath, "rag-archive", ManifestHash, ArchiveHash, installRoot, ExpandedSize, InstallSize);
        public void Dispose() { if (Directory.Exists(DataRoot)) Directory.Delete(DataRoot, recursive: true); }

        private static void Write(ZipArchive archive, string name, byte[] content)
        {
            using var output = archive.CreateEntry(name).Open();
            output.Write(content);
        }

        private static string Hash(byte[] content) => Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant();
    }
}
