using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.Distribution;
using YouTuber.Launcher.Downloads;
using YouTuber.Launcher.Prerequisites;
using YouTuber.Launcher.Setup;
using YouTuber.Launcher.Uninstall;
using Xunit;

namespace YouTuber.Launcher.Tests.Setup;

public sealed class ProductionPrerequisiteProvisionerTests
{
    [Fact]
    public async Task Missing_WebView2_downloads_only_the_manifest_pinned_artifact_then_runs_publisher_verification()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        var download = new RecordingDownload();
        var webView = new RecordingWebView2(available: false);
        using var provisioner = new ProductionPrerequisiteProvisioner(paths, webView, download);

        await provisioner.EnsureAsync(FixtureManifest(), CancellationToken.None);

        var request = Assert.Single(download.Requests);
        Assert.Equal("huggingface.co", request.Source.Host);
        Assert.Equal(new string('9', 64), request.ExpectedSha256);
        Assert.Equal(1, request.ExpectedSize);
        Assert.Equal("Microsoft Corporation", Assert.Single(webView.Installers).ExpectedPublisher);
    }

    [Fact]
    public async Task Installed_WebView2_does_not_download_or_execute_an_installer()
    {
        using var root = new TemporaryDirectory();
        var download = new RecordingDownload();
        var webView = new RecordingWebView2(available: true);
        using var provisioner = new ProductionPrerequisiteProvisioner(AppPaths.ForBaseDirectory(root.Path), webView, download);

        await provisioner.EnsureAsync(FixtureManifest(), CancellationToken.None);

        Assert.Empty(download.Requests);
        Assert.Empty(webView.Installers);
    }

    [Fact]
    public async Task Downloaded_WebView2_installer_is_owned_even_when_installation_fails()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        using var provisioner = new ProductionPrerequisiteProvisioner(paths, new RecordingWebView2(available: false, fail: true), new RecordingDownload());

        await Assert.ThrowsAsync<InvalidOperationException>(() => provisioner.EnsureAsync(FixtureManifest(), CancellationToken.None));

        var installer = Path.Combine(paths.DownloadsRoot, "webview2-installer.exe");
        Assert.Contains(installer, new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData).Files, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Partially_written_WebView2_download_is_already_owned_when_download_throws()
    {
        using var root = new TemporaryDirectory();
        var paths = AppPaths.ForBaseDirectory(root.Path);
        using var provisioner = new ProductionPrerequisiteProvisioner(paths, new RecordingWebView2(available: false), new PartialFailureDownload());

        await Assert.ThrowsAsync<IOException>(() => provisioner.EnsureAsync(FixtureManifest(), CancellationToken.None));

        var installer = Path.Combine(paths.DownloadsRoot, "webview2-installer.exe");
        Assert.Contains(installer, new OwnedDataInventory().Plan(paths.DataRoot, UninstallPreparation.InstallId, UninstallChoice.AppAndOwnedData).Files, StringComparer.OrdinalIgnoreCase);
    }

    private static FetchedDistributionManifest FixtureManifest()
        => new(ManifestValidator.ParseDistribution(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "distribution-valid.json"))), new string('c', 64));

    private sealed class RecordingDownload : ISetupDownloadClient
    {
        public List<DownloadRequest> Requests { get; } = [];
        public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            Requests.Add(request);
            Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
            File.WriteAllBytes(request.TargetPath, new byte[checked((int)request.ExpectedSize)]);
            return Task.CompletedTask;
        }
        public void Dispose() { }
    }

    private sealed class RecordingWebView2(bool available, bool fail = false) : IWebView2Manager
    {
        public List<InstallerArtifact> Installers { get; } = [];
        public bool IsAvailable() => available;
        public Task EnsureAvailableAsync(InstallerArtifact artifact, CancellationToken cancellationToken = default)
        {
            Installers.Add(artifact);
            return fail ? Task.FromException(new InvalidOperationException("installer failed")) : Task.CompletedTask;
        }
    }

    private sealed class PartialFailureDownload : ISetupDownloadClient
    {
        public Task DownloadAsync(DownloadRequest request, CancellationToken cancellationToken)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(request.TargetPath)!);
            File.WriteAllText(request.TargetPath, "partial");
            throw new IOException("network failed after partial write");
        }
        public void Dispose() { }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory() { Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-prereq-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(Path); }
        public string Path { get; }
        public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
    }
}
