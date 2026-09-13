using YouTuber.Launcher.Configuration;
using Xunit;

namespace YouTuber.Launcher.Tests.Configuration;

public sealed class SettingsStoreTests : IDisposable
{
    private readonly string _testRoot = Path.Combine(Path.GetTempPath(), "YouTuberStudioTests", Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task SaveThenLoadRoundTripsSettingsAndLeavesNoTemporaryFile()
    {
        var paths = AppPaths.ForBaseDirectory(_testRoot);
        var settings = new LauncherSettings(1, paths.DataRoot, "stable", true, true, DateTimeOffset.Parse("2026-07-31T12:00:00+00:00"));
        var store = new SettingsStore(paths);

        await store.SaveAsync(settings);
        var loaded = await store.LoadAsync();

        Assert.Equal(settings, loaded);
        Assert.Empty(Directory.EnumerateFiles(paths.StateRoot, "*.tmp"));
    }

    [Fact]
    public async Task LoadRejectsUnknownJsonProperties()
    {
        var paths = AppPaths.ForBaseDirectory(_testRoot);
        Directory.CreateDirectory(paths.StateRoot);
        await File.WriteAllTextAsync(
            paths.SettingsFile,
            "{\"schema\":1,\"dataRoot\":\"C:\\\\Data\",\"releaseChannel\":\"stable\",\"licenseAccepted\":true,\"disclosureAccepted\":true,\"lastUpdateCheckUtc\":null,\"unexpected\":true}");
        var store = new SettingsStore(paths);

        await Assert.ThrowsAsync<System.Text.Json.JsonException>(() => store.LoadAsync());
    }

    [Fact]
    public void DownloadsRequireCurrentSchemaStableChannelAndAcceptances()
    {
        var paths = AppPaths.ForBaseDirectory(_testRoot);
        var ready = new LauncherSettings(1, paths.DataRoot, "stable", true, true, null);
        var missingDisclosure = ready with { DisclosureAccepted = false };
        var previewChannel = ready with { ReleaseChannel = "preview" };
        var olderSchema = ready with { Schema = 0 };

        Assert.True(ready.CanDownload);
        Assert.False(missingDisclosure.CanDownload);
        Assert.False(previewChannel.CanDownload);
        Assert.False(olderSchema.CanDownload);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testRoot))
        {
            Directory.Delete(_testRoot, recursive: true);
        }
    }
}
