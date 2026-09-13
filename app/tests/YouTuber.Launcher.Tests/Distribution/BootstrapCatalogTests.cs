using YouTuber.Launcher.Distribution;
using System.Diagnostics;
using System.Text;
using Xunit;

namespace YouTuber.Launcher.Tests.Distribution;

public sealed class BootstrapCatalogTests
{
    [Fact]
    public void EmbeddedCatalogIsExplicitlyDevelopmentOnlyAndUsesTheFixtureEndpoint()
    {
        var catalog = BootstrapCatalogLoader.LoadEmbedded();

        Assert.Equal("development", catalog.Environment);
        Assert.Equal("https://127.0.0.1/manifest.json", catalog.ManifestUrl);
        Assert.Equal("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", catalog.ManifestSha256);
    }

    [Fact]
    public void ProductionCatalogRejectsAnUnknownManifestHost()
    {
        const string json = """
            {"schema":"youtuber.bootstrap-catalog.v1","environment":"production","manifest_url":"https://untrusted.example/manifest.json","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
            """;

        Assert.Throws<ManifestValidationException>(() => BootstrapCatalogLoader.Parse(json));
    }

    [Fact]
    public void ProductionCatalogCannotRebrandACustomAllowlistedHostAsCanonicalHuggingFace()
    {
        const string json = """
            {"schema":"youtuber.bootstrap-catalog.v1","environment":"production","manifest_url":"https://evil.example/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
            """;

        Assert.Throws<ManifestValidationException>(() => BootstrapCatalogLoader.Parse(json, ["evil.example"]));
    }

    [Fact]
    public void ProductionCatalogAcceptsTheBuiltInHuggingFaceHost()
    {
        const string json = """
            {"schema":"youtuber.bootstrap-catalog.v1","environment":"production","manifest_url":"https://huggingface.co/your-account/youtuber/resolve/1111111111111111111111111111111111111111/manifest.json","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
            """;

        var catalog = BootstrapCatalogLoader.Parse(json);

        Assert.Equal("production", catalog.Environment);
    }

    [Fact]
    public void ProductionCatalogStreamUsesTheBuiltInHuggingFaceHost()
    {
        const string json = """
            {"schema":"youtuber.bootstrap-catalog.v1","environment":"production","manifest_url":"https://huggingface.co/your-account/youtuber/resolve/1111111111111111111111111111111111111111/manifest.json","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
            """;
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(json));

        var catalog = BootstrapCatalogLoader.Load(stream);

        Assert.Equal("production", catalog.Environment);
    }

    [Theory]
    [InlineData("https://huggingface.co/owner/repo/resolve/main/manifest.json")]
    [InlineData("https://huggingface.co/owner/repo/blob/1111111111111111111111111111111111111111/manifest.json")]
    [InlineData("https://huggingface.co/owner/repo/resolve/111111111111111111111111111111111111111/manifest.json")]
    [InlineData("https://user:secret@huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json")]
    [InlineData("https://huggingface.co:444/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json")]
    [InlineData("https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json?download=true")]
    [InlineData("https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json#fragment")]
    [InlineData("https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/folder%2Fmanifest.json")]
    [InlineData("https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/folder/../manifest.json")]
    public void ProductionCatalogRejectsNonCanonicalOrMutableHuggingFaceUrls(string manifestUrl)
    {
        var json = $$"""
            {"schema":"youtuber.bootstrap-catalog.v1","environment":"production","manifest_url":"{{manifestUrl}}","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
            """;

        Assert.Throws<ManifestValidationException>(() => BootstrapCatalogLoader.Parse(json));
    }

    [Fact]
    public async Task PublishRejectsDevelopmentCatalogWithEscapedEnvironmentProperty()
    {
        var root = FindRepositoryRoot();
        var project = Path.Combine(root, "app", "src", "YouTuber.Launcher", "YouTuber.Launcher.csproj");
        var catalog = Path.Combine(root, "tests", "fixtures", "distribution", "bootstrap-catalog-development-escaped-environment.json");
        using var process = Process.Start(new ProcessStartInfo("dotnet")
        {
            WorkingDirectory = root,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            ArgumentList = { "publish", project, "-c", "Release", "--no-restore", $"-p:BootstrapCatalogPath={catalog}" },
        }) ?? throw new InvalidOperationException("Could not start dotnet publish.");

        var output = await process.StandardOutput.ReadToEndAsync();
        var error = await process.StandardError.ReadToEndAsync();
        await process.WaitForExitAsync();

        Assert.NotEqual(0, process.ExitCode);
        Assert.Contains("development bootstrap catalog", output + error, StringComparison.OrdinalIgnoreCase);
    }

    private static string FindRepositoryRoot()
    {
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "app", "YouTuberStudio.sln")))
            {
                return directory.FullName;
            }
        }

        throw new InvalidOperationException("Repository root was not found.");
    }
}
