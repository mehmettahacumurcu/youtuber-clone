using YouTuber.Launcher.Distribution;
using Xunit;

namespace YouTuber.Launcher.Tests.Distribution;

public sealed class ManifestValidatorTests
{
    [Fact]
    public void Distribution_manifest_requires_an_immutable_ollama_publication()
    {
        var manifest = new DistributionManifest(
            "youtuber.distribution.v1", "1.0.0", new MinimumRequirements("10.0.19045", 1, 1, 1, 1),
            new Dictionary<string, DistributionComponent>
            {
                ["rag"] = new("owner/repo", "1111111111111111111111111111111111111111", "artifact.zip", 1, new string('a', 64), "rag", 1, 1, 1, "worker.exe", ["--healthcheck"], new Compatibility(1), ["NOTICE.txt"]),
            });

        Assert.Throws<ManifestValidationException>(() => ManifestValidator.Validate(manifest));
    }

    [Fact]
    public void Ollama_publication_requires_a_valid_fetched_manifest_identity()
    {
        var manifest = ManifestValidator.ParseDistribution(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "distribution-valid.json")));

        Assert.Throws<ManifestValidationException>(() => OllamaPublicationFactory.Create(manifest, new string('A', 64)));
    }

    [Fact]
    public void Distribution_manifest_carries_immutable_prerequisites_and_model_roles()
    {
        var manifest = ManifestValidator.ParseDistribution(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "distribution-valid.json")));

        Assert.Equal("Microsoft Corporation", manifest.WebView2!.Installer.Publisher);
        Assert.Equal("qwen3:4b", manifest.Ollama!.Models.Verifier.Name);
        Assert.Equal(3, manifest.Ollama.Models.Verifier.Source.Size);
        Assert.Equal(4, manifest.Ollama.Models.Verifier.InstalledSize);
        Assert.Equal(5, manifest.Ollama.Models.Speaker.InstalledSize);
        Assert.Equal(2, manifest.Ollama.Installer.InstallSize);
        Assert.Equal(6, manifest.WebView2.Installer.InstallSize);
        Assert.Equal(2, manifest.Components["voice_runtime"].ExpandedSize);
        Assert.Equal("speaker-v5-a636", manifest.Ollama.Models.Speaker.Name);
        Assert.Equal(DistributionPackageType.RuntimeArchive, manifest.Components["voice_runtime"].PackageType);
        Assert.Equal(DistributionInstallRoot.Runtime, manifest.Components["voice_runtime"].InstallRoot);
    }

    [Fact]
    public void Model_roles_are_bound_by_the_fetched_manifest_hash_without_a_self_referential_json_field()
    {
        var manifest = ManifestValidator.ParseDistribution(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "distribution-valid.json")));
        var fetchedHash = new string('c', 64);

        var publication = OllamaPublicationFactory.Create(manifest, fetchedHash);

        Assert.Equal(fetchedHash, publication.ManifestIdentity);
        Assert.Equal("qwen3:4b", publication.Models.Verifier.Name);
        Assert.Equal("speaker-v5-a636", publication.Models.Speaker.Name);
    }

    [Fact]
    public void Runtime_component_kinds_cannot_be_reclassified_as_data_assets()
    {
        var json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", "distribution-valid.json"))
            .Replace("\"package_type\": \"runtime_archive\"", "\"package_type\": \"data_archive\"", StringComparison.Ordinal)
            .Replace("\"install_root\": \"runtime\"", "\"install_root\": \"models\"", StringComparison.Ordinal);

        Assert.Throws<ManifestValidationException>(() => ManifestValidator.ParseDistribution(json));
    }
    public static TheoryData<string, bool> ContractFixtures => new()
    {
        { "distribution-valid.json", true },
        { "distribution-invalid-main-revision.json", false },
        { "distribution-invalid-uppercase-hash.json", false },
        { "distribution-invalid-zero-size.json", false },
        { "distribution-invalid-traversal.json", false },
        { "distribution-invalid-runtime-api-2.json", false },
        { "distribution-invalid-duplicate-install-path.json", false },
        { "distribution-invalid-zero-hash.json", false },
        { "distribution-invalid-unknown-property.json", false },
        { "distribution-invalid-missing-repo.json", false },
        { "distribution-invalid-string-size.json", false },
        { "distribution-invalid-boolean-runtime-api.json", false },
        { "distribution-invalid-empty-path-segment.json", false },
        { "distribution-invalid-dot-path-segment.json", false },
        { "distribution-invalid-whitespace-path.json", false },
        { "distribution-invalid-model-role.json", false },
    };

    [Theory]
    [MemberData(nameof(ContractFixtures))]
    public void DistributionFixturesHaveTheSameAcceptanceContract(string filename, bool accepted)
    {
        var json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "fixtures", "distribution", filename));

        if (accepted)
        {
            var manifest = ManifestValidator.ParseDistribution(json);
            Assert.Equal("1.0.0", manifest.Release);
            Assert.Equal("1111111111111111111111111111111111111111", manifest.Components["voice_runtime"].Revision);
        }
        else
        {
            Assert.Throws<ManifestValidationException>(() => ManifestValidator.ParseDistribution(json));
        }
    }
}
