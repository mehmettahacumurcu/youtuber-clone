using System.Text.Json.Serialization;

namespace YouTuber.Launcher.Distribution;

public sealed record DistributionManifest(
    string Schema,
    string Release,
    MinimumRequirements Minimum,
    IReadOnlyDictionary<string, DistributionComponent> Components)
{
    public OllamaPublicationMetadata? Ollama { get; init; }
    [JsonPropertyName("webview2")]
    public WebView2PublicationMetadata? WebView2 { get; init; }
}

public sealed record OllamaPublicationMetadata(
    SignedInstallerMetadata Installer,
    string ReleaseVersion,
    string ReleaseIdentity,
    OllamaModelRolesMetadata Models);

public sealed record WebView2PublicationMetadata(SignedInstallerMetadata Installer);

public sealed record SignedInstallerMetadata(
    string Repo,
    string Revision,
    string Path,
    long Size,
    string Sha256,
    string Publisher,
    long InstallSize);

public sealed record ImmutableArtifactMetadata(
    string Repo,
    string Revision,
    string Path,
    long Size,
    string Sha256);

public sealed record OllamaVerifierMetadata(
    string Name,
    ImmutableArtifactMetadata Source,
    string InstalledDigest,
    long InstalledSize);

public sealed record OllamaModelRolesMetadata(OllamaVerifierMetadata Verifier, SpeakerModelMetadata Speaker);

public sealed record SpeakerModelMetadata(
    string Name,
    ImmutableArtifactMetadata Gguf,
    ImmutableArtifactMetadata Modelfile,
    long InstalledSize);

public static class DistributionPackageType
{
    public const string RuntimeArchive = "runtime_archive";
    public const string DataArchive = "data_archive";
}

public static class DistributionInstallRoot
{
    public const string Runtime = "runtime";
    public const string Models = "models";
    public const string Rag = "rag";
}

public sealed record MinimumRequirements(
    string Windows,
    long VramBytes,
    long RamBytes,
    long DiskBytes,
    int RuntimeApi);

public sealed record DistributionComponent(
    string Repo,
    string Revision,
    string Path,
    long Size,
    string Sha256,
    string ArchiveRoot,
    long ExpandedSize,
    long InstallSize,
    long PeakSpace,
    string Entrypoint,
    IReadOnlyList<string> Healthcheck,
    Compatibility Compatibility,
    IReadOnlyList<string> Licenses)
{
    public string? ManifestSha256 { get; init; }
    public string? PackageType { get; init; }
    public string? InstallRoot { get; init; }
}

public sealed record Compatibility(int RuntimeApi);

public sealed record BootstrapCatalog(
    string Schema,
    string Environment,
    string ManifestUrl,
    string ManifestSha256);

[JsonSourceGenerationOptions(
    GenerationMode = JsonSourceGenerationMode.Metadata,
    PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower,
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
[JsonSerializable(typeof(DistributionManifest))]
[JsonSerializable(typeof(BootstrapCatalog))]
internal partial class ManifestJsonContext : JsonSerializerContext
{
}
