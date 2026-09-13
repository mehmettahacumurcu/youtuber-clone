using System.Text.Json;
using System.Text.RegularExpressions;
using YouTuber.Launcher.Downloads;

namespace YouTuber.Launcher.Distribution;

public sealed class ManifestValidationException : Exception
{
    public ManifestValidationException(string message, Exception? innerException = null)
        : base(message, innerException)
    {
    }
}

public static partial class ManifestValidator
{
    public const string DistributionSchema = "youtuber.distribution.v1";
    public const int RuntimeApiVersion = 1;

    private static readonly HashSet<string> ComponentKinds = new(StringComparer.Ordinal)
    {
        "studio_runtime", "voice_runtime", "llm", "voice", "rag", "embedder", "reranker", "ollama_installer", "webview2_runtime",
    };

    public static DistributionManifest ParseDistribution(string json)
    {
        try
        {
            var manifest = JsonSerializer.Deserialize(json, ManifestJsonContext.Default.DistributionManifest)
                ?? throw new ManifestValidationException("Distribution manifest is empty.");
            Validate(manifest);
            return manifest;
        }
        catch (ManifestValidationException)
        {
            throw;
        }
        catch (Exception exception) when (exception is JsonException or OverflowException)
        {
            throw new ManifestValidationException("Distribution manifest is invalid.", exception);
        }
    }

    public static void Validate(DistributionManifest manifest)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        Require(manifest.Schema == DistributionSchema, "Distribution schema is not supported.");
        Require(IsSemVer(manifest.Release), "Release must be a SemVer version.");
        ValidateMinimum(manifest.Minimum);
        var components = manifest.Components ?? throw new ManifestValidationException("Distribution manifest must contain components.");
        Require(components.Count > 0, "Distribution manifest must contain components.");
        ValidateOllama(manifest.Ollama);
        ValidateWebView2(manifest.WebView2);

        var installPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        long requiredBytes = manifest.Minimum.DiskBytes;
        try
        {
            checked
            {
                requiredBytes += manifest.Ollama!.Installer.Size;
                requiredBytes += manifest.Ollama.Installer.InstallSize;
                requiredBytes += manifest.WebView2!.Installer.Size;
                requiredBytes += manifest.WebView2.Installer.InstallSize;
                requiredBytes += manifest.Ollama.Models.Verifier.Source.Size;
                requiredBytes += manifest.Ollama.Models.Verifier.InstalledSize;
                requiredBytes += manifest.Ollama.Models.Speaker.Gguf.Size;
                requiredBytes += manifest.Ollama.Models.Speaker.Modelfile.Size;
                requiredBytes += manifest.Ollama.Models.Speaker.InstalledSize;
                foreach (var (kind, component) in components)
                {
                    Require(ComponentKinds.Contains(kind), $"Unsupported component kind '{kind}'.");
                    ValidateComponent(kind, component, manifest.Minimum.RuntimeApi);
                    Require(installPaths.Add(NormalizeWindowsPath(component.ArchiveRoot)), "Components must not share an install path.");
                    requiredBytes += component.Size;
                    requiredBytes += component.ExpandedSize;
                    requiredBytes += component.InstallSize;
                    requiredBytes += component.PeakSpace;
                }
            }
        }
        catch (OverflowException exception)
        {
            throw new ManifestValidationException("Distribution byte requirements overflow Int64.", exception);
        }

        _ = requiredBytes;
    }

    public static void ValidateBootstrapCatalog(BootstrapCatalog catalog, IEnumerable<string>? allowedProductionManifestHosts = null)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        Require(catalog.Schema == "youtuber.bootstrap-catalog.v1", "Bootstrap catalog schema is not supported.");
        Require(catalog.Environment is "development" or "production", "Bootstrap catalog environment is invalid.");
        Require(IsSha256(catalog.ManifestSha256) && !IsAllZeroes(catalog.ManifestSha256), "Bootstrap manifest SHA-256 is invalid.");
        if (!Uri.TryCreate(catalog.ManifestUrl, UriKind.Absolute, out var manifestUri) || manifestUri.Scheme != Uri.UriSchemeHttps)
        {
            throw new ManifestValidationException("Bootstrap manifest URL must use HTTPS.");
        }

        if (catalog.Environment == "development")
        {
            Require(manifestUri.Host == "127.0.0.1", "Development bootstrap catalogs may only use 127.0.0.1.");
            return;
        }

        var allowedHosts = new HashSet<string>(allowedProductionManifestHosts ?? [], StringComparer.OrdinalIgnoreCase);
        Require(!manifestUri.IsLoopback && allowedHosts.Contains(manifestUri.Host), "Production bootstrap manifest host is not allowed.");
        Require(HuggingFaceUrl.IsCanonicalImmutableResolveUri(manifestUri), "Production bootstrap manifest URL must be a canonical immutable Hugging Face resolve URL.");
    }

    private static void ValidateMinimum(MinimumRequirements? minimum)
    {
        if (minimum is null)
        {
            throw new ManifestValidationException("Minimum requirements are required.");
        }
        Require(!string.IsNullOrWhiteSpace(minimum.Windows), "Minimum Windows version is required.");
        Require(minimum.VramBytes > 0 && minimum.RamBytes > 0 && minimum.DiskBytes > 0, "Minimum byte requirements must be positive.");
        Require(minimum.RuntimeApi == RuntimeApiVersion, "Distribution runtime API is incompatible.");
    }

    private static void ValidateComponent(string kind, DistributionComponent? component, int expectedRuntimeApi)
    {
        if (component is null)
        {
            throw new ManifestValidationException("Component is required.");
        }
        Require(component.Repo is not null && RepoPattern().IsMatch(component.Repo), "Component repository is invalid.");
        Require(component.Revision is not null && RevisionPattern().IsMatch(component.Revision), "Component revision must be an immutable lowercase Git SHA.");
        Require(IsSha256(component.Sha256) && !IsAllZeroes(component.Sha256), "Component SHA-256 is invalid.");
        Require(IsSha256(component.ManifestSha256) && !IsAllZeroes(component.ManifestSha256), "Component manifest SHA-256 is invalid.");
        Require(component.Size > 0 && component.ExpandedSize > 0 && component.InstallSize > 0 && component.PeakSpace > 0, "Component byte requirements must be positive.");
        Require(component.Compatibility is not null && component.Compatibility.RuntimeApi == RuntimeApiVersion && component.Compatibility.RuntimeApi == expectedRuntimeApi, "Component runtime API is incompatible.");
        Require(component.Healthcheck is { Count: > 0 }, "Component health check is required.");
        Require(component.Licenses is { Count: > 0 }, "Component license references are required.");
        Require(component.PackageType is DistributionPackageType.RuntimeArchive or DistributionPackageType.DataArchive, "Component package type is invalid.");

        var isRuntime = component.PackageType == DistributionPackageType.RuntimeArchive;
        Require(isRuntime == component.InstallRoot?.Equals(DistributionInstallRoot.Runtime, StringComparison.Ordinal), "Runtime components must install below the fixed runtime root.");
        if (!isRuntime)
        {
            Require(component.InstallRoot is DistributionInstallRoot.Models or DistributionInstallRoot.Rag, "Data components must install below a fixed data root.");
        }
        var expected = kind switch
        {
            "studio_runtime" or "voice_runtime" => (DistributionPackageType.RuntimeArchive, DistributionInstallRoot.Runtime),
            "rag" => (DistributionPackageType.DataArchive, DistributionInstallRoot.Rag),
            "llm" or "voice" or "embedder" or "reranker" => (DistributionPackageType.DataArchive, DistributionInstallRoot.Models),
            _ => throw new ManifestValidationException($"Component kind '{kind}' must be represented by its dedicated immutable publication record."),
        };
        Require(component.PackageType == expected.Item1 && component.InstallRoot == expected.Item2, $"Component kind '{kind}' has an invalid package classification.");

        ValidateRelativePosixPath(component.Path, "Component archive path");
        ValidateRelativePosixPath(component.ArchiveRoot, "Component archive root");
        ValidateRelativePosixPath(component.Entrypoint, "Component entrypoint");
        foreach (var license in component.Licenses)
        {
            ValidateRelativePosixPath(license, "Component license reference");
        }
    }

    private static void ValidateOllama(OllamaPublicationMetadata? ollama)
    {
        var publication = ollama ?? throw new ManifestValidationException("Distribution manifest must contain an immutable Ollama publication.");
        ValidateInstaller(publication.Installer, "Ollama", "Ollama, Inc.");
        Require(Version.TryParse(publication.ReleaseVersion, out _), "Ollama release version is invalid.");
        Require(!string.IsNullOrWhiteSpace(publication.ReleaseIdentity), "Ollama release identity is required.");
        var models = publication.Models ?? throw new ManifestValidationException("Ollama model roles are required.");
        var verifier = models.Verifier ?? throw new ManifestValidationException("The verifier model role is required.");
        Require(string.Equals(verifier.Name, "qwen3:4b", StringComparison.Ordinal), "The verifier model role must be qwen3:4b.");
        ValidateArtifact(verifier.Source, "Verifier source");
        Require(IsSha256(verifier.InstalledDigest) && !IsAllZeroes(verifier.InstalledDigest), "The verifier installed digest is invalid.");
        Require(verifier.InstalledSize > 0, "The verifier installed size must be positive.");
        var speaker = models.Speaker ?? throw new ManifestValidationException("The Speaker model role is required.");
        Require(ModelNamePattern().IsMatch(speaker.Name), "The Speaker model role is invalid.");
        ValidateArtifact(speaker.Gguf, "Speaker GGUF");
        ValidateArtifact(speaker.Modelfile, "Speaker Modelfile");
        Require(speaker.InstalledSize > 0, "The Speaker installed size must be positive.");
    }

    private static void ValidateWebView2(WebView2PublicationMetadata? webView2)
    {
        var publication = webView2 ?? throw new ManifestValidationException("Distribution manifest must contain an immutable WebView2 publication.");
        ValidateInstaller(publication.Installer, "WebView2", "Microsoft Corporation");
    }

    private static void ValidateInstaller(SignedInstallerMetadata? installer, string name, string publisher)
    {
        if (installer is null) throw new ManifestValidationException($"{name} installer metadata is required.");
        ValidateArtifact(new ImmutableArtifactMetadata(installer.Repo, installer.Revision, installer.Path, installer.Size, installer.Sha256), $"{name} installer");
        Require(string.Equals(installer.Publisher, publisher, StringComparison.Ordinal), $"{name} publisher is invalid.");
        Require(installer.InstallSize > 0, $"{name} install size must be positive.");
    }

    private static void ValidateArtifact(ImmutableArtifactMetadata? artifact, string name)
    {
        if (artifact is null) throw new ManifestValidationException($"{name} metadata is required.");
        Require(artifact.Repo is not null && RepoPattern().IsMatch(artifact.Repo), $"{name} repository is invalid.");
        Require(artifact.Revision is not null && RevisionPattern().IsMatch(artifact.Revision), $"{name} revision must be immutable.");
        ValidateRelativePosixPath(artifact.Path, $"{name} path");
        Require(artifact.Size > 0, $"{name} size must be positive.");
        Require(IsSha256(artifact.Sha256) && !IsAllZeroes(artifact.Sha256), $"{name} SHA-256 is invalid.");
    }

    private static bool IsSemVer(string? value) => value is not null && SemVerPattern().IsMatch(value);

    private static bool IsSha256(string? value) => value is not null && Sha256Pattern().IsMatch(value);

    private static bool IsAllZeroes(string? value) => value is not null && value.All(character => character == '0');

    private static void ValidateRelativePosixPath(string? value, string name)
    {
        try
        {
            _ = NormalizeWindowsPath(value);
        }
        catch (ManifestValidationException exception)
        {
            throw new ManifestValidationException($"{name} is invalid.", exception);
        }
    }

    private static string NormalizeWindowsPath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) || value.StartsWith('/') || value.Contains('\\') || value.Contains(':'))
        {
            throw new ManifestValidationException("Path must be a relative POSIX path.");
        }
        var parts = value.Split('/', StringSplitOptions.None);
        Require(parts.Length > 0, "Path must have a name.");
        var normalized = new string[parts.Length];
        for (var index = 0; index < parts.Length; index++)
        {
            var part = parts[index];
            Require(part is not "" and not "." and not "..", "Path must not traverse parents.");
            normalized[index] = part.TrimEnd(' ', '.');
            Require(normalized[index].Length > 0, "Path contains a Windows-empty name.");
        }

        return string.Join('/', normalized);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new ManifestValidationException(message);
        }
    }

    [GeneratedRegex("^[0-9]+\\.[0-9]+\\.[0-9]+$", RegexOptions.CultureInvariant)]
    private static partial Regex SemVerPattern();

    [GeneratedRegex("^[0-9a-f]{64}$", RegexOptions.CultureInvariant)]
    private static partial Regex Sha256Pattern();

    [GeneratedRegex("^[0-9a-f]{40}$", RegexOptions.CultureInvariant)]
    private static partial Regex RevisionPattern();

    [GeneratedRegex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", RegexOptions.CultureInvariant)]
    private static partial Regex RepoPattern();

    [GeneratedRegex("^[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?$", RegexOptions.CultureInvariant)]
    private static partial Regex ModelNamePattern();
}
