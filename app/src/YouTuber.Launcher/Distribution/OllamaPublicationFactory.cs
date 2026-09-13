using YouTuber.Launcher.Downloads;
using YouTuber.Launcher.Ollama;
using YouTuber.Launcher.Prerequisites;

namespace YouTuber.Launcher.Distribution;

/// <summary>Creates the trusted Ollama installer contract from the signed distribution manifest only.</summary>
public static class OllamaPublicationFactory
{
    public static OllamaPublication Create(DistributionManifest manifest, string fetchedManifestSha256)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        if (fetchedManifestSha256 is null || fetchedManifestSha256.Length != 64 || fetchedManifestSha256.Any(value => value is not (>= '0' and <= '9' or >= 'a' and <= 'f')))
            throw new ManifestValidationException("Fetched manifest identity is invalid.");
        ManifestValidator.Validate(manifest);
        var source = manifest.Ollama!;
        var installer = source.Installer;
        var artifact = new InstallerArtifact(
            HuggingFaceUrl.Create(installer.Repo, installer.Revision, installer.Path).AbsoluteUri,
            installer.Size,
            installer.Sha256,
            installer.Publisher);
        var publication = new OllamaPublication(artifact, installer.Publisher, source.ReleaseVersion, source.ReleaseIdentity, fetchedManifestSha256, source.Models);
        publication.Validate();
        return publication;
    }
}
