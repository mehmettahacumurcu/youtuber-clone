using System.Collections.Frozen;
using System.IO;
using System.Text.Json;

namespace YouTuber.Launcher.Distribution;

public static class BootstrapCatalogLoader
{
    private const string ResourceName = "YouTuber.Launcher.Resources.bootstrap-catalog.json";
    private static readonly FrozenSet<string> ProductionManifestHosts = new[] { "huggingface.co" }
        .ToFrozenSet(StringComparer.OrdinalIgnoreCase);

    public static BootstrapCatalog LoadEmbedded()
    {
        var assembly = typeof(BootstrapCatalogLoader).Assembly;
        using var stream = assembly.GetManifestResourceStream(ResourceName)
            ?? throw new ManifestValidationException("Embedded bootstrap catalog is missing.");
        return Load(stream);
    }

    public static BootstrapCatalog Load(Stream stream, IEnumerable<string>? allowedProductionManifestHosts = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        try
        {
            var catalog = JsonSerializer.Deserialize(stream, ManifestJsonContext.Default.BootstrapCatalog)
                ?? throw new ManifestValidationException("Bootstrap catalog is empty.");
            ManifestValidator.ValidateBootstrapCatalog(catalog, allowedProductionManifestHosts ?? ProductionManifestHosts);
            return catalog;
        }
        catch (ManifestValidationException)
        {
            throw;
        }
        catch (JsonException exception)
        {
            throw new ManifestValidationException("Bootstrap catalog is invalid.", exception);
        }
    }

    public static BootstrapCatalog Parse(string json, IEnumerable<string>? allowedProductionManifestHosts = null)
    {
        try
        {
            var catalog = JsonSerializer.Deserialize(json, ManifestJsonContext.Default.BootstrapCatalog)
                ?? throw new ManifestValidationException("Bootstrap catalog is empty.");
            ManifestValidator.ValidateBootstrapCatalog(catalog, allowedProductionManifestHosts ?? ProductionManifestHosts);
            return catalog;
        }
        catch (ManifestValidationException)
        {
            throw;
        }
        catch (JsonException exception)
        {
            throw new ManifestValidationException("Bootstrap catalog is invalid.", exception);
        }
    }
}
