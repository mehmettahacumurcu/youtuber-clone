using System.Net.Http;
using System.Security.Cryptography;
using System.Text;

namespace YouTuber.Launcher.Distribution;

public sealed record FetchedDistributionManifest(DistributionManifest Manifest, string Sha256);

/// <summary>Fetches exactly the immutable manifest selected by the embedded catalog.</summary>
public sealed class DistributionManifestFetcher : IDisposable
{
    private const int MaximumManifestBytes = 4 * 1024 * 1024;
    private readonly HttpClient _http;

    public DistributionManifestFetcher(HttpMessageHandler? handler = null)
    {
        _http = new HttpClient(handler ?? new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false }, disposeHandler: true)
        {
            Timeout = TimeSpan.FromSeconds(30),
        };
    }

    public async Task<FetchedDistributionManifest> FetchAsync(BootstrapCatalog catalog, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ManifestValidator.ValidateBootstrapCatalog(catalog, ["huggingface.co"]);
        var uri = new Uri(catalog.ManifestUrl, UriKind.Absolute);
        using var response = await _http.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        if (!response.IsSuccessStatusCode || response.RequestMessage?.RequestUri != uri || response.Content.Headers.ContentLength is > MaximumManifestBytes)
            throw new ManifestValidationException("Distribution manifest request was rejected.");
        await response.Content.LoadIntoBufferAsync(MaximumManifestBytes).WaitAsync(cancellationToken);
        var bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken);
        if (bytes.Length > MaximumManifestBytes) throw new ManifestValidationException("Distribution manifest exceeds the maximum size.");
        var actual = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        if (!CryptographicOperations.FixedTimeEquals(Encoding.ASCII.GetBytes(actual), Encoding.ASCII.GetBytes(catalog.ManifestSha256)))
            throw new ManifestValidationException("Distribution manifest hash does not match the embedded catalog.");
        return new FetchedDistributionManifest(ManifestValidator.ParseDistribution(Encoding.UTF8.GetString(bytes)), actual);
    }

    public void Dispose() => _http.Dispose();
}
