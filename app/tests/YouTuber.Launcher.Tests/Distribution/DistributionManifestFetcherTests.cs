using System.Net;
using System.Text;
using YouTuber.Launcher.Distribution;
using Xunit;

namespace YouTuber.Launcher.Tests.Distribution;

public sealed class DistributionManifestFetcherTests
{
    [Fact]
    public async Task Rejects_a_manifest_that_is_not_bound_to_the_embedded_catalog_hash()
    {
        var catalog = BootstrapCatalogLoader.Parse("""{"schema":"youtuber.bootstrap-catalog.v1","environment":"production","manifest_url":"https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/distribution.json","manifest_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}""");
        using var fetcher = new DistributionManifestFetcher(new StaticHandler("{}"));

        await Assert.ThrowsAsync<ManifestValidationException>(() => fetcher.FetchAsync(catalog));
    }

    private sealed class StaticHandler(string body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                RequestMessage = request,
                Content = new StringContent(body, Encoding.UTF8, "application/json"),
            });
    }
}
