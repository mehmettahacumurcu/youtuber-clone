using System.Net;
using System.Net.Sockets;
using YouTuber.Launcher.Web;
using Xunit;

namespace YouTuber.Launcher.Tests.Web;

public sealed class FixtureManifestBindingTests
{
    [Fact]
    public async Task Fixture_manifest_rejects_non_http_loopback_url()
    {
        await Assert.ThrowsAsync<InvalidOperationException>(() => LauncherSessionPlanFactory.CreateAsync(Arguments("https://127.0.0.1/manifest.json", new string('0', 64))));
    }

    [Fact]
    public async Task Fixture_manifest_rejects_unreachable_loopback_url()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();

        await Assert.ThrowsAnyAsync<Exception>(() => LauncherSessionPlanFactory.CreateAsync(Arguments($"http://127.0.0.1:{port}/manifest.json", new string('0', 64))));
    }

    [Fact]
    public async Task Fixture_manifest_rejects_hash_mismatch_before_worker_creation()
    {
        using var listener = new HttpListener();
        var port = GetUnusedLoopbackPort();
        listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        listener.Start();
        var address = $"http://127.0.0.1:{port}/";
        var serving = Task.Run(async () =>
        {
            var context = await listener.GetContextAsync();
            var body = System.Text.Encoding.UTF8.GetBytes("{\"fixture\":true}");
            context.Response.ContentType = "application/json";
            await context.Response.OutputStream.WriteAsync(body);
            context.Response.Close();
        });

        await Assert.ThrowsAsync<InvalidOperationException>(() => LauncherSessionPlanFactory.CreateAsync(Arguments(address + "manifest.json", new string('0', 64))));
        await serving;
    }

    private static IReadOnlyList<string> Arguments(string manifestUrl, string hash)
        => ["--fixture-workers", $"--data-root={Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"))}", $"--fixture-manifest-url={manifestUrl}", $"--fixture-manifest-sha256={hash}"];

    private static int GetUnusedLoopbackPort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        return ((IPEndPoint)listener.LocalEndpoint).Port;
    }
}
