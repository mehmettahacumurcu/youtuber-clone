using System.Net;
using System.Net.Sockets;
using YouTuber.Launcher.Processes;
using Xunit;

namespace YouTuber.Launcher.Tests.Processes;

public sealed class SessionSecretTests
{
    [Fact]
    public void Create_generates_unique_32_byte_base64url_values()
    {
        var secrets = Enumerable.Range(0, 1_000).Select(_ => SessionSecret.Create()).ToArray();

        Assert.Equal(1_000, secrets.Distinct(StringComparer.Ordinal).Count());
        Assert.All(secrets, secret =>
        {
            Assert.Equal(43, secret.Length);
            Assert.Matches("^[A-Za-z0-9_-]{43}$", secret);
            Assert.Equal(32, SessionSecret.Decode(secret).Length);
        });
    }

    [Fact]
    public void Reserve_does_not_return_an_already_listening_port()
    {
        using var held = new TcpListener(IPAddress.Loopback, 0);
        held.Start();
        var heldPort = ((IPEndPoint)held.LocalEndpoint).Port;

        using var allocation = new PortAllocator().Reserve(3);

        Assert.Equal(3, allocation.Ports.Count);
        Assert.Equal(3, allocation.Ports.Distinct().Count());
        Assert.DoesNotContain(heldPort, allocation.Ports);
    }
}
