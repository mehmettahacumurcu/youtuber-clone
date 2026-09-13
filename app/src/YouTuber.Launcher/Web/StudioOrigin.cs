namespace YouTuber.Launcher.Web;

public sealed class StudioOrigin
{
    public StudioOrigin(int port)
    {
        if (port is < 1 or > 65535) throw new ArgumentOutOfRangeException(nameof(port));
        Port = port;
        Origin = new Uri($"http://127.0.0.1:{port}/");
    }
    public int Port { get; }
    public Uri Origin { get; }
    public bool AllowsNavigation(string? value)
    {
        if (string.Equals(value, "about:blank", StringComparison.Ordinal)) return true;
        if (!Uri.TryCreate(value, UriKind.Absolute, out var uri)) return false;
        var expectedAuthority = "http://127.0.0.1:" + Port;
        if (!value.StartsWith(expectedAuthority + "/", StringComparison.Ordinal)) return false;
        return uri.Scheme == Uri.UriSchemeHttp && uri.Host == "127.0.0.1" && uri.Port == Port && string.IsNullOrEmpty(uri.UserInfo) && string.IsNullOrEmpty(uri.Fragment);
    }
    public bool IsLandingPage(string? value) => string.Equals(value, Origin.AbsoluteUri, StringComparison.Ordinal);
    public Uri Bootstrap => new(Origin, "bootstrap");
    public Uri BrowserBootstrap => new(Origin, "v1/browser-bootstrap");
}
