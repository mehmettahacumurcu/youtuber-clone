using YouTuber.Launcher.Processes;

namespace YouTuber.Launcher.Web;

/// <summary>A bearer bootstrap request whose diagnostic representation is always redacted.</summary>
public sealed class StudioBootstrapRequest
{
    private StudioBootstrapRequest(Uri uri, string authorization)
    {
        Uri = uri;
        Authorization = authorization;
    }

    public Uri Uri { get; }
    public string Method => "POST";
    public string Authorization { get; }

    public static StudioBootstrapRequest Create(StudioOrigin origin, string sessionSecret)
    {
        ArgumentNullException.ThrowIfNull(origin);
        if (SessionSecret.Decode(sessionSecret).Length != SessionSecret.ByteLength) throw new ArgumentException("Invalid session credential.", nameof(sessionSecret));
        return new StudioBootstrapRequest(origin.Bootstrap, "Bearer " + sessionSecret);
    }

    internal string WebViewHeaders => $"Authorization: {Authorization}";
    public override string ToString() => $"{Method} {Uri.AbsoluteUri} [Authorization: Bearer <redacted>]";
}
