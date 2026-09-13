using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using YouTuber.Launcher.Processes;

namespace YouTuber.Launcher.Web;

public sealed class BrowserBootstrapClient : IDisposable
{
    private const int MaximumResponseBytes = 4096;
    private readonly HttpClient _http;
    private readonly StudioOrigin _origin;

    public BrowserBootstrapClient(StudioOrigin origin)
        : this(new SocketsHttpHandler { AllowAutoRedirect = false, UseCookies = false, UseProxy = false }, origin)
    {
    }

    public BrowserBootstrapClient(HttpMessageHandler handler, StudioOrigin origin)
    {
        ArgumentNullException.ThrowIfNull(handler);
        _origin = origin ?? throw new ArgumentNullException(nameof(origin));
        _http = new HttpClient(handler, disposeHandler: true) { Timeout = TimeSpan.FromSeconds(5) };
    }

    public async Task<Uri> AcquireUrlAsync(string sessionSecret, CancellationToken cancellationToken = default)
    {
        if (SessionSecret.Decode(sessionSecret).Length != SessionSecret.ByteLength) throw new ArgumentException("Invalid session credential.", nameof(sessionSecret));
        using var request = new HttpRequestMessage(HttpMethod.Post, _origin.BrowserBootstrap) { Content = new ByteArrayContent([]) };
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", sessionSecret);
        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        if (response.StatusCode != HttpStatusCode.OK || (response.RequestMessage?.RequestUri is Uri finalUri && finalUri != _origin.BrowserBootstrap))
        {
            throw new InvalidOperationException("Browser bootstrap did not return a direct trusted response.");
        }
        if (response.Content.Headers.ContentType?.MediaType is not "application/json") throw new InvalidOperationException("Browser bootstrap returned an unexpected content type.");
        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var bounded = new MemoryStream();
        var buffer = new byte[1024];
        while (true)
        {
            var read = await stream.ReadAsync(buffer, cancellationToken);
            if (read == 0) break;
            if (bounded.Length + read > MaximumResponseBytes) throw new InvalidOperationException("Browser bootstrap response was too large.");
            bounded.Write(buffer, 0, read);
        }
        bounded.Position = 0;
        using var document = await JsonDocument.ParseAsync(bounded, cancellationToken: cancellationToken);
        var value = document.RootElement.TryGetProperty("url", out var element) ? element.GetString() : null;
        return ValidateExchangeUrl(value);
    }

    private Uri ValidateExchangeUrl(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) || !_origin.AllowsNavigation(value) || !Uri.TryCreate(value, UriKind.Absolute, out var uri))
            throw new InvalidOperationException("Browser bootstrap returned an untrusted URL.");
        if (uri.AbsolutePath != "/v1/browser-exchange" || !string.IsNullOrEmpty(uri.Fragment) || !uri.Query.StartsWith("?nonce=", StringComparison.Ordinal) || uri.Query.Count(character => character == '&') != 0)
            throw new InvalidOperationException("Browser bootstrap returned an invalid exchange URL.");
        var nonce = uri.Query[7..];
        if (nonce.Length != 22 || nonce.Any(character => !char.IsAsciiLetterOrDigit(character) && character is not '-' and not '_'))
            throw new InvalidOperationException("Browser bootstrap nonce is not 128-bit base64url.");
        try
        {
            if (Convert.FromBase64String(nonce.Replace('-', '+').Replace('_', '/') + "==").Length != 16) throw new FormatException();
        }
        catch (FormatException exception) { throw new InvalidOperationException("Browser bootstrap nonce is not 128-bit base64url.", exception); }
        return uri;
    }

    public void Dispose() => _http.Dispose();
}

public interface IExternalBrowserLauncher { void Open(Uri uri); }

public sealed class ShellExecuteBrowserLauncher : IExternalBrowserLauncher
{
    public void Open(Uri uri)
    {
        ArgumentNullException.ThrowIfNull(uri);
        Process.Start(new ProcessStartInfo(uri.AbsoluteUri) { UseShellExecute = true });
    }
}
