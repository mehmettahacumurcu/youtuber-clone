using Microsoft.Web.WebView2.Core;
using System.IO;

namespace YouTuber.Launcher.Prerequisites;

public interface IWebView2Manager
{
    bool IsAvailable();
    Task EnsureAvailableAsync(InstallerArtifact artifact, CancellationToken cancellationToken = default);
}

public sealed class WebView2Manager : IWebView2Manager
{
    public const string ExpectedPublisher = "Microsoft Corporation";
    private readonly Func<bool> _isAvailable;
    private readonly AuthenticodeVerifier _verifier;
    private readonly InstallerRunner _runner;

    public WebView2Manager(Func<bool>? isAvailable = null, AuthenticodeVerifier? verifier = null, InstallerRunner? runner = null)
    {
        _isAvailable = isAvailable ?? ProbeAvailability;
        _verifier = verifier ?? new AuthenticodeVerifier();
        _runner = runner ?? new InstallerRunner();
    }

    public bool IsAvailable() => _isAvailable();

    public async Task EnsureAvailableAsync(InstallerArtifact artifact, CancellationToken cancellationToken = default)
    {
        if (IsAvailable()) return;
        if (!string.Equals(artifact.ExpectedPublisher, ExpectedPublisher, StringComparison.OrdinalIgnoreCase))
        {
            throw new InstallerVerificationException("The WebView2 installer must be signed by Microsoft Corporation.");
        }

        var verified = await _verifier.VerifyAsync(artifact, cancellationToken);
        var result = await _runner.RunAsync(verified, ["/silent", "/install"], TimeSpan.FromMinutes(5), cancellationToken: cancellationToken);
        if (result.TimedOut || result.ExitCode != 0 || !IsAvailable())
        {
            throw new InvalidOperationException("Microsoft Edge WebView2 Runtime installation failed.");
        }
    }

    private static bool ProbeAvailability()
    {
        try { return !string.IsNullOrWhiteSpace(CoreWebView2Environment.GetAvailableBrowserVersionString()); }
        catch (Exception exception) when (exception is InvalidOperationException or FileNotFoundException) { return false; }
    }
}
