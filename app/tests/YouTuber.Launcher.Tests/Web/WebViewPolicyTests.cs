using System.Net;
using System.Net.Http.Headers;
using System.Text;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.ViewModels;
using YouTuber.Launcher.Web;
using Xunit;

namespace YouTuber.Launcher.Tests.Web;

public sealed class WebViewPolicyTests
{
    [Theory]
    [InlineData("http://127.0.0.1:32100/")]
    [InlineData("http://127.0.0.1:32100/path?x=1")]
    [InlineData("about:blank")]
    public void Allows_only_selected_origin_and_blank(string value) => Assert.True(new StudioOrigin(32100).AllowsNavigation(value));

    [Theory]
    [InlineData("http://localhost:32100/")]
    [InlineData("http://127.0.0.1:32101/")]
    [InlineData("https://127.0.0.1:32100/")]
    [InlineData("http://[::1]:32100/")]
    [InlineData("http://127.0.0.1@evil.example:32100/")]
    [InlineData("http://127.0.0.1:32100/#fragment")]
    [InlineData("file:///C:/x")]
    [InlineData("data:text/html,x")]
    [InlineData("http://2130706433:32100/")]
    public void Rejects_every_alternate_origin_encoding(string value) => Assert.False(new StudioOrigin(32100).AllowsNavigation(value));

    [Theory]
    [InlineData("http://127.0.0.1:32100/bootstrap")]
    [InlineData("http://127.0.0.1:32100/?error=1")]
    [InlineData("about:blank")]
    public void Bootstrap_completion_accepts_only_the_exact_redirect_landing_page(string value)
        => Assert.False(new StudioOrigin(32100).IsLandingPage(value));

    [Fact]
    public void Bootstrap_completion_accepts_the_exact_redirect_landing_page()
        => Assert.True(new StudioOrigin(32100).IsLandingPage("http://127.0.0.1:32100/"));

    [Fact]
    public void Policy_disables_every_privileged_or_persistent_webview_capability()
    {
        var restrictions = new WebViewPolicy(new StudioOrigin(32100)).Restrictions;

        Assert.False(restrictions.ContextMenus);
        Assert.False(restrictions.DevTools);
        Assert.False(restrictions.StatusBar);
        Assert.False(restrictions.PasswordAutosave);
        Assert.False(restrictions.GeneralAutofill);
        Assert.False(restrictions.HostObjects);
        Assert.False(restrictions.WebMessages);
        Assert.False(restrictions.ScriptDialogs);
        Assert.False(restrictions.ExternalDrop);
    }

    [Fact]
    public void Bootstrap_request_keeps_the_secret_only_in_the_authorization_header()
    {
        var secret = SessionSecret.Create();
        var request = StudioBootstrapRequest.Create(new StudioOrigin(32100), secret);

        Assert.Equal("POST", request.Method);
        Assert.Equal("http://127.0.0.1:32100/bootstrap", request.Uri.AbsoluteUri);
        Assert.DoesNotContain(secret, request.Uri.AbsoluteUri, StringComparison.Ordinal);
        Assert.DoesNotContain(secret, request.ToString(), StringComparison.Ordinal);
        Assert.Equal("Bearer " + secret, request.Authorization);
    }

    [Fact]
    public async Task Browser_bootstrap_posts_bearer_and_accepts_only_a_valid_single_use_nonce_url()
    {
        var origin = new StudioOrigin(32100);
        var secret = SessionSecret.Create();
        var handler = new RecordingHandler(_ => Json(HttpStatusCode.OK, "{\"url\":\"http://127.0.0.1:32100/v1/browser-exchange?nonce=AAAAAAAAAAAAAAAAAAAAAA\"}"));
        using var client = new BrowserBootstrapClient(handler, origin);

        var result = await client.AcquireUrlAsync(secret);

        Assert.Equal("http://127.0.0.1:32100/v1/browser-exchange?nonce=AAAAAAAAAAAAAAAAAAAAAA", result.AbsoluteUri);
        Assert.Equal(HttpMethod.Post, handler.Request!.Method);
        Assert.Equal(origin.BrowserBootstrap, handler.Request.RequestUri);
        Assert.Equal(new AuthenticationHeaderValue("Bearer", secret), handler.Request.Headers.Authorization);
        Assert.DoesNotContain(secret, result.AbsoluteUri, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(HttpStatusCode.Redirect, "{\"url\":\"http://127.0.0.1:32100/v1/browser-exchange?nonce=AAAAAAAAAAAAAAAAAAAAAA\"}")]
    [InlineData(HttpStatusCode.OK, "{\"url\":\"https://evil.example/v1/browser-exchange?nonce=AAAAAAAAAAAAAAAAAAAAAA\"}")]
    [InlineData(HttpStatusCode.OK, "{\"url\":\"http://127.0.0.1:32100/wrong?nonce=AAAAAAAAAAAAAAAAAAAAAA\"}")]
    [InlineData(HttpStatusCode.OK, "{\"url\":\"http://127.0.0.1:32100/v1/browser-exchange?nonce=short\"}")]
    public async Task Browser_bootstrap_rejects_redirects_and_untrusted_exchange_urls(HttpStatusCode status, string body)
    {
        using var client = new BrowserBootstrapClient(new RecordingHandler(_ => Json(status, body)), new StudioOrigin(32100));

        await Assert.ThrowsAsync<InvalidOperationException>(() => client.AcquireUrlAsync(SessionSecret.Create()));
    }

    [Fact]
    public void Shell_hides_web_content_during_worker_recovery_and_requires_a_new_bootstrap()
    {
        var viewModel = new ShellViewModel();
        viewModel.MarkBootstrapComplete();
        Assert.True(viewModel.IsStudioVisible);

        viewModel.MarkRecoveryRequired("Studio stopped unexpectedly.");

        Assert.False(viewModel.IsStudioVisible);
        Assert.True(viewModel.NeedsBootstrap);
        Assert.Contains("stopped", viewModel.StatusMessage, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Shell_releases_the_browser_action_when_disposed()
    {
        var viewModel = new ShellViewModel();
        viewModel.ConfigureBrowserAction(_ => Task.CompletedTask);
        viewModel.MarkBootstrapComplete();
        Assert.True(viewModel.OpenInBrowserCommand.CanExecute(null));

        viewModel.ClearBrowserAction();

        Assert.False(viewModel.OpenInBrowserCommand.CanExecute(null));
    }

    [Fact]
    public async Task Browser_troubleshooting_failure_is_reported_without_escaping_the_ui_command()
    {
        var viewModel = new ShellViewModel();
        viewModel.ConfigureBrowserAction(_ => throw new HttpRequestException("internal details"));
        viewModel.MarkBootstrapComplete();

        await viewModel.OpenInBrowserCommand.ExecuteAsync();

        Assert.Contains("could not be opened", viewModel.BrowserStatusMessage, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("internal details", viewModel.BrowserStatusMessage, StringComparison.Ordinal);
    }

    private static HttpResponseMessage Json(HttpStatusCode status, string body) => new(status)
    {
        Content = new StringContent(body, Encoding.UTF8, "application/json"),
    };

    private sealed class RecordingHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        public HttpRequestMessage? Request { get; private set; }
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Request = request;
            return Task.FromResult(respond(request));
        }
    }
}
