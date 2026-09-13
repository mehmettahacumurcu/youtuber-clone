using System.IO;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.Wpf;

namespace YouTuber.Launcher.Web;

public sealed class StudioWebViewAdapter
{
    private readonly StudioOrigin _origin;
    private readonly string _userDataRoot;
    private readonly SemaphoreSlim _lifecycle = new(1, 1);
    private bool _initialized;
    private bool _bootstrapIssued;
    private bool _bootstrapPending;
    private MemoryStream? _bootstrapContent;

    public StudioWebViewAdapter(StudioOrigin origin, string dataRoot)
    {
        _origin = origin ?? throw new ArgumentNullException(nameof(origin));
        ArgumentException.ThrowIfNullOrWhiteSpace(dataRoot);
        _userDataRoot = Path.Combine(Path.GetFullPath(dataRoot), "state", "webview2");
    }

    public string UserDataRoot => _userDataRoot;
    public event EventHandler? BootstrapCompleted;
    public event EventHandler? RecoveryRequired;

    public async Task InitializeAsync(WebView2 view, string secret, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(view);
        await _lifecycle.WaitAsync(cancellationToken);
        try
        {
            if (_initialized) throw new InvalidOperationException("The Studio WebView is already initialized.");
            Directory.CreateDirectory(_userDataRoot);
            Trace("create-environment");
            var options = new CoreWebView2EnvironmentOptions { ExclusiveUserDataFolderAccess = true };
            var environment = await CoreWebView2Environment.CreateAsync(null, _userDataRoot, options);
            Trace("ensure-control");
            await view.EnsureCoreWebView2Async(environment);
            Trace("configure");
            Configure(view);
            view.CoreWebView2.CookieManager.DeleteAllCookies();
            Trace("bootstrap");
            IssueBootstrap(view.CoreWebView2, StudioBootstrapRequest.Create(_origin, secret));
            _initialized = true;
            Trace("issued");
        }
        finally { _lifecycle.Release(); }
    }

    public async Task RebootstrapAsync(WebView2 view, string secret, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(view);
        await _lifecycle.WaitAsync(cancellationToken);
        try
        {
            if (!_initialized || view.CoreWebView2 is null) throw new InvalidOperationException("The Studio WebView is not initialized.");
            _bootstrapIssued = false;
            _bootstrapPending = false;
            view.CoreWebView2.CookieManager.DeleteAllCookies();
            IssueBootstrap(view.CoreWebView2, StudioBootstrapRequest.Create(_origin, secret));
        }
        finally { _lifecycle.Release(); }
    }

    public void InvalidateSession()
    {
        _bootstrapIssued = false;
        _bootstrapPending = false;
        _bootstrapContent?.Dispose();
        _bootstrapContent = null;
    }

    private void Configure(WebView2 view)
    {
        var core = view.CoreWebView2;
        new WebViewPolicy(_origin).Apply(core.Settings);
        core.Settings.IsBuiltInErrorPageEnabled = false;
        view.AllowExternalDrop = false;
        core.DownloadStarting += (_, args) => { args.Cancel = true; args.Handled = true; };
        core.NewWindowRequested += (_, args) => args.Handled = true;
        core.PermissionRequested += (_, args) => { args.State = CoreWebView2PermissionState.Deny; args.Handled = true; };
        core.NavigationStarting += (_, args) => { if (!_origin.AllowsNavigation(args.Uri)) args.Cancel = true; };
        core.FrameNavigationStarting += (_, args) => { if (!_origin.AllowsNavigation(args.Uri)) args.Cancel = true; };
        core.LaunchingExternalUriScheme += (_, args) => args.Cancel = true;
        core.NavigationCompleted += (_, args) =>
        {
            if (!_bootstrapPending) return;
            _bootstrapPending = false;
            _bootstrapContent?.Dispose();
            _bootstrapContent = null;
            if (args.IsSuccess && _origin.IsLandingPage(core.Source)) BootstrapCompleted?.Invoke(this, EventArgs.Empty);
            else { _bootstrapIssued = false; RecoveryRequired?.Invoke(this, EventArgs.Empty); }
        };
        core.ProcessFailed += (_, _) => { InvalidateSession(); RecoveryRequired?.Invoke(this, EventArgs.Empty); };
    }

    private void IssueBootstrap(CoreWebView2 core, StudioBootstrapRequest request)
    {
        if (_bootstrapIssued) throw new InvalidOperationException("Studio bootstrap has already been issued for this worker session.");
        _bootstrapContent?.Dispose();
        _bootstrapContent = new MemoryStream([], writable: false);
        var webRequest = core.Environment.CreateWebResourceRequest(request.Uri.AbsoluteUri, request.Method, _bootstrapContent, request.WebViewHeaders);
        _bootstrapIssued = true;
        _bootstrapPending = true;
        core.NavigateWithWebResourceRequest(webRequest);
    }

    [System.Diagnostics.Conditional("DEBUG")]
    private void Trace(string stage)
        => File.AppendAllText(Path.Combine(_userDataRoot, "adapter-stage.log"), stage + Environment.NewLine);
}
