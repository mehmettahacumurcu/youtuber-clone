using System.Windows;
using System.Windows.Controls;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.ViewModels;
using YouTuber.Launcher.Web;

namespace YouTuber.Launcher.Views;

public partial class ShellView : UserControl, IDisposable
{
    private readonly ShellViewModel _viewModel = new();
    private StudioWebViewAdapter? _adapter;
    private BrowserBootstrapClient? _browserClient;
    private string? _sessionSecret;
    private bool _webViewInitialized;

    public ShellView()
    {
        InitializeComponent();
        DataContext = _viewModel;
        Unloaded += (_, _) => Dispose();
    }

    public async Task StartAsync(StudioOrigin origin, string dataRoot, string sessionSecret, IExternalBrowserLauncher? browser = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(origin);
        _sessionSecret = sessionSecret;
        _adapter = new StudioWebViewAdapter(origin, dataRoot);
        _adapter.BootstrapCompleted += (_, _) => Dispatcher.Invoke(ShowStudio);
        _adapter.RecoveryRequired += (_, _) => Dispatcher.Invoke(() => ShowRecovery("Studio stopped unexpectedly. Restart the local services to reconnect."));
        _browserClient?.Dispose();
        _browserClient = new BrowserBootstrapClient(origin);
        var browserLauncher = browser ?? new ShellExecuteBrowserLauncher();
        _viewModel.ConfigureBrowserAction(async token => browserLauncher.Open(await _browserClient.AcquireUrlAsync(sessionSecret, token)));
        _viewModel.MarkStarting();
        ApplyVisibility();
        try
        {
            await _adapter.InitializeAsync(Studio, sessionSecret, cancellationToken);
            _webViewInitialized = true;
            ApplyVisibility();
        }
        catch { ShowRecovery("Studio could not start. Review the launcher logs and retry."); throw; }
    }

    public void AttachWorkerSupervisor(IWorkerSupervisor supervisor)
    {
        ArgumentNullException.ThrowIfNull(supervisor);
        supervisor.RecoveryRequired += (_, _) => Dispatcher.Invoke(() => HandleWorkerRecovery());
        supervisor.WorkerRestarting += (_, _) => Dispatcher.Invoke(HandleWorkerRecovery);
        supervisor.WorkerRestarted += (_, _) => _ = HandleWorkerRestartedAsync();
    }

    public void HandleWorkerRecovery()
    {
        _adapter?.InvalidateSession();
        ShowRecovery("A local worker stopped. Studio is hidden until recovery finishes.");
    }

    public async Task HandleWorkersReadyAsync(CancellationToken cancellationToken = default)
    {
        if (_adapter is null || _sessionSecret is null) throw new InvalidOperationException("Studio shell is not configured.");
        _viewModel.MarkStarting("Local workers are ready. Re-establishing the Studio session...");
        ApplyVisibility();
        await _adapter.RebootstrapAsync(Studio, _sessionSecret, cancellationToken);
    }

    private async Task HandleWorkerRestartedAsync()
    {
        try
        {
            if (!Dispatcher.CheckAccess())
            {
                await Dispatcher.InvokeAsync(() => HandleWorkerRestartedAsync()).Task.Unwrap();
                return;
            }
            await HandleWorkersReadyAsync();
        }
        catch
        {
            if (Dispatcher.CheckAccess()) ShowRecovery("Local worker recovery finished, but Studio could not establish a new session.");
            else Dispatcher.Invoke(() => ShowRecovery("Local worker recovery finished, but Studio could not establish a new session."));
        }
    }

    private void ShowStudio()
    {
        _viewModel.MarkBootstrapComplete();
        ApplyVisibility();
    }

    private void ShowRecovery(string message)
    {
        _viewModel.MarkRecoveryRequired(message);
        ApplyVisibility();
    }

    public void ShowStartupFailure(string message) => ShowRecovery(message);

    private void ApplyVisibility()
    {
        Studio.Visibility = !_webViewInitialized || _viewModel.IsStudioVisible ? Visibility.Visible : Visibility.Hidden;
        Studio.IsHitTestVisible = _viewModel.IsStudioVisible;
        NativeStatus.Visibility = _viewModel.IsStudioVisible ? Visibility.Collapsed : Visibility.Visible;
    }

    public void Dispose()
    {
        _viewModel.ClearBrowserAction();
        _browserClient?.Dispose();
        _browserClient = null;
        _sessionSecret = null;
        Studio.Dispose();
    }
}
