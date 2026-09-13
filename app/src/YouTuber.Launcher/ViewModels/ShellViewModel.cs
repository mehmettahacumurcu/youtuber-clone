using YouTuber.Launcher.Web;

namespace YouTuber.Launcher.ViewModels;

public sealed class ShellViewModel : ObservableObject
{
    private bool _isStudioVisible;
    private bool _needsBootstrap = true;
    private string _statusMessage = "Starting YouTuber Studio...";
    private string? _browserStatusMessage;
    private Func<CancellationToken, Task>? _openBrowser;

    public bool IsStudioVisible { get => _isStudioVisible; private set { _isStudioVisible = value; Raise(); } }
    public bool NeedsBootstrap { get => _needsBootstrap; private set { _needsBootstrap = value; Raise(); } }
    public string StatusMessage { get => _statusMessage; private set { _statusMessage = value; Raise(); } }
    public string BrowserWarning => "This opens a one-time local sign-in link. Do not share the link while it is valid.";
    public string? BrowserStatusMessage { get => _browserStatusMessage; private set { _browserStatusMessage = value; Raise(); } }
    public AsyncRelayCommand OpenInBrowserCommand { get; }

    public ShellViewModel()
    {
        OpenInBrowserCommand = new AsyncRelayCommand(
            OpenBrowserAsync,
            () => _openBrowser is not null && IsStudioVisible);
    }

    public void ConfigureBrowserAction(Func<CancellationToken, Task> openBrowser)
    {
        _openBrowser = openBrowser ?? throw new ArgumentNullException(nameof(openBrowser));
        BrowserStatusMessage = null;
        OpenInBrowserCommand.RaiseCanExecuteChanged();
    }

    public void ClearBrowserAction()
    {
        _openBrowser = null;
        BrowserStatusMessage = null;
        OpenInBrowserCommand.RaiseCanExecuteChanged();
    }

    private async Task OpenBrowserAsync(CancellationToken cancellationToken)
    {
        try
        {
            BrowserStatusMessage = null;
            if (_openBrowser is not null) await _openBrowser(cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { }
        catch
        {
            BrowserStatusMessage = "The one-time browser link could not be opened. Retry after the local services are ready.";
        }
    }

    public void MarkStarting(string message = "Starting YouTuber Studio...")
    {
        StatusMessage = message;
        IsStudioVisible = false;
        NeedsBootstrap = true;
        OpenInBrowserCommand.RaiseCanExecuteChanged();
    }

    public void MarkBootstrapComplete()
    {
        StatusMessage = "YouTuber Studio is ready.";
        NeedsBootstrap = false;
        IsStudioVisible = true;
        OpenInBrowserCommand.RaiseCanExecuteChanged();
    }

    public void MarkRecoveryRequired(string message)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message);
        StatusMessage = message;
        IsStudioVisible = false;
        NeedsBootstrap = true;
        OpenInBrowserCommand.RaiseCanExecuteChanged();
    }
}
