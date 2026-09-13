using System.Windows;
using YouTuber.Launcher.Processes;
using YouTuber.Launcher.ViewModels;
using YouTuber.Launcher.Views;
using YouTuber.Launcher.Web;

namespace YouTuber.Launcher;

public partial class MainWindow : Window, IStudioShellHost
{
    private bool _setupCloseApproved;
    public MainWindow()
    {
        InitializeComponent();
    }

    public Task StartStudioAsync(StudioOrigin origin, string dataRoot, string sessionSecret, IExternalBrowserLauncher? browser = null, CancellationToken cancellationToken = default)
        => Shell.StartAsync(origin, dataRoot, sessionSecret, browser, cancellationToken);

    public void AttachWorkerSupervisor(IWorkerSupervisor supervisor) => Shell.AttachWorkerSupervisor(supervisor);
    public Task HandleWorkersReadyAsync(CancellationToken cancellationToken = default) => Shell.HandleWorkersReadyAsync(cancellationToken);
    Task IStudioShellHost.StartStudioAsync(StudioOrigin origin, string dataRoot, string sessionSecret, CancellationToken cancellationToken)
        => Shell.StartAsync(origin, dataRoot, sessionSecret, cancellationToken: cancellationToken);
    public void ShowStartupFailure(string message) => Shell.ShowStartupFailure(message);

    public void ShowDataRootSelection(DataRootSelectionViewModel viewModel)
    {
        ArgumentNullException.ThrowIfNull(viewModel);
        SetupHost.Content = new DataRootSelectionView(viewModel);
        SetupHost.Visibility = Visibility.Visible;
        Shell.Visibility = Visibility.Collapsed;
    }

    public void ShowSetup(SetupViewModel viewModel, SetupViewActions? actions = null)
    {
        ArgumentNullException.ThrowIfNull(viewModel);
        var setup = new SetupView(viewModel, actions);
        setup.CloseApproved += (_, _) =>
        {
            _setupCloseApproved = true;
            Close();
        };
        SetupHost.Content = setup;
        SetupHost.Visibility = Visibility.Visible;
        Shell.Visibility = Visibility.Collapsed;
    }

    public void ShowStudioShell()
    {
        SetupHost.Content = null;
        SetupHost.Visibility = Visibility.Collapsed;
        Shell.Visibility = Visibility.Visible;
    }

    public bool ConsumeSetupCloseApproval()
    {
        var approved = _setupCloseApproved;
        _setupCloseApproved = false;
        return approved;
    }
}
