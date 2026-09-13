using System.Windows;
using System.Windows.Controls;
using YouTuber.Launcher.ViewModels;

namespace YouTuber.Launcher.Views;

/// <summary>First-run UI. The composition root supplies setup execution callbacks.</summary>
public partial class SetupView : UserControl
{
    private readonly SetupViewModel? _viewModel;
    private readonly SetupViewActions _actions;
    public event EventHandler? CloseApproved;

    public SetupView()
    {
        InitializeComponent();
        _actions = new SetupViewActions();
    }

    public SetupView(SetupViewModel viewModel, SetupViewActions? actions = null)
    {
        ArgumentNullException.ThrowIfNull(viewModel);
        InitializeComponent();
        _viewModel = viewModel;
        _actions = actions ?? new SetupViewActions();
        DataContext = viewModel;
    }

    private void AcceptTerms_Click(object sender, RoutedEventArgs e) => _viewModel?.AcceptTerms();
    private async void Continue_Click(object sender, RoutedEventArgs e) => await RunActionAsync(_actions.ContinueAsync, "Setup could not continue. Retry the current step.");
    private async void Retry_Click(object sender, RoutedEventArgs e) => await RunActionAsync(_actions.RetryAsync, "Setup retry failed. Check the error and retry again.");

    private async void CloseSafely_Click(object sender, RoutedEventArgs e)
    {
        if (_viewModel is null) return;
        var canClose = _actions.RequestCloseAsync is not null
            ? await _actions.RequestCloseAsync(CancellationToken.None)
            : await _viewModel.RequestCloseAsync();
        if (canClose) CloseApproved?.Invoke(this, EventArgs.Empty);
        else StatusText.Text = "Setup is applying a change. Wait for a safe checkpoint before closing.";
    }

    private async Task RunActionAsync(Func<CancellationToken, Task>? action, string failureMessage)
    {
        if (action is null)
        {
            StatusText.Text = "Setup is not configured yet. Close safely and retry after configuration is available.";
            return;
        }
        try { StatusText.Text = null; await action(CancellationToken.None); }
        catch (OperationCanceledException) { StatusText.Text = "Setup was cancelled."; }
        catch { StatusText.Text = failureMessage; }
    }
}

public sealed record SetupViewActions(
    Func<CancellationToken, Task>? ContinueAsync = null,
    Func<CancellationToken, Task>? RetryAsync = null,
    Func<CancellationToken, Task<bool>>? RequestCloseAsync = null);
