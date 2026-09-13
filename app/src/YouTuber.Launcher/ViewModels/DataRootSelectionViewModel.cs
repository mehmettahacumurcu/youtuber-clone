using System.IO;
using YouTuber.Launcher.Configuration;

namespace YouTuber.Launcher.ViewModels;

public interface IDataRootPicker
{
    Task<string?> PickAsync(string initialPath, CancellationToken cancellationToken = default);
}

public sealed class WindowsDataRootPicker : IDataRootPicker
{
    public Task<string?> PickAsync(string initialPath, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var dialog = new Microsoft.Win32.OpenFolderDialog
        {
            Title = "Choose where YouTuber Studio stores its models and data",
            InitialDirectory = initialPath,
            Multiselect = false,
        };
        var selected = dialog.ShowDialog() == true ? dialog.FolderName : null;
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(selected);
    }
}

public sealed class DataRootSelectionViewModel : ObservableObject
{
    private readonly DataRootLocator _locator;
    private readonly IDataRootPicker _picker;
    private readonly TaskCompletionSource<string> _selection = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private string _selectedPath;
    private string? _error;

    public DataRootSelectionViewModel(DataRootLocator locator, string defaultPath, IDataRootPicker picker)
    {
        ArgumentNullException.ThrowIfNull(locator);
        ArgumentException.ThrowIfNullOrWhiteSpace(defaultPath);
        ArgumentNullException.ThrowIfNull(picker);
        _locator = locator;
        _picker = picker;
        _selectedPath = defaultPath;
        BrowseCommand = new AsyncRelayCommand(BrowseAsync);
        ContinueCommand = new AsyncRelayCommand(ContinueAsync);
    }

    public string SelectedPath
    {
        get => _selectedPath;
        set
        {
            if (string.Equals(_selectedPath, value, StringComparison.Ordinal)) return;
            _selectedPath = value;
            Error = null;
            Raise();
        }
    }

    public string? Error
    {
        get => _error;
        private set
        {
            if (string.Equals(_error, value, StringComparison.Ordinal)) return;
            _error = value;
            Raise();
        }
    }

    public Task<string> Selection => _selection.Task;

    public AsyncRelayCommand BrowseCommand { get; }

    public AsyncRelayCommand ContinueCommand { get; }

    private async Task BrowseAsync(CancellationToken cancellationToken)
    {
        var selected = await _picker.PickAsync(SelectedPath, cancellationToken);
        if (!string.IsNullOrWhiteSpace(selected)) SelectedPath = selected;
    }

    private async Task ContinueAsync(CancellationToken cancellationToken)
    {
        try
        {
            await _locator.SaveAsync(SelectedPath, cancellationToken);
            var persisted = _locator.LoadRequired();
            Error = null;
            _selection.TrySetResult(persisted);
        }
        catch (Exception exception) when (exception is ArgumentException or InvalidDataException or IOException)
        {
            Error = "Choose an absolute folder on a local NTFS drive that is not a drive root, protected folder, or link.";
        }
    }
}
