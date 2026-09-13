using YouTuber.Launcher.Configuration;
using YouTuber.Launcher.ViewModels;
using Xunit;

namespace YouTuber.Launcher.Tests.ViewModels;

public sealed class DataRootSelectionViewModelTests
{
    [Fact]
    public async Task Browse_updates_selection_without_creating_or_persisting_data_root()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var selected = Path.Combine(directory.Path, "custom-data");
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        var picker = new RecordingPicker(selected);
        var viewModel = new DataRootSelectionViewModel(locator, Path.Combine(localAppData, "YouTuberStudio"), picker);

        await viewModel.BrowseCommand.ExecuteAsync();

        Assert.Equal(selected, viewModel.SelectedPath);
        Assert.Equal(Path.Combine(localAppData, "YouTuberStudio"), picker.InitialPath);
        Assert.False(Directory.Exists(selected));
        Assert.Null(locator.TryLoad());
    }

    [Fact]
    public async Task Continue_validates_and_durably_persists_before_completing_selection()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var selected = Path.Combine(directory.Path, "custom-data");
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        var viewModel = new DataRootSelectionViewModel(locator, selected, new RecordingPicker(null));

        await viewModel.ContinueCommand.ExecuteAsync();

        Assert.Equal(Path.GetFullPath(selected), await viewModel.Selection);
        Assert.Equal(Path.GetFullPath(selected), locator.LoadRequired());
        Assert.False(Directory.Exists(selected));
        Assert.Null(viewModel.Error);
    }

    [Fact]
    public async Task Invalid_selection_keeps_chooser_open_and_does_not_replace_locator()
    {
        using var directory = new TemporaryDirectory();
        var localAppData = Directory.CreateDirectory(Path.Combine(directory.Path, "local")).FullName;
        var existing = Path.Combine(directory.Path, "existing-data");
        var locator = DataRootLocator.ForBaseDirectory(localAppData);
        await locator.SaveAsync(existing);
        var viewModel = new DataRootSelectionViewModel(locator, Path.GetPathRoot(directory.Path)!, new RecordingPicker(null));

        await viewModel.ContinueCommand.ExecuteAsync();

        Assert.False(viewModel.Selection.IsCompleted);
        Assert.NotNull(viewModel.Error);
        Assert.Equal(Path.GetFullPath(existing), locator.LoadRequired());
    }

    private sealed class RecordingPicker(string? result) : IDataRootPicker
    {
        public string? InitialPath { get; private set; }

        public Task<string?> PickAsync(string initialPath, CancellationToken cancellationToken = default)
        {
            InitialPath = initialPath;
            return Task.FromResult(result);
        }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "youtuber-selection-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            try { Directory.Delete(Path, recursive: true); }
            catch { }
        }
    }
}
