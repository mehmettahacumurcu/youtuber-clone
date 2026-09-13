using YouTuber.Launcher.Setup;
namespace YouTuber.Launcher.ViewModels;
public sealed class ComponentProgressViewModel : ObservableObject
{
    private readonly SetupCoordinator _coordinator;
    private ComponentProgress _value;
    public string Name => _value.Name;
    public string Version => _value.Version;
    public long BytesReceived => _value.BytesReceived;
    public long TotalBytes => _value.TotalBytes;
    public double Speed => _value.BytesPerSecond;
    public TimeSpan? Eta => _value.Eta;
    public string Status => _value.Status;
    public string? Error => _value.Error;
    public string AccessibleName => $"{Name} {Version} {Status}".Trim();
    public string TransferredText => $"{FormatBytes(BytesReceived)} / {FormatBytes(TotalBytes)}";
    public string SpeedText => Speed <= 0 ? "—" : $"{FormatBytes((long)Speed)}/s";
    public string EtaText => Eta is null ? "—" : Eta.Value.TotalHours >= 1 ? $"{(int)Eta.Value.TotalHours}:{Eta.Value.Minutes:00}:{Eta.Value.Seconds:00}" : $"{Eta.Value.Minutes}:{Eta.Value.Seconds:00}";
    public AsyncRelayCommand PauseCommand { get; }
    public AsyncRelayCommand ResumeCommand { get; }
    public AsyncRelayCommand RetryCommand { get; }
    public ComponentProgressViewModel(ComponentProgress value, SetupCoordinator coordinator)
    {
        _value = value;
        _coordinator = coordinator;
        PauseCommand = new AsyncRelayCommand(token => _coordinator.PauseComponentAsync(Name, token), () => _coordinator.CanOperateComponent(Name, "Downloading"));
        ResumeCommand = new AsyncRelayCommand(token => _coordinator.ResumeComponentAsync(Name, token), () => _coordinator.CanOperateComponent(Name, "Paused"));
        RetryCommand = new AsyncRelayCommand(token => _coordinator.RetryComponentAsync(Name, token), () => _coordinator.CanOperateComponent(Name, "Failed"));
    }
    public void Update(ComponentProgress value) { _value = value; foreach (var property in new[] { nameof(Name), nameof(Version), nameof(BytesReceived), nameof(TotalBytes), nameof(Speed), nameof(Eta), nameof(Status), nameof(Error), nameof(AccessibleName), nameof(TransferredText), nameof(SpeedText), nameof(EtaText) }) Raise(property); PauseCommand.RaiseCanExecuteChanged(); ResumeCommand.RaiseCanExecuteChanged(); RetryCommand.RaiseCanExecuteChanged(); }

    private static string FormatBytes(long bytes)
    {
        string[] units = ["B", "KB", "MB", "GB", "TB"];
        var value = Math.Max(0, bytes);
        var unit = 0;
        var display = (double)value;
        while (display >= 1024 && unit < units.Length - 1) { display /= 1024; unit++; }
        return unit == 0 ? $"{value} {units[unit]}" : $"{display:0.0} {units[unit]}";
    }
}
