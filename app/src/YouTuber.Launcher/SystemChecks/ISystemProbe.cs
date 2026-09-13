using YouTuber.Launcher.Configuration;

namespace YouTuber.Launcher.SystemChecks;

public interface ISystemProbe
{
    Task<SystemProfile> ProbeAsync(AppPaths paths, CancellationToken cancellationToken = default);
}

public sealed class WindowsSystemProbe : ISystemProbe
{
    private readonly WindowsProbe _windowsProbe;
    private readonly NvidiaProbe _nvidiaProbe;
    private readonly MemoryProbe _memoryProbe;
    private readonly DiskProbe _diskProbe;
    private readonly PrerequisiteProbe _prerequisiteProbe;

    public WindowsSystemProbe(
        WindowsProbe? windowsProbe = null,
        NvidiaProbe? nvidiaProbe = null,
        MemoryProbe? memoryProbe = null,
        DiskProbe? diskProbe = null,
        PrerequisiteProbe? prerequisiteProbe = null)
    {
        _windowsProbe = windowsProbe ?? new WindowsProbe();
        _nvidiaProbe = nvidiaProbe ?? new NvidiaProbe();
        _memoryProbe = memoryProbe ?? new MemoryProbe();
        _diskProbe = diskProbe ?? new DiskProbe();
        _prerequisiteProbe = prerequisiteProbe ?? new PrerequisiteProbe();
    }

    public async Task<SystemProfile> ProbeAsync(AppPaths paths, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var ollamaDrive = _prerequisiteProbe.GetExistingOllamaStoreDriveRoot(paths.DataRoot);
        var disk = _diskProbe.GetAvailableBytes(paths, ollamaDrive);
        return new SystemProfile(
            _windowsProbe.GetCurrentBuild(),
            await _nvidiaProbe.ProbeAsync(cancellationToken).ConfigureAwait(false),
            _memoryProbe.GetPhysicalMemoryBytes(),
            disk,
            DiskProbe.DriveRoot(paths.DataRoot),
            ollamaDrive,
            _prerequisiteProbe.Probe());
    }
}
