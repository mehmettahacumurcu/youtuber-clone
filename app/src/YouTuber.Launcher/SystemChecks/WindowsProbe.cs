using Microsoft.Win32;

namespace YouTuber.Launcher.SystemChecks;

public sealed class WindowsProbe
{
    private const string CurrentVersionKey = @"SOFTWARE\Microsoft\Windows NT\CurrentVersion";

    public int GetCurrentBuild()
    {
        using var key = Registry.LocalMachine.OpenSubKey(CurrentVersionKey, writable: false);
        var build = key?.GetValue("CurrentBuildNumber") as string;
        if (!int.TryParse(build, out var parsedBuild) || parsedBuild <= 0)
        {
            throw new InvalidOperationException("Unable to read the Windows build number from the registry.");
        }

        return parsedBuild;
    }
}
