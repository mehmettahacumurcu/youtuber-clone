using Microsoft.Web.WebView2.Core;
using System.IO;
using System.Runtime.InteropServices;

namespace YouTuber.Launcher.SystemChecks;

public sealed class PrerequisiteProbe
{
    public PrerequisiteStatus Probe()
    {
        var ollamaAvailable = FindExecutableOnPath("ollama.exe") is not null;
        var webView2Available = IsWebView2Available();
        return new PrerequisiteStatus(
            ollamaAvailable,
            webView2Available,
            ollamaAvailable ? "Ollama was found on PATH." : "Ollama was not found on PATH.",
            webView2Available ? "Microsoft Edge WebView2 Runtime is available." : "Microsoft Edge WebView2 Runtime is unavailable.");
    }

    public string GetExistingOllamaStoreDriveRoot(string fallbackPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fallbackPath);
        var configured = Environment.GetEnvironmentVariable("OLLAMA_MODELS");
        var store = string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".ollama", "models")
            : configured;
        return Directory.Exists(store) ? DiskProbe.DriveRoot(store) : DiskProbe.DriveRoot(fallbackPath);
    }

    private static bool IsWebView2Available()
    {
        try
        {
            return !string.IsNullOrWhiteSpace(CoreWebView2Environment.GetAvailableBrowserVersionString());
        }
        catch (Exception exception) when (exception is COMException or FileNotFoundException or InvalidOperationException)
        {
            return false;
        }
    }

    private static string? FindExecutableOnPath(string executable)
    {
        var path = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrWhiteSpace(path))
        {
            return null;
        }

        foreach (var directory in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var candidate = Path.Combine(directory, executable);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }
}
