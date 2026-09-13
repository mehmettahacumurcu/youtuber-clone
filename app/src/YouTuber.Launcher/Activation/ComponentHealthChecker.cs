using System.Diagnostics;
using System.IO;

namespace YouTuber.Launcher.Activation;

public sealed class ComponentHealthCheckException : ComponentActivationException
{
    public ComponentHealthCheckException(string message, Exception? innerException = null) : base(message, innerException) { }
}

public sealed record ComponentHealthCheckRequest(string ExecutablePath, IReadOnlyList<string> Arguments, string WorkingDirectory);

public interface IComponentHealthChecker
{
    Task CheckAsync(ComponentHealthCheckRequest request, CancellationToken cancellationToken = default);
}

public sealed class ComponentHealthChecker : IComponentHealthChecker
{
    public static readonly TimeSpan Timeout = TimeSpan.FromSeconds(180);

    public async Task CheckAsync(ComponentHealthCheckRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var arguments = request.Arguments;
        if (!Path.IsPathFullyQualified(request.ExecutablePath) || !Path.IsPathFullyQualified(request.WorkingDirectory) || !File.Exists(request.ExecutablePath) || arguments is null || arguments.Any(argument => argument is null || argument.IndexOf('\0') >= 0))
        {
            throw new ComponentHealthCheckException("The component health-check request is invalid.");
        }

        var relative = Path.GetRelativePath(request.WorkingDirectory, request.ExecutablePath);
        if (Path.IsPathRooted(relative) || relative == ".." || relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal)) throw new ComponentHealthCheckException("The health-check entrypoint escapes its component directory.");
        var start = new ProcessStartInfo
        {
            FileName = request.ExecutablePath,
            WorkingDirectory = request.WorkingDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = false,
            RedirectStandardError = false,
        };
        foreach (var argument in arguments) start.ArgumentList.Add(argument);
        var windowsRoot = Path.GetDirectoryName(Environment.SystemDirectory);
        if (string.IsNullOrWhiteSpace(windowsRoot) || !Path.IsPathFullyQualified(windowsRoot) || !Directory.Exists(windowsRoot))
            throw new ComponentHealthCheckException("The Windows bootstrap directory is unavailable.");
        start.Environment.Clear();
        start.Environment["SystemRoot"] = windowsRoot;
        start.Environment["WINDIR"] = windowsRoot;
        start.Environment["YOUTUBER_COMPONENT_ROOT"] = request.WorkingDirectory;
        start.Environment["YOUTUBER_HEALTHCHECK"] = "1";

        using var process = new Process { StartInfo = start };
        var started = false;
        try
        {
            if (!process.Start()) throw new ComponentHealthCheckException("The component health-check process did not start.");
            started = true;
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            deadline.CancelAfter(Timeout);
            await process.WaitForExitAsync(deadline.Token);
            if (process.ExitCode != 0) throw new ComponentHealthCheckException($"The component health check exited with code {process.ExitCode}.");
        }
        catch (OperationCanceledException exception) when (!cancellationToken.IsCancellationRequested)
        {
            TryKill(process);
            throw new ComponentHealthCheckException("The component health check timed out after 180 seconds.", exception);
        }
        catch (System.ComponentModel.Win32Exception exception)
        {
            throw new ComponentHealthCheckException("The component health-check process could not be started.", exception);
        }
        finally
        {
            if (started && !process.HasExited) TryKill(process);
        }
    }

    private static void TryKill(Process process)
    {
        try { process.Kill(entireProcessTree: true); }
        catch (InvalidOperationException) { }
    }
}
