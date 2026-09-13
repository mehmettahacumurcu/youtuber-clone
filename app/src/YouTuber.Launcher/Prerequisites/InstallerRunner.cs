using System.Diagnostics;

namespace YouTuber.Launcher.Prerequisites;

public sealed record InstallerRunResult(int ExitCode, string StandardOutput, string StandardError, bool TimedOut);

public interface IInstallerRunner
{
    Task<InstallerRunResult> RunAsync(VerifiedInstaller installer, IEnumerable<string> arguments, TimeSpan timeout, IReadOnlyDictionary<string, string>? environment = null, CancellationToken cancellationToken = default);
}

public sealed class InstallerRunner : IInstallerRunner
{
    public async Task<InstallerRunResult> RunAsync(VerifiedInstaller installer, IEnumerable<string> arguments, TimeSpan timeout, IReadOnlyDictionary<string, string>? environment = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(installer);
        ArgumentNullException.ThrowIfNull(arguments);
        if (timeout <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(timeout));
        installer.EnsureUsable();

        var startInfo = new ProcessStartInfo(installer.Path)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (var argument in arguments) startInfo.ArgumentList.Add(argument);
        if (environment is not null)
        {
            foreach (var pair in environment) startInfo.Environment[pair.Key] = pair.Value;
        }

        using var process = new Process { StartInfo = startInfo };
        try
        {
            if (!process.Start()) throw new InvalidOperationException("The verified prerequisite installer could not be started.");
            var output = process.StandardOutput.ReadToEndAsync();
            var error = process.StandardError.ReadToEndAsync();
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(timeout);
            try
            {
                await process.WaitForExitAsync(timeoutCts.Token);
            }
            catch (OperationCanceledException)
            {
                if (!process.HasExited) process.Kill(entireProcessTree: true);
                await process.WaitForExitAsync(CancellationToken.None);
                if (cancellationToken.IsCancellationRequested) throw;
                return new InstallerRunResult(-1, await output, await error, true);
            }

            return new InstallerRunResult(process.ExitCode, await output, await error, false);
        }
        finally
        {
            installer.Dispose();
        }
    }
}
