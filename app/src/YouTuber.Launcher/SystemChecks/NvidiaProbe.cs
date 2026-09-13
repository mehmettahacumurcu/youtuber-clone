using System.Diagnostics;
using System.ComponentModel;
using System.Globalization;
using System.IO;

namespace YouTuber.Launcher.SystemChecks;

public sealed record NvidiaSmiProcessResult(
    int? ExitCode,
    string StandardOutput,
    string StandardError,
    bool TimedOut,
    Exception? StartException);

public sealed class NvidiaProbe
{
    private static readonly string[] NvidiaSmiArguments =
    ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"];

    public async Task<NvidiaProbeResult> ProbeAsync(CancellationToken cancellationToken = default)
    {
        var result = await RunNvidiaSmiAsync(cancellationToken).ConfigureAwait(false);
        return ClassifyProcessResult(result);
    }

    public static NvidiaProbeResult ClassifyProcessResult(NvidiaSmiProcessResult result)
    {
        ArgumentNullException.ThrowIfNull(result);

        if (result.StartException is FileNotFoundException or Win32Exception)
        {
            return NvidiaProbeResult.Failure("nvidia-smi was not found. Install a supported NVIDIA driver and restart the launcher.");
        }

        if (result.StartException is not null)
        {
            return NvidiaProbeResult.Failure($"nvidia-smi could not be started: {result.StartException.Message}");
        }

        if (result.TimedOut)
        {
            return NvidiaProbeResult.Failure("nvidia-smi timed out after 10 seconds. Restart the NVIDIA driver or reinstall it.");
        }

        if (result.ExitCode is not 0)
        {
            return NvidiaProbeResult.Failure($"nvidia-smi exited with exit code {result.ExitCode}: {result.StandardError.Trim()}");
        }

        return ParseCsvOutput(result.StandardOutput);
    }

    public static NvidiaProbeResult ParseCsvOutput(string output)
    {
        if (string.IsNullOrWhiteSpace(output))
        {
            return NvidiaProbeResult.Failure("nvidia-smi returned no GPU data.");
        }

        var gpus = new List<NvidiaGpu>();
        foreach (var line in output.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var fields = line.Split(',', StringSplitOptions.TrimEntries);
            if (fields.Length != 3 || string.IsNullOrWhiteSpace(fields[0]) || string.IsNullOrWhiteSpace(fields[2]) ||
                !long.TryParse(fields[1], NumberStyles.None, CultureInfo.InvariantCulture, out var memoryMegabytes) || memoryMegabytes <= 0)
            {
                return NvidiaProbeResult.Failure("nvidia-smi returned malformed GPU memory data.");
            }

            try
            {
                gpus.Add(new NvidiaGpu(fields[0], checked(memoryMegabytes * 1024 * 1024), fields[2]));
            }
            catch (OverflowException)
            {
                return NvidiaProbeResult.Failure("nvidia-smi returned malformed GPU memory data.");
            }
        }

        return NvidiaProbeResult.Success(gpus);
    }

    private static async Task<NvidiaSmiProcessResult> RunNvidiaSmiAsync(CancellationToken cancellationToken)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = "nvidia-smi",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (var argument in NvidiaSmiArguments)
        {
            startInfo.ArgumentList.Add(argument);
        }

        try
        {
            using var process = new Process { StartInfo = startInfo };
            if (!process.Start())
            {
                return new NvidiaSmiProcessResult(null, string.Empty, string.Empty, false, new InvalidOperationException("Process did not start."));
            }

            var outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
            var errorTask = process.StandardError.ReadToEndAsync(cancellationToken);
            using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeout.CancelAfter(TimeSpan.FromSeconds(10));
            try
            {
                await process.WaitForExitAsync(timeout.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
            {
                process.Kill(entireProcessTree: true);
                await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);
                return new NvidiaSmiProcessResult(null, await outputTask.ConfigureAwait(false), await errorTask.ConfigureAwait(false), true, null);
            }

            return new NvidiaSmiProcessResult(process.ExitCode, await outputTask.ConfigureAwait(false), await errorTask.ConfigureAwait(false), false, null);
        }
        catch (Exception exception) when (exception is Win32Exception or FileNotFoundException)
        {
            return new NvidiaSmiProcessResult(null, string.Empty, string.Empty, false, exception);
        }
    }
}
