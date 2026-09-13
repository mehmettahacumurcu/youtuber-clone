using YouTuber.Launcher.SystemChecks;
using Xunit;

namespace YouTuber.Launcher.Tests.SystemChecks;

public sealed class NvidiaProbeTests
{
    [Fact]
    public void ParseCsvOutputSelectsTheLargestGpuAndRetainsEveryIdentityForDiagnostics()
    {
        var result = NvidiaProbe.ParseCsvOutput("NVIDIA RTX 4070, 12282, 555.85\nNVIDIA RTX 4090, 24564, 555.99");

        Assert.True(result.IsAvailable);
        Assert.Equal("NVIDIA RTX 4090", result.SelectedGpu!.Name);
        Assert.Equal(24_564L * 1024 * 1024, result.SelectedGpu.VramBytes);
        Assert.Equal(["NVIDIA RTX 4070", "NVIDIA RTX 4090"], result.Gpus.Select(gpu => gpu.Name));
    }

    [Theory]
    [InlineData("NVIDIA RTX 4090, not-a-number, 555.99")]
    [InlineData("NVIDIA RTX 4090, -1, 555.99")]
    [InlineData("NVIDIA RTX 4090, 24564")]
    public void ParseCsvOutputRejectsMalformedMemoryAsAnUnavailableNvidiaProbe(string output)
    {
        var result = NvidiaProbe.ParseCsvOutput(output);

        Assert.False(result.IsAvailable);
        Assert.Contains("malformed", result.Diagnostic, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ClassifyProcessFailureRejectsMissingNvidiaSmi()
    {
        var result = NvidiaProbe.ClassifyProcessResult(new NvidiaSmiProcessResult(null, "", "", false, new FileNotFoundException()));

        Assert.False(result.IsAvailable);
        Assert.Contains("not found", result.Diagnostic, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ClassifyProcessFailureRejectsTimeout()
    {
        var result = NvidiaProbe.ClassifyProcessResult(new NvidiaSmiProcessResult(null, "", "", true, null));

        Assert.False(result.IsAvailable);
        Assert.Contains("timed out", result.Diagnostic, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ClassifyProcessFailureRejectsNonzeroExitCode()
    {
        var result = NvidiaProbe.ClassifyProcessResult(new NvidiaSmiProcessResult(9, "", "driver error", false, null));

        Assert.False(result.IsAvailable);
        Assert.Contains("exit code 9", result.Diagnostic, StringComparison.OrdinalIgnoreCase);
    }
}
