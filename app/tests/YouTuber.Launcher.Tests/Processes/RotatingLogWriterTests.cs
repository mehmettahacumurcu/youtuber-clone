using YouTuber.Launcher.Processes;
using Xunit;

namespace YouTuber.Launcher.Tests.Processes;

public sealed class RotatingLogWriterTests
{
    [Fact]
    public void WriteLine_never_exceeds_the_configured_byte_limit_for_an_oversized_utf8_record()
    {
        using var directory = new TemporaryDirectory();
        using var writer = new RotatingLogWriter(directory.Path, "worker.log", maximumBytes: 10, rotations: 5);

        writer.WriteLine("ğğğğğğğğğğ");

        Assert.True(new FileInfo(System.IO.Path.Combine(directory.Path, "worker.log")).Length <= 10);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public TemporaryDirectory() { Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "YouTuberLogs-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(Path); }
        public string Path { get; }
        public void Dispose() { if (Directory.Exists(Path)) Directory.Delete(Path, recursive: true); }
    }
}
